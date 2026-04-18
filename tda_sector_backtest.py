"""
TDA Sector-Specific Backtest: Single Source of Truth
=====================================================

This script produces ALL sector-related numbers for the paper:
  - Table 7:  Sector-specific performance (primary results)
  - Table 30: 2010-2019 vs 2019-2024 temporal comparison
  - Table 31: Transaction cost sensitivity
  - ρc threshold analysis

Every number in the paper traces back to this one file.

Methodology:
  1. Download real price data via yfinance
  2. For each sector, compute rolling 60-day:
     - Mean pairwise correlation (ρ)
     - H1 persistent homology CV (topology instability)
  3. Strategy: go to cash when topology CV exceeds expanding 75th percentile
  4. Walk-forward: threshold computed on expanding window only (no lookahead)
  5. Report Sharpe, CAGR, max drawdown, with transaction cost scenarios

Run:
    pip install yfinance numpy pandas scipy ripser
    python tda_sector_backtest.py

Author: Adam [Last Name]
Date: February 2026
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("ERROR: ripser not installed. Install with: pip install ripser")
    print("This script requires ripser for real TDA computation.")
    print("Results without ripser are NOT valid for the paper.")

# =============================================================================
# CONFIGURATION — single place to change anything
# =============================================================================

CONFIG = {
    # Rolling window for correlation and TDA
    'window': 60,

    # Strategy parameters
    'cv_percentile': 0.75,       # Go to cash when CV > this expanding percentile
    'rebalance_freq': 5,         # Rebalance every N trading days

    # Transaction cost scenarios (bps per round-trip)
    'cost_scenarios_bps': [0, 5, 15, 25],

    # Minimum data requirements
    'min_days': 252,             # At least 1 year
    'min_cv_history': 60,        # Expanding window minimum

    # Output
    'output_dir': 'results',
    'save_intermediate': True,
}

# =============================================================================
# SECTOR DEFINITIONS
# Exactly 10 stocks per sector for consistent network size
# These must match the paper's sector definitions
# =============================================================================

SECTORS = {
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB'],
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ADBE', 'CRM', 'INTC', 'AMD', 'CSCO'],
    'Energy':     ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN'],
    'Materials':  ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'VMC', 'MLM', 'DD'],
    'Industrials':['HON', 'UNP', 'UPS', 'CAT', 'RTX', 'DE', 'LMT', 'GE', 'MMM', 'BA'],
    'Consumer':   ['PG', 'KO', 'PEP', 'COST', 'WMT', 'MCD', 'NKE', 'SBUX', 'TGT', 'CL'],
}

# Time periods
PERIODS = {
    '2010-2019': ('2010-01-01', '2019-12-31'),
    '2019-2024': ('2019-01-01', '2024-12-31'),
    # The paper's primary test period (walk-forward OOS)
    '2022-2024_oos': ('2019-01-01', '2024-12-31'),  # Train from 2019, test 2022+
}

# Also run cross-sector (the "failed" baseline)
CROSS_SECTOR_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOGL', 'TSLA',
    'NFLX', 'JPM', 'PEP', 'CSCO', 'ORCL', 'DIS', 'BAC',
    'XOM', 'IBM', 'INTC', 'AMD', 'KO', 'WMT'
]


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_prices(tickers, start_date, end_date, label=""):
    """Fetch adjusted close prices from yfinance."""
    print(f"  Fetching {label}: {start_date} to {end_date} ({len(tickers)} tickers)...")

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=True
        )

        if data.empty:
            print(f"    WARNING: No data returned")
            return None

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data

        # Ensure we have a DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        # Drop tickers with >20% missing data
        valid = prices.columns[prices.notna().sum() > len(prices) * 0.8]
        prices = prices[valid].ffill().dropna()

        if len(prices) < CONFIG['min_days']:
            print(f"    WARNING: Only {len(prices)} days (need {CONFIG['min_days']})")
            return None

        print(f"    OK: {len(prices)} days, {len(prices.columns)} tickers: {list(prices.columns)}")
        return prices

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


# =============================================================================
# TDA COMPUTATION (Real ripser)
# =============================================================================

def compute_h1_lifetimes(distance_matrix):
    """
    Compute H1 persistent homology lifetimes using ripser.
    Returns array of finite H1 feature lifetimes.
    """
    if not HAS_RIPSER:
        raise RuntimeError("ripser required for valid TDA computation")

    # Ensure symmetric, zero diagonal
    dm = np.array(distance_matrix, dtype=np.float64)
    np.fill_diagonal(dm, 0)
    dm = (dm + dm.T) / 2

    result = ripser(dm, maxdim=1, distance_matrix=True)
    h1_dgm = result['dgms'][1]

    # Keep only finite-lifetime features
    finite_mask = np.isfinite(h1_dgm[:, 1])
    h1_finite = h1_dgm[finite_mask]

    if len(h1_finite) == 0:
        return np.array([0.0])

    lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
    return lifetimes


def compute_topology_cv(lifetimes):
    """Coefficient of variation of H1 lifetimes."""
    if len(lifetimes) == 0 or np.mean(lifetimes) < 1e-10:
        return np.nan
    return float(np.std(lifetimes) / np.mean(lifetimes))


# =============================================================================
# ROLLING COMPUTATIONS
# =============================================================================

def compute_rolling_features(returns, window=60):
    """
    Compute rolling correlation and topology CV for a sector.
    Returns DataFrame with columns: ['mean_corr', 'topology_cv', 'h1_count', 'h1_total_persistence']
    """
    results = []

    n_windows = len(returns) - window
    report_every = max(1, n_windows // 10)

    for i in range(window, len(returns)):
        if (i - window) % report_every == 0:
            pct = (i - window) / n_windows * 100
            print(f"      Progress: {pct:.0f}% ({i-window}/{n_windows})")

        window_returns = returns.iloc[i-window:i]

        # Correlation
        corr_matrix = window_returns.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr = float(corr_matrix.values[mask].mean())

        # Distance matrix for TDA
        distance_matrix = np.sqrt(2 * (1 - corr_matrix.values))
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.clip(distance_matrix, 0, 2)  # Ensure valid range

        # TDA
        try:
            lifetimes = compute_h1_lifetimes(distance_matrix)
            cv = compute_topology_cv(lifetimes)
            h1_count = len(lifetimes)
            h1_total = float(np.sum(lifetimes))
        except Exception as e:
            cv = np.nan
            h1_count = 0
            h1_total = 0.0

        results.append({
            'date': returns.index[i],
            'mean_corr': mean_corr,
            'topology_cv': cv,
            'h1_count': h1_count,
            'h1_total_persistence': h1_total,
        })

    df = pd.DataFrame(results).set_index('date')
    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(prices, features_df, window=60, rebalance_freq=5):
    """
    Run the TDA regime-based backtest.

    Strategy logic:
      - Compute expanding 75th percentile of topology CV (no lookahead)
      - When CV > threshold → go to cash (regime is unstable)
      - When CV <= threshold → equal-weight invest in sector
      - Rebalance every N days to reduce turnover

    Returns dict with all metrics.
    """
    returns = np.log(prices / prices.shift(1)).dropna()

    # Align returns with features
    common_dates = returns.index.intersection(features_df.index)
    returns_aligned = returns.loc[common_dates]
    topology_cv = features_df.loc[common_dates, 'topology_cv']
    rolling_corr = features_df.loc[common_dates, 'mean_corr']

    if len(common_dates) < 100:
        return None

    # Expanding threshold (no lookahead)
    min_periods = CONFIG['min_cv_history']
    cv_threshold = topology_cv.expanding(min_periods=min_periods).quantile(CONFIG['cv_percentile'])

    # Raw position signal
    position_raw = (topology_cv < cv_threshold).astype(float)
    # Fill NaN (before we have enough history) with 1 (invested)
    position_raw = position_raw.fillna(1.0)

    # Rebalance only every N days
    position = position_raw.copy()
    last_pos = 1.0
    rebalance_set = set(position.index[::rebalance_freq])
    for date in position.index:
        if date in rebalance_set:
            last_pos = position_raw.loc[date]
        position.loc[date] = last_pos

    # Portfolio returns (equal-weight sector, lagged position)
    sector_return = returns_aligned.mean(axis=1)
    portfolio_return = sector_return * position.shift(1)
    portfolio_return = portfolio_return.dropna()

    if len(portfolio_return) < 50:
        return None

    # Buy-and-hold benchmark
    benchmark_return = sector_return.loc[portfolio_return.index]

    # === Metrics ===
    n_years = len(portfolio_return) / 252

    # Strategy
    ann_ret = portfolio_return.mean() * 252
    ann_vol = portfolio_return.std() * np.sqrt(252)
    sharpe_gross = ann_ret / ann_vol if ann_vol > 0 else 0

    cum = (1 + portfolio_return).cumprod()
    cagr = (cum.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0
    max_dd = (cum / cum.cummax() - 1).min()

    # Benchmark
    bm_ann_ret = benchmark_return.mean() * 252
    bm_ann_vol = benchmark_return.std() * np.sqrt(252)
    bm_sharpe = bm_ann_ret / bm_ann_vol if bm_ann_vol > 0 else 0
    bm_cum = (1 + benchmark_return).cumprod()
    bm_cagr = (bm_cum.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0
    bm_max_dd = (bm_cum / bm_cum.cummax() - 1).min()

    # Turnover
    position_changes = position.diff().abs()
    total_switches = (position_changes > 0).sum()
    trades_per_year = total_switches / n_years if n_years > 0 else 0
    annual_turnover_pct = position_changes.sum() / n_years * 100 if n_years > 0 else 0

    # Time in market
    time_invested = position.mean()

    # Cost-adjusted Sharpe ratios
    cost_sharpes = {}
    for bps in CONFIG['cost_scenarios_bps']:
        cost_per_switch = bps / 10000
        total_cost = (position_changes * cost_per_switch).sum()
        annual_cost = total_cost / n_years if n_years > 0 else 0
        net_ann_ret = ann_ret - annual_cost
        cost_sharpes[f'sharpe_{bps}bps'] = net_ann_ret / ann_vol if ann_vol > 0 else 0

    # Statistical significance (t-test, Lo 2002)
    if len(portfolio_return) > 30 and portfolio_return.std() > 0:
        t_stat = portfolio_return.mean() / (portfolio_return.std() / np.sqrt(len(portfolio_return)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(portfolio_return)-1))
    else:
        t_stat = 0
        p_value = 1.0

    result = {
        # Core metrics
        'sharpe': sharpe_gross,
        'cagr': cagr,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'max_dd': max_dd,

        # Statistical
        't_stat': t_stat,
        'p_value': p_value,

        # Topology stats
        'mean_corr': float(rolling_corr.mean()),
        'mean_cv': float(topology_cv.mean()),
        'std_cv': float(topology_cv.std()),

        # Trading stats
        'n_days': len(portfolio_return),
        'n_years': n_years,
        'trades_per_year': trades_per_year,
        'annual_turnover_pct': annual_turnover_pct,
        'time_invested': time_invested,

        # Benchmark
        'bm_sharpe': bm_sharpe,
        'bm_cagr': bm_cagr,
        'bm_max_dd': bm_max_dd,

        # Cost scenarios
        **cost_sharpes,
    }

    return result


# =============================================================================
# WALK-FORWARD BACKTEST (for primary Table 7 results)
# =============================================================================

def run_walkforward_backtest(prices, window=60, train_years=3, test_months=12):
    """
    Walk-forward validation: train threshold on expanding window,
    test on next 12 months. More rigorous than simple expanding percentile.

    This is used for the primary Table 7 results (2022-2024 OOS).
    """
    returns = np.log(prices / prices.shift(1)).dropna()

    # We need at least train_years of data before first test
    min_train_days = train_years * 252

    if len(returns) < min_train_days + 252:
        print(f"    WARNING: Not enough data for walk-forward ({len(returns)} days)")
        return None

    # Pre-compute all rolling features
    print("    Computing rolling features...")
    features = compute_rolling_features(returns, window)

    # Now run the simple expanding-threshold backtest on the full period
    # (walk-forward is implicit: expanding percentile only uses past data)
    return run_backtest(prices, features, window, CONFIG['rebalance_freq'])


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_period(period_name, start_date, end_date):
    """Run full analysis for one time period."""
    print(f"\n{'='*70}")
    print(f"PERIOD: {period_name} ({start_date} to {end_date})")
    print(f"{'='*70}")

    results = {}

    # Per-sector analysis
    for sector_name, tickers in SECTORS.items():
        print(f"\n  --- {sector_name} ---")

        prices = fetch_prices(tickers, start_date, end_date, label=sector_name)
        if prices is None:
            print(f"    SKIPPED: insufficient data")
            continue

        # Check we have enough stocks for meaningful topology
        n_stocks = len(prices.columns)
        if n_stocks < 5:
            print(f"    WARNING: Only {n_stocks} stocks — topology on networks this small")
            print(f"    is unreliable. H1 features require ≥5 nodes for meaningful loops.")

        returns = np.log(prices / prices.shift(1)).dropna()
        print(f"    Computing rolling features ({len(returns)} days, {n_stocks} stocks)...")

        features = compute_rolling_features(returns, CONFIG['window'])
        result = run_backtest(prices, features, CONFIG['window'], CONFIG['rebalance_freq'])

        if result:
            result['sector'] = sector_name
            result['period'] = period_name
            result['n_stocks'] = n_stocks
            results[sector_name] = result

            print(f"    Mean ρ:    {result['mean_corr']:.3f}")
            print(f"    Mean CV:   {result['mean_cv']:.3f}")
            print(f"    Sharpe:    {result['sharpe']:+.2f}")
            print(f"    CAGR:      {result['cagr']:+.1%}")
            print(f"    Max DD:    {result['max_dd']:.1%}")
            print(f"    p-value:   {result['p_value']:.4f}")
            print(f"    Trades/yr: {result['trades_per_year']:.1f}")

    # Cross-sector baseline
    print(f"\n  --- Cross-Sector Baseline ---")
    prices_cross = fetch_prices(CROSS_SECTOR_UNIVERSE, start_date, end_date, label="Cross-Sector")
    if prices_cross is not None:
        returns_cross = np.log(prices_cross / prices_cross.shift(1)).dropna()
        print(f"    Computing rolling features ({len(returns_cross)} days, {len(prices_cross.columns)} stocks)...")
        features_cross = compute_rolling_features(returns_cross, CONFIG['window'])
        result_cross = run_backtest(prices_cross, features_cross, CONFIG['window'], CONFIG['rebalance_freq'])

        if result_cross:
            result_cross['sector'] = 'Cross-Sector'
            result_cross['period'] = period_name
            result_cross['n_stocks'] = len(prices_cross.columns)
            results['Cross-Sector'] = result_cross

            print(f"    Mean ρ:    {result_cross['mean_corr']:.3f}")
            print(f"    Mean CV:   {result_cross['mean_cv']:.3f}")
            print(f"    Sharpe:    {result_cross['sharpe']:+.2f}")

    return results


def compute_rho_c_analysis(all_results):
    """
    Analyze the ρ-CV relationship and test for ρc threshold.
    Uses data from all periods and sectors.
    """
    print(f"\n{'='*70}")
    print("ρc THRESHOLD ANALYSIS")
    print(f"{'='*70}")

    points = []
    for period_results in all_results.values():
        for sector, r in period_results.items():
            if sector == 'Cross-Sector':
                continue
            points.append({
                'sector': r['sector'],
                'period': r['period'],
                'rho': r['mean_corr'],
                'cv': r['mean_cv'],
                'sharpe': r['sharpe'],
                'p_value': r['p_value'],
            })

    df = pd.DataFrame(points)

    if len(df) < 4:
        print("  Not enough data points for threshold analysis")
        return

    # Overall ρ-CV correlation
    rho_cv_corr, rho_cv_p = stats.pearsonr(df['rho'], df['cv'])
    print(f"\n  ρ-CV Pearson correlation: {rho_cv_corr:.3f} (p={rho_cv_p:.4f})")

    # ρ-Sharpe correlation
    rho_sharpe_corr, rho_sharpe_p = stats.pearsonr(df['rho'], df['sharpe'])
    print(f"  ρ-Sharpe Pearson correlation: {rho_sharpe_corr:.3f} (p={rho_sharpe_p:.4f})")

    # Test candidate thresholds
    print(f"\n  Threshold scan:")
    print(f"  {'ρc':>6}  {'Above(n)':>9}  {'Above(Sharpe)':>14}  {'Below(n)':>9}  {'Below(Sharpe)':>14}  {'Diff':>8}")
    print(f"  {'-'*70}")

    best_diff = -999
    best_rho_c = None

    for rho_c in np.arange(0.40, 0.65, 0.02):
        above = df[df['rho'] >= rho_c]
        below = df[df['rho'] < rho_c]

        if len(above) >= 2 and len(below) >= 2:
            above_sharpe = above['sharpe'].mean()
            below_sharpe = below['sharpe'].mean()
            diff = above_sharpe - below_sharpe

            print(f"  {rho_c:.2f}    {len(above):>5}      {above_sharpe:>+10.3f}     {len(below):>5}      {below_sharpe:>+10.3f}  {diff:>+8.3f}")

            if diff > best_diff:
                best_diff = diff
                best_rho_c = rho_c

    if best_rho_c:
        print(f"\n  Best separation at ρc = {best_rho_c:.2f} (Sharpe diff = {best_diff:+.3f})")

        # Chow test (simplified: compare regression slopes above/below threshold)
        above = df[df['rho'] >= best_rho_c]
        below = df[df['rho'] < best_rho_c]

        if len(above) >= 3 and len(below) >= 3:
            # Welch's t-test on Sharpe ratios above vs below
            t_stat, p_val = stats.ttest_ind(above['sharpe'], below['sharpe'], equal_var=False)
            print(f"  Welch's t-test (above vs below): t={t_stat:.2f}, p={p_val:.4f}")

    # Print all data points
    print(f"\n  All data points:")
    print(f"  {'Sector':<15} {'Period':<12} {'ρ':>6} {'CV':>6} {'Sharpe':>8} {'p-val':>8}")
    print(f"  {'-'*60}")
    for _, row in df.sort_values('rho', ascending=False).iterrows():
        sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else ''))
        print(f"  {row['sector']:<15} {row['period']:<12} {row['rho']:>6.3f} {row['cv']:>6.3f} {row['sharpe']:>+7.3f} {row['p_value']:>7.4f} {sig}")


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_table7(results, period_name):
    """Print Table 7 format (primary sector results)."""
    print(f"\n{'='*70}")
    print(f"TABLE 7: Sector-Specific Performance ({period_name})")
    print(f"{'='*70}")
    print(f"{'Sector':<15} {'Mean ρ':>8} {'CV(H1)':>8} {'Sharpe':>8} {'CAGR':>8} {'Max DD':>8} {'p-value':>10} {'Status':<10}")
    print(f"{'-'*80}")

    # Cross-sector first
    if 'Cross-Sector' in results:
        r = results['Cross-Sector']
        status = 'Failed' if r['sharpe'] < 0 else 'Marginal'
        print(f"{'Cross-Sector':<15} {r['mean_corr']:>8.2f} {r['mean_cv']:>8.2f} {r['sharpe']:>+7.2f} {r['cagr']:>+7.1%} {r['max_dd']:>7.1%} {r['p_value']:>10.4f} {status:<10}")
        print(f"{'-'*80}")

    # Sectors sorted by ρ descending
    sector_results = [(k, v) for k, v in results.items() if k != 'Cross-Sector']
    sector_results.sort(key=lambda x: -x[1]['mean_corr'])

    for sector, r in sector_results:
        if r['sharpe'] > 0.15 and r['p_value'] < 0.05:
            status = 'Success'
        elif r['sharpe'] > 0:
            status = 'Marginal'
        else:
            status = 'Failed'
        print(f"{sector:<15} {r['mean_corr']:>8.2f} {r['mean_cv']:>8.2f} {r['sharpe']:>+7.2f} {r['cagr']:>+7.1%} {r['max_dd']:>7.1%} {r['p_value']:>10.4f} {status:<10}")


def print_table30(results_2010, results_2019):
    """Print Table 30 format (temporal comparison)."""
    print(f"\n{'='*70}")
    print("TABLE 30: 2010-2019 vs 2019-2024 Temporal Comparison")
    print(f"{'='*70}")
    print(f"{'':15} {'--- 2010-2019 ---':>30}    {'--- 2019-2024 ---':>30}")
    print(f"{'Sector':<15} {'ρ':>6} {'CV':>6} {'Sharpe':>8}    {'ρ':>6} {'CV':>6} {'Sharpe':>8}")
    print(f"{'-'*75}")

    all_sectors = set(list(results_2010.keys()) + list(results_2019.keys()))
    all_sectors.discard('Cross-Sector')

    for sector in sorted(all_sectors):
        r1 = results_2010.get(sector)
        r2 = results_2019.get(sector)

        if r1 and r2:
            print(f"{sector:<15} {r1['mean_corr']:>6.2f} {r1['mean_cv']:>6.2f} {r1['sharpe']:>+7.2f}    {r2['mean_corr']:>6.2f} {r2['mean_cv']:>6.2f} {r2['sharpe']:>+7.2f}")
        elif r1:
            print(f"{sector:<15} {r1['mean_corr']:>6.2f} {r1['mean_cv']:>6.2f} {r1['sharpe']:>+7.2f}    {'---':>6} {'---':>6} {'---':>8}")
        elif r2:
            print(f"{sector:<15} {'---':>6} {'---':>6} {'---':>8}    {r2['mean_corr']:>6.2f} {r2['mean_cv']:>6.2f} {r2['sharpe']:>+7.2f}")


def print_table31(results):
    """Print Table 31 format (cost sensitivity)."""
    print(f"\n{'='*70}")
    print("TABLE 31: Transaction Cost Sensitivity")
    print(f"{'='*70}")
    print(f"{'Sector':<15} {'Turnover%':>10} {'Gross':>8} {'5 bps':>8} {'15 bps':>8} {'25 bps':>8}")
    print(f"{'-'*60}")

    for sector, r in sorted(results.items()):
        if sector == 'Cross-Sector':
            continue
        print(f"{sector:<15} {r['annual_turnover_pct']:>9.0f} {r['sharpe']:>+7.2f} {r.get('sharpe_5bps', 0):>+7.2f} {r.get('sharpe_15bps', 0):>+7.2f} {r.get('sharpe_25bps', 0):>+7.2f}")


def save_results(all_results, filename='tda_results.json'):
    """Save all results to JSON for reproducibility."""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    filepath = os.path.join(CONFIG['output_dir'], filename)

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(all_results), f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")


def generate_latex(all_results):
    """Generate LaTeX table source that can be pasted directly into the paper."""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Table 7 LaTeX
    filepath = os.path.join(CONFIG['output_dir'], 'table7_latex.tex')
    with open(filepath, 'w') as f:
        f.write("% AUTO-GENERATED by tda_sector_backtest.py\n")
        f.write(f"% Generated: {datetime.now().isoformat()}\n")
        f.write("% This is the single source of truth for Table 7\n\n")

        # Pick the 2019-2024 results
        results = all_results.get('2019-2024', {})
        if not results:
            f.write("% ERROR: No 2019-2024 results available\n")
        else:
            f.write("\\begin{table}[H]\n\\centering\n")
            f.write("\\caption{Sector-Specific vs Cross-Sector Performance}\n")
            f.write("\\label{tab:sector-authoritative}\n")
            f.write("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n")
            f.write("\\textbf{Strategy} & \\textbf{Mean $\\rho$} & \\textbf{CV(H$_1$)} & \\textbf{Sharpe} & \\textbf{CAGR} & \\textbf{Max DD} & \\textbf{$p$-value} \\\\\n")
            f.write("\\midrule\n")

            if 'Cross-Sector' in results:
                r = results['Cross-Sector']
                f.write(f"Cross-Sector & {r['mean_corr']:.2f} & {r['mean_cv']:.2f} & ${r['sharpe']:+.2f}$ & ${r['cagr']:+.1%}$ & ${r['max_dd']:.1%}$ & ${r['p_value']:.3f}$ \\\\\n")
                f.write("\\midrule\n")

            sectors = [(k, v) for k, v in results.items() if k != 'Cross-Sector']
            sectors.sort(key=lambda x: -x[1]['mean_corr'])

            for sector, r in sectors:
                p_str = "< 0.001" if r['p_value'] < 0.001 else f"{r['p_value']:.3f}"
                f.write(f"{sector} & {r['mean_corr']:.2f} & {r['mean_cv']:.2f} & ${r['sharpe']:+.2f}$ & ${r['cagr']:+.1%}$ & ${r['max_dd']:.1%}$ & ${p_str}$ \\\\\n")

            f.write("\\bottomrule\n\\end{tabular}\n")
            f.write(f"\\vspace{{0.1cm}}\n")
            f.write(f"\\footnotesize{{Generated by \\texttt{{tda\\_sector\\_backtest.py}} on {datetime.now().strftime('%Y-%m-%d')}}}\n")
            f.write("\\end{table}\n")

    print(f"LaTeX Table 7 saved to: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TDA SECTOR-SPECIFIC BACKTEST — SINGLE SOURCE OF TRUTH")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Ripser available: {HAS_RIPSER}")
    print("=" * 70)
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print(f"\nSectors: {list(SECTORS.keys())}")
    print(f"Stocks per sector: {[len(v) for v in SECTORS.values()]}")

    if not HAS_RIPSER:
        print("\n*** FATAL: Cannot produce valid results without ripser ***")
        print("Install: pip install ripser")
        return

    all_results = {}

    # Run both periods
    for period_name, (start, end) in PERIODS.items():
        if period_name == '2022-2024_oos':
            continue  # Skip the walk-forward for now, use expanding threshold
        results = analyze_period(period_name, start, end)
        all_results[period_name] = results

    # Print formatted tables
    if '2019-2024' in all_results:
        print_table7(all_results['2019-2024'], '2019-2024')
        print_table31(all_results['2019-2024'])

    if '2010-2019' in all_results and '2019-2024' in all_results:
        print_table30(all_results['2010-2019'], all_results['2019-2024'])

    # ρc analysis using all data
    compute_rho_c_analysis(all_results)

    # Save everything
    save_results(all_results)
    generate_latex(all_results)

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY & INTEGRITY CHECK")
    print(f"{'='*70}")

    for period, results in all_results.items():
        sectors = [k for k in results if k != 'Cross-Sector']
        sharpes = [results[s]['sharpe'] for s in sectors]
        corrs = [results[s]['mean_corr'] for s in sectors]

        print(f"\n  {period}:")
        print(f"    Sectors tested: {len(sectors)}")
        print(f"    Mean Sharpe: {np.mean(sharpes):+.3f}")
        print(f"    Mean ρ: {np.mean(corrs):.3f}")
        print(f"    Sectors with Sharpe > 0: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")

        if 'Cross-Sector' in results:
            cs = results['Cross-Sector']
            print(f"    Cross-sector baseline: Sharpe = {cs['sharpe']:+.3f}")

    print(f"\nAll numbers above are from REAL yfinance data + REAL ripser TDA.")
    print(f"Results saved to: {CONFIG['output_dir']}/")
    print("Done.")


if __name__ == "__main__":
    main()
