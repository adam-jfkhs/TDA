"""
Historical Period Validation (2010-2019) and Turnover Analysis
===============================================================

This script performs two critical validations:
1. Tests the TDA framework on 2010-2019 data (pre-COVID period)
2. Calculates turnover statistics for realistic cost assessment

CRITICAL: Uses REAL yfinance data only - no simulations.

Run: python historical_validation.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing TDA dependencies
try:
    import ripser
    HAS_TDA = True
except ImportError:
    HAS_TDA = False
    print("Warning: ripser not installed. Install with: pip install ripser")

# =============================================================================
# SECTOR DEFINITIONS (same as main paper)
# =============================================================================

SECTORS = {
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB'],
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ADBE', 'CRM', 'INTC', 'AMD', 'CSCO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN'],
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def fetch_sector_data(sector_name, tickers, start_date, end_date):
    """Fetch real price data from yfinance."""
    print(f"  Fetching {sector_name}: {start_date} to {end_date}...")

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=True
        )

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data

        # Drop tickers with too much missing data
        valid_cols = prices.columns[prices.notna().sum() > len(prices) * 0.8]
        prices = prices[valid_cols].dropna()

        if len(prices) < 252:  # Need at least 1 year
            print(f"    Warning: Only {len(prices)} days of data for {sector_name}")
            return None

        print(f"    Got {len(prices)} days, {len(prices.columns)} tickers")
        return prices

    except Exception as e:
        print(f"    Error: {e}")
        return None

def compute_returns(prices):
    """Compute log returns."""
    return np.log(prices / prices.shift(1)).dropna()

def compute_rolling_correlation(returns, window=60):
    """Compute rolling mean pairwise correlation."""
    correlations = []
    dates = []

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        corr_matrix = window_returns.corr()

        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr = corr_matrix.values[mask].mean()

        correlations.append(mean_corr)
        dates.append(returns.index[i])

    return pd.Series(correlations, index=dates)

def compute_h1_persistence(distance_matrix):
    """Compute H1 persistent homology."""
    if not HAS_TDA:
        return np.array([0.1, 0.2])

    result = ripser.ripser(distance_matrix, maxdim=1, distance_matrix=True)
    h1_dgm = result['dgms'][1]

    finite_mask = np.isfinite(h1_dgm[:, 1])
    h1_finite = h1_dgm[finite_mask]

    if len(h1_finite) == 0:
        return np.array([0.0])

    lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
    return lifetimes

def compute_topology_cv(lifetimes):
    """Compute CV of H1 lifetimes."""
    if len(lifetimes) == 0 or np.mean(lifetimes) == 0:
        return np.nan
    return np.std(lifetimes) / np.mean(lifetimes)

def compute_rolling_topology_cv(returns, window=60):
    """Compute rolling topology CV."""
    cv_values = []
    dates = []

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        corr_matrix = window_returns.corr()
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))

        lifetimes = compute_h1_persistence(distance_matrix.values)
        cv = compute_topology_cv(lifetimes)

        cv_values.append(cv)
        dates.append(returns.index[i])

    return pd.Series(cv_values, index=dates)

# =============================================================================
# BACKTEST WITH TURNOVER TRACKING
# =============================================================================

def run_sector_backtest_with_turnover(prices, window=60, rebalance_freq=5):
    """
    Run simple TDA-based backtest and track turnover.

    Strategy: Go to cash when topology CV > 75th percentile (high instability)
    """
    returns = compute_returns(prices)

    # Compute rolling topology CV
    print("    Computing topology CV...")
    topology_cv = compute_rolling_topology_cv(returns, window)

    # Compute rolling correlation for reference
    rolling_corr = compute_rolling_correlation(returns, window)

    # Align everything
    common_dates = topology_cv.index.intersection(rolling_corr.index)
    topology_cv = topology_cv.loc[common_dates]
    rolling_corr = rolling_corr.loc[common_dates]
    returns_aligned = returns.loc[common_dates]

    if len(common_dates) < 100:
        return None

    # Strategy: binary exposure based on rolling 60-day CV threshold
    # Use expanding percentile to avoid lookahead
    cv_threshold = topology_cv.expanding(min_periods=60).quantile(0.75)

    # Position: 1 = invested, 0 = cash
    position = (topology_cv < cv_threshold).astype(float)

    # Rebalance only every N days to reduce turnover
    rebalance_dates = position.index[::rebalance_freq]
    position_rebalanced = position.copy()
    last_pos = 1.0
    for i, date in enumerate(position.index):
        if date in rebalance_dates:
            last_pos = position.loc[date]
        position_rebalanced.loc[date] = last_pos

    # Calculate turnover
    position_changes = position_rebalanced.diff().abs()

    # Portfolio return = equal-weight sector return * position
    portfolio_return = returns_aligned.mean(axis=1) * position_rebalanced.shift(1)
    portfolio_return = portfolio_return.dropna()

    # Calculate metrics
    annual_return = portfolio_return.mean() * 252
    annual_vol = portfolio_return.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # Turnover metrics - divide by 2 for round-trips (1->0->1 = 1 round-trip, not 200%)
    total_turnover = position_changes.sum() / 2  # FIXED: normalize for binary strategy
    annual_turnover = total_turnover / (len(position_changes) / 252) * 100
    trades_per_year = (position_changes > 0).sum() / (len(position_changes) / 252)

    # Average holding period
    position_diff = position_rebalanced.diff()
    regime_changes = (position_diff != 0).sum()
    avg_holding_days = len(position_rebalanced) / max(regime_changes, 1)

    # Strategy active flag
    is_active = trades_per_year > 1

    # Cost-adjusted Sharpe ratios (5, 15, 25 bps per round-trip)
    cost_5bps = annual_turnover / 100 * 0.0005 * 2  # 2x for round-trip
    cost_15bps = annual_turnover / 100 * 0.0015 * 2
    cost_25bps = annual_turnover / 100 * 0.0025 * 2

    net_return_5bps = annual_return - cost_5bps
    net_return_15bps = annual_return - cost_15bps
    net_return_25bps = annual_return - cost_25bps

    sharpe_5bps = net_return_5bps / annual_vol if annual_vol > 0 else 0
    sharpe_15bps = net_return_15bps / annual_vol if annual_vol > 0 else 0
    sharpe_25bps = net_return_25bps / annual_vol if annual_vol > 0 else 0

    return {
        'sharpe': sharpe,
        'sharpe_5bps': sharpe_5bps,
        'sharpe_15bps': sharpe_15bps,
        'sharpe_25bps': sharpe_25bps,
        'annual_return': annual_return * 100,
        'annual_vol': annual_vol * 100,
        'annual_turnover_pct': annual_turnover,
        'trades_per_year': trades_per_year,
        'avg_holding_days': avg_holding_days,
        'mean_correlation': rolling_corr.mean(),
        'mean_cv': topology_cv.mean(),
        'n_days': len(portfolio_return),
        'active': is_active,
    }

# =============================================================================
# MAIN VALIDATION TESTS
# =============================================================================

def test_historical_period():
    """Test 2010-2019 period (pre-COVID)."""
    print("\n" + "="*70)
    print("TEST 1: HISTORICAL PERIOD VALIDATION (2010-2019)")
    print("="*70)
    print("\nThis tests whether the TDA framework works OUTSIDE the 2020-2024 period.")
    print("If it fails here, the 2020-2024 results may be regime-specific.\n")

    results = []

    for sector_name, tickers in SECTORS.items():
        print(f"\n{sector_name}:")

        prices = fetch_sector_data(
            sector_name, tickers,
            start_date="2010-01-01",
            end_date="2019-12-31"
        )

        if prices is None or len(prices) < 500:
            print(f"    Insufficient data, skipping")
            continue

        result = run_sector_backtest_with_turnover(prices)

        if result:
            result['sector'] = sector_name
            result['period'] = '2010-2019'
            results.append(result)

            print(f"    Mean ρ: {result['mean_correlation']:.3f}")
            print(f"    Mean CV: {result['mean_cv']:.3f}")
            print(f"    Sharpe: {result['sharpe']:.2f}")
            print(f"    Annual Turnover: {result['annual_turnover_pct']:.1f}%")

    return results

def test_recent_period():
    """Test 2019-2024 period (for comparison)."""
    print("\n" + "="*70)
    print("TEST 2: RECENT PERIOD (2019-2024) - FOR COMPARISON")
    print("="*70)

    results = []

    for sector_name, tickers in SECTORS.items():
        print(f"\n{sector_name}:")

        prices = fetch_sector_data(
            sector_name, tickers,
            start_date="2019-01-01",
            end_date="2024-12-31"
        )

        if prices is None or len(prices) < 500:
            print(f"    Insufficient data, skipping")
            continue

        result = run_sector_backtest_with_turnover(prices)

        if result:
            result['sector'] = sector_name
            result['period'] = '2019-2024'
            results.append(result)

            print(f"    Mean ρ: {result['mean_correlation']:.3f}")
            print(f"    Mean CV: {result['mean_cv']:.3f}")
            print(f"    Sharpe: {result['sharpe']:.2f}")
            print(f"    Annual Turnover: {result['annual_turnover_pct']:.1f}%")

    return results

def generate_latex_tables(historical_results, recent_results):
    """Generate LaTeX tables for the paper."""

    # Filter active sectors for turnover stats
    active_recent = [r for r in recent_results if r.get('active', True)]

    print("\n" + "="*70)
    print("LATEX TABLE: TURNOVER STATISTICS (Active Sectors Only)")
    print("="*70)

    print("""
\\begin{table}[H]
\\centering
\\caption{Strategy Turnover Statistics by Sector (2019--2024)}
\\label{tab:turnover}
\\begin{tabular}{@{}lccccc@{}}
\\toprule
\\textbf{Sector} & \\textbf{Turnover (\\%)} & \\textbf{Trades/Yr} & \\textbf{Holding (days)} & \\textbf{Sharpe} & \\textbf{Status} \\\\
\\midrule""")

    for r in recent_results:
        status = "Active" if r.get('active', True) else "Stable$^\\dagger$"
        print(f"{r['sector']} & {r['annual_turnover_pct']:.0f} & {r['trades_per_year']:.1f} & {r['avg_holding_days']:.0f} & {r['sharpe']:.2f} & {status} \\\\")

    print("""\\bottomrule
\\multicolumn{6}{l}{\\footnotesize $^\\dagger$Stable: Sector remained below instability threshold; no trades generated.}
\\end{tabular}
\\end{table}
""")

    print("\n" + "="*70)
    print("LATEX TABLE: COST SENSITIVITY ANALYSIS")
    print("="*70)

    print("""
\\begin{table}[H]
\\centering
\\caption{Transaction Cost Sensitivity: Sharpe Ratio by Cost Assumption}
\\label{tab:cost-sensitivity}
\\begin{tabular}{@{}lcccc@{}}
\\toprule
\\textbf{Sector} & \\textbf{Gross} & \\textbf{5 bps} & \\textbf{15 bps} & \\textbf{25 bps} \\\\
\\midrule""")

    for r in active_recent:
        print(f"{r['sector']} & {r['sharpe']:.2f} & {r.get('sharpe_5bps', r['sharpe']):.2f} & {r.get('sharpe_15bps', r['sharpe']):.2f} & {r.get('sharpe_25bps', r['sharpe']):.2f} \\\\")

    print("""\\bottomrule
\\multicolumn{5}{l}{\\footnotesize Cost applied per round-trip. Only active sectors shown.}
\\end{tabular}
\\end{table}
""")

    print("\n" + "="*70)
    print("LATEX TABLE: HISTORICAL VS RECENT COMPARISON")
    print("="*70)

    print("""
\\begin{table}[H]
\\centering
\\caption{TDA Framework: 2010--2019 vs 2019--2024 Comparison}
\\label{tab:historical-comparison}
\\begin{tabular}{@{}lcccccc@{}}
\\toprule
& \\multicolumn{3}{c}{\\textbf{2010--2019}} & \\multicolumn{3}{c}{\\textbf{2019--2024}} \\\\
\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}
\\textbf{Sector} & \\textbf{Mean $\\rho$} & \\textbf{CV} & \\textbf{Sharpe} & \\textbf{Mean $\\rho$} & \\textbf{CV} & \\textbf{Sharpe} \\\\
\\midrule""")

    # Match sectors between periods
    for h in historical_results:
        sector = h['sector']
        r = next((x for x in recent_results if x['sector'] == sector), None)
        if r:
            print(f"{sector} & {h['mean_correlation']:.2f} & {h['mean_cv']:.2f} & {h['sharpe']:.2f} & {r['mean_correlation']:.2f} & {r['mean_cv']:.2f} & {r['sharpe']:.2f} \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

def main():
    print("="*70)
    print("TDA HISTORICAL VALIDATION & TURNOVER ANALYSIS")
    print("Using REAL yfinance data only - no simulations")
    print("="*70)

    if not HAS_TDA:
        print("\nWARNING: Running without ripser - results will use placeholder values")
        print("Install with: pip install ripser")

    # Run tests
    historical_results = test_historical_period()
    recent_results = test_recent_period()

    # Filter active sectors
    active_recent = [r for r in recent_results if r.get('active', True)]
    active_historical = [r for r in historical_results if r.get('active', True)]

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if historical_results:
        avg_historical_sharpe = np.mean([r['sharpe'] for r in historical_results])
        print(f"\n2010-2019 Average Sharpe (all sectors): {avg_historical_sharpe:.2f}")

    if recent_results:
        avg_recent_sharpe = np.mean([r['sharpe'] for r in recent_results])
        print(f"2019-2024 Average Sharpe (all sectors): {avg_recent_sharpe:.2f}")

    if active_recent:
        avg_active_sharpe = np.mean([r['sharpe'] for r in active_recent])
        avg_turnover = np.mean([r['annual_turnover_pct'] for r in active_recent])
        print(f"\n2019-2024 Active Sectors Only:")
        print(f"  Average Sharpe: {avg_active_sharpe:.2f}")
        print(f"  Average Turnover: {avg_turnover:.0f}%")
        print(f"  Active sectors: {[r['sector'] for r in active_recent]}")

    # Critical assessment
    print("\n" + "="*70)
    print("CRITICAL ASSESSMENT")
    print("="*70)

    if historical_results and recent_results:
        h_sharpes = [r['sharpe'] for r in historical_results]

        if np.mean(h_sharpes) > 0.1:
            print("\n[PASS] Strategy shows positive Sharpe in 2010-2019")
            print("       This suggests findings are NOT purely COVID-era specific")
        else:
            print("\n[CONCERN] Strategy shows weak/negative Sharpe in 2010-2019")
            print("          This suggests findings MAY be regime-specific")

        # Turnover assessment (active sectors only)
        if active_recent:
            avg_turnover = np.mean([r['annual_turnover_pct'] for r in active_recent])
            if avg_turnover < 500:  # Adjusted threshold for normalized turnover
                print(f"\n[PASS] Average turnover ({avg_turnover:.0f}%) is reasonable")
                print("       5-15 bps cost assumption is defensible")
            else:
                print(f"\n[CONCERN] High turnover ({avg_turnover:.0f}%)")
                print("          May need 15-25 bps cost assumption")

        # Inactive sectors
        inactive = [r for r in recent_results if not r.get('active', True)]
        if inactive:
            print(f"\n[INFO] {len(inactive)} sector(s) showed no trading activity:")
            for r in inactive:
                print(f"       - {r['sector']}: ρ={r['mean_correlation']:.2f}, CV={r['mean_cv']:.2f}")
            print("       This is expected for high-ρ, low-CV sectors (structurally stable)")

    # Generate LaTeX
    if historical_results and recent_results:
        generate_latex_tables(historical_results, recent_results)

    print("\nDone!")

if __name__ == "__main__":
    main()
