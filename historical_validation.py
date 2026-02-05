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

    # Turnover metrics
    total_turnover = position_changes.sum()
    annual_turnover = total_turnover / (len(position_changes) / 252) * 100
    trades_per_year = (position_changes > 0).sum() / (len(position_changes) / 252)

    # Average holding period
    position_diff = position_rebalanced.diff()
    regime_changes = (position_diff != 0).sum()
    avg_holding_days = len(position_rebalanced) / max(regime_changes, 1)

    return {
        'sharpe': sharpe,
        'annual_return': annual_return * 100,
        'annual_vol': annual_vol * 100,
        'annual_turnover_pct': annual_turnover,
        'trades_per_year': trades_per_year,
        'avg_holding_days': avg_holding_days,
        'mean_correlation': rolling_corr.mean(),
        'mean_cv': topology_cv.mean(),
        'n_days': len(portfolio_return),
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

    print("\n" + "="*70)
    print("LATEX TABLE: TURNOVER STATISTICS")
    print("="*70)

    print("""
\\begin{table}[H]
\\centering
\\caption{Strategy Turnover Statistics by Sector}
\\label{tab:turnover}
\\begin{tabular}{@{}lcccc@{}}
\\toprule
\\textbf{Sector} & \\textbf{Annual Turnover (\\%)} & \\textbf{Trades/Year} & \\textbf{Avg Holding (days)} & \\textbf{Sharpe} \\\\
\\midrule""")

    for r in recent_results:
        print(f"{r['sector']} & {r['annual_turnover_pct']:.1f} & {r['trades_per_year']:.1f} & {r['avg_holding_days']:.0f} & {r['sharpe']:.2f} \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    print("\n" + "="*70)
    print("LATEX TABLE: HISTORICAL VS RECENT COMPARISON")
    print("="*70)

    print("""
\\begin{table}[H]
\\centering
\\caption{TDA Framework: 2010-2019 vs 2019-2024 Comparison}
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

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if historical_results:
        avg_historical_sharpe = np.mean([r['sharpe'] for r in historical_results])
        print(f"\n2010-2019 Average Sharpe: {avg_historical_sharpe:.2f}")

    if recent_results:
        avg_recent_sharpe = np.mean([r['sharpe'] for r in recent_results])
        avg_turnover = np.mean([r['annual_turnover_pct'] for r in recent_results])
        print(f"2019-2024 Average Sharpe: {avg_recent_sharpe:.2f}")
        print(f"Average Annual Turnover: {avg_turnover:.1f}%")

    # Critical assessment
    print("\n" + "="*70)
    print("CRITICAL ASSESSMENT")
    print("="*70)

    if historical_results and recent_results:
        h_sharpes = [r['sharpe'] for r in historical_results]
        r_sharpes = [r['sharpe'] for r in recent_results]

        if np.mean(h_sharpes) > 0.1:
            print("\n[PASS] Strategy shows positive Sharpe in 2010-2019")
            print("       This suggests findings are NOT purely COVID-era specific")
        else:
            print("\n[CONCERN] Strategy shows weak/negative Sharpe in 2010-2019")
            print("          This suggests findings MAY be regime-specific")

        # Turnover assessment
        avg_turnover = np.mean([r['annual_turnover_pct'] for r in recent_results])
        if avg_turnover < 200:
            print(f"\n[PASS] Average turnover ({avg_turnover:.0f}%) supports 5bps cost assumption")
        else:
            print(f"\n[CONCERN] High turnover ({avg_turnover:.0f}%) - may need higher cost assumption")

    # Generate LaTeX
    if historical_results and recent_results:
        generate_latex_tables(historical_results, recent_results)

    print("\nDone!")

if __name__ == "__main__":
    main()
