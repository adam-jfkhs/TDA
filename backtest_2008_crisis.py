"""
2008-2009 Crisis Backtest
==========================

Critical stress test: Does the ρ ≥ 0.50 threshold hold when
correlations spike to 0.95+ during systemic crisis?

Tests sector-specific topology strategy on:
- Pre-crisis: 2007
- Crisis peak: 2008-2009
- Recovery: 2010-2011

Hypothesis: Strategy fails when ALL correlations → 1.0 (no sector differentiation)

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("2008-2009 CRISIS BACKTEST")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

START_DATE = '2007-01-01'
END_DATE = '2011-12-31'

SECTORS = {
    'Financials': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SPGI',
                   'AXP', 'BK', 'USB', 'PNC', 'TFC', 'COF', 'SCHW'],
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'ORCL', 'CSCO', 'INTC',
                   'IBM', 'QCOM', 'TXN', 'ADI', 'AMAT', 'LRCX', 'KLAC'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX',
               'PXD', 'OXY', 'KMI', 'WMB', 'HAL', 'DVN'],
}

OUTPUT_DIR = Path('crisis_2008_backtest')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_sector_data(sector_name, tickers):
    """Download sector data for crisis period"""
    print(f"\n📥 Downloading {sector_name}...")

    returns_list = []
    successful = []

    for ticker in tickers:
        try:
            data = pdr.DataReader(ticker, 'yahoo', START_DATE, END_DATE)

            if len(data) > 500:  # Minimum 2+ years
                returns = data['Adj Close'].pct_change()
                returns.name = ticker
                returns_list.append(returns)
                successful.append(ticker)
                print(f"  ✅ {ticker}: {len(returns)} days")
            else:
                print(f"  ❌ {ticker}: Insufficient data")

        except Exception as e:
            print(f"  ❌ {ticker}: {str(e)[:50]}")

    if len(returns_list) < 10:
        print(f"  ⚠️  WARNING: Only {len(returns_list)} stocks (need 10+)")
        return None

    returns_df = pd.concat(returns_list, axis=1).dropna()
    print(f"  💾 {sector_name}: {len(returns_df)} days × {len(returns_df.columns)} stocks")

    return returns_df

# ============================================================================
# ROLLING METRICS
# ============================================================================

def compute_rolling_metrics(returns_df, window=252):
    """Compute rolling correlation and CV"""

    metrics = {
        'date': [],
        'mean_rho': [],
        'cv': [],
        'max_rho': [],  # Track correlation spike
        'min_rho': []
    }

    for i in range(window, len(returns_df), 21):  # Monthly
        window_data = returns_df.iloc[i-window:i]
        corr_matrix = window_data.corr()

        # Mean correlation
        upper_tri = np.triu_indices_from(corr_matrix, k=1)
        corrs = corr_matrix.values[upper_tri]
        mean_rho = np.mean(corrs)

        # CV
        cv = np.std(corrs) / mean_rho if mean_rho > 0 else np.nan

        metrics['date'].append(returns_df.index[i])
        metrics['mean_rho'].append(mean_rho)
        metrics['cv'].append(cv)
        metrics['max_rho'].append(np.max(corrs))
        metrics['min_rho'].append(np.min(corrs))

    return pd.DataFrame(metrics)

# ============================================================================
# CRISIS PERIOD ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CRISIS PERIOD BREAKDOWN")
print("=" * 80)

PERIODS = {
    'Pre-Crisis': ('2007-01-01', '2007-12-31'),
    'Crisis Peak': ('2008-01-01', '2009-06-30'),
    'Recovery': ('2009-07-01', '2011-12-31')
}

all_sector_results = {}

for sector_name, tickers in SECTORS.items():
    print(f"\n{'='*80}")
    print(f"SECTOR: {sector_name}")
    print(f"{'='*80}")

    # Download
    returns_df = download_sector_data(sector_name, tickers)

    if returns_df is None:
        continue

    # Compute metrics
    metrics_df = compute_rolling_metrics(returns_df, window=252)

    # Analyze by period
    period_results = []

    for period_name, (start, end) in PERIODS.items():
        period_data = metrics_df[
            (metrics_df['date'] >= start) &
            (metrics_df['date'] <= end)
        ]

        if len(period_data) == 0:
            continue

        result = {
            'Sector': sector_name,
            'Period': period_name,
            'Mean_Rho': period_data['mean_rho'].mean(),
            'Mean_CV': period_data['cv'].mean(),
            'Max_Rho': period_data['max_rho'].max(),
            'N_Windows': len(period_data)
        }

        period_results.append(result)

        print(f"\n  📊 {period_name}:")
        print(f"     Mean ρ: {result['Mean_Rho']:.3f}")
        print(f"     Mean CV: {result['Mean_CV']:.3f}")
        print(f"     Max ρ: {result['Max_Rho']:.3f}")

        # Critical threshold check
        if result['Mean_Rho'] >= 0.50:
            print(f"     ✅ ABOVE THRESHOLD (ρ ≥ 0.50) → Predicted viable")
        else:
            print(f"     ❌ BELOW THRESHOLD (ρ < 0.50) → Predicted non-viable")

    all_sector_results[sector_name] = pd.DataFrame(period_results)

    # Save
    metrics_df.to_csv(OUTPUT_DIR / f'{sector_name.lower()}_2008_metrics.csv', index=False)

# ============================================================================
# CROSS-PERIOD COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("CROSS-PERIOD COMPARISON")
print("=" * 80)

# Combine all sectors
combined_results = pd.concat(all_sector_results.values(), ignore_index=True)

print("\n" + combined_results.to_string(index=False))

# Pivot table
pivot = combined_results.pivot_table(
    index='Period',
    columns='Sector',
    values='Mean_Rho',
    aggfunc='mean'
)

print("\n📊 Mean Correlation by Period and Sector:")
print(pivot.round(3).to_string())

# ============================================================================
# CRITICAL FINDING
# ============================================================================

print("\n" + "=" * 80)
print("CRITICAL FINDING: ρ THRESHOLD DURING CRISIS")
print("=" * 80)

crisis_data = combined_results[combined_results['Period'] == 'Crisis Peak']

if len(crisis_data) > 0:
    crisis_mean_rho = crisis_data['Mean_Rho'].mean()
    crisis_max_rho = crisis_data['Max_Rho'].max()

    print(f"\n🔥 Crisis Peak (2008-2009):")
    print(f"   Avg sector ρ: {crisis_mean_rho:.3f}")
    print(f"   Max correlation: {crisis_max_rho:.3f}")

    if crisis_mean_rho >= 0.50:
        print(f"\n   ✅ SECTORS REMAIN ABOVE THRESHOLD")
        print(f"   Strategy should still work (sector differentiation persists)")
    else:
        print(f"\n   ❌ SECTORS FALL BELOW THRESHOLD")
        print(f"   Strategy likely fails (correlations collapse)")

    if crisis_max_rho >= 0.95:
        print(f"\n   ⚠️  WARNING: Correlations spiked to {crisis_max_rho:.3f}")
        print(f"   All assets moving together → no diversification → strategy risk")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("IMPLICATIONS FOR THESIS")
print("=" * 80)

print("\n1. If sectors stayed above ρ ≥ 0.50 during 2008:")
print("   → Add to Section 7: 'Threshold robust even during systemic crisis'")
print("   → Strengthens generalization claim")

print("\n2. If sectors fell below ρ < 0.50:")
print("   → Add to Limitations: 'Strategy may fail during extreme correlations'")
print("   → Propose regime-aware position sizing")

print("\n3. If correlations spiked to 0.95+:")
print("   → Add to Section 11: 'Crisis regime detection via correlation spike'")
print("   → Use topology as 'circuit breaker' signal")

combined_results.to_csv(OUTPUT_DIR / 'crisis_period_summary.csv', index=False)

print("\n💾 Results saved to:", OUTPUT_DIR)

print("\n" + "=" * 80)
print("✅ 2008 CRISIS BACKTEST COMPLETE")
print("=" * 80)
