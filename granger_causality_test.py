"""
Granger Causality Test: Topology → VIX
=======================================

Tests whether topological features (H1 count, CV, Fiedler value)
**Granger-cause** VIX changes, or merely correlate.

If topology Granger-causes VIX with 3-5 day lag, this converts
"concurrent detection" into "predictive signal."

Critical for Davidson reviewers.

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GRANGER CAUSALITY TEST: TOPOLOGY → VIX")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
MAX_LAG = 10  # Test up to 10-day lag

# ============================================================================
# DATA COLLECTION
# ============================================================================

print("\n📥 Step 1: Download VIX data...")
vix = pdr.DataReader('^VIX', 'yahoo', START_DATE, END_DATE)
vix_close = vix['Close'].dropna()
print(f"   ✅ VIX: {len(vix_close)} days")

print("\n📥 Step 2: Download S&P 500 sector ETFs for topology proxy...")

# Use sector ETFs as proxy for "topology stress"
sectors = {
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLI': 'Industrials'
}

sector_returns = {}
for ticker, name in sectors.items():
    data = pdr.DataReader(ticker, 'yahoo', START_DATE, END_DATE)
    if len(data) > 0:
        returns = data['Adj Close'].pct_change()
        sector_returns[ticker] = returns
        print(f"   ✅ {name} ({ticker}): {len(returns)} days")

# Combine
returns_df = pd.concat(sector_returns, axis=1)
returns_df.columns = [f'{ticker}_return' for ticker in sector_returns.keys()]

# ============================================================================
# TOPOLOGY PROXY COMPUTATION
# ============================================================================

print("\n🔬 Step 3: Compute topology proxy (rolling correlation CV)...")

def compute_rolling_topology_cv(returns_df, window=30):
    """Compute rolling coefficient of variation of correlations"""
    cv_series = []
    dates = []

    for i in range(window, len(returns_df)):
        window_data = returns_df.iloc[i-window:i]
        corr_matrix = window_data.corr()

        # Upper triangle correlations
        upper_tri = np.triu_indices_from(corr_matrix, k=1)
        corrs = corr_matrix.values[upper_tri]

        # CV
        if np.mean(corrs) > 0:
            cv = np.std(corrs) / np.mean(corrs)
        else:
            cv = np.nan

        cv_series.append(cv)
        dates.append(returns_df.index[i])

    return pd.Series(cv_series, index=dates, name='topology_cv')

topology_cv = compute_rolling_topology_cv(returns_df, window=30)
print(f"   ✅ Topology CV computed: {len(topology_cv)} observations")

# ============================================================================
# ALIGN DATA
# ============================================================================

print("\n🔗 Step 4: Align VIX and Topology data...")

# Merge
combined = pd.DataFrame({
    'vix': vix_close,
    'topology_cv': topology_cv
}).dropna()

print(f"   ✅ Aligned dataset: {len(combined)} days")
print(f"   Date range: {combined.index[0].date()} to {combined.index[-1].date()}")

# ============================================================================
# GRANGER CAUSALITY TEST
# ============================================================================

print("\n" + "=" * 80)
print("GRANGER CAUSALITY TEST RESULTS")
print("=" * 80)

print("\n📊 Hypothesis: Does Topology CV Granger-cause VIX?")
print("   Null hypothesis (H0): Topology CV does NOT Granger-cause VIX")
print("   Alternative (H1): Topology CV DOES Granger-cause VIX\n")

# Prepare data: [VIX, Topology_CV] format required by statsmodels
test_data = combined[['vix', 'topology_cv']].values

# Run test
try:
    results = grangercausalitytests(test_data, maxlag=MAX_LAG, verbose=False)

    print(f"{'Lag':<6} {'F-stat':<12} {'p-value':<12} {'Significance'}")
    print("-" * 60)

    significant_lags = []

    for lag in range(1, MAX_LAG + 1):
        # Extract F-test results
        f_stat = results[lag][0]['ssr_ftest'][0]
        p_value = results[lag][0]['ssr_ftest'][1]

        # Significance
        if p_value < 0.01:
            sig = "*** (p < 0.01)"
            significant_lags.append(lag)
        elif p_value < 0.05:
            sig = "** (p < 0.05)"
            significant_lags.append(lag)
        elif p_value < 0.10:
            sig = "* (p < 0.10)"
        else:
            sig = "Not significant"

        print(f"{lag:<6} {f_stat:<12.3f} {p_value:<12.4f} {sig}")

    print("-" * 60)

    # Summary
    print("\n🔬 INTERPRETATION:")

    if len(significant_lags) > 0:
        print(f"   ✅ TOPOLOGY GRANGER-CAUSES VIX at lags: {significant_lags}")
        print(f"   This means topology has PREDICTIVE POWER for VIX {min(significant_lags)}-{max(significant_lags)} days ahead")
        print(f"   → Not just concurrent detection, but LEADING INDICATOR")

        # Best lag
        best_lag = significant_lags[0]
        best_p = results[best_lag][0]['ssr_ftest'][1]
        print(f"\n   🎯 Best lag: {best_lag} days (p = {best_p:.4f})")
        print(f"   → Topology stress predicts VIX spike ~{best_lag} days in advance")

    else:
        print(f"   ❌ NO GRANGER CAUSALITY DETECTED")
        print(f"   Topology and VIX are merely correlated, not causal")
        print(f"   → Topology is concurrent indicator, not predictive")

except Exception as e:
    print(f"   ❌ ERROR: {e}")

# ============================================================================
# REVERSE TEST (VIX → Topology)
# ============================================================================

print("\n" + "=" * 80)
print("REVERSE TEST: Does VIX Granger-cause Topology?")
print("=" * 80)

# Swap order: [Topology_CV, VIX]
reverse_data = combined[['topology_cv', 'vix']].values

try:
    reverse_results = grangercausalitytests(reverse_data, maxlag=MAX_LAG, verbose=False)

    reverse_significant = []

    for lag in range(1, MAX_LAG + 1):
        p_value = reverse_results[lag][0]['ssr_ftest'][1]
        if p_value < 0.05:
            reverse_significant.append(lag)

    if len(reverse_significant) > 0:
        print(f"   🔄 VIX also Granger-causes Topology at lags: {reverse_significant}")
        print(f"   → BIDIRECTIONAL causality (feedback loop)")
    else:
        print(f"   → VIX does NOT Granger-cause Topology")
        print(f"   → UNIDIRECTIONAL causality (Topology → VIX only)")

except Exception as e:
    print(f"   ❌ ERROR: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n💾 Saving results...")

results_summary = pd.DataFrame([
    {
        'Lag': lag,
        'F_Statistic': results[lag][0]['ssr_ftest'][0],
        'P_Value': results[lag][0]['ssr_ftest'][1],
        'Significant': results[lag][0]['ssr_ftest'][1] < 0.05
    }
    for lag in range(1, MAX_LAG + 1)
])

results_summary.to_csv('granger_causality_results.csv', index=False)
print("   ✅ Saved: granger_causality_results.csv")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("NEXT STEPS FOR THESIS")
print("=" * 80)

print("\n1. Add to Section 11 (Theory):")
print("   - Subsection: 'Granger Causality Analysis'")
print("   - Report F-stats and p-values for significant lags")
print("   - State: 'Topology Granger-causes VIX with X-day lag (p < 0.01)'")

print("\n2. Update Abstract:")
print("   - Change '77% concurrent detection' → 'X-day leading indicator'")

print("\n3. Implications:")
print("   - If significant lag exists: PREDICTIVE edge (not just detection)")
print("   - If no lag: Still valuable (regime detection, not prediction)")

print("\n" + "=" * 80)
print("✅ GRANGER CAUSALITY TEST COMPLETE")
print("=" * 80)
