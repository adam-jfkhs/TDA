"""
Phase 4: Cryptocurrency Market Analysis (FIXED)
===============================================

Tests if topology works in high-volatility, 24/7 crypto markets.

Author: Adam Levine
Date: January 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from ripser import ripser
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 4: CRYPTOCURRENCY MARKET ANALYSIS")
print("=" * 80)

# Configuration
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
DATA_DIR = Path('data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Major cryptocurrencies (top by market cap)
CRYPTO_TICKERS = [
    'BTC-USD',   # Bitcoin
    'ETH-USD',   # Ethereum
    'BNB-USD',   # Binance Coin
    'XRP-USD',   # Ripple
    'ADA-USD',   # Cardano
    'DOGE-USD',  # Dogecoin
    'SOL-USD',   # Solana
    'MATIC-USD', # Polygon
    'DOT-USD',   # Polkadot
    'LTC-USD',   # Litecoin
    'AVAX-USD',  # Avalanche
    'LINK-USD',  # Chainlink
]

print("\n₿ Downloading cryptocurrency data...")
print(f"Period: {START_DATE} to {END_DATE}")

returns_list = []
successful_tickers = []

for ticker in CRYPTO_TICKERS:
    try:
        print(f"  Downloading {ticker}...", end=' ')

        # Fixed: Use Ticker object first, then history
        crypto = yf.Ticker(ticker)
        data = crypto.history(start=START_DATE, end=END_DATE)

        if len(data) > 100:
            returns = data['Close'].pct_change().dropna()
            returns_list.append(returns)
            successful_tickers.append(ticker)
            print(f"✅ {len(data)} days")
        else:
            print(f"❌ Insufficient data ({len(data)} days)")
    except Exception as e:
        print(f"❌ Failed: {str(e)[:50]}")

if len(successful_tickers) < 8:
    print(f"\n❌ Insufficient cryptocurrencies ({len(successful_tickers)} < 8)")
    print("\n⚠️  For your thesis: You can mention attempted crypto analysis")
    print("    or use simulated data based on expected correlations")
    exit(1)

# Combine into dataframe
returns_df = pd.concat(returns_list, axis=1, keys=successful_tickers)
returns_df = returns_df.dropna()

print(f"\n✅ Successfully downloaded {len(successful_tickers)} cryptocurrencies")
print(f"   Total observations: {len(returns_df)} days")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CRYPTOCURRENCY CORRELATION ANALYSIS")
print("=" * 80)

corr_matrix = returns_df.corr()

# Mean pairwise correlation
upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
mean_corr = upper_tri.stack().mean()

print(f"\nMean Pairwise Correlation: {mean_corr:.3f}")

# Compare to US tech stocks (expected ~0.58)
us_tech_corr = 0.578  # From Phase 2
diff = mean_corr - us_tech_corr

print(f"US Technology Sector:      {us_tech_corr:.3f}")
print(f"Difference:                {diff:+.3f} ({diff/us_tech_corr*100:+.1f}%)")

if mean_corr < 0.45:
    print("\n⚠️  Lower correlations than equities (expected for crypto)")
elif mean_corr > 0.55:
    print("\n✅ High correlations (similar to equity sectors)")
else:
    print("\n🟡 Moderate correlations")

# Volatility
volatility = returns_df.std() * np.sqrt(365)  # 365 days for crypto (24/7)
mean_vol = volatility.mean()

print(f"\nMean Annualized Volatility: {mean_vol:.1f}%")
print(f"US Technology Volatility:   28.4%")
print(f"Crypto / Equity Ratio:      {mean_vol/28.4:.1f}×")

# ============================================================================
# TOPOLOGY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CRYPTOCURRENCY TOPOLOGY ANALYSIS")
print("=" * 80)

# Distance matrix
distance_matrix = np.sqrt(2 * (1 - corr_matrix))

# Compute persistent homology
print("\nComputing persistent homology...")
result = ripser(distance_matrix, distance_matrix=True, maxdim=1)

diagrams = result['dgms']

# Extract H0 and H1 persistence
h0_persistence = diagrams[0][:, 1] - diagrams[0][:, 0]
h1_persistence = diagrams[1][:, 1] - diagrams[1][:, 0]

# Count significant features
h0_count = np.sum(h0_persistence > 0.1)
h1_count = np.sum(h1_persistence > 0.1)

# Statistics
mean_h1 = h1_count
std_h1 = np.std(h1_persistence[h1_persistence > 0.1]) if len(h1_persistence[h1_persistence > 0.1]) > 0 else 0
cv = std_h1 / mean_h1 if mean_h1 > 0 else 0

print(f"\nTopology Features:")
print(f"  H₀ (Components): {h0_count}")
print(f"  H₁ (Loops):      {h1_count}")
print(f"  CV:              {cv:.3f}")

# Compare to US tech
us_tech_cv = 0.451  # From Phase 2
print(f"\nComparison to US Technology:")
print(f"  US Tech CV:      {us_tech_cv:.3f}")
print(f"  Crypto CV:       {cv:.3f}")
print(f"  Difference:      {cv - us_tech_cv:+.3f} ({(cv - us_tech_cv)/us_tech_cv*100:+.1f}%)")

# ============================================================================
# PREDICTION TEST
# ============================================================================

print("\n" + "=" * 80)
print("TESTING SECTION 7 PREDICTION")
print("=" * 80)

# From Section 7: correlation-CV relationship
# ρ_correlation_cv = -0.87
# Using regression: CV ≈ -1.5 * correlation + 1.32

predicted_cv = -1.5 * mean_corr + 1.32
prediction_error = abs(cv - predicted_cv)

print(f"\nBased on Section 7 correlation-CV relationship:")
print(f"  Given correlation: {mean_corr:.3f}")
print(f"  Predicted CV:      {predicted_cv:.3f} ± 0.10")
print(f"  Actual CV:         {cv:.3f}")
print(f"  Prediction error:  {prediction_error:.3f}")

if prediction_error < 0.15:
    print(f"\n✅ PREDICTION VALIDATED! Error within ±0.15")
    print("   → Section 7 relationship generalizes to crypto!")
else:
    print(f"\n⚠️  Prediction off by {prediction_error:.3f}")
    print("   → Crypto may have different topology dynamics")

# ============================================================================
# TRADING VIABILITY
# ============================================================================

print("\n" + "=" * 80)
print("TRADING VIABILITY ASSESSMENT")
print("=" * 80)

print(f"\nCriteria from Section 7:")
print(f"  ✅ Good:     Correlation > 0.5 AND CV < 0.6")
print(f"  🟡 Marginal: Correlation > 0.4 OR CV < 0.7")
print(f"  ❌ Poor:     Correlation < 0.4 AND CV > 0.7")

print(f"\nCryptocurrency Assessment:")
print(f"  Correlation: {mean_corr:.3f}")
print(f"  CV:          {cv:.3f}")

if mean_corr > 0.5 and cv < 0.6:
    status = "✅ GOOD - Ready for TDA trading"
    recommendation = "Can use standard sector-specific approach"
elif mean_corr > 0.4 or cv < 0.7:
    status = "🟡 MARGINAL - Needs adaptation"
    recommendation = "Use longer lookback (90 days), adaptive thresholds"
else:
    status = "❌ POOR - Not recommended"
    recommendation = "Focus on higher-correlation crypto groups only"

print(f"\nStatus: {status}")
print(f"Recommendation: {recommendation}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save topology features
topology_data = {
    'date': [returns_df.index[-1]],
    'h0_count': [h0_count],
    'h1_count': [h1_count],
    'cv': [cv],
    'mean_correlation': [mean_corr],
    'mean_volatility': [mean_vol]
}

topology_df = pd.DataFrame(topology_data)
topology_df.to_csv(DATA_DIR / 'cryptocurrency_topology.csv')

# Save returns
returns_df.to_csv(DATA_DIR / 'cryptocurrency_returns.csv')

# Save correlation matrix
corr_matrix.to_csv(DATA_DIR / 'cryptocurrency_correlations.csv')

print(f"\n💾 Files saved:")
print(f"   - cryptocurrency_topology.csv")
print(f"   - cryptocurrency_returns.csv")
print(f"   - cryptocurrency_correlations.csv")

print("\n" + "=" * 80)
print("CRYPTOCURRENCY ANALYSIS COMPLETE")
print("=" * 80)

print("\nKey Findings:")
print(f"  1. Crypto correlations {diff/us_tech_corr*100:+.1f}% {'lower' if diff < 0 else 'higher'} than tech equities")
print(f"  2. Volatility {mean_vol/28.4:.1f}× higher than equities")
print(f"  3. Topology CV is {cv:.3f} (vs {us_tech_cv:.3f} for tech)")
print(f"  4. Trading viability: {status.split('-')[1].strip()}")

print("\nNext: Run 03_cross_market_comparison.py")
