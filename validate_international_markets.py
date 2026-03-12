"""
Real International Market Validation
=====================================

Replaces Section 9 simulated data with ACTUAL market data from:
- FTSE 100 (UK)
- DAX (Germany)
- Nikkei 225 (Japan)
- Cryptocurrency basket

Uses yfinance to pull real correlation and topology data.
Computes actual ρ-CV relationship to validate (or refute) simulation.

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from pandas_datareader import data as pdr
import pandas_datareader as pdr_module

print("=" * 80)
print("REAL INTERNATIONAL MARKET VALIDATION")
print("=" * 80)

# Configuration
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
OUTPUT_DIR = Path('international_validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MARKET DEFINITIONS (Real Tickers)
# ============================================================================

MARKETS = {
    'FTSE_100': {
        'name': 'UK FTSE 100',
        'tickers': [
            'HSBA.L', 'BP.L', 'SHEL.L', 'AZN.L', 'GSK.L',
            'RIO.L', 'ULVR.L', 'DGE.L', 'REL.L', 'NG.L',
            'LLOY.L', 'BARC.L', 'VOD.L', 'PRU.L', 'LSEG.L'
        ]
    },
    'DAX': {
        'name': 'Germany DAX',
        'tickers': [
            'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'BAS.DE',
            'BAYN.DE', 'BMW.DE', 'MUV2.DE', 'DB1.DE', 'VOW3.DE',
            'ADS.DE', 'HEN3.DE', 'CON.DE', 'MRK.DE', 'FRE.DE'
        ]
    },
    'Nikkei_225': {
        'name': 'Japan Nikkei 225',
        'tickers': [
            '7203.T', '6758.T', '9984.T', '6861.T', '8306.T',
            '9432.T', '6501.T', '7267.T', '4503.T', '8035.T',
            '6902.T', '7751.T', '4568.T', '6273.T', '6954.T'
        ]
    },
    'Crypto': {
        'name': 'Cryptocurrency Basket',
        'tickers': [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
            'SOL-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD',
            'AVAX-USD', 'LINK-USD'
        ]
    }
}

# ============================================================================
# DATA DOWNLOAD FUNCTION
# ============================================================================

def download_market_data(market_code, market_info):
    """Download and clean market data"""
    print(f"\n📥 Downloading {market_info['name']}...")

    returns_list = []
    successful_tickers = []
    failed_tickers = []

    for ticker in market_info['tickers']:
        try:
            # Try Yahoo Finance via pandas_datareader
            data = pdr.DataReader(ticker, 'yahoo', START_DATE, END_DATE)

            if len(data) > 100:  # Minimum data requirement
                returns = data['Adj Close'].pct_change().dropna()
                returns.name = ticker
                returns_list.append(returns)
                successful_tickers.append(ticker)
                print(f"  ✅ {ticker}: {len(returns)} days")
            else:
                failed_tickers.append(ticker)
                print(f"  ❌ {ticker}: Insufficient data ({len(data)} days)")

        except Exception as e:
            failed_tickers.append(ticker)
            print(f"  ❌ {ticker}: {str(e)[:50]}")

    if len(returns_list) < 10:
        print(f"  ⚠️  WARNING: Only {len(returns_list)} tickers succeeded (need 10+)")
        return None

    # Combine returns
    returns_df = pd.concat(returns_list, axis=1)
    returns_df = returns_df.dropna(thresh=int(0.8 * len(returns_df.columns)))  # Keep days with 80%+ data

    print(f"  💾 Final: {len(returns_df)} days × {len(returns_df.columns)} stocks")
    print(f"  📊 Success rate: {len(successful_tickers)}/{len(market_info['tickers'])}")

    return returns_df

# ============================================================================
# TOPOLOGY COMPUTATION (Simple Proxy)
# ============================================================================

def compute_topology_metrics(returns_df, window=252):
    """Compute rolling correlation and topology CV proxy"""

    print(f"  🔬 Computing topology metrics (rolling {window}-day window)...")

    cv_values = []
    rho_values = []
    dates = []

    for i in range(window, len(returns_df), 21):  # Monthly steps
        window_data = returns_df.iloc[i-window:i]

        # Correlation matrix
        corr_matrix = window_data.corr()

        # Mean correlation
        upper_tri = np.triu_indices_from(corr_matrix, k=1)
        upper_corrs = corr_matrix.values[upper_tri]
        mean_rho = np.mean(upper_corrs)

        # CV proxy: std of correlations / mean
        std_rho = np.std(upper_corrs)
        cv_proxy = std_rho / mean_rho if mean_rho > 0 else np.nan

        cv_values.append(cv_proxy)
        rho_values.append(mean_rho)
        dates.append(returns_df.index[i])

    results_df = pd.DataFrame({
        'date': dates,
        'mean_rho': rho_values,
        'cv': cv_values
    })

    print(f"  ✅ Computed {len(results_df)} time windows")

    return results_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

all_results = []

for market_code, market_info in MARKETS.items():
    print("\n" + "=" * 80)
    print(f"MARKET: {market_info['name']}")
    print("=" * 80)

    # Download data
    returns_df = download_market_data(market_code, market_info)

    if returns_df is None:
        continue

    # Compute metrics
    metrics_df = compute_topology_metrics(returns_df)

    # Aggregate statistics
    mean_rho = metrics_df['mean_rho'].mean()
    mean_cv = metrics_df['cv'].mean()

    result = {
        'Market': market_info['name'],
        'Market_Code': market_code,
        'Mean_Rho': mean_rho,
        'Mean_CV': mean_cv,
        'N_Stocks': len(returns_df.columns),
        'N_Days': len(returns_df),
        'N_Windows': len(metrics_df)
    }

    all_results.append(result)

    print(f"\n  📊 SUMMARY:")
    print(f"     Mean ρ: {mean_rho:.3f}")
    print(f"     Mean CV: {mean_cv:.3f}")
    print(f"     Stocks: {len(returns_df.columns)}")
    print(f"     Days: {len(returns_df)}")

    # Save data
    returns_df.to_csv(OUTPUT_DIR / f'{market_code}_returns.csv')
    metrics_df.to_csv(OUTPUT_DIR / f'{market_code}_metrics.csv')

# ============================================================================
# CROSS-MARKET ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CROSS-MARKET CORRELATION-CV RELATIONSHIP")
print("=" * 80)

results_df = pd.DataFrame(all_results)
print("\n" + results_df.to_string(index=False))

# Compute global ρ-CV correlation
if len(results_df) >= 3:
    global_corr = np.corrcoef(results_df['Mean_Rho'], results_df['Mean_CV'])[0, 1]

    print(f"\n🔬 CRITICAL FINDING:")
    print(f"   Global ρ-CV Correlation: {global_corr:.3f}")
    print(f"   (Simulation predicted: ρ ≈ -0.97)")

    if global_corr < -0.7:
        print(f"   ✅ STRONG GENERALIZATION (matches simulation)")
    elif -0.7 <= global_corr < -0.5:
        print(f"   🟡 PARTIAL GENERALIZATION (weaker than simulation)")
    else:
        print(f"   ❌ DOES NOT GENERALIZE (simulation was optimistic)")

# Save summary
results_df.to_csv(OUTPUT_DIR / 'international_market_summary.csv', index=False)

print(f"\n💾 Results saved to: {OUTPUT_DIR}/")
print("\n" + "=" * 80)
print("✅ VALIDATION COMPLETE")
print("=" * 80)

print("\nNext steps:")
print("1. Review OUTPUT in international_validation/")
print("2. Compare to Section 9 simulated values")
print("3. Update thesis Section 9 with REAL data")
print("4. Revise abstract/conclusion based on actual results")
