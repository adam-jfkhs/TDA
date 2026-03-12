# International Validation Output Directory

This directory contains output from `validate_international_markets.py`.

## Expected Files (After Running Script)

When you run `validate_international_markets.py` in an environment with internet access, this directory will contain:

### Market-Specific Files

- `FTSE_100_returns.csv` - Daily return time series for UK FTSE 100 stocks
- `FTSE_100_metrics.csv` - Rolling correlation (ρ) and topology CV metrics
- `DAX_returns.csv` - Daily return time series for German DAX stocks
- `DAX_metrics.csv` - Rolling metrics for DAX
- `Nikkei_225_returns.csv` - Daily return time series for Japan Nikkei 225 stocks
- `Nikkei_225_metrics.csv` - Rolling metrics for Nikkei
- `Crypto_returns.csv` - Daily return time series for cryptocurrency basket
- `Crypto_metrics.csv` - Rolling metrics for crypto

### Summary File

- `international_market_summary.csv` - Cross-market comparison table with:
  - Market name
  - Mean correlation (ρ)
  - Mean topology CV
  - Number of stocks
  - Number of days
  - Number of rolling windows

## Current Status

⚠️ **Empty**: Scripts could not run in sandbox environment due to network restrictions (HTTP 403 when accessing Yahoo Finance).

✅ **Ready**: Run `validate_international_markets.py` locally to populate this directory with real empirical data.

## Usage

See `../REAL_DATA_ANALYSIS_README.md` for full instructions on running the analysis script and integrating results into the thesis.
