#!/bin/bash
# Run all real-data pipelines end-to-end. Produces:
#   risk_report.pdf                     (Section 12 verified backtest, Sharpe -0.77)
#   ml_integration_real.{csv,txt}       (Section 10 ML — replaces simulated numbers)
#   strategy_variants_real.{csv,txt}    (Section 8 variants — replaces look-ahead numbers)
#
# Requires internet access (Yahoo Finance + Ken French library).
# Designed for Google Colab. Run with:
#     !bash run_all_real.sh
#
# Once these complete, you can remove the Simulation / Projection Disclosure
# banners from sections 8 and 10 and replace the placeholder Sharpe numbers
# with the real values printed in *.txt.

set -e

echo "=== Installing dependencies ==="
pip install --quiet yfinance pandas-datareader matplotlib scipy statsmodels scikit-learn

echo
echo "=== [1/3] Section 12: Verified backtest (risk_report.py) ==="
python3 risk_report.py

echo
echo "=== [2/3] Section 10: ML on real data (ml_integration_real.py) ==="
python3 ml_integration_real.py

echo
echo "=== [3/3] Section 8: Strategy variants on real data (strategy_variants_real.py) ==="
python3 strategy_variants_real.py

echo
echo "=== Output files ==="
ls -la risk_report.pdf ml_integration_real.* strategy_variants_real.*

echo
echo "Paste the contents of *.txt into the corresponding thesis sections,"
echo "then delete the Simulation/Projection Disclosure banners at the top of"
echo "thesis_latex/sections/sec08_variants.tex and sec10_ml.tex."
