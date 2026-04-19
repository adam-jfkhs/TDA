# TDA Trading Strategy — Adam Levine

Quantitative research into topological data analysis (TDA) applied to equity market regime detection.

## Finding

No strategy tested produced positive risk-adjusted returns in a look-ahead-free, factor-regression-verified backtest. The best approach (beta-spread + Fiedler regime filter) achieves Sharpe −0.77 over 2022–2024. Topology provides measurable risk-reduction benefit over no filter, but is outperformed by a simpler realized-volatility filter.

The mathematical framework (correlation-CV bound, spectral gap analysis, ghost loop detection) is a genuine contribution independent of strategy performance.

## Repository Structure

```
thesis_latex/               LaTeX source for the full thesis
  sections/                 Individual section files
  tables/                   Authoritative result tables
  figures/                  Figure manifest and generated figures
  references.bib            Complete bibliography (24 entries)

thesis_expansion/           Phase-by-phase Python analysis code
  phase1_intraday/          Intraday topology analysis
  phase2_sector_topology/   Sector-specific backtest code
  phase3_strategy_variants/
  phase4_cross_market/      Calibrated simulation (not live data)
  phase5_ml_integration/
  phase6_theory/

risk_report.py              Verified backtest — run this on Colab
generate_all_thesis_figures.py  Reproduces all thesis figures
```

## Running the Verified Backtest

```bash
# On Google Colab:
pip install yfinance pandas-datareader matplotlib scipy statsmodels
python risk_report.py
# Output: risk_report.pdf (6-panel risk report) + console summary table
```

Produces equity curves, rolling Sharpe, drawdown, monthly heatmap, Fama-French factor regression, and return distribution for four strategies vs SPY.

## Compiling the Thesis

```bash
cd thesis_latex
pdflatex thesis_main.tex
bibtex thesis_main
pdflatex thesis_main.tex
pdflatex thesis_main.tex
```
