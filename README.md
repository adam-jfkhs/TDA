# TDA Trading Strategy — Adam Levine

Quantitative research into topological data analysis (TDA) applied to equity market regime detection.

## Finding

No strategy tested produced positive risk-adjusted returns in a look-ahead-free, factor-regression-verified backtest. The best approach (beta-spread + Fiedler regime filter) achieves Sharpe −0.77 over 2022–2024. Topology provides measurable risk-reduction benefit over no filter, but is outperformed by a simpler realized-volatility filter.

The mathematical framework (correlation-CV bound, spectral gap analysis, ghost loop detection) is a genuine contribution independent of strategy performance.

## Repository Structure

```
thesis_latex/                     LaTeX source for the full thesis
  sections/                       Individual section files
  tables/                         Authoritative result tables
  figures/                        Figure manifest and generated figures
  references.bib                  Complete bibliography (24 entries)

risk_report.py                    Section 12 verified backtest (Sharpe −0.77)
ml_integration_real.py            Section 10 ML benchmark on REAL data, 1-day lag
strategy_variants_real.py         Section 8 variants on REAL data, 1-day lag
run_all_real.sh                   One-shot Colab runner for all three pipelines

null_model_test.py                Null-model TDA significance test
validation_tests.py               Cross-market validation on real FTSE/DAX/Nikkei
visualizations_3d.py              Interactive 3D topology visualisations

thesis_expansion/                 Phase-by-phase analysis code (legacy)
  phase1_intraday/                Intraday topology
  phase2_sector_topology/         Section 7 source (look-ahead biased; fixed in
                                  risk_report.py Strategy D)
  phase3_strategy_variants/       Look-ahead biased; replaced by
                                  strategy_variants_real.py
  phase4_cross_market/            Calibrated simulation; replaced by
                                  validation_tests.py for real cross-market
  phase5_ml_integration/          Synthetic-data ML; replaced by
                                  ml_integration_real.py
  phase6_theory/                  Theoretical bound on simulated correlation
                                  matrices (standard for theoretical validation)
```

## Reproducing All Numbers (Google Colab)

```bash
!bash run_all_real.sh
```

This produces:
- `risk_report.pdf` — 6-panel risk report, Strategy A/B/C/D vs SPY (2022–2024)
- `ml_integration_real.{csv,txt}` — F1, AUC, precision, recall, Sharpe per ML model
- `strategy_variants_real.{csv,txt}` — V1/V2/V3/V4 metrics

After running, paste the contents of the `*.txt` files into Sections 8 and 10
of the thesis, then remove the `Simulation Disclosure` and `Projection Disclosure`
banners at the top of `sec08_variants.tex` and `sec10_ml.tex` — the disclosures
are only needed while those sections still reference simulated/projected numbers.

## Compiling the Thesis

```bash
cd thesis_latex
pdflatex thesis_main.tex
bibtex thesis_main
pdflatex thesis_main.tex
pdflatex thesis_main.tex
```

