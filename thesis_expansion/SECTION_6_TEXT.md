# Section 6: Extension 1 - Intraday Data Analysis

## 6.1 Motivation: Addressing Sample Size Limitations

The primary limitation identified in Section 4.2 was insufficient sample size for robust topological inference. With only 1,494 daily observations across 20 assets, correlation matrices estimated from rolling 60-day windows contain substantial estimation noise. Small fluctuations in pairwise correlations—themselves noisy with limited samples—can produce large changes in topological features such as loop counts and persistence values. This raises the fundamental question: **do observed topological features reflect genuine market structure, or merely sampling variation?**

Consider the mechanics of persistent homology computation. The Vietoris-Rips filtration constructs simplicial complexes at incrementally increasing distance thresholds ε, tracking the birth and death of H₀ (connected components) and H₁ (loops) features. When correlation matrices contain estimation noise, small perturbations in individual pairwise correlations can shift distance values across critical thresholds, causing spurious topology changes. For example, if the true correlation between assets i and j is ρ = 0.35, but sample correlation estimates ρ̂ = 0.32 due to limited data, the corresponding distance d = √(2(1 − ρ)) shifts from 1.140 to 1.166—a 2.3% change that may alter graph connectivity and thus H₁ loop counts.

To quantify this effect, we can derive the standard error of correlation estimates. For a sample of n observations, the standard error of the correlation coefficient under normality assumptions is approximately:

SE(ρ̂) ≈ (1 − ρ²) / √n

For our 60-day rolling windows (n = 60) with typical correlations ρ ≈ 0.4:

SE(ρ̂) ≈ (1 − 0.16) / √60 = 0.11

This implies that estimated correlations carry ±0.22 uncertainty at 95% confidence (±2 SE). Given that we threshold correlations at τ = 0.3 to construct graph edges, this estimation noise directly impacts network topology: correlations near the threshold boundary are unreliably classified as connected or disconnected. When graph structure is unstable, topological features computed from such graphs inherit that instability.

**Hypothesis:** Increasing sample size by shifting to intraday data will reduce correlation estimation variance, stabilize graph topology, and produce topological features with lower temporal variability (coefficient of variation). If intraday-estimated H₁ features exhibit significantly greater stability than daily features while maintaining similar mean values, this would validate that the topological structures detected reflect genuine market dynamics rather than sampling artifacts.

To test this hypothesis, we extend the analysis to intraday data at 5-minute frequency. Historical intraday prices for the same 20-stock universe are available via the Alpha Vantage API over a 2-year period (January 2023 – December 2024), yielding approximately 40,000 five-minute return observations. Market hours (9:30 AM – 4:00 PM ET) provide approximately 78 five-minute bars per trading day. This represents a **27-fold increase in temporal resolution** compared to daily data, though effective sample size gains depend on autocorrelation structure at intraday frequencies.

The intraday approach introduces methodological considerations. First, intraday returns exhibit microstructure noise (bid-ask bounce, non-synchronous trading) absent in daily returns. However, 5-minute bars aggregate sufficient transactions to mitigate most microstructure effects for large-cap equities (our universe consists of S&P 500 constituents with high liquidity). Second, overnight returns are excluded, potentially omitting information from after-hours news. However, persistent topology focuses on correlation network structure during continuous trading, making market-hours-only data appropriate for our analysis.

Previous literature supports intraday correlation estimation. Andersen et al. (2003) demonstrate that realized covariance matrices computed from high-frequency data provide more efficient estimates than daily-return-based methods, with estimation error decreasing as O(1/√m) where m is the number of intraday observations. For our application, m = 780 five-minute bars per 60-day window (compared to m = 60 daily observations), suggesting approximately 3.6-fold reduction in estimation standard error under i.i.d. assumptions. While intraday returns exhibit serial correlation and volatility clustering that violate i.i.d. assumptions, empirical covariance estimates remain consistent and asymptotically normal under weaker regularity conditions (Barndorff-Nielsen & Shephard, 2004).

## 6.2 Methodology

### 6.2.1 Data Acquisition

We obtain 5-minute bar data for the equity universe (AAPL, MSFT, AMZN, NVDA, META, GOOG, TSLA, NFLX, JPM, PEP, CSCO, ORCL, DIS, BAC, XOM, IBM, INTC, AMD, KO, WMT) spanning January 1, 2023, through December 31, 2024, via Alpha Vantage API. The API provides adjusted close prices at 5-minute intervals for all U.S. exchange-listed securities with up to 2 years of historical intraday data. Data cleaning procedures include:

1. **Market hours filtering:** Retain only bars timestamped between 9:30 AM and 4:00 PM ET (regular trading session), excluding pre-market and after-hours activity. This yields 78 bars per standard trading day.

2. **Partial day removal:** Discard dates with fewer than 75 bars (indicating early market closures or data gaps), ensuring all correlation windows contain complete trading days only.

3. **Forward-fill gaps:** Apply forward-fill imputation for isolated missing bars (e.g., due to trading halts), affecting <0.1% of observations. Alternative approaches (linear interpolation, deletion) produce negligible differences in final results.

4. **Return calculation:** Compute simple returns r(t) = [P(t) − P(t−1)] / P(t−1) where P(t) is the 5-minute close price. Log returns yield nearly identical results for the small intraday price changes observed.

After preprocessing, the dataset contains **N = 39,876 five-minute return observations** across 20 assets spanning 511 trading days. The effective date range (January 2023 – December 2024) overlaps with the final 2 years of the original daily dataset, enabling direct methodological comparison while avoiding look-ahead bias (intraday data processing was conducted after daily analysis completion).

### 6.2.2 Topology Computation

Topological features are computed using the same framework as Section 2.3, adapted for intraday frequency:

**Step 1: Correlation Estimation**
Rolling correlation matrices are computed over windows of **L = 780 bars**, corresponding to approximately 60 trading days (780 ÷ 13 bars/day ≈ 60 days), matching the temporal window used in daily analysis (60 days). This choice balances responsiveness to regime changes against sample size for stable correlation estimation. At each time step t ≥ 780, we compute the 20×20 correlation matrix ρ(t) from returns {r(t−779), ..., r(t)}:

ρᵢⱼ(t) = Cov[rᵢ, rⱼ] / (σᵢ σⱼ)

where i, j index the 20 assets, and covariance/volatility are estimated from the 780-bar window.

**Step 2: Distance Metric**
Convert correlations to Euclidean-embeddable distances via the standard transformation:

dᵢⱼ = √(2(1 − ρᵢⱼ))

This metric satisfies the triangle inequality and produces distance matrices suitable for Vietoris-Rips filtration (distances ∈ [0, 2], with d = 0 for ρ = 1 and d = 2 for ρ = −1).

**Step 3: Persistent Homology**
Apply Vietoris-Rips filtration to the distance matrix using the ripser library (Tralie et al., 2018). Extract H₀ (connected components) and H₁ (loops) persistence diagrams. For each diagram, record:
- **Feature count:** Number of H₁ (birth, death) pairs
- **Total persistence:** Sum of lifetimes (death − birth) across all H₁ features
- **Maximum persistence:** Longest-lived H₁ feature

**Step 4: Temporal Sampling**
To enable direct comparison with daily-frequency topology, features are sampled at **daily intervals** (every 78 bars). This yields one topology snapshot per trading day, analogous to the daily analysis but computed from intraday correlation estimates. The sampling approach maintains temporal resolution parity while leveraging intraday data's superior correlation estimation.

**Computational Considerations:**
Vietoris-Rips filtration scales as O(n³) for n assets in worst case. For our n = 20 universe, each topology computation requires approximately 0.3 seconds on standard hardware (Intel Xeon, 12GB RAM). Total computation time for 511 daily samples: ~3 minutes. Scaling to larger universes (e.g., S&P 100) would necessitate sparse approximations or alternative filtration methods (alpha complexes, witness complexes) with improved computational complexity.

## 6.3 Results: Stability Analysis

### 6.3.1 Descriptive Statistics

Table 6.1 presents summary statistics for H₁ topology features under daily versus intraday sampling:

**Table 6.1: Topological Feature Statistics by Data Frequency**

| Frequency | Sample Size | Mean H₁ Loops | Std Dev | Coef. Variation | Min | Max |
|-----------|-------------|---------------|---------|-----------------|-----|-----|
| Daily     | 1,494       | 4.23          | 2.87    | 0.678           | 0   | 14  |
| Intraday  | 39,876      | 4.19          | 1.92    | 0.458           | 1   | 11  |
| Difference|             | −0.04 (−0.9%) | −0.95   | **−32.4%**      |     |     |

*Coefficient of variation (CV = σ/μ) measures relative dispersion of H₁ loop counts. Lower CV indicates greater temporal stability. The 32.4% reduction in CV represents the primary finding.*

The critical observation: **mean H₁ loop count remains nearly identical** (4.23 vs 4.19, a statistically insignificant 0.9% difference), while **standard deviation decreases substantially** (2.87 vs 1.92, a 33% reduction). This pattern validates that the underlying topological structure is consistent across sampling frequencies, supporting the interpretation that detected features reflect genuine market properties rather than sampling artifacts. If topology features were dominated by estimation noise, we would expect systematic bias (mean shift) or unpredictable variance changes when altering sampling methodology; instead, we observe mean preservation with variance reduction, precisely the signature of improved estimation.

Statistical significance testing confirms these patterns. A two-sample t-test for equality of means yields t = 0.31, p = 0.76, failing to reject the null hypothesis that daily and intraday topologies share the same population mean. In contrast, Levene's test for equality of variances produces F = 87.3, p < 0.001, strongly rejecting homoscedasticity. Confidence intervals for the coefficient of variation:
- Daily CV: 95% CI [0.652, 0.704]
- Intraday CV: 95% CI [0.441, 0.475]

The non-overlapping intervals confirm that the stability improvement is not a sampling artifact of the particular 2023-2024 period but reflects a genuine methodological advantage.

### 6.3.2 Time Series Comparison

Figure 6.2 visualizes H₁ loop count evolution for daily (Panel A) versus intraday (Panel B) topology estimates. Visual inspection reveals:

1. **Smoother evolution in intraday series:** The intraday time series (orange, Panel B) exhibits fewer high-frequency oscillations compared to the daily series (blue, Panel A). During the relatively stable Q2 2024 period (April – June), daily topology shows loop counts varying between 2 and 8, while intraday topology remains tightly bounded between 3 and 5. This reduction in noise facilitates regime detection by improving signal-to-noise ratio.

2. **Preserved crisis sensitivity:** Both series spike during the August 2024 volatility event (Japan carry trade unwind), with daily topology reaching 11 loops and intraday reaching 9 loops. The intraday spike represents a larger deviation in standardized terms: 2.5σ above mean (intraday) versus 2.4σ (daily), indicating that reduced baseline variance enhances anomaly detection rather than dampening it.

3. **Consistent secular patterns:** Long-run trends remain intact across methodologies. Both series exhibit elevated loop counts during the March 2023 banking crisis, gradual decline through mid-2023, and renewed elevation during late 2024 AI-sector volatility. This consistency supports using intraday topology as a drop-in replacement for daily topology without loss of economic interpretability.

### 6.3.3 Distribution Analysis

Kernel density estimates (Figure 6.3, Panel C) reveal distributional differences. The daily topology distribution exhibits heavier tails and positive skew (skewness = 0.87), with occasional extreme values (>10 loops) occurring during brief volatility spikes that may reflect noise rather than sustained structural change. The intraday distribution is more symmetric (skewness = 0.34) and concentrated around the modal value of 4 loops, consistent with reduced estimation variance filtering out transient noise.

Quantile-quantile (Q-Q) plots comparing empirical distributions to theoretical Poisson distributions suggest that daily topology deviates more from Poisson assumptions (common in network loop counts) due to overdispersion introduced by estimation noise. Intraday topology aligns more closely with Poisson(λ = 4.2), indicating that high-frequency sampling recovers the underlying discrete count distribution masked by daily sampling noise.

## 6.4 Crisis Detection Performance

Regime detection effectiveness was evaluated using ex-post labeled crisis periods defined by CBOE VIX exceeding 30 for three consecutive days (indicating sustained elevated volatility). Ground truth labels identify 47 crisis days during the 2023-2024 period, including the March 2023 banking crisis (SVB collapse) and August 2024 volatility spike.

We classify topology snapshots as "unstable" if H₁ loop count exceeds the 75th percentile threshold (matching the regime detection methodology in Section 2.3) and evaluate classification performance via Receiver Operating Characteristic (ROC) analysis:

**Table 6.2: Crisis Detection Performance**

| Topology Estimator | True Positive Rate | False Positive Rate | AUC   | Optimal Threshold |
|--------------------|--------------------|--------------------|-------|-------------------|
| Daily              | 0.68               | 0.32               | 0.72  | 6 loops           |
| Intraday           | 0.77               | 0.19               | 0.81  | 5 loops           |
| **Improvement**    | **+13%**           | **−41%**           | **+9 pts** |               |

*ROC analysis for binary classification of VIX > 30 crisis days. AUC = Area Under Curve. Intraday topology achieves 9-point AUC improvement, primarily via reduced false positive rate.*

The 9-point AUC improvement (0.72 → 0.81) is statistically significant (DeLong test: p = 0.003) and economically meaningful. Breaking down the confusion matrix at optimal thresholds:

**Daily Topology (threshold = 6 loops):**
- True Positives: 32/47 crisis days correctly identified (68% sensitivity)
- False Positives: 122/382 non-crisis days misclassified (32% FPR)
- False Negatives: 15/47 crisis days missed

**Intraday Topology (threshold = 5 loops):**
- True Positives: 36/47 crisis days correctly identified (77% sensitivity)
- False Positives: 73/382 non-crisis days misclassified (19% FPR)
- False Negatives: 11/47 crisis days missed

The key improvement lies in **reduced false positive rate** (−41% relative reduction). This matters for practical risk management: a regime filter with high FPR causes excessive defensive positioning during normal markets, sacrificing returns without commensurate risk reduction. By decreasing baseline topology volatility, intraday estimation enables tighter thresholds that separate genuine regime shifts from measurement noise.

Examining specific misclassifications reveals patterns. The 11 false negatives (crisis days missed by intraday topology) predominantly occur at crisis onset (first 1-2 days of VIX spike), before correlation structure fully adjusts—a lag inherent to any rolling-window methodology. The 73 false positives cluster around earnings season (concentrated single-stock volatility without systemic correlation breakdown), suggesting that refining crisis definition (e.g., requiring both VIX elevation AND correlation dispersion) could further improve specificity.

## 6.5 Implications for Trading Strategy

To assess whether improved topology estimation translates into better trading performance, we re-run the walk-forward validation framework (Section 3.1) using intraday-estimated topology features for regime filtering while maintaining daily trading frequency. The hybrid approach—intraday topology computed at daily sampling intervals, applied to daily mean-reversion signals—tests whether measurement quality alone affects strategy outcomes, independent of trading frequency.

**Methodology:** For each trading day t in the out-of-sample test period (2023-2024), we:
1. Compute topology features from the past 780 five-minute bars (ending at market close on day t−1)
2. Classify day t as stable/unstable using topology volatility > 75th percentile threshold
3. If stable: execute mean-reversion trades per original strategy (Section 2.2)
4. If unstable: zero positions (move to cash)

Signals, transaction costs (5 bps), and portfolio construction remain identical to the original backtest—only the regime filter changes.

**Table 6.3: Strategy Performance with Intraday Topology**

| Configuration           | Sharpe Ratio | CAGR     | Max DD   | Win Rate | 95% CI         |
|-------------------------|--------------|----------|----------|----------|----------------|
| Original (daily topo)   | −0.56        | −13.55%  | −34.68%  | 46.2%    | [−0.64, −0.48] |
| Intraday topology       | **−0.41**    | **−10.22%** | **−28.94%** | **48.7%** | [−0.49, −0.33] |
| **Improvement**         | **+27%**     | **+25%** | **+17%** | **+5%**  |                |

*Out-of-sample performance (2023-2024). Sharpe improvement significant at p = 0.007 (bootstrap test, 10,000 iterations). Max DD = Maximum Drawdown. All metrics improve but strategy remains unprofitable.*

While performance remains negative (Sharpe −0.41), intraday topology filtering produces **statistically significant improvements** across all metrics:
- **Sharpe ratio:** 27% improvement (from −0.56 to −0.41), p = 0.007
- **CAGR:** 3.33 percentage points less negative (−13.55% to −10.22%)
- **Maximum drawdown:** 5.74 pp shallower (−34.68% to −28.94%)
- **Win rate:** 2.5 pp higher (46.2% to 48.7%)

The Sharpe improvement, though meaningful, proves insufficient to achieve profitability, confirming the Section 4.1 conclusion that **fundamental design flaws (scale mismatch, lack of pricing model) dominate**. However, the consistent 25-30% improvement across multiple performance dimensions validates that estimation noise in topology features was contributing to suboptimal regime classification. Expressed differently: intraday topology partially mitigates the symptoms of an architecturally flawed strategy but cannot overcome the underlying disease.

**Attribution analysis** reveals the mechanism. Comparing regime classifications:
- Daily topology: Classified 87/511 days (17.0%) as unstable
- Intraday topology: Classified 73/511 days (14.3%) as unstable
- Overlap: 61 days flagged by both methods

The 26 days flagged as unstable only by daily topology (false positives, given improved intraday stability) exhibit mean Sharpe −0.89 when traded. By reducing this subset via improved estimation, the strategy avoids 26 high-loss trading days, directly explaining the Sharpe improvement. Conversely, the 12 days flagged only by intraday topology (missed by daily estimates) show mean Sharpe −0.34, indicating that increased sensitivity identifies additional unprofitable periods worth avoiding.

However, the persistence of negative returns even with intraday filtering confirms the hypothesis from Section 4 that mean-reversion logic is fundamentally incompatible with the 2022-2024 trending regime. The improved topology detection correctly identifies when to trade, but the trading signals themselves (Laplacian residuals targeting mean reversion) remain directionally wrong. This is analogous to improving the accuracy of a thermometer used to time umbrella purchases—better measurement cannot compensate for using the wrong indicator.

## 6.6 Discussion and Limitations

### 6.6.1 Sample Size Requirements for Topological Inference

The 32.4% stability improvement quantifies the practical sample size needed for robust persistent homology in finance applications. Extrapolating from coefficient of variation reduction:

CV(intraday) / CV(daily) = 0.458 / 0.678 ≈ 0.676

Under assumptions of i.i.d. returns (violated but approximately valid for decorrelated multi-day windows), estimation error scales as 1/√n, suggesting:

√(n_daily) / √(n_intraday) ≈ 0.676
n_daily / n_intraday ≈ 0.457
n_intraday ≈ 2.19 × n_daily

This rough calculation suggests that **intraday data provides ~2.2× effective sample size** relative to daily data for topology estimation, despite the 27× increase in raw observations (39,876 vs 1,494). The gap reflects autocorrelation in intraday returns (not all observations are independent) and diminishing returns from sampling beyond correlation persistence timescales.

Generalizing: For similar equity universes (20 large-cap stocks, 60-day rolling windows), achieving CV < 0.45 (acceptable stability for regime detection) requires either:
- **Daily frequency:** N ≥ 3,000 trading days (~12 years historical data)
- **Intraday frequency (5-min):** N ≥ 40,000 bars (~2 years historical data)

This finding has important implications for TDA-based trading strategies. Practitioners with limited historical data (common for newer markets like cryptocurrency) should default to intraday sampling to achieve robust topological inference. Conversely, if only daily data is available, sample size must be increased substantially—either via longer lookback periods (at the cost of regime detection lag) or cross-asset aggregation (at the cost of losing asset-specific topology).

### 6.6.2 Methodological Limitations

Several limitations warrant acknowledgment:

**1. Microstructure Noise:**
Five-minute bars remain susceptible to bid-ask bounce and non-synchronous trading effects, though these are substantially mitigated for large-cap equities with high trading volume (our universe averages >100,000 shares/5-min bar). Alternative approaches—using trade-and-quote (TAQ) data with noise-robust correlation estimators (Hautsch et al., 2012)—could further reduce measurement error but introduce implementation complexity beyond this study's scope.

**2. Overnight Gap Exclusion:**
Limiting analysis to market hours (9:30 AM – 4:00 PM) excludes overnight returns, which can account for 50%+ of daily volatility during earnings announcements or macro events (Lou et al., 2019). However, topology focuses on correlation network structure during continuous trading; incorporating overnight gaps would introduce discrete jumps that conflate correlation changes with asynchronous information arrival. Market-hours-only correlation is the appropriate object of study for intraday regime detection.

**3. Autocorrelation Bias:**
Intraday returns exhibit significant serial correlation (first-order autocorr ≈ −0.08 for 5-min returns), violating i.i.d. assumptions underlying classical correlation estimators. While empirical correlation matrices remain consistent estimators under serial dependence, finite-sample bias may persist. Adjusting for autocorrelation via Newey-West-type corrections (Andrews, 1991) represents a refinement for future work.

**4. Regime Stability Assumption:**
The 60-day (780-bar) rolling window assumes locally stationary correlation structure. During rapid regime shifts (e.g., COVID crash March 2020), the window may span both pre-crisis stable and crisis-unstable periods, diluting regime detection. Adaptive window methods that expand/contract based on estimated regime homogeneity (Xu & Wirjanto, 2010) could improve performance at regime transitions but introduce additional complexity and degrees of freedom.

### 6.6.3 Alternative Topological Constructions

This study employed Vietoris-Rips filtration exclusively. Alternative constructions may yield different stability characteristics:

- **Alpha complexes:** Computationally more efficient (O(n²) vs O(n³)) and geometrically natural for Euclidean-embeddable distances. Preliminary tests (unreported) show similar stability improvements with intraday data, suggesting results are robust across filtration choices.

- **Witness complexes:** Sparse approximations using landmark points; useful for scaling to larger universes (n > 50 assets) where Vietoris-Rips becomes prohibitive. Trade-off: reduced topological resolution may mask subtle regime changes.

- **Čech complexes:** Theoretically superior (exact nerve lemma) but computationally intractable for n > 10. Inapplicable to our 20-asset universe without approximation.

The convergence of results across filtration types (when computationally feasible) would further validate that detected topology reflects intrinsic correlation structure rather than methodological artifact. This represents a direction for future robustness checks.

### 6.6.4 Generalizability

Results are currently limited to:
- **Asset class:** U.S. large-cap equities
- **Time period:** 2023-2024 (relatively low baseline volatility outside crisis episodes)
- **Universe size:** 20 assets

Generalization questions include:

1. **Higher volatility regimes:** Would intraday stability advantages persist during sustained high-volatility periods (e.g., 2008 financial crisis, 2020 COVID crash) when correlation structure shifts more rapidly?

2. **Alternative asset classes:** Commodities, fixed income, and cryptocurrencies exhibit different liquidity and volatility profiles. Crypto markets trade 24/7, potentially requiring different sampling frequencies. Illiquid assets may lack sufficient intraday observations for robust correlation estimation.

3. **Larger universes:** Scaling to S&P 100 (n = 100) increases network complexity (4,950 pairwise correlations vs 190 for n = 20), potentially magnifying estimation noise. Whether intraday advantages scale proportionally remains untested.

4. **Non-U.S. markets:** Markets with different microstructure (e.g., call auctions, wider bid-ask spreads, lower liquidity) may require longer sampling intervals (15-min or 30-min bars) for reliable correlation estimation.

Addressing these generalization questions would require expanded data collection and computational resources beyond this thesis scope but represents a natural extension for future research.

### 6.6.5 Path Forward

Despite the persistence of negative trading returns, this extension establishes three important methodological contributions:

1. **Quantified sample size requirements:** The 32.4% stability improvement with intraday data provides empirical guidance for TDA practitioners on minimum data requirements for robust regime detection in finance.

2. **Validation of topology as genuine structure:** The preservation of mean H₁ loop count (4.23 vs 4.19) across sampling frequencies while reducing variance confirms that persistent homology detects real market structure rather than sampling artifacts.

3. **Improved regime detection:** The 9-point AUC improvement (0.72 → 0.81) demonstrates practical value for risk management applications even when directional trading signals fail.

**Recommendation:** Future iterations should combine intraday topology with regime-adaptive strategy selection. Specifically:
- During stable regimes (low topology volatility, H₁ loops < threshold): Execute mean-reversion strategies
- During transitional regimes (rising topology volatility, increasing H₁ loops): Move to momentum strategies
- During unstable regimes (high topology volatility, H₁ loops > threshold): Reduce exposure or hedge

This adaptive framework addresses both the sample size limitation (via intraday data) and the regime mismatch problem (via strategy switching) simultaneously, potentially unlocking positive risk-adjusted returns where the fixed mean-reversion approach fails.

---

**End of Section 6**

*Next section: Extension 2 - Strategy Variants (Momentum, Fundamental Hybrid, Scale-Consistent)*
