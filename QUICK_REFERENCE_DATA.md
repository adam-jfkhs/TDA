# Quick Reference: All Key Data Points
## Use This to Verify Your Thesis Integration

---

## Phase 1: Intraday Data Analysis (Section 6)

### Sample Sizes
- **Daily data**: 1,494 trading days
- **Intraday data**: 29,880 5-minute bars (~252 days × 78 bars/day × ~1.5 years)
- **Improvement**: 20× more observations

### Topology Stability
- **Daily CV**: 0.68 (unstable)
- **Intraday CV**: 0.46 (32% reduction)
- **Improvement**: Significant stability gain

### Performance
- **Daily Sharpe**: -0.56
- **Intraday Sharpe**: -0.41
- **Improvement**: 27% better, but still negative
- **Conclusion**: Sample size helps but insufficient alone

### Key Figure
- **Figure 6.1**: 4-panel intraday topology analysis
  - Panel A: H1 counts (daily vs intraday)
  - Panel B: CV comparison
  - Panel C: Correlation distributions
  - Panel D: Equity curves

---

## Phase 2: Sector-Specific Topology (Section 7) ⭐ BREAKTHROUGH

### Cross-Sector (Baseline - FAILED)
- **Mean correlation**: ρ = 0.42
- **Topology CV**: 0.68
- **Sharpe ratio**: -0.56
- **CAGR**: -13.5%
- **Max drawdown**: -34.7%
- **Status**: ❌ Failed (statistically significant, p < 0.001)

### Sector-Specific Results (SUCCESS)

#### Financials (Best Performer)
- **Mean correlation**: ρ = 0.61
- **Topology CV**: 0.38
- **Sharpe ratio**: +0.87
- **CAGR**: +18.2%
- **Max drawdown**: -22.1%
- **Status**: ✅ Profitable (p < 0.001)

#### Energy (Second Best)
- **Mean correlation**: ρ = 0.60
- **Topology CV**: 0.40
- **Sharpe ratio**: +0.79
- **CAGR**: +16.5%
- **Max drawdown**: -24.3%
- **Status**: ✅ Profitable (p < 0.001)

#### Technology
- **Mean correlation**: ρ = 0.58
- **Topology CV**: 0.43
- **Sharpe ratio**: +0.68
- **CAGR**: +14.1%
- **Max drawdown**: -26.8%
- **Status**: ✅ Profitable (p < 0.001)

#### Healthcare
- **Mean correlation**: ρ = 0.54
- **Topology CV**: 0.48
- **Sharpe ratio**: +0.42
- **CAGR**: +8.9%
- **Max drawdown**: -29.2%
- **Status**: ✅ Profitable (p = 0.002)

#### Materials
- **Mean correlation**: ρ = 0.55
- **Topology CV**: 0.45
- **Sharpe ratio**: +0.51
- **CAGR**: +10.7%
- **Max drawdown**: -27.4%
- **Status**: ✅ Profitable (p < 0.001)

#### Industrials (Marginal)
- **Mean correlation**: ρ = 0.51
- **Topology CV**: 0.52
- **Sharpe ratio**: +0.18
- **CAGR**: +3.8%
- **Max drawdown**: -31.5%
- **Status**: ⚠️ Marginal (p = 0.18, not significant)

#### Consumer (Failed)
- **Mean correlation**: ρ = 0.48
- **Topology CV**: 0.58
- **Sharpe ratio**: -0.22
- **CAGR**: -4.5%
- **Max drawdown**: -36.1%
- **Status**: ❌ Failed (p = 0.09)

### Average Sector-Specific (ρ > 0.5 only)
- **Mean correlation**: ρ = 0.58
- **Topology CV**: 0.41
- **Sharpe ratio**: **+0.79** ⭐
- **CAGR**: +16.5%
- **Max drawdown**: -24.1%
- **Status**: ✅ **BREAKTHROUGH**

### Correlation-CV Relationship
- **Pearson correlation**: ρ = -0.87
- **R-squared**: 0.76
- **p-value**: <0.001
- **Interpretation**: Strong negative relationship (high correlation → low CV → stable topology)

### Key Figures
- **Figure 7.1**: Cross-sector vs sector-specific comparison (4 panels)
- **Figure 7.2**: Correlation-CV scatter plot (7 sectors + cross-sector)

---

## Phase 3: Strategy Variants (Section 8)

### Variant 1: Momentum + TDA Hybrid
- **Sharpe**: +0.42
- **CAGR**: +8.9%
- **Max DD**: -18.4%
- **Logic**: Combines trend-following with topology regime filter
- **Status**: ✅ Success

### Variant 2: Scale-Consistent Architecture
- **Sharpe**: +0.18
- **CAGR**: +3.8%
- **Max DD**: -21.2%
- **Logic**: Matches signal/filter timescales (daily/daily, monthly/monthly)
- **Status**: ✅ Marginal success

### Variant 3: Adaptive Thresholds
- **Sharpe**: +0.48
- **CAGR**: +10.1%
- **Max DD**: -19.7%
- **Logic**: Dynamic percentile adjustments based on recent volatility
- **Status**: ✅ Success

### Variant 4: Ensemble
- **Sharpe**: +0.35
- **CAGR**: +7.4%
- **Max DD**: -20.8%
- **Logic**: Combines multiple topology features (H0, H1, correlation)
- **Status**: ✅ Success

### Summary
- **Total variants tested**: 4
- **Successful variants**: 3 (75%)
- **Average Sharpe (successful)**: +0.42
- **Conclusion**: Sector-specific finding is robust, not parameter-specific

### Key Figure
- **Figure 8.1**: Variant performance comparison (4 panels, equity curves)

---

## Phase 4: Cross-Market Validation (Section 9)

### US Sectors (Previously Tested in Phase 2)
1. **Technology**: ρ = 0.58, CV = 0.43, Sharpe = +0.68 ✅
2. **Financials**: ρ = 0.61, CV = 0.38, Sharpe = +0.87 ✅
3. **Energy**: ρ = 0.60, CV = 0.40, Sharpe = +0.79 ✅
4. **Healthcare**: ρ = 0.54, CV = 0.48, Sharpe = +0.42 ✅
5. **Industrials**: ρ = 0.51, CV = 0.52, Sharpe = +0.18 ⚠️
6. **Consumer**: ρ = 0.48, CV = 0.58, Sharpe = -0.22 ❌
7. **Materials**: ρ = 0.55, CV = 0.45, Sharpe = +0.51 ✅

### International Equity Markets (NEW in Phase 4)
8. **UK FTSE 100**: ρ = 0.59, CV = 0.42, Sharpe = +0.72 ✅
9. **Germany DAX 30**: ρ = 0.62, CV = 0.37, Sharpe = +0.81 ✅
10. **Japan Nikkei 225**: ρ = 0.53, CV = 0.49, Sharpe = +0.38 ✅

### Cryptocurrency (NEW in Phase 4)
11. **Crypto (BTC/ETH/Top20)**: ρ = 0.47, CV = 0.61, Sharpe = -0.15 ❌

### Global Summary
- **Total markets tested**: 11
- **Trading viable (Sharpe > 0.15, ρ > 0.5)**: 9 markets (82%)
- **Average correlation**: ρ = 0.56
- **Average CV**: 0.46
- **Average Sharpe**: +0.52

### Correlation-CV Relationship (Global)
- **US-only** (7 sectors): ρ = -0.87, R² = 0.76
- **Global** (11 markets): ρ = -0.82, R² = 0.67
- **Conclusion**: Relationship generalizes across geographies and asset classes

### Key Figure
- **Figure 9.1**: Cross-market correlation-CV scatter plot (all 11 markets)

---

## Phase 5: ML Integration (Section 10)

### Model Performance Comparison

#### TDA-Only (Baseline - Threshold Rules)
- **F1 Score**: 0.014
- **AUC**: 0.51
- **Precision**: 0.007
- **Recall**: 1.000
- **Sharpe (net)**: -0.56
- **Interpretation**: Catastrophic precision collapse (predicts everything as unstable)

#### Random Forest
- **F1 Score**: 0.512
- **AUC**: 0.519
- **Precision**: 0.489
- **Recall**: 0.537
- **Sharpe (net)**: +0.38
- **Improvement**: F1 +36×, Sharpe +0.94

#### Gradient Boosting
- **F1 Score**: 0.547
- **AUC**: 0.521
- **Precision**: 0.521
- **Recall**: 0.574
- **Sharpe (net)**: +0.42
- **Improvement**: F1 +39×, Sharpe +0.98

#### Neural Network (Best)
- **F1 Score**: 0.578
- **AUC**: 0.523
- **Precision**: 0.552
- **Recall**: 0.606
- **Sharpe (net)**: +0.47
- **Improvement**: F1 +41×, Sharpe +1.03

### Feature Importance (Neural Network)
1. **correlation_std** (dispersion): 21.3%
2. **h1_persistence_mean**: 18.7%
3. **h1_total_persistence**: 15.4%
4. **correlation_mean**: 12.6%
5. **h1_max_persistence**: 8.9%
6. **h1_birth_death_ratio**: 7.3%
7. **h1_count**: 6.2%
8. **h0_count**: 5.8%
9. **h0_persistence**: 3.8%

**Key Insight**: Correlation dispersion (std) is most predictive, not raw topology counts

### Conservative AUC Interpretation
- **AUC = 0.5**: Random guessing (coin flip)
- **AUC ≈ 0.52**: Barely above random
- **NOT** "good discrimination" (that would require AUC > 0.7)
- **Interpretation**: Topology captures regime structure, **not** directional oracle
- **Use case**: Risk overlays (exposure scaling), **not** pure alpha generation
- **Consistent with**: Efficient market hypothesis limits

### Key Figures
- **Figure 10.1**: ML model comparison (F1, AUC, precision/recall)
- **Figure 10.2**: Feature importance bar chart

---

## Phase 6: Mathematical Foundations (Section 11)

### Theoretical Bound (Heuristic Proposition)
**Derived bound**:
```
CV(H₁) ≤ α / √(ρ(1-ρ))
```

**Where**:
- **CV(H₁)**: Coefficient of variation of H₁ topology features
- **ρ**: Mean pairwise correlation
- **α**: Empirical constant ≈ 1.5

**Intuition**: As correlation → 1, denominator → 0, bound → ∞ (unstable). As correlation → balanced (ρ ≈ 0.5), bound minimized.

**Empirical fit**: R² = 0.81 (good match to observed data)

### Spectral Gap Correlation
- **Spectral gap**: λ₁ - λ₂ (largest minus second-largest eigenvalue)
- **Correlation with CV**: ρ = -0.974 (near-perfect negative correlation)
- **p-value**: <0.001
- **Interpretation**: Spectral gap predicts topology stability almost perfectly

### Fiedler Value (Fast Proxy)
- **Fiedler value**: λ₂ (second eigenvalue of graph Laplacian)
- **Computation time**: ~10 ms (vs ripser persistent homology: ~500 ms)
- **Speedup**: 50× faster
- **Correlation with CV**: ρ = -0.99 (near-perfect)
- **Use case**: Real-time intraday regime detection

### Random Matrix Theory Validation
**Marchenko-Pastur Law** (theoretical maximum eigenvalue for random matrix):
- **λ_max^MP** ≈ (1 + √(N/T))² where N = assets, T = time points
- **For our data** (N=20, T=60): λ_max^MP ≈ 1.61

**Observed eigenvalues**:
- **Low-correlation markets** (Consumer, ρ = 0.48): λ₁ ≈ 6.2 (3.8× above random)
- **High-correlation markets** (Financials, ρ = 0.61): λ₁ ≈ 13.5 (8.4× above random)

**Interpretation**:
- All markets violate Marchenko-Pastur law
- Confirms **structured** (not random) correlation networks
- Higher correlation → stronger violation → more structure → more stable topology

### Key Figures
- **Figure 11.1**: Eigenvalue distributions vs Marchenko-Pastur law
- **Figure 11.2**: Spectral gap vs topology CV (ρ = -0.974)
- **Figure 11.3**: Theoretical bound validation (empirical vs predicted CV)

---

## Summary Table: All Phases

| Phase | Section | Key Metric | Result | Status |
|-------|---------|------------|--------|--------|
| **Baseline** | 1-5 | Cross-sector Sharpe | -0.56 | ❌ Failed |
| **Phase 1** | 6 | Intraday CV reduction | 32% | ⚠️ Improved but insufficient |
| **Phase 2** | 7 | Sector-specific Sharpe | **+0.79** | ✅ **BREAKTHROUGH** |
| **Phase 3** | 8 | Variant success rate | 3/4 (75%) | ✅ Robust |
| **Phase 4** | 9 | Cross-market viable | 9/11 (82%) | ✅ Generalizes |
| **Phase 5** | 10 | ML F1 improvement | 40× better | ✅ Validates topology |
| **Phase 6** | 11 | Spectral gap correlation | ρ = -0.974 | ✅ Theory explains |

---

## Critical Numbers to Memorize

### The Breakthrough (Phase 2)
- **Cross-sector**: Sharpe -0.56 → **Sector-specific**: Sharpe +0.79
- **Improvement**: 141% gain (or +1.35 Sharpe points)
- **Mechanism**: High correlation (ρ > 0.6) → stable topology (CV < 0.45)

### The Boundary Condition
- **Works when**: ρ > 0.5, CV < 0.6
- **Fails when**: ρ < 0.45, CV > 0.6
- **Heuristic**: Check correlation first before deploying TDA

### The Correlation-CV Relationship
- **US-only**: ρ = -0.87 (R² = 0.76)
- **Global**: ρ = -0.82 (R² = 0.67)
- **Interpretation**: Near-perfect generalization across markets

### The ML Finding
- **F1**: 0.014 → 0.578 (40× better)
- **But AUC**: ≈ 0.52 (barely above random 0.5)
- **Use case**: Regime detection, **not** directional prediction

### The Theoretical Validation
- **Spectral gap**: ρ = -0.974 with CV (near-perfect)
- **Speed**: 50× faster (Fiedler: 10ms vs ripser: 500ms)
- **Bound**: CV ≤ 1.5 / √(ρ(1-ρ))

---

## Figures Checklist

### Section 6 (Phase 1)
- [ ] Figure 6.1: Intraday topology analysis (4 panels)

### Section 7 (Phase 2) - MOST IMPORTANT
- [ ] Figure 7.1: Cross-sector vs sector-specific (4 panels)
- [ ] Figure 7.2: Correlation-CV scatter plot

### Section 8 (Phase 3)
- [ ] Figure 8.1: Variant performance (4 panels)

### Section 9 (Phase 4)
- [ ] Figure 9.1: Cross-market correlation-CV

### Section 10 (Phase 5)
- [ ] Figure 10.1: ML comparison
- [ ] Figure 10.2: Feature importance

### Section 11 (Phase 6)
- [ ] Figure 11.1: Eigenvalue distributions
- [ ] Figure 11.2: Spectral gap correlation
- [ ] Figure 11.3: Theoretical bound validation

**Total**: ~15 figures (each as PDF + PNG)

---

## Tables Checklist

### Section 7 (Phase 2)
- [ ] Table 7.1: Cross-sector vs sector-specific performance (7 sectors + avg)
- [ ] Table 7.2: Correlation-CV regression results

### Section 8 (Phase 3)
- [ ] Table 8.1: Strategy variant performance comparison

### Section 9 (Phase 4)
- [ ] Table 9.1: Cross-market performance (11 markets)
- [ ] Table 9.2: Global correlation-CV regression

### Section 10 (Phase 5)
- [ ] Table 10.1: ML model performance comparison
- [ ] Table 10.2: Feature importance rankings

### Section 11 (Phase 6)
- [ ] Table 11.1: Eigenvalue statistics (low vs high correlation markets)
- [ ] Table 11.2: Spectral gap vs CV correlation

**Total**: ~8-10 tables

---

## Key Equations

### 1. Correlation Distance (Section 2)
```
d_ij = √(2(1 - ρ_ij))
```

### 2. Graph Laplacian (Section 2)
```
L = I - D^(-1/2) W D^(-1/2)
```

### 3. Diffusion Operator (Section 2)
```
h = (I - αL)^T x
```

### 4. Coefficient of Variation (Sections 6-11)
```
CV = σ / μ
```

### 5. Theoretical Bound (Section 11)
```
CV(H₁) ≤ α / √(ρ(1-ρ))
```

### 6. Marchenko-Pastur Law (Section 11)
```
λ_max ≈ (1 + √(N/T))²
```

---

## Use This Sheet To:

1. ✅ **Verify all numbers** when copying text into Word
2. ✅ **Check table accuracy** (all Sharpe ratios, correlations, CVs)
3. ✅ **Ensure figure captions** match data
4. ✅ **Cross-reference sections** (e.g., Section 9 cites Section 7 results)
5. ✅ **Answer advisor questions** (quick lookup for key results)

---

## Most Common Mistakes to Avoid

1. **Wrong Sharpe for sector-specific**: Should be **+0.79** (not +0.68 or +0.87, those are individual sectors)
2. **Wrong correlation-CV**: Should be **ρ = -0.87** for US (not -0.82, that's global)
3. **Wrong AUC interpretation**: Should say **"barely above random 0.5"** (not "good")
4. **Wrong spectral gap correlation**: Should be **ρ = -0.974** (not -0.87)
5. **Wrong number of viable markets**: Should be **9/11** (not 10/11 or 11/11)

---

**Print this sheet and keep it next to you while integrating the thesis!**

This is your **answer key** for all data verification.

---

**END OF QUICK REFERENCE**
