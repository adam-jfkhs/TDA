"""
Real-data ML Integration (replaces phase5_ml_integration/ML_INTEGRATION.py)
==========================================================================
Re-runs Section 10 of the thesis using REAL Yahoo Finance data with strict
1-day signal lag. No simulated regime-switching data.

Outputs:
    ml_integration_real.csv     — per-model metrics (F1, AUC, precision, recall, Sharpe)
    ml_integration_features.csv — feature importance from Random Forest
    ml_integration_real.txt     — plain-text summary table (paste into Section 10)

Strategy logic:
    - Universe: same 3 sectors as risk_report.py (Tech, Financials, Energy)
    - Daily features (rolling 60-day window):
        - mean_corr, std_corr (correlation moments)
        - lambda1, lambda2, spectral_gap, fiedler  (eigenvalue features —
          Section 11 shows Fiedler -0.991 correlated with topology CV;
          uses these instead of expensive ripser persistent homology)
        - realized_vol (annualised)
    - Target (binary): sign of next-day equal-weight sector return
    - Models: Random Forest, Gradient Boosting, Neural Net
    - Split: walk-forward, train pre-2022, test 2022-01-01 to 2024-12-31
    - All predictions made from features at date t, evaluated against return
      at date t+1 (no look-ahead).

Run on Google Colab:
    !pip install yfinance scikit-learn pandas numpy scipy
    !python ml_integration_real.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.linalg as la
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, accuracy_score)

# ── CONFIG ────────────────────────────────────────────────────────────────────
SECTORS = {
    "Financials": ["JPM", "BAC", "WFC", "GS",  "MS",   "BLK"],
    "Energy":     ["XOM", "CVX", "COP", "SLB",  "MPC",  "VLO"],
    "Technology": ["AAPL","MSFT","NVDA","AMD",  "INTC", "ORCL"],
}
START      = "2019-01-01"
END        = "2024-12-31"
TRAIN_END  = "2021-12-31"
TEST_START = "2022-01-01"
LOOKBACK   = 60
CORR_THR   = 0.30

# ── DATA ──────────────────────────────────────────────────────────────────────

def download():
    import yfinance as yf
    tickers = sorted({t for ts in SECTORS.values() for t in ts})
    raw = yf.download(tickers, start=START, end=END,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return np.log(prices / prices.shift(1)).dropna()

# ── FEATURES ──────────────────────────────────────────────────────────────────

def laplacian_eigs(corr: np.ndarray, threshold: float):
    """Return (lambda1, lambda2, fiedler) of normalised graph Laplacian."""
    W = np.where(corr > threshold, corr, 0.0)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    d = np.where(d == 0, 1.0, d)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(len(d)) - D_inv_sqrt @ W @ D_inv_sqrt
    vals = np.sort(la.eigvalsh(L))
    return float(vals[-1]), float(vals[-2]), float(vals[1])

def compute_features(returns: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Daily feature matrix from rolling correlation."""
    sect = returns[tickers].dropna()
    rows = []
    for i in range(LOOKBACK, len(sect)):
        window = sect.iloc[i-LOOKBACK:i]
        corr = window.corr().values
        upper = corr[np.triu_indices(len(tickers), k=1)]
        l1, l2, fv = laplacian_eigs(corr, CORR_THR)
        rows.append({
            "date":          sect.index[i],
            "mean_corr":     upper.mean(),
            "std_corr":      upper.std(),
            "lambda1":       l1,
            "lambda2":       l2,
            "spectral_gap":  l1 - l2,
            "fiedler":       fv,
            "realized_vol":  window.std().mean() * np.sqrt(252),
        })
    return pd.DataFrame(rows).set_index("date")

# ── TARGET (NEXT-DAY SECTOR RETURN SIGN, NO LOOK-AHEAD) ───────────────────────

def build_dataset(returns: pd.DataFrame):
    feats_all, targets_all = [], []
    for sec_name, tickers in SECTORS.items():
        avail = [t for t in tickers if t in returns.columns]
        if len(avail) < 4:
            continue
        feats = compute_features(returns, avail)
        sec_eq = returns[avail].mean(axis=1)
        # target_t = sign(return at t+1); features at t predict t+1
        next_ret = sec_eq.shift(-1)
        target = (next_ret > 0).astype(int)
        df = feats.join(target.rename("target"), how="inner").dropna()
        df["sector"] = sec_name
        df["next_ret"] = next_ret.reindex(df.index).values
        feats_all.append(df)
    return pd.concat(feats_all).sort_index()

# ── MODELS ────────────────────────────────────────────────────────────────────

FEATURE_COLS = ["mean_corr", "std_corr", "lambda1", "lambda2",
                "spectral_gap", "fiedler", "realized_vol"]

def evaluate(name, model, X_tr, y_tr, X_te, y_te, ret_te):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
    pred  = model.predict(X_te)

    # Strategy: long when model predicts 1 (next-day positive), flat else.
    # PnL is realised next-day return (already lagged in build_dataset).
    pos = (pred == 1).astype(float) - (pred == 0).astype(float)  # long/short
    strat_ret = pos * ret_te.values
    sr = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() else np.nan

    return {
        "model":     name,
        "f1":        f1_score(y_te, pred, zero_division=0),
        "auc":       roc_auc_score(y_te, proba) if proba is not None else np.nan,
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall":    recall_score(y_te, pred, zero_division=0),
        "accuracy":  accuracy_score(y_te, pred),
        "sharpe":    sr,
        "n_test":    len(y_te),
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[1/4] Downloading {sum(len(v) for v in SECTORS.values())} tickers …")
    returns = download()
    print(f"      {returns.shape[0]} days × {returns.shape[1]} tickers")

    print("[2/4] Computing features (mean_corr, spectral_gap, Fiedler, vol) …")
    df = build_dataset(returns)
    print(f"      {len(df)} rows total ({len(SECTORS)} sectors stacked)")

    train_mask = df.index <= TRAIN_END
    test_mask  = df.index >= TEST_START
    X_tr = df.loc[train_mask, FEATURE_COLS]
    y_tr = df.loc[train_mask, "target"]
    X_te = df.loc[test_mask, FEATURE_COLS]
    y_te = df.loc[test_mask, "target"]
    ret_te = df.loc[test_mask, "next_ret"]
    print(f"      train: {len(y_tr)}  test: {len(y_te)}  base rate: {y_tr.mean():.3f}")

    # Threshold baseline = predict 1 iff fiedler below training 25th pct
    print("[3/4] Fitting models …")
    fv_thr = X_tr["fiedler"].quantile(0.25)
    threshold_pred = (X_te["fiedler"] < fv_thr).astype(int)
    threshold_ret = ((threshold_pred == 1).astype(float)
                     - (threshold_pred == 0).astype(float)) * ret_te.values
    threshold_sr = (threshold_ret.mean() / threshold_ret.std() * np.sqrt(252)
                    if threshold_ret.std() else np.nan)

    results = [
        {"model": "Threshold (Fiedler<25pct)",
         "f1":        f1_score(y_te, threshold_pred, zero_division=0),
         "auc":       roc_auc_score(y_te, -X_te["fiedler"].values),
         "precision": precision_score(y_te, threshold_pred, zero_division=0),
         "recall":    recall_score(y_te, threshold_pred, zero_division=0),
         "accuracy":  accuracy_score(y_te, threshold_pred),
         "sharpe":    threshold_sr,
         "n_test":    len(y_te)},
        evaluate("Random Forest",
                 RandomForestClassifier(n_estimators=200, max_depth=8,
                                        min_samples_leaf=20, random_state=42,
                                        n_jobs=-1),
                 X_tr, y_tr, X_te, y_te, ret_te),
        evaluate("Gradient Boosting",
                 GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=42),
                 X_tr, y_tr, X_te, y_te, ret_te),
        evaluate("Neural Network",
                 MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500,
                               early_stopping=True, random_state=42),
                 X_tr, y_tr, X_te, y_te, ret_te),
    ]

    # Feature importance from RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    feat_imp = pd.Series(rf.feature_importances_,
                         index=FEATURE_COLS).sort_values(ascending=False)

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    print("[4/4] Writing outputs …")
    res_df = pd.DataFrame(results).set_index("model").round(4)
    res_df.to_csv("ml_integration_real.csv")
    feat_imp.round(4).to_csv("ml_integration_features.csv",
                             header=["importance"])

    lines = []
    lines.append("=" * 80)
    lines.append("  ML INTEGRATION — REAL DATA, 1-DAY LAG, OUT-OF-SAMPLE 2022–2024")
    lines.append("=" * 80)
    lines.append(f"  Universe: {sum(len(v) for v in SECTORS.values())} stocks "
                 f"across {len(SECTORS)} sectors")
    lines.append(f"  Train:    pre-{TRAIN_END}  ({len(y_tr)} obs)")
    lines.append(f"  Test:     {TEST_START} – {END}  ({len(y_te)} obs)")
    lines.append("")
    lines.append(f"{'Model':<32}{'F1':>8}{'AUC':>8}{'Prec':>8}{'Recall':>8}{'Sharpe':>8}")
    lines.append("-" * 80)
    for r in results:
        lines.append(f"{r['model']:<32}"
                     f"{r['f1']:>8.3f}{r['auc']:>8.3f}"
                     f"{r['precision']:>8.3f}{r['recall']:>8.3f}{r['sharpe']:>8.2f}")
    lines.append("")
    lines.append("Feature importance (Random Forest):")
    for f, imp in feat_imp.items():
        lines.append(f"  {f:<18}{imp:>6.3f}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  Features at day t predict sign of return at t+1 (strict lag).")
    lines.append("  Spectral-graph features (Fiedler) replace ripser persistent")
    lines.append("  homology — Section 11 shows -0.991 correlation with topology CV.")
    lines.append("=" * 80)
    text = "\n".join(lines)
    print(text)
    Path("ml_integration_real.txt").write_text(text)

if __name__ == "__main__":
    main()
