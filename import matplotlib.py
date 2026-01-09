"""
Topology + Neural Net Regime Overlay (Walk-forward) with Performance Visualizations

What it does:
- Downloads prices via yfinance (robust to MultiIndex columns)
- Computes rolling structure features from correlation networks:
    * mean_corr, corr_std, fiedler (normalized Laplacian 2nd eigenvalue)
- Defines a "risk regime" label using FUTURE realized vol of SPY (threshold set on TRAIN only per fold)
- Trains an MLPClassifier (neural net) to estimate P(risk_regime)
- Trades a base strategy (momentum) and scales exposure by (1 - p_risk)
- Walk-forward backtest with transaction costs
- Plots: equity (linear+log), drawdown, rolling Sharpe, baseline vs overlay, p_risk, exposure, features

Install:
pip install yfinance numpy pandas scipy scikit-learn matplotlib
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt
from scipy.linalg import eigh

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    tickers: List[str]

    start: str = "2010-01-01"
    end: Optional[str] = None  # None => today

    # Try "Adj Close" first, fallback will auto-handle if not present
    price_field: str = "Adj Close"

    corr_lookback: int = 60

    # Base strategy (momentum)
    momentum_lookback: int = 252  # 12m momentum
    holding_period: int = 21      # rebalance monthly-ish
    top_n: int = 3                # long top_n assets equally

    # Regime label (risk): future realized vol quantile of SPY
    label_horizon: int = 21
    risk_quantile: float = 0.75

    # Walk-forward splits (calendar-year based)
    train_years: int = 5
    test_years: int = 1

    # Costs (simple turnover model)
    cost_bps: float = 5.0  # 5 bps per 1.0 daily turnover

    # NN
    hidden_layers: Tuple[int, ...] = (16, 16)
    alpha_l2: float = 1e-3
    max_iter: int = 500
    random_state: int = 42


# -----------------------------
# Data Fetching (ROBUST)
# -----------------------------

def fetch_prices(cfg: Config) -> pd.DataFrame:
    """
    Robust yfinance download:
    - Handles both MultiIndex and single-level columns
    - Handles cases where 'Adj Close' is missing by falling back to 'Close'
    """
    raw = yf.download(
        cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=True,
    )

    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data. Check tickers/network.")

    def extract_field(field: str) -> Optional[pd.DataFrame]:
        if isinstance(raw.columns, pd.MultiIndex):
            if field in raw.columns.get_level_values(0):
                return raw[field].copy()
            return None
        else:
            if field in raw.columns:
                return raw[field].copy()
            return None

    df = extract_field(cfg.price_field)
    if df is None:
        # fallback
        df = extract_field("Close")
        if df is None:
            raise KeyError(f"Neither '{cfg.price_field}' nor 'Close' found in yfinance columns.")

    # Ensure DataFrame (single ticker can sometimes produce Series in some setups)
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Clean
    df = df.dropna(how="all").ffill().dropna()
    # Sort columns consistently
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()


# -----------------------------
# Graph / Topology Proxy Features
# -----------------------------

def corr_features(window_rets: pd.DataFrame) -> Dict[str, float]:
    """
    Fast structure features from correlation matrix:
    - mean_corr: mean off-diagonal correlation
    - corr_std: std off-diagonal correlation
    - fiedler: 2nd smallest eigenvalue of normalized Laplacian of W=max(corr,0)
    """
    C = window_rets.corr().values
    n = C.shape[0]
    if n < 3:
        return {"mean_corr": np.nan, "corr_std": np.nan, "fiedler": np.nan}

    off = C[np.triu_indices(n, k=1)]
    mean_corr = float(np.nanmean(off))
    corr_std = float(np.nanstd(off))

    # weights: nonnegative correlations only
    W = np.clip(C, 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)

    if np.any(d <= 1e-12):
        # disconnected / isolates -> fiedler approx 0
        return {"mean_corr": mean_corr, "corr_std": corr_std, "fiedler": 0.0}

    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    evals = eigh(L, eigvals_only=True)
    evals = np.sort(np.real(evals))
    fiedler = float(evals[1]) if len(evals) > 1 else 0.0

    return {"mean_corr": mean_corr, "corr_std": corr_std, "fiedler": fiedler}


def build_feature_matrix(rets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    idx = []

    for t in range(cfg.corr_lookback, len(rets)):
        end = rets.index[t]
        window = rets.iloc[t - cfg.corr_lookback : t]
        rows.append(corr_features(window))
        idx.append(end)

    X = pd.DataFrame(rows, index=pd.Index(idx, name="date"))
    return X


# -----------------------------
# Labels (Risk Regime)
# -----------------------------

def make_future_vol_series(rets: pd.DataFrame, cfg: Config, benchmark: str = "SPY") -> pd.Series:
    """
    Future realized vol proxy: std of benchmark returns over next horizon (annualized).
    Label thresholding happens per-fold using TRAIN only (prevents leakage).
    """
    if benchmark not in rets.columns:
        raise ValueError(f"Benchmark {benchmark} must be included in tickers for labels.")

    fwd = rets[benchmark].rolling(cfg.label_horizon).std().shift(-cfg.label_horizon)
    fwd = fwd * np.sqrt(252)
    return fwd.rename("future_vol")


# -----------------------------
# Base Strategy (Momentum)
# -----------------------------

def momentum_scores(px: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return px.pct_change(lookback)


def compute_positions_momentum(px: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Monthly rebalance: long top_n by momentum, equal weight.
    Held constant between rebalance points via forward-fill.
    """
    scores = momentum_scores(px, cfg.momentum_lookback)

    # Rebalance schedule
    rebal_dates = px.index[::cfg.holding_period]
    pos = pd.DataFrame(0.0, index=px.index, columns=px.columns)

    for d in rebal_dates:
        if d not in scores.index:
            continue
        row = scores.loc[d].dropna()
        if len(row) < cfg.top_n:
            continue
        top = row.sort_values(ascending=False).head(cfg.top_n).index.tolist()
        w = 1.0 / cfg.top_n
        pos.loc[d, top] = w

    # Forward-fill positions
    pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)
    return pos


# -----------------------------
# Backtest Helpers
# -----------------------------

def apply_costs(positions: pd.DataFrame, rets: pd.DataFrame, cost_bps: float) -> pd.Series:
    """
    Simple turnover cost model:
    cost = cost_bps * sum(abs(delta_weights)) per day
    """
    pos = positions.reindex(rets.index).fillna(0.0)
    turnover = pos.diff().abs().sum(axis=1)
    costs = (cost_bps / 1e4) * turnover
    pnl_gross = (pos.shift(1) * rets).sum(axis=1)
    pnl_net = pnl_gross - costs
    return pnl_net.rename("strategy_ret")


def sharpe(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 10 or daily_ret.std() == 0:
        return 0.0
    return float(np.sqrt(252) * daily_ret.mean() / daily_ret.std())


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1)


# -----------------------------
# Walk-forward Splits
# -----------------------------

def walk_forward_splits(dates: pd.DatetimeIndex, cfg: Config) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    years = sorted(list({d.year for d in dates}))
    splits = []

    for i in range(0, len(years)):
        train_start_year = years[i]
        train_end_year = train_start_year + cfg.train_years - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + cfg.test_years - 1

        if test_end_year > years[-1]:
            break

        train_start = pd.Timestamp(f"{train_start_year}-01-01")
        train_end = pd.Timestamp(f"{train_end_year}-12-31")
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end = pd.Timestamp(f"{test_end_year}-12-31")

        splits.append((train_start, train_end, test_start, test_end))

    return splits


# -----------------------------
# Main System
# -----------------------------

def run_system(cfg: Config) -> Dict[str, object]:
    px = fetch_prices(cfg)
    rets = returns_from_prices(px)

    # Features
    X = build_feature_matrix(rets, cfg)

    # Future vol target series (thresholding per fold)
    future_vol = make_future_vol_series(rets, cfg, benchmark="SPY")

    # Base positions (momentum)
    base_pos = compute_positions_momentum(px, cfg)

    # Align indices
    common = X.index.intersection(future_vol.index).intersection(rets.index).intersection(base_pos.index)
    X = X.loc[common].dropna()
    future_vol = future_vol.loc[X.index]
    rets_aligned = rets.loc[X.index]
    base_pos_aligned = base_pos.loc[X.index]

    splits = walk_forward_splits(X.index, cfg)

    all_net_overlay = []
    all_net_base = []
    fold_stats = []

    # diagnostics
    all_p_risk = []
    all_exposure_overlay = []
    all_exposure_base = []
    all_X_te = []

    for (tr_s, tr_e, te_s, te_e) in splits:
        tr_mask = (X.index >= tr_s) & (X.index <= tr_e)
        te_mask = (X.index >= te_s) & (X.index <= te_e)

        X_tr, X_te = X.loc[tr_mask], X.loc[te_mask]
        vol_tr, vol_te = future_vol.loc[tr_mask], future_vol.loc[te_mask]

        if len(X_tr) < 200 or len(X_te) < 50:
            continue

        # TRAIN-only threshold
        thr = float(np.nanquantile(vol_tr.dropna(), cfg.risk_quantile))
        y_tr = (vol_tr >= thr).astype(int).fillna(0).values

        # Model
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=cfg.hidden_layers,
                activation="relu",
                alpha=cfg.alpha_l2,
                max_iter=cfg.max_iter,
                random_state=cfg.random_state
            ))
        ])
        model.fit(X_tr.values, y_tr)

        # Predict risk probability on TEST
        p_risk = pd.Series(model.predict_proba(X_te.values)[:, 1], index=X_te.index, name="p_risk")
        overlay_scale = (1.0 - p_risk).clip(0.0, 1.0)

        # Positions
        pos_base = base_pos_aligned.loc[X_te.index].copy()
        pos_overlay = pos_base.mul(overlay_scale, axis=0)

        # Diagnostics exposures
        exp_base = pos_base.abs().sum(axis=1).rename("exposure_base")
        exp_overlay = pos_overlay.abs().sum(axis=1).rename("exposure_overlay")

        # Returns
        rets_te = rets_aligned.loc[X_te.index]
        net_base = apply_costs(pos_base, rets_te, cfg.cost_bps).rename("base_ret")
        net_overlay = apply_costs(pos_overlay, rets_te, cfg.cost_bps).rename("overlay_ret")

        all_net_base.append(net_base)
        all_net_overlay.append(net_overlay)

        # store diagnostics
        all_p_risk.append(p_risk)
        all_exposure_base.append(exp_base)
        all_exposure_overlay.append(exp_overlay)
        all_X_te.append(X_te)

        # fold stats
        eq_base = (1.0 + net_base).cumprod()
        eq_overlay = (1.0 + net_overlay).cumprod()

        fold_stats.append({
            "test_start": str(te_s.date()),
            "test_end": str(te_e.date()),
            "base_sharpe": sharpe(net_base),
            "overlay_sharpe": sharpe(net_overlay),
            "base_maxdd": max_drawdown(eq_base),
            "overlay_maxdd": max_drawdown(eq_overlay),
            "avg_p_risk": float(p_risk.mean()),
            "avg_exp_base": float(exp_base.mean()),
            "avg_exp_overlay": float(exp_overlay.mean()),
        })

    if not all_net_overlay:
        raise RuntimeError("No valid folds produced. Expand date range or adjust cfg.")

    base = pd.concat(all_net_base).sort_index()
    overlay = pd.concat(all_net_overlay).sort_index()

    eq_base = (1.0 + base).cumprod()
    eq_overlay = (1.0 + overlay).cumprod()

    p_risk_all = pd.concat(all_p_risk).sort_index() if all_p_risk else pd.Series(dtype=float)
    exp_base_all = pd.concat(all_exposure_base).sort_index() if all_exposure_base else pd.Series(dtype=float)
    exp_overlay_all = pd.concat(all_exposure_overlay).sort_index() if all_exposure_overlay else pd.Series(dtype=float)
    feats_all = pd.concat(all_X_te).sort_index() if all_X_te else pd.DataFrame()

    return {
        "base_returns": base,
        "overlay_returns": overlay,
        "base_equity": eq_base,
        "overlay_equity": eq_overlay,
        "base_sharpe": sharpe(base),
        "overlay_sharpe": sharpe(overlay),
        "base_maxdd": max_drawdown(eq_base),
        "overlay_maxdd": max_drawdown(eq_overlay),
        "base_cagr": cagr(eq_base),
        "overlay_cagr": cagr(eq_overlay),
        "fold_stats": pd.DataFrame(fold_stats),
        "p_risk": p_risk_all,
        "exposure_base": exp_base_all,
        "exposure_overlay": exp_overlay_all,
        "features": feats_all,
        "config": cfg,
    }


# -----------------------------
# Plotting
# -----------------------------

def plot_performance(out: Dict[str, object], title: str = "Backtest Performance"):
    base = out["base_returns"].dropna()
    overlay = out["overlay_returns"].dropna()

    eq_base = out["base_equity"].dropna()
    eq_overlay = out["overlay_equity"].dropna()

    # Drawdowns
    dd_base = eq_base / eq_base.cummax() - 1.0
    dd_overlay = eq_overlay / eq_overlay.cummax() - 1.0

    # Rolling Sharpe
    win = 252
    def rolling_sharpe(r: pd.Series) -> pd.Series:
        m = r.rolling(win).mean()
        s = r.rolling(win).std()
        return np.sqrt(252) * (m / s)

    rs_base = rolling_sharpe(base)
    rs_overlay = rolling_sharpe(overlay)

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(title)

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(eq_base.index, eq_base.values, label="Base")
    ax1.plot(eq_overlay.index, eq_overlay.values, label="Overlay")
    ax1.set_title("Equity (Linear)")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Equity")
    ax1.legend()

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(eq_base.index, np.log(eq_base.values), label="Base")
    ax2.plot(eq_overlay.index, np.log(eq_overlay.values), label="Overlay")
    ax2.set_title("Equity (Log)")
    ax2.set_xlabel("Date"); ax2.set_ylabel("log(Equity)")
    ax2.legend()

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(dd_base.index, dd_base.values, label="Base")
    ax3.plot(dd_overlay.index, dd_overlay.values, label="Overlay")
    ax3.set_title("Drawdown")
    ax3.set_xlabel("Date"); ax3.set_ylabel("Drawdown")
    ax3.axhline(0.0, linewidth=1)
    ax3.legend()

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(rs_base.index, rs_base.values, label="Base")
    ax4.plot(rs_overlay.index, rs_overlay.values, label="Overlay")
    ax4.set_title("Rolling Sharpe (252d)")
    ax4.set_xlabel("Date"); ax4.set_ylabel("Rolling Sharpe")
    ax4.axhline(0.0, linewidth=1)
    ax4.legend()

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.hist(base.dropna().values, bins=50, alpha=0.6, label="Base")
    ax5.hist(overlay.dropna().values, bins=50, alpha=0.6, label="Overlay")
    ax5.set_title("Daily Return Distribution")
    ax5.set_xlabel("Daily Return"); ax5.set_ylabel("Count")
    ax5.legend()

    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot((1 + base).cumprod().index, (1 + base).cumprod().values, label="Base")
    ax6.plot((1 + overlay).cumprod().index, (1 + overlay).cumprod().values, label="Overlay")
    ax6.set_title("Equity (Redundant Check)")
    ax6.set_xlabel("Date"); ax6.set_ylabel("Equity")
    ax6.legend()

    plt.tight_layout()
    plt.show()


def plot_regime_diagnostics(out: Dict[str, object], title: str = "Regime Diagnostics"):
    p_risk = out.get("p_risk", pd.Series(dtype=float)).dropna()
    exp_b = out.get("exposure_base", pd.Series(dtype=float)).dropna()
    exp_o = out.get("exposure_overlay", pd.Series(dtype=float)).dropna()
    X = out.get("features", pd.DataFrame()).dropna()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title)

    ax1 = fig.add_subplot(3, 1, 1)
    if len(p_risk) > 0:
        ax1.plot(p_risk.index, p_risk.values)
    ax1.set_title("NN Risk Probability (p_risk)")
    ax1.set_xlabel("Date"); ax1.set_ylabel("p_risk")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = fig.add_subplot(3, 1, 2)
    if len(exp_b) > 0:
        ax2.plot(exp_b.index, exp_b.values, label="Base Exposure")
    if len(exp_o) > 0:
        ax2.plot(exp_o.index, exp_o.values, label="Overlay Exposure")
    ax2.set_title("Exposure Over Time")
    ax2.set_xlabel("Date"); ax2.set_ylabel("Sum |weights|")
    ax2.legend()

    ax3 = fig.add_subplot(3, 1, 3)
    if isinstance(X, pd.DataFrame) and len(X) > 0:
        for col in ["mean_corr", "corr_std", "fiedler"]:
            if col in X.columns:
                ax3.plot(X.index, X[col].values, label=col)
        ax3.legend()
    ax3.set_title("Structure Features Over Time")
    ax3.set_xlabel("Date"); ax3.set_ylabel("Value")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    cfg = Config(
        # Include SPY because we use it for the risk label benchmark
        tickers=["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"],
        start="2010-01-01",
        price_field="Adj Close",  # robust fetcher will fall back to Close automatically
        corr_lookback=60,
        momentum_lookback=252,
        holding_period=21,
        top_n=3,
        cost_bps=5.0,
        train_years=5,
        test_years=1,
        risk_quantile=0.75,
        hidden_layers=(16, 16),
        alpha_l2=1e-3,
        max_iter=500,
        random_state=42,
    )

    out = run_system(cfg)

    print("\n=== Overall Summary ===")
    print(f"Base Sharpe:    {out['base_sharpe']:.3f}")
    print(f"Overlay Sharpe: {out['overlay_sharpe']:.3f}")
    print(f"Base CAGR:      {out['base_cagr']:.2%}")
    print(f"Overlay CAGR:   {out['overlay_cagr']:.2%}")
    print(f"Base MaxDD:     {out['base_maxdd']:.2%}")
    print(f"Overlay MaxDD:  {out['overlay_maxdd']:.2%}")

    fs = out["fold_stats"]
    if isinstance(fs, pd.DataFrame) and len(fs) > 0:
        print("\n=== Fold-by-fold stats ===")
        print(fs.to_string(index=False))

    plot_performance(out, "Base vs Topology+NN Overlay (Momentum Base)")
    plot_regime_diagnostics(out, "NN Risk + Exposure + Structure Features")