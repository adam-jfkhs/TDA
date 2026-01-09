"""
Option C: Topology+NN Risk Overlay on a Short-Vol / Crash-Sensitive Proxy (Walk-forward)

Goal:
- Use topology/NN as a crash-risk gate for a strategy with negative skew (short-vol like).
- No option chains required: we create a crash-sensitive proxy from SPY returns.

Key design choices:
- Base = "short-vol proxy" from SPY daily returns (negative convexity)
- Overlay = stepwise exposure gating based on NN-estimated p_risk
- Walk-forward: threshold computed TRAIN-only each fold to avoid leakage

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
    end: Optional[str] = None

    price_field: str = "Adj Close"

    corr_lookback: int = 60

    # Risk label (future vol)
    label_horizon: int = 21
    risk_quantile: float = 0.75

    # Walk-forward
    train_years: int = 5
    test_years: int = 1

    # Costs (applied on exposure changes; this is a proxy so keep small)
    cost_bps: float = 2.0

    # NN
    hidden_layers: Tuple[int, ...] = (16, 16)
    alpha_l2: float = 1e-3
    max_iter: int = 500
    random_state: int = 42

    # Short-vol proxy strength (bigger => more negative skew / more crash-sensitive)
    crash_lambda: float = 2.0

    # Stepwise exposure gating thresholds:
    # p_risk > 0.80 => 0% exposure
    # p_risk > 0.60 => 25%
    # p_risk > 0.40 => 60%
    # else => 100%
    gate_hi: float = 0.80
    gate_mid: float = 0.60
    gate_lo: float = 0.40
    exp_hi: float = 0.00
    exp_mid: float = 0.25
    exp_lo: float = 0.60
    exp_ok: float = 1.00


# -----------------------------
# Data Fetching (ROBUST)
# -----------------------------

def fetch_prices(cfg: Config) -> pd.DataFrame:
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
        df = extract_field("Close")
        if df is None:
            raise KeyError(f"Neither '{cfg.price_field}' nor 'Close' found in yfinance columns.")

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.dropna(how="all").ffill().dropna()
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()


# -----------------------------
# Features (Topology Proxy)
# -----------------------------

def corr_features(window_rets: pd.DataFrame) -> Dict[str, float]:
    C = window_rets.corr().values
    n = C.shape[0]
    if n < 3:
        return {"mean_corr": np.nan, "corr_std": np.nan, "fiedler": np.nan}

    off = C[np.triu_indices(n, k=1)]
    mean_corr = float(np.nanmean(off))
    corr_std = float(np.nanstd(off))

    W = np.clip(C, 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)

    if np.any(d <= 1e-12):
        return {"mean_corr": mean_corr, "corr_std": corr_std, "fiedler": 0.0}

    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    evals = eigh(L, eigvals_only=True)
    evals = np.sort(np.real(evals))
    fiedler = float(evals[1]) if len(evals) > 1 else 0.0

    return {"mean_corr": mean_corr, "corr_std": corr_std, "fiedler": fiedler}


def build_feature_matrix(rets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows, idx = [], []
    for t in range(cfg.corr_lookback, len(rets)):
        end = rets.index[t]
        window = rets.iloc[t - cfg.corr_lookback : t]
        rows.append(corr_features(window))
        idx.append(end)
    return pd.DataFrame(rows, index=pd.Index(idx, name="date"))


# -----------------------------
# Labels (Future Vol)
# -----------------------------

def make_future_vol_series(rets: pd.DataFrame, cfg: Config, benchmark: str = "SPY") -> pd.Series:
    if benchmark not in rets.columns:
        raise ValueError(f"Benchmark {benchmark} must be included in tickers for labels.")
    fwd = rets[benchmark].rolling(cfg.label_horizon).std().shift(-cfg.label_horizon)
    fwd = fwd * np.sqrt(252)
    return fwd.rename("future_vol")


# -----------------------------
# Short-vol / Crash-sensitive Proxy
# -----------------------------

def short_vol_proxy_returns(spy_ret: pd.Series, crash_lambda: float) -> pd.Series:
    """
    A simple negative-skew proxy:
    proxy = r - lambda * I(r<0) * |r|
    so losses are amplified vs gains, like short convexity.
    """
    r = spy_ret.copy()
    penalty = crash_lambda * (r < 0).astype(float) * np.abs(r)
    proxy = r - penalty
    return proxy.rename("shortvol_proxy_ret")


def stepwise_exposure(p_risk: pd.Series, cfg: Config) -> pd.Series:
    """
    Step function gating: only reduce exposure when risk is high.
    """
    exp = pd.Series(cfg.exp_ok, index=p_risk.index, dtype=float)

    exp[p_risk > cfg.gate_lo] = cfg.exp_lo
    exp[p_risk > cfg.gate_mid] = cfg.exp_mid
    exp[p_risk > cfg.gate_hi] = cfg.exp_hi

    return exp.rename("exposure")


# -----------------------------
# Backtest Helpers
# -----------------------------

def apply_exposure_and_costs(base_ret: pd.Series, exposure: pd.Series, cost_bps: float) -> pd.Series:
    """
    Strategy return = exposure_{t-1} * base_ret_t - costs
    Costs proportional to change in exposure (turnover proxy).
    """
    base_ret = base_ret.reindex(exposure.index).dropna()
    exposure = exposure.reindex(base_ret.index).fillna(method="ffill").fillna(1.0)

    # cost on exposure changes
    turnover = exposure.diff().abs().fillna(0.0)
    costs = (cost_bps / 1e4) * turnover

    net = exposure.shift(1).fillna(exposure.iloc[0]) * base_ret - costs
    return net.rename("net_ret")


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
# Walk-forward splits
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
# Main system
# -----------------------------

def run_system(cfg: Config) -> Dict[str, object]:
    px = fetch_prices(cfg)
    rets = returns_from_prices(px)

    # Features
    X = build_feature_matrix(rets, cfg)

    # Labels (future vol)
    future_vol = make_future_vol_series(rets, cfg, benchmark="SPY")

    # Base short-vol proxy returns from SPY
    spy_ret = rets["SPY"].copy()
    base_proxy = short_vol_proxy_returns(spy_ret, cfg.crash_lambda)

    # Align
    common = X.index.intersection(future_vol.index).intersection(base_proxy.index)
    X = X.loc[common].dropna()
    future_vol = future_vol.loc[X.index]
    base_proxy = base_proxy.loc[X.index]

    splits = walk_forward_splits(X.index, cfg)

    all_base = []
    all_overlay = []
    all_p_risk = []
    all_exposure = []
    all_X_te = []
    fold_stats = []

    for (tr_s, tr_e, te_s, te_e) in splits:
        tr_mask = (X.index >= tr_s) & (X.index <= tr_e)
        te_mask = (X.index >= te_s) & (X.index <= te_e)

        X_tr, X_te = X.loc[tr_mask], X.loc[te_mask]
        vol_tr, vol_te = future_vol.loc[tr_mask], future_vol.loc[te_mask]
        base_te = base_proxy.loc[X_te.index]

        if len(X_tr) < 200 or len(X_te) < 50:
            continue

        # TRAIN-only threshold for regime label
        thr = float(np.nanquantile(vol_tr.dropna(), cfg.risk_quantile))
        y_tr = (vol_tr >= thr).astype(int).fillna(0).values

        # NN
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

        p_risk = pd.Series(model.predict_proba(X_te.values)[:, 1], index=X_te.index, name="p_risk")

        # Stepwise exposure gating
        exposure = stepwise_exposure(p_risk, cfg)

        # Base vs overlay returns
        # Base: full exposure (1.0) to the proxy
        base_ret = apply_exposure_and_costs(base_te, pd.Series(1.0, index=base_te.index), cfg.cost_bps).rename("base_ret")
        overlay_ret = apply_exposure_and_costs(base_te, exposure, cfg.cost_bps).rename("overlay_ret")

        all_base.append(base_ret)
        all_overlay.append(overlay_ret)
        all_p_risk.append(p_risk)
        all_exposure.append(exposure)
        all_X_te.append(X_te)

        eq_b = (1 + base_ret).cumprod()
        eq_o = (1 + overlay_ret).cumprod()

        fold_stats.append({
            "test_start": str(te_s.date()),
            "test_end": str(te_e.date()),
            "base_sharpe": sharpe(base_ret),
            "overlay_sharpe": sharpe(overlay_ret),
            "base_maxdd": max_drawdown(eq_b),
            "overlay_maxdd": max_drawdown(eq_o),
            "avg_p_risk": float(p_risk.mean()),
            "avg_exposure": float(exposure.mean()),
        })

    if not all_overlay:
        raise RuntimeError("No valid folds produced. Expand date range or adjust cfg.")

    base = pd.concat(all_base).sort_index()
    overlay = pd.concat(all_overlay).sort_index()

    eq_base = (1 + base).cumprod()
    eq_overlay = (1 + overlay).cumprod()

    out = {
        "base_returns": base,
        "overlay_returns": overlay,
        "base_equity": eq_base,
        "overlay_equity": eq_overlay,
        "base_sharpe": sharpe(base),
        "overlay_sharpe": sharpe(overlay),
        "base_cagr": cagr(eq_base),
        "overlay_cagr": cagr(eq_overlay),
        "base_maxdd": max_drawdown(eq_base),
        "overlay_maxdd": max_drawdown(eq_overlay),
        "fold_stats": pd.DataFrame(fold_stats),
        "p_risk": pd.concat(all_p_risk).sort_index(),
        "exposure": pd.concat(all_exposure).sort_index(),
        "features": pd.concat(all_X_te).sort_index(),
        "config": cfg,
    }
    return out


# -----------------------------
# Plotting
# -----------------------------

def plot_performance(out: Dict[str, object], title: str = "Short-Vol Proxy: Base vs NN Overlay"):
    base = out["base_returns"].dropna()
    overlay = out["overlay_returns"].dropna()
    eq_b = out["base_equity"].dropna()
    eq_o = out["overlay_equity"].dropna()

    dd_b = eq_b / eq_b.cummax() - 1
    dd_o = eq_o / eq_o.cummax() - 1

    win = 252
    def rolling_sharpe(r: pd.Series) -> pd.Series:
        m = r.rolling(win).mean()
        s = r.rolling(win).std()
        return np.sqrt(252) * (m / s)

    rs_b = rolling_sharpe(base)
    rs_o = rolling_sharpe(overlay)

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(title)

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(eq_b.index, eq_b.values, label="Base")
    ax1.plot(eq_o.index, eq_o.values, label="Overlay")
    ax1.set_title("Equity (Linear)")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Equity")
    ax1.legend()

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(eq_b.index, np.log(eq_b.values), label="Base")
    ax2.plot(eq_o.index, np.log(eq_o.values), label="Overlay")
    ax2.set_title("Equity (Log)")
    ax2.set_xlabel("Date"); ax2.set_ylabel("log(Equity)")
    ax2.legend()

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(dd_b.index, dd_b.values, label="Base")
    ax3.plot(dd_o.index, dd_o.values, label="Overlay")
    ax3.set_title("Drawdown")
    ax3.set_xlabel("Date"); ax3.set_ylabel("Drawdown")
    ax3.axhline(0.0, linewidth=1)
    ax3.legend()

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(rs_b.index, rs_b.values, label="Base")
    ax4.plot(rs_o.index, rs_o.values, label="Overlay")
    ax4.set_title("Rolling Sharpe (252d)")
    ax4.set_xlabel("Date"); ax4.set_ylabel("Rolling Sharpe")
    ax4.axhline(0.0, linewidth=1)
    ax4.legend()

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.hist(base.values, bins=60, alpha=0.6, label="Base")
    ax5.hist(overlay.values, bins=60, alpha=0.6, label="Overlay")
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
    p_risk = out["p_risk"].dropna()
    exposure = out["exposure"].dropna()
    X = out["features"].dropna()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(p_risk.index, p_risk.values)
    ax1.set_title("NN Risk Probability (p_risk)")
    ax1.set_xlabel("Date"); ax1.set_ylabel("p_risk")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(exposure.index, exposure.values)
    ax2.set_title("Stepwise Exposure (Gate Output)")
    ax2.set_xlabel("Date"); ax2.set_ylabel("Exposure")
    ax2.set_ylim(-0.05, 1.05)

    ax3 = fig.add_subplot(3, 1, 3)
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
        # Include SPY for proxy + label benchmark
        tickers=["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"],
        start="2010-01-01",
        price_field="Adj Close",
        corr_lookback=60,
        label_horizon=21,
        risk_quantile=0.75,
        train_years=5,
        test_years=1,
        cost_bps=2.0,
        hidden_layers=(16, 16),
        alpha_l2=1e-3,
        max_iter=500,
        random_state=42,
        crash_lambda=2.0,   # try 1.5, 2.0, 2.5
        gate_hi=0.80, gate_mid=0.60, gate_lo=0.40,
        exp_hi=0.00, exp_mid=0.25, exp_lo=0.60, exp_ok=1.00
    )

    out = run_system(cfg)

    print("\n=== Overall Summary (Short-Vol Proxy) ===")
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

    plot_performance(out, "Short-Vol Proxy: Base vs Topology+NN Overlay")
    plot_regime_diagnostics(out, "NN Risk + Step Exposure + Structure Features")