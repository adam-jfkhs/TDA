"""
Real-data Strategy Variants (replaces phase3_strategy_variants/)
================================================================
Re-runs Section 8 of the thesis using REAL Yahoo Finance data with strict
1-day signal lag. Replaces the look-ahead-biased Phase 3 scripts.

Variants tested (each on Technology, Financials, Energy):
  V1: Mean-Reversion (Phase 2 baseline, fixed: t→t+1 lag)
  V2: Momentum + TDA Hybrid
        - Fiedler in stressed regime (low) → mean-revert (long losers, short winners)
        - Fiedler in calm regime (high)   → momentum (long winners, short losers)
  V3: Adaptive Threshold (rolling 60-day Z-score on Fiedler)
        - z > +1: strongly stressed → mean reversion
        - z < -1: strongly calm     → momentum
        - else flat
  V4: Ensemble — equal-weight combination of V1, V2, V3

Outputs:
    strategy_variants_real.csv     — per-variant metrics
    strategy_variants_real.txt     — summary table for Section 8

Run on Google Colab:
    !pip install yfinance pandas-datareader scipy
    !python strategy_variants_real.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.linalg as la
from pathlib import Path

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
TC_BPS     = 5
N_LONG     = 2
N_SHORT    = 2
MOM_WIN    = 20

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

def fiedler_value(corr: np.ndarray, threshold: float) -> float:
    W = np.where(corr > threshold, corr, 0.0)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    d = np.where(d == 0, 1.0, d)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(len(d)) - D_inv_sqrt @ W @ D_inv_sqrt
    vals = np.sort(la.eigvalsh(L))
    return float(vals[1]) if len(vals) > 1 else 0.0

def sector_signals(returns: pd.DataFrame, tickers: list) -> pd.DataFrame:
    sect = returns[tickers].dropna()
    rows = []
    for i in range(LOOKBACK, len(sect)):
        window = sect.iloc[i-LOOKBACK:i]
        corr = window.corr().values
        rows.append({
            "date":     sect.index[i],
            "fiedler":  fiedler_value(corr, CORR_THR),
            "mom_20":   sect.iloc[i-MOM_WIN:i].sum().to_dict(),
        })
    sig = pd.DataFrame(rows).set_index("date")
    return sig

def select_positions(returns_window: pd.DataFrame, tickers: list,
                     mode: str) -> np.ndarray:
    """
    mode = 'mean_rev' → long worst, short best
    mode = 'momentum' → long best,  short worst
    mode = 'flat'     → all zeros
    """
    pos = np.zeros(len(tickers))
    if mode == "flat":
        return pos
    last_ret = returns_window.iloc[-1].values
    if mode == "mean_rev":
        idx = np.argsort(last_ret)               # ascending
        pos[idx[:N_LONG]]   =  1.0 / N_LONG
        pos[idx[-N_SHORT:]] = -1.0 / N_SHORT
    else:  # momentum
        mom = returns_window.iloc[-MOM_WIN:].sum().values
        idx = np.argsort(mom)                    # ascending
        pos[idx[:N_LONG]]   = -1.0 / N_LONG      # short losers
        pos[idx[-N_SHORT:]] =  1.0 / N_SHORT     # long winners
    return pos

def run_variant(returns: pd.DataFrame, tickers: list, variant: str,
                fv_train_q25: float, fv_train_q75: float,
                fv_train_mu: float, fv_train_sd: float):
    """
    Returns daily strategy returns for one variant on one sector.
    All positions taken on day t use signals from day t-1 (no look-ahead).
    """
    sect = returns[tickers].dropna()
    dates = sect.index
    rows = []
    prev_pos = np.zeros(len(tickers))

    fv_history = []          # for adaptive z-score
    for i in range(LOOKBACK + 1, len(sect)):
        # Compute signal from data up to (and including) day t-1
        window_prev = sect.iloc[i-1-LOOKBACK : i-1]
        corr_prev = window_prev.corr().values
        fv_prev = fiedler_value(corr_prev, CORR_THR)
        fv_history.append(fv_prev)

        # Decide regime
        if variant == "V1":   # baseline mean-rev (no regime filter)
            mode = "mean_rev"
        elif variant == "V2":  # Fiedler-conditioned hybrid
            mode = "mean_rev" if fv_prev < fv_train_q25 else "momentum"
        elif variant == "V3":  # Adaptive Z-score
            recent = np.array(fv_history[-LOOKBACK:])
            if len(recent) < LOOKBACK:
                mode = "flat"
            else:
                z = (fv_prev - recent.mean()) / (recent.std() + 1e-9)
                if z > 1.0:    mode = "mean_rev"
                elif z < -1.0: mode = "momentum"
                else:          mode = "flat"
        else:
            raise ValueError(variant)

        pos = select_positions(window_prev, tickers, mode)
        day_ret = sect.iloc[i].values            # day t actual returns
        turnover = np.abs(pos - prev_pos).sum()
        tc = turnover * TC_BPS / 10_000
        ret = float(pos @ day_ret) - tc
        prev_pos = pos
        rows.append({"date": dates[i], "ret": ret, "regime": mode})

    return pd.DataFrame(rows).set_index("date")

def calibrate_fiedler(returns: pd.DataFrame, tickers: list):
    sect = returns[tickers].dropna()
    fvs = []
    for i in range(LOOKBACK, len(sect)):
        if sect.index[i] > pd.Timestamp(TRAIN_END):
            break
        window = sect.iloc[i-LOOKBACK:i]
        fvs.append(fiedler_value(window.corr().values, CORR_THR))
    fvs = np.array(fvs)
    return (np.percentile(fvs, 25), np.percentile(fvs, 75),
            fvs.mean(), fvs.std())

def perf_metrics(daily_ret: pd.Series, label: str) -> dict:
    test = daily_ret.loc[TEST_START:]
    ann = test.mean() * 252
    vol = test.std() * np.sqrt(252)
    sr  = ann / vol if vol > 0 else np.nan
    cum = (1 + test).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return {"variant": label,
            "ann_ret": ann, "ann_vol": vol, "sharpe": sr,
            "max_dd": dd.min(),
            "hit_rate": (test > 0).mean(),
            "n_obs": len(test)}

def main():
    print(f"[1/3] Downloading data …")
    returns = download()
    print(f"      {returns.shape[0]} days × {returns.shape[1]} tickers")

    all_variants = {"V1": [], "V2": [], "V3": []}
    print(f"[2/3] Running variants per sector (1-day lag, strict) …")
    for sec_name, tickers in SECTORS.items():
        avail = [t for t in tickers if t in returns.columns]
        if len(avail) < 4:
            continue
        q25, q75, mu, sd = calibrate_fiedler(returns, avail)
        for v in ["V1", "V2", "V3"]:
            df = run_variant(returns, avail, v, q25, q75, mu, sd)
            all_variants[v].append(df["ret"].rename(sec_name))
        print(f"      {sec_name:<12} fv25={q25:.3f}  fv75={q75:.3f}  "
              f"fv_mu={mu:.3f}")

    combined = {}
    for v in ["V1", "V2", "V3"]:
        stacked = pd.concat(all_variants[v], axis=1).dropna()
        combined[v] = stacked.mean(axis=1)
    combined["V4"] = pd.concat([combined[k] for k in ("V1","V2","V3")],
                               axis=1).mean(axis=1)

    print(f"[3/3] Writing outputs …")
    metrics = [
        perf_metrics(combined["V1"], "V1: Mean-Reversion (baseline)"),
        perf_metrics(combined["V2"], "V2: Momentum+TDA Hybrid"),
        perf_metrics(combined["V3"], "V3: Adaptive Threshold (Z)"),
        perf_metrics(combined["V4"], "V4: Ensemble (V1+V2+V3)"),
    ]
    res = pd.DataFrame(metrics).set_index("variant").round(4)
    res.to_csv("strategy_variants_real.csv")

    lines = []
    lines.append("=" * 80)
    lines.append("  STRATEGY VARIANTS — REAL DATA, 1-DAY LAG, OUT-OF-SAMPLE 2022–2024")
    lines.append("=" * 80)
    lines.append(f"  Sectors: {', '.join(SECTORS)}  ({sum(len(v) for v in SECTORS.values())} stocks)")
    lines.append(f"  Train:   pre-{TRAIN_END}    Test: {TEST_START} – {END}")
    lines.append(f"  TC:      {TC_BPS} bps      Lookback: {LOOKBACK}d   Mom: {MOM_WIN}d")
    lines.append("")
    lines.append(f"{'Variant':<35}{'AnnRet':>10}{'AnnVol':>9}{'Sharpe':>9}{'MaxDD':>9}{'Hit':>7}")
    lines.append("-" * 80)
    for m in metrics:
        lines.append(f"{m['variant']:<35}{m['ann_ret']:>9.1%}{m['ann_vol']:>9.1%}"
                     f"{m['sharpe']:>9.2f}{m['max_dd']:>9.1%}{m['hit_rate']:>7.1%}")
    lines.append("=" * 80)
    text = "\n".join(lines)
    print(text)
    Path("strategy_variants_real.txt").write_text(text)

if __name__ == "__main__":
    main()
