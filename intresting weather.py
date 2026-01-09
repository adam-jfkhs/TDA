import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------------------
# Earnings from yfinance
# ----------------------------
def get_earnings_dates_yf(ticker: str) -> pd.DatetimeIndex:
    tk = yf.Ticker(ticker)
    try:
        cal = tk.get_earnings_dates(limit=80)
        if cal is not None and len(cal) > 0:
            idx = pd.to_datetime(cal.index).tz_localize(None)
            return pd.DatetimeIndex(sorted(idx.unique()))
    except Exception:
        pass
    return pd.DatetimeIndex([])

def any_in_window(dates: pd.DatetimeIndex, earn_dates_by_ticker: dict, lookahead_days: int) -> pd.Series:
    out = pd.Series(False, index=dates)
    for _, ed in earn_dates_by_ticker.items():
        if len(ed) == 0:
            continue
        ed = pd.DatetimeIndex(ed)
        # vector-ish approach: for each date, find next earnings by scanning (ok for our scale)
        for d in dates:
            future = ed[ed >= d]
            if len(future) and (future[0] - d).days <= lookahead_days:
                out.loc[d] = True
    return out

# ----------------------------
# USER SETTINGS (APPAREL ONLY)
# ----------------------------
START = "2015-01-01"
END   = None

# Keep apparel-only; avoid tickers that Yahoo breaks for you
MALL_HEAVY = ["AEO","ANF","URBN"]
RESILIENT  = ["LEVI","RL","COLM","KTB","VFC","PVH","HBI"]

ENTRY_LOOKAHEAD_DAYS = 21
HOLD_DAYS = 5
SHOCK_ENTRY_Z = 2.0
COST_BPS = 2.0

# ----------------------------
# Helpers
# ----------------------------
def fetch_prices_robust(tickers, start, end=None):
    """
    Downloads prices and drops tickers that failed (all-NaN).
    Returns (prices_df, kept_tickers, dropped_tickers)
    """
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False, group_by="column")

    if raw is None or raw.empty:
        return pd.DataFrame(), [], tickers

    # extract price panel
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            px = raw["Adj Close"].copy()
        else:
            px = raw["Close"].copy()
    else:
        px = raw["Adj Close"].to_frame() if "Adj Close" in raw.columns else raw["Close"].to_frame()

    # Identify tickers that are entirely missing
    kept = [c for c in px.columns if not px[c].dropna().empty]
    dropped = [c for c in px.columns if c not in kept]

    px = px[kept].dropna(how="all").ffill().dropna()

    return px, kept, dropped

def returns(px):
    return px.pct_change().dropna()

def apply_costs(signal: pd.Series, cost_bps: float) -> pd.Series:
    turnover = signal.diff().abs().fillna(0.0)
    return (cost_bps / 1e4) * turnover

def sharpe(x):
    x = x.dropna()
    if len(x) < 50 or x.std() == 0:
        return 0.0
    return float(np.sqrt(252) * x.mean() / x.std())

def maxdd(eq):
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1).min())

# ----------------------------
# Main
# ----------------------------
def run():
    universe = sorted(set(MALL_HEAVY + RESILIENT + ["XRT"]))
    px, kept, dropped = fetch_prices_robust(universe, START, END)

    print("Kept tickers:", kept)
    if dropped:
        print("Dropped tickers (no data):", dropped)

    # Must have XRT + at least 1 ticker on each side
    if px.empty or "XRT" not in px.columns:
        print("No usable data. Try different tickers or check network/Yahoo.")
        return

    mall = [t for t in MALL_HEAVY if t in px.columns]
    res  = [t for t in RESILIENT if t in px.columns]

    if len(mall) == 0 or len(res) == 0:
        print("Not enough tickers after dropping failures.")
        print("Mall side:", mall, "Res side:", res)
        return

    rets = returns(px)

    # Shock proxy: XRT 5d vol z-score (plumbing test only)
    xrt_vol = rets["XRT"].rolling(5).std()
    shock = (xrt_vol - xrt_vol.rolling(252).mean()) / xrt_vol.rolling(252).std()
    shock = shock.reindex(rets.index).fillna(0.0)

    # Earnings window: ANY ticker in either basket has earnings soon
    earn_dates = {t: get_earnings_dates_yf(t) for t in (mall + res)}
    earn_window = any_in_window(rets.index, earn_dates, ENTRY_LOOKAHEAD_DAYS)

    shock_event = (shock.abs() >= SHOCK_ENTRY_Z)
    entry = (earn_window & shock_event).astype(int)

    # Hold for HOLD_DAYS after entry
    position = pd.Series(0.0, index=rets.index)
    entry_ix = np.where(entry.values == 1)[0]
    for i in entry_ix:
        j = min(i + HOLD_DAYS, len(position))
        position.iloc[i:j] = 1.0

    # Basket spread return
    mall_ret = rets[mall].mean(axis=1)
    res_ret  = rets[res].mean(axis=1)
    spread_ret = res_ret - mall_ret

    pos_lag = position.shift(1).fillna(0.0)
    costs = apply_costs(position, COST_BPS)
    strat_ret = pos_lag * spread_ret - costs
    equity = (1 + strat_ret.fillna(0.0)).cumprod()

    print("\n=== Results (Apparel Basket Spread) ===")
    print(f"Mall basket: {mall}")
    print(f"Res basket:  {res}")
    print(f"Days in market: {int((position>0).sum())} / {len(position)}")
    print(f"Entries: {int(entry.sum())}")
    print(f"Sharpe: {sharpe(strat_ret):.3f}")
    print(f"MaxDD:  {maxdd(equity):.2%}")
    print(f"Total:  {(equity.iloc[-1]-1):.2%}")

    plt.figure(figsize=(12,4))
    plt.plot(equity.index, equity.values)
    plt.title("Equity | Long RES basket, Short Mall basket | earnings+shock filter")
    plt.xlabel("Date"); plt.ylabel("Equity")
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(shock.index, shock.values)
    plt.axhline(SHOCK_ENTRY_Z); plt.axhline(-SHOCK_ENTRY_Z)
    plt.title("Shock Proxy: XRT 5d vol z-score (plumbing)")
    plt.xlabel("Date"); plt.ylabel("shock")
    plt.show()

    plt.figure(figsize=(12,3))
    plt.plot(position.index, position.values)
    plt.ylim(-0.05, 1.05)
    plt.title("Position")
    plt.show()

if __name__ == "__main__":
    run()