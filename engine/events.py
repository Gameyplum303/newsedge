"""Market data -> news events: every event is a trade you could actually have made.

Each article is matched to the first price available once it was public (the next open for news
before the open, the close for news during market hours). Back-filled Google News items only carry
a date, so they count from the end of that date. Each event records the move from that entry to the
next close: the raw return (what a trade earned) and the abnormal return (minus beta x the market).
"""
import numpy as np
import pandas as pd

from . import config
from .linking import INDIRECT_LINKS
from .market_clock import ET
from .sentiment import dominant_category

# Every event is measured as a trade you could actually have made: from the first price available
# after the news was public to the following close. Back-filled Google News items only carry a date
# (stamped midnight Pacific), so they count as known once that date is over, never earlier.
DATE_ONLY_UTC_HOURS = (7, 8)          # midnight Pacific in summer / winter
REGULAR_OPEN_MIN, REGULAR_CLOSE_MIN = 9 * 60 + 30, 16 * 60


def abnormal_returns(closes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Returns (raw daily returns, abnormal daily returns, trailing abnormal volatility, betas)."""
    rets = closes.pct_change()
    mkt = rets[config.MARKET_BENCHMARK]
    abn, vol, betas = {}, {}, {}
    for t in rets.columns.drop(config.MARKET_BENCHMARK):
        r = rets[t]
        ok = r.notna() & mkt.notna()
        betas[t] = float(np.clip(np.cov(r[ok], mkt[ok])[0, 1] / mkt[ok].var(), 0.0, 3.0)) if ok.sum() > 30 else 1.0
        a = r - betas[t] * mkt
        abn[t] = a
        vol[t] = a.rolling(60, min_periods=20).std().shift(1).fillna(a.std())
    return rets, pd.DataFrame(abn), pd.DataFrame(vol), betas


def available_at(published: pd.Series) -> pd.Series:
    """When an article was certainly public: its timestamp, or the end of its date for date-only items."""
    date_only = (published.dt.minute == 0) & (published.dt.second == 0) & published.dt.hour.isin(DATE_ONLY_UTC_HOURS)
    return published + pd.to_timedelta(date_only.astype(int) * 24, unit="h")


def entry_points(available: pd.Series, trading_days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """For each article: index of the session you could first trade in, and whether at its open.

    Before the open -> that day's open. During the session -> its close (only daily prices exist for
    history, so the close is the first price we can be sure of). After the close or on a non-trading
    day -> the next session's open. Index len(trading_days) means the session hasn't happened yet.
    """
    et = available.dt.tz_convert(ET)
    day = et.dt.tz_localize(None).dt.normalize()
    minutes = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    first = np.searchsorted(trading_days.values, day.values)
    is_session = (first < len(trading_days)) & (trading_days.values[np.minimum(first, len(trading_days) - 1)] == day.values)
    next_day = np.searchsorted(trading_days.values, (day + pd.Timedelta(days=1)).values)
    during = is_session & (minutes >= REGULAR_OPEN_MIN) & (minutes < REGULAR_CLOSE_MIN)
    after = is_session & (minutes >= REGULAR_CLOSE_MIN)
    return np.where(after, next_day, first), ~during


def build_events(articles: pd.DataFrame, closes: pd.DataFrame, last_complete: pd.Timestamp,
                 opens: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (ticker, entry) with aggregated news features and the tradeable move that followed.

    `opens` are daily opening prices; without them the previous close stands in for the open.
    """
    if articles.empty or closes.empty:
        return pd.DataFrame()
    days = closes.index
    opens = closes.shift(1) if opens is None else opens.reindex(index=days, columns=closes.columns)
    rets, abn, vol, betas = abnormal_returns(closes)
    # The stock's normal daily swing before the news (whole-history value where there's too little history).
    daily_vol = rets.rolling(60, min_periods=20).std().shift(1).fillna(rets.std())

    a = articles[articles["ticker"].isin(abn.columns)]
    entry_idx, at_open = entry_points(available_at(a["published"]), days)
    exit_idx = entry_idx + (~at_open).astype(int)
    ok = (entry_idx > 0) & (exit_idx < len(days))
    ok[ok] &= days[exit_idx[ok]] <= last_complete      # only trades that have closed
    a = a[ok].assign(entry_idx=entry_idx[ok], at_open=at_open[ok])
    if a.empty:
        return pd.DataFrame()
    a["weight"] = a["relevance"] * a["source_weight"]
    a["weighted_sentiment"] = a["weight"] * a["sentiment"]
    a["indirect_weight"] = a["weight"] * a["link_kind"].isin(INDIRECT_LINKS) if "link_kind" in a else 0.0

    g = a.groupby(["ticker", "entry_idx", "at_open"])
    ev = pd.DataFrame({
        "sentiment": g["weighted_sentiment"].sum() / g["weight"].sum().clip(lower=1e-9),
        "n": g.size(),
        "indirect_share": g["indirect_weight"].sum() / g["weight"].sum().clip(lower=1e-9),
        "category": g.apply(dominant_category, include_groups=False),
    }).reset_index()

    i, at_open = ev["entry_idx"].to_numpy(), ev["at_open"].to_numpy()
    out_i = i + (~at_open).astype(int)
    col = {t: k for k, t in enumerate(closes.columns)}
    c, o = closes.to_numpy(), opens.to_numpy()
    tk = ev["ticker"].map(col).to_numpy()
    spy = col[config.MARKET_BENCHMARK]
    entry_px = np.where(at_open, o[i, tk], c[i, tk])
    entry_mkt = np.where(at_open, o[i, spy], c[i, spy])
    ev["ret"] = c[out_i, tk] / entry_px - 1
    market = c[out_i, spy] / entry_mkt - 1
    ev["abnormal"] = ev["ret"] - ev["ticker"].map(betas).to_numpy() * market
    ev["sigma"] = [vol.iat[k, vol.columns.get_loc(t)] for k, t in zip(i, ev["ticker"])]
    ev["daily_vol"] = [daily_vol.iat[k, daily_vol.columns.get_loc(t)] for k, t in zip(i, ev["ticker"])]
    ev["date"] = days[out_i]                       # the session the trade was closed in
    ev["entry"] = np.where(at_open, "open", "close")
    ev = ev.drop(columns=["entry_idx", "at_open"]).dropna(subset=["ret", "abnormal", "sigma", "daily_vol"])
    ev = ev[ev["sigma"] > 0]
    ev["z"] = ev["abnormal"] / ev["sigma"]
    ev["impact"] = (ev["z"].abs() > config.IMPACT_Z).astype(int)
    return ev.sort_values("date").reset_index(drop=True)
