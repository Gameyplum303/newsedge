"""Market data: daily history, intraday bars and live quotes (Yahoo Finance, optional Finnhub).

Quote accuracy rules:
  * `price` is the most recent trade from any source, including pre-market and after-hours.
  * `time` is when that trade happened (never the time we asked), so staleness is visible.
  * `prev_close` is the official, unadjusted close of the previous trading day.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

from . import config, market_clock

LIVE_WITHIN_S = 90   # a quote whose last trade is this recent counts as live


@dataclass(frozen=True)
class Quote:
    price: float
    prev_close: float | None
    regular_close: float | None   # today's 4 PM close, once the session is over
    time: datetime                # UTC time of the last trade
    source: str

    @property
    def change_pct(self) -> float:
        return self.price / self.prev_close - 1 if self.prev_close else 0.0

    @property
    def session(self) -> str:
        return market_clock.session_of(self.time)

    def age_s(self, now: datetime | None = None) -> float:
        return max(0.0, ((now or datetime.now(timezone.utc)) - self.time).total_seconds())


# ---------------------------------------------------------------- yfinance helpers
def _column(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """One field (e.g. "Close") from a yf.download result as a ticker-column frame."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        return raw[field]
    return raw[[field]].rename(columns={field: tickers[0]})


def _naive_dates(index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    return (idx.tz_localize(None) if idx.tz is not None else idx).normalize()


# ---------------------------------------------------------------- history
def daily_prices(tickers: list[str], period: str = config.PRICE_HISTORY_PERIOD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted daily (closes, opens), one column per ticker plus the benchmark, on the same dates."""
    tickers = sorted(set(tickers) | {config.MARKET_BENCHMARK})
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False, threads=True)
    closes = _column(raw, "Close", tickers).dropna(how="all")
    if closes.empty:
        return closes, closes
    opens = _column(raw, "Open", tickers).reindex(closes.index)
    closes.index = opens.index = _naive_dates(closes.index)
    return closes, opens


def intraday(ticker: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """Unadjusted bars incl. extended hours, so chart prices match what the exchange printed."""
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, prepost=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    idx = pd.to_datetime(raw.index)
    raw.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return raw


# ---------------------------------------------------------------- live quotes
def official_closes(tickers: list[str], now: datetime | None = None) -> dict[str, dict]:
    """Per ticker: the previous trading day's official close and, once the session ends, today's."""
    now = (now or datetime.now(market_clock.ET)).astimezone(market_clock.ET)
    raw = yf.download(tickers, period="10d", interval="1d", auto_adjust=False, progress=False, threads=True)
    closes = _column(raw, "Close", tickers)
    if closes.empty:
        return {}
    closes.index = _naive_dates(closes.index)
    td = pd.Timestamp(market_clock.current_trading_date(now))
    session_over = now >= market_clock.session_bounds(td.date())[1]
    out = {}
    for t in closes.columns:
        s = closes[t].dropna()
        before = s[s.index < td]
        if not before.empty:
            out[t] = {"prev_close": float(before.iloc[-1]),
                      "regular_close": float(s.loc[td]) if session_over and td in s.index else None}
    return out


def yahoo_last_trades(tickers: list[str]) -> dict[str, tuple[float, datetime]]:
    """Latest 1-minute bar per ticker, extended hours included (Yahoo labels it Nasdaq real-time)."""
    raw = yf.download(tickers, period="1d", interval="1m", prepost=True, auto_adjust=False,
                      progress=False, threads=True)
    closes = _column(raw, "Close", tickers)
    out = {}
    for t in closes.columns:
        s = closes[t].dropna()
        if not s.empty:
            ts = pd.Timestamp(s.index[-1])
            ts = ts.tz_localize("UTC") if ts.tz is None else ts
            out[t] = (float(s.iloc[-1]), ts.to_pydatetime().astimezone(timezone.utc))
    return out


def _finnhub_rest(ticker: str, api_key: str) -> tuple[float, datetime, float] | None:
    r = requests.get("https://finnhub.io/api/v1/quote", params={"symbol": ticker, "token": api_key}, timeout=8)
    r.raise_for_status()
    q = r.json()
    if not q or not q.get("c") or not q.get("t"):
        return None
    return float(q["c"]), datetime.fromtimestamp(q["t"], timezone.utc), float(q.get("pc") or 0)


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def quotes(tickers: list[str], *, finnhub_key: str = "", stream=None,
           closes_ref: dict | None = None, yahoo: dict | None = None) -> dict[str, Quote]:
    """Most accurate available quote per ticker: the newest trade across all sources wins.

    `stream` is an optional live FinnhubStream; `closes_ref` and `yahoo` let callers pass cached
    results of official_closes() and yahoo_last_trades().
    """
    tickers = list(tickers)
    candidates: dict[str, list[tuple[float, datetime, str]]] = {t: [] for t in tickers}

    if stream is not None:
        stream.subscribe(tickers)
        for t in tickers:
            hit = stream.latest(t)
            if hit:
                candidates[t].append((hit[0], datetime.fromtimestamp(hit[1], timezone.utc), "Finnhub live stream"))

    yahoo = yahoo if yahoo is not None else (_safe(yahoo_last_trades, tickers) or {})
    for t, (p, ts) in yahoo.items():
        if t in candidates:
            candidates[t].append((p, ts, "Yahoo Finance (Nasdaq real-time)"))

    finnhub_prev = {}
    if finnhub_key:
        # With a stream, REST is only a fallback for symbols that have not traded on it yet.
        missing = [t for t in tickers if not candidates[t]] if stream is not None else tickers
        with ThreadPoolExecutor(max_workers=8) as pool:
            for t, res in zip(missing, pool.map(lambda x: _safe(_finnhub_rest, x, finnhub_key), missing)):
                if res:
                    candidates[t].append((res[0], res[1], "Finnhub"))
                    finnhub_prev[t] = res[2]

    ref = closes_ref if closes_ref is not None else (_safe(official_closes, tickers) or {})
    out = {}
    for t, cands in candidates.items():
        if cands:
            price, ts, source = max(cands, key=lambda c: c[1])
            r = ref.get(t, {})
            out[t] = Quote(price=price, prev_close=r.get("prev_close") or finnhub_prev.get(t),
                           regular_close=r.get("regular_close"), time=ts, source=source)
    return out


def freshness(q: Quote, market_state: str, now: datetime | None = None) -> tuple[str, str]:
    """(short status label, detailed tooltip) describing how current a quote is."""
    now = now or datetime.now(timezone.utc)
    t_et = q.time.astimezone(market_clock.ET)
    same_day = t_et.date() == now.astimezone(market_clock.ET).date()
    when = f"{t_et:%I:%M:%S %p} ET" if same_day else f"{t_et:%b %d %I:%M %p} ET"

    tip = f"Last trade {when} · {q.session} · {q.source}"
    if q.session == "After-hours" and q.regular_close:
        tip += f" · 4:00 PM close ${q.regular_close:,.2f}"
    if q.prev_close:
        tip += f" · previous close ${q.prev_close:,.2f}"

    if market_state == "closed":
        return f"⚪ Closed · last trade {when}", tip
    age = q.age_s(now)
    if age <= LIVE_WITHIN_S:
        return f"🟢 Live · {q.session.lower()}", tip
    mins = int(age // 60)
    return (f"🟡 Last trade {mins}m ago" if mins < 60 else f"🔴 No trade for {mins // 60}h"), tip
