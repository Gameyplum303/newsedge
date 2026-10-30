"""Reusable dashboard widgets: market clock, live prices, auto-refresh and tables."""
import time

import pandas as pd
import streamlit as st

from engine import forecast, jobs, market_clock, prices, store
from engine.prices import Quote
from engine.sanitize import safe_link
from engine.sentiment import CATEGORY_LABELS
from . import data
from .format import as_percent, esc
from .theme import MARKET_STATE_COLORS

LIVE_PRICES_PER_ROW = 5


# ---------------------------------------------------------------- live widgets (Streamlit fragments)
@st.fragment(run_every=1)
def market_clock_bar():
    s = market_clock.status()
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    extra = f" · {s['holiday']}" if s["holiday"] else (" · early close 1:00 PM" if s["early_close"] else "")
    c1.markdown(
        f"<div style='font-size:0.85rem;opacity:.7'>US market (NYSE / Nasdaq)</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{MARKET_STATE_COLORS[s['state']]}'>"
        f"● {s['state'].upper()}</div>"
        f"<div style='font-size:0.85rem;opacity:.7'>{s['now']:%a %b %d · %I:%M:%S %p} ET{extra}</div>",
        unsafe_allow_html=True)
    is_open = s["state"] == "open"
    c2.metric("Time until close" if is_open else "Next session closes in",
              market_clock.format_duration(s["seconds_to_close"]))
    c3.metric("Next open in" if is_open else "Time until open", market_clock.format_duration(s["seconds_to_open"]))
    c4.metric("Next open", f"{s['next_open']:%a %b %d, %I:%M %p} ET")


def _price_metric(container, label: str, q: Quote | None, market_state: str):
    if not q:
        container.metric(label, "—")
        container.caption("no quote")
        return
    status, tooltip = prices.freshness(q, market_state)
    container.metric(label, f"${q.price:,.2f}", f"{q.change_pct:+.2%}", help=tooltip)
    container.caption(status)


@st.fragment(run_every=5)
def live_prices(tickers: tuple, finnhub_key: str):
    quotes = data.load_quotes(tickers, finnhub_key)
    state = market_clock.status()["state"]
    for i in range(0, len(tickers), LIVE_PRICES_PER_ROW):
        row = tickers[i:i + LIVE_PRICES_PER_ROW]
        for col, t in zip(st.columns(LIVE_PRICES_PER_ROW), row):
            _price_metric(col, t, quotes.get(t), state)


@st.fragment(run_every=5)
def card_price(ticker: str, tickers: tuple, finnhub_key: str):
    q = data.load_quotes(tickers, finnhub_key).get(ticker)
    _price_metric(st, "Price", q, market_clock.status()["state"])


@st.fragment(run_every=5)
def close_forecast(ticker: str, tickers: tuple, finnhub_key: str, daily_vol: float, p_impact: float,
                   expected_move: float, scales: dict):
    """Predicted close and its ranges, recomputed from the live price and the time left."""
    q = data.load_quotes(tickers, finnhub_key).get(ticker)
    if not q:
        st.caption("Predicted close: no live price yet.")
        return
    f = forecast.forecast_close(q.price, daily_vol, p_impact, expected_move, scales)
    c1, c2, c3 = st.columns([1, 1.6, 1])
    c1.metric("Predicted close", f"${f.price:,.2f}",
              help="Most likely closing price. A single price can't carry a confidence, so the ranges show it.")
    c1.caption(f.label[0].upper() + f.label[1:])
    c2.markdown(esc(f"**68% range:** ${f.low68:,.2f} – ${f.high68:,.2f}  \n"
                    f"**90% range:** ${f.low90:,.2f} – ${f.high90:,.2f}"))
    c3.metric("Chance it closes above now", f"{f.p_above_now:.0%}")
    if f.news_shifts_price:
        st.caption("Outside market hours: the news direction shifts the next close; more news widens the range.")
    else:
        st.caption("During the session the forecast stays centred on the current price: in 10 years of history, "
                   "news was only tradeable from the next session. More news widens the range; it narrows as the "
                   "close approaches.")


@st.fragment(run_every=5)
def auto_refresher(enabled: bool, every: int):
    """Reruns the whole page every `every` seconds (the news cache refreshes itself every 5 minutes)."""
    last = st.session_state.setdefault("last_full_run", time.time())
    if enabled and time.time() - last >= every:
        data.load_yahoo_trades.clear()
        st.rerun(scope="app")


def mark_full_run():
    st.session_state["last_full_run"] = time.time()


_FINISHED = ("done", "stopped", "failed", "cancelled")
_ICONS = {"done": "✅", "stopped": "⏸️", "failed": "❌", "cancelled": "🚫"}


@st.fragment(run_every=5)
def setup_jobs_panel():
    """Progress of background history downloads; retrains the model when one finishes."""
    recent = store.recent_jobs(hours=6)
    seen = st.session_state.setdefault("seen_jobs", {j["id"] for j in recent if j["status"] in _FINISHED})
    if any(j["status"] == "queued" for j in recent):
        jobs.ensure_worker()   # e.g. after a restart, pick queued work back up
    for j in recent:
        if j["status"] in ("queued", "running"):
            st.progress(float(j["progress"] or 0), text=f"**{j['ticker']}** · {j['message']}")
        else:
            st.caption(f"{_ICONS[j['status']]} **{j['ticker']}** · {j['message']}")
            if j["status"] == "stopped" and st.button(f"Resume {j['ticker']}", key=f"resume_{j['id']}"):
                jobs.enqueue(j["ticker"], j["names"], j["kinds"], j["years"])
    newly_finished = [j for j in recent if j["status"] in _FINISHED and j["id"] not in seen]
    if newly_finished:
        seen.update(j["id"] for j in newly_finished)
        data.load_articles.clear()   # new articles in -> the model retrains on the next run
        st.toast(f"History updated for {', '.join(j['ticker'] for j in newly_finished)}; retraining the model.",
                 icon="🧠")
        st.rerun(scope="app")


# ---------------------------------------------------------------- tables
TRADE_COLUMNS = ["ticker", "action", "confidence", "entry", "target_pct", "target_price", "stop_pct", "stop_price",
                 "p_target", "p_stop", "expected_return", "days_to_target"]
_TRADE_PERCENT = ["confidence", "target_pct", "stop_pct", "p_target", "p_stop", "expected_return", "daily_vol"]


def _trade_column_config() -> dict:
    cc = st.column_config
    return {
        "ticker": "Ticker",
        "action": "Action",
        "signal": "Signal",
        "confidence": cc.ProgressColumn("Direction conf.", format="%.0f%%", min_value=0, max_value=100),
        "entry": cc.NumberColumn("Entry", format="$%.2f"),
        "target_pct": cc.NumberColumn("Profit target", format="+%.2f%%"),
        "target_price": cc.NumberColumn("Target price", format="$%.2f"),
        "stop_pct": cc.NumberColumn("Stop-loss", format="−%.2f%%"),
        "stop_price": cc.NumberColumn("Stop price", format="$%.2f"),
        "p_target": cc.ProgressColumn("Win probability", format="%.0f%%", min_value=0, max_value=100,
                                      help="Chance the target is reached before the stop-loss"),
        "p_stop": cc.NumberColumn("P(stopped out)", format="%.0f%%"),
        "expected_return": cc.NumberColumn("Expected return", format="%+.2f%%",
                                           help="Average result across all simulated outcomes"),
        "days_to_target": cc.NumberColumn("Days to target", format="%.0f"),
        "daily_vol": cc.NumberColumn("Daily vol", format="%.2f%%"),
    }


def trade_table(trades: pd.DataFrame, key: str, extra: list[str] | None = None):
    show = as_percent(trades[TRADE_COLUMNS + (extra or [])], _TRADE_PERCENT)
    st.dataframe(show, hide_index=True, width="stretch", key=key, column_config=_trade_column_config())


def top_pick(trade: pd.Series):
    verb = "BUY" if trade["side"] == 1 else "SELL SHORT"
    st.markdown(esc(
        f"**Top pick: {verb} {trade['ticker']}** at ~${trade['entry']:,.2f} → target **${trade['target_price']:,.2f}** "
        f"(+{trade['target_pct']:.2%}) · stop ${trade['stop_price']:,.2f} (−{trade['stop_pct']:.2%})"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win probability", f"{trade['p_target']:.0%}", help="Chance of reaching the target before the stop")
    m2.metric("Expected return", f"{trade['expected_return']:+.2%}")
    m3.metric("Direction confidence", f"{trade['confidence']:.0%}")
    days = trade["days_to_target"]
    m4.metric("Typical days to target", f"{days:.0f}" if pd.notna(days) else "—")


def article_table(articles: pd.DataFrame, show_ticker: bool = False):
    df = pd.DataFrame({
        "Ticker": articles["ticker"],
        "Published (ET)": articles["published"].dt.tz_convert(market_clock.ET).dt.strftime("%b %d %I:%M %p"),
        "Headline": articles["title"],
        "Source": articles["source"],
        "Event": articles["category"].map(CATEGORY_LABELS) + ("" if "via" not in articles else
                  articles["via"].map(lambda v: f" · via {v}" if v else "")),
        "Sentiment": articles["sentiment"],
        "Link": articles["link"].fillna("").map(safe_link),   # older stored rows predate the ingest check
    })
    if "influence" in articles:
        df.insert(5, "Influence", articles["influence"] * 100)
    if not show_ticker:
        df = df.drop(columns="Ticker")
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "Headline": st.column_config.TextColumn(width="large"),
        "Sentiment": st.column_config.NumberColumn(format="%+.2f"),
        "Influence": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Link": st.column_config.LinkColumn(display_text="open ↗"),
    })
