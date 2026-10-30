"""Signals view: summary table plus one card per stock with reasoning, articles and a chart."""
import numpy as np
import pandas as pd
import streamlit as st

from engine import config, forecast, market_clock, store
from engine.sentiment import CATEGORY_LABELS
from .. import charts, data
from ..components import article_table, card_price, close_forecast
from ..format import NO_PREDICTION, PREDICTION_LABELS, direction_md, esc
from ..state import Page


def _summary_table(page: Page):
    rows = []
    scales = page.analysis.model.range_scale
    for s in page.analysis.signals:
        q = page.quotes.get(s.ticker)
        f = forecast.forecast_close(q.price, s.daily_vol, s.p_impact, s.expected_move, scales) if q else None
        rows.append({
            "Ticker": s.ticker,
            "Price": q.price if q else None,
            "Today": q.change_pct * 100 if q else np.nan,
            "Last trade (ET)": q.time.astimezone(market_clock.ET).strftime("%I:%M:%S %p") if q else "—",
            "Predicted close": f.price if f else None,
            "68% range": f"${f.low68:,.2f} – ${f.high68:,.2f}" if f else "—",
            "P(close > now)": f.p_above_now * 100 if f else np.nan,
            "Prediction": PREDICTION_LABELS.get(s.direction, NO_PREDICTION),
            "Confidence": s.confidence * 100,
            "Impact probability": s.p_impact * 100,
            "Expected move": s.expected_move * 100,
            "Articles (24h)": s.n_articles,
            "Event": CATEGORY_LABELS.get(s.category, s.category) if s.n_articles else "—",
        })
    cc = st.column_config
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Price": cc.NumberColumn(format="$%.2f"),
        "Today": cc.NumberColumn(format="%+.2f%%"),
        "Predicted close": cc.NumberColumn(format="$%.2f", help="Most likely close (today's during market "
                                           "hours, otherwise the next session's). Refreshes with the page; the "
                                           "stock cards below update it every 5 seconds."),
        "68% range": cc.TextColumn(help="The close should land in this range about 68% of the time "
                                   "(calibrated on past news events; see the History tab)"),
        "P(close > now)": cc.NumberColumn(format="%.0f%%", help="Chance the close is above the current price"),
        "Confidence": cc.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                        help="Probability the predicted direction is right"),
        "Impact probability": cc.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                                help=f"Probability of an outsized move (>{config.IMPACT_Z}σ vs market)"),
        "Expected move": cc.NumberColumn(format="%+.2f%%", help="Expected move vs the market from your next chance to trade (next open, or the close for news during market hours) to the following close, before trading costs"),
    })


def _card(page: Page, s):
    tickers = tuple(page.controls.tickers)
    with st.container(border=True):
        head, m1, m2, m3, m4 = st.columns([1.3, 1, 1, 1, 1])
        head.markdown(f"## {s.ticker}  \n{direction_md(s.direction)}")
        with m1:
            card_price(s.ticker, tickers, page.controls.finnhub_key)
        m2.metric("Direction confidence", f"{s.confidence:.0%}")
        m3.metric("Impact probability", f"{s.p_impact:.0%}")
        m4.metric("Expected move", f"{s.expected_move:+.2%}")
        close_forecast(s.ticker, tickers, page.controls.finnhub_key, s.daily_vol, s.p_impact, s.expected_move,
                       page.analysis.model.range_scale)
        job = store.open_job(s.ticker)
        if job:
            st.info(f"⏳ {s.ticker}'s news history is still downloading ({float(job['progress'] or 0):.0%}). "
                    "Its confidence is provisional until the model retrains with it.", icon="⏳")
        st.markdown("**Why:**\n" + "\n".join(f"- {esc(r)}" for r in s.reasons))
        if s.n_articles:
            with st.expander(f"Source articles ({s.n_articles})"):
                article_table(s.articles)
        with st.expander("Price chart with news markers (5 days, incl. extended hours)"):
            bars = data.load_intraday(s.ticker)
            if bars.empty:
                st.caption("Intraday data unavailable.")
            else:
                arts = page.analysis.articles
                st.plotly_chart(charts.price_with_news(bars, arts[arts["ticker"] == s.ticker], s.ticker),
                                width="stretch", key=f"chart_{s.ticker}")


def render(page: Page):
    _summary_table(page)
    for s in page.analysis.signals:
        _card(page, s)
