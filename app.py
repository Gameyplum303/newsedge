"""NewsEdge dashboard. Run with:  streamlit run app.py"""
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(page_title="NewsEdge", page_icon="📈", layout="wide")

from engine import config, market_clock  # noqa: E402  (Streamlit needs set_page_config first)
from engine.pipeline import analyse  # noqa: E402
from ui import components, data, sidebar  # noqa: E402
from ui.state import Page  # noqa: E402
from ui.views import action_plan, companies, history, news_feed, signals, trade_ideas  # noqa: E402

VIEWS = {"Signals": signals, "Action plan": action_plan, "News feed": news_feed, "History & model": history,
         "Companies": companies}

controls = sidebar.render()
tickers = tuple(controls.tickers)

# ---------------------------------------------------------------- header, clock, live prices
st.title("📈 NewsEdge")
st.caption("News-driven signals for US stocks · probabilities, not promises · not financial advice")
with st.container(border=True):
    components.market_clock_bar()
with st.container(border=True):
    components.live_prices(tickers, controls.finnhub_key)
components.mark_full_run()
components.auto_refresher(controls.auto_refresh, controls.refresh_secs)

# ---------------------------------------------------------------- analysis
report, new_articles, fetched_at = data.refresh_news(controls.watch_items)
articles = data.load_articles(controls.watch_items, fetched_at)
closes, opens = data.load_prices(tickers)
if closes.empty:
    st.error("Could not download price history from Yahoo Finance. Check your internet connection and refresh.")
    st.stop()

now = datetime.now(timezone.utc)
model = data.train_model(articles, closes, opens, market_clock.last_complete_session(closes.index))
quotes = data.load_quotes(tickers, controls.finnhub_key)
analysis = analyse(articles, closes, controls.watchlist, {t: q.price for t, q in quotes.items()},
                   controls.plan, now, model=model)
page = Page(controls, analysis, quotes, report, fetched_at, new_articles, now)

if model.n_events < config.MIN_EVENTS_TO_TRAIN:
    st.info(f"**First run?** The model has only {model.n_events} historical news events to learn from, so it is "
            "using default weights. Click **📚 Build history** in the sidebar to back-fill past headlines and "
            "calibrate the confidence scores on how these stocks actually reacted.", icon="💡")

# ---------------------------------------------------------------- page body
trade_ideas.render(page)
view = st.radio("View", list(VIEWS), horizontal=True, label_visibility="collapsed", key="view")
VIEWS[view].render(page)

st.divider()
st.caption("NewsEdge is an educational research tool. Signals are statistical estimates from public headlines and "
           "past price reactions; they can be wrong, markets can gap through stops, and past relationships change. "
           "Nothing here is investment advice.")
