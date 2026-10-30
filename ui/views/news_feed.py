"""News feed view: recent articles with filters, plus per-feed health."""
import pandas as pd
import streamlit as st

from engine import linking, market_clock
from engine.sentiment import WORLD_EVENT_CATEGORIES
from ..components import article_table
from ..state import Page

WINDOW_HOURS = 72
LINK_FILTERS = {"Any": None, "Direct": {linking.DIRECT, linking.FEED}, "Government": {linking.GOVERNMENT},
                "Related companies": {linking.RELATION}, "World exposure": {linking.EXPOSURE}}


def render(page: Page):
    articles = page.analysis.articles
    tickers = page.controls.tickers
    recent = articles[articles["published"] >= page.now - pd.Timedelta(hours=WINDOW_HOURS)]
    c1, c2 = st.columns([2, 1])
    picked = c1.multiselect("Filter tickers", tickers, default=tickers, key="feed_tickers")
    min_abs = c2.slider("Min |sentiment|", 0.0, 1.0, 0.0, 0.05)
    c3, c4 = st.columns(2)
    kind = c3.radio("News type", ["All", "Business", "World events"], horizontal=True, key="feed_kind",
                    help="World events: disasters, accidents, outages and cyberattacks, conflict, strikes, outbreaks.")
    link = c4.radio("Linked to the stock", list(LINK_FILTERS), horizontal=True, key="feed_link",
                    help="Direct: names the company or its products. Indirect: government business, related "
                         "companies (suppliers, customers, partners, competitors) or world-event exposure.")
    recent = recent[recent["ticker"].isin(picked) & (recent["sentiment"].abs() >= min_abs)]
    if kind != "All":
        is_world = recent["category"].isin(WORLD_EVENT_CATEGORIES)
        recent = recent[is_world if kind == "World events" else ~is_world]
    if LINK_FILTERS[link] and "link_kind" in recent:
        recent = recent[recent["link_kind"].isin(LINK_FILTERS[link])]
    fetched = pd.Timestamp(page.fetched_at).tz_convert(market_clock.ET)
    st.caption(f"{len(recent):,} articles in the last {WINDOW_HOURS}h · last collected {fetched:%I:%M:%S %p} ET · "
               f"{page.new_articles} new on that pass")
    article_table(recent, show_ticker=True)
    with st.expander("Feed health"):
        st.dataframe(pd.DataFrame([{"Feed": k, "Status": "ok" if v is not None else "unreachable",
                                    "Articles matched": v or 0} for k, v in sorted(page.feed_report.items())]),
                     hide_index=True, width="stretch")
