"""Action plan view: best trade per stock and the profit-target trade-off chart."""
import streamlit as st

from .. import charts
from ..components import trade_table
from ..format import range_text, stop_text
from ..state import Page


def render(page: Page):
    s = page.controls.plan
    a = page.analysis
    st.markdown(f"Best trade per stock with a profit target between **{range_text(s)}**, {stop_text(s)} and up "
                f"to **{s.horizon_days}** trading days. Ranked by expected return; only signals above "
                f"{s.min_confidence:.0%} confidence are actionable.")
    if a.plan.empty:
        st.warning("No quotes available yet.")
    else:
        trade_table(a.plan, key="tbl_plan", extra=["signal", "daily_vol"])

        st.subheader("Target trade-off")
        ticker = st.selectbox("Stock", a.plan["ticker"].tolist(), key="tradeoff_ticker")
        side = -1 if a.signal(ticker).direction == "DOWN" else 1
        grid = a.candidates[(a.candidates["ticker"] == ticker) & (a.candidates["side"] == side)]
        st.plotly_chart(charts.target_tradeoff(grid.sort_values("target_pct"), ticker, side), width="stretch")
        st.caption("Read it left to right: tiny targets are hit most often, big targets pay more but rarely hit. "
                   "The expected-return line shows whether any target in the range is worth taking at all.")
    st.caption("Shorting and options carry extra risk (unlimited loss on shorts, time decay on puts). "
               "This is a statistical research tool, not investment advice.")
