"""Top-of-page trade ideas: highest-confidence and highest-return trades."""
import streamlit as st

from ..components import top_pick, trade_table
from ..format import range_text, stop_text
from ..state import Page


def _wait(page: Page):
    s = page.controls.plan
    plan = page.analysis.plan
    with st.container(border=True):
        st.markdown(f"**WAIT.** No stock currently has a news signal strong enough to clear your "
                    f"{s.min_confidence:.0%} confidence bar with a positive expected return at any target in "
                    f"{range_text(s)}. Doing nothing is a valid position.")
        directional = plan[plan["signal"] != "NEUTRAL"] if not plan.empty else plan
        if directional.empty:
            return
        c = directional.sort_values("confidence", ascending=False).iloc[0]
        why = (f"confidence {c['confidence']:.0%} is below your {s.min_confidence:.0%} bar"
               if c["confidence"] < s.min_confidence else
               f"its best expected return is only {c['expected_return']:+.2%} once stop-loss gaps and "
               f"normal volatility are accounted for")
        st.caption(f"Closest candidate: **{c['ticker']}** ({'▲' if c['signal'] == 'UP' else '▼'} "
                   f"{c['signal'].lower()}), but {why}.")


def _section(title: str, caption: str, trades, key: str):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption(caption)
        top_pick(trades.iloc[0])
        trade_table(trades, key=key)


def render(page: Page):
    s = page.controls.plan
    a = page.analysis
    st.subheader("🎯 Trade ideas")
    st.caption(f"Every stock is tested at each profit target from **{range_text(s)}** with {stop_text(s)} over up to "
               f"{s.horizon_days} trading days ({s.simulations:,} simulated 30-minute price paths each). Only trades "
               f"in the direction of a news signal with ≥{s.min_confidence:.0%} confidence and a positive expected "
               f"return are listed.")
    if a.top_confidence.empty:
        _wait(page)
        return
    _section("🛡️ Highest confidence trades",
             "For each stock, the profit target most likely to be reached. Smaller targets win more often "
             "but earn less per trade.", a.top_confidence, "tbl_conf")
    _section("🚀 Highest return trades",
             "For each stock, the profit target with the largest expected return. Bigger targets pay more "
             "when they hit but hit less often.", a.top_return, "tbl_ret")
    st.caption("Simulated in 30-minute steps: targets fill at the limit price, stops fill where the price "
               "actually is, so gaps past a stop cost extra. Very small targets rarely pay once that slippage is "
               "counted. Commissions and spreads are not included. Not investment advice.")
