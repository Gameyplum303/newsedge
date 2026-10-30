"""History & model view: out-of-sample accuracy and what the model learned."""
import pandas as pd
import streamlit as st

from engine.evaluation import CONFIDENT
from engine.sentiment import CATEGORY_LABELS
from .. import charts
from ..format import as_percent
from ..state import Page


def _earnings(ev: dict):
    st.subheader("What following the signals would have earned")
    st.caption(f"Every call on the {ev['test_events']:,} news events from {ev['test_period']}, which the model never "
               "saw in training: buy when it said up, short when it said down, at the first price available after "
               "the news (next open, or the close for news during market hours), sold at the following close, "
               f"minus {ev['trading_cost']:.2%} round-trip trading costs.")
    rows = pd.DataFrame(ev["backtest"])
    rows = as_percent(rows[rows["trades"] > 0], ["threshold", "share", "win_rate", "avg_return", "total_return"])
    st.dataframe(rows, hide_index=True, width="stretch", column_config={
        "threshold": st.column_config.NumberColumn("Only trade at confidence ≥", format="%.0f%%"),
        "trades": "Trades",
        "share": st.column_config.NumberColumn("Share of news events", format="%.1f%%"),
        "win_rate": st.column_config.NumberColumn("Profitable trades", format="%.0f%%"),
        "avg_return": st.column_config.NumberColumn("Average per trade", format="%+.3f%%"),
        "total_return": st.column_config.NumberColumn("Sum of all trades", format="%+.1f%%"),
    })
    best = max(ev["backtest"], key=lambda r: r["avg_return"] if r["trades"] >= 30 else -1)
    if best["trades"] >= 30 and best["avg_return"] > 0:
        st.caption(f"Best: trading only calls at ≥{best['threshold']:.0%} confidence averaged "
                   f"{best['avg_return']:+.3%} per trade after costs over {best['trades']:,} trades. Small samples "
                   "and single test periods are noisy; treat a small positive number with caution.")
    else:
        st.warning("No confidence level would have made money after trading costs on these unseen years. "
                   "Headline sentiment alone has not been a profitable trading signal here.", icon="⚠️")


def _out_of_sample(ev: dict):
    if ev.get("backtest"):
        _earnings(ev)
    st.subheader("Out-of-sample check")
    st.caption(f"Trained on the first 70% of history ({ev['train_events']} events), tested on the most recent "
               f"{ev['test_events']} events ({ev['test_period']}) that it never saw.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Direction accuracy", f"{ev['direction_accuracy']:.1%}",
              f"{(ev['direction_accuracy'] - ev['baseline_accuracy']) * 100:+.1f} pts vs always-guess-majority")
    conf_acc = ev["confident_accuracy"]
    c2.metric(f"Accuracy when ≥{CONFIDENT:.0%} confident", f"{conf_acc:.1%}" if pd.notna(conf_acc) else "—",
              f"on {ev['confident_share']:.0%} of events", delta_color="off")
    c3.metric("Impact Brier score", f"{ev['impact_brier']:.3f}",
              f"{ev['impact_brier'] - ev['impact_base_brier']:+.3f} vs base rate", delta_color="inverse")
    c4.metric("Training events", f"{ev['train_events']:,}")
    st.caption("If direction accuracy is not clearly above the baseline, treat the signals as weak. "
               "Headline sentiment alone is a noisy predictor; that's normal and why probabilities are shown.")
    if ev.get("calibration"):
        _calibration(ev)
    if ev.get("close_ranges"):
        _close_ranges(ev["close_ranges"])


def _close_ranges(cr: dict):
    st.subheader("Are the predicted-close ranges honest?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Closes inside the 68% range", f"{cr['inside_68']:.0%}", f"{(cr['inside_68'] - 0.68) * 100:+.0f} pts vs 68%",
              delta_color="off")
    c2.metric("Closes inside the 90% range", f"{cr['inside_90']:.0%}", f"{(cr['inside_90'] - 0.90) * 100:+.0f} pts vs 90%",
              delta_color="off")
    c3.metric("Unseen news events checked", f"{cr['events']:,}")
    st.caption("Range widths were sized on the training years only, then checked on these later events: how often "
               "the real close landed inside each range. Numbers near 68% and 90% mean the ranges on the stock cards "
               "can be taken at face value.")


def _calibration(ev: dict):
    st.subheader("Is the confidence honest?")
    left, right = st.columns(2)
    cal = as_percent(pd.DataFrame(ev["calibration"]), ["claimed", "actual"])
    left.caption("When the model claimed this confidence on unseen events, how often it was actually right. "
                 "Close numbers mean the % shown on the dashboard can be taken at face value.")
    left.dataframe(cal, hide_index=True, width="stretch", column_config={
        "band": "Confidence band",
        "claimed": st.column_config.NumberColumn("Model claimed", format="%.0f%%"),
        "actual": st.column_config.NumberColumn("Actually right", format="%.0f%%"),
        "events": "Events",
    })
    thr = as_percent(pd.DataFrame(ev["by_threshold"]), ["threshold", "share", "accuracy"])
    right.caption("Acting only on stronger calls: higher accuracy, but far fewer chances. Set the sidebar's "
                  "minimum confidence to match the trade-off you want.")
    right.dataframe(thr, hide_index=True, width="stretch", column_config={
        "threshold": st.column_config.NumberColumn("Only act at ≥", format="%.0f%%"),
        "share": st.column_config.NumberColumn("Share of news days", format="%.1f%%"),
        "events": "Events",
        "accuracy": st.column_config.NumberColumn("Accuracy", format="%.0f%%"),
    })


def _category_table(cs: pd.DataFrame):
    st.subheader("By event type")
    if cs.empty:
        return
    t = cs.reset_index().rename(columns={"category": "Event"})
    t["Event"] = t["Event"].map(CATEGORY_LABELS)
    t = as_percent(t[["Event", "n", "mean_abnormal", "sentiment_hit_rate", "impact_rate"]],
                   ["mean_abnormal", "sentiment_hit_rate", "impact_rate"])
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "n": "Events",
        "mean_abnormal": st.column_config.NumberColumn("Avg abnormal move", format="%+.2f%%"),
        "sentiment_hit_rate": st.column_config.NumberColumn("Followed sentiment", format="%.0f%%"),
        "impact_rate": st.column_config.NumberColumn("Big-move rate", format="%.0f%%"),
    })


def _ticker_table(ts: pd.DataFrame):
    st.subheader("By ticker")
    if ts.empty:
        return
    t = ts.reset_index().rename(columns={"index": "Ticker"})
    t = as_percent(t[["Ticker", "events", "hit_rate", "news_day_abs_move", "normal_day_abs_move", "daily_vol"]],
                   ["hit_rate", "news_day_abs_move", "normal_day_abs_move", "daily_vol"])
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "events": "News days",
        "hit_rate": st.column_config.NumberColumn("Followed sentiment", format="%.0f%%"),
        "news_day_abs_move": st.column_config.NumberColumn("Avg |move| news day", format="%.2f%%"),
        "normal_day_abs_move": st.column_config.NumberColumn("Avg |move| any day", format="%.2f%%"),
        "daily_vol": st.column_config.NumberColumn("Daily volatility", format="%.2f%%"),
    })


def render(page: Page):
    m = page.analysis.model
    st.markdown(f"The model has learned from **{m.n_events:,} news events** (each a tradeable entry after news) across "
                f"**{len(page.analysis.articles):,} stored articles**. Status: "
                + ("**calibrated on your data** ✅" if m.trained else "**default weights** – build more history"))
    if m.evaluation:
        _out_of_sample(m.evaluation)
    else:
        st.info("Not enough history yet for an out-of-sample test. Build history in the sidebar.")
    if not m.events.empty:
        st.plotly_chart(charts.news_events(m.events), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        _category_table(m.category_stats)
    with c2:
        _ticker_table(m.ticker_stats)
