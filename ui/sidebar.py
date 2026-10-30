"""Sidebar: watchlist, data refresh, trade-plan settings and price-source status."""
import streamlit as st

from engine import config, jobs, profiles, store
from engine.sanitize import valid_ticker
from engine.strategy import PlanSettings
from . import components, data
from .state import Controls


def _add_company(ticker: str, company: str = "") -> list[str] | None:
    """Set up a newly picked company; shows an error and returns None if it can't be modeled."""
    with st.spinner(f"Looking up {ticker}…"):
        names = profiles.set_up_company(ticker, company)
    if names is None:
        st.error(f"{ticker} {f'({company}) ' if company else ''}has no price history on Yahoo Finance yet "
                 "(e.g. a very recent listing), so it can't be modeled.")
    return names


def _company_picker(watchlist: dict) -> list[tuple[str, str]]:
    """Search box over every US-listed company. Returns the picked (ticker, company) pairs."""
    directory = data.load_listings()
    if directory.empty:   # offline on first run: fall back to typing a ticker
        typed = st.text_input("Add a company by ticker", placeholder="e.g. SHOP", max_chars=12).strip().upper()
        if typed and not valid_ticker(typed):
            st.error("Enter a ticker symbol such as SHOP or BRK-B.")
            return []
        return [(typed, "")] if typed and typed not in watchlist else []
    directory = directory[~directory["yahoo"].isin(watchlist)]
    labels = dict(zip(directory["yahoo"] + " · " + directory["company"] + " (" + directory["exchange"] + ")",
                      zip(directory["yahoo"], directory["company"])))
    picked = st.multiselect("Add companies", list(labels), key="wl_add",
                            placeholder=f"Search {len(directory):,} US-listed companies…",
                            help="Every company listed on NASDAQ, plus NYSE and the other US exchanges, from NASDAQ's "
                                 "official directory. Its news history downloads in the background once you save "
                                 "(about 10–15 minutes), and the model retrains when it's done.")
    return [labels[p] for p in picked]


def _watchlist_section() -> dict[str, list[str]]:
    watchlist = store.load_watchlist()
    st.header("Watchlist")
    chosen = st.multiselect("Tracking", list(watchlist), default=list(watchlist), key="wl_select",
                            help="Remove a company to stop tracking it; any history download in progress is cancelled.")
    picked = _company_picker(watchlist)
    if st.button("Save watchlist", width="stretch"):
        updated = {t: watchlist[t] for t in chosen}
        for ticker, company in picked:
            names = _add_company(ticker, company)
            if names is None:
                return watchlist
            updated[ticker] = names
        if not updated:
            st.warning("Keep at least one company.")
            return watchlist
        store.save_watchlist(updated)
        queued, cancelled = jobs.sync_watchlist(watchlist, updated)
        if queued:
            st.toast(f"Downloading history for {', '.join(queued)} in the background. The model retrains when "
                     "it's done.", icon="⏳")
        if cancelled:
            st.toast(f"Stopped downloading for {', '.join(cancelled)}.", icon="🚫")
        st.cache_data.clear()
        st.rerun()
    return watchlist


def _plan_section() -> PlanSettings:
    st.header("Trade plan settings")
    lo, hi = st.slider("Profit target range (%)", 0.1, 10.0, (0.1, 10.0), 0.1, format="%.1f%%",
                       help="Every target in this range is tested for each stock; the trade lists "
                            "pick the best one per stock.")
    stop_mode = st.radio("Stop-loss", ["Fixed %", "Scale with target"], horizontal=True,
                         help="'Scale with target' sets each trade's stop as a multiple of its profit target, "
                              "so a +0.5% target gets a proportionally tight stop.")
    stop_pct, stop_ratio = 5.0, None
    if stop_mode == "Fixed %":
        stop_pct = st.slider("Stop-loss (%)", 0.5, 20.0, 5.0, 0.5, format="%.1f%%")
    else:
        stop_ratio = st.slider("Stop = target ×", 0.25, 3.0, 1.0, 0.25,
                               help="1.0 risks $1 to make $1; 0.5 risks $1 to make $2.")
    horizon = st.slider("Max holding period (trading days)", 1, 60, 20)
    min_conf = st.slider("Minimum direction confidence to act", 0.50, 0.90, 0.58, 0.01)
    return PlanSettings(min_target=lo / 100, max_target=hi / 100, stop=stop_pct / 100, stop_ratio=stop_ratio,
                        horizon_days=horizon, min_confidence=min_conf)


def _price_source_section() -> str:
    st.header("Price data")
    default = st.session_state.setdefault("finnhub_key", config.FINNHUB_API_KEY_DEFAULT)
    entered = st.text_input("Finnhub API key (optional)", type="password", value="",
                            placeholder="set" if default else "paste key for tick-by-tick prices",
                            help="Free at finnhub.io. Streams every US trade in real time. Kept only while the app "
                                 "runs; to keep it permanently run: setx FINNHUB_API_KEY yourkey").strip()
    if entered:
        st.session_state["finnhub_key"] = entered
    key = st.session_state["finnhub_key"]
    if not key:
        st.caption("Yahoo Finance (Nasdaq real-time, 1-minute resolution, incl. pre-market and after-hours). "
                   "Add a Finnhub key for tick-by-tick prices.")
        return key
    stream = data.live_stream(key)
    if stream.status == "live":
        st.caption(f"🟢 Finnhub live stream connected · {stream.trades_received:,} trades received")
    elif "401" in stream.error or "403" in stream.error:
        st.caption("🔴 Finnhub rejected the key. Check it at finnhub.io/dashboard. Using Yahoo meanwhile.")
    else:
        st.caption(f"🟡 Finnhub stream {stream.status}… Using Yahoo meanwhile.")
    return key


def render() -> Controls:
    with st.sidebar:
        watchlist = _watchlist_section()

        st.divider()
        st.header("Data")
        if st.button("🔄 Refresh news now", width="stretch"):
            data.refresh_news.clear()
            data.load_yahoo_trades.clear()
        auto = st.toggle("Auto-refresh", value=True, help="Re-collects news every 5 minutes and quotes every refresh.")
        refresh_secs = st.select_slider("Refresh every", options=[15, 30, 60, 120, 300], value=30,
                                        format_func=lambda s: f"{s}s" if s < 60 else f"{s // 60} min")
        years = st.slider("History to back-fill (years)", 1, 10, 10)
        if st.button("📚 Build history", width="stretch",
                     help="Downloads past headlines for every stock in the background (Google News archive) so "
                          "the model can learn how each one reacted to past news. Finished weeks are skipped, so "
                          "this is quick for stocks that are already set up."):
            for t, names in watchlist.items():
                jobs.enqueue(t, names, years=years)
            st.toast("History download queued for your watchlist.", icon="📚")
        components.setup_jobs_panel()

        st.divider()
        plan = _plan_section()

        st.divider()
        key = _price_source_section()

    return Controls(watchlist=watchlist, auto_refresh=auto, refresh_secs=refresh_secs, plan=plan, finnhub_key=key)
