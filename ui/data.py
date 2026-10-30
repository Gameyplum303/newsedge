"""Cached data access for the dashboard. Every network or heavy call goes through here."""
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from engine import articles, listings, livefeed, market_clock, news, prices, store
from engine.model import NewsModel, train
from engine.prices import Quote
from .state import WatchItems


def _as_dict(items: WatchItems) -> dict[str, list[str]]:
    return {t: list(names) for t, names in items}


# ---------------------------------------------------------------- news & history
@st.cache_data(ttl=300, show_spinner="Collecting the latest business news…")
def refresh_news(items: WatchItems) -> tuple[dict, int, str]:
    """Fetch live feeds and store new articles. Returns (feed report, new article count, fetch time)."""
    articles, report = news.fetch_live(_as_dict(items))
    return report, store.save_articles(articles), datetime.now(timezone.utc).isoformat()


@st.cache_data(ttl=300, show_spinner=False)
def load_articles(items: WatchItems, fetched_at: str) -> pd.DataFrame:
    """Stored articles linked to each stock (relevance, link, per-stock sentiment) under the current rules."""
    watchlist = _as_dict(items)
    with st.spinner("Updating stored article scores…"):
        articles.rescore_stored_articles()
    return articles.annotate_articles(store.load_articles(list(watchlist)), watchlist)


@st.cache_data(ttl=6 * 3600, show_spinner="Loading the list of US-listed companies…")
def load_listings() -> pd.DataFrame:
    return listings.load()


# ---------------------------------------------------------------- prices & model
@st.cache_data(ttl=3600, show_spinner="Loading price history…")
def load_prices(tickers: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily (closes, opens)."""
    return prices.daily_prices(list(tickers))


@st.cache_data(ttl=600, show_spinner="Calibrating the model on past news and price moves…")
def train_model(articles: pd.DataFrame, closes: pd.DataFrame, opens: pd.DataFrame,
                last_complete: pd.Timestamp) -> NewsModel:
    return train(articles, closes, last_complete, opens)


@st.cache_data(ttl=300, show_spinner=False)
def load_intraday(ticker: str) -> pd.DataFrame:
    return prices.intraday(ticker)


# ---------------------------------------------------------------- live quotes
@st.cache_data(ttl=10, show_spinner=False)
def load_yahoo_trades(tickers: tuple) -> dict:
    try:
        return prices.yahoo_last_trades(list(tickers))
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _official_closes(tickers: tuple, session_key) -> dict:
    try:
        return prices.official_closes(list(tickers))
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def live_stream(api_key: str) -> livefeed.FinnhubStream:
    return livefeed.FinnhubStream(api_key)


def load_quotes(tickers: tuple, finnhub_key: str = "") -> dict[str, Quote]:
    """Newest trade per ticker from the live stream (if any) and Yahoo, vs the official previous close."""
    # Re-fetch the official closes once the session ends so after-hours quotes know today's close.
    state = market_clock.status()["state"]
    session_key = (market_clock.current_trading_date(), state in ("after-hours", "closed"))
    return prices.quotes(list(tickers), finnhub_key=finnhub_key,
                         stream=live_stream(finnhub_key) if finnhub_key else None,
                         closes_ref=_official_closes(tickers, session_key),
                         yahoo=load_yahoo_trades(tickers))
