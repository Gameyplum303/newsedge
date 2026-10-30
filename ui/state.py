"""Typed bundles passed between the sidebar, the data layer and the views."""
from dataclasses import dataclass
from datetime import datetime

from engine.pipeline import Analysis
from engine.prices import Quote
from engine.strategy import PlanSettings

WatchItems = tuple[tuple[str, tuple[str, ...]], ...]   # hashable watchlist, for Streamlit caching


@dataclass(frozen=True)
class Controls:
    """Everything the user set in the sidebar."""
    watchlist: dict[str, list[str]]
    auto_refresh: bool
    refresh_secs: int
    plan: PlanSettings
    finnhub_key: str

    @property
    def tickers(self) -> list[str]:
        return sorted(self.watchlist)

    @property
    def watch_items(self) -> WatchItems:
        return tuple((t, tuple(self.watchlist[t])) for t in self.tickers)


@dataclass
class Page:
    """What the views render from."""
    controls: Controls
    analysis: Analysis
    quotes: dict[str, Quote]
    feed_report: dict
    fetched_at: str
    new_articles: int
    now: datetime
