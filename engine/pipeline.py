"""The analysis in one call: stored articles + prices -> model, signals and trade ideas."""
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from . import market_clock, signals, strategy
from .model import NewsModel, train
from .signals import Signal


@dataclass
class Analysis:
    articles: pd.DataFrame
    model: NewsModel
    signals: list[Signal]
    candidates: pd.DataFrame      # every (ticker, side, target) the planner tested
    plan: pd.DataFrame            # best trade (or WAIT) per ticker
    top_confidence: pd.DataFrame
    top_return: pd.DataFrame

    def signal(self, ticker: str) -> Signal:
        return next(s for s in self.signals if s.ticker == ticker)


def analyse(articles: pd.DataFrame, closes: pd.DataFrame, watchlist: dict[str, list[str]],
            prices: dict[str, float], settings: strategy.PlanSettings, now: datetime,
            model: NewsModel | None = None, opens: pd.DataFrame | None = None) -> Analysis:
    """Run the full analysis.

    `articles` should already be annotated (see articles.annotate_articles).
    Pass `model` to reuse an already trained one.
    """
    if model is None:
        model = train(articles, closes, market_clock.last_complete_session(closes.index, now), opens)
    sigs = signals.build(articles, model, list(watchlist), now)
    candidates = strategy.plan(sigs, prices, model, settings)
    return Analysis(articles, model, sigs, candidates,
                    strategy.per_ticker_summary(candidates),
                    strategy.highest_confidence(candidates),
                    strategy.highest_return(candidates))
