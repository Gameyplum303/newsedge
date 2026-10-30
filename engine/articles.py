"""Prepare stored articles for the model: keep their scores current and link them to stocks."""
import numpy as np
import pandas as pd

from . import linking, sentiment, store


def rescore_stored_articles() -> int:
    """Bring stored sentiment and event types up to the current scorer. Returns headlines re-scored.

    Everything is re-scored when sentiment.SCORER_VERSION changes; otherwise only headlines fetched
    since the last pass (e.g. written by an older process still running a back-fill).
    """
    version = store.get_meta("scorer_version")
    since = store.get_meta("scored_through") if version == sentiment.SCORER_VERSION else None
    rows = store.articles_fetched_since(since)
    if rows.empty:
        return 0
    summaries = rows["summary"].fillna("")
    store.update_scores([(round(sentiment.score_text(t, s), 4), sentiment.classify(t, s), i)
                         for i, t, s in zip(rows["id"], rows["title"], summaries)])
    store.set_meta("scorer_version", sentiment.SCORER_VERSION)
    store.set_meta("scored_through", rows["fetched_at"].max())
    return len(rows)


def annotate_articles(df: pd.DataFrame, watchlist: dict[str, list[str]]) -> pd.DataFrame:
    """Derive per-stock relevance, link and direction for stored articles under the current rules.

    Adds `link_kind` (see linking), `via` (how an indirect link was made, "" otherwise) and
    `raw_sentiment`. `sentiment` becomes the sentiment *for that stock*: flipped for competitors and
    for stocks that tend to gain from the bad news. Safe to apply twice.
    """
    if df.empty:
        return df.assign(link_kind=pd.Series(dtype=str), via=pd.Series(dtype=str),
                         raw_sentiment=pd.Series(dtype=float))
    matchers = linking.build_matchers(watchlist)
    raw = df["raw_sentiment"] if "raw_sentiment" in df else df["sentiment"]
    links = [linking.link(matchers.get(t), t, title, summary, cat, linking.RELEVANCE_TICKER_FEED)
             for t, title, summary, cat in zip(df["ticker"], df["title"], df["summary"].fillna(""), df["category"])]
    low_signal = df["category"].map(sentiment.LOW_SIGNAL_CATEGORIES).fillna(1.0).to_numpy()
    return df.assign(relevance=np.array([lk.relevance for lk in links]) * low_signal,
                     link_kind=[lk.kind for lk in links],
                     via=[lk.via for lk in links],
                     raw_sentiment=raw,
                     sentiment=raw.to_numpy() * np.array([lk.sign for lk in links]))
