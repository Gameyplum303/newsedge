"""Collect business and world news from RSS feeds, Google News searches and Alpaca (Benzinga).

Each article is tagged with the watchlist stocks it relates to (see linking.py for how).
"""
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable
from urllib.parse import quote, quote_plus

import feedparser
import requests

from . import alpaca, config, linking, profiles, sentiment
from .linking import CompanyMatcher
from .sanitize import plain_text, safe_link

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 NewsEdge/1.0"}

Tagger = Callable[[str, str, str], list[tuple[str, float]]]   # (title, summary, category) -> [(ticker, relevance)]


# ---------------------------------------------------------------- tagging articles with stocks
def _broad_tagger(matchers: dict[str, CompanyMatcher]) -> Tagger:
    """For business and world feeds: any watchlist company the article names or is linked to."""
    def tag(title, summary, category):
        links = ((t, linking.link(m, t, title, summary, category)) for t, m in matchers.items())
        return [(t, lk.relevance) for t, lk in links if lk.relevance > 0]
    return tag


def _ticker_tagger(matcher: CompanyMatcher) -> Tagger:
    """For a company's own feed: never zero relevance, since the feed chose the article."""
    return lambda title, summary, category: [
        (matcher.ticker, linking.link(matcher, matcher.ticker, title, summary, category,
                                      linking.RELEVANCE_TICKER_FEED).relevance)]


def _linked_tagger(matcher: CompanyMatcher) -> Tagger:
    """For searches about a company's partners and government business: keep only real links."""
    def tag(title, summary, category):
        lk = linking.link(matcher, matcher.ticker, title, summary, category)
        return [(matcher.ticker, lk.relevance)] if lk.relevance > 0 else []
    return tag


# ---------------------------------------------------------------- feed parsing
def source_weight(source: str) -> float:
    s = (source or "").lower()
    for name, w in config.SOURCE_WEIGHTS.items():
        if name in s:
            return w
    return config.DEFAULT_SOURCE_WEIGHT


MAX_FEED_BYTES = 5_000_000   # real feeds are well under 1 MB; refuse anything far larger


def _get_feed(url: str, timeout: int = 12):
    try:
        with requests.get(url, headers=HEADERS, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            body = bytearray()
            for chunk in resp.iter_content(64 * 1024):
                body += chunk
                if len(body) > MAX_FEED_BYTES:
                    return None
        return feedparser.parse(bytes(body))   # feedparser never resolves external XML entities
    except Exception:
        return None


def _published(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    return None


def _article_id(title: str) -> str:
    # Keyed on the normalised headline so the same story from two feeds is stored once.
    norm = re.sub(r"[^a-z0-9]", "", title.lower())
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


_SKIP = {name: re.compile(p) for name, p in config.FEED_SKIP_PATTERNS.items()}


def _entries_to_articles(feed, default_source: str, tagger: Tagger, now: datetime,
                         max_age: timedelta | None, skip: re.Pattern | None = None) -> list[dict]:
    out = []
    if not feed:
        return out
    for e in feed.entries:
        title = plain_text(e.get("title", ""))
        if not title or (skip and skip.search(title)):
            continue
        source = default_source
        src = e.get("source")
        if src and getattr(src, "get", None) and src.get("title"):
            source = src.get("title")
            # Google News appends " - Outlet" to every headline.
            if title.endswith(" - " + source):
                title = title[: -len(source) - 3].strip()
        summary = plain_text(e.get("summary", ""))
        if summary.startswith(title[:40]):  # Google News summaries just repeat the headline
            summary = ""
        published = _published(e) or now
        if max_age and now - published > max_age:
            continue
        out += _article_rows(title, summary, e.get("link", ""), source, published, tagger)
    return out


def _article_rows(title: str, summary: str, link: str, source: str, published: datetime, tagger: Tagger) -> list[dict]:
    """One stored row per stock the article relates to (none if it relates to no stock)."""
    category = sentiment.classify(title, summary)
    tags = tagger(title, summary, category)
    if not tags:
        return []
    common = {
        "id": _article_id(title),
        "title": title,
        "summary": summary[:800],
        "link": safe_link(link),
        "source": source,
        "published": published,
        "sentiment": round(sentiment.score_text(title, summary), 4),
        "category": category,
        "source_weight": source_weight(source),
    }
    low_signal = sentiment.LOW_SIGNAL_CATEGORIES.get(category, 1.0)
    return [{**common, "ticker": ticker, "relevance": relevance * low_signal} for ticker, relevance in tags]


# ---------------------------------------------------------------- Alpaca (Benzinga) news
ROUNDUP_SYMBOLS = 4      # an item listing this many symbols or more is a market roundup, not company news


def _alpaca_to_articles(items: list[dict], matchers: dict[str, CompanyMatcher]) -> list[dict]:
    """Alpaca news items -> stored rows for the watchlist stocks each item lists.

    Alpaca says which symbols an item is about, so a company item counts even without naming the
    company (like its own feed). Roundups listing many symbols count only where they name the company.
    """
    out = []
    for item in items:
        title = plain_text(item.get("headline", ""))
        listed = [s for s in item.get("symbols") or [] if s in matchers]
        if not title or not listed:
            continue
        published = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
        source = (item.get("source") or "benzinga").title()
        summary = plain_text(item.get("summary", ""))
        roundup = len(item.get("symbols") or []) >= ROUNDUP_SYMBOLS
        for t in listed:
            tag = _linked_tagger(matchers[t]) if roundup else _ticker_tagger(matchers[t])
            out += _article_rows(title, summary, item.get("url", ""), source, published, tag)
    return out


def _fetch_alpaca_live(watchlist: dict[str, list[str]], now: datetime, max_age: timedelta) -> list[dict] | None:
    items = alpaca.news(list(watchlist), now - max_age, max_pages=6)
    return None if items is None else _alpaca_to_articles(items, linking.build_matchers(watchlist))


def _dedupe(articles: list[dict]) -> list[dict]:
    best = {}
    for a in articles:
        key = (a["id"], a["ticker"])
        cur = best.get(key)
        # Keep the earliest timestamp (closest to when the news broke) and the strongest link.
        if cur is None:
            best[key] = a
        else:
            cur["published"] = min(cur["published"], a["published"])
            cur["relevance"] = max(cur["relevance"], a["relevance"])
    return list(best.values())


# ---------------------------------------------------------------- Google News searches
QUERY_TOPICS = {"news": "stock", "incidents": f"({config.INCIDENT_TERMS})"}


def _google_query(ticker: str, names: list[str], extra: str, kind: str = "news") -> str | None:
    """Google News search: the company plus a topic ("news", "incidents"), or for "related" its
    partners and government topics *without* the company, to catch news that doesn't name it.
    None when there is nothing to search for."""
    if kind == "related":
        terms = profiles.related_terms(ticker)
        if not terms:
            return None
        quoted = " OR ".join(f'"{t}"' for t in terms)
        return quote_plus(f"({quoted}) {extra}".strip())
    who = " OR ".join([f'"{n}"' for n in names[:2]] + [ticker])
    return quote_plus(f"({who}) {QUERY_TOPICS[kind]} {extra}".strip())


def _google_url(query: str) -> str:
    return config.GOOGLE_NEWS_FEED.format(query=query)


# ---------------------------------------------------------------- public API
@dataclass(frozen=True)
class _Feed:
    name: str       # shown in the feed-health report
    source: str     # default outlet name for its articles
    url: str
    tagger: Tagger


def _live_feeds(watchlist: dict[str, list[str]]) -> list[_Feed]:
    matchers = linking.build_matchers(watchlist)
    broad = _broad_tagger(matchers)
    feeds = [_Feed(name, name, url, broad) for name, url in {**config.GENERAL_FEEDS, **config.WORLD_FEEDS}.items()]
    for ticker, names in watchlist.items():
        own = _ticker_tagger(matchers[ticker])
        sym = quote(ticker, safe="")   # tickers are validated on load; encode anyway before building URLs
        feeds += [
            _Feed(f"Yahoo:{ticker}", "Yahoo", config.YAHOO_TICKER_FEED.format(ticker=sym), own),
            _Feed(f"Google News:{ticker}", "Google News", _google_url(_google_query(ticker, names, "when:3d")), own),
            _Feed(f"Google News incidents:{ticker}", "Google News",
                  _google_url(_google_query(ticker, names, "when:3d", "incidents")), own),
            _Feed(f"Seeking Alpha:{ticker}", "Seeking Alpha", config.SEEKING_ALPHA_TICKER_FEED.format(ticker=sym), own),
        ]
        related = _google_query(ticker, names, "when:3d", "related")
        if related:
            feeds.append(_Feed(f"Google News related:{ticker}", "Google News", _google_url(related),
                               _linked_tagger(matchers[ticker])))
    return feeds


def fetch_live(watchlist: dict[str, list[str]], max_age_hours: int = 72) -> tuple[list[dict], dict]:
    """Latest articles for the watchlist from business, world and per-ticker feeds.

    Returns (articles, report) where report maps feed name -> matched articles (None if unreachable).
    """
    now = datetime.now(timezone.utc)

    def run(feed: _Feed):
        parsed = _get_feed(feed.url)
        arts = _entries_to_articles(parsed, feed.source, feed.tagger, now, timedelta(hours=max_age_hours),
                                    _SKIP.get(feed.name))
        return feed.name, parsed is not None, arts

    articles, report = [], {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        benzinga = pool.submit(_fetch_alpaca_live, watchlist, now, timedelta(hours=max_age_hours)) \
            if alpaca.available() else None
        for name, ok, arts in pool.map(run, _live_feeds(watchlist)):
            report[name] = len(arts) if ok else None
            articles.extend(arts)
        if benzinga is not None:
            arts = benzinga.result()
            report["Alpaca (Benzinga)"] = None if arts is None else len(arts)
            articles.extend(arts or [])
    return _dedupe(articles), report


def fetch_window(ticker: str, names: list[str], start: date, end: date, kind: str = "news") -> list[dict] | None:
    """Past articles about one ticker published in [start, end), via Google News date-range search.

    `kind` is "news" (stock coverage), "incidents" (accidents, outages, disasters naming the company),
    "related" (partners, suppliers, competitors and government business, without the company) or
    "alpaca" (Benzinga news from Alpaca, with exact publication times instead of dates).
    Returns None when the search could not be reached (e.g. rate limited), so callers can retry.
    """
    if kind == "alpaca":
        items = alpaca.news([ticker], start, end)
        return None if items is None else _dedupe(_alpaca_to_articles(items, linking.build_matchers({ticker: names})))
    query = _google_query(ticker, names, f"after:{start.isoformat()} before:{end.isoformat()}", kind)
    if query is None:
        return []
    feed = _get_feed(_google_url(query))
    if feed is None:
        return None
    matcher = linking.build_matchers({ticker: names})[ticker]
    tag = _linked_tagger(matcher) if kind == "related" else _ticker_tagger(matcher)
    return _dedupe(_entries_to_articles(feed, "Google News", tag, datetime.now(timezone.utc), None))
