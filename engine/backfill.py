"""Resumable back-fill of historical headlines.

Three kinds of search per ticker: "news" (stock coverage, weekly windows), "incidents" (accidents,
outages, disasters naming the company) and "related" (partners, suppliers, competitors and government
business, without the company); the last two use two-week windows since they are rarer. With Alpaca
keys, a fourth kind, "alpaca", downloads Benzinga news with exact publication times (2015 on) in
monthly windows; its pages aren't capped, so longer windows lose nothing.
Each finished (kind, ticker, window) is recorded in the database, so an interrupted or rate-limited run
picks up where it stopped, even on a later day when the windows fall on different dates: a window is
skipped when all of its days are already covered. Weekly searches return about twice as many articles
per month as monthly ones, because Google News caps every search at 100 results.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from collections.abc import Callable

from . import alpaca, news, profiles, store

WINDOW_DAYS = {"news": 7, "incidents": 14, "related": 14, "alpaca": 30}
KINDS = tuple(WINDOW_DAYS)
RECENT_DAYS = 3          # windows ending this recently are fetched again next time (news still arriving)
MAX_RETRIES = 3
RETRY_WAIT_S = 20        # first back-off when a search fails; doubles each retry
GIVE_UP_AFTER = 5        # consecutive failed windows before stopping (Google is rate limiting)

Progress = Callable[[str, int, int], None]   # ("TICKER kind", windows done, windows total)


@dataclass
class BackfillResult:
    ticker: str
    kind: str
    windows_total: int
    windows_fetched: int = 0
    windows_skipped: int = 0     # already done in an earlier run
    new_articles: int = 0
    stopped_early: bool = False  # gave up after repeated failures; run again later to resume


def windows(years: float, today: date | None = None, window_days: int = 7) -> list[tuple[date, date]]:
    """Consecutive [start, end) windows covering the last `years`, newest first."""
    end = (today or date.today()) + timedelta(days=1)
    oldest = end - timedelta(days=round(365.25 * years))
    out = []
    while end > oldest:
        start = max(end - timedelta(days=window_days), oldest)
        out.append((start, end))
        end = start
    return out


def covered_days(ranges) -> set[date]:
    """Every day inside the given (start, end) ISO ranges, end exclusive."""
    days = set()
    for start, end in ranges:
        d, stop = date.fromisoformat(start), date.fromisoformat(end)
        while d < stop:
            days.add(d)
            d += timedelta(days=1)
    return days


def _is_covered(start: date, end: date, covered: set[date]) -> bool:
    return all(start + timedelta(days=k) in covered for k in range((end - start).days))


def applicable_kinds(ticker: str, kinds=KINDS) -> list[str]:
    """The searches worth running for a company: "related" needs partners or government business,
    "alpaca" needs Alpaca keys."""
    return [k for k in kinds if (k != "related" or profiles.related_terms(ticker))
            and (k != "alpaca" or alpaca.available())]


def coverage(ticker: str, kind: str = "news", years: float = 10, today: date | None = None) -> float:
    """Share of the last `years` already downloaded for a ticker and search kind (0..1)."""
    span = windows(years, today, WINDOW_DAYS[kind])
    start, end = span[-1][0], span[0][1]
    covered = covered_days(store.completed_ranges(ticker, kind))
    return sum(start + timedelta(days=k) in covered for k in range((end - start).days)) / (end - start).days


def backfill_ticker(ticker: str, names: list[str], years: float, kind: str = "news", pause: float = 0.6,
                    progress: Progress | None = None, today: date | None = None) -> BackfillResult:
    if kind not in applicable_kinds(ticker, [kind]):
        return BackfillResult(ticker, kind, 0)   # no partners or government business to search for
    today = today or date.today()
    todo = windows(years, today, WINDOW_DAYS[kind])
    covered = covered_days(store.completed_ranges(ticker, kind))
    result = BackfillResult(ticker, kind, len(todo))
    failures_in_a_row = 0
    for i, (start, end) in enumerate(todo, 1):
        if _is_covered(start, end, covered):
            result.windows_skipped += 1
        else:
            articles = None
            for attempt in range(MAX_RETRIES):
                articles = news.fetch_window(ticker, names, start, end, kind)
                if articles is not None:
                    break
                time.sleep(RETRY_WAIT_S * 2 ** attempt)
            if articles is None:
                failures_in_a_row += 1
                if failures_in_a_row >= GIVE_UP_AFTER:
                    result.stopped_early = True
                    break
            else:
                failures_in_a_row = 0
                result.new_articles += store.save_articles(articles)
                result.windows_fetched += 1
                if (today - end).days >= RECENT_DAYS:
                    store.mark_window(ticker, start.isoformat(), end.isoformat(), len(articles), kind)
            time.sleep(pause)
        if progress:
            progress(f"{ticker} {kind}", i, len(todo))
    return result


def backfill(watchlist: dict[str, list[str]], years: float, kinds=KINDS, workers: int = 3,
             progress: Progress | None = None) -> list[BackfillResult]:
    """Back-fill every ticker and kind, a few in parallel (more workers hit Google's rate limits sooner)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(backfill_ticker, t, names, years, kind, progress=progress)
                   for kind in kinds for t, names in watchlist.items()]
        return [f.result() for f in futures]
