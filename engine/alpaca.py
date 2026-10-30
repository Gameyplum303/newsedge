"""Alpaca market-data API: Benzinga news with exact timestamps (2015 on) and minute price bars.

Keys come from the APCA_API_KEY_ID / APCA_API_SECRET_KEY environment variables (free account at
https://alpaca.markets, paper-trading keys are enough; see config.secret). Without keys every
function here returns nothing and NewsEdge runs on its other sources.
"""
import threading
import time
from datetime import date, datetime, timezone

import requests

from . import config

DATA_URL = "https://data.alpaca.markets"
PAGE_LIMIT = 50              # most news items the API returns per request
MIN_INTERVAL_S = 0.32        # free plan allows 200 requests a minute; stay a little under
RETRIES = 4

_lock = threading.Lock()
_last_call = [0.0]


def credentials() -> tuple[str, str] | None:
    key, secret = config.secret("APCA_API_KEY_ID"), config.secret("APCA_API_SECRET_KEY")
    return (key, secret) if key and secret else None


def available() -> bool:
    return credentials() is not None


def _get(path: str, params: dict, timeout: int = 20) -> dict | None:
    """One rate-limited GET; None when unreachable, unauthorised or still rate limited after retries."""
    creds = credentials()
    if creds is None:
        return None
    headers = {"APCA-API-KEY-ID": creds[0], "APCA-API-SECRET-KEY": creds[1]}
    for attempt in range(RETRIES):
        with _lock:   # one request at a time across threads, spaced to respect the rate limit
            wait = _last_call[0] + MIN_INTERVAL_S - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        try:
            # No redirects: requests would forward the key headers to whatever host a redirect names.
            resp = requests.get(DATA_URL + path, params=params, headers=headers, timeout=timeout,
                                allow_redirects=False)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429:
            time.sleep(10 * 2 ** attempt)
            continue
        if resp.status_code != 200:
            return None
        try:
            body = resp.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None
    return None


def _iso(t: date | datetime) -> str:
    """UTC timestamp for the API; a plain date means midnight UTC."""
    if not isinstance(t, datetime):
        t = datetime(t.year, t.month, t.day, tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def news(symbols: list[str], start: date | datetime, end: date | datetime | None = None,
         max_pages: int | None = None) -> list[dict] | None:
    """News items mentioning any of `symbols` published in [start, end), newest first.

    Each item has id, headline, summary, author, source, url, symbols, created_at (exact, UTC).
    None when the API could not be reached, so callers can retry.
    """
    params = {"symbols": ",".join(symbols), "start": _iso(start), "limit": PAGE_LIMIT,
              "sort": "desc", "include_content": "false", "exclude_contentless": "false"}
    if end is not None:
        params["end"] = _iso(end)
    items, pages = [], 0
    while True:
        page = _get("/v1beta1/news", params)
        if page is None:
            return None
        items += page.get("news") or []
        pages += 1
        token = page.get("next_page_token")
        if not token or (max_pages and pages >= max_pages):
            return items
        params["page_token"] = token


def minute_bars(symbol: str, start: date | datetime, end: date | datetime,
                timeframe: str = "1Min") -> list[dict] | None:
    """Price bars for one symbol: t (UTC), o, h, l, c, v. Consolidated (SIP) prices, which the free
    plan serves for anything older than 15 minutes. None when unreachable."""
    params = {"timeframe": timeframe, "start": _iso(start), "end": _iso(end), "limit": 10000,
              "adjustment": "split", "feed": "sip"}
    bars = []
    while True:
        page = _get(f"/v2/stocks/{symbol}/bars", params)
        if page is None:
            return None
        bars += page.get("bars") or []
        token = page.get("next_page_token")
        if not token:
            return bars
        params["page_token"] = token
