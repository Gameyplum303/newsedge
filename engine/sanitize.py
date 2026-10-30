"""Rules for untrusted text: feed content, hand-edited files and typed input.

Every headline, summary, link and source comes from outside, so it is cleaned before it is stored
and escaped before it is shown. Tickers end up in URLs, file names and searches, so only plain
symbols are accepted.
"""
import html
import re
from urllib.parse import urlsplit

TICKER_RE = re.compile(r"[A-Z][A-Z0-9]{0,6}([.-][A-Z0-9]{1,3})?")   # AAPL, BRK-B, BF.B
_TAG = re.compile(r"<[^>]+>")
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]()<>#!|~])")


def valid_ticker(ticker) -> bool:
    return isinstance(ticker, str) and TICKER_RE.fullmatch(ticker) is not None


def plain_text(text: str) -> str:
    """Feed HTML -> plain text on one line."""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", text or ""))).strip()


def safe_link(url: str) -> str:
    """The URL if it is a plain http(s) link, else "" (no javascript:, data: or other schemes)."""
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    return url if parts.scheme in ("http", "https") and parts.hostname else ""


def markdown_literal(text) -> str:
    """Text shown as-is inside Markdown: no links, images or formatting from a headline.
    ("$" is left to the dashboard, which escapes it for all Markdown.)"""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))
