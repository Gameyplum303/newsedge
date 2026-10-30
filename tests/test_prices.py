from datetime import datetime, timedelta, timezone

import pytest

from engine import prices
from engine.prices import Quote

NOW = datetime(2026, 10, 20, 20, 30, tzinfo=timezone.utc)   # 4:30 PM ET, after-hours


class FakeStream:
    def __init__(self, latest):
        self._latest, self.subscribed = latest, set()

    def subscribe(self, symbols):
        self.subscribed |= set(symbols)

    def latest(self, symbol):
        return self._latest.get(symbol)


def test_newest_trade_wins_and_uses_official_prev_close():
    yahoo = {"AAPL": (340.07, NOW - timedelta(seconds=30)), "JPM": (335.26, NOW - timedelta(minutes=9))}
    stream = FakeStream({"AAPL": (340.20, (NOW - timedelta(seconds=2)).timestamp())})
    ref = {"AAPL": {"prev_close": 342.94, "regular_close": 339.50}, "JPM": {"prev_close": 333.53, "regular_close": None}}
    q = prices.quotes(["AAPL", "JPM", "NONE"], stream=stream, closes_ref=ref, yahoo=yahoo)

    assert set(q) == {"AAPL", "JPM"}
    assert q["AAPL"].price == 340.20 and q["AAPL"].source == "Finnhub live stream"
    assert q["JPM"].source.startswith("Yahoo")
    assert q["AAPL"].change_pct == pytest.approx(340.20 / 342.94 - 1)
    assert stream.subscribed == {"AAPL", "JPM", "NONE"}


def test_quote_session_and_freshness():
    q = Quote(price=340.07, prev_close=342.94, regular_close=339.50, time=NOW - timedelta(seconds=30), source="Yahoo")
    assert q.session == "After-hours"
    label, tip = prices.freshness(q, "after-hours", now=NOW)
    assert label.startswith("🟢 Live")
    assert "4:00 PM close $339.50" in tip and "previous close $342.94" in tip

    stale = Quote(335.26, 333.53, None, NOW - timedelta(minutes=9), "Yahoo")
    assert prices.freshness(stale, "after-hours", now=NOW)[0] == "🟡 Last trade 9m ago"
    assert prices.freshness(stale, "closed", now=NOW)[0].startswith("⚪ Closed")


def test_missing_prev_close_gives_zero_change():
    assert Quote(10.0, None, None, NOW, "x").change_pct == 0.0
