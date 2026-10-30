from datetime import date, datetime

import pandas as pd
import pytest

from engine import market_clock as mc

ET = mc.ET


@pytest.mark.parametrize("when, state, next_open, next_close", [
    (datetime(2026, 10, 20, 10, 0, tzinfo=ET), "open", "2026-10-21 09:30", "2026-10-20 16:00"),
    (datetime(2026, 10, 20, 8, 0, tzinfo=ET), "pre-market", "2026-10-20 09:30", "2026-10-20 16:00"),
    (datetime(2026, 10, 20, 3, 0, tzinfo=ET), "closed", "2026-10-20 09:30", "2026-10-20 16:00"),
    (datetime(2026, 10, 20, 17, 0, tzinfo=ET), "after-hours", "2026-10-21 09:30", "2026-10-21 16:00"),
    (datetime(2026, 10, 23, 21, 0, tzinfo=ET), "closed", "2026-10-26 09:30", "2026-10-26 16:00"),   # Friday night
    (datetime(2026, 11, 25, 18, 0, tzinfo=ET), "after-hours", "2026-11-27 09:30", "2026-11-27 13:00"),  # Thanksgiving
    (datetime(2026, 11, 27, 12, 0, tzinfo=ET), "open", "2026-11-30 09:30", "2026-11-27 13:00"),     # early close
    (datetime(2026, 12, 25, 12, 0, tzinfo=ET), "closed", "2026-12-28 09:30", "2026-12-28 16:00"),   # Christmas
])
def test_status(when, state, next_open, next_close):
    s = mc.status(when)
    assert s["state"] == state
    assert s["next_open"] == pd.Timestamp(next_open, tz=ET)
    assert s["next_close"] == pd.Timestamp(next_close, tz=ET)


def test_holiday_and_early_close_flags():
    assert mc.status(datetime(2026, 12, 25, 12, tzinfo=ET))["holiday"] == "Christmas Day"
    assert mc.status(datetime(2026, 11, 27, 12, tzinfo=ET))["early_close"]


@pytest.mark.parametrize("when, expected", [
    (datetime(2026, 10, 20, 3, 59, tzinfo=ET), date(2026, 10, 19)),   # before pre-market: previous session
    (datetime(2026, 10, 20, 4, 0, tzinfo=ET), date(2026, 10, 20)),
    (datetime(2026, 10, 24, 12, 0, tzinfo=ET), date(2026, 10, 23)),   # Saturday -> Friday
    (datetime(2026, 11, 26, 12, 0, tzinfo=ET), date(2026, 11, 25)),   # Thanksgiving -> Wednesday
])
def test_current_trading_date(when, expected):
    assert mc.current_trading_date(when) == expected


@pytest.mark.parametrize("when, session", [
    (datetime(2026, 10, 20, 7, 0, tzinfo=ET), "Pre-market"),
    (datetime(2026, 10, 20, 12, 0, tzinfo=ET), "Regular"),
    (datetime(2026, 10, 20, 16, 0, tzinfo=ET), "After-hours"),
    (datetime(2026, 11, 27, 13, 30, tzinfo=ET), "After-hours"),
    (datetime(2026, 10, 24, 12, 0, tzinfo=ET), "Closed"),
])
def test_session_of(when, session):
    assert mc.session_of(when) == session


def test_last_complete_session():
    days = pd.DatetimeIndex(["2026-10-16", "2026-10-19", "2026-10-20"])
    assert mc.last_complete_session(days, datetime(2026, 10, 20, 15, 59, tzinfo=ET)) == pd.Timestamp("2026-10-19")
    assert mc.last_complete_session(days, datetime(2026, 10, 20, 16, 0, tzinfo=ET)) == pd.Timestamp("2026-10-20")


def test_format_duration():
    assert mc.format_duration(3661) == "01:01:01"
    assert mc.format_duration(2 * 86400 + 5) == "2d 00:00:05"
