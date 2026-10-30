"""US equity market (NYSE/Nasdaq) calendar: status, next open/close, countdowns."""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

PRE_MARKET = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
AFTER_HOURS_END = time(20, 0)

# NYSE full-day holidays. Extend each year from https://www.nyse.com/markets/hours-calendars
HOLIDAYS = {
    date(2026, 1, 1): "New Year's Day",
    date(2026, 1, 19): "Martin Luther King Jr. Day",
    date(2026, 2, 16): "Washington's Birthday",
    date(2026, 4, 3): "Good Friday",
    date(2026, 5, 25): "Memorial Day",
    date(2026, 6, 19): "Juneteenth",
    date(2026, 7, 3): "Independence Day (observed)",
    date(2026, 9, 7): "Labor Day",
    date(2026, 11, 26): "Thanksgiving Day",
    date(2026, 12, 25): "Christmas Day",
    date(2027, 1, 1): "New Year's Day",
    date(2027, 1, 18): "Martin Luther King Jr. Day",
    date(2027, 2, 15): "Washington's Birthday",
    date(2027, 3, 26): "Good Friday",
    date(2027, 5, 31): "Memorial Day",
    date(2027, 6, 18): "Juneteenth (observed)",
    date(2027, 7, 5): "Independence Day (observed)",
    date(2027, 9, 6): "Labor Day",
    date(2027, 11, 25): "Thanksgiving Day",
    date(2027, 12, 24): "Christmas Day (observed)",
}

# Sessions that close at 1:00 PM ET.
EARLY_CLOSES = {date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26)}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


def session_bounds(d: date) -> tuple[datetime, datetime]:
    close = EARLY_CLOSE if d in EARLY_CLOSES else REGULAR_CLOSE
    return datetime.combine(d, REGULAR_OPEN, ET), datetime.combine(d, close, ET)


def next_trading_day(d: date) -> date:
    """First trading day on or after d."""
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def previous_trading_day(d: date) -> date:
    """Last trading day strictly before d."""
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def current_trading_date(now: datetime | None = None) -> date:
    """The session prices are measured against: today from 4:00 AM ET on a trading day, else the last one."""
    now = (now or datetime.now(ET)).astimezone(ET)
    d = now.date()
    if is_trading_day(d) and now.time() >= PRE_MARKET:
        return d
    return previous_trading_day(d)


def session_of(ts: datetime) -> str:
    """Which part of the trading day a timestamp falls in: Pre-market, Regular, After-hours or Closed."""
    ts = ts.astimezone(ET)
    if not is_trading_day(ts.date()):
        return "Closed"
    open_dt, close_dt = session_bounds(ts.date())
    if ts < open_dt:
        return "Pre-market"
    return "Regular" if ts < close_dt else "After-hours"


def last_complete_session(days, now: datetime | None = None):
    """Latest date in a daily price index whose regular session has closed (today counts after the close)."""
    now = (now or datetime.now(ET)).astimezone(ET)
    today = pd.Timestamp(now.date())
    if today in days and now >= session_bounds(now.date())[1]:
        return today
    earlier = days[days < today]
    return earlier[-1] if len(earlier) else days[0]


def status(now: datetime | None = None) -> dict:
    """Market state plus the next open and close as ET datetimes."""
    now = (now or datetime.now(ET)).astimezone(ET)
    today = now.date()

    if is_trading_day(today):
        open_dt, close_dt = session_bounds(today)
        if open_dt <= now < close_dt:
            state = "open"
            next_close = close_dt
            next_open = session_bounds(next_trading_day(today + timedelta(days=1)))[0]
        elif now < open_dt:
            state = "pre-market" if now.time() >= PRE_MARKET else "closed"
            next_open, next_close = open_dt, close_dt
        else:
            state = "after-hours" if now.time() < AFTER_HOURS_END else "closed"
            next_open, next_close = session_bounds(next_trading_day(today + timedelta(days=1)))
    else:
        state = "closed"
        next_open, next_close = session_bounds(next_trading_day(today))

    return {
        "now": now,
        "state": state,
        "next_open": next_open,
        "next_close": next_close,
        "seconds_to_open": max(0, (next_open - now).total_seconds()),
        "seconds_to_close": max(0, (next_close - now).total_seconds()),
        "holiday": HOLIDAYS.get(today),
        "early_close": today in EARLY_CLOSES,
    }


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}d {h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"
