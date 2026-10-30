"""The model may only credit news with moves you could have traded after it was public."""
import numpy as np
import pandas as pd
import pytest

from engine import config
from engine.evaluation import _backtest
from engine.events import available_at, build_events, entry_points

ET = "America/New_York"
DAYS = pd.bdate_range("2026-10-12", "2026-10-23")     # Mon 12 .. Fri 23


def _at(ts, tz=ET):
    return pd.Series([pd.Timestamp(ts, tz=tz)]).dt.tz_convert("UTC")


def test_date_only_items_count_from_the_end_of_their_date():
    date_only = pd.Series([pd.Timestamp("2026-10-20 07:00:00", tz="UTC")])   # Google's "midnight Pacific"
    precise = pd.Series([pd.Timestamp("2026-10-20 07:00:13", tz="UTC")])
    assert available_at(date_only).iloc[0] == pd.Timestamp("2026-10-21 07:00:00", tz="UTC")
    assert available_at(precise).iloc[0] == precise.iloc[0]


@pytest.mark.parametrize("published, day, at_open", [
    ("2026-10-20 08:00", "2026-10-20", True),     # before the open -> that open
    ("2026-10-20 11:00", "2026-10-20", False),    # during the session -> that close
    ("2026-10-20 17:00", "2026-10-21", True),     # after the close -> next open
    ("2026-10-17 12:00", "2026-10-19", True),     # Saturday -> Monday's open
])
def test_entry_is_the_first_tradeable_price(published, day, at_open):
    idx, opens = entry_points(_at(published), DAYS)
    assert DAYS[idx[0]] == pd.Timestamp(day) and opens[0] == at_open


def test_a_date_only_item_is_never_traded_on_its_own_date():
    pub = pd.Series([pd.Timestamp("2026-10-20 07:00:00", tz="UTC")])   # some time on Tue 20th
    idx, at_open = entry_points(available_at(pub), DAYS)
    assert DAYS[idx[0]] == pd.Timestamp("2026-10-21") and at_open[0]


def _market():
    closes = pd.DataFrame({"SPY": np.full(len(DAYS), 100.0), "AAA": np.linspace(50, 59, len(DAYS))}, index=DAYS)
    opens = closes.shift(1).fillna(50.0)
    opens.loc["2026-10-20", "AAA"] = 55.0             # Tue 20 opens at 55; closes are 50, 51, ... so Tue closes at 56
    return closes, opens


def _article(published, tz=ET):
    return {"ticker": "AAA", "published": pd.Timestamp(published, tz=tz).tz_convert("UTC"), "sentiment": 0.5,
            "category": "general", "relevance": 1.0, "source_weight": 1.0, "link_kind": "direct"}


def test_returns_run_from_entry_to_the_next_close():
    closes, opens = _market()
    arts = pd.DataFrame([_article("2026-10-20 08:00"), _article("2026-10-21 11:00")])
    ev = build_events(arts, closes, DAYS[-1], opens).set_index("entry")
    assert ev.at["open", "ret"] == pytest.approx(56 / 55 - 1)                 # Tue open -> Tue close
    assert ev.at["close", "ret"] == pytest.approx(58 / 57 - 1)                # Wed close -> Thu close
    assert ev.at["close", "date"] == pd.Timestamp("2026-10-22")


def test_trades_that_have_not_closed_are_left_out():
    closes, opens = _market()
    arts = pd.DataFrame([_article("2026-10-22 11:00")])                      # exits Fri 23
    assert build_events(arts, closes, pd.Timestamp("2026-10-22"), opens).empty
    assert len(build_events(arts, closes, pd.Timestamp("2026-10-23"), opens)) == 1


def test_backtest_charges_trading_costs():
    p_up = np.array([0.7, 0.3, 0.52])
    ret = np.array([0.01, -0.02, -0.005])
    rows = {r["threshold"]: r for r in _backtest(p_up, ret)}
    cost = config.ROUND_TRIP_COST
    assert rows[0.5]["trades"] == 3
    assert rows[0.5]["avg_return"] == pytest.approx(np.mean([0.01 - cost, 0.02 - cost, -0.005 - cost]))
    assert rows[0.65]["trades"] == 2 and rows[0.65]["win_rate"] == 1.0
