"""Predicted close: ranges that react to news and time left, and honest calibration."""
from datetime import datetime

import numpy as np
import pytest

from engine import forecast
from engine.forecast import NORMAL_IMPACT_RATE, Z68, Z90, fit_scales, forecast_close, volatility_multiplier
from engine.market_clock import ET


def _et(*args):
    return datetime(*args, tzinfo=ET)


MORNING, LATE, EVENING = _et(2026, 10, 20, 10, 0), _et(2026, 10, 20, 15, 30), _et(2026, 10, 20, 17, 0)


def test_more_news_means_wider_ranges():
    assert volatility_multiplier(NORMAL_IMPACT_RATE) == pytest.approx(1.0)
    assert volatility_multiplier(0.05) < 1.0 < volatility_multiplier(0.3) < volatility_multiplier(0.6)
    calm = forecast_close(100, 0.02, 0.05, 0.0, now=MORNING)
    busy = forecast_close(100, 0.02, 0.5, 0.0, now=MORNING)
    assert busy.high68 - busy.low68 > calm.high68 - calm.low68


def test_ranges_narrow_as_the_close_approaches():
    early = forecast_close(100, 0.02, 0.13, 0.0, now=MORNING)
    late = forecast_close(100, 0.02, 0.13, 0.0, now=LATE)
    assert early.low90 < early.low68 < 100 < early.high68 < early.high90
    assert late.high90 - late.low90 < early.high90 - early.low90
    assert "today" in late.label


def test_intraday_forecast_stays_centred_on_the_price():
    f = forecast_close(100, 0.02, 0.13, 0.01, now=MORNING)
    assert f.price == 100 and f.p_above_now == 0.5 and not f.news_shifts_price


def test_after_hours_news_shifts_the_next_close():
    up = forecast_close(100, 0.02, 0.13, 0.01, now=EVENING)
    down = forecast_close(100, 0.02, 0.13, -0.01, now=EVENING)
    assert up.close_at.date().isoformat() == "2026-10-21" and up.news_shifts_price
    assert up.price == pytest.approx(101) and down.price == pytest.approx(99)
    assert up.p_above_now > 0.5 > down.p_above_now


def test_scales_are_one_when_moves_match_the_assumed_size():
    rng = np.random.default_rng(0)
    sigma = np.full(20_000, 0.02)
    s = fit_scales(np.abs(rng.normal(0, sigma)), sigma)
    assert s["68"] == pytest.approx(1.0, abs=0.03) and s["90"] == pytest.approx(1.0, abs=0.03)
    assert fit_scales([0.01] * 5, [0.02] * 5) == forecast.DEFAULT_SCALES    # too few to fit


def test_coverage_counts_moves_inside_each_range():
    c = forecast.coverage([0.5 * Z68, 1.5 * Z68, 1.5 * Z90, 2 * Z90], 1.0, 1.0)
    assert c == {"inside_68": 0.25, "inside_90": 0.5, "events": 4}


def test_model_calibrates_ranges_and_checks_them_out_of_sample(trained_model):
    assert "daily_vol" in trained_model.events and trained_model.events["daily_vol"].gt(0).all()
    assert set(trained_model.range_scale) == {"68", "90"}
    cr = trained_model.evaluation["close_ranges"]
    assert cr["events"] == trained_model.evaluation["test_events"]
    assert 0.5 < cr["inside_68"] < 0.85 and 0.8 < cr["inside_90"] <= 1.0
