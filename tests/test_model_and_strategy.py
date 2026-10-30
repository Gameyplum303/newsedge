from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from engine import signals, strategy
from engine.model import NewsModel
from engine.pipeline import analyse
from engine.signals import Signal
from tests.conftest import make_article

NOW = pd.Timestamp("2026-10-20 15:00", tz="UTC").to_pydatetime()


# ---------------------------------------------------------------- model
def test_model_finds_planted_news_effect(trained_model):
    m = trained_model
    assert m.trained and m.n_events > 400
    ev = m.evaluation
    assert ev["direction_accuracy"] > ev["baseline_accuracy"] + 0.03   # real out-of-sample edge
    assert ev["confident_accuracy"] > ev["direction_accuracy"]
    assert m.dir_w[1] > 0   # positive sentiment -> up


def test_untrained_model_uses_defaults_and_bounds():
    m = NewsModel()
    p = m.predict("ZZZ", 1.0, 1000, "earnings_beat")
    assert not m.trained and 0.5 < p.p_up <= 0.97 and 0.01 <= p.p_impact <= 0.95 and p.reliability == 1.0
    assert m.predict("ZZZ", -1.0, 3, "earnings_miss").p_up < 0.5


# ---------------------------------------------------------------- signals
def test_signals_follow_fresh_news(synthetic_market, trained_model):
    articles, _ = synthetic_market
    live = pd.DataFrame([
        make_article("AAA", NOW - timedelta(hours=2), 0.8, "earnings_beat", title="AAA beats estimates"),
        make_article("BBB", NOW - timedelta(hours=3), -0.7, "analyst_downgrade", title="BBB downgraded"),
    ])
    sigs = {s.ticker: s for s in signals.build(pd.concat([articles, live]), trained_model, ["AAA", "BBB", "CCC"], NOW)}
    assert sigs["AAA"].direction == "UP" and sigs["AAA"].expected_move > 0
    assert sigs["BBB"].direction == "DOWN" and sigs["BBB"].expected_move < 0
    assert sigs["CCC"].direction == "NEUTRAL" and sigs["CCC"].n_articles == 0
    assert any("Earnings beat" in r for r in sigs["AAA"].reasons)


# ---------------------------------------------------------------- strategy
class FlatModel:
    def daily_vol(self, ticker):
        return 0.02


def _signal(ticker, direction="UP", conf=0.62, move=0.0025):
    return Signal(ticker, 10, 0.3, "general", conf, 0.2, direction, conf, move, 0.015)


def test_targets_cover_the_requested_range():
    s = strategy.PlanSettings(min_target=0.003, max_target=0.05)
    t = s.targets()
    assert t[0] == 0.003 and t[-1] == 0.05 and list(t) == sorted(t)
    assert strategy.PlanSettings(stop_ratio=0.5).stop_for(0.001) == strategy.MIN_STOP
    assert strategy.PlanSettings(stop=0.07).stop_for(0.02) == 0.07


def test_no_news_means_no_edge():
    cands = strategy.plan([_signal("FLAT", "NEUTRAL", 0.5, 0.0)], {"FLAT": 100.0}, FlatModel(), strategy.PlanSettings())
    assert not cands["actionable"].any()
    assert (cands["expected_return"] <= 0.0005).all()   # gaps past stops make no-news trades slightly negative


def test_rankings_pick_safest_and_most_profitable_targets():
    sigs = [_signal("AAA"), _signal("BBB", move=0.004)]
    cands = strategy.plan(sigs, {"AAA": 100.0, "BBB": 50.0}, FlatModel(), strategy.PlanSettings())
    conf, ret = strategy.highest_confidence(cands), strategy.highest_return(cands)
    for ticker in ("AAA", "BBB"):
        act = cands[(cands["ticker"] == ticker) & cands["actionable"]]
        assert conf.set_index("ticker").at[ticker, "p_target"] == act["p_target"].max()
        assert ret.set_index("ticker").at[ticker, "expected_return"] == act["expected_return"].max()
    assert (cands.loc[cands["side"] == -1, "actionable"] == False).all()   # never trade against an UP signal  # noqa: E712
    row = cands.iloc[0]
    assert row["target_price"] == pytest.approx(row["entry"] * (1 + row["target_pct"]))


def test_simulation_is_deterministic():
    args = ([_signal("AAA")], {"AAA": 100.0}, FlatModel(), strategy.PlanSettings())
    pd.testing.assert_frame_equal(strategy.plan(*args), strategy.plan(*args))


def test_summary_marks_wait_when_nothing_qualifies():
    cands = strategy.plan([_signal("AAA", conf=0.51)], {"AAA": 100.0}, FlatModel(), strategy.PlanSettings())
    (row,) = strategy.per_ticker_summary(cands).to_dict("records")
    assert row["action"] == strategy.WAIT and np.isnan(row["target_price"])


# ---------------------------------------------------------------- pipeline
def test_analyse_end_to_end(synthetic_market):
    articles, closes = synthetic_market
    wl = {"AAA": ["AAA"], "BBB": ["BBB"], "CCC": ["CCC"]}
    live = pd.DataFrame([make_article("AAA", NOW - timedelta(hours=1), 0.9, "earnings_beat", title="AAA beats")])
    a = analyse(pd.concat([articles, live]), closes, wl, {t: 100.0 for t in wl}, strategy.PlanSettings(), NOW)
    assert a.model.trained
    assert {s.ticker for s in a.signals} == set(wl)
    assert set(a.plan["ticker"]) == set(wl)
    assert a.signal("AAA").direction == "UP"
