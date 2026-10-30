"""Out-of-sample checks: train on the oldest 70% of news events, score the newest 30%.

Reports accuracy, calibration (claimed vs actual confidence) and a backtest of what trading the
model's calls would have earned after costs.
"""
from typing import Callable

import numpy as np
import pandas as pd

from . import config, forecast

OOS_TRAIN_SHARE = 0.7
CONFIDENT = 0.6
CALIBRATION_BINS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0)
THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def _calibration(conf: np.ndarray, correct: np.ndarray) -> list[dict]:
    """Per confidence band: what the model claimed vs how often it was actually right."""
    rows = []
    for lo, hi in zip(CALIBRATION_BINS, CALIBRATION_BINS[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            rows.append({"band": f"{lo:.0%}–{hi:.0%}", "claimed": float(conf[m].mean()),
                         "actual": float(correct[m].mean()), "events": int(m.sum())})
    return rows


def _by_threshold(conf: np.ndarray, correct: np.ndarray) -> list[dict]:
    """If we only acted on calls at least this confident: how often, and how accurate."""
    rows = []
    for t in THRESHOLDS:
        m = conf >= t
        rows.append({"threshold": t, "share": float(m.mean()), "events": int(m.sum()),
                     "accuracy": float(correct[m].mean()) if m.any() else np.nan})
    return rows


def _backtest(p_up: np.ndarray, ret: np.ndarray) -> list[dict]:
    """What trading the model's calls would have earned: buy when it said up, short when it said down,
    from the first tradeable price after the news to the next close, minus round-trip trading costs."""
    side = np.where(p_up > 0.5, 1.0, -1.0)
    pnl = side * ret - config.ROUND_TRIP_COST
    conf = np.maximum(p_up, 1 - p_up)
    rows = []
    for t in THRESHOLDS:
        m = conf >= t
        rows.append({"threshold": t, "trades": int(m.sum()), "share": float(m.mean()),
                     "win_rate": float((pnl[m] > 0).mean()) if m.any() else np.nan,
                     "avg_return": float(pnl[m].mean()) if m.any() else np.nan,
                     "total_return": float(pnl[m].sum()) if m.any() else 0.0})
    return rows


def evaluate_out_of_sample(ev: pd.DataFrame, fit: Callable[[pd.DataFrame], object]) -> dict:
    """Fit on the oldest 70% of events and score the newest 30%, which the model never saw."""
    if len(ev) < config.MIN_EVENTS_TO_TRAIN * 1.5:
        return {}
    cut = ev["date"].quantile(OOS_TRAIN_SHARE)
    train_ev, test_ev = ev[ev["date"] <= cut], ev[ev["date"] > cut]
    if len(test_ev) < 15:
        return {}
    m = fit(train_ev)
    preds = [m.predict(r.ticker, r.sentiment, r.n, r.category, r.indirect_share) for r in test_ev.itertuples()]
    p_up = np.array([p.p_up for p in preds])
    p_imp = np.array([p.p_impact for p in preds])
    actual_up = (test_ev["abnormal"] > 0).values
    correct = (p_up > 0.5) == actual_up
    conf = np.maximum(p_up, 1 - p_up)
    confident = conf >= CONFIDENT
    impact = test_ev["impact"].values
    close_ranges = forecast.coverage(test_ev["ret"].abs().to_numpy(),
                                     forecast.event_sigma(test_ev, p_imp, m.range_scale["68"]),
                                     forecast.event_sigma(test_ev, p_imp, m.range_scale["90"]))
    return {
        "close_ranges": close_ranges,
        "calibration": _calibration(conf, correct),
        "by_threshold": _by_threshold(conf, correct),
        "backtest": _backtest(p_up, test_ev["ret"].to_numpy()),
        "trading_cost": config.ROUND_TRIP_COST,
        "train_events": len(train_ev),
        "test_events": len(test_ev),
        "test_period": f"{test_ev['date'].min():%Y-%m-%d} to {test_ev['date'].max():%Y-%m-%d}",
        "direction_accuracy": float(correct.mean()),
        "baseline_accuracy": float(max(actual_up.mean(), 1 - actual_up.mean())),
        "confident_share": float(confident.mean()),
        "confident_accuracy": float(correct[confident].mean()) if confident.any() else np.nan,
        "impact_brier": float(np.mean((p_imp - impact) ** 2)),
        "impact_base_brier": float(np.mean((m.base_impact_rate - impact) ** 2)),
    }
