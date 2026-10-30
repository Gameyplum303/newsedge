"""Find trades with a profit target anywhere in a range (default +0.1% to +10%), with Monte-Carlo odds.

For each signal we simulate price paths in 30-minute steps (13 per trading day, the first step of
each day being the overnight gap). Day 1's opening gap carries the news-driven expected move with
news-day volatility; after that prices follow a zero-drift random walk at the stock's normal
volatility (news edges decay fast). Intraday steps matter for small targets: a +0.1% target is
usually touched within the day even if the close is lower, and stop orders trigger near the stop
price except when the price gaps through it. On the same paths we test every profit target in the
range, for a long and a short position, and measure how often it is reached before the stop-loss.

Two rankings come out of that grid:
  * highest confidence: per stock, the target most likely to be reached (expected return > 0)
  * highest return:     per stock, the target with the largest expected return
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .signals import Signal

TARGET_GRID = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03,
               0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
MIN_EDGE = 0.0002   # expected return must beat this (0.02%) to count as a trade
STEPS_PER_DAY = 13  # 30-minute bars
GAP_SHARE = config.OVERNIGHT_GAP_SHARE
MIN_STOP = 0.001    # never place a stop closer than 0.1%

LONG, SHORT, WAIT = "BUY (go long)", "SELL SHORT / buy puts", "WAIT – no edge"


@dataclass(frozen=True)
class PlanSettings:
    min_target: float = 0.001     # +0.1%
    max_target: float = 0.10      # +10%
    stop: float = 0.05            # fixed stop-loss, e.g. 0.05 = -5%
    stop_ratio: float | None = None  # if set, stop = target x ratio (1.0 = risk 1 to make 1)
    horizon_days: int = 20        # trading days to hold at most
    min_confidence: float = 0.58
    cost: float = config.ROUND_TRIP_COST   # subtracted from every trade's result
    simulations: int = 8000       # antithetic pairs, so 4,000 independent draws

    def targets(self) -> tuple:
        grid = [t for t in TARGET_GRID if self.min_target - 1e-12 <= t <= self.max_target + 1e-12]
        return tuple(sorted({round(self.min_target, 6), *grid, round(self.max_target, 6)}))

    def stop_for(self, target: float) -> float:
        return max(MIN_STOP, target * self.stop_ratio) if self.stop_ratio else self.stop


@lru_cache(maxsize=4)
def _paths(mu1: float, sd1: float, sd: float, horizon: int, sims: int) -> np.ndarray:
    k = STEPS_PER_DAY
    sds = np.tile(np.r_[np.sqrt(GAP_SHARE) * sd, np.full(k - 1, np.sqrt((1 - GAP_SHARE) / (k - 1)) * sd)], horizon)
    sds[0] = np.sqrt(GAP_SHARE) * sd1                        # day-1 opening gap: news volatility
    sds[1:k] = np.sqrt((1 - GAP_SHARE) / (k - 1)) * sd1      # rest of day 1
    rng = np.random.default_rng(7)                           # fixed seed: same inputs, same answer
    z = rng.standard_normal((sims // 2, len(sds)))
    r = np.vstack([z, -z]) * sds                             # antithetic pairs cut simulation noise
    r[:, 0] += mu1                                           # news move is priced in at the open
    return np.cumprod(1.0 + r, axis=1, dtype=np.float32) - 1.0


def _evaluate(pnl: np.ndarray, target: float, stop: float, cost: float = 0.0) -> dict:
    n, steps = pnl.shape
    hit_t = pnl >= target
    hit_s = pnl <= -stop
    first_t = np.where(hit_t.any(1), hit_t.argmax(1), steps + 1)
    first_s = np.where(hit_s.any(1), hit_s.argmax(1), steps + 1)
    won = first_t < first_s
    lost = first_s < first_t
    # Targets fill at the limit price; stops fill at the bar's price, which can gap past the stop.
    stop_fill = pnl[np.arange(n), np.minimum(first_s, steps - 1)]
    outcome = np.where(won, target, np.where(lost, stop_fill, pnl[:, -1]))
    return {
        "p_target": float(won.mean()),
        "p_stop": float(lost.mean()),
        "expected_return": float(outcome.mean()) - cost,
        "days_to_target": float(np.median(first_t[won] // STEPS_PER_DAY) + 1) if won.any() else np.nan,
    }


@lru_cache(maxsize=512)
def _grid(mu1: float, sd1: float, sd: float, side: int, settings: PlanSettings) -> tuple:
    pnl = side * _paths(mu1, sd1, sd, settings.horizon_days, settings.simulations)
    return tuple((t, settings.stop_for(t), _evaluate(pnl, t, settings.stop_for(t), settings.cost))
                 for t in settings.targets())


def plan(signals: list[Signal], prices: dict[str, float], model, settings: PlanSettings) -> pd.DataFrame:
    """Every (ticker, side, target) candidate with its odds; `actionable` marks real trade ideas.

    `prices` maps ticker -> current price (the entry price).
    """
    rows = []
    for sig in signals:
        price = prices.get(sig.ticker)
        if price is None:
            continue
        daily_vol = model.daily_vol(sig.ticker)
        sd1 = max(daily_vol, sig.typical_news_move * 1.25)
        for side in (+1, -1):
            aligned = sig.direction == ("UP" if side == 1 else "DOWN")
            # Rounded inputs let unchanged signals reuse cached simulations between refreshes.
            grid = _grid(round(sig.expected_move, 5), round(sd1, 5), round(daily_vol, 5), side, settings)
            for target, stop, res in grid:
                rows.append({
                    "ticker": sig.ticker,
                    "action": LONG if side == 1 else SHORT,
                    "side": side,
                    "signal": sig.direction,
                    "confidence": sig.confidence,
                    "entry": price,
                    "target_pct": target,
                    "stop_pct": stop,
                    "target_price": price * (1 + side * target),
                    "stop_price": price * (1 - side * stop),
                    **res,
                    "daily_vol": daily_vol,
                    "actionable": (aligned and sig.confidence >= settings.min_confidence
                                   and res["expected_return"] > MIN_EDGE),
                })
    return pd.DataFrame(rows)


def _best_per_ticker(cands: pd.DataFrame, by: list[str], top: int) -> pd.DataFrame:
    act = cands[cands["actionable"]] if not cands.empty else cands
    if act.empty:
        return act
    return act.sort_values(by, ascending=False).drop_duplicates("ticker").head(top).reset_index(drop=True)


def highest_confidence(cands: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Per stock, the target most likely to be hit; ranked by that probability."""
    return _best_per_ticker(cands, ["p_target", "expected_return"], top)


def highest_return(cands: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Per stock, the target with the largest expected return; ranked by expected return."""
    return _best_per_ticker(cands, ["expected_return", "p_target"], top)


def per_ticker_summary(cands: pd.DataFrame) -> pd.DataFrame:
    """One row per stock: its best-return trade, or WAIT when nothing qualifies."""
    if cands.empty:
        return cands
    out = []
    for _, g in cands.groupby("ticker", sort=False):
        act = g[g["actionable"]]
        if not act.empty:
            out.append(act.sort_values("expected_return", ascending=False).iloc[0])
        else:
            row = g.sort_values("expected_return", ascending=False).iloc[0].copy()
            row["action"] = WAIT
            row[["target_price", "stop_price"]] = np.nan
            out.append(row)
    df = pd.DataFrame(out)
    return df.sort_values(["actionable", "expected_return"], ascending=[False, False]).reset_index(drop=True)