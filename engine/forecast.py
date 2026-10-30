"""Predicted closing price, with ranges that say how sure the model is.

A single price can't carry a confidence (the chance of closing on exactly that cent is ~0), so the
forecast is a most-likely close plus 68% and 90% ranges and the chance of closing above the current
price. It moves with:
  * the live price and the time left until the close (the range narrows as the close approaches)
  * how much news there is: the model's chance of an outsized move sets how wide the range is
  * the news direction outside market hours, when it shifts the next close (historically, news
    during the session is only tradeable from the following session, so intraday forecasts stay
    centred on the current price)
Range widths are calibrated on past news events so that "68%" contains about 68% of real closes;
evaluation.py checks that on years the model never saw.
"""
import math
from dataclasses import dataclass
from datetime import datetime
from statistics import NormalDist

import numpy as np

from . import config, market_clock

Z68, Z90 = NormalDist().inv_cdf(0.84135), NormalDist().inv_cdf(0.95)
NORMAL_IMPACT_RATE = 2 * (1 - NormalDist().cdf(config.IMPACT_Z))   # ~13% of moves exceed IMPACT_Z sigma
DEFAULT_SCALES = {"68": 1.0, "90": 1.0}


def volatility_multiplier(p_impact) -> np.ndarray | float:
    """How much bigger than usual today's moves should be, given the chance of an outsized move.

    If moves were normal at the usual size, P(|move| > IMPACT_Z sigma) would be ~13%. A higher chance
    means a wider distribution; solve for the width that gives the model's chance.
    """
    p = np.clip(np.asarray(p_impact, dtype=float), 0.02, 0.9)
    inv = np.vectorize(lambda q: NormalDist().inv_cdf(1 - q / 2))(p)
    out = np.clip(config.IMPACT_Z / inv, 0.6, 3.0)
    return float(out) if out.ndim == 0 else out


def horizon_sigma(daily_vol, p_impact, variance_share, scale: float = 1.0):
    """Expected size of the move until the close, as a fraction of the price."""
    return np.asarray(daily_vol) * volatility_multiplier(p_impact) * np.sqrt(variance_share) * scale


def event_sigma(ev, p_impact, scale: float = 1.0) -> np.ndarray:
    """Expected size of each past news event's move: an open entry misses the overnight gap, a close
    entry gets the gap and a full session."""
    share = np.where(ev["entry"] == "open", 1 - config.OVERNIGHT_GAP_SHARE, 1.0)
    return horizon_sigma(ev["daily_vol"].values, p_impact, share, scale)


def coverage(abs_return, sigma68, sigma90) -> dict:
    """How often real moves landed inside the 68% / 90% ranges."""
    r = np.asarray(abs_return)
    return {"inside_68": float(np.mean(r <= Z68 * np.asarray(sigma68))),
            "inside_90": float(np.mean(r <= Z90 * np.asarray(sigma90))), "events": int(len(r))}


def fit_scales(abs_return, sigma) -> dict[str, float]:
    """Factors that make the 68% / 90% ranges contain 68% / 90% of past outcomes."""
    ratio = np.asarray(abs_return) / np.asarray(sigma)
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) < 30:
        return dict(DEFAULT_SCALES)
    return {"68": float(np.quantile(ratio, 0.6827) / Z68), "90": float(np.quantile(ratio, 0.90) / Z90)}


@dataclass(frozen=True)
class CloseForecast:
    close_at: datetime        # the session close being forecast (ET)
    label: str                # "today's close" or e.g. "Fri Oct 30 close"
    price: float              # most likely close
    low68: float
    high68: float
    low90: float
    high90: float
    p_above_now: float        # chance the close is above the current price
    news_shifts_price: bool   # False during market hours (news only widens the range then)


def forecast_close(price: float, daily_vol: float, p_impact: float, expected_move: float,
                   scales: dict | None = None, now: datetime | None = None) -> CloseForecast:
    scales = scales or DEFAULT_SCALES
    s = market_clock.status(now)
    close_at = s["next_close"]
    if s["state"] == "open":
        open_dt, _ = market_clock.session_bounds(close_at.date())
        left = max(0.0, (close_at - s["now"]).total_seconds()) / (close_at - open_dt).total_seconds()
        share, drift = (1 - config.OVERNIGHT_GAP_SHARE) * left, 0.0
    else:
        share, drift = 1.0, expected_move   # overnight gap plus a full session, shifted by the news
    s68 = float(horizon_sigma(daily_vol, p_impact, share, scales["68"]))
    s90 = float(horizon_sigma(daily_vol, p_impact, share, scales["90"]))
    centre = price * (1 + drift)
    today = close_at.date() == s["now"].date()
    return CloseForecast(
        close_at=close_at,
        label=f"today's close ({close_at:%I:%M %p} ET)" if today else f"{close_at:%a %b %d} close",
        price=centre,
        low68=centre * math.exp(-Z68 * s68), high68=centre * math.exp(Z68 * s68),
        low90=centre * math.exp(-Z90 * s90), high90=centre * math.exp(Z90 * s90),
        p_above_now=NormalDist().cdf(drift / s68) if s68 > 0 else 0.5,
        news_shifts_price=s["state"] != "open",
    )
