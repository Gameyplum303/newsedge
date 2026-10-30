"""Learn how news has moved prices in the past, and turn that into calibrated probabilities.

Pipeline
  1. Each article is matched to the first price you could have traded at once it was public:
     the next open for news before the open (or the day after, for date-only historical items),
     the close for news during market hours. Nothing is credited to news before it was published.
  2. Articles are grouped into "news events" (one per ticker per entry).
  3. Each event gets the move from that entry to the next close, as a raw return (what a trade
     earned) and as an *abnormal* return: minus beta x the S&P 500 (SPY) move over the same span.
  4. Two logistic regressions are fitted on the events:
       direction : P(abnormal return > 0)        from sentiment, event type, article count
       impact    : P(|abnormal| > IMPACT_Z sigma) from |sentiment|, article count, event type
  5. A chronological 70/30 split reports out-of-sample accuracy and a backtest of what trading the
     calls would have earned after costs, so you can see whether there is any real edge.
"""
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import config, forecast
from .evaluation import evaluate_out_of_sample
from .events import abnormal_returns, build_events
from .sentiment import CATEGORY_PRIOR

# Features: sentiment, event type and article count, plus indirect share: how much of the day's news
# weight came from indirect links (government business, related companies, world-event exposures)
# rather than articles about the company itself. The model learns how much less (or more) those matter.
DEFAULT_DIR_W = np.array([0.0, 1.2, 0.8, 0.3, -0.4])        # bias, sentiment, category z, sentiment*log(n), sentiment*indirect
DEFAULT_IMP_W = np.array([-2.6, 1.0, 0.45, 3.0, -0.5])      # bias, |sentiment|, log(n), category impact rate, indirect
DEFAULT_DAILY_VOL = 0.02                                     # until a stock has price history
MIN_DIRECTIONAL_SENTIMENT = 0.05   # weaker sentiment doesn't count as calling a direction


class Prediction(NamedTuple):
    p_up: float
    p_impact: float
    reliability: float   # how much this stock's own track record scaled the call


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _sentiment_hits(ev: pd.DataFrame) -> pd.Series:
    """For events that called a direction, whether the abnormal move went that way."""
    d = ev[ev["sentiment"].abs() >= MIN_DIRECTIONAL_SENTIMENT]
    return np.sign(d["sentiment"]) == np.sign(d["abnormal"])


def _shrunk_hit_rate(hits: pd.Series, k: float) -> float:
    """Hit rate pulled toward 50% by k pseudo-observations, so small samples stay modest."""
    return float((hits.sum() + 0.5 * k) / (len(hits) + k))


@dataclass
class NewsModel:
    dir_w: np.ndarray = field(default_factory=lambda: DEFAULT_DIR_W.copy())
    imp_w: np.ndarray = field(default_factory=lambda: DEFAULT_IMP_W.copy())
    trained: bool = False
    n_events: int = 0
    base_impact_rate: float = 0.13
    pooled_hit: float = 0.5
    n_cap: float = 8.0               # article counts above this were never seen in training
    category_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    ticker_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    evaluation: dict = field(default_factory=dict)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    range_scale: dict = field(default_factory=lambda: dict(forecast.DEFAULT_SCALES))   # close-range calibration

    # ---- per-feature lookups -------------------------------------------------------
    def category_z(self, cat: str) -> float:
        cs = self.category_stats
        if not cs.empty and cat in cs.index:
            return float(cs.at[cat, "z_shrunk"])
        return 0.3 * CATEGORY_PRIOR.get(cat, 0.0)

    def category_impact(self, cat: str) -> float:
        cs = self.category_stats
        if not cs.empty and cat in cs.index:
            return float(cs.at[cat, "impact_shrunk"])
        return self.base_impact_rate

    def ticker_row(self, ticker: str) -> dict:
        ts = self.ticker_stats
        if not ts.empty and ticker in ts.index:
            return ts.loc[ticker].to_dict()
        return {}

    def daily_vol(self, ticker: str) -> float:
        """The stock's normal daily swing (std of daily returns), or a typical large-cap value."""
        v = self.ticker_row(ticker).get("daily_vol")
        return float(v) if v is not None and np.isfinite(v) and v > 0 else DEFAULT_DAILY_VOL

    def features(self, sentiment, n, categories, indirect_share=None) -> tuple[np.ndarray, np.ndarray]:
        """Direction and impact feature matrices (without the bias column)."""
        s = np.asarray(sentiment, dtype=float)
        ind = np.zeros_like(s) if indirect_share is None else np.asarray(indirect_share, dtype=float)
        logn = np.log1p(np.minimum(np.asarray(n, dtype=float), self.n_cap))
        cz = np.array([self.category_z(c) for c in categories])
        ci = np.array([self.category_impact(c) for c in categories])
        return np.c_[s, cz, s * logn, s * ind], np.c_[np.abs(s), logn, ci, ind]

    # ---- prediction ----------------------------------------------------------------
    def predict(self, ticker: str, sent: float, n: int, category: str, indirect_share: float = 0.0) -> Prediction:
        x_dir, x_imp = self.features([sent], [n], [category], [indirect_share])
        p_up = float(_sigmoid(np.r_[1.0, x_dir[0]] @ self.dir_w))
        p_imp = float(_sigmoid(np.r_[1.0, x_imp[0]] @ self.imp_w))

        # Scale conviction by how reliably this stock has followed its news in the past.
        tr = self.ticker_row(ticker)
        reliability = float(np.clip(1.0 + 4.0 * (tr["hit_shrunk"] - self.pooled_hit), 0.7, 1.3)) if tr else 1.0
        p_up = 0.5 + (p_up - 0.5) * reliability
        return Prediction(float(np.clip(p_up, 0.03, 0.97)), float(np.clip(p_imp, 0.01, 0.95)), reliability)


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 2.0, iters: int = 30) -> np.ndarray:
    """L2-regularised logistic regression via Newton's method. X excludes the bias column."""
    X1 = np.c_[np.ones(len(X)), X]
    w = np.zeros(X1.shape[1])
    reg = np.eye(len(w)) * l2
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = _sigmoid(X1 @ w)
        grad = X1.T @ (p - y) + reg @ w
        hess = X1.T @ (X1 * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


# ---- statistics & training ----------------------------------------------------------------
def _category_stats(ev: pd.DataFrame, base_impact: float, k: float = 15.0) -> pd.DataFrame:
    g = ev.groupby("category")
    cs = pd.DataFrame({
        "n": g.size(),
        "mean_abnormal": g["abnormal"].mean(),
        "mean_z": g["z"].mean(),
        "impact_rate": g["impact"].mean(),
    })
    # Shrink toward neutral / the base rate so rare categories don't get extreme values.
    cs["z_shrunk"] = g["z"].sum() / (cs["n"] + k)
    cs["impact_shrunk"] = (g["impact"].sum() + k * base_impact) / (cs["n"] + k)
    hits = _sentiment_hits(ev)
    cs["sentiment_hit_rate"] = hits.groupby(ev.loc[hits.index, "category"]).mean()
    return cs.sort_values("n", ascending=False)


def _ticker_stats(ev: pd.DataFrame, rets: pd.DataFrame, abn: pd.DataFrame, k: float = 12.0) -> pd.DataFrame:
    rows = {}
    for t in abn.columns:
        e = ev[ev["ticker"] == t]
        hits = _sentiment_hits(e)
        abn_vol = float(abn[t].tail(250).std())
        rows[t] = {
            "events": len(e),
            "directional_events": len(hits),
            "hit_rate": int(hits.sum()) / len(hits) if len(hits) else np.nan,
            "hit_shrunk": _shrunk_hit_rate(hits, k),
            "news_day_abs_move": float(e["abnormal"].abs().mean()) if len(e) >= 5 else abn_vol * 1.2,
            "normal_day_abs_move": float(abn[t].tail(250).abs().mean()),
            "impact_rate": float(e["impact"].mean()) if len(e) else np.nan,
            "daily_vol": float(rets[t].tail(90).std()) if t in rets else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _fit(ev: pd.DataFrame, rets: pd.DataFrame, abn: pd.DataFrame) -> NewsModel:
    m = NewsModel(n_events=len(ev))
    if ev.empty:
        return m
    m.base_impact_rate = float(ev["impact"].mean()) if len(ev) >= 20 else m.base_impact_rate
    m.n_cap = float(max(3.0, ev["n"].quantile(0.9)))
    m.category_stats = _category_stats(ev, m.base_impact_rate)
    m.ticker_stats = _ticker_stats(ev, rets, abn)
    m.pooled_hit = _shrunk_hit_rate(_sentiment_hits(ev), k=20.0)

    y_dir = (ev["abnormal"] > 0).astype(float).values
    if len(ev) >= config.MIN_EVENTS_TO_TRAIN and 0 < y_dir.mean() < 1 and 0 < ev["impact"].mean() < 1:
        Xd, Xi = m.features(ev["sentiment"].values, ev["n"].values, ev["category"], ev["indirect_share"].values)
        m.dir_w = _fit_logistic(Xd, y_dir)
        m.imp_w = _fit_logistic(Xi, ev["impact"].values.astype(float))
        m.trained = True
        # Size the predicted-close ranges so they held the right share of past moves.
        p_imp = _sigmoid(np.c_[np.ones(len(Xi)), Xi] @ m.imp_w)
        m.range_scale = forecast.fit_scales(np.abs(ev["ret"].values), forecast.event_sigma(ev, p_imp))
    return m


def train(articles: pd.DataFrame, closes: pd.DataFrame, last_complete: pd.Timestamp,
          opens: pd.DataFrame | None = None) -> NewsModel:
    rets, abn, _, _ = abnormal_returns(closes)
    ev = build_events(articles, closes, last_complete, opens)
    model = _fit(ev, rets, abn)
    model.evaluation = evaluate_out_of_sample(ev, lambda part: _fit(part, rets, abn))
    model.events = ev
    return model
