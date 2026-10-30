"""Turn the latest articles into a per-ticker signal with confidence and human-readable reasoning."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from . import companies, config, profiles, sentiment
from .linking import INDIRECT_LINKS
from .model import NewsModel, Prediction
from .sanitize import markdown_literal
from .sentiment import CATEGORY_LABELS, dominant_category

NEUTRAL_BELOW = 0.53   # direction confidence under this is reported as no clear move


@dataclass
class Signal:
    ticker: str
    n_articles: int
    sentiment: float
    category: str
    p_up: float
    p_impact: float
    direction: str            # "UP", "DOWN" or "NEUTRAL"
    confidence: float         # probability the direction call is right
    expected_move: float      # expected signed abnormal move from the next tradeable entry to the following close
    typical_news_move: float
    reasons: list[str] = field(default_factory=list)
    articles: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_vol: float = 0.02   # the stock's normal daily swing (sizes the predicted-close ranges)


@dataclass
class _Evidence:
    """What a signal was built from; the reasoning text is written from this."""
    ticker: str
    articles: pd.DataFrame       # weighted articles in the signal window
    sentiment: float
    category: str
    prediction: Prediction
    stats: dict                  # the model's statistics for this ticker ({} if none yet)
    typical_move: float
    expected_move: float
    indirect_share: float


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


def build(articles: pd.DataFrame, model: NewsModel, tickers: list[str], now: datetime) -> list[Signal]:
    window_start = now - timedelta(hours=config.SIGNAL_WINDOW_HOURS)
    signals = []
    for t in tickers:
        a = articles[(articles["ticker"] == t) & (articles["published"] >= window_start)].copy()
        tr = model.ticker_row(t)
        typical = float(tr.get("news_day_abs_move", 0.015) or 0.015)
        daily_vol = model.daily_vol(t)

        if a.empty:
            signals.append(Signal(t, 0, 0.0, "general", 0.5, model.base_impact_rate, "NEUTRAL", 0.5, 0.0,
                                  typical, [f"No {t} articles in the last {config.SIGNAL_WINDOW_HOURS}h, so there is no news signal."],
                                  daily_vol=daily_vol))
            continue

        age_h = (now - a["published"]).dt.total_seconds().clip(lower=0) / 3600
        a["weight"] = 0.5 ** (age_h / config.RECENCY_HALF_LIFE_HOURS) * a["relevance"] * a["source_weight"]
        total_w = a["weight"].sum()
        sent = float((a["weight"] * a["sentiment"]).sum() / total_w) if total_w > 0 else 0.0
        a["influence"] = a["weight"] * a["sentiment"].abs()
        a["influence"] = a["influence"] / max(a["influence"].sum(), 1e-9)

        category = dominant_category(a)
        indirect = a["link_kind"].isin(INDIRECT_LINKS) if "link_kind" in a else pd.Series(False, index=a.index)
        indirect_share = float(a.loc[indirect, "weight"].sum() / total_w) if total_w > 0 else 0.0

        n = len(a)
        pred = model.predict(t, sent, n, category, indirect_share)
        p_up, p_imp = pred.p_up, pred.p_impact
        conf = max(p_up, 1 - p_up)
        direction = "NEUTRAL" if conf < NEUTRAL_BELOW else ("UP" if p_up > 0.5 else "DOWN")
        expected = (2 * p_up - 1) * typical

        reasons = _reasons(_Evidence(t, a, sent, category, pred, tr, typical, expected, indirect_share), model)
        signals.append(Signal(t, n, sent, category, p_up, p_imp, direction, conf, expected, typical,
                              reasons, a.sort_values("influence", ascending=False), daily_vol))
    return sorted(signals, key=lambda s: (s.direction == "NEUTRAL", -s.confidence * (0.5 + s.p_impact)))


def _relationship_reasons(ev: _Evidence) -> list[str]:
    out = []
    areas = profiles.government(ev.ticker)
    if areas:
        def level(w):
            return "high" if w >= 0.6 else "moderate" if w >= 0.2 else "low"
        desc = ", ".join(f"{companies.GOVERNMENT_AREA_LABELS[k]} ({level(w)})"
                         for k, w in sorted(areas.items(), key=lambda kv: -kv[1]))
        note = (" War and conflict news counts as a positive for it; ceasefires and defense cuts as negatives."
                if areas.get("defense", 0) >= 0.2 else "")
        out.append(f"🏛️ **Government business:** {desc}.{note}")
    if "via" in ev.articles:
        via = ev.articles.loc[ev.articles["via"] != "", "via"].value_counts()
        if not via.empty:
            listed = ", ".join(f"{markdown_literal(name)} ×{n}" for name, n in via.head(5).items())
            out.append(f"🔗 **{via.sum()} indirect article(s)**, {ev.indirect_share:.0%} of today's news weight: {listed}. "
                       "The model weighs indirect news by how much it has moved prices in the past.")
    return out


def _reasons(ev: _Evidence, model: NewsModel) -> list[str]:
    t, a, sent, category, pred, tr = ev.ticker, ev.articles, ev.sentiment, ev.category, ev.prediction, ev.stats
    pos = int((a["sentiment"] >= sentiment.POSITIVE).sum())
    neg = int((a["sentiment"] <= sentiment.NEGATIVE).sum())
    neu = len(a) - pos - neg
    srcs = a["source"].nunique()
    r = [f"**{len(a)} article(s)** from {srcs} source(s) in the last {config.SIGNAL_WINDOW_HOURS}h "
         f"({pos} positive, {neg} negative, {neu} neutral). Recency- and source-weighted sentiment is "
         f"**{sent:+.2f}** ({sentiment.label(sent)})."]

    world = a[a["category"].isin(sentiment.WORLD_EVENT_CATEGORIES)]
    if not world.empty:
        kinds = ", ".join(CATEGORY_LABELS[c].lower() for c in world["category"].value_counts().index[:3])
        r.append(f"🌍 **{len(world)} world-event article(s)** ({kinds}).")
    r += _relationship_reasons(ev)

    top = a.sort_values("influence", ascending=False).iloc[0]
    r.append(f"Most influential headline: *\"{markdown_literal(top['title'])}\"* "
             f"({markdown_literal(top['source'])}, sentiment {top['sentiment']:+.2f}).")

    cs = model.category_stats
    lbl = CATEGORY_LABELS.get(category, category)
    if not cs.empty and category in cs.index and cs.at[category, "n"] >= 5:
        row = cs.loc[category]
        hit = row.get("sentiment_hit_rate")
        hit_txt = f", and the move matched headline sentiment {hit:.0%} of the time" if pd.notna(hit) else ""
        r.append(f"Dominant event type: **{lbl}**. In {int(row['n'])} past events of this type, stocks had an "
                 f"average abnormal move of {_pct(row['mean_abnormal'], 2)}{hit_txt}.")
    else:
        prior = sentiment.CATEGORY_PRIOR.get(category, 0)
        lean = "upward" if prior > 0.2 else "downward" if prior < -0.2 else "no clear"
        r.append(f"Dominant event type: **{lbl}** (not enough history yet; prior assumption: {lean} bias).")

    if tr and tr.get("directional_events", 0) >= 5:
        r.append(f"Historically {t} has moved in the direction of its news sentiment "
                 f"**{tr['hit_rate']:.0%}** of the time ({int(tr['directional_events'])} news days) vs "
                 f"{model.pooled_hit:.0%} across the watchlist, so confidence is scaled ×{pred.reliability:.2f}.")
    if tr:
        r.append(f"Typical move for {t} after news, from the first price you could trade at to the next close: "
                 f"**±{ev.typical_move:.1%}** vs the market. Expected move now: **{_pct(ev.expected_move, 2)}**, "
                 f"before {config.ROUND_TRIP_COST:.2%} trading costs.")
    r.append(f"Probability of an outsized move (beyond {config.IMPACT_Z}σ vs the market): **{pred.p_impact:.0%}** "
             f"(base rate {model.base_impact_rate:.0%}).")
    if not model.trained:
        r.append(f"⚠️ Model is running on default weights ({model.n_events} historical news events so far). "
                 "Use **Build history** in the sidebar to calibrate it on real past articles and price moves.")
    return r
