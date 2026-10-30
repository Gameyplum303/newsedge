"""Plotly figures used by the views."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from engine.market_clock import ET
from engine.sentiment import CATEGORY_LABELS
from .theme import ACCENT, DOWN, UP, sentiment_color, style_figure


def price_with_news(bars: pd.DataFrame, articles: pd.DataFrame, ticker: str) -> go.Figure:
    """Intraday price line with a marker at each article's publish time, coloured by sentiment."""
    fig = go.Figure(go.Scatter(x=bars.index.tz_convert(ET), y=bars["Close"], mode="lines",
                               line=dict(color=ACCENT, width=1.6), name=ticker,
                               hovertemplate="%{x|%b %d %I:%M %p}<br>$%{y:,.2f}<extra></extra>"))
    a = articles[articles["published"] >= bars.index[0]]
    if not a.empty:
        pos = np.clip(bars.index.searchsorted(a["published"]), 0, len(bars) - 1)
        fig.add_trace(go.Scatter(
            x=a["published"].dt.tz_convert(ET), y=bars["Close"].values[pos], mode="markers",
            marker=dict(size=10, color=[sentiment_color(s) for s in a["sentiment"]], line=dict(width=1, color="white")),
            # Hover text is rendered as HTML: escape headlines so a feed can't inject tags or links.
            text=[f"{html.escape(t[:90])}<br>{html.escape(str(src))} · sentiment {s:+.2f}"
                  for t, src, s in zip(a["title"], a["source"], a["sentiment"])],
            hovertemplate="%{text}<extra></extra>", name="Articles"))
    return style_figure(fig, 300, showlegend=False,
                        xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[20, 4], pattern="hour")]))


def target_tradeoff(grid: pd.DataFrame, ticker: str, side: int) -> go.Figure:
    """Win probability and expected return across the tested profit targets."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid["target_pct"] * 100, y=grid["p_target"] * 100, name="Win probability (%)",
                             mode="lines+markers", line=dict(color=ACCENT)))
    fig.add_trace(go.Scatter(x=grid["target_pct"] * 100, y=grid["expected_return"] * 100, name="Expected return (%)",
                             mode="lines+markers", line=dict(color=UP), yaxis="y2"))
    return style_figure(
        fig, 340,
        title=f"{ticker} {'long' if side == 1 else 'short'}: win probability vs. expected return by target",
        xaxis_title="Profit target (%)", yaxis=dict(title="Win probability (%)", range=[0, 100]),
        yaxis2=dict(title="Expected return (%)", overlaying="y", side="right", zeroline=True),
        legend=dict(orientation="h", y=-0.25))


def news_events(events: pd.DataFrame) -> go.Figure:
    """Each past news event: weighted sentiment vs the tradeable move that followed."""
    fig = go.Figure(go.Scatter(
        x=events["sentiment"], y=events["abnormal"] * 100, mode="markers",
        marker=dict(size=np.clip(np.log1p(events["n"]) * 5, 4, 18), opacity=0.6,
                    color=np.where(events["abnormal"] > 0, UP, DOWN)),
        text=[f"{t} {d:%Y-%m-%d}<br>{CATEGORY_LABELS.get(c, c)} · {n} articles"
              for t, d, c, n in zip(events["ticker"], events["date"], events["category"], events["n"])],
        hovertemplate="%{text}<br>sentiment %{x:+.2f}<br>abnormal move %{y:+.2f}%<extra></extra>"))
    return style_figure(fig, 380, title="Past news events: headline sentiment vs. the move you could have traded",
                        xaxis_title="Weighted sentiment", yaxis_title="Abnormal move (%)")
