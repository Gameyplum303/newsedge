"""Colours and chart styling shared across the dashboard."""
import plotly.graph_objects as go

from engine.sentiment import NEGATIVE, POSITIVE

UP = "#22c55e"
DOWN = "#ef4444"
FLAT = "#94a3b8"
ACCENT = "#60a5fa"
WARN = "#f59e0b"

MARKET_STATE_COLORS = {"open": UP, "pre-market": WARN, "after-hours": WARN, "closed": DOWN}


def sentiment_color(score: float) -> str:
    return UP if score >= POSITIVE else DOWN if score <= NEGATIVE else FLAT


def style_figure(fig: go.Figure, height: int, **layout) -> go.Figure:
    """Dark, transparent plotly layout that sits cleanly on the Streamlit dark theme."""
    top = 30 if "title" in layout else 10
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=top, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **layout)
    return fig
