"""Text formatting helpers for the dashboard."""
import pandas as pd

from engine.strategy import PlanSettings

PREDICTION_LABELS = {"UP": "▲ Increase", "DOWN": "▼ Decrease"}
NO_PREDICTION = "● No clear move"


def esc(text: str) -> str:
    """Stop Streamlit markdown from treating dollar amounts as LaTeX."""
    return text.replace("$", "\\$")


def direction_md(direction: str) -> str:
    return {"UP": ":green[**▲ UP**]", "DOWN": ":red[**▼ DOWN**]"}.get(direction, ":gray[**● NEUTRAL**]")


def as_percent(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Copy with fractional columns scaled to percent (Streamlit formats "%.1f%%" literally)."""
    out = df.copy()
    present = [c for c in columns if c in out]
    out[present] = out[present] * 100
    return out


def range_text(s: PlanSettings) -> str:
    return f"+{s.min_target:.1%} to +{s.max_target:.1%}"


def stop_text(s: PlanSettings) -> str:
    return f"a −{s.stop:.1%} stop" if s.stop_ratio is None else f"a stop of {s.stop_ratio:g}× the target"
