"""Shared fixtures: a synthetic market where news sentiment genuinely moves prices."""
import numpy as np
import pandas as pd
import pytest

from engine.market_clock import ET

STRENGTH = {"earnings_beat": 0.03, "analyst_downgrade": 0.025, "general": 0.006, "product": 0.01}


@pytest.fixture(autouse=True)
def no_alpaca_keys(monkeypatch):
    """Tests run offline: ignore any Alpaca keys saved on this machine."""
    from engine import alpaca
    monkeypatch.setattr(alpaca, "credentials", lambda: None)


def make_article(ticker, published, sentiment, category="general", title=None, **extra):
    return {"id": f"{ticker}-{published}-{title}", "ticker": ticker, "title": title or f"{ticker} news",
            "summary": "", "link": "", "source": "CNBC", "published": pd.Timestamp(published),
            "sentiment": sentiment, "category": category, "relevance": 1.0, "source_weight": 1.0, **extra}


@pytest.fixture(scope="session")
def synthetic_market():
    """Two years of prices for AAA/BBB/CCC + SPY, and ~30% of days with news that moves the stock."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-10-01", "2026-10-19")
    mkt = rng.normal(0.0004, 0.01, len(days))
    rets, rows = {"SPY": mkt}, []
    for t in ["AAA", "BBB", "CCC"]:
        effect = np.zeros(len(days))
        for i, day in enumerate(days):
            if rng.random() >= 0.3:
                continue
            s = float(np.clip(rng.normal(0, 0.45), -1, 1))
            cat = rng.choice(list(STRENGTH))
            effect[i] = s * STRENGTH[cat]
            for j in range(rng.integers(1, 6)):
                pub = pd.Timestamp(day).tz_localize(ET) + pd.Timedelta(hours=8 + j)   # before the close
                rows.append(make_article(t, pub.tz_convert("UTC"), s + rng.normal(0, 0.1), cat, title=f"{t} {i} {j}"))
        rets[t] = 1.1 * mkt + rng.normal(0, 0.015, len(days)) + effect
    closes = (1 + pd.DataFrame(rets, index=days)).cumprod() * 100
    return pd.DataFrame(rows), closes


@pytest.fixture(scope="session")
def trained_model(synthetic_market):
    from engine.model import train
    articles, closes = synthetic_market
    return train(articles, closes, closes.index[-1])
