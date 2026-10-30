import pandas as pd
import pytest

from engine import articles, companies, linking, news, sentiment


@pytest.mark.parametrize("headline, category", [
    ("Nvidia beats Q3 estimates as data-center sales soar", "earnings_beat"),
    ("Tesla cuts full-year delivery outlook", "guidance_down"),
    ("Morgan Stanley upgrades Apple to overweight", "analyst_upgrade"),
    ("Is it time to buy Apple stock?", "opinion"),
    ("Microsoft to acquire gaming studio for $2B", "mna"),
    ("FTC opens antitrust probe into Amazon", "legal_regulatory"),
    ("Apple shares are trading today", "general"),
])
def test_classify(headline, category):
    assert sentiment.classify(headline) == category


def test_score_direction():
    assert sentiment.score_text("Nvidia soars after record profit") > 0.3
    assert sentiment.score_text("Tesla plunges on fraud probe") < -0.3


def test_dominant_category_prefers_specific_events():
    df = pd.DataFrame({"category": ["general", "general", "analyst_upgrade"], "weight": [1.0, 1.0, 0.5]})
    assert sentiment.dominant_category(df) == "analyst_upgrade"
    df = pd.DataFrame({"category": ["general"] * 9 + ["product"], "weight": [1.0] * 9 + [0.1]})
    assert sentiment.dominant_category(df) == "general"   # 1% share is not meaningful


@pytest.fixture(scope="module")
def matchers():
    return linking.build_matchers({t: companies.COMPANIES[t] for t in ["JPM", "GS", "AAPL", "MSFT", "INTC"]})


@pytest.mark.parametrize("ticker, headline, relevance", [
    ("JPM", "INTC Stock Revives on JPMorgan Intel Upgrade Despite the BIS Chip Restrictions", linking.RELEVANCE_COMMENTATOR),
    ("JPM", "JPMorgan upgrades Nvidia to overweight", linking.RELEVANCE_COMMENTATOR),
    ("JPM", "JPMorgan beats estimates as trading revenue jumps", linking.RELEVANCE_HEADLINE),
    ("JPM", "Apple upgraded to buy by JPMorgan", linking.RELEVANCE_COMMENTATOR),
    ("GS", "Goldman Sachs analysts see S&P 500 at 7,500", linking.RELEVANCE_COMMENTATOR),
    ("GS", "Goldman Sachs profit tops expectations", linking.RELEVANCE_HEADLINE),
    ("JPM", "JPMorgan raises Apple's price target to $300", linking.RELEVANCE_COMMENTATOR),
    ("AAPL", "Layoffs at Apple hit services unit", linking.RELEVANCE_HEADLINE),
    ("MSFT", "Activision acquired by Microsoft in record deal", linking.RELEVANCE_HEADLINE),
    ("AAPL", "Stocks rise, according to Apple supplier data", linking.RELEVANCE_COMMENTATOR),
    ("AAPL", "Shares of (NASDAQ: AAPL) edge higher", linking.RELEVANCE_HEADLINE),
    ("AAPL", "Oil prices climb", 0.0),
])
def test_relevance(matchers, ticker, headline, relevance):
    assert matchers[ticker].relevance(headline, "") == relevance


def test_summary_only_mention(matchers):
    assert matchers["AAPL"].relevance("Tech stocks rally", "Apple led gains.") == linking.RELEVANCE_SUMMARY


def test_annotate_applies_low_signal_weight():
    df = pd.DataFrame([
        {"ticker": "AAPL", "title": "Apple beats estimates", "summary": "", "category": "earnings_beat"},
        {"ticker": "AAPL", "title": "Is it time to buy Apple?", "summary": "", "category": "opinion"},
        {"ticker": "AAPL", "title": "Chip stocks slide", "summary": None, "category": "general"},
    ])
    out = articles.annotate_articles(df.assign(sentiment=0.5), {"AAPL": ["Apple"]})
    assert out["relevance"].tolist() == pytest.approx([1.0, 0.4, linking.RELEVANCE_TICKER_FEED])


def test_dedupe_keeps_earliest_time_and_strongest_link():
    t0, t1 = pd.Timestamp("2026-10-20 10:00", tz="UTC"), pd.Timestamp("2026-10-20 12:00", tz="UTC")
    arts = [{"id": "x", "ticker": "AAPL", "published": t1, "relevance": 1.0},
            {"id": "x", "ticker": "AAPL", "published": t0, "relevance": 0.45}]
    (a,) = news._dedupe(arts)
    assert a["published"] == t0 and a["relevance"] == 1.0
