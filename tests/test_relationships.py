import numpy as np
import pandas as pd
import pytest

from engine import articles, linking, news, profiles, sentiment
from engine.events import build_events
from engine.model import NewsModel

WATCH = {"LMT": ["Lockheed Martin", "Lockheed"], "MSFT": ["Microsoft"], "UNH": ["UnitedHealth"],
         "NVDA": ["Nvidia"], "AMD": ["AMD"], "AAPL": ["Apple"]}


@pytest.fixture(scope="module")
def m():
    return linking.build_matchers(WATCH)


def _link(m, ticker, headline):
    return linking.link(m[ticker], ticker, headline, "", sentiment.classify(headline))


@pytest.mark.parametrize("headline, category", [
    ("Lockheed Martin wins $2 billion Army missile contract", "government_contract"),
    ("Pentagon awards Palantir AI targeting contract", "government_contract"),
    ("Government shutdown enters its second week", "government_policy"),
    ("Medicare cuts payments to private insurers", "government_policy"),
    ("Company announces budget cuts and layoffs", "layoffs"),        # not a government budget story
])
def test_government_classification(headline, category):
    assert sentiment.classify(headline) == category


def test_war_news_is_positive_for_defense_contractors_scaled_by_exposure(m):
    lmt = _link(m, "LMT", "Russia launches missile strikes on Kyiv")
    msft = _link(m, "MSFT", "Russia launches missile strikes on Kyiv")
    assert (lmt.kind, lmt.sign, lmt.relevance) == ("government", -1, pytest.approx(linking.RELEVANCE_GOVERNMENT))
    assert msft.sign == -1 and msft.relevance == pytest.approx(linking.RELEVANCE_GOVERNMENT * 0.1)
    assert _link(m, "AAPL", "Russia launches missile strikes on Kyiv").relevance == 0   # no defense business


def test_budget_and_policy_news_reads_as_is_for_government_dependent_firms(m):
    unh = _link(m, "UNH", "Medicare cuts payments to private insurers")
    assert (unh.kind, unh.sign, unh.via) == ("government", 1, "government healthcare programs")
    assert unh.relevance == pytest.approx(linking.RELEVANCE_GOVERNMENT * 0.5)
    shutdown = _link(m, "LMT", "Government shutdown enters its second week")
    assert shutdown.sign == 1 and shutdown.relevance > 0


def test_a_rivals_contract_win_is_not_good_news(m):
    lk = _link(m, "LMT", "Pentagon awards Anduril $1 billion drone contract")
    assert (lk.kind, lk.sign, lk.via) == ("relation", -1, "Anduril (competitor)")


def test_supplier_and_competitor_news(m):
    supplier = _link(m, "NVDA", "TSMC beats estimates as AI chip demand soars")
    assert (supplier.kind, supplier.sign, supplier.via) == ("relation", 1, "TSMC (supplier)")
    assert supplier.relevance == pytest.approx(linking.RELEVANCE_RELATION * 0.8)
    rival = _link(m, "AMD", "Nvidia beats estimates on record data center sales")
    assert (rival.sign, rival.via) == (-1, "Nvidia (competitor)")
    assert _link(m, "NVDA", "TSMC shares edge higher").relevance == 0   # too vague to pass on


def test_strongest_indirect_link_wins(m):
    headline = "Earthquake halts TSMC chip plants in Taiwan"
    assert _link(m, "NVDA", headline).via == "TSMC (supplier)"        # 0.5 x 0.8 beats exposure 0.35
    assert _link(m, "AAPL", headline).via == "Taiwan chip supply"     # 0.5 x 0.6 loses to exposure 0.35


def test_direct_mention_beats_any_indirect_link(m):
    assert _link(m, "LMT", "Lockheed Martin wins $2 billion Army missile contract").kind == "direct"


def test_annotate_flips_competitor_news_and_labels_links():
    df = pd.DataFrame([{"ticker": "AMD", "title": "Nvidia beats estimates on record data center sales", "summary": "",
                        "category": "earnings_beat", "sentiment": 0.6}])
    out = articles.annotate_articles(df, WATCH)
    assert out[["link_kind", "via"]].iloc[0].tolist() == ["relation", "Nvidia (competitor)"]
    assert out["sentiment"].iloc[0] == pytest.approx(-0.6) and out["raw_sentiment"].iloc[0] == 0.6


def test_related_search_terms():
    terms = profiles.related_terms("LMT")
    assert "Anduril" in terms and "Pentagon" in terms
    assert news._google_query("ZZZZ", ["Nobody"], "when:3d", kind="related") is None


def test_model_treats_indirect_news_more_cautiously_by_default():
    model = NewsModel()
    direct = model.predict("X", 0.6, 5, "general", indirect_share=0.0)
    indirect = model.predict("X", 0.6, 5, "general", indirect_share=1.0)
    assert direct.p_up > indirect.p_up > 0.5
    assert direct.p_impact > indirect.p_impact


def test_events_record_indirect_share():
    days = pd.bdate_range("2026-09-01", periods=40)
    closes = pd.DataFrame({"SPY": np.linspace(100, 104, 40), "AAA": np.linspace(50, 55, 40)}, index=days)
    pub = pd.Timestamp(days[30]).tz_localize("America/New_York") + pd.Timedelta(hours=10)
    arts = pd.DataFrame([
        {"ticker": "AAA", "published": pub.tz_convert("UTC"), "sentiment": 0.5, "category": "general",
         "relevance": 1.0, "source_weight": 1.0, "link_kind": "direct"},
        {"ticker": "AAA", "published": pub.tz_convert("UTC"), "sentiment": 0.5, "category": "general",
         "relevance": 1.0, "source_weight": 1.0, "link_kind": "relation"},
    ])
    ev = build_events(arts, closes, days[-1])
    assert ev["indirect_share"].tolist() == [pytest.approx(0.5)]


def test_mentioning_weapons_is_not_war_news():
    assert sentiment.classify("Robot factories and hypersonic missiles: a bank's $5 billion bet") != "geopolitical"
    assert sentiment.classify("Russia launches missile strikes on Kyiv") == "geopolitical"
