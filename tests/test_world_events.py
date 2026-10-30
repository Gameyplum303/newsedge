from types import SimpleNamespace

import pandas as pd
import pytest

from engine import articles, linking, news, sentiment

WATCH = {"AAPL": ["Apple"], "NVDA": ["Nvidia"], "AMZN": ["Amazon"], "XOM": ["Exxon"], "BA": ["Boeing"],
         "PFE": ["Pfizer"], "JPM": ["JPMorgan"]}


@pytest.mark.parametrize("headline, category", [
    ("Magnitude 7.4 earthquake strikes eastern Taiwan", "disaster"),
    ("Hurricane forces evacuations along the Gulf Coast", "disaster"),
    ("Train derails in Ohio, releasing toxic chemicals", "accident"),
    ("Explosion at Texas refinery injures workers", "accident"),
    ("AWS outage knocks major websites offline", "cyber_outage"),
    ("Ransomware attack hits hospital chain", "cyber_outage"),
    ("Missile strikes hit oil tankers in the Red Sea", "geopolitical"),
    ("Dockworkers strike shuts East Coast ports", "labor"),
    ("WHO declares bird flu outbreak a public health emergency", "health"),
    ("Apple beats estimates on strong iPhone sales", "earnings_beat"),   # business rules still win
    ("Nvidia fires back at critics of its AI chips", "general"),        # "fires" is not a fire
])
def test_world_event_classification(headline, category):
    assert sentiment.classify(headline) == category


def test_world_events_score_negative():
    assert sentiment.score_text("Deadly earthquake and tsunami devastate coastal towns") < -0.5


@pytest.fixture(scope="module")
def matchers():
    return linking.build_matchers(WATCH)


def test_product_aliases_link_directly(matchers):
    assert matchers["AMZN"].relevance("AWS outage knocks websites offline", "") == linking.RELEVANCE_HEADLINE
    assert matchers["BA"].relevance("FAA orders new inspections of 737 MAX jets", "") == linking.RELEVANCE_HEADLINE
    assert matchers["AAPL"].relevance("iPhone factory hit by power cuts", "") == linking.RELEVANCE_HEADLINE


@pytest.mark.parametrize("ticker, headline, category, exposure, sign", [
    ("NVDA", "Magnitude 7.4 earthquake strikes eastern Taiwan", "disaster", "Taiwan chip supply", 1),
    ("AAPL", "Magnitude 7.4 earthquake strikes eastern Taiwan", "disaster", "Taiwan chip supply", 1),
    ("XOM", "Missile strikes hit oil tankers in the Red Sea", "geopolitical", "Middle East oil supply", -1),
    ("BA", "Passenger jet makes emergency landing after engine failure", "accident", "Commercial aviation safety", 1),
    ("PFE", "WHO declares bird flu outbreak a public health emergency", "health", "Disease outbreak", -1),
])
def test_exposure_links(matchers, ticker, headline, category, exposure, sign):
    lk = linking.link(matchers[ticker], ticker, headline, "", category)
    assert (lk.relevance, lk.via, lk.sign, lk.kind) == (linking.RELEVANCE_EXPOSURE, exposure, sign, "exposure")


def test_exposure_needs_matching_event_type(matchers):
    # Taiwan in a business headline is not a supply-chain shock.
    assert linking.link(matchers["NVDA"], "NVDA", "Taiwan's central bank holds rates", "", "macro").relevance == 0
    assert linking.link(matchers["JPM"], "JPM", "Magnitude 7.4 earthquake strikes eastern Taiwan", "", "disaster").relevance == 0


def test_broad_tagger_links_world_event_to_exposed_watchlist_stocks(matchers):
    tags = dict(news._broad_tagger(matchers)("Magnitude 7.4 earthquake strikes eastern Taiwan", "", "disaster"))
    assert tags == {"AAPL": linking.RELEVANCE_EXPOSURE, "NVDA": linking.RELEVANCE_EXPOSURE}


def test_annotate_flips_sentiment_for_beneficiaries_and_is_idempotent():
    df = pd.DataFrame([
        {"ticker": "XOM", "title": "Missile strikes hit oil tankers in the Red Sea", "summary": "",
         "category": "geopolitical", "sentiment": -0.6},
        {"ticker": "NVDA", "title": "Magnitude 7.4 earthquake strikes eastern Taiwan", "summary": "",
         "category": "disaster", "sentiment": -0.6},
    ])
    once = articles.annotate_articles(df, WATCH)
    assert once["sentiment"].tolist() == [0.6, -0.6]          # oil producer benefits, chip maker is hurt
    assert once["via"].tolist() == ["Middle East oil supply", "Taiwan chip supply"]
    twice = articles.annotate_articles(once, WATCH)
    assert twice["sentiment"].tolist() == [0.6, -0.6]


def test_minor_disaster_alerts_are_skipped():
    feed = SimpleNamespace(entries=[{"title": "Green earthquake (Magnitude 4.6M) in Taiwan"},
                                    {"title": "Red earthquake (Magnitude 7.4M) in Taiwan"}])
    tagger = news._broad_tagger(linking.build_matchers(WATCH))
    arts = news._entries_to_articles(feed, "GDACS", tagger, pd.Timestamp.now(tz="UTC").to_pydatetime(), None,
                                     news._SKIP["GDACS disasters"])
    assert {a["title"] for a in arts} == {"Red earthquake (Magnitude 7.4M) in Taiwan"}


def test_world_events_come_from_the_headline_only():
    assert sentiment.classify("How major US stock indexes fared Tuesday", "Stocks fell as war fears grew.") != "geopolitical"
    assert sentiment.classify("Explosion at Texas refinery", "") == "accident"
