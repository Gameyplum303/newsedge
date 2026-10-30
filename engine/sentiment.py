"""Headline sentiment (VADER + finance lexicon) and market-event classification."""
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Bump when the lexicon or EVENT_RULES change: stored articles are then re-scored (articles.rescore_stored_articles).
SCORER_VERSION = "2026-10-20.government-2"

# Finance-specific word scores layered over VADER's general lexicon (scale -4..+4).
FINANCE_LEXICON = {
    "beat": 2.0, "beats": 2.0, "tops": 1.8, "topped": 1.8, "exceeds": 1.8, "surpasses": 1.8,
    "miss": -2.0, "misses": -2.0, "missed": -2.0, "shortfall": -2.0,
    "upgrade": 2.3, "upgrades": 2.3, "upgraded": 2.3,
    "downgrade": -2.3, "downgrades": -2.3, "downgraded": -2.3,
    "outperform": 1.8, "overweight": 1.5, "underperform": -1.8, "underweight": -1.5,
    "surge": 2.5, "surges": 2.5, "soar": 2.6, "soars": 2.6, "jumps": 2.0, "jump": 1.8,
    "rally": 2.0, "rallies": 2.0, "climbs": 1.6, "gains": 1.5, "rebound": 1.5, "rebounds": 1.5,
    "plunge": -3.0, "plunges": -3.0, "tumble": -2.6, "tumbles": -2.6, "slump": -2.4,
    "slumps": -2.4, "sinks": -2.2, "slides": -1.8, "falls": -1.6, "drops": -1.6, "crash": -3.2,
    "selloff": -2.4, "sell-off": -2.4, "plummets": -3.0, "tanks": -2.8,
    "bullish": 2.2, "bearish": -2.2, "record": 1.4, "strong": 1.4, "robust": 1.5, "weak": -1.8,
    "weaker": -1.8, "soft": -1.0, "headwinds": -1.5, "tailwinds": 1.5, "growth": 1.0,
    "raises": 1.2, "raised": 1.2, "lowers": -1.4, "lowered": -1.4, "cuts": -1.4, "slashes": -2.2,
    "buyback": 1.6, "repurchase": 1.4, "dividend": 0.8,
    "approval": 2.0, "approved": 2.0, "approves": 2.0, "rejected": -2.2, "rejects": -2.2,
    "lawsuit": -1.8, "sued": -1.8, "sues": -1.4, "probe": -1.8, "investigation": -1.8,
    "antitrust": -1.4, "subpoena": -2.0, "recall": -2.0, "fraud": -3.4, "default": -2.6,
    "bankruptcy": -3.6, "delisting": -3.0, "halted": -2.0, "layoffs": -1.0, "restructuring": -0.8,
    "volatile": -0.8, "uncertainty": -1.2, "warns": -2.0, "warning": -1.8, "profit": 1.2,
    "loss": -1.6, "losses": -1.6, "outage": -1.8, "breach": -2.2, "hack": -2.2,
    # World events: disasters, accidents, conflict, disruption.
    "earthquake": -2.5, "quake": -2.3, "hurricane": -2.2, "typhoon": -2.2, "cyclone": -2.0, "tornado": -2.2,
    "flood": -2.0, "flooding": -2.0, "wildfire": -2.3, "tsunami": -2.8, "eruption": -2.0, "landslide": -2.2,
    "explosion": -2.6, "blast": -2.2, "derailment": -2.6, "derailed": -2.4, "collapse": -2.4,
    "spill": -2.0, "leak": -1.6, "grounded": -2.0, "evacuation": -1.8, "evacuated": -1.8,
    "cyberattack": -2.7, "ransomware": -2.8, "hacked": -2.4, "disruption": -1.6, "shutdown": -1.6,
    "invasion": -2.8, "missile": -2.0, "airstrike": -2.6, "sanctions": -1.6, "embargo": -1.8, "blockade": -2.0,
    "outbreak": -2.2, "pandemic": -2.4, "epidemic": -2.2, "walkout": -1.6, "ceasefire": 1.2,
}

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(FINANCE_LEXICON)

# (category, label, pattern, prior direction -1..+1). Earlier rules win.
EVENT_RULES = [
    ("earnings_beat", "Earnings beat",
     r"\b(beats?|tops?|topped|exceed(s|ed)?|surpass(es|ed)?|crush(es|ed)?)\b.{0,60}\b(estimates?|expectations|forecasts?|consensus|views?)\b|\bearnings beat\b", 1.0),
    ("earnings_miss", "Earnings miss",
     r"\b(miss(es|ed)?|falls? short|fell short|below|lags?)\b.{0,60}\b(estimates?|expectations|forecasts?|consensus|views?)\b|\bearnings miss\b", -1.0),
    ("guidance_up", "Guidance raised",
     r"\b(raises?|raised|boosts?|lifts?|hikes?|ups)\b.{0,40}\b(guidance|outlook|forecasts?|targets?)\b", 1.0),
    ("guidance_down", "Guidance cut",
     r"\b(cuts?|lowers?|lowered|slash(es|ed)?|reduces?|trims?|withdraws?)\b.{0,40}\b(guidance|outlook|forecasts?)\b|\b(weak|disappointing|soft)\b.{0,15}\b(guidance|outlook|forecast)\b", -1.0),
    ("analyst_upgrade", "Analyst upgrade / PT raise",
     r"\bupgrad(e|es|ed)\b|\b(raises?|lifts?|hikes?|boosts?)\b.{0,25}\bprice target\b|\bprice target\b.{0,15}\b(raised|hiked|lifted|boosted)\b|\binitiat\w+\b.{0,30}\b(buy|outperform|overweight)\b", 1.0),
    ("analyst_downgrade", "Analyst downgrade / PT cut",
     r"\bdowngrad(e|es|ed)\b|\b(cuts?|lowers?|slash(es)?|trims?)\b.{0,25}\bprice target\b|\bprice target\b.{0,15}\b(cut|lowered|slashed|trimmed)\b", -1.0),
    ("opinion", "Opinion / listicle",
     r"\?\s*$|^\s*\d+\s+(reasons|stocks|things|ways)\b|\b(should you buy|is it time to|better buy|buy now\??|millionaire|forever stock|no-brainer)\b", 0.0),
    # Government business. Before world events so "Army awards missile contract" is a contract, not a war.
    ("government_contract", "Government contract",
     r"\b(award(s|ed)?|wins?|won|secures?|secured|lands?|landed|gets?)\b.{0,50}\b(contracts?|deals?|orders?)\b.{0,50}\b(Pentagon|Army|Navy|Air Force|Space Force|Marines|Department of (Defense|War)|DoD|NASA|federal|government|Homeland Security|DHS|Veterans Affairs|NATO)\b|\b(award(s|ed)?|wins?|won|secures?|secured|lands?|landed|gets?)\b.{0,50}\b(Pentagon|Army|Navy|Air Force|Space Force|Marines|DoD|NASA|federal|government|NATO)\b.{0,40}\b(contracts?|deals?|orders?)\b|\b(Pentagon|Army|Navy|Air Force|Space Force|DoD|NASA|federal government)\b.{0,30}\b(awards?|picks|selects?|taps)\b|\bgovernment contracts?\b", 0.8),
    ("government_policy", "Government budget / policy",
     r"\b(government shutdown|shutdown ends|federal budget|spending bill|continuing resolution|debt ceiling|appropriations|defen[cs]e (budget|spending|bill)|federal (budget |spending )?cuts|executive order|Medicare (rates|payments|cuts)|drug price negotiations?)\b", 0.0),
    # World events. Placed before business categories so "lawsuit over the crash" counts as an accident.
    ("disaster", "Natural disaster",
     r"\b(earthquakes?|quakes?|hurricanes?|typhoons?|cyclones?|tornado(es)?|flood(s|ing|ed)?|wildfires?|tsunami|volcan\w*|eruption|landslides?|drought|heat ?wave|blizzard|winter storm|tropical storm)\b", -0.5),
    ("accident", "Accident / industrial incident",
     r"\b(plane|jet|helicopter|train|bus|car|vehicle|truck)\s+crash(es|ed)?\b|\bcrash(es|ed)? (kills|killed|landing)\b|\bderail\w*|\bexplo(sion|sions|ded|des)\b|\bblasts?\b|\b(caught|catches) fire\b|\bfire (at|in|breaks out|engulfs|destroys)\b|\b(factory|plant|warehouse|refinery|mine) (fire|blaze|accident)\b|\bblaze\b|\bcollaps(e|es|ed)\b|\b(oil|chemical) (spill|leak)\b|\bgas leak\b|\bemergency landing\b|\bgrounds? (its |the )?(fleet|planes|jets|aircraft)\b|\bfatal\b", -0.7),
    ("cyber_outage", "Outage / cyberattack",
     r"\b(outages?|cyber ?attacks?|hack(ed|ers?)?|ransomware|data breach|security breach|ddos|went down|service disruption)\b", -0.7),
    ("geopolitical", "War / geopolitical",
     r"\b(wars?|invasion|invad\w+|missile (strikes?|attacks?|barrage|launch(es)?)|airstrikes?|military (strike|action|escalation)|sanctions|embargo|coup|terror\w*|ceasefire|blockade|armed conflict|troops)\b", -0.3),
    ("labor", "Strike / labor action",
     r"\b(workers|union|employees|staff|pilots|dockworkers)\b.{0,20}\b(strike|walk ?out|walk off)\b|\bon strike\b|\bstrike (ends|begins|vote|deadline)\b|\bwalkout\b|\bpicket\w*", -0.5),
    ("health", "Outbreak / health emergency",
     r"\b(pandemic|outbreak|epidemic|ebola|bird flu|H5N1|covid|measles|mpox|cholera|public health emergency)\b", -0.3),
    ("regulatory_approval", "Regulatory approval",
     r"\b(FDA|EMA|FCC|regulators?)\b.{0,40}\b(approv\w+|clear\w*|authori[sz]\w+|green ?light\w*)\b|\bapproval\b", 0.6),
    ("legal_regulatory", "Legal / regulatory risk",
     r"\b(lawsuit|sued|sues|probe|investigat\w+|antitrust|subpoena|indict\w*|fined|penalt(y|ies)|class action|recall\w*|DOJ|FTC)\b|\bSEC\b", -0.8),
    ("mna", "M&A / deal",
     r"\b(acquir\w+|acquisition|merger|merge|takeover|buyout|tender offer|to buy|agrees to buy|spin-?off|divest\w*)\b", 0.2),
    ("capital_return", "Buyback / dividend",
     r"\b(buybacks?|share repurchase|repurchase program|dividend (hike|increase|raise)|raises dividend|special dividend)\b", 0.6),
    ("layoffs", "Layoffs / restructuring",
     r"\b(layoffs?|lay off|job cuts|cut(s|ting)? (\d[\d,]* )?jobs|restructur\w+|workforce reduction)\b", 0.0),
    ("leadership", "Leadership change",
     r"\b(CEO|CFO|COO|chief executive|chairman)\b.{0,40}\b(steps? down|resign\w*|depart\w*|ousted|exit\w*|named|appoint\w*|replac\w*|retir\w*)\b", -0.2),
    ("partnership", "Partnership / contract win",
     r"\b(partner(s|ship|ing)?|contract|deal with|agreement with|collaborat\w+|wins?|won|selected by|teams up)\b", 0.5),
    ("product", "Product / launch",
     r"\b(launch\w*|unveil\w*|debut\w*|introduc\w*|roll(s|ing)? out|new (chip|model|product|device|phone|service))\b", 0.3),
    ("earnings_event", "Earnings report / preview",
     r"\b(earnings|quarterly results|Q[1-4] results|revenue|EPS)\b", 0.0),
    ("macro", "Macro / market-wide",
     r"\b(Fed|Federal Reserve|inflation|CPI|PPI|interest rates?|tariffs?|recession|jobs report|payrolls|treasury yields?|stock market today|Dow|S&P 500|Nasdaq composite)\b", 0.0),
]
_COMPILED = [(c, lbl, re.compile(p, re.I), d) for c, lbl, p, d in EVENT_RULES]

CATEGORY_LABELS = {c: lbl for c, lbl, _, _ in EVENT_RULES} | {"general": "General news"}
CATEGORY_PRIOR = {c: d for c, _, _, d in EVENT_RULES} | {"general": 0.0}
# Categories that are low-information; these articles get down-weighted.
LOW_SIGNAL_CATEGORIES = {"opinion": 0.4, "macro": 0.6}
# Categories too broad to describe what a day's news was about, unless nothing more specific is.
GENERIC_CATEGORIES = {"general", "opinion", "macro", "earnings_event"}
# Too vague to say anything about a company's partners or competitors.
NOT_RELATIONAL_CATEGORIES = {"general", "opinion", "macro"}
# Disasters, accidents and other non-business events.
WORLD_EVENT_CATEGORIES = {"disaster", "accident", "cyber_outage", "geopolitical", "labor", "health"}
# Sentiment at or beyond these levels counts as positive / negative.
POSITIVE, NEGATIVE = 0.15, -0.15


def score_text(title: str, summary: str = "") -> float:
    """Compound sentiment in [-1, 1]; the headline counts more than the summary."""
    t = _analyzer.polarity_scores(title or "")["compound"]
    if not summary:
        return t
    s = _analyzer.polarity_scores(summary[:600])["compound"]
    return 0.7 * t + 0.3 * s


def classify(title: str, summary: str = "") -> str:
    """Headline is checked first, then the summary.

    World-event types only count from the headline: summaries mention wars or storms in passing.
    """
    for cat, _, rx, _ in _COMPILED:
        if rx.search(title or ""):
            return cat
    for cat, _, rx, _ in _COMPILED:
        if cat not in WORLD_EVENT_CATEGORIES and rx.search(summary or ""):
            return cat
    return "general"


def dominant_category(articles: pd.DataFrame, min_share: float = 0.15) -> str:
    """Specific event type carrying the most article weight, if it is a meaningful share of coverage.

    `articles` needs "category" and "weight" columns.
    """
    by_cat = articles.groupby("category")["weight"].sum()
    total = by_cat.sum()
    specific = by_cat.drop(labels=[c for c in GENERIC_CATEGORIES if c in by_cat.index])
    if not specific.empty and specific.max() >= min_share * total:
        return specific.idxmax()
    generic = by_cat.reindex([c for c in ("earnings_event", "macro") if c in by_cat.index])
    if not generic.empty and generic.max() >= min_share * total:
        return generic.idxmax()
    return "general"


def label(score: float) -> str:
    if score >= 0.5:
        return "strongly positive"
    if score >= POSITIVE:
        return "positive"
    if score > NEGATIVE:
        return "neutral"
    if score > -0.5:
        return "negative"
    return "strongly negative"
