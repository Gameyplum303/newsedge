"""How an article relates to a stock, and how strongly.

An article is linked to a stock in one of these ways (the strongest wins):
  * directly: it names the company, its ticker or one of its products ("AWS outage" -> AMZN)
  * government business: war, budget and policy news for companies that depend on government
    contracts; war news reads as positive for defense contractors
  * relationships: news about a supplier, customer, partner or competitor; a competitor's good
    news reads as negative
  * exposure: a world event that hits a supply chain or market the company depends on
    ("earthquake in Taiwan" -> chip makers)
Company-specific inputs come from profiles.py; event-wide ones from companies.py.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

from . import companies, profiles, sentiment

# How directly an article is about a company.
RELEVANCE_HEADLINE = 1.0      # named in the headline
RELEVANCE_SUMMARY = 0.6       # named only in the summary
RELEVANCE_TICKER_FEED = 0.45  # listed in the company's own feed without naming it
RELEVANCE_GOVERNMENT = 0.45   # x the company's government weight: war, budget and policy news
RELEVANCE_RELATION = 0.5      # x the relationship weight: news about a supplier, customer, partner, competitor
RELEVANCE_EXPOSURE = 0.35     # world event hitting a supply chain or market the company depends on
RELEVANCE_COMMENTATOR = 0.15  # named only as the analyst/commentator ("JPMorgan upgrades Intel")
SUMMARY_ONLY = 0.6            # indirect link found only in the summary, not the headline

# Link kinds.
DIRECT, FEED, GOVERNMENT, RELATION, EXPOSURE = "direct", "feed", "government", "relation", "exposure"
INDIRECT_LINKS = {GOVERNMENT, RELATION, EXPOSURE}


@dataclass(frozen=True)
class Link:
    relevance: float
    via: str = ""        # how an indirect link was made, e.g. "TSMC (supplier)" or "Taiwan chip supply"
    sign: int = 1        # -1 flips sentiment: competitors, and stocks that tend to gain from the bad news
    kind: str = DIRECT


# ---------------------------------------------------------------- direct mentions
@dataclass(frozen=True)
class CompanyMatcher:
    ticker: str
    mention: re.Pattern            # company/product names or explicit symbols like $AAPL, (AAPL), NASDAQ:AAPL
    commentary: re.Pattern | None  # spans where the company is only the commentator

    def relevance(self, title: str, summary: str) -> float:
        """RELEVANCE_* level for an article (0 = company not mentioned)."""
        if self._direct(title):
            return RELEVANCE_HEADLINE
        if self.mention.search(title):
            return RELEVANCE_COMMENTATOR
        if self._direct(summary):
            return RELEVANCE_SUMMARY
        return 0.0

    def _direct(self, text: str) -> bool:
        return bool(self.mention.search(self.commentary.sub(" ", text) if self.commentary else text))


def _alternation(names) -> str:
    return "|".join(re.escape(n) for n in names if n)


def build_matchers(watchlist: dict[str, list[str]]) -> dict[str, CompanyMatcher]:
    all_companies = {**companies.COMPANIES, **watchlist}
    matchers = {}
    for ticker, names in watchlist.items():
        sym = rf"(?:\$|\((?:(?:NASDAQ|NYSE|Nasdaq|NYSEARCA)\s*:\s*)?|(?:NASDAQ|NYSE)\s*:\s*){re.escape(ticker)}\b"
        alts = _alternation(list(names) + [a for a in profiles.aliases(ticker) if a not in names])
        mention = re.compile(rf"\b(?:{alts})\b|{sym}" if alts else sym)
        commentary = None
        if names:
            own = _alternation(names)   # only the company itself can be a commentator, not its products
            others = _alternation(n for t, ns in all_companies.items() if t != ticker for n in ns) or "(?!)"
            commentary = re.compile(
                # Company names stay case-sensitive; the analyst vocabulary around them is not (Title Case headlines).
                rf"\b(?:{own})(?:'s)?\s+(?:\w+\s+)?(?i:analysts?|strategists?|economists?|research|upgrades|downgrades|"
                rf"initiates|reiterates|maintains|resumes|(?:raises|lifts|hikes|cuts|lowers|trims|boosts)\s+\S+(?:'s)?\s+"
                rf"(?:price target|target|rating)|says\s+\S+\s+(?:stock|shares)|sees)\b"
                rf"|\b(?i:according to)\s+(?:{own})\b"
                rf"|\b(?:{own})(?:'s)?\s+(?:{others})(?:'s)?\s+(?i:upgrade|downgrade|price target|rating|call)\b"
                rf"|\b(?i:upgraded|downgraded|initiated|rated|reiterated)\b(?:\s+\S+){{0,4}}?\s+(?i:by|at)\s+(?:{own})\b")
        matchers[ticker] = CompanyMatcher(ticker, mention, commentary)
    return matchers


# ---------------------------------------------------------------- indirect links
@dataclass(frozen=True)
class Exposure:
    name: str
    pattern: re.Pattern
    categories: frozenset
    tickers: dict            # ticker -> +1 (hurt by bad news) or -1 (tends to benefit)

    def applies(self, title: str, summary: str, category: str) -> bool:
        return category in self.categories and bool(self.pattern.search(f"{title} {summary}"))


EXPOSURES = [Exposure(e["name"], re.compile(e["pattern"], re.I), frozenset(e["categories"]), e["tickers"])
             for e in companies.EXPOSURES]


@dataclass(frozen=True)
class Relation:
    label: str               # e.g. "TSMC"
    relation: str            # supplier / customer / partner / competitor
    weight: float
    pattern: re.Pattern      # the related company's names (case-sensitive)

    @property
    def sign(self) -> int:
        return companies.RELATION_SIGN[self.relation]


@lru_cache(maxsize=1024)
def _relations(ticker: str, profiles_version: float) -> tuple[Relation, ...]:
    """Compiled relationships for a ticker; rebuilt when profiles are edited."""
    return tuple(Relation(r["names"][0], r["relation"], float(r["weight"]), re.compile(rf"\b(?:{_alternation(r['names'])})\b"))
                 for r in profiles.relationships(ticker) if r.get("names"))


_GOVERNMENT_TOPICS = {area: re.compile(p, re.I) for area, p in companies.GOVERNMENT_TOPICS.items()}


def _government_links(ticker: str, title: str, summary: str, category: str) -> list[Link]:
    areas = profiles.government(ticker)
    if not areas:
        return []
    if category == "geopolitical":
        # War and conflict tend to lift defense contractors (and ceasefires to weigh on them).
        w = areas.get("defense", 0)
        return [Link(RELEVANCE_GOVERNMENT * w, "defense contractor (conflict news)", -1, GOVERNMENT)] if w else []
    if category == "government_contract":
        return []   # a contract named for someone else; the winner is linked directly
    out = []
    for text, scale in ((title, 1.0), (summary, SUMMARY_ONLY)):
        for area, w in areas.items():
            if _GOVERNMENT_TOPICS[area].search(text) or _GOVERNMENT_TOPICS["all"].search(text):
                label = f"government {companies.GOVERNMENT_AREA_LABELS[area]}"
                out.append(Link(RELEVANCE_GOVERNMENT * w * scale, label, 1, GOVERNMENT))
    return out


def _relation_links(ticker: str, title: str, summary: str, category: str) -> list[Link]:
    if category in sentiment.NOT_RELATIONAL_CATEGORIES:
        return []
    return [Link(RELEVANCE_RELATION * r.weight * scale, f"{r.label} ({r.relation})", r.sign, RELATION)
            for r in _relations(ticker, profiles.version())
            for text, scale in ((title, 1.0), (summary, SUMMARY_ONLY)) if r.pattern.search(text)]


def _exposure_links(ticker: str, title: str, summary: str, category: str) -> list[Link]:
    return [Link(RELEVANCE_EXPOSURE, e.name, e.tickers[ticker], EXPOSURE)
            for e in EXPOSURES if ticker in e.tickers and e.applies(title, summary, category)]


# ---------------------------------------------------------------- public API
def link(matcher: CompanyMatcher | None, ticker: str, title: str, summary: str, category: str,
         fallback: float = 0.0) -> Link:
    """How an article relates to one stock: the strongest of a direct mention, government business,
    a related company and a world-event exposure; failing all of them, `fallback`."""
    direct = matcher.relevance(title, summary) if matcher else 0.0
    if direct > 0:
        return Link(direct)
    indirect = (_government_links(ticker, title, summary, category) + _relation_links(ticker, title, summary, category)
                + _exposure_links(ticker, title, summary, category))
    best = max(indirect, key=lambda lk: lk.relevance, default=None)
    return best if best is not None else Link(fallback, kind=FEED)
