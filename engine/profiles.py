"""Company profiles: product names, government business and relationships.

Built-in profiles live in companies.py. Profiles for companies you add, and any edits you make in
the Companies tab, are saved to data/profiles.json and take precedence field by field.
"""
import json
import os
import re

import yfinance as yf

from . import companies, config
from .names import clean_company_name, headline_names

PROFILES_PATH = os.path.join(config.DATA_DIR, "profiles.json")
FIELDS = ("aliases", "government", "relationships")

# Industry (as Yahoo Finance names it) -> suggested government business.
INDUSTRY_GOVERNMENT = {
    "Aerospace & Defense": {"defense": 0.6},
    "Healthcare Plans": {"healthcare": 0.5},
    "Medical Care Facilities": {"healthcare": 0.4},
    "Drug Manufacturers - General": {"healthcare": 0.25},
    "Drug Manufacturers—General": {"healthcare": 0.25},
    "Biotechnology": {"healthcare": 0.15},
    "Information Technology Services": {"civilian": 0.2},
    "Engineering & Construction": {"civilian": 0.2},
    "Security & Protection Services": {"civilian": 0.2},
}
# Phrases in a company's business description that point to government customers.
DESCRIPTION_GOVERNMENT = [
    (r"Department of Defense|U\.S\. (military|armed forces|Army|Navy|Air Force)|defense (agencies|customers)", "defense", 0.3),
    (r"U\.S\. government|federal (government|agencies)|government (agencies|customers)", "civilian", 0.2),
    (r"\bMedicare\b|\bMedicaid\b", "healthcare", 0.3),
]

_cache = {"mtime": None, "data": {}}


def _user_profiles() -> dict:
    try:
        mtime = os.path.getmtime(PROFILES_PATH)
    except OSError:
        return {}
    if mtime != _cache["mtime"]:
        try:
            with open(PROFILES_PATH, encoding="utf-8") as f:
                _cache["data"] = json.load(f)
        except (OSError, ValueError):
            _cache["data"] = {}
        _cache["mtime"] = mtime
    return _cache["data"]


def version() -> float:
    """Changes whenever profiles are edited (used to refresh cached matchers)."""
    _user_profiles()
    return _cache["mtime"] or 0.0


def _field(ticker: str, field: str, default):
    user = _user_profiles().get(ticker, {})
    return user[field] if field in user else default


def aliases(ticker: str) -> list[str]:
    return _field(ticker, "aliases", companies.COMPANY_ALIASES.get(ticker, []))


def government(ticker: str) -> dict[str, float]:
    return {k: v for k, v in _field(ticker, "government", companies.GOVERNMENT_EXPOSURE.get(ticker, {})).items() if v}


def relationships(ticker: str) -> list[dict]:
    return _field(ticker, "relationships", companies.RELATIONSHIPS.get(ticker, []))


def related_terms(ticker: str) -> list[str]:
    """Search terms for a company's partners, suppliers, competitors and government business."""
    terms = [r["names"][0] for r in relationships(ticker) if r.get("names")]
    for area in government(ticker):
        terms += companies.GOVERNMENT_QUERY_TERMS[area]
    return list(dict.fromkeys(terms))


def profile(ticker: str) -> dict:
    user = _user_profiles().get(ticker, {})
    return {"aliases": aliases(ticker), "government": government(ticker), "relationships": relationships(ticker),
            "sector": user.get("sector", ""), "industry": user.get("industry", ""), "custom": bool(user)}


def save_profile(ticker: str, **fields) -> None:
    data = dict(_user_profiles())
    data[ticker] = {**data.get(ticker, {}), **fields}
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    _cache["mtime"] = None


def suggest_government(industry: str, description: str) -> dict[str, float]:
    gov = dict(INDUSTRY_GOVERNMENT.get(industry, {}))
    for pattern, area, weight in DESCRIPTION_GOVERNMENT:
        if re.search(pattern, description or ""):
            gov[area] = max(gov.get(area, 0.0), weight)
    return gov


def lookup(ticker: str) -> dict | None:
    """Name, sector, industry and description for a ticker, or None if it can't be found or traded."""
    try:
        t = yf.Ticker(ticker)
        if t.history(period="5d").empty:
            return None
        info = t.info or {}
    except Exception:
        return None
    name = clean_company_name(info.get("shortName") or info.get("longName") or "")
    return {"ticker": ticker, "names": [name] if name else [], "sector": info.get("sector", ""),
            "industry": info.get("industry", ""), "description": info.get("longBusinessSummary", "")}


def suggest(found: dict) -> dict:
    """A starting profile for a newly added company. Relationships can't be looked up from free data."""
    return {"aliases": [], "government": suggest_government(found["industry"], found["description"]),
            "relationships": [], "sector": found["sector"], "industry": found["industry"]}


def set_up_company(ticker: str, company: str = "") -> list[str] | None:
    """Prepare a company being added: check it can be modeled, save a suggested profile (unless one
    exists) and return the names to match in headlines. None if Yahoo Finance has no price history
    for it (e.g. a very recent listing). Nothing else about the company is downloaded here."""
    found = lookup(ticker)
    if found is None:
        return None
    if ticker in companies.COMPANIES:
        return companies.COMPANIES[ticker]
    if not profile(ticker)["custom"]:
        save_profile(ticker, **suggest(found))
    name = company or (found["names"][0] if found["names"] else "")
    return headline_names(name) if name else []
