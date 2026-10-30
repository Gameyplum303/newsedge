"""Every company that can be added: NASDAQ's official directory of US-listed securities.

NASDAQ Trader publishes two daily files: NASDAQ-listed securities and those listed on other US
exchanges (NYSE, NYSE American, NYSE Arca, Cboe, IEX). Only this directory is downloaded up front;
news and prices for a company are downloaded once you add it to the watchlist.
"""
import io
import os
import re
import time

import pandas as pd
import requests

from . import config
from .names import clean_company_name

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
CACHE_PATH = os.path.join(config.DATA_DIR, "listings.csv")
MAX_AGE_S = 24 * 3600
EXCHANGES = {"Q": "NASDAQ", "N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}

# Securities that aren't an operating company's shares.
NOT_COMPANY_SHARES = re.compile(r"\b(?:warrants?|units?|rights?|notes?|debentures?|preferred|"
                                r"subordinated|when issued|fund)\b", re.I)   # funds aren't operating companies
SHARE_CLASS_SUFFIX = re.compile(r"\s*(?:-\s*)?(?:Class [A-Z]\s+)?(?:Common Stock|Common Shares|Ordinary Shares|"
                                r"American Depositary Shares|Depositary Shares|Shares of Beneficial Interest).*$", re.I)


def _read(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.strip().splitlines() if not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)


def company_name(security_name: str) -> str:
    """'Apple Inc. - Common Stock' -> 'Apple'; 'JPMorgan Chase & Co. Common Stock' -> 'JPMorgan Chase & Co.'"""
    name = security_name.split(" - ")[0]
    return clean_company_name(SHARE_CLASS_SUFFIX.sub("", name).strip()) or name.strip()


def parse(nasdaq_text: str, other_text: str) -> pd.DataFrame:
    """Companies from the two directory files: symbol, Yahoo symbol, name, company, exchange."""
    nq = _read(nasdaq_text).rename(columns={"Symbol": "symbol", "Security Name": "security"})
    nq["exchange"] = "Q"
    ot = _read(other_text).rename(columns={"ACT Symbol": "symbol", "Security Name": "security", "Exchange": "exchange"})
    df = pd.concat([nq[["symbol", "security", "exchange", "ETF", "Test Issue"]],
                    ot[["symbol", "security", "exchange", "ETF", "Test Issue"]]], ignore_index=True)
    df = df[(df["ETF"] != "Y") & (df["Test Issue"] != "Y") & (df["symbol"] != "")]
    df = df[~df["security"].str.contains(NOT_COMPANY_SHARES)]
    df = df[~df["symbol"].str.contains(r"[\$\^]")]          # preferred-share and test notations
    out = pd.DataFrame({
        "symbol": df["symbol"].str.strip(),
        "yahoo": df["symbol"].str.strip().str.replace(".", "-", regex=False),   # BRK.B -> BRK-B
        "security": df["security"],
        "company": df["security"].map(company_name),
        "exchange": df["exchange"].map(EXCHANGES).fillna("Other US"),
    })
    return out.drop_duplicates("yahoo").sort_values("yahoo").reset_index(drop=True)


def load(force: bool = False) -> pd.DataFrame:
    """The directory, refreshed from NASDAQ once a day; the last good copy is used if offline."""
    fresh = os.path.exists(CACHE_PATH) and time.time() - os.path.getmtime(CACHE_PATH) < MAX_AGE_S
    if fresh and not force:
        return pd.read_csv(CACHE_PATH, dtype=str, keep_default_na=False)
    try:
        headers = {"User-Agent": "Mozilla/5.0 NewsEdge/1.0"}
        texts = [requests.get(u, headers=headers, timeout=20) for u in (NASDAQ_URL, OTHER_URL)]
        for r in texts:
            r.raise_for_status()
        df = parse(texts[0].text, texts[1].text)
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        return df
    except Exception:
        if os.path.exists(CACHE_PATH):
            return pd.read_csv(CACHE_PATH, dtype=str, keep_default_na=False)
        return pd.DataFrame(columns=["symbol", "yahoo", "security", "company", "exchange"])


def find(ticker: str, listings: pd.DataFrame | None = None) -> dict | None:
    """The directory entry for a ticker (as Yahoo writes it), or None if it isn't a listed company."""
    listings = load() if listings is None else listings
    hit = listings[listings["yahoo"] == ticker.upper()]
    return hit.iloc[0].to_dict() if not hit.empty else None
