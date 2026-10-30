"""Settings: paths, data sources and model parameters. Company data lives in companies.py."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the project folder
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "newsedge.db")
WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.json")


def secret(name: str) -> str:
    """An API key from the environment. On Windows, keys saved with `setx` are also read from the
    user's registry, so an app started before they were saved still finds them. Keys are never
    written to files or logs."""
    value = os.environ.get(name, "")
    if not value and os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                value = str(winreg.QueryValueEx(k, name)[0])
        except OSError:
            value = ""
    return value.strip()


# Optional: free key from https://finnhub.io streams real-time US trades.
# Without it, quotes come from Yahoo Finance (Nasdaq real-time, 1-minute resolution).
FINNHUB_API_KEY_DEFAULT = secret("FINNHUB_API_KEY")

MARKET_BENCHMARK = "SPY"
PRICE_HISTORY_PERIOD = "11y"   # must cover the oldest back-filled articles

# General business-news RSS feeds. Articles are mapped to tickers by company name/symbol.
GENERAL_FEEDS = {
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    "CNBC Earnings": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135",
    "CNBC Tech": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910",
    "CNBC Investing": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "MarketWatch Pulse": "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "WSJ Markets": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
    "Investing.com": "https://www.investing.com/rss/news_25.rss",
    "Business Insider": "https://markets.businessinsider.com/rss/news",
}

# World and disaster feeds: articles count for a stock when they name the company (or one of its
# products), or hit one of its exposures (companies.EXPOSURES).
WORLD_FEEDS = {
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC US & Canada": "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
    "NPR News": "https://feeds.npr.org/1001/rss.xml",
    "NYT World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "NYT US": "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
    "Guardian World": "https://www.theguardian.com/world/rss",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "CNBC World": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
    "ABC News": "https://abcnews.go.com/abcnews/topstories",
    "CBS News": "https://www.cbsnews.com/latest/rss/main",
    "GDACS disasters": "https://www.gdacs.org/xml/rss.xml",
    "USGS significant quakes": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.atom",
}
# Feed entries to ignore, by headline pattern (GDACS "Green" alerts are minor events).
FEED_SKIP_PATTERNS = {"GDACS disasters": r"^Green\b"}

# Extra per-ticker search for incidents that business coverage may not label as "stock" news.
INCIDENT_TERMS = ("fire OR explosion OR outage OR recall OR crash OR strike OR earthquake OR hurricane OR flood "
                  "OR cyberattack OR hack OR breach OR accident OR evacuation OR shutdown OR derailment OR spill")

# Per-ticker feeds. {ticker} and {query} are filled in.
YAHOO_TICKER_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_NEWS_FEED = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
SEEKING_ALPHA_TICKER_FEED = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"

# How much each outlet's article counts. Unlisted outlets get DEFAULT_SOURCE_WEIGHT.
SOURCE_WEIGHTS = {
    "reuters": 1.0, "bloomberg": 1.0, "wall street journal": 1.0, "wsj": 1.0,
    "cnbc": 1.0, "marketwatch": 0.95, "barron's": 0.95, "financial times": 1.0,
    "associated press": 0.9, "ap news": 0.9, "yahoo finance": 0.85, "business insider": 0.8,
    "investing.com": 0.8, "seeking alpha": 0.75, "the motley fool": 0.5, "motley fool": 0.5,
    "zacks": 0.5, "investorplace": 0.5, "benzinga": 0.65, "simply wall st": 0.5, "tipranks": 0.6,
    "bbc": 0.95, "npr": 0.9, "nyt": 0.95, "new york times": 0.95, "guardian": 0.9, "al jazeera": 0.85,
    "abc news": 0.85, "cbs news": 0.85, "gdacs": 0.9, "usgs": 1.0,
}
DEFAULT_SOURCE_WEIGHT = 0.7

# Signal model settings
SIGNAL_WINDOW_HOURS = 24      # articles newer than this feed the live signal
RECENCY_HALF_LIFE_HOURS = 8   # an article's weight halves every N hours
IMPACT_Z = 1.5                # "big move" = abnormal return beyond 1.5 standard deviations
MIN_EVENTS_TO_TRAIN = 40      # below this, default model weights are used
OVERNIGHT_GAP_SHARE = 0.25    # share of a day's price variance that arrives in the overnight gap
ROUND_TRIP_COST = 0.001       # spread + slippage + fees for buying and selling once (0.10%), large US stocks
