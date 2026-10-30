# NewsEdge

News-driven signals for US stocks. NewsEdge collects business news, links each article to the
company it is about, learns how those stocks reacted to similar news in the past, and shows a
calibrated probability of where each stock moves next, along with the reasoning and source articles.

> Educational research tool. Probabilities, not promises. Not investment advice.

## Setup

```bash
python -m pip install -r requirements.txt
```

Optional: real-time exchange quotes from Finnhub (free key at https://finnhub.io):

```bash
setx FINNHUB_API_KEY your_key_here
```

Recommended: Benzinga news with exact publication times (2015 on) from a free Alpaca account
(https://alpaca.markets, paper-trading keys are enough). Save the keys on your computer, never in code:

```bash
setx APCA_API_KEY_ID your_key_id
setx APCA_API_SECRET_KEY your_secret
python backfill.py --years 11 --kinds alpaca
```

## Security

- The dashboard has no login, so it only listens on this computer (`server.address = "localhost"` in
  `.streamlit/config.toml`). Don't expose it on a network without adding authentication.
- API keys are read from environment variables, sent only over HTTPS to their own service (Alpaca
  requests never follow redirects), and never logged or shown.
- Feeds are untrusted: article links must be plain http(s), headlines are escaped before they reach
  Markdown or chart hover text, and oversized feed responses are refused.
- Tickers are checked against a strict pattern before they are used in URLs, files or searches, and
  SQL always uses parameters.

## Run

```bash
python -m streamlit run app.py
```

Or double-click `run.bat`. On first run, back-fill history so the model can calibrate on real price reactions:

```bash
python backfill.py            # 10 years: news, incident and related-company searches; resumable
```

Or use **📚 Build history** in the sidebar. Finished weeks are remembered, so a stopped or
rate-limited run picks up where it left off.

## Project layout

```
app.py              page outline: sidebar -> analysis -> views
engine/             all logic, no Streamlit (importable and testable on its own)
  config.py         settings: paths, news feeds, source weights, model parameters
  companies.py      built-in company data: names, products, government business, relationships, exposures
  profiles.py       company profiles: built-in defaults plus your edits (data/profiles.json)
  names.py          company names: listing names to the short names headlines use
  market_clock.py   NYSE calendar: sessions, holidays, countdowns, trading dates
  sentiment.py      headline sentiment, event types, dominant event of a day
  linking.py        how an article relates to a stock: direct, government, related company, exposure
  news.py           collecting: RSS feeds and Google News searches, live and historical
  articles.py       preparing stored articles: re-scoring and per-stock linking
  events.py         news events as tradeable moves: entry after the news, exit at the next close
  model.py          the news-impact model (logistic regressions, per-stock reliability)
  evaluation.py     out-of-sample accuracy, calibration and what trading the calls would have earned
  forecast.py       predicted close with 68% / 90% ranges, sized by news and time left
  signals.py        per-stock signal with plain-English reasoning
  strategy.py       Monte-Carlo trade planner over the profit-target range
  prices.py         price history and accurate live quotes (Quote)
  livefeed.py       Finnhub real-time trade stream
  alpaca.py         Alpaca API: Benzinga news with exact timestamps, minute price bars
  sanitize.py       rules for untrusted text: links, tickers, feed HTML, Markdown escaping
  store.py          SQLite: articles, download progress, background jobs; the watchlist file
  pipeline.py       analyse(): model + signals + trade ideas in one call
  backfill.py       resumable history download (also: python backfill.py)
  jobs.py           background set-up of newly added companies (worker process)
  listings.py       every US-listed company that can be added (NASDAQ's official directory)
ui/                 Streamlit dashboard
  sidebar.py        user controls -> Controls
  data.py           every cached network/heavy call
  components.py     clock, live prices, auto-refresh, trade/article tables
  charts.py         plotly figures
  views/            trade ideas, signals, action plan, news feed, history, companies
tests/              pytest suite (synthetic market, no network)
data/               articles database and watchlist (created on first run)
```

## Tests

```bash
python -m pytest
```

The tests run offline in a few seconds. They cover the market calendar, event classification,
company matching, quote selection and freshness, the model (on a synthetic market with a planted
news effect), the trade planner and the full pipeline.

## How each requirement is met

| # | Requirement | Where |
|---|---|---|
| 1 | Collect business **and world** news and tie it to companies | `engine/news.py`: 11 business feeds (CNBC, MarketWatch, WSJ, Yahoo Finance, Seeking Alpha, Investing.com, Business Insider), 12 world and disaster feeds (BBC, NPR, NYT, Guardian, Al Jazeera, ABC, CBS, CNBC World, GDACS, USGS), plus per-ticker Yahoo, Google News, Google News incident searches and Seeking Alpha feeds. Matches company names, tickers and product names, links world events through supply-chain exposures, and removes cross-feed duplicates. |
| 2 | % confidence that news moves the stock, and which way | `engine/sentiment.py` scores headlines with VADER plus a finance lexicon and classifies 17 event types. `engine/model.py` outputs **direction confidence** and **impact probability**. |
| 3 | Learn from past price moves vs past articles | `engine/model.py` maps each article to the session it affected, computes the market-adjusted (abnormal) return, fits logistic models, and reports out-of-sample accuracy. |
| 4 | Clean GUI with confidence, reasoning and articles | `ui/`: signal table, per-stock cards with "Why" bullets, source articles with links, and price charts with news markers. |
| 5 | Best action for a +0.1% to +10% return | `engine/strategy.py` tests 16 profit targets across the range on simulated 30-minute price paths, for long and short. The dashboard shows two lists: **Highest confidence trades** (the target most likely to be reached) and **Highest return trades** (the target with the largest expected return). The stop-loss can be fixed or scale with the target. |
| 6 | Real-time data and market clock | `engine/prices.py` and `engine/livefeed.py` (Yahoo, optional Finnhub stream) and `engine/market_clock.py` (NYSE hours, 2026–27 holidays, early closes) with live open/close countdowns. |

## Adding a company

Use the sidebar's **Add companies** search: it covers every company listed on NASDAQ, plus NYSE and
the other US exchanges, from NASDAQ's official daily symbol directory (ETFs, funds, warrants, units,
rights and preferred shares are left out). Pick one or more, then click **Save watchlist**. NewsEdge then:

1. checks the company has price history and looks up its sector, industry and description
2. sets the names to look for in headlines (e.g. "Palantir Technologies" and "Palantir") and suggests
   a profile: government business is guessed from the industry and business description
3. downloads 10 years of news, incident and related-company history **in the background**
   (about 10–15 minutes; the sidebar shows progress, and you can keep using the dashboard)
4. retrains the model automatically when the download finishes

**Nothing about a company is downloaded until you add it.** The only thing fetched up front is the
directory itself (refreshed daily, cached in `data/listings.csv`). Removing a company cancels any
download still running for it, and the background worker refuses to start one for a company that
isn't on the watchlist. Articles already downloaded for a removed company are kept, so re-adding
it is quick.

Downloads run one company at a time in a separate worker (`python -m engine.jobs`), so they keep
going if you close or reload the page and resume after an interruption. If Google News starts
rate-limiting, the job pauses and a **Resume** button appears; finished weeks are kept.

Open the **Companies** tab to check each company's history coverage and edit its profile: product
names, government business, and related companies. Relationships can't be looked up reliably from
free data, so add suppliers, customers, partners and competitors there.
## World events (disasters, accidents, outages, conflict)

Six event types beyond business news: natural disaster, accident / industrial incident, outage /
cyberattack, war / geopolitical, strike / labor action, outbreak / health emergency. They are
detected from the headline (summaries mention wars or storms in passing).

An event affects a stock when it:

- **names the company or a product** (`COMPANY_ALIASES` in `engine/companies.py`: "AWS outage" -> AMZN,
  "737 MAX" -> BA), or
- **hits an exposure** (`EXPOSURES` in `engine/companies.py`), such as a Taiwan earthquake -> chip makers,
  or Red Sea attacks on tankers -> oil producers. Exposure links count for less than a direct mention.
  Stocks marked `-1` tend to benefit from the bad news, so their sentiment is flipped.

Edit both lists in `engine/companies.py`. When you change the lexicon or event rules in
`engine/sentiment.py`, bump `SCORER_VERSION` and stored articles are re-scored automatically.

## Government contracts and company relationships

Each company has a relationship profile (defaults in `engine/companies.py`, editable in the Companies tab):

- **`GOVERNMENT_EXPOSURE`**: how much defense, civilian IT and healthcare-program business matters
  (0 to 1). War and conflict news counts as **positive** for defense contractors, and ceasefires
  as negative. Budget, shutdown and policy news (Pentagon budget, Medicare rates, CHIPS Act) is
  read as-is for companies that depend on that area. The weight scales how much it counts:
  Lockheed (1.0) far more than Microsoft (0.1).
- **`RELATIONSHIPS`**: suppliers, customers, partners and competitors, each with a weight. News about
  TSMC reaches Nvidia, AMD and Apple; news about OpenAI reaches Microsoft. A **competitor's** good
  news counts as negative, and a rival's contract win is not good news.
- Two event types: **Government contract** (awards; positive) and **Government budget / policy**.
- A per-company **related** search (partners plus government topics, without the company's name)
  finds this news live and in the back-fill.

An article's link to a stock is the strongest of: direct mention, government business, related
company, world-event exposure. The stock card lists the indirect links ("TSMC (supplier) ×3"), and
the News feed can filter by link type.

The model gets an **indirect-share** input: how much of a day's news weight came from indirect links.
It learns from history how much less (or more) that news moves prices than news about the company itself.

Limits: sentiment is scored per headline, so a headline about *ending* a war can still read as
negative before it is flipped for defense names. Relationship lists are hand-curated and should be
reviewed as partnerships change.
## How the confidence is computed

1. Each article gets a sentiment score in [-1, 1] and an event type (earnings beat, downgrade, lawsuit, …).
2. Articles from the last 24h are weighted by recency (half-life 8h), source quality and how directly
   they mention the company.
3. Two logistic regressions trained on your stored history convert (sentiment, event type, article
   count, share of indirect news) into P(stock beats the market) and P(outsized move > 1.5σ), over a
   move you **could actually have traded**: from the first price available once the news was public
   (the next open for news before the open, the close for news during market hours) to the next close.
   Back-filled Google News items only carry a date, so they count from the open after that date;
   the model never gets credit for a move that happened before the news was out.
4. Confidence is scaled by how reliably that stock has followed its news historically.
5. The **History & model** tab tests the model on the most recent 30% of history, which it never
   trained on, and shows **what trading its calls would have earned after 0.10% round-trip costs**.
   Expected returns in the trade plans are also after costs.

### Predicted close

Each stock card shows a **predicted close** with its confidence, updated every 5 seconds from the live
price (`engine/forecast.py`). A single price can't carry a confidence (the chance of closing on exactly
that cent is ~0), so the card gives:

- the **most likely close**: today's during market hours, otherwise the next session's
- a **68% range** and a **90% range**, where the close should land 68% / 90% of the time
- the **chance it closes above the current price**

How it reacts:

- **Time left.** The ranges narrow as the close approaches.
- **News volume.** More news widens the ranges, using the model's chance of an outsized move. A 13%
  chance is a normal day; 40% roughly doubles the width.
- **News direction.** Outside market hours, the next close shifts by the expected move. During the
  session the forecast stays centred on the current price, because the 10-year test found that
  intraday news is only tradeable from the next session.

The range widths are sized on the training years. The **History & model** tab then reports how often
real closes landed inside each range on years the model never saw.

### What the 10-year test found

Over 2023–2026 (7,445 unseen news events, 10 large US stocks), trading on headline sentiment was
right 49% of the time and lost about 0.1% per trade after costs. The model now rarely claims more
than 55% confidence, so the trade ideas mostly say **WAIT**: an honest result, not a fault. Earlier
apparent edges came from same-day moves that free feeds report only after they happen.

## Data notes and limits

- Uses RSS feeds, which the publishers provide for automated reading. Scraping full article pages
  usually violates site terms and breaks often.
- **Price accuracy.** Each price is the newest trade from any source, including pre-market and
  after-hours. It shows the time of that trade and a freshness label (🟢 live, 🟡 last trade X
  minutes ago, ⚪ market closed). Daily % change is measured against the official, unadjusted
  previous close. Prices refresh on their own every 5 seconds.
  - Without a key: Yahoo Finance (Nasdaq real-time, 1-minute resolution) via ``yfinance``, an
    unofficial library.
  - With a free Finnhub key (sidebar, or ``setx FINNHUB_API_KEY yourkey``): a live websocket stream
    of every trade. The newest trade from either source wins.
  - For a production setup, use a licensed feed such as Polygon or Alpaca.
- Headline sentiment is a weak predictor of moves you can still trade: free feeds publish minutes
  after trading firms have reacted. Large moves usually happen before any feed carries the story.
- Historical news timestamps from Google News are dates only. Live articles keep their exact time,
  so the database slowly builds an exactly timed history that future tests can use.
- Holiday calendar covers 2026–2027. Extend `HOLIDAYS` in `engine/market_clock.py` each year.
