from datetime import date

from engine import alpaca, backfill, linking, news

WATCHLIST = {"AAPL": ["Apple"], "MSFT": ["Microsoft"]}


def item(headline, symbols, created="2026-10-29T13:05:07Z", **extra):
    return {"id": 1, "headline": headline, "summary": "", "url": "https://x", "source": "benzinga",
            "symbols": symbols, "created_at": created, **extra}


def test_items_keep_exact_time_and_tag_listed_watchlist_stocks():
    arts = news._alpaca_to_articles([item("Apple Q4 EPS $1.64 Beats $1.60 Estimate", ["AAPL", "XYZ"])],
                                    linking.build_matchers(WATCHLIST))
    assert [a["ticker"] for a in arts] == ["AAPL"]
    assert arts[0]["published"].isoformat() == "2026-10-29T13:05:07+00:00"   # to the second, not a date
    assert arts[0]["source"] == "Benzinga" and arts[0]["relevance"] > 0


def test_company_item_counts_without_naming_the_company():
    arts = news._alpaca_to_articles([item("iPhone 18 shipments rise", ["AAPL"])], linking.build_matchers(WATCHLIST))
    assert [a["ticker"] for a in arts] == ["AAPL"]


def test_roundups_count_only_where_they_name_the_company():
    roundup = item("Microsoft Leads Tech Higher: Stock Market Today", ["AAPL", "MSFT", "SPY", "QQQ", "IWM"])
    arts = news._alpaca_to_articles([roundup], linking.build_matchers(WATCHLIST))
    assert [a["ticker"] for a in arts] == ["MSFT"]


def test_alpaca_backfill_needs_keys(monkeypatch):
    assert "alpaca" not in backfill.applicable_kinds("AAPL")
    monkeypatch.setattr(alpaca, "credentials", lambda: ("id", "secret"))
    assert "alpaca" in backfill.applicable_kinds("AAPL")


def test_dates_are_sent_as_utc_midnight():
    assert alpaca._iso(date(2026, 10, 1)) == "2026-10-01T00:00:00Z"


def test_fetch_window_reports_unreachable_as_none(monkeypatch):
    monkeypatch.setattr(alpaca, "news", lambda *a, **k: None)
    assert news.fetch_window("AAPL", ["Apple"], date(2026, 10, 1), date(2026, 10, 31), "alpaca") is None
    monkeypatch.setattr(alpaca, "news", lambda *a, **k: [item("Apple raises dividend", ["AAPL"])])
    assert len(news.fetch_window("AAPL", ["Apple"], date(2026, 10, 1), date(2026, 10, 31), "alpaca")) == 1
