from datetime import date

import pytest

from engine import backfill, config, jobs, listings, store
from engine.names import clean_company_name, headline_names

NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
PLTR|Palantir Technologies Inc. - Class A Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ABCDW|ABCD Acquisition Corp. - Warrant|S|N|N|100|N|N
ABCDU|ABCD Acquisition Corp. - Units|S|N|N|100|N|N
ZXZZT|NASDAQ TEST STOCK|Q|Y|N|100|N|N
File Creation Time: 1020202617:01|||||||"""

OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
JPM|JP Morgan Chase & Co. Common Stock|N|JPM|N|100|N|JPM
BRK.B|Berkshire Hathaway Inc. New Common Stock|N|BRK.B|N|100|N|BRK=B
JPM$C|JP Morgan Chase & Co. Depositary Shares Preferred Series C|N|JPMpC|N|100|N|JPM-C
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
File Creation Time: 1020202617:01||||||"""


def test_directory_keeps_listed_companies_only():
    df = listings.parse(NASDAQ, OTHER)
    assert df["yahoo"].tolist() == ["AAPL", "BRK-B", "JPM", "PLTR"]   # no ETFs, warrants, units, tests, preferreds
    rows = df.set_index("yahoo")
    assert rows.at["AAPL", "exchange"] == "NASDAQ" and rows.at["JPM", "exchange"] == "NYSE"
    assert rows.at["AAPL", "company"] == "Apple"
    assert rows.at["JPM", "company"] == "JP Morgan Chase"
    assert rows.at["PLTR", "company"] == "Palantir Technologies"


def test_find_uses_yahoo_symbols():
    df = listings.parse(NASDAQ, OTHER)
    assert listings.find("brk-b", df)["company"] == "Berkshire Hathaway"
    assert listings.find("QQQ", df) is None


@pytest.mark.parametrize("raw, clean", [
    ("JPMorgan Chase & Co.", "JPMorgan Chase"),
    ("Shopify Inc. Class A", "Shopify"),
    ("3M Company", ""),            # "3M" is too short to match headlines safely on its own
])
def test_clean_company_name(raw, clean):
    assert clean_company_name(raw) == clean


@pytest.mark.parametrize("company, names", [
    ("Palantir Technologies", ["Palantir Technologies", "Palantir"]),
    ("Taiwan Semiconductor Manufacturing", ["Taiwan Semiconductor Manufacturing", "Taiwan Semiconductor"]),
    ("Shopify", ["Shopify"]),
    ("United Holdings", ["United Holdings"]),             # "United" alone would match too much
    ("First Hawaiian", ["First Hawaiian"]),
])
def test_headline_names(company, names):
    assert headline_names(company) == names


# ---------------------------------------------------------------- download only for watchlist companies
@pytest.fixture
def temp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(config, "WATCHLIST_PATH", str(tmp_path / "watchlist.json"))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    return tmp_path


def test_sync_queues_added_and_cancels_removed(temp_data):
    old = {"AAPL": ["Apple"]}
    jobs.enqueue("AAPL", ["Apple"])
    queued, cancelled = jobs.sync_watchlist(old, {"SHOP": ["Shopify"]})
    assert queued == ["SHOP"] and cancelled == ["AAPL"]
    assert store.open_job("AAPL") is None and store.open_job("SHOP")["status"] == "queued"


def test_worker_never_downloads_for_companies_not_on_the_watchlist(temp_data, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs.backfill, "backfill_ticker", lambda t, *a, **k: calls.append(t) or
                        backfill.BackfillResult(t, "news", 1, 1))
    monkeypatch.setattr(jobs.articles, "rescore_stored_articles", lambda: 0)
    store.save_watchlist({"SHOP": ["Shopify"]})
    store.create_job("NOPE", ["Not added"], ["news"], 1)
    store.create_job("SHOP", ["Shopify"], ["news"], 1)
    jobs.run_worker()
    assert calls == ["SHOP"]
    assert [j["ticker"] for j in store.jobs_with_status("cancelled")] == ["NOPE"]


def test_removing_a_company_stops_its_running_download(temp_data, monkeypatch):
    store.save_watchlist({"SHOP": ["Shopify"]})
    job_id = store.create_job("SHOP", ["Shopify"], ["news"], 1)
    searched = []

    def fake_fetch(ticker, names, start, end, kind):
        searched.append(start)
        if len(searched) == 3:      # the user removes SHOP mid-download
            store.update_job(job_id, status="cancelled", message="Cancelled: removed from the watchlist")
        return []

    monkeypatch.setattr(jobs.backfill.news, "fetch_window", fake_fetch)
    monkeypatch.setattr(jobs.backfill.time, "sleep", lambda s: None)
    monkeypatch.setattr(jobs, "HEARTBEAT_EVERY_S", 0)
    monkeypatch.setattr(jobs.articles, "rescore_stored_articles", lambda: 0)
    jobs.run_worker()
    assert len(searched) <= 4 < len(backfill.windows(1, date.today()))
    assert store.job_status(job_id) == "cancelled"


def test_investment_funds_are_not_companies():
    other = OTHER.replace("File Creation Time", "HYI|Western Asset High Yield Opportunity Fund Inc.|N|HYI|N|100|N|HYI\nFile Creation Time")
    assert "HYI" not in listings.parse(NASDAQ, other)["yahoo"].tolist()
