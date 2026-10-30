from datetime import date

import pytest

from engine import backfill, companies, config, jobs, linking, profiles, store


@pytest.fixture
def temp_data(tmp_path, monkeypatch):
    """Point the database and profiles file at a temporary folder."""
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(config, "WATCHLIST_PATH", str(tmp_path / "watchlist.json"))
    store.save_watchlist({t: [] for t in ("SHOP", "UBER", "BAD")})   # jobs only run for watchlist companies
    monkeypatch.setattr(profiles, "PROFILES_PATH", str(tmp_path / "profiles.json"))
    profiles._cache.update(mtime=None, data={})
    yield tmp_path
    profiles._cache.update(mtime=None, data={})


# ---------------------------------------------------------------- profiles
def test_builtin_profiles_and_user_overrides(temp_data):
    assert profiles.government("LMT") == {"defense": 1.0}
    assert "AWS" in profiles.aliases("AMZN")
    profiles.save_profile("LMT", government={"defense": 0.5})
    assert profiles.government("LMT") == {"defense": 0.5}
    assert profiles.relationships("LMT") == companies.RELATIONSHIPS["LMT"]   # untouched fields keep defaults
    assert profiles.profile("LMT")["custom"]


def test_profile_edits_change_news_links_without_restart(temp_data):
    m = linking.build_matchers({"SHOP": ["Shopify"]})["SHOP"]
    headline = "Stripe beats estimates as payment volumes surge"
    assert linking.link(m, "SHOP", headline, "", "earnings_beat").relevance == 0
    profiles.save_profile("SHOP", relationships=[{"names": ["Stripe"], "relation": "partner", "weight": 0.6}])
    lk = linking.link(m, "SHOP", headline, "", "earnings_beat")
    assert (lk.kind, lk.via) == ("relation", "Stripe (partner)")
    assert "Stripe" in profiles.related_terms("SHOP")


@pytest.mark.parametrize("industry, description, expected", [
    ("Aerospace & Defense", "", {"defense": 0.6}),
    ("Software—Application", "Serves U.S. government agencies and the Department of Defense.",
     {"defense": 0.3, "civilian": 0.2}),
    ("Healthcare Plans", "Medicare Advantage plans", {"healthcare": 0.5}),
    ("Internet Retail", "Online marketplace for consumers.", {}),
])
def test_government_suggestion(industry, description, expected):
    assert profiles.suggest_government(industry, description) == expected


# ---------------------------------------------------------------- job queue
def test_setup_needed_until_history_is_downloaded(temp_data):
    assert jobs.needs_setup("SHOP")
    for start, end in backfill.windows(10, date.today())[: int(522 * 0.95)]:
        store.mark_window("SHOP", start.isoformat(), end.isoformat(), 10, "news")
    assert not jobs.needs_setup("SHOP") and jobs.history_coverage("SHOP") > 0.9


def test_enqueue_reuses_open_job_and_starts_one_worker(temp_data, monkeypatch):
    spawned = []
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    first = jobs.enqueue("SHOP", ["Shopify"])
    again = jobs.enqueue("SHOP", ["Shopify"])
    jobs.enqueue("UBER", ["Uber"])
    assert first == again and len(spawned) == 1          # second call sees the worker's heartbeat
    assert [j["ticker"] for j in store.jobs_with_status("queued")] == ["SHOP", "UBER"]


def test_worker_runs_jobs_in_order_and_reports_progress(temp_data, monkeypatch):
    calls = []

    def fake_backfill(ticker, names, years, kind, progress=None, **_):
        total = len(backfill.windows(years, None, backfill.WINDOW_DAYS[kind]))
        progress(f"{ticker} {kind}", total, total)
        calls.append((ticker, kind))
        return backfill.BackfillResult(ticker, kind, total, total, 0, 5)

    monkeypatch.setattr(jobs.backfill, "backfill_ticker", fake_backfill)
    monkeypatch.setattr(jobs.articles, "rescore_stored_articles", lambda: 0)
    monkeypatch.setattr(jobs, "HEARTBEAT_EVERY_S", 0)
    a = store.create_job("SHOP", ["Shopify"], ["news", "incidents"], 1)
    b = store.create_job("UBER", ["Uber"], ["news"], 1)
    jobs.run_worker()

    done = {j["id"]: j for j in store.jobs_with_status("done")}
    assert set(done) == {a, b} and done[a]["progress"] == 1.0 and "10 new articles" in done[a]["message"]
    assert [c[0] for c in calls] == ["SHOP", "SHOP", "UBER"]
    assert store.get_meta("worker_heartbeat") == ""        # worker released when the queue emptied


def test_rate_limited_job_pauses_and_crashed_jobs_resume(temp_data, monkeypatch):
    monkeypatch.setattr(jobs.backfill, "backfill_ticker",
                        lambda t, n, y, k, **_: backfill.BackfillResult(t, k, 10, stopped_early=True))
    monkeypatch.setattr(jobs.articles, "rescore_stored_articles", lambda: 0)
    crashed = store.create_job("SHOP", ["Shopify"], ["news"], 1)
    store.update_job(crashed, status="running")             # left behind by a worker that died
    jobs.run_worker()
    (job,) = store.jobs_with_status("stopped")
    assert job["id"] == crashed and "rate-limiting" in job["message"]


def test_failed_job_does_not_stop_the_queue(temp_data, monkeypatch):
    def boom(ticker, *a, **k):
        if ticker == "BAD":
            raise RuntimeError("network down")
        return backfill.BackfillResult(ticker, "news", 1, 1)

    monkeypatch.setattr(jobs.backfill, "backfill_ticker", boom)
    monkeypatch.setattr(jobs.articles, "rescore_stored_articles", lambda: 0)
    store.create_job("BAD", [], ["news"], 1)
    store.create_job("SHOP", ["Shopify"], ["news"], 1)
    jobs.run_worker()
    assert [j["ticker"] for j in store.jobs_with_status("failed")] == ["BAD"]
    assert [j["ticker"] for j in store.jobs_with_status("done")] == ["SHOP"]


# ---------------------------------------------------------------- adding a company
def _found(name="Palantir Technologies Inc.", industry="Software—Infrastructure", description=""):
    return {"ticker": "X", "names": [name], "sector": "Technology", "industry": industry, "description": description}


def test_set_up_new_company_saves_suggested_profile(temp_data, monkeypatch):
    monkeypatch.setattr(profiles, "lookup", lambda t: _found(description="Serves the Department of Defense."))
    names = profiles.set_up_company("PLTRX", "Palantir Technologies")
    assert names == ["Palantir Technologies", "Palantir"]
    assert profiles.government("PLTRX") == {"defense": 0.3}


def test_set_up_builtin_company_keeps_curated_names_and_profile(temp_data, monkeypatch):
    monkeypatch.setattr(profiles, "lookup", lambda t: _found())
    assert profiles.set_up_company("LMT", "Lockheed Martin") == companies.COMPANIES["LMT"]
    assert not profiles.profile("LMT")["custom"]


def test_set_up_company_without_price_history(temp_data, monkeypatch):
    monkeypatch.setattr(profiles, "lookup", lambda t: None)
    assert profiles.set_up_company("NEWIPO", "New IPO") is None
