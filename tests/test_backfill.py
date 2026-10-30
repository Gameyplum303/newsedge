from datetime import date, timedelta

import pytest

from engine import backfill


def test_windows_are_contiguous_weeks_newest_first():
    w = backfill.windows(1, today=date(2026, 10, 20))
    assert w[0] == (date(2026, 10, 14), date(2026, 10, 21))
    assert all(a[0] == b[1] for a, b in zip(w, w[1:]))            # no gaps or overlaps
    assert all((e - s).days <= 7 for s, e in w)
    assert w[-1][0] == date(2026, 10, 21) - timedelta(days=365)   # covers exactly one year
    assert len(backfill.windows(10, today=date(2026, 10, 20))) == 522
    assert len(backfill.windows(10, today=date(2026, 10, 20), window_days=14)) == 261


@pytest.fixture
def fake_store(monkeypatch):
    marked, saved = {}, []
    monkeypatch.setattr(backfill.store, "completed_ranges",
                        lambda t, k: [(s, e) for (tt, kk, s), e in marked.items() if (tt, kk) == (t, k)])
    monkeypatch.setattr(backfill.store, "mark_window", lambda t, s, e, n, k: marked.__setitem__((t, k, s), e))
    monkeypatch.setattr(backfill.store, "save_articles", lambda arts: saved.extend(arts) or len(arts))
    monkeypatch.setattr(backfill.time, "sleep", lambda s: None)
    return marked, saved


def test_resume_skips_finished_weeks_and_refetches_recent_ones(monkeypatch, fake_store):
    marked, _ = fake_store
    calls = []
    monkeypatch.setattr(backfill.news, "fetch_window", lambda t, n, s, e, k: calls.append(s) or [{"id": s}])
    today = date(2026, 10, 20)

    first = backfill.backfill_ticker("AAA", ["A"], 0.1, today=today)
    assert first.windows_fetched == first.windows_total and first.new_articles == first.windows_total
    assert len(marked) == first.windows_total - 1      # the current week is not marked done

    calls.clear()
    second = backfill.backfill_ticker("AAA", ["A"], 0.1, today=today)
    assert second.windows_skipped == first.windows_total - 1 and len(calls) == 1


def test_gives_up_when_rate_limited(monkeypatch, fake_store):
    monkeypatch.setattr(backfill.news, "fetch_window", lambda *a: None)
    r = backfill.backfill_ticker("AAA", ["A"], 1, today=date(2026, 10, 20))
    assert r.stopped_early and r.windows_fetched == 0
    assert not fake_store[0]


def test_news_and_incident_progress_are_tracked_separately(monkeypatch, fake_store):
    monkeypatch.setattr(backfill.news, "fetch_window", lambda *a: [{"id": "x"}])
    today = date(2026, 10, 20)
    backfill.backfill_ticker("AAA", ["A"], 0.1, "news", today=today)
    inc = backfill.backfill_ticker("AAA", ["A"], 0.1, "incidents", today=today)
    assert inc.windows_skipped == 0 and inc.windows_fetched == inc.windows_total

def test_related_search_is_skipped_when_there_is_nothing_to_search(monkeypatch, fake_store):
    calls = []
    monkeypatch.setattr(backfill.news, "fetch_window", lambda *a: calls.append(a) or [])
    r = backfill.backfill_ticker("ZZZZ", ["Nobody"], 1, "related", today=date(2026, 10, 20))
    assert r.windows_total == 0 and not calls          # no 26 empty searches with pauses in between


def test_resuming_on_a_later_day_skips_history_already_covered(monkeypatch, fake_store):
    calls = []
    monkeypatch.setattr(backfill.news, "fetch_window", lambda t, n, s, e, k: calls.append(s) or [])
    backfill.backfill_ticker("AAA", ["A"], 0.5, "related", today=date(2026, 10, 20))
    monkeypatch.setattr(backfill.profiles, "related_terms", lambda t: ["x"])
    calls.clear()
    backfill.backfill_ticker("AAA", ["A"], 0.5, "related", today=date(2026, 10, 20))   # first real run
    first = len(calls)
    calls.clear()
    later = backfill.backfill_ticker("AAA", ["A"], 0.5, "related", today=date(2026, 10, 21))   # windows shift a day
    assert first > 10 and len(calls) <= 2 and later.windows_skipped >= later.windows_total - 2
