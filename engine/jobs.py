"""Background data set-up for companies: download their history so the model can retrain.

Nothing is downloaded for a company until it is added. Adding one queues a job; removing it cancels
its job. A single background worker process (`python -m engine.jobs`) runs
queued jobs one company at a time, so Google News isn't hit by several downloads at once. It
reports progress to the database, survives the dashboard being closed or reloaded, and resumes
interrupted jobs. The dashboard retrains the model when a job finishes.
"""
import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timezone

from . import articles, backfill, config, store

DEFAULT_YEARS = 10
HEARTBEAT_EVERY_S = 2
STALE_AFTER_S = 300      # longer than the slowest rate-limit back-off in backfill
READY_COVERAGE = 0.9     # a company counts as set up once 90% of its 10-year news history is in


def _now() -> datetime:
    return datetime.now(timezone.utc)


def expected_windows(ticker: str, years: float, kinds, today: date | None = None) -> dict[str, int]:
    """Searches each applicable kind needs."""
    return {k: len(backfill.windows(years, today, backfill.WINDOW_DAYS[k]))
            for k in backfill.applicable_kinds(ticker, kinds)}


def history_coverage(ticker: str, years: float = DEFAULT_YEARS, today: date | None = None) -> float:
    """Share of the company's news history already downloaded (0..1)."""
    return backfill.coverage(ticker, "news", years, today)


def needs_setup(ticker: str, years: float = DEFAULT_YEARS) -> bool:
    return history_coverage(ticker, years) < READY_COVERAGE


class Cancelled(Exception):
    """The company was removed from the watchlist while its history was downloading."""


# ---------------------------------------------------------------- queue (called from the dashboard)
def sync_watchlist(old: dict[str, list[str]], new: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """After a watchlist change: download history for added companies, cancel it for removed ones.

    Returns (tickers queued, tickers cancelled).
    """
    queued = [t for t in new if t not in old and needs_setup(t)]
    for t in queued:
        enqueue(t, new[t])
    cancelled = [t for t in old if t not in new and cancel(t)]
    return queued, cancelled


def cancel(ticker: str) -> bool:
    """Cancel the open job for a ticker, if any. A running download stops within a search or two."""
    job = store.open_job(ticker)
    if job:
        store.update_job(job["id"], status="cancelled", message="Cancelled: removed from the watchlist")
    return job is not None


def enqueue(ticker: str, names: list[str], kinds=backfill.KINDS, years: float = DEFAULT_YEARS) -> int:
    """Queue a set-up job (or return the one already open for this ticker) and make sure a worker runs."""
    job = store.open_job(ticker)
    job_id = job["id"] if job else store.create_job(ticker, names, kinds, years)
    ensure_worker()
    return job_id


def worker_alive() -> bool:
    beat = store.get_meta("worker_heartbeat")
    return bool(beat) and (_now() - datetime.fromisoformat(beat)).total_seconds() < STALE_AFTER_S


def ensure_worker() -> None:
    """Start the background worker unless one is already running."""
    if worker_alive() or not store.jobs_with_status("queued"):
        return
    store.set_meta("worker_heartbeat", _now().isoformat())   # claim first, so two clicks start one worker
    os.makedirs(config.DATA_DIR, exist_ok=True)
    detach = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
              if os.name == "nt" else {"start_new_session": True})
    # The worker gets its own copy of the handle, so this process can close its copy right away.
    with open(os.path.join(config.DATA_DIR, "jobs.log"), "a", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, "-m", "engine.jobs"], cwd=config.BASE_DIR, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, **detach)


# ---------------------------------------------------------------- worker
def _beat() -> None:
    store.set_meta("worker_heartbeat", _now().isoformat())


def run_job(job: dict) -> None:
    ticker, names, years = job["ticker"], job["names"], job["years"] or DEFAULT_YEARS
    expected = expected_windows(ticker, years, job["kinds"] or backfill.KINDS)
    total = sum(expected.values()) or 1
    done = dict.fromkeys(expected, 0)
    lock, last, stop = threading.Lock(), [0.0], threading.Event()

    def progress(label, n, _total):
        if stop.is_set():
            raise Cancelled
        with lock:
            done[label.split()[-1]] = n
            if time.time() - last[0] >= HEARTBEAT_EVERY_S:
                last[0] = time.time()
                if store.job_status(job["id"]) == "cancelled":
                    stop.set()
                    raise Cancelled
                finished = sum(done.values())
                store.update_job(job["id"], progress=finished / total,
                                 message=f"Downloading {years:g} years of history: {finished:,}/{total:,} searches")
                _beat()

    store.update_job(job["id"], status="running", progress=0.0, message="Starting download")
    results = backfill.backfill({ticker: names}, years, kinds=list(expected), workers=max(1, len(expected)),
                                progress=progress)
    store.update_job(job["id"], message="Updating article scores")
    articles.rescore_stored_articles()

    added = sum(r.new_articles for r in results)
    if any(r.stopped_early for r in results):
        store.update_job(job["id"], status="stopped", progress=sum(done.values()) / total,
                         message=f"Paused: Google News is rate-limiting. {added:,} articles saved so far; "
                                 "resume later and finished weeks are kept.")
    else:
        store.update_job(job["id"], status="done", progress=1.0,
                         message=f"Ready: {added:,} new articles. The model retrains with them automatically.")


def run_worker() -> None:
    """Run queued jobs until none are left. Jobs left 'running' by a crashed worker are resumed."""
    _beat()
    for job in store.jobs_with_status("running"):
        store.update_job(job["id"], status="queued", message="Resuming after an interruption")
    while (job := store.next_queued_job()) is not None:
        if job["ticker"] not in store.load_watchlist():
            # Only download for companies on the watchlist.
            store.update_job(job["id"], status="cancelled", message="Cancelled: not on the watchlist")
            continue
        try:
            run_job(job)
        except Cancelled:
            pass
        except Exception as e:   # keep going with the next company
            store.update_job(job["id"], status="failed", message=f"Failed: {e}"[:300])
        _beat()
    store.set_meta("worker_heartbeat", "")


if __name__ == "__main__":
    print(f"[{_now():%Y-%m-%d %H:%M:%S}] worker started", flush=True)
    run_worker()
    print(f"[{_now():%Y-%m-%d %H:%M:%S}] worker finished", flush=True)
