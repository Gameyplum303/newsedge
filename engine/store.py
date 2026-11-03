"""SQLite persistence: articles, history-download progress, background jobs, settings; plus the watchlist file."""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import companies, config
from .sanitize import valid_ticker

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    title TEXT,
    summary TEXT,
    link TEXT,
    source TEXT,
    published TEXT,
    sentiment REAL,
    category TEXT,
    relevance REAL,
    source_weight REAL,
    fetched_at TEXT,
    PRIMARY KEY (id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_articles_ticker_pub ON articles (ticker, published);
CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles (fetched_at);   -- finds new articles to re-score
CREATE TABLE IF NOT EXISTS backfill_windows (   -- legacy (news searches only); read, no longer written
    ticker TEXT NOT NULL,
    start TEXT NOT NULL,
    "end" TEXT NOT NULL,
    articles INTEGER,
    fetched_at TEXT,
    PRIMARY KEY (ticker, start)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);   -- scorer version, worker heartbeat
CREATE TABLE IF NOT EXISTS jobs (               -- background data set-up for a company
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    names TEXT,                  -- JSON list of company names
    kinds TEXT,                  -- comma-separated back-fill kinds
    years REAL,
    status TEXT NOT NULL,        -- queued / running / done / stopped / failed / cancelled
    progress REAL DEFAULT 0,
    message TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS backfill_progress (
    kind TEXT NOT NULL,          -- news / incidents / related
    ticker TEXT NOT NULL,
    start TEXT NOT NULL,
    "end" TEXT NOT NULL,
    articles INTEGER,
    fetched_at TEXT,
    PRIMARY KEY (kind, ticker, start)
);
"""

COLUMNS = ["id", "ticker", "title", "summary", "link", "source", "published",
           "sentiment", "category", "relevance", "source_weight"]


def _connect() -> sqlite3.Connection:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    # The dashboard, a back-fill and re-scoring can write at the same time: wait for locks instead
    # of failing, and use WAL so readers and writers don't block each other.
    con = sqlite3.connect(config.DB_PATH, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def save_articles(articles: list[dict]) -> int:
    """Insert new articles; returns how many were new."""
    if not articles:
        return 0
    fetched = _now()
    rows = [(a["id"], a["ticker"], a["title"], a["summary"], a["link"], a["source"],
             a["published"].isoformat(), a["sentiment"], a["category"], a["relevance"],
             a["source_weight"], fetched) for a in articles]
    with _connect() as con:
        before = con.total_changes
        con.executemany("INSERT OR IGNORE INTO articles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        return con.total_changes - before


def get_meta(key: str) -> str | None:
    with _connect() as con:
        row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(key: str, value: str) -> None:
    with _connect() as con:
        con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))


def articles_fetched_since(since: str | None) -> pd.DataFrame:
    """Distinct stored headlines (id, title, summary) fetched after `since` (ISO), or all if None.

    Rows are fetched through the fetched_at index and grouped in pandas: grouping in SQL makes SQLite
    walk the whole table in id order, which takes minutes on a hard drive.
    """
    sql = "SELECT id, title, summary, fetched_at FROM articles" + (" WHERE fetched_at > ?" if since else "")  # nosec B608: fixed text, value is a parameter
    with _connect() as con:
        rows = pd.read_sql_query(sql, con, params=[since] if since else None)
    rows["summary"] = rows["summary"].fillna("")
    return rows.groupby("id", as_index=False).agg({"title": "max", "summary": "max", "fetched_at": "max"})


def update_scores(rows: list[tuple[float, str, str]], batch: int = 2000) -> None:
    """rows: (sentiment, category, id); applies to every ticker row of that headline.

    Commits in batches so a long re-score never holds the write lock for more than a moment.
    """
    for i in range(0, len(rows), batch):
        with _connect() as con:
            con.executemany("UPDATE articles SET sentiment = ?, category = ? WHERE id = ?", rows[i:i + batch])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(ticker: str, names: list[str], kinds, years: float) -> int:
    with _connect() as con:
        cur = con.execute("INSERT INTO jobs (ticker, names, kinds, years, status, progress, message, created_at, "
                          "updated_at) VALUES (?, ?, ?, ?, 'queued', 0, 'Waiting to start', ?, ?)",
                          (ticker, json.dumps(list(names)), ",".join(kinds), years, _now(), _now()))
        return cur.lastrowid


JOB_FIELDS = {"status", "progress", "message"}   # column names can't be SQL parameters: only these


def update_job(job_id: int, **fields) -> None:
    unknown = set(fields) - JOB_FIELDS
    if unknown:
        raise ValueError(f"not a job field: {', '.join(sorted(unknown))}")
    fields["updated_at"] = _now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as con:
        # nosec B608: column names were checked against JOB_FIELDS above; values are parameters.
        con.execute(f"UPDATE jobs SET {cols} WHERE id = ?", [*fields.values(), job_id])  # nosec B608


def _job_rows(sql: str, params=()) -> list[dict]:
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(sql, params)]
    for r in rows:
        r["names"] = json.loads(r["names"] or "[]")
        r["kinds"] = [k for k in (r["kinds"] or "").split(",") if k]
    return rows


def job_status(job_id: int) -> str | None:
    with _connect() as con:
        row = con.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row[0] if row else None


def next_queued_job() -> dict | None:
    rows = _job_rows("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1")
    return rows[0] if rows else None


def open_job(ticker: str) -> dict | None:
    """The queued or running job for a ticker, if any."""
    rows = _job_rows("SELECT * FROM jobs WHERE ticker = ? AND status IN ('queued', 'running') ORDER BY id DESC LIMIT 1",
                     (ticker,))
    return rows[0] if rows else None


def jobs_with_status(*statuses: str) -> list[dict]:
    marks = ",".join("?" * len(statuses))
    return _job_rows(f"SELECT * FROM jobs WHERE status IN ({marks}) ORDER BY id", statuses)  # nosec B608: only ? placeholders


def recent_jobs(hours: float = 24) -> list[dict]:
    """Latest job per ticker that is open or changed within `hours`."""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return _job_rows("SELECT * FROM jobs WHERE id IN (SELECT MAX(id) FROM jobs GROUP BY ticker) "
                     "AND (status IN ('queued', 'running') OR updated_at >= ?) ORDER BY id", (since,))


def reset_windows(ticker: str, kind: str) -> None:
    """Forget back-fill progress so the next run searches that history again."""
    with _connect() as con:
        con.execute("DELETE FROM backfill_progress WHERE ticker = ? AND kind = ?", (ticker, kind))
        if kind == "news":
            con.execute("DELETE FROM backfill_windows WHERE ticker = ?", (ticker,))


def completed_ranges(ticker: str, kind: str = "news") -> list[tuple[str, str]]:
    """(start, end) ISO dates of history windows already back-filled for a ticker and search kind."""
    with _connect() as con:
        done = con.execute('SELECT start, "end" FROM backfill_progress WHERE kind = ? AND ticker = ?',
                           (kind, ticker)).fetchall()
        if kind == "news":
            done += con.execute('SELECT start, "end" FROM backfill_windows WHERE ticker = ?', (ticker,)).fetchall()
    return done


def mark_window(ticker: str, start: str, end: str, articles: int, kind: str = "news") -> None:
    with _connect() as con:
        con.execute("INSERT OR REPLACE INTO backfill_progress VALUES (?,?,?,?,?,?)",
                    (kind, ticker, start, end, articles, _now()))


def load_articles(tickers: list[str] | None = None) -> pd.DataFrame:
    """Stored articles, newest first. Read in one sequential pass and filtered in pandas: looking rows
    up through the ticker index jumps all over the file, which is very slow on a hard drive."""
    with _connect() as con:
        df = pd.read_sql_query("SELECT * FROM articles", con)
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["published"] = pd.to_datetime(df["published"], utc=True, format="ISO8601")
    return df.sort_values("published", ascending=False).reset_index(drop=True)


def _clean_watchlist(data) -> dict[str, list[str]]:
    """Valid tickers with string names only (the file can be edited by hand)."""
    if not isinstance(data, dict):
        return {}
    return {t: [n for n in names if isinstance(n, str) and n.strip()][:10]
            for t, names in data.items() if valid_ticker(t) and isinstance(names, list)}


def load_watchlist() -> dict[str, list[str]]:
    """Ticker -> company names used for matching."""
    try:
        with open(config.WATCHLIST_PATH, encoding="utf-8") as f:
            data = _clean_watchlist(json.load(f))
        if data:
            return data
    except (OSError, ValueError):
        pass
    return {t: companies.COMPANIES.get(t, []) for t in companies.DEFAULT_WATCHLIST}


def save_watchlist(watchlist: dict[str, list[str]]) -> None:
    bad = [t for t in watchlist if not valid_ticker(t)]
    if bad:
        raise ValueError(f"invalid ticker: {', '.join(map(str, bad))}")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, indent=2)
