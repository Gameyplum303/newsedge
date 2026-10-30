"""Back-fill years of historical headlines from the command line (resumable).

    python backfill.py                 # 10 years of news + incident searches for the saved watchlist
    python backfill.py --years 5 --tickers AAPL MSFT --kinds incidents

Safe to stop and re-run: finished weeks are remembered and skipped.
"""
import argparse
import threading
import time

from engine import articles, backfill, store


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tickers", nargs="*", help="default: the saved watchlist")
    ap.add_argument("--kinds", nargs="*", default=list(backfill.KINDS), choices=list(backfill.KINDS))
    args = ap.parse_args()

    watchlist = store.load_watchlist()
    if args.tickers:
        watchlist = {t: watchlist.get(t, []) for t in args.tickers}

    status, lock, started = {}, threading.Lock(), time.time()

    def progress(label, done, total):
        with lock:
            status[label] = (done, total)
            if done % 25 == 0 or done == total:
                overall = sum(d for d, _ in status.values()), sum(t for _, t in status.values())
                print(f"[{time.time() - started:6.0f}s] {label:16s} {done}/{total} windows | "
                      f"all tickers {overall[0]}/{overall[1]}", flush=True)

    print(f"Back-filling {args.years:g} years ({', '.join(args.kinds)}) for {', '.join(watchlist)}", flush=True)
    results = backfill.backfill(watchlist, args.years, kinds=args.kinds, workers=args.workers, progress=progress)

    print("\nticker  kind       fetched  skipped  new articles")
    for r in results:
        note = "  <- stopped: Google rate limit, run again later to resume" if r.stopped_early else ""
        print(f"{r.ticker:6s}  {r.kind:9s}  {r.windows_fetched:7d}  {r.windows_skipped:7d}  {r.new_articles:12,d}{note}")
    print(f"\nTotal new articles: {sum(r.new_articles for r in results):,} in {time.time() - started:.0f}s")
    print(f"Re-scored {articles.rescore_stored_articles():,} stored headlines with the current scorer.")


if __name__ == "__main__":
    main()
