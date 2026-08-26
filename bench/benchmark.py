"""MiniDB benchmark.

Three experiments, all run against the real implementation - no faked numbers:

1. scan vs. index   - point-query latency as the table grows (1k/10k/100k)
2. WAL cost         - insert throughput with durability on vs. off
3. page size        - how 1K/4K/8K pages change tree height and lookup cost

Run from the repo root::

    python -m bench.benchmark                 # all three
    python -m bench.benchmark --quick         # skip the 100k row case
    python -m bench.benchmark --rows 1000,5000
"""

from __future__ import annotations

import argparse
import platform
import tempfile
import time
from pathlib import Path

from minidb.database import Database
from minidb.executor import execute_select
from minidb.query import Condition, SelectQuery
from minidb.record import Column, ColumnType, Schema

SCHEMA = Schema(
    (
        Column("id", ColumnType.INT),
        Column("name", ColumnType.TEXT),
        Column("age", ColumnType.INT),
    )
)


def _fill(db: Database, n: int) -> float:
    table = db.open_table("t")
    t0 = time.perf_counter()
    for i in range(n):
        table.insert((i, f"user{i}", i % 100))
    return time.perf_counter() - t0


def build(path: str, n: int, page_size: int, *, wal: bool = False) -> tuple[float, int]:
    with Database(path, page_size=page_size, wal=wal) as db:
        db.create_table("t", SCHEMA)
        secs = _fill(db, n)
        return secs, db.page_count


def best_ms(fn, trials: int = 7) -> float:
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def scan_vs_index(counts: list[int], page_size: int) -> None:
    print("## 1. sequential scan vs. B+ tree index  (point query: WHERE id = k)")
    print()
    hdr = (
        f"{'rows':>8} {'pages':>7} {'height':>6} {'leaf/int cap':>13} "
        f"{'insert/s':>9} {'scan (ms)':>10} {'index (ms)':>11} {'speedup':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in counts:
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "b.db")
            insert_s, pages = build(path, n, page_size)
            with Database(path, page_size=page_size, wal=False) as db:
                idx = db.open_table("t").index
                height, caps = idx.height(), f"{idx.leaf_capacity}/{idx.internal_capacity}"
                q = SelectQuery("t", None, Condition("id", "=", n - 1))
                ms_scan = best_ms(lambda: execute_select(db, q, use_index=False))
                ms_index = best_ms(lambda: execute_select(db, q, use_index=True))
            rate = n / insert_s if insert_s else 0.0
            speed = ms_scan / ms_index if ms_index else float("inf")
            print(
                f"{n:>8} {pages:>7} {height:>6} {caps:>13} "
                f"{rate:>9.0f} {ms_scan:>10.2f} {ms_index:>11.3f} {speed:>8.0f}x"
            )
    print()


def wal_cost(n: int = 2000) -> None:
    print(f"## 2. the WAL's cost  (insert throughput, {n} rows)")
    print()
    for wal in (False, True):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "b.db")
            insert_s, _ = build(path, n, 4096, wal=wal)
        label = "wal=on  (fsync per commit)" if wal else "wal=off (bulk load)"
        print(f"  {label:<28} {n / insert_s:>9.0f} inserts/s")
    print()


def page_size_sweep(n: int = 10_000) -> None:
    print(f"## 3. page size  ({n} rows, index lookup WHERE id = k)")
    print()
    hdr = (
        f"{'page':>6} {'pages':>7} {'height':>6} {'leaf/int cap':>13} "
        f"{'index (ms)':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ps in (1024, 4096, 8192):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "b.db")
            build(path, n, ps)
            with Database(path, page_size=ps, wal=False) as db:
                idx = db.open_table("t").index
                height = idx.height()
                caps = f"{idx.leaf_capacity}/{idx.internal_capacity}"
                pages = db.page_count
                q = SelectQuery("t", None, Condition("id", "=", n - 1))
                ms = best_ms(lambda: execute_select(db, q, use_index=True))
            print(f"{ps:>6} {pages:>7} {height:>6} {caps:>13} {ms:>11.3f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1000,10000,100000")
    ap.add_argument("--quick", action="store_true", help="skip the 100k row case")
    args = ap.parse_args()
    counts = [int(x) for x in args.rows.split(",")]
    if args.quick:
        counts = [c for c in counts if c <= 10_000]

    print(f"# MiniDB benchmark")
    print(f"# {platform.platform()}  |  Python {platform.python_version()}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M')}")
    print()
    scan_vs_index(counts, 4096)
    wal_cost()
    page_size_sweep()
    print("notes:")
    print("  - scan is O(rows); index lookup is ~height page reads + 1 heap read")
    print("  - the tree gains one level per ~300x more rows (fanout ~300 at 4K)")
    print("  - wal=on pays one fsync per statement - the price of durability")
    print("  - bigger pages -> higher fanout -> shorter tree, but more I/O per node")


if __name__ == "__main__":
    main()
