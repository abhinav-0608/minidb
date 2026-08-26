"""MiniDB benchmark - sequential scan vs. B+ tree index.

Builds a fresh database of N rows and, for a point query ``WHERE id = k``,
times it BOTH ways: forced full scan vs. index lookup. Also reports the
B+ tree's height and node capacities so you can see the tree stay shallow as
N grows.

All numbers come from running the real implementation. Nothing is faked.

Run from the repo root::

    python -m bench.benchmark                 # 1k / 10k / 100k rows
    python -m bench.benchmark --rows 1000,5000 --page-size 4096
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


def build(path: str, n: int, page_size: int) -> tuple[float, int]:
    # wal=False: an fsync per insert would dominate; this measures storage +
    # index cost. The WAL's cost is discussed separately (Stage 10).
    with Database(path, page_size=page_size, wal=False) as db:
        db.create_table("t", SCHEMA)
        table = db.open_table("t")
        t0 = time.perf_counter()
        for i in range(n):
            table.insert((i, f"user{i}", i % 100))
        return time.perf_counter() - t0, db.page_count


def best_ms(fn, trials: int = 7) -> float:
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1000,10000,100000")
    ap.add_argument("--page-size", type=int, default=4096)
    args = ap.parse_args()
    counts = [int(x) for x in args.rows.split(",")]
    ps = args.page_size

    print("# MiniDB benchmark - sequential scan vs. B+ tree index")
    print(f"# {platform.platform()}  |  Python {platform.python_version()}")
    print(f"# page_size = {ps}")
    print()
    hdr = (
        f"{'rows':>8} {'pages':>7} {'height':>6} {'leaf/int cap':>13} "
        f"{'insert/s':>9} {'scan id=k':>11} {'index id=k':>11} {'speedup':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for n in counts:
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "bench.db")
            insert_s, pages = build(path, n, ps)
            with Database(path, page_size=ps) as db:
                idx = db.open_table("t").index
                height = idx.height()
                caps = f"{idx.leaf_capacity}/{idx.internal_capacity}"
                k = n - 1  # a present key, worst case for the scan
                q = SelectQuery("t", None, Condition("id", "=", k))
                ms_scan = best_ms(lambda: execute_select(db, q, use_index=False))
                ms_index = best_ms(lambda: execute_select(db, q, use_index=True))
            rate = n / insert_s if insert_s else 0.0
            speed = ms_scan / ms_index if ms_index else float("inf")
            print(
                f"{n:>8} {pages:>7} {height:>6} {caps:>13} "
                f"{rate:>9.0f} {ms_scan:>10.2f}m {ms_index:>10.3f}m {speed:>8.0f}x"
            )

    print()
    print("scan id=k walks every row; index id=k is height page reads + 1 heap")
    print("read. The tree gains ~one level per ~300x more rows.")


if __name__ == "__main__":
    main()
