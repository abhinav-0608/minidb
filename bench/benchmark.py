"""Stage 4 baseline benchmark - sequential scan only.

Shows how a full-table SELECT scales with row count BEFORE any index exists.
Stage 6 re-runs this next to an indexed lookup; Stage 10 is the full suite.

All numbers come from running the real implementation. Nothing is faked.

Run from the repo root::

    python -m bench.benchmark                 # 1k / 10k / 100k rows
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


def build(path: str, n: int) -> tuple[float, int]:
    with Database(path) as db:
        db.create_table("t", SCHEMA)
        table = db.open_table("t")
        t0 = time.perf_counter()
        for i in range(n):
            table.insert((i, f"user{i}", i % 100))
        insert_s = time.perf_counter() - t0
        return insert_s, db.page_count


def best_ms(fn, trials: int = 5) -> float:
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1000,10000,100000")
    counts = [int(x) for x in ap.parse_args().rows.split(",")]

    print("# MiniDB Stage 4 baseline - sequential scan only")
    print(f"# {platform.platform()}")
    print(f"# Python {platform.python_version()}")
    print()
    hdr = (
        f"{'rows':>8} {'pages':>7} {'insert/s':>10} "
        f"{'SELECT * (ms)':>15} {'id=last (ms)':>14} {'id=absent (ms)':>16}"
    )
    print(hdr)
    print("-" * len(hdr))

    for n in counts:
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "bench.db")
            insert_s, pages = build(path, n)
            with Database(path) as db:
                q_all = SelectQuery("t", None)
                q_hit = SelectQuery("t", None, Condition("id", "=", n - 1))
                q_miss = SelectQuery("t", None, Condition("id", "=", -1))
                ms_all = best_ms(lambda: execute_select(db, q_all))
                ms_hit = best_ms(lambda: execute_select(db, q_hit))
                ms_miss = best_ms(lambda: execute_select(db, q_miss))
            rate = n / insert_s if insert_s else 0.0
            print(
                f"{n:>8} {pages:>7} {rate:>10.0f} "
                f"{ms_all:>15.2f} {ms_hit:>14.2f} {ms_miss:>16.2f}"
            )

    print()
    print("SELECT * materialises every row; WHERE id=... still scans every row")
    print("(no index yet). All three times should grow ~linearly with rows.")


if __name__ == "__main__":
    main()
