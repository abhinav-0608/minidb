"""Stage 6 tests - Table (heap + id index kept in step).

* TestInsertScan   - insert returns a RID, scan keeps order, index gets the id
* TestUniqueId     - a duplicate id is refused and never reaches the heap
* TestLookupById   - index lookup == what a scan+filter would return
* TestConsistency  - the heap and the index describe the same set of rows
* TestPersistence  - a table (via Database) survives a restart intact
"""

from __future__ import annotations

import random

import pytest

from minidb.database import Database
from minidb.errors import MiniDBError
from minidb.heap import Heap
from minidb.index import BTreeIndex
from minidb.page import PageType
from minidb.pager import Pager
from minidb.record import Column, ColumnType as CT, Schema
from minidb.table import Table

USERS = Schema((Column("id", CT.INT), Column("name", CT.TEXT), Column("age", CT.INT)))


def make_table(pager, schema=USERS):
    heap = Heap.create(pager, schema, page_type=PageType.HEAP)
    index = BTreeIndex.create(pager)
    return Table(pager, schema, heap.first_page_id, index.root_page_id)


# ---------------------------------------------------------------------------
class TestInsertScan:
    def test_insert_returns_rid_and_scan_keeps_order(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            t = make_table(pg)
            rids = [t.insert((i, f"u{i}", i)) for i in (3, 1, 2)]
            assert all(isinstance(r, tuple) and len(r) == 2 for r in rids)
            assert list(t.scan()) == [(3, "u3", 3), (1, "u1", 1), (2, "u2", 2)]

    def test_index_receives_every_id(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            t = make_table(pg)
            for i in (7, 2, 9, 4):
                t.insert((i, "x", i))
            assert sorted(k for k, _ in t.index.items()) == [2, 4, 7, 9]

    def test_single_id_column_table(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            t = make_table(pg, Schema((Column("id", CT.INT),)))
            t.insert((42,))
            assert list(t.scan()) == [(42,)]
            assert t.lookup_by_id(42) == (42,)


# ---------------------------------------------------------------------------
class TestUniqueId:
    def test_duplicate_id_is_refused(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            t = make_table(pg)
            t.insert((1, "Alice", 24))
            with pytest.raises(MiniDBError):
                t.insert((1, "Other", 99))

    def test_rejected_duplicate_never_reaches_the_heap(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            t = make_table(pg)
            t.insert((1, "Alice", 24))
            with pytest.raises(MiniDBError):
                t.insert((1, "Ghost", 0))
            assert list(t.scan()) == [(1, "Alice", 24)]  # no orphan row
            assert len(list(t.index.items())) == 1

    def test_duplicate_refused_after_a_split(self, tmp_path):
        with Pager(str(tmp_path / "t.db"), page_size=512) as pg:
            t = make_table(pg)
            for i in range(300):
                t.insert((i, "x", i))
            for i in (0, 150, 299):
                with pytest.raises(MiniDBError):
                    t.insert((i, "y", 1))
            assert len(list(t.scan())) == 300
            assert len(list(t.index.items())) == 300


# ---------------------------------------------------------------------------
class TestLookupById:
    def test_lookup_matches_scan(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            t = make_table(pg)
            rows = [
                (i, f"u{i}", i % 7)
                for i in random.Random(1).sample(range(1000), 400)
            ]
            for r in rows:
                t.insert(r)
            by_id = {r[0]: r for r in rows}
            for k in list(by_id) + [-1, 5000, 999]:
                assert t.lookup_by_id(k) == by_id.get(k)

    def test_lookup_across_a_page_chain(self, tmp_path):
        with Pager(str(tmp_path / "t.db"), page_size=512) as pg:
            t = make_table(pg)
            for i in range(600):
                t.insert((i, f"u{i}", i))
            assert pg.page_count > 10
            for k in (0, 199, 200, 599):
                assert t.lookup_by_id(k) == (k, f"u{k}", k)
            assert t.lookup_by_id(600) is None


# ---------------------------------------------------------------------------
class TestConsistency:
    def test_heap_and_index_describe_the_same_rows(self, tmp_path):
        with Pager(str(tmp_path / "t.db"), page_size=512) as pg:
            t = make_table(pg)
            ids = random.Random(3).sample(range(100_000), 2000)
            for i in ids:
                t.insert((i, f"n{i}", i % 50))
            scanned = list(t.scan())
            assert {r[0] for r in scanned} == {k for k, _ in t.index.items()}
            assert len(list(t.index.items())) == len(scanned)
            for r in scanned:
                assert t.lookup_by_id(r[0]) == r


# ---------------------------------------------------------------------------
class TestPersistence:
    def test_table_survives_restart(self, tmp_path):
        path = str(tmp_path / "t.db")
        ids = random.Random(4).sample(range(100_000), 2500)
        with Database(path, page_size=512) as db:
            db.create_table("users", USERS)
            tb = db.open_table("users")
            for i in ids:
                tb.insert((i, f"n{i}", i % 100))
            root, height = tb.index.root_page_id, tb.index.height()
        with Database(path, page_size=512) as db:
            tb = db.open_table("users")
            assert tb.index.root_page_id == root
            assert tb.index.height() == height
            idset = set(ids)
            for k in ids[:200] + [-1, 999_999]:
                row = tb.lookup_by_id(k)
                assert row == (k, f"n{k}", k % 100) if k in idset else row is None
            assert sorted(r[0] for r in tb.scan()) == sorted(ids)
