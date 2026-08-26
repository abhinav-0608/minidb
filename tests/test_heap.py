"""Stage 2 tests - the heap file.

* TestCreateAndScan    - a new heap, empty scan, RIDs from insert
* TestPersistence      - insert, restart, scan matches  (the Stage 2 goal)
* TestPageChaining     - rows spill onto linked pages and read back in order
* TestInvalidRows      - an oversized row is refused and changes nothing
* TestIndependence     - two heaps in one file do not see each other's rows
* TestCorruptChain     - a cyclic page chain is detected, not looped on
"""

from __future__ import annotations

import pytest

from minidb.errors import MiniDBError
from minidb.heap import Heap
from minidb.page import PageType, SlottedPage
from minidb.pager import Pager
from minidb.record import Column, ColumnType, Schema

USERS = Schema(
    (
        Column("id", ColumnType.INT),
        Column("name", ColumnType.TEXT),
        Column("age", ColumnType.INT),
    )
)


# ---------------------------------------------------------------------------
class TestCreateAndScan:
    def test_new_heap_is_empty(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            h = Heap.create(pg, USERS)
            assert h.first_page_id == 1
            assert list(h.scan()) == []

    def test_first_page_is_a_heap_page(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            h = Heap.create(pg, USERS)
            page = SlottedPage(bytearray(pg.read_page(h.first_page_id)))
            assert page.page_type == PageType.HEAP

    def test_insert_returns_rids(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            h = Heap.create(pg, USERS)
            assert h.insert((1, "Alice", 24)) == (1, 0)
            assert h.insert((2, "Bob", 31)) == (1, 1)

    def test_scan_returns_rows_in_insert_order(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            h = Heap.create(pg, USERS)
            rows = [(3, "c", 3), (1, "a", 1), (2, "b", 2)]
            for r in rows:
                h.insert(r)
            assert list(h.scan()) == rows


# ---------------------------------------------------------------------------
class TestPersistence:
    def test_insert_restart_scan(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Pager(path) as pg:
            h = Heap.create(pg, USERS)
            h.insert((1, "Alice", 24))
            h.insert((2, "Bob", 31))
            first = h.first_page_id
        with Pager(path) as pg:
            h = Heap(pg, USERS, first)
            assert list(h.scan()) == [(1, "Alice", 24), (2, "Bob", 31)]

    def test_appends_go_to_the_tail_after_reopen(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Pager(path) as pg:
            h = Heap.create(pg, USERS)
            for i in range(5):
                h.insert((i, "x", i))
            first = h.first_page_id
        with Pager(path) as pg:
            h = Heap(pg, USERS, first)  # must re-find the last page
            for i in range(5, 10):
                h.insert((i, "x", i))
        with Pager(path) as pg:
            h = Heap(pg, USERS, first)
            assert list(h.scan()) == [(i, "x", i) for i in range(10)]

    def test_empty_string_and_unicode_survive(self, tmp_path):
        path = str(tmp_path / "t.db")
        rows = [(1, "", 0), (2, "café \U0001f600", 99)]
        with Pager(path) as pg:
            h = Heap.create(pg, USERS)
            for r in rows:
                h.insert(r)
            first = h.first_page_id
        with Pager(path) as pg:
            assert list(Heap(pg, USERS, first).scan()) == rows


# ---------------------------------------------------------------------------
class TestPageChaining:
    def test_rows_spill_onto_new_pages_and_read_back(self, tmp_path):
        path = str(tmp_path / "t.db")
        expected = [(i, "x" * 20, i * 2) for i in range(200)]
        with Pager(path, page_size=512) as pg:
            h = Heap.create(pg, USERS)
            for r in expected:
                h.insert(r)
            first = h.first_page_id
            assert pg.page_count > 3  # chain definitely grew
        with Pager(path, page_size=512) as pg:
            assert list(Heap(pg, USERS, first).scan()) == expected

    def test_rid_page_id_advances_when_a_page_fills(self, tmp_path):
        with Pager(str(tmp_path / "t.db"), page_size=512) as pg:
            h = Heap.create(pg, USERS)
            page_ids = {h.insert((i, "x" * 20, i))[0] for i in range(200)}
            assert len(page_ids) > 1  # rows landed on more than one page

    def test_new_pages_are_heap_pages_and_linked(self, tmp_path):
        with Pager(str(tmp_path / "t.db"), page_size=512) as pg:
            h = Heap.create(pg, USERS)
            for i in range(200):
                h.insert((i, "x" * 20, i))
            # walk the chain by hand
            page_id, count = h.first_page_id, 0
            while page_id != 0:
                page = SlottedPage(bytearray(pg.read_page(page_id)))
                assert page.page_type == PageType.HEAP
                count += page.num_slots
                page_id = page.next_page_id
            assert count == 200


# ---------------------------------------------------------------------------
class TestInvalidRows:
    def test_row_bigger_than_a_page_is_refused_and_changes_nothing(self, tmp_path):
        with Pager(str(tmp_path / "t.db"), page_size=512) as pg:
            h = Heap.create(pg, USERS)
            h.insert((1, "small", 1))
            before = pg.page_count
            with pytest.raises(MiniDBError):
                h.insert((2, "y" * 900, 2))  # ~919 bytes > 499-byte page capacity
            assert pg.page_count == before
            assert list(h.scan()) == [(1, "small", 1)]

    def test_malformed_row_is_refused_before_any_write(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            h = Heap.create(pg, USERS)
            with pytest.raises(MiniDBError):
                h.insert((1, "Alice"))  # wrong arity
            assert list(h.scan()) == []


# ---------------------------------------------------------------------------
class TestIndependence:
    def test_two_heaps_in_one_file_stay_separate(self, tmp_path):
        path = str(tmp_path / "t.db")
        other = Schema((Column("k", ColumnType.INT), Column("v", ColumnType.TEXT)))
        with Pager(path) as pg:
            a = Heap.create(pg, USERS)
            b = Heap.create(pg, other)
            a.insert((1, "Alice", 24))
            b.insert((100, "hundred"))
            a.insert((2, "Bob", 31))
            first_a, first_b = a.first_page_id, b.first_page_id
        with Pager(path) as pg:
            assert list(Heap(pg, USERS, first_a).scan()) == [
                (1, "Alice", 24),
                (2, "Bob", 31),
            ]
            assert list(Heap(pg, other, first_b).scan()) == [(100, "hundred")]


# ---------------------------------------------------------------------------
class TestCorruptChain:
    def test_cyclic_page_chain_is_detected(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Pager(path, page_size=512) as pg:
            h = Heap.create(pg, USERS)
            for i in range(80):  # force at least two pages
                h.insert((i, "x" * 20, i))
            first = h.first_page_id
            p1 = SlottedPage(bytearray(pg.read_page(first)))
            second_id = p1.next_page_id
            assert second_id != 0
            p2 = SlottedPage(bytearray(pg.read_page(second_id)))
            p2.next_page_id = first  # close the loop
            pg.write_page(second_id, p2.buffer)
        with Pager(path, page_size=512) as pg:
            with pytest.raises(MiniDBError):
                Heap(pg, USERS, first)  # _walk_to_last_page must not spin
