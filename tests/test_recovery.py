"""Stage 9 tests - crash recovery, end to end through Database.

* TestCleanLifecycle - no crash: WAL empty after close, data persists
* TestPagerTxn       - begin/commit/abort semantics at the pager level
* TestDurability     - crash after WAL fsync -> committed change survives
* TestAtomicity      - crash before COMMIT record -> change vanishes, db usable
* TestCheckpoint     - crash mid-checkpoint -> WAL redo makes it consistent
* TestTornTail       - garbage / truncated WAL tail -> good txns still recover
* TestIdempotent     - replaying the same WAL twice is safe
* TestConsistency    - heap and index agree after every crash point
* TestWalOff         - wal=False keeps the pre-Stage-9 behaviour
"""

from __future__ import annotations

import os

import pytest

from minidb.database import Database
from minidb.errors import MiniDBError
from minidb.executor import execute_sql
from minidb.heap import Heap
from minidb.pager import Pager, SimulatedCrash
from minidb.record import Column, ColumnType, Schema


def crash_during(db, sql, at):
    db._pager.crash_at = at
    with pytest.raises(SimulatedCrash):
        execute_sql(db, sql)
    db._pager._abandon()  # simulate the process dying


def rows_of(path, sql="SELECT * FROM t"):
    with Database(path) as db:  # opening runs recovery
        return execute_sql(db, sql).rows


@pytest.fixture
def seeded(tmp_path):
    """A database with table t and one committed, checkpointed row."""
    p = str(tmp_path / "t.db")
    with Database(p) as db:
        execute_sql(db, "CREATE TABLE t (id INT, v TEXT)")
        execute_sql(db, "INSERT INTO t VALUES (1, 'first')")
    assert os.path.getsize(p + "-wal") == 0
    return p


# ---------------------------------------------------------------------------
class TestCleanLifecycle:
    def test_wal_is_empty_after_a_clean_close(self, tmp_path):
        p = str(tmp_path / "t.db")
        with Database(p) as db:
            execute_sql(db, "CREATE TABLE t (id INT, v TEXT)")
            for i in range(20):
                execute_sql(db, f"INSERT INTO t VALUES ({i}, 'v{i}')")
        assert os.path.getsize(p + "-wal") == 0

    def test_data_persists_without_a_crash(self, seeded):
        assert rows_of(seeded) == [(1, "first")]

    def test_reopen_works_even_if_the_empty_wal_is_deleted(self, seeded):
        os.remove(seeded + "-wal")
        assert rows_of(seeded) == [(1, "first")]


# ---------------------------------------------------------------------------
class TestPagerTxn:
    def test_writes_need_a_transaction_in_wal_mode(self, tmp_path):
        with Pager(str(tmp_path / "t.db"), wal=True) as pg:
            with pytest.raises(MiniDBError):
                pg.allocate_page()
            pg.begin()
            pid = pg.allocate_page()
            pg.write_page(pid, b"\x01" * pg.page_size)
            pg.commit()
            with pytest.raises(MiniDBError):
                pg.write_page(pid, b"\x02" * pg.page_size)

    def test_double_begin_raises(self, tmp_path):
        with Pager(str(tmp_path / "t.db"), wal=True) as pg:
            pg.begin()
            with pytest.raises(MiniDBError):
                pg.begin()

    def test_abort_rolls_back_pages_and_page_count(self, tmp_path):
        with Pager(str(tmp_path / "t.db"), wal=True) as pg:
            pg.begin()
            a = pg.allocate_page()
            pg.write_page(a, b"A" * pg.page_size)
            pg.commit()
            base = pg.page_count

            pg.begin()
            b = pg.allocate_page()
            pg.write_page(b, b"B" * pg.page_size)
            pg.write_page(a, b"Z" * pg.page_size)  # touch a committed page
            pg.abort()

            assert pg.page_count == base
            assert pg.read_page(a) == b"A" * pg.page_size  # pre-image restored
            with pytest.raises(MiniDBError):
                pg.read_page(b)  # allocation undone


# ---------------------------------------------------------------------------
class TestDurability:
    def test_committed_insert_survives_a_pre_checkpoint_crash(self, seeded):
        with Database(seeded) as db:
            crash_during(db, "INSERT INTO t VALUES (2, 'second')", "after_wal_fsync")
        assert os.path.getsize(seeded + "-wal") > 0  # committed txn still there
        assert rows_of(seeded) == [(1, "first"), (2, "second")]

    def test_all_prior_commits_survive_too(self, tmp_path):
        p = str(tmp_path / "t.db")
        with Database(p) as db:
            execute_sql(db, "CREATE TABLE t (id INT, v TEXT)")
            for i in range(30):
                execute_sql(db, f"INSERT INTO t VALUES ({i}, 'v{i}')")
            crash_during(db, "INSERT INTO t VALUES (99, 'last')", "after_wal_fsync")
        assert rows_of(p) == [(i, f"v{i}") for i in range(30)] + [(99, "last")]


# ---------------------------------------------------------------------------
class TestAtomicity:
    def test_uncommitted_insert_vanishes(self, seeded):
        with Database(seeded) as db:
            crash_during(db, "INSERT INTO t VALUES (2, 'gone')", "after_page_records")
        assert rows_of(seeded) == [(1, "first")]

    def test_db_is_usable_and_the_id_is_free_again(self, seeded):
        with Database(seeded) as db:
            crash_during(db, "INSERT INTO t VALUES (2, 'gone')", "after_page_records")
        with Database(seeded) as db:
            execute_sql(db, "INSERT INTO t VALUES (2, 'real')")
            assert execute_sql(db, "SELECT * FROM t").rows == [
                (1, "first"), (2, "real")
            ]

    def test_half_created_table_is_rolled_back(self, tmp_path):
        p = str(tmp_path / "t.db")
        with Database(p) as db:
            execute_sql(db, "CREATE TABLE a (id INT)")
            crash_during(db, "CREATE TABLE b (id INT, x TEXT)", "after_page_records")
        with Database(p) as db:
            assert db.table_names() == ["a"]
            execute_sql(db, "INSERT INTO a VALUES (5)")
            assert execute_sql(db, "SELECT * FROM a").rows == [(5,)]


# ---------------------------------------------------------------------------
class TestCheckpoint:
    def _two_rows_then_checkpoint_crash(self, path, at):
        with Database(path) as db:
            execute_sql(db, "CREATE TABLE t (id INT, v TEXT)")
            execute_sql(db, "INSERT INTO t VALUES (1, 'a')")
            execute_sql(db, "INSERT INTO t VALUES (2, 'b')")
            db._pager.crash_at = at
            with pytest.raises(SimulatedCrash):
                db._pager.checkpoint()
            db._pager._abandon()

    def test_crash_before_header_write(self, tmp_path):
        p = str(tmp_path / "t.db")
        self._two_rows_then_checkpoint_crash(p, "checkpoint_before_header")
        assert rows_of(p) == [(1, "a"), (2, "b")]

    def test_crash_before_wal_truncate(self, tmp_path):
        p = str(tmp_path / "t.db")
        self._two_rows_then_checkpoint_crash(p, "checkpoint_before_truncate")
        assert rows_of(p) == [(1, "a"), (2, "b")]
        # a clean reopen+close reconciles the WAL
        with Database(p) as db:
            pass
        assert os.path.getsize(p + "-wal") == 0


# ---------------------------------------------------------------------------
class TestTornTail:
    def test_garbage_after_a_committed_txn(self, seeded):
        with Database(seeded) as db:
            crash_during(db, "INSERT INTO t VALUES (2, 'keep')", "after_wal_fsync")
        with open(seeded + "-wal", "ab") as f:
            f.write(b"\xff" * 200)
        assert rows_of(seeded) == [(1, "first"), (2, "keep")]

    def test_truncated_wal_tail(self, seeded):
        with Database(seeded) as db:
            execute_sql(db, "INSERT INTO t VALUES (2, 'keep')")
            crash_during(db, "INSERT INTO t VALUES (3, 'partial')", "after_page_records")
        with open(seeded + "-wal", "r+b") as f:
            f.truncate(max(0, os.path.getsize(seeded + "-wal") - 5))
        assert rows_of(seeded) == [(1, "first"), (2, "keep")]


# ---------------------------------------------------------------------------
class TestIdempotent:
    def test_replaying_the_same_wal_twice(self, seeded):
        with Database(seeded) as db:
            crash_during(db, "INSERT INTO t VALUES (2, 'x')", "after_wal_fsync")
        wal_bytes = open(seeded + "-wal", "rb").read()
        assert wal_bytes

        assert rows_of(seeded) == [(1, "first"), (2, "x")]  # recovery #1
        assert os.path.getsize(seeded + "-wal") == 0  # ...truncated it

        with open(seeded + "-wal", "wb") as f:
            f.write(wal_bytes)  # restore -> forces recovery #2
        assert rows_of(seeded) == [(1, "first"), (2, "x")]  # same result


# ---------------------------------------------------------------------------
class TestConsistency:
    @pytest.mark.parametrize("point", ["after_page_records", "after_wal_fsync"])
    def test_heap_and_index_agree_after_a_crash(self, tmp_path, point):
        p = str(tmp_path / "t.db")
        with Database(p, page_size=512) as db:
            execute_sql(db, "CREATE TABLE t (id INT, v TEXT)")
            for i in range(40):
                execute_sql(db, f"INSERT INTO t VALUES ({i}, 'v{i}')")
            crash_during(db, "INSERT INTO t VALUES (999, 'crash')", point)

        with Database(p, page_size=512) as db:
            t = db.open_table("t")
            scanned = list(t.scan())
            index_items = list(t.index.items())
            assert {r[0] for r in scanned} == {k for k, _ in index_items}
            assert len(index_items) == len(scanned)
            for r in scanned:
                assert t.lookup_by_id(r[0]) == r

            ids = sorted(r[0] for r in scanned)
            if point == "after_wal_fsync":
                assert ids == list(range(40)) + [999]
            else:
                assert ids == list(range(40))


# ---------------------------------------------------------------------------
class TestWalOff:
    def test_wal_false_writes_no_wal_file_and_still_persists(self, tmp_path):
        p = str(tmp_path / "t.db")
        with Database(p, wal=False) as db:
            execute_sql(db, "CREATE TABLE t (id INT)")
            execute_sql(db, "INSERT INTO t VALUES (1)")
        assert not os.path.exists(p + "-wal")
        with Database(p, wal=False) as db:
            assert execute_sql(db, "SELECT * FROM t").rows == [(1,)]

    def test_bare_pager_needs_no_transaction(self, tmp_path):
        s = Schema((Column("id", ColumnType.INT),))
        p = str(tmp_path / "raw.db")
        with Pager(p) as pg:  # wal=False by default
            h = Heap.create(pg, s)
            h.insert((1,))
            h.insert((2,))
            first = h.first_page_id
        with Pager(p) as pg:
            assert list(Heap(pg, s, first).scan()) == [(1,), (2,)]
