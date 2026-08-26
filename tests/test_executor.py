"""Stage 4 tests - the sequential executor.

* TestSelectAll    - SELECT * shape and ordering, incl. empty tables
* TestProjection   - column subset / reorder / duplicate / list input
* TestWhere        - equality filter on INT and TEXT, matches and misses
* TestWhereIdOracle- WHERE id = k returns exactly the right row for every k
                     (the sequential result Stage 6's index must reproduce)
* TestRejections   - unknown table / column, bad operator, type mismatch
* TestAcrossPages  - scan + filter over a multi-page heap chain
"""

from __future__ import annotations

import pytest

from minidb.database import Database
from minidb.errors import MiniDBError
from minidb.executor import execute_select
from minidb.query import Condition, SelectQuery
from minidb.record import Column, ColumnType as CT, Schema

USERS = Schema((Column("id", CT.INT), Column("name", CT.TEXT), Column("age", CT.INT)))
ROWS = [
    (1, "Alice", 24),
    (2, "Bob", 31),
    (3, "Carol", 24),
    (4, "Dave", 19),
    (5, "Eve", 31),
]


@pytest.fixture
def users_db(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.create_table("users", USERS)
    table = db.open_table("users")
    for r in ROWS:
        table.insert(r)
    yield db
    db.close()


# ---------------------------------------------------------------------------
class TestSelectAll:
    def test_returns_every_row_in_insert_order(self, users_db):
        r = execute_select(users_db, SelectQuery("users", None))
        assert r.column_names == ["id", "name", "age"]
        assert r.rows == ROWS

    def test_empty_table_gives_headers_and_no_rows(self, tmp_path):
        with Database(str(tmp_path / "e.db")) as db:
            db.create_table("t", USERS)
            r = execute_select(db, SelectQuery("t", None))
            assert r.column_names == ["id", "name", "age"]
            assert r.rows == []

    def test_rows_are_tuples_and_column_names_a_list(self, users_db):
        r = execute_select(users_db, SelectQuery("users", None))
        assert isinstance(r.column_names, list)
        assert all(isinstance(row, tuple) for row in r.rows)


# ---------------------------------------------------------------------------
class TestProjection:
    def test_single_column(self, users_db):
        r = execute_select(users_db, SelectQuery("users", ("name",)))
        assert r.column_names == ["name"]
        assert r.rows == [(row[1],) for row in ROWS]

    def test_reordered(self, users_db):
        r = execute_select(users_db, SelectQuery("users", ("age", "id")))
        assert r.column_names == ["age", "id"]
        assert r.rows == [(row[2], row[0]) for row in ROWS]

    def test_duplicate_column_allowed(self, users_db):
        r = execute_select(users_db, SelectQuery("users", ("id", "id")))
        assert r.column_names == ["id", "id"]
        assert r.rows == [(row[0], row[0]) for row in ROWS]

    def test_star_equals_explicit_all_columns(self, users_db):
        a = execute_select(users_db, SelectQuery("users", None))
        b = execute_select(users_db, SelectQuery("users", ("id", "name", "age")))
        assert (a.column_names, a.rows) == (b.column_names, b.rows)

    def test_columns_may_be_a_list(self, users_db):
        r = execute_select(users_db, SelectQuery("users", ["name"]))
        assert r.rows == [(row[1],) for row in ROWS]


# ---------------------------------------------------------------------------
class TestWhere:
    def test_int_column_multiple_matches_keep_order(self, users_db):
        r = execute_select(
            users_db, SelectQuery("users", None, Condition("age", "=", 24))
        )
        assert r.rows == [(1, "Alice", 24), (3, "Carol", 24)]

    def test_text_column(self, users_db):
        r = execute_select(
            users_db, SelectQuery("users", None, Condition("name", "=", "Carol"))
        )
        assert r.rows == [(3, "Carol", 24)]

    def test_no_match_is_empty(self, users_db):
        r = execute_select(
            users_db, SelectQuery("users", None, Condition("age", "=", 99))
        )
        assert r.rows == []

    def test_match_on_last_row(self, users_db):
        r = execute_select(
            users_db, SelectQuery("users", None, Condition("name", "=", "Eve"))
        )
        assert r.rows == [(5, "Eve", 31)]

    def test_where_with_projection(self, users_db):
        r = execute_select(
            users_db, SelectQuery("users", ("name",), Condition("age", "=", 31))
        )
        assert r.column_names == ["name"]
        assert r.rows == [("Bob",), ("Eve",)]

    def test_where_on_empty_table(self, tmp_path):
        with Database(str(tmp_path / "e.db")) as db:
            db.create_table("t", USERS)
            r = execute_select(
                db, SelectQuery("t", None, Condition("age", "=", 1))
            )
            assert r.rows == []


# ---------------------------------------------------------------------------
class TestWhereIdOracle:
    def test_each_present_key_returns_its_row(self, users_db):
        for row in ROWS:
            r = execute_select(
                users_db, SelectQuery("users", None, Condition("id", "=", row[0]))
            )
            assert r.rows == [row]

    @pytest.mark.parametrize("missing", [-1, 0, 6, 999])
    def test_absent_key_returns_nothing(self, users_db, missing):
        r = execute_select(
            users_db, SelectQuery("users", None, Condition("id", "=", missing))
        )
        assert r.rows == []


# ---------------------------------------------------------------------------
class TestRejections:
    def test_unknown_table(self, users_db):
        with pytest.raises(MiniDBError):
            execute_select(users_db, SelectQuery("ghost", None))

    def test_unknown_projected_column(self, users_db):
        with pytest.raises(MiniDBError):
            execute_select(users_db, SelectQuery("users", ("nope",)))

    def test_unknown_where_column(self, users_db):
        with pytest.raises(MiniDBError):
            execute_select(
                users_db, SelectQuery("users", None, Condition("nope", "=", 1))
            )

    @pytest.mark.parametrize("op", [">", "<", ">=", "<=", "!=", "=="])
    def test_non_equality_operator(self, users_db, op):
        with pytest.raises(MiniDBError):
            execute_select(
                users_db, SelectQuery("users", None, Condition("id", op, 1))
            )

    @pytest.mark.parametrize("value", ["x", 5.0, True, None])
    def test_int_column_rejects_non_int_literal(self, users_db, value):
        with pytest.raises(MiniDBError):
            execute_select(
                users_db, SelectQuery("users", None, Condition("age", "=", value))
            )

    @pytest.mark.parametrize("value", [5, 5.0, True, None])
    def test_text_column_rejects_non_str_literal(self, users_db, value):
        with pytest.raises(MiniDBError):
            execute_select(
                users_db, SelectQuery("users", None, Condition("name", "=", value))
            )


# ---------------------------------------------------------------------------
class TestAcrossPages:
    def test_scan_and_filter_over_a_page_chain(self, tmp_path):
        path = str(tmp_path / "big.db")
        expected = [(i, f"u{i}", i % 7) for i in range(500)]
        with Database(path, page_size=512) as db:
            db.create_table("t", USERS)
            table = db.open_table("t")
            for r in expected:
                table.insert(r)
            assert db.page_count > 5  # heap definitely chained

            assert execute_select(db, SelectQuery("t", None)).rows == expected

            got = execute_select(
                db, SelectQuery("t", ("id",), Condition("age", "=", 3))
            ).rows
            assert got == [(i,) for i in range(500) if i % 7 == 3]
