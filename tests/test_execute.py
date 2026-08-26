"""Stage 8 tests - the executor dispatcher (execute / execute_sql / execute_script).

* TestDispatch     - each query type routes to the right operation
* TestRoundTrip    - CREATE -> INSERT -> SELECT through SQL strings, + persistence
* TestErrors       - every failure path is a delegated MiniDBError, DB untouched
* TestIndexParity  - `WHERE id = k` through SQL still matches a scan
* TestScript       - execute_script returns one result per statement
"""

from __future__ import annotations

import random

import pytest

from minidb.database import Database
from minidb.errors import MiniDBError
from minidb.executor import SelectResult, execute, execute_script, execute_sql
from minidb.query import CreateTableQuery, InsertQuery, SelectQuery
from minidb.record import Column, ColumnType as CT

DDL = "CREATE TABLE users (id INT, name TEXT, age INT)"


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "x.db"))
    yield d
    d.close()


# ---------------------------------------------------------------------------
class TestDispatch:
    def test_create_returns_none_and_makes_the_table(self, db):
        assert execute_sql(db, DDL) is None
        assert db.table_names() == ["users"]

    def test_insert_returns_none_and_adds_a_row(self, db):
        execute_sql(db, DDL)
        assert execute_sql(db, "INSERT INTO users VALUES (1, 'Alice', 24)") is None
        assert list(db.open_table("users").scan()) == [(1, "Alice", 24)]

    def test_select_returns_a_selectresult(self, db):
        execute_sql(db, DDL)
        execute_sql(db, "INSERT INTO users VALUES (1, 'Alice', 24)")
        r = execute_sql(db, "SELECT name FROM users WHERE id = 1")
        assert isinstance(r, SelectResult)
        assert (r.column_names, r.rows) == (["name"], [("Alice",)])

    def test_execute_takes_a_query_object_directly(self, db):
        execute(db, CreateTableQuery("t", (Column("id", CT.INT),)))
        execute(db, InsertQuery("t", (5,)))
        assert execute(db, SelectQuery("t", None, None)).rows == [(5,)]

    @pytest.mark.parametrize("thing", ["SELECT 1", 42, None])
    def test_execute_rejects_a_non_query(self, db, thing):
        with pytest.raises(MiniDBError):
            execute(db, thing)


# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_full_session_through_sql_strings(self, db):
        execute_sql(db, DDL)
        for tup in ("(1, 'Alice', 24)", "(2, 'Bob', 31)", "(3, 'Carol', 24)"):
            execute_sql(db, f"INSERT INTO users VALUES {tup}")
        assert execute_sql(db, "SELECT * FROM users").rows == [
            (1, "Alice", 24), (2, "Bob", 31), (3, "Carol", 24)
        ]
        assert execute_sql(db, "SELECT name FROM users WHERE id = 2").rows == [("Bob",)]
        assert execute_sql(db, "SELECT id FROM users WHERE age = 24").rows == [(1,), (3,)]

    def test_persistence_across_reopen(self, tmp_path):
        path = str(tmp_path / "p.db")
        with Database(path) as db:
            execute_sql(db, "CREATE TABLE t (id INT, v TEXT)")
            execute_sql(db, "INSERT INTO t VALUES (1, 'one')")
            execute_sql(db, "INSERT INTO t VALUES (2, 'two')")
        with Database(path) as db:
            assert execute_sql(db, "SELECT * FROM t").rows == [(1, "one"), (2, "two")]
            assert execute_sql(db, "SELECT v FROM t WHERE id = 2").rows == [("two",)]

    def test_escaped_quote_round_trips(self, db):
        execute_sql(db, "CREATE TABLE t (id INT, v TEXT)")
        execute_sql(db, "INSERT INTO t VALUES (1, 'it''s here')")
        assert execute_sql(db, "SELECT v FROM t WHERE id = 1").rows == [("it's here",)]

    def test_negative_int_round_trips(self, db):
        execute_sql(db, "CREATE TABLE t (id INT, n INT)")
        execute_sql(db, "INSERT INTO t VALUES (-5, -100)")
        assert execute_sql(db, "SELECT * FROM t WHERE id = -5").rows == [(-5, -100)]


# ---------------------------------------------------------------------------
class TestErrors:
    @pytest.mark.parametrize(
        "sql,fragment",
        [
            ("CREATE TABLE users (id INT)", "already exists"),
            ("CREATE TABLE bad (uid INT)", "id INT"),
            ("CREATE TABLE bad (id TEXT)", "id INT"),
            ("CREATE TABLE bad (id INT, x INT, x INT)", "duplicate"),
            ("INSERT INTO nope VALUES (1)", "no such table"),
            ("INSERT INTO users VALUES (1)", "3 columns but 1"),
            ("INSERT INTO users VALUES (1, 2, 3)", "TEXT but got"),
            ("INSERT INTO users VALUES (1, 'a', 'b')", "INT but got"),
            ("SELECT missing FROM users", "no column"),
            ("SELECT * FROM users WHERE age = 'x'", "cannot compare"),
            ("SELECT * FROM nope", "no such table"),
            ("SELECT FROM users", "column name"),
        ],
    )
    def test_error_is_reported_with_a_useful_message(self, db, sql, fragment):
        execute_sql(db, DDL)
        with pytest.raises(MiniDBError) as exc:
            execute_sql(db, sql)
        assert fragment in str(exc.value)

    def test_duplicate_id_is_refused_and_leaves_one_row(self, db):
        execute_sql(db, DDL)
        execute_sql(db, "INSERT INTO users VALUES (1, 'Alice', 24)")
        with pytest.raises(MiniDBError):
            execute_sql(db, "INSERT INTO users VALUES (1, 'Other', 9)")
        assert execute_sql(db, "SELECT * FROM users").rows == [(1, "Alice", 24)]

    def test_failed_statement_leaves_db_usable(self, db):
        execute_sql(db, DDL)
        with pytest.raises(MiniDBError):
            execute_sql(db, "INSERT INTO users VALUES (1)")  # wrong arity
        execute_sql(db, "INSERT INTO users VALUES (1, 'Alice', 24)")
        assert execute_sql(db, "SELECT * FROM users").rows == [(1, "Alice", 24)]


# ---------------------------------------------------------------------------
class TestIndexParity:
    def test_where_id_matches_a_scan_through_sql(self, tmp_path):
        path = str(tmp_path / "b.db")
        ids = random.Random(9).sample(range(50_000), 700)
        with Database(path, page_size=512) as db:
            execute_sql(db, "CREATE TABLE u (id INT, n TEXT)")
            for i in ids:
                execute_sql(db, f"INSERT INTO u VALUES ({i}, 'n{i}')")
            present = set(ids)
            for k in ids[:150] + [-1, 99_999, 50_000]:
                rows = execute_sql(db, f"SELECT n FROM u WHERE id = {k}").rows
                assert rows == ([(f"n{k}",)] if k in present else [])


# ---------------------------------------------------------------------------
class TestScript:
    def test_one_result_per_statement(self, db):
        results = execute_script(
            db,
            "CREATE TABLE t (id INT);"
            "INSERT INTO t VALUES (1);"
            "INSERT INTO t VALUES (2);"
            "SELECT * FROM t;",
        )
        assert results[:3] == [None, None, None]
        assert isinstance(results[3], SelectResult)
        assert results[3].rows == [(1,), (2,)]

    def test_empty_script(self, db):
        assert execute_script(db, "") == []
        assert execute_script(db, "-- comment only\n") == []

    def test_error_propagates_but_earlier_statements_ran(self, db):
        with pytest.raises(MiniDBError):
            execute_script(
                db, "CREATE TABLE t (id INT); INSERT INTO t VALUES (1, 2);"
            )
        assert db.table_names() == ["t"]  # the CREATE happened
