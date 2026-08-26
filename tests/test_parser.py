"""Stage 7 tests - the SQL parser.

* TestExamples    - the brief's five statements, pinned to exact objects
* TestCreate      - CREATE TABLE shapes
* TestInsert      - INSERT INTO shapes, value types preserved
* TestSelect      - *, column lists, optional WHERE
* TestWhitespace  - messy whitespace and optional semicolons
* TestMalformed   - a catalogue of syntax errors, all rejected
* TestErrorText   - error messages carry position and are specific
"""

from __future__ import annotations

import pytest

from minidb.errors import MiniDBError
from minidb.query import Condition, CreateTableQuery, InsertQuery, SelectQuery
from minidb.record import Column, ColumnType as CT
from minidb.sql import parse, parse_script


# ---------------------------------------------------------------------------
class TestExamples:
    def test_the_five_brief_statements(self):
        assert parse("CREATE TABLE users (id INT, name TEXT, age INT);") == (
            CreateTableQuery(
                "users",
                (
                    Column("id", CT.INT),
                    Column("name", CT.TEXT),
                    Column("age", CT.INT),
                ),
            )
        )
        assert parse("INSERT INTO users VALUES (1, 'Alice', 24);") == InsertQuery(
            "users", (1, "Alice", 24)
        )
        assert parse("INSERT INTO users VALUES (2, 'Bob', 31);") == InsertQuery(
            "users", (2, "Bob", 31)
        )
        assert parse("SELECT * FROM users;") == SelectQuery("users", None, None)
        assert parse("SELECT name FROM users WHERE id = 1;") == SelectQuery(
            "users", ("name",), Condition("id", "=", 1)
        )


# ---------------------------------------------------------------------------
class TestCreate:
    def test_single_column(self):
        assert parse("CREATE TABLE t (id INT)") == CreateTableQuery(
            "t", (Column("id", CT.INT),)
        )

    def test_column_order_and_types_preserved(self):
        q = parse("CREATE TABLE t (c INT, a TEXT, b INT)")
        assert [(c.name, c.type) for c in q.columns] == [
            ("c", CT.INT), ("a", CT.TEXT), ("b", CT.INT)
        ]

    def test_columns_is_a_tuple_of_Column(self):
        q = parse("CREATE TABLE t (id INT)")
        assert isinstance(q.columns, tuple)
        assert isinstance(q.columns[0], Column)

    def test_lowercase_and_mixed_case_keywords(self):
        assert parse("create table t (id int, n text)") == CreateTableQuery(
            "t", (Column("id", CT.INT), Column("n", CT.TEXT))
        )
        assert parse("CrEaTe TaBlE t (id InT)") == CreateTableQuery(
            "t", (Column("id", CT.INT),)
        )

    def test_semicolon_is_optional(self):
        assert parse("CREATE TABLE t (id INT)") == parse("CREATE TABLE t (id INT);")

    def test_newlines_and_extra_spaces(self):
        q = parse("CREATE\n  TABLE   t\n(\n  id   INT ,\n  name TEXT\n)\n;")
        assert q == CreateTableQuery(
            "t", (Column("id", CT.INT), Column("name", CT.TEXT))
        )


# ---------------------------------------------------------------------------
class TestInsert:
    def test_basic(self):
        assert parse("INSERT INTO t VALUES (1, 'Alice', 24)") == InsertQuery(
            "t", (1, "Alice", 24)
        )

    def test_single_value(self):
        assert parse("INSERT INTO t VALUES (42)") == InsertQuery("t", (42,))

    def test_negative_int(self):
        assert parse("INSERT INTO t VALUES (-5, 'x')") == InsertQuery("t", (-5, "x"))

    def test_escaped_quote_in_value(self):
        assert parse("INSERT INTO t VALUES ('it''s')") == InsertQuery("t", ("it's",))

    def test_value_types_are_preserved(self):
        q = parse("INSERT INTO t VALUES (1, 'a', 2)")
        assert [type(v) for v in q.values] == [int, str, int]

    def test_many_values(self):
        assert parse("INSERT INTO t VALUES (1, 2, 3, 4, 5, 6)").values == (
            1, 2, 3, 4, 5, 6
        )

    def test_semicolon_optional(self):
        assert parse("INSERT INTO t VALUES (1)") == parse("INSERT INTO t VALUES (1);")


# ---------------------------------------------------------------------------
class TestSelect:
    def test_star(self):
        assert parse("SELECT * FROM t") == SelectQuery("t", None, None)

    def test_one_column(self):
        assert parse("SELECT a FROM t") == SelectQuery("t", ("a",), None)

    def test_column_list(self):
        assert parse("SELECT a, b, c FROM t") == SelectQuery("t", ("a", "b", "c"), None)

    def test_column_case_preserved(self):
        assert parse("SELECT Name, AGE FROM t") == SelectQuery(
            "t", ("Name", "AGE"), None
        )

    def test_no_space_around_comma(self):
        assert parse("SELECT a,b FROM t") == SelectQuery("t", ("a", "b"), None)

    def test_where_int(self):
        assert parse("SELECT * FROM t WHERE id = 1") == SelectQuery(
            "t", None, Condition("id", "=", 1)
        )

    def test_where_negative_int(self):
        assert parse("SELECT * FROM t WHERE x = -9") == SelectQuery(
            "t", None, Condition("x", "=", -9)
        )

    def test_where_string(self):
        assert parse("SELECT * FROM t WHERE name = 'Bob'") == SelectQuery(
            "t", None, Condition("name", "=", "Bob")
        )

    def test_where_with_projection(self):
        assert parse("SELECT a FROM t WHERE b = 2") == SelectQuery(
            "t", ("a",), Condition("b", "=", 2)
        )


# ---------------------------------------------------------------------------
class TestWhitespace:
    def test_all_statements_tolerate_messy_whitespace(self):
        assert parse("\n\tSELECT\n*\nFROM\nt\n") == SelectQuery("t", None, None)
        assert parse("  INSERT\tINTO t VALUES( 1 ,'a', -2 )  ") == InsertQuery(
            "t", (1, "a", -2)
        )
        assert parse("CREATE TABLE t ( id INT , n TEXT )") == CreateTableQuery(
            "t", (Column("id", CT.INT), Column("n", CT.TEXT))
        )

    def test_semicolon_variants(self):
        a = parse("SELECT * FROM t")
        assert a == parse("SELECT * FROM t;") == parse("SELECT * FROM t ;")


# ---------------------------------------------------------------------------
class TestMalformed:
    BAD = [
        "", "   ",
        "SELECT", "SELECT *", "SELECT * FROM", "SELECT FROM t",
        "SELECT *, a FROM t", "SELECT a b FROM t", "SELECT a, FROM t",
        "SELECT a FROM t x", "SELECT * t", "SELECT * FROM 123",
        "SELECT * FROM t WHERE", "SELECT * FROM t WHERE id",
        "SELECT * FROM t WHERE id =", "SELECT * FROM t WHERE id >= 1",
        "SELECT * FROM t WHERE 1 = 1",
        "INSERT t VALUES (1)", "INSERT INTO VALUES (1)", "INSERT INTO t (1)",
        "INSERT INTO t VALUES 1", "INSERT INTO t VALUES ()",
        "INSERT INTO t VALUES (1,)", "INSERT INTO t VALUES (1 2)",
        "INSERT INTO t VALUES (NULL)", "INSERT INTO t VALUES (id)",
        "INSERT INTO t VALUES ('unterminated)",
        "CREATE users (id INT)", "CREATE TABLE (id INT)", "CREATE TABLE t id INT",
        "CREATE TABLE t ()", "CREATE TABLE t (id)", "CREATE TABLE t (id BLOB)",
        "CREATE TABLE t (id INT,)", "CREATE TABLE t (id INT name TEXT)",
        "DELETE FROM t", "UPDATE t SET x = 1", "DROP TABLE t",
        "SELECT * FROM t; SELECT * FROM t", "SELECT * FROM t;;",
        "SELECT * FROM t junk", "123", "'string'", "* FROM t",
    ]

    @pytest.mark.parametrize("sql", BAD)
    def test_rejected(self, sql):
        with pytest.raises(MiniDBError):
            parse(sql)


# ---------------------------------------------------------------------------
class TestErrorText:
    def test_errors_carry_a_position(self):
        with pytest.raises(MiniDBError) as exc:
            parse("SELECT * FROM t WHERE")
        assert "position" in str(exc.value)

    def test_unsupported_operator_mentions_equals(self):
        with pytest.raises(MiniDBError) as exc:
            parse("SELECT * FROM t WHERE id >= 1")
        assert "'='" in str(exc.value)

    def test_null_message(self):
        with pytest.raises(MiniDBError) as exc:
            parse("INSERT INTO t VALUES (NULL)")
        assert "NULL" in str(exc.value)

    def test_unknown_statement_message_lists_the_options(self):
        with pytest.raises(MiniDBError) as exc:
            parse("DELETE FROM t")
        msg = str(exc.value)
        assert "SELECT" in msg and "INSERT" in msg and "CREATE" in msg


# ---------------------------------------------------------------------------
class TestParseScript:
    def test_empty_and_whitespace_and_semicolons(self):
        assert parse_script("") == []
        assert parse_script("  \n  ") == []
        assert parse_script(";;;") == []
        assert parse_script("-- just a comment\n") == []

    def test_single_statement_semicolon_optional(self):
        want = [SelectQuery("t", None, None)]
        assert parse_script("SELECT * FROM t") == want
        assert parse_script("SELECT * FROM t;") == want

    def test_three_statements(self):
        qs = parse_script(
            "CREATE TABLE t (id INT);"
            "INSERT INTO t VALUES (1);"
            "SELECT * FROM t;"
        )
        assert [type(q).__name__ for q in qs] == [
            "CreateTableQuery", "InsertQuery", "SelectQuery"
        ]

    def test_blank_statements_are_ignored(self):
        assert parse_script(";SELECT * FROM a;; ;SELECT * FROM b;;") == [
            SelectQuery("a", None, None), SelectQuery("b", None, None)
        ]

    def test_semicolon_inside_a_string_is_not_a_separator(self):
        qs = parse_script("INSERT INTO t VALUES ('a;b;c'); SELECT * FROM t;")
        assert qs[0] == InsertQuery("t", ("a;b;c",))
        assert len(qs) == 2

    def test_comments_between_statements(self):
        qs = parse_script("-- setup\nCREATE TABLE t (id INT);\n-- done\n")
        assert [type(q).__name__ for q in qs] == ["CreateTableQuery"]

    def test_missing_separator_between_statements(self):
        with pytest.raises(MiniDBError):
            parse_script("SELECT * FROM a SELECT * FROM b")

    def test_syntax_error_in_a_later_statement_propagates(self):
        with pytest.raises(MiniDBError):
            parse_script("SELECT * FROM a; SELECT FROM b;")
