"""Stage 8 tests - the REPL and the .sql script runner.

* TestPrint        - _print output format (table, NULL, OK, row count)
* TestRunScript    - full-script output; parse-time vs execute-time failure
* TestRunRepl      - line accumulation, continuation, dot-commands, recovery
* TestMain         - `python -m minidb <db> [script]` entry point
"""

from __future__ import annotations

import io

import pytest

from minidb.database import Database
from minidb.executor import SelectResult, execute_sql
from minidb.repl import CONT, PROMPT, _print, main, run_repl, run_script

DDL = "CREATE TABLE users (id INT, name TEXT, age INT);"


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "r.db"))
    yield d
    d.close()


def feed(db, text: str) -> str:
    out = io.StringIO()
    run_repl(db, stdin=io.StringIO(text), stdout=out)
    return out.getvalue()


# ---------------------------------------------------------------------------
class TestPrint:
    def test_table_with_rows(self):
        out = io.StringIO()
        _print(SelectResult(["id", "name"], [(1, "Alice"), (2, "Bob")]), out)
        assert out.getvalue() == (
            "id | name\n1 | Alice\n2 | Bob\n(2 rows)\n"
        )

    def test_singular_row_count(self):
        out = io.StringIO()
        _print(SelectResult(["x"], [(9,)]), out)
        assert out.getvalue().endswith("(1 row)\n")

    def test_empty_result_keeps_the_header(self):
        out = io.StringIO()
        _print(SelectResult(["a", "b"], []), out)
        assert out.getvalue() == "a | b\n(0 rows)\n"

    def test_null_renders_as_the_word_NULL(self):
        out = io.StringIO()
        _print(SelectResult(["a", "b"], [(1, None), (None, "x")]), out)
        assert out.getvalue() == "a | b\n1 | NULL\nNULL | x\n(2 rows)\n"

    def test_none_result_prints_OK(self):
        out = io.StringIO()
        _print(None, out)
        assert out.getvalue() == "OK\n"


# ---------------------------------------------------------------------------
class TestRunScript:
    def test_create_insert_select_output(self, db):
        out = io.StringIO()
        run_script(
            db,
            DDL + "INSERT INTO users VALUES (1, 'Alice', 24);SELECT * FROM users;",
            out,
        )
        assert out.getvalue() == (
            "OK\nOK\nid | name | age\n1 | Alice | 24\n(1 row)\n"
        )

    def test_runs_the_example_script(self, db, tmp_path):
        out = io.StringIO()
        with open("examples/demo.sql", encoding="utf-8") as f:
            run_script(db, f.read(), out)
        text = out.getvalue()
        assert "1 | Alice | 24" in text
        assert "(4 rows)" in text
        assert text.count("OK\n") == 5  # 1 CREATE + 4 INSERTs

    def test_syntax_error_stops_the_whole_batch(self, db):
        out = io.StringIO()
        run_script(db, "CREATE TABLE t (id INT); SELECT FROM t;", out)
        assert out.getvalue().startswith("error:")
        assert db.table_names() == []  # the CREATE never ran

    def test_execute_error_stops_after_earlier_statements(self, db):
        out = io.StringIO()
        run_script(
            db,
            "CREATE TABLE t (id INT);"
            "INSERT INTO t VALUES (1, 2);"
            "SELECT * FROM t;",
            out,
        )
        text = out.getvalue()
        assert text.startswith("OK\n")  # the CREATE ran
        assert "error:" in text
        assert "row)" not in text  # the trailing SELECT did not run
        assert db.table_names() == ["t"]


# ---------------------------------------------------------------------------
class TestRunRepl:
    def test_single_statement(self, db):
        text = feed(db, DDL + "\nSELECT * FROM users;\n")
        assert "id | name | age" in text
        assert "(0 rows)" in text

    def test_multi_line_statement_is_accumulated(self, db):
        text = feed(
            db,
            DDL + "\n"
            "INSERT INTO users VALUES (1, 'Alice', 24);\n"
            "SELECT name\n"
            "   FROM users\n"
            "   WHERE id = 1;\n",
        )
        assert "name\nAlice\n(1 row)\n" in text

    def test_continuation_prompt_appears_for_unfinished_input(self, db):
        text = feed(db, "SELECT *\nFROM users;\n")
        assert CONT in text

    def test_dot_tables(self, db):
        text = feed(db, DDL + "\n.tables\n")
        assert "users\n" in text  # .tables printed the table name

    def test_dot_tables_when_empty(self, db):
        assert "(no tables)" in feed(db, ".tables\n")

    def test_dot_exit_stops_processing(self, db):
        text = feed(db, ".exit\nSELECT * FROM users;\n")
        assert "id | name" not in text

    def test_an_error_does_not_kill_the_session(self, db):
        text = feed(db, "SELECT bogus;\n" + DDL + "\nSELECT * FROM users;\n")
        assert "error:" in text
        assert "id | name | age" in text  # later statements still ran

    def test_semicolon_inside_a_string_is_not_a_terminator(self, db):
        text = feed(
            db,
            "CREATE TABLE t (id INT, v TEXT);\n"
            "INSERT INTO t VALUES (1, 'a;b');\n"
            "SELECT v FROM t WHERE id = 1;\n",
        )
        assert "v\na;b\n(1 row)\n" in text

    def test_line_ending_in_semicolon_inside_unterminated_string_fails_safely(self, db):
        text = feed(db, "INSERT INTO t VALUES (1, 'oops;\n")
        assert "error:" in text and "unterminated" in text

    def test_trailing_buffer_runs_at_eof_without_semicolon(self, db):
        execute_sql(db, DDL.rstrip(";"))
        text = feed(db, "SELECT * FROM users")  # no ';', no newline
        assert "id | name | age" in text


# ---------------------------------------------------------------------------
class TestMain:
    def test_runs_a_script_file_and_persists(self, tmp_path, capsys):
        db_path = str(tmp_path / "m.db")
        script = tmp_path / "s.sql"
        script.write_text(
            DDL + "\nINSERT INTO users VALUES (1, 'Alice', 24);\n"
            "SELECT * FROM users;\n"
        )
        assert main([db_path, str(script)]) == 0
        assert "1 | Alice | 24" in capsys.readouterr().out
        with Database(db_path) as db:
            assert execute_sql(db, "SELECT * FROM users").rows == [(1, "Alice", 24)]

    def test_no_args_is_a_usage_error(self, capsys):
        assert main([]) == 2
        assert "usage" in capsys.readouterr().err

    def test_starts_the_repl_when_no_script(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO(DDL + "\n.exit\n"))
        assert main([str(tmp_path / "m.db")]) == 0
        assert "OK" in capsys.readouterr().out
