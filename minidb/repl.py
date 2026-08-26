"""Stage 8 - the interactive shell and the .sql script runner.

    python -m minidb mydb.db              # interactive
    python -m minidb mydb.db setup.sql    # run a script, then exit
"""

from __future__ import annotations

import sys

from .database import Database
from .errors import MiniDBError
from .executor import SelectResult, execute
from .sql import parse_script

PROMPT = "MiniDB> "
CONT = "   ...> "


def run_script(db: Database, sql: str, out=None) -> None:
    """Parse and run every statement in ``sql``, printing results.

    Stops at the first error (parse or execute) - later statements are not
    run against a database left in an unexpected state.
    """
    out = out if out is not None else sys.stdout
    try:
        statements = parse_script(sql)
    except MiniDBError as exc:
        out.write(f"error: {exc}\n")
        return
    for query in statements:
        try:
            result = execute(db, query)
        except MiniDBError as exc:
            out.write(f"error: {exc}\n")
            return
        _print(result, out)


def run_repl(db: Database, *, stdin=None, stdout=None) -> None:
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    buffer = ""
    stdout.write(PROMPT)
    stdout.flush()
    for raw in stdin:
        if not buffer.strip():  # at a fresh prompt: dot-commands and blanks
            line = raw.strip()
            if line in (".exit", ".quit"):
                break
            if line == ".tables":
                names = db.table_names()
                stdout.write((", ".join(names) or "(no tables)") + "\n")
                stdout.write(PROMPT)
                stdout.flush()
                continue
            if not line:
                stdout.write(PROMPT)
                stdout.flush()
                continue

        buffer += raw
        if buffer.strip().endswith(";"):
            run_script(db, buffer, stdout)
            buffer = ""
            stdout.write(PROMPT)
        else:
            stdout.write(CONT)
        stdout.flush()

    if buffer.strip():  # EOF with an unterminated buffer: try it anyway
        run_script(db, buffer, stdout)


def _print(result, out) -> None:
    if not isinstance(result, SelectResult):
        out.write("OK\n")
        return
    out.write(" | ".join(result.column_names) + "\n")
    for row in result.rows:
        out.write(
            " | ".join("NULL" if v is None else str(v) for v in row) + "\n"
        )
    n = len(result.rows)
    out.write(f"({n} row{'' if n == 1 else 's'})\n")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m minidb <database.db> [script.sql]", file=sys.stderr)
        return 2

    db_path = argv[0]
    script_path = argv[1] if len(argv) > 1 else None
    with Database(db_path) as db:
        if script_path is not None:
            with open(script_path, encoding="utf-8") as f:
                run_script(db, f.read())
        else:
            run_repl(db)
    return 0
