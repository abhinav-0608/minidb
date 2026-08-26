"""Stage 4/6 - the executor.

Runs a ``SelectQuery`` against a ``Database``. The one optimisation: when the
condition is ``<id> = <const>`` (column 0 is always the unique, indexed id),
it does a B+ tree lookup instead of a full scan. Everything else is a scan.

``use_index=False`` forces the scan path - the benchmark and the "index must
agree with scan" tests rely on that.
"""

from __future__ import annotations

from dataclasses import dataclass

from .database import Database
from .errors import MiniDBError
from .query import CreateTableQuery, InsertQuery, SelectQuery
from .record import Column, ColumnType, Schema
from .sql import parse, parse_script


@dataclass(frozen=True)
class SelectResult:
    column_names: list[str]
    rows: list[tuple]


def execute_select(
    db: Database, query: SelectQuery, *, use_index: bool = True
) -> SelectResult:
    schema = db.schema_of(query.table)  # raises "no such table" if unknown
    table = db.open_table(query.table)

    # projection: which column indices to return, in the requested order
    if query.columns is None:
        proj = list(range(len(schema.columns)))
    else:
        proj = [_resolve_column(schema, name) for name in query.columns]
    out_names = [schema.columns[i].name for i in proj]

    # optional WHERE <column> = <value>
    cond = query.condition
    cond_index = -1
    if cond is not None:
        if cond.op != "=":
            raise MiniDBError(
                f"unsupported operator {cond.op!r} (only '=' is supported)"
            )
        cond_index = _resolve_column(schema, cond.column)
        _check_comparable(schema.columns[cond_index], cond.value)

    # fast path: equality on column 0 (the unique, indexed id)
    if use_index and cond is not None and cond_index == 0:
        row = table.lookup_by_id(cond.value)
        rows = [] if row is None else [tuple(row[i] for i in proj)]
        return SelectResult(out_names, rows)

    rows = []
    for row in table.scan():
        if cond is not None and row[cond_index] != cond.value:
            continue
        rows.append(tuple(row[i] for i in proj))

    return SelectResult(out_names, rows)


def execute(db: Database, query):
    """Run one typed query. Returns a SelectResult for SELECT, else None."""
    if isinstance(query, SelectQuery):
        return execute_select(db, query)
    if isinstance(query, InsertQuery):
        db.open_table(query.table).insert(query.values)
        return None
    if isinstance(query, CreateTableQuery):
        db.create_table(query.table, Schema(query.columns))
        return None
    raise MiniDBError(f"cannot execute a {type(query).__name__}")


def execute_sql(db: Database, sql: str):
    """Parse and run exactly one SQL statement."""
    return execute(db, parse(sql))


def execute_script(db: Database, sql: str) -> list:
    """Parse and run a ';'-separated script; returns one result per statement."""
    return [execute(db, q) for q in parse_script(sql)]


def _resolve_column(schema: Schema, name: str) -> int:
    try:
        return schema.column_index(name)
    except MiniDBError:
        raise MiniDBError(
            f"no column {name!r} in table "
            f"(have: {', '.join(schema.column_names)})"
        ) from None


def _check_comparable(column: Column, value) -> None:
    if column.type is ColumnType.INT and type(value) is not int:
        raise MiniDBError(
            f"cannot compare INT column {column.name!r} with a "
            f"{type(value).__name__}"
        )
    if column.type is ColumnType.TEXT and type(value) is not str:
        raise MiniDBError(
            f"cannot compare TEXT column {column.name!r} with a "
            f"{type(value).__name__}"
        )
