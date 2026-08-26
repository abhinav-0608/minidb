"""Stage 4 - the executor (sequential).

Runs a ``SelectQuery`` against a ``Database`` by scanning the whole table.
No index, no planning: every SELECT reads every row of every page. Stage 6
adds an index fast-path for ``WHERE id = <const>``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .database import Database
from .errors import MiniDBError
from .query import SelectQuery
from .record import Column, ColumnType, Schema


@dataclass(frozen=True)
class SelectResult:
    column_names: list[str]
    rows: list[tuple]


def execute_select(db: Database, query: SelectQuery) -> SelectResult:
    schema = db.schema_of(query.table)  # raises "no such table" if unknown
    heap = db.open_table(query.table)

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

    rows: list[tuple] = []
    for row in heap.scan():
        if cond is not None and row[cond_index] != cond.value:
            continue
        rows.append(tuple(row[i] for i in proj))

    return SelectResult(out_names, rows)


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
