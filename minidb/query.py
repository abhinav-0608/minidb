"""Typed query objects - the interface between the SQL parser (Stage 7) and
the executor (Stage 4+).

The grammar has no nested expressions, so one flat frozen dataclass per
statement kind is enough: no general AST, no visitor. Stage 4 needs only
``SelectQuery``; ``CreateTableQuery`` and ``InsertQuery`` arrive with the
parser.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    column: str
    op: str  # only "=" is supported for now
    value: object  # int for an INT column, str for a TEXT column


@dataclass(frozen=True)
class SelectQuery:
    table: str
    columns: tuple[str, ...] | None  # None means SELECT *
    condition: Condition | None = None

    def __post_init__(self) -> None:
        if self.columns is not None:
            object.__setattr__(self, "columns", tuple(self.columns))
