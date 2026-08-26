"""Stage 3 - the catalog.

Table metadata lives *inside* the database file, in a table of its own. The
catalog is a Heap (Stage 2) with a fixed schema, rooted at a known page
(``CATALOG_ROOT_PAGE_ID``, which is 1 - the same convention SQLite uses).

One catalog row per user table::

    table_name    TEXT   "users"
    first_page_id INT    2
    column_defs   TEXT   "id INT, name TEXT, age TEXT"

``column_defs`` is the user table's column list rendered as text and parsed
back into a Schema on open. Column names are validated as identifiers when a
table is created, so the ", " and " " separators are unambiguous.

Not crash-atomic yet: a crash partway through creating a table can leak an
allocated-but-unreferenced heap page. Stage 9's WAL makes it atomic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import CATALOG_ROOT_PAGE_ID
from .errors import MiniDBError
from .heap import Heap
from .page import PageType
from .pager import Pager
from .record import Column, ColumnType, Schema

CATALOG_SCHEMA = Schema(
    (
        Column("table_name", ColumnType.TEXT),
        Column("first_page_id", ColumnType.INT),
        Column("column_defs", ColumnType.TEXT),
    )
)


@dataclass(frozen=True)
class TableInfo:
    name: str
    first_page_id: int
    schema: Schema  # the *user* table's schema


def encode_columns(schema: Schema) -> str:
    return ", ".join(f"{c.name} {c.type.value}" for c in schema.columns)


def decode_columns(spec: str) -> Schema:
    columns = []
    for part in spec.split(","):
        tokens = part.split()
        if len(tokens) != 2:
            raise MiniDBError(f"corrupt catalog: bad column definition {part!r}")
        name, typename = tokens
        try:
            col_type = ColumnType(typename)
        except ValueError:
            raise MiniDBError(
                f"corrupt catalog: unknown column type {typename!r}"
            ) from None
        columns.append(Column(name, col_type))
    try:
        return Schema(tuple(columns))
    except MiniDBError as e:
        raise MiniDBError(f"corrupt catalog: {e}") from e


class Catalog:
    def __init__(self, heap: Heap) -> None:
        self._heap = heap
        self._tables: dict[str, TableInfo] = {}
        for name, first_page_id, defs in heap.scan():
            self._tables[name] = TableInfo(
                name, first_page_id, decode_columns(defs)
            )

    @classmethod
    def open(cls, pager: Pager) -> "Catalog":
        if pager.page_count <= CATALOG_ROOT_PAGE_ID:
            # fresh file: nothing past the header page yet, so make the catalog
            heap = Heap.create(pager, CATALOG_SCHEMA, page_type=PageType.CATALOG)
            if heap.first_page_id != CATALOG_ROOT_PAGE_ID:  # pragma: no cover
                raise MiniDBError(
                    f"catalog landed on page {heap.first_page_id}, "
                    f"expected {CATALOG_ROOT_PAGE_ID}"
                )
        else:
            # existing file: page 1 must parse as a catalog page
            heap = Heap(
                pager,
                CATALOG_SCHEMA,
                CATALOG_ROOT_PAGE_ID,
                page_type=PageType.CATALOG,
            )
        return cls(heap)

    def has(self, name: str) -> bool:
        return name in self._tables

    def get(self, name: str) -> TableInfo:
        try:
            return self._tables[name]
        except KeyError:
            raise MiniDBError(f"no such table: {name}") from None

    def table_names(self) -> list[str]:
        return list(self._tables)

    def add(self, name: str, first_page_id: int, schema: Schema) -> None:
        self._heap.insert((name, first_page_id, encode_columns(schema)))
        self._tables[name] = TableInfo(name, first_page_id, schema)
