"""Stage 3 - the Database.

The first component that ties two subsystems together: it owns the pager and
the catalog, and exposes table-level operations. Query execution (Stage 4+)
sits on top of this.
"""

from __future__ import annotations

import re

from .catalog import Catalog
from .constants import DEFAULT_PAGE_SIZE
from .errors import MiniDBError
from .heap import Heap
from .index import BTreeIndex
from .page import PageType
from .pager import Pager
from .record import ColumnType, Schema
from .table import Table

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Database:
    def __init__(self, path: str, *, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        self._pager = Pager(path, page_size=page_size)
        try:
            self._catalog = Catalog.open(self._pager)
        except BaseException:
            self._pager.close()
            raise

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._pager.close()

    @property
    def page_count(self) -> int:
        """Total pages in the file (for diagnostics and benchmarks)."""
        return self._pager.page_count

    # -- tables ------------------------------------------------------------

    def create_table(self, name: str, schema: Schema) -> None:
        if not _IDENT.fullmatch(name):
            raise MiniDBError(f"invalid table name: {name!r}")
        for c in schema.columns:
            if not _IDENT.fullmatch(c.name):
                raise MiniDBError(f"invalid column name: {c.name!r}")

        first = schema.columns[0]
        if first.name != "id" or first.type is not ColumnType.INT:
            raise MiniDBError(
                "the first column of every table must be 'id INT' "
                f"(got {first.name!r} {first.type.value})"
            )

        if self._catalog.has(name):
            raise MiniDBError(f"table {name!r} already exists")

        # heap page and index root first, catalog row last: a crash in between
        # only leaks pages - it never leaves the catalog pointing at a page
        # that is not there.
        heap = Heap.create(self._pager, schema, page_type=PageType.HEAP)
        index = BTreeIndex.create(self._pager)
        self._catalog.add(
            name, heap.first_page_id, index.root_page_id, schema
        )

    def table_names(self) -> list[str]:
        return self._catalog.table_names()

    def schema_of(self, name: str) -> Schema:
        return self._catalog.get(name).schema

    def open_table(self, name: str) -> Table:
        info = self._catalog.get(name)
        return Table(
            self._pager,
            info.schema,
            info.first_page_id,
            info.index_root_page_id,
        )
