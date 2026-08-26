"""Stage 3 + Stage 9 - the Database.

Owns the pager and the catalog and exposes table-level operations. With
``wal=True`` (the default) every mutating call - ``create_table`` and
``Table.insert`` - runs as one transaction: it commits on success, aborts on
error, and survives a crash via the write-ahead log.
"""

from __future__ import annotations

import contextlib
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
    def __init__(
        self,
        path: str,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        wal: bool = True,
    ) -> None:
        self._pager = Pager(path, page_size=page_size, wal=wal)
        try:
            # On a brand-new file this creates the catalog page, so it must be
            # a transaction. On an existing file it is a read-only load.
            with self._transaction():
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
        return self._pager.page_count

    # -- transactions ---------------------------------------------------

    @contextlib.contextmanager
    def _transaction(self):
        """Own a transaction if none is active, else join the caller's. On an
        aborted owned transaction, reload the catalog from the rolled-back
        pages so in-memory state matches disk."""
        own = self._pager.wal_enabled and not self._pager.in_transaction
        if own:
            self._pager.begin()
        try:
            yield
        except BaseException:
            if own:
                self._pager.abort()
                self._catalog = Catalog.open(self._pager)
            raise
        else:
            if own:
                self._pager.commit()

    # -- tables -----------------------------------------------------------

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

        with self._transaction():
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
