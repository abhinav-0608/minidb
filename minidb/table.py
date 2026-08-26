"""Stage 6 - a Table: a heap plus its id index.

Index maintenance has to live where both structures are owned. ``Table``
mirrors the parts of the ``Heap`` interface earlier stages use (``insert``,
``scan``, ``first_page_id``, ``schema``), so it drops in as the return of
``Database.open_table``.

Not crash-atomic yet: a crash between the heap write and the index write
leaves a row the index cannot find. Stage 9's WAL makes the pair atomic.
"""

from __future__ import annotations

from collections.abc import Iterator

from .errors import MiniDBError
from .heap import Heap
from .index import BTreeIndex
from .page import PageType, SlottedPage
from .record import Schema


class Table:
    def __init__(
        self,
        pager,
        schema: Schema,
        heap_first_page_id: int,
        index_root_page_id: int,
    ) -> None:
        self._pager = pager
        self._schema = schema
        self._heap = Heap(pager, schema, heap_first_page_id, page_type=PageType.HEAP)
        self._index = BTreeIndex(pager, index_root_page_id)

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def first_page_id(self) -> int:
        return self._heap.first_page_id

    @property
    def index(self) -> BTreeIndex:
        return self._index

    def insert(self, values) -> tuple[int, int]:
        """Append a row and index its id. Returns the RID."""
        values = tuple(values)
        key = values[0]  # column 0 is always the unique 'id' (decision #4)
        if key in self._index:
            raise MiniDBError(f"duplicate id {key!r} in table")
        rid = self._heap.insert(values)
        self._index.insert(key, rid)
        return rid

    def scan(self) -> Iterator[tuple]:
        return self._heap.scan()

    def lookup_by_id(self, key) -> tuple | None:
        """Row for ``key`` via the index, or ``None`` if absent."""
        rid = self._index.search(key)
        if rid is None:
            return None
        page_id, slot_id = rid
        page = SlottedPage(bytearray(self._pager.read_page(page_id)))
        return self._schema.decode_row(page.read_record(slot_id))
