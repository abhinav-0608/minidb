"""Stage 2 - the heap file.

A table's rows live in a singly linked list of slotted pages ("a heap
file"): unordered, append-only for now.

    first_page_id = 1
    page 1 --next--> page 4 --next--> page 7 --next--> 0   (0 = end of chain)
    [rows...]        [rows...]        [rows...]

``insert`` appends to the last page, allocating and linking a fresh page
when the last one is full. Small gaps left in earlier pages are not
reclaimed - there is no DELETE or VACUUM.

The catalog (Stage 3) will persist each table's ``first_page_id``; until
then the caller holds on to it.
"""

from __future__ import annotations

from collections.abc import Iterator

from .errors import MiniDBError
from .page import PageType, SlottedPage
from .pager import Pager
from .record import Schema


class Heap:
    def __init__(self, pager: Pager, schema: Schema, first_page_id: int) -> None:
        self._pager = pager
        self._schema = schema
        self._first_page_id = first_page_id
        # walk the chain once so inserts can append in O(1)
        self._last_page_id = self._walk_to_last_page(first_page_id)

    @classmethod
    def create(cls, pager: Pager, schema: Schema) -> "Heap":
        page_id = pager.allocate_page()
        empty = SlottedPage.init_empty(pager.page_size, PageType.HEAP)
        pager.write_page(page_id, empty.buffer)
        return cls(pager, schema, page_id)

    @property
    def first_page_id(self) -> int:
        return self._first_page_id

    @property
    def schema(self) -> Schema:
        return self._schema

    # -- write ---------------------------------------------------------------

    def insert(self, values) -> tuple[int, int]:
        """Append a row. Returns its RID ``(page_id, slot_id)``."""
        data = self._schema.encode_row(values)
        max_record = (
            self._pager.page_size
            - SlottedPage.HEADER_SIZE
            - SlottedPage.SLOT_SIZE
        )
        if len(data) > max_record:
            raise MiniDBError(
                f"row encodes to {len(data)} bytes; the most a "
                f"{self._pager.page_size}-byte page can hold is {max_record}"
            )

        last = SlottedPage(bytearray(self._pager.read_page(self._last_page_id)))
        if last.can_fit(len(data)):
            slot_id = last.add_record(data)
            self._pager.write_page(self._last_page_id, last.buffer)
            return (self._last_page_id, slot_id)

        # last page is full: allocate a fresh one and link it on
        new_id = self._pager.allocate_page()
        new_page = SlottedPage.init_empty(self._pager.page_size, PageType.HEAP)
        slot_id = new_page.add_record(data)
        last.next_page_id = new_id
        self._pager.write_page(self._last_page_id, last.buffer)  # persist the link
        self._pager.write_page(new_id, new_page.buffer)
        self._last_page_id = new_id
        return (new_id, slot_id)

    # -- read ------------------------------------------------------------

    def scan(self) -> Iterator[tuple]:
        """Yield every row, in chain then slot order."""
        seen: set[int] = set()
        page_id = self._first_page_id
        while page_id != 0:
            if page_id in seen:
                raise MiniDBError(f"heap page chain has a cycle at page {page_id}")
            seen.add(page_id)
            page = SlottedPage(bytearray(self._pager.read_page(page_id)))
            for slot_id in range(page.num_slots):
                yield self._schema.decode_row(page.read_record(slot_id))
            page_id = page.next_page_id

    # -- internals -----------------------------------------------------------

    def _walk_to_last_page(self, page_id: int) -> int:
        seen: set[int] = set()
        while True:
            if page_id in seen:
                raise MiniDBError(f"heap page chain has a cycle at page {page_id}")
            seen.add(page_id)
            page = SlottedPage(bytearray(self._pager.read_page(page_id)))
            if page.next_page_id == 0:
                return page_id
            page_id = page.next_page_id
