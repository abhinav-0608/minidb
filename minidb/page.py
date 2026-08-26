"""Stage 2 - the slotted page.

Turns a fixed-size byte buffer into a container of variable-length records::

    byte 0
    +---------------------------------------------------+
    | header (9 bytes)                                  |
    |   page_type u8 . num_slots u16 . free_end u16 .   |
    |   next_page_id u32                                |
    +---------------------------------------------------+
    | slot directory - grows DOWN, 4 bytes per slot     |
    |   slot 0: (offset, length)                        |
    |   slot 1: (offset, length)                        |
    +----------------- free_start ---------------------+   free_start = 9 + num_slots*4
    |                 FREE SPACE                        |   free_space  = free_end - free_start
    +----------------- free_end -----------------------+
    |   record 1 bytes                                  |
    |   record 0 bytes  - grows UP                      |
    +---------------------------------------------------+
    byte page_size

``slot_id`` is an index into the directory; slot i always lives at byte
``9 + i*4``. The record bytes may sit anywhere, so ``(page_id, slot_id)`` is
a stable row identifier.

This module does no I/O. The heap layer reads a page through the pager, wraps
the returned bytes in a ``bytearray``, mutates it here, and writes it back.
"""

from __future__ import annotations

import enum
import struct

from .errors import MiniDBError

# page_type u8 | num_slots u16 | free_end u16 | next_page_id u32   (little-endian)
_HEADER = struct.Struct("<BHHI")
# record offset u16 | record length u16
_SLOT = struct.Struct("<HH")


class PageType(enum.IntEnum):
    HEAP = 1
    CATALOG = 2
    BTREE_INTERNAL = 3
    BTREE_LEAF = 4


class SlottedPage:
    HEADER_SIZE = _HEADER.size  # 9
    SLOT_SIZE = _SLOT.size  # 4

    def __init__(self, buf: bytearray, *, expected_type=PageType.HEAP) -> None:
        if not isinstance(buf, bytearray):
            raise MiniDBError("SlottedPage needs a bytearray it can mutate")
        self._buf = buf
        page_type, num_slots, free_end, next_page_id = _HEADER.unpack_from(buf, 0)
        if expected_type is not None and page_type != int(expected_type):
            want = getattr(expected_type, "name", expected_type)
            raise MiniDBError(
                f"expected a {want} page but found page_type {page_type}"
            )
        self._page_type = page_type
        self._num_slots = num_slots
        self._free_end = free_end
        self._next_page_id = next_page_id

    @classmethod
    def init_empty(cls, page_size: int, page_type=PageType.HEAP) -> "SlottedPage":
        buf = bytearray(page_size)
        # a brand-new page: no slots, all bytes below the header are free
        _HEADER.pack_into(buf, 0, int(page_type), 0, page_size, 0)
        return cls(buf, expected_type=page_type)

    # -- header-backed state -----------------------------------------------

    @property
    def buffer(self) -> bytes:
        """The page bytes, ready to hand to ``pager.write_page``."""
        return bytes(self._buf)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def page_type(self) -> int:
        return self._page_type

    @property
    def next_page_id(self) -> int:
        return self._next_page_id

    @next_page_id.setter
    def next_page_id(self, value: int) -> None:
        if not (0 <= value < (1 << 32)):
            raise MiniDBError(f"next_page_id {value} out of range")
        self._next_page_id = value
        self._write_header()

    @property
    def _free_start(self) -> int:
        return self.HEADER_SIZE + self._num_slots * self.SLOT_SIZE

    @property
    def free_space(self) -> int:
        return self._free_end - self._free_start

    def can_fit(self, record_len: int) -> bool:
        """True if a record of this length plus its new slot would fit."""
        return record_len + self.SLOT_SIZE <= self.free_space

    # -- records ---------------------------------------------------------

    def add_record(self, data: bytes) -> int:
        n = len(data)
        if not self.can_fit(n):
            raise MiniDBError(
                f"record of {n} bytes does not fit: page has {self.free_space} "
                f"bytes free, needs {n + self.SLOT_SIZE} (record + slot)"
            )
        new_free_end = self._free_end - n
        self._buf[new_free_end : new_free_end + n] = data
        # the new slot goes at the current end of the directory
        slot_id = self._num_slots
        _SLOT.pack_into(self._buf, self._free_start, new_free_end, n)
        self._num_slots += 1
        self._free_end = new_free_end
        self._write_header()
        return slot_id

    def read_record(self, slot_id: int) -> bytes:
        if not (0 <= slot_id < self._num_slots):
            raise MiniDBError(
                f"slot {slot_id} out of range; page has {self._num_slots} slots"
            )
        offset, length = _SLOT.unpack_from(
            self._buf, self.HEADER_SIZE + slot_id * self.SLOT_SIZE
        )
        return bytes(self._buf[offset : offset + length])

    def __iter__(self):
        for slot_id in range(self._num_slots):
            yield self.read_record(slot_id)

    # -- internals -----------------------------------------------------------

    def _write_header(self) -> None:
        _HEADER.pack_into(
            self._buf,
            0,
            self._page_type,
            self._num_slots,
            self._free_end,
            self._next_page_id,
        )
