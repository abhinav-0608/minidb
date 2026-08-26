"""Stage 1 - the pager.

The pager is the ONLY part of MiniDB that touches the database file. Every
other component asks for a page by number, gets back exactly ``page_size``
bytes, and hands back a full page of bytes to store. Nothing above this
layer knows about byte offsets, file handles, or fsync.

On-disk model
-------------
The file is a contiguous array of fixed-size pages::

    +----------+----------+----------+----------+-----
    | page 0   | page 1   | page 2   | page 3   | ...
    | (header) | (data)   | (data)   | (data)   |
    +----------+----------+----------+----------+-----
    0         PS        2*PS       3*PS       4*PS      byte offsets, PS = page_size

    byte offset of page N  =  N * page_size

Page 0 is reserved for the file header and is owned by the pager. Callers
get page ids starting at 1 (``constants.FIRST_DATA_PAGE_ID``).

In-memory model
---------------
Pages pass through a write-back cache - a plain dict ``{page_id: bytearray}``:

* ``read_page``  - returns the cached copy, loading it from the file on a miss.
* ``write_page`` - replaces the cached copy and marks the page dirty. It does
  NOT touch the file.
* ``flush``      - writes every dirty page to the file, then fsyncs.
* ``close``      - flush, release the lock, close the handle.

Because writes stay in the cache until ``flush()``, a read after a write
(with no flush in between) returns what you just wrote. Durability happens
only at ``flush()`` / ``close()``. Stage 1 therefore guarantees persistence
across a *clean* shutdown; surviving a crash mid-write is Stage 9 (the WAL).

The cache is intentionally dumb: unbounded, no eviction, no LRU. A real
buffer pool has a fixed set of frames and an eviction policy; we will not
need that here.

Concurrency
-----------
Single process, single thread. On open the pager takes an exclusive advisory
lock (``flock``) on the file so a second MiniDB process cannot open the same
database and corrupt it. There is no in-process locking.
"""

from __future__ import annotations

import fcntl
import os
import struct

from .constants import (
    DEFAULT_PAGE_SIZE,
    FIRST_DATA_PAGE_ID,
    FORMAT_VERSION,
    HEADER_PAGE_ID,
    MAGIC,
    MAX_PAGE_SIZE,
    MIN_PAGE_SIZE,
)
from .errors import MiniDBError

__all__ = ["Pager", "MiniDBError"]


# File header, stored at the start of page 0. Little-endian, no padding:
#   8s  magic           - identifies the file as a MiniDB database
#   I   format_version  - on-disk layout version (constants.FORMAT_VERSION)
#   I   page_size       - bytes per page; fixed for the life of the file
#   Q   page_count      - total number of pages, including page 0
# The rest of page 0 stays zero, reserved for later stages.
_HEADER = struct.Struct("<8sIIQ")


class Pager:
    def __init__(self, path: str, *, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        self._path = path
        self._file = None
        self._locked = False
        self._cache: dict[int, bytearray] = {}
        self._dirty: set[int] = set()
        self._page_size = 0
        self._page_count = 0

        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        self._file = os.fdopen(fd, "r+b")
        try:
            self._acquire_lock()
            if os.fstat(fd).st_size == 0:
                self._init_new_file(page_size)
            else:
                self._open_existing_file(page_size)
        except BaseException:
            self._file.close()
            self._file = None
            raise

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "Pager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._file is None else f"{self._page_count} pages"
        return f"<Pager {self._path!r} page_size={self._page_size} {state}>"

    # -- read-only properties ------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def page_count(self) -> int:
        """Total pages in the file, including the header page 0."""
        return self._page_count

    # -- public API ----------------------------------------------------------

    def read_page(self, page_id: int) -> bytes:
        self._check_open()
        self._check_data_page_id(page_id)
        if page_id not in self._cache:
            self._file.seek(page_id * self._page_size)
            data = self._file.read(self._page_size)
            if len(data) != self._page_size:
                raise MiniDBError(
                    f"short read on page {page_id}: expected {self._page_size} "
                    f"bytes, got {len(data)} (file truncated or corrupt)"
                )
            self._cache[page_id] = bytearray(data)
        # Hand back an immutable copy so a caller cannot mutate the cache
        # behind our back without going through write_page().
        return bytes(self._cache[page_id])

    def write_page(self, page_id: int, data: bytes) -> None:
        self._check_open()
        self._check_data_page_id(page_id)
        if len(data) != self._page_size:
            raise MiniDBError(
                f"page must be exactly {self._page_size} bytes, got {len(data)}"
            )
        # Copy the caller's bytes so their later edits do not leak into the cache.
        self._cache[page_id] = bytearray(data)
        self._dirty.add(page_id)

    def allocate_page(self) -> int:
        """Add one new zero-filled page and return its id.

        In-memory only; the file grows on the next ``flush()``, when the
        dirty page is written at ``new_id * page_size``.
        """
        self._check_open()
        page_id = self._page_count
        self._page_count += 1
        self._cache[page_id] = bytearray(self._page_size)
        self._dirty.add(page_id)
        return page_id

    def flush(self) -> None:
        """Write every dirty page to the file, then fsync.

        Two buffer layers sit between us and the platter::

            this process --write()--> OS page cache --fsync()--> disk

        ``file.flush()`` pushes the first arrow, ``os.fsync()`` the second.
        """
        self._check_open()
        # Page 0 must always reflect the current page_count.
        self._cache[HEADER_PAGE_ID] = self._build_header_page()
        self._dirty.add(HEADER_PAGE_ID)
        # Ascending order so the file is extended contiguously.
        for page_id in sorted(self._dirty):
            self._file.seek(page_id * self._page_size)
            self._file.write(self._cache[page_id])
        self._file.flush()
        os.fsync(self._file.fileno())
        self._dirty.clear()

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self.flush()
        finally:
            if self._locked:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                self._locked = False
            self._file.close()
            self._file = None

    # -- setup helpers -----------------------------------------------------

    def _acquire_lock(self) -> None:
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise MiniDBError(
                f"cannot lock {self._path!r}: another MiniDB process has this "
                f"database open"
            )
        self._locked = True

    def _init_new_file(self, page_size: int) -> None:
        if not (MIN_PAGE_SIZE <= page_size <= MAX_PAGE_SIZE):
            raise MiniDBError(
                f"page_size {page_size} outside "
                f"[{MIN_PAGE_SIZE}, {MAX_PAGE_SIZE}]"
            )
        self._page_size = page_size
        self._page_count = 1  # page 0, the header
        self._cache[HEADER_PAGE_ID] = self._build_header_page()
        self._dirty.add(HEADER_PAGE_ID)
        self.flush()  # make the file a valid MiniDB database on disk now

    def _open_existing_file(self, requested_page_size: int) -> None:
        size = os.fstat(self._file.fileno()).st_size
        if size < _HEADER.size:
            raise MiniDBError(
                f"{self._path!r} is only {size} bytes, too small to be a "
                f"MiniDB database"
            )
        self._file.seek(0)
        magic, version, page_size, page_count = _HEADER.unpack(
            self._file.read(_HEADER.size)
        )
        if magic != MAGIC:
            raise MiniDBError(
                f"{self._path!r} is not a MiniDB database (bad magic)"
            )
        if version != FORMAT_VERSION:
            raise MiniDBError(
                f"{self._path!r} is on-disk format v{version}; this build only "
                f"understands v{FORMAT_VERSION}"
            )
        if not (MIN_PAGE_SIZE <= page_size <= MAX_PAGE_SIZE):
            raise MiniDBError(
                f"{self._path!r} header has an implausible page_size "
                f"({page_size})"
            )
        if (
            requested_page_size != DEFAULT_PAGE_SIZE
            and requested_page_size != page_size
        ):
            raise MiniDBError(
                f"{self._path!r} was created with page_size {page_size}, but "
                f"page_size {requested_page_size} was requested"
            )
        if size % page_size != 0:
            raise MiniDBError(
                f"{self._path!r} is {size} bytes, not a whole number of "
                f"{page_size}-byte pages (truncated or corrupt)"
            )
        pages_on_disk = size // page_size
        if pages_on_disk != page_count:
            raise MiniDBError(
                f"{self._path!r} header says {page_count} pages but the file "
                f"holds {pages_on_disk} (truncated or corrupt)"
            )
        self._page_size = page_size
        self._page_count = page_count

    def _build_header_page(self) -> bytearray:
        page = bytearray(self._page_size)
        _HEADER.pack_into(
            page, 0, MAGIC, FORMAT_VERSION, self._page_size, self._page_count
        )
        return page

    # -- validation helpers ----------------------------------------------

    def _check_open(self) -> None:
        if self._file is None:
            raise MiniDBError("pager is closed")

    def _check_data_page_id(self, page_id: int) -> None:
        if not isinstance(page_id, int) or isinstance(page_id, bool):
            raise MiniDBError(
                f"page_id must be an int, got {type(page_id).__name__}"
            )
        if page_id < FIRST_DATA_PAGE_ID:
            raise MiniDBError(
                f"page_id {page_id} is reserved (page 0 is the file header); "
                f"data pages start at {FIRST_DATA_PAGE_ID}"
            )
        if page_id >= self._page_count:
            raise MiniDBError(
                f"page_id {page_id} is out of range; the file has "
                f"{self._page_count} pages "
                f"(valid data ids 1..{self._page_count - 1})"
            )
