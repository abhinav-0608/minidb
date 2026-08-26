"""Stage 1 + Stage 9 - the pager.

The pager is the only part of MiniDB that touches the database file. It hands
callers ``page_size``-byte pages by number, keeps them in a write-back cache,
and (optionally) protects writes with a write-ahead log.

Two modes:

* ``wal=False`` (default for a bare Pager): writes go to the cache and reach
  the file on ``flush()`` / ``close()``. No transactions, no crash safety.
  This is the primitive that the storage-layer unit tests exercise directly.

* ``wal=True`` (what ``Database`` uses): all writes happen inside a
  transaction. ``commit()`` appends full-page after-images to the WAL and
  fsyncs it; only then may pages reach the data file, and only at a
  ``checkpoint()`` (every 128 commits, and on close). ``abort()`` restores
  every touched page from a pre-image. On open, a non-empty WAL is replayed
  (redo of committed transactions) before the file is trusted.

On-disk file layout is unchanged from Stage 1::

    +----------+----------+----------+-----
    | page 0   | page 1   | page 2   | ...
    | (header) | (data)   | (data)   |
    +----------+----------+----------+-----
    byte offset of page N  =  N * page_size

Page 0 (magic, format version, page size, page count) is written directly by
the pager at init / checkpoint / close / end-of-recovery - never through a
transaction.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import struct

from .constants import (
    DEFAULT_PAGE_SIZE,
    FIRST_DATA_PAGE_ID,
    FORMAT_VERSION,
    MAGIC,
    MAX_PAGE_SIZE,
    MIN_PAGE_SIZE,
)
from .errors import MiniDBError
from .wal import WriteAheadLog, recover

__all__ = ["Pager", "MiniDBError", "SimulatedCrash"]

# magic (8) | format_version (u32) | page_size (u32) | page_count (u64) = 24 bytes
_HEADER = struct.Struct("<8sIIQ")

_CHECKPOINT_EVERY = 128  # commits between automatic checkpoints (bounds WAL size)


class SimulatedCrash(BaseException):
    """Raised by the ``crash_at`` test hook. A BaseException, so it bypasses
    ordinary ``except Exception`` handlers the way a real crash would."""


class Pager:
    def __init__(
        self,
        path: str,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        wal: bool = False,
    ) -> None:
        self._path = path
        self._file = None
        self._locked = False
        self._cache: dict[int, bytearray] = {}
        self._dirty: set[int] = set()  # wal=False only
        self._page_size = 0
        self._page_count = 0

        # --- WAL / transaction state (wal=True only) ---
        self._wal_enabled = wal
        self._wal_path = path + "-wal"
        self._wal: WriteAheadLog | None = None
        self._txn_active = False
        self._txn_id = 0
        self._next_txn_id = 1
        self._txn_modified: set[int] = set()
        self._txn_preimages: dict[int, bytearray | None] = {}
        self._txn_base_page_count = 0
        self._pending: set[int] = set()  # committed, not yet in the data file
        self._commits_since_checkpoint = 0
        self.crash_at: str | None = None  # test hook

        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        self._file = os.fdopen(fd, "r+b")
        try:
            self._acquire_lock()
            if os.fstat(fd).st_size == 0:
                self._init_new_file(page_size)
            else:
                self._open_existing_file(page_size)
                if self._wal_enabled:
                    self._recover_if_needed()
            if self._wal_enabled:
                self._wal = WriteAheadLog(self._wal_path)
        except BaseException:
            if self._wal is not None:
                self._wal.close()
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
        wal = " wal" if self._wal_enabled else ""
        return f"<Pager {self._path!r} page_size={self._page_size} {state}{wal}>"

    # -- read-only properties ------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def page_count(self) -> int:
        return self._page_count

    @property
    def wal_enabled(self) -> bool:
        return self._wal_enabled

    @property
    def in_transaction(self) -> bool:
        return self._txn_active

    # -- page access -------------------------------------------------------

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
        return bytes(self._cache[page_id])

    def write_page(self, page_id: int, data: bytes) -> None:
        self._check_open()
        self._check_data_page_id(page_id)
        if len(data) != self._page_size:
            raise MiniDBError(
                f"page must be exactly {self._page_size} bytes, got {len(data)}"
            )
        if self._wal_enabled:
            if not self._txn_active:
                raise MiniDBError("write_page outside a transaction (WAL mode)")
            if page_id not in self._txn_preimages:
                self._txn_preimages[page_id] = self._cache.get(page_id)
            self._txn_modified.add(page_id)
        else:
            self._dirty.add(page_id)
        self._cache[page_id] = bytearray(data)

    def allocate_page(self) -> int:
        self._check_open()
        if self._wal_enabled and not self._txn_active:
            raise MiniDBError("allocate_page outside a transaction (WAL mode)")
        page_id = self._page_count
        self._page_count += 1
        self._cache[page_id] = bytearray(self._page_size)
        if self._wal_enabled:
            self._txn_preimages[page_id] = None  # brand new: rollback == evict
            self._txn_modified.add(page_id)
        else:
            self._dirty.add(page_id)
        return page_id

    # -- transactions (wal=True) -----------------------------------------

    @contextlib.contextmanager
    def transaction(self):
        """Own a transaction if none is active, otherwise join the caller's."""
        own = self._wal_enabled and not self._txn_active
        if own:
            self.begin()
        try:
            yield
        except BaseException:
            if own:
                self.abort()
            raise
        else:
            if own:
                self.commit()

    def begin(self) -> None:
        self._check_open()
        if not self._wal_enabled:
            return
        if self._txn_active:
            raise MiniDBError("a transaction is already active")
        self._txn_active = True
        self._txn_id = self._next_txn_id
        self._next_txn_id += 1
        self._txn_modified = set()
        self._txn_preimages = {}
        self._txn_base_page_count = self._page_count

    def commit(self) -> None:
        self._check_open()
        if not self._wal_enabled or not self._txn_active:
            return
        modified = sorted(self._txn_modified)
        if modified:
            for pid in modified:
                self._wal.append_page(self._txn_id, pid, bytes(self._cache[pid]))
            self._maybe_crash("after_page_records")
            self._wal.append_commit(self._txn_id)
            self._wal.fsync()
            self._maybe_crash("after_wal_fsync")
            self._pending.update(modified)
            self._commits_since_checkpoint += 1
        self._txn_modified = set()
        self._txn_preimages = {}
        self._txn_active = False
        if self._commits_since_checkpoint >= _CHECKPOINT_EVERY:
            self.checkpoint()

    def abort(self) -> None:
        self._check_open()
        if not self._wal_enabled or not self._txn_active:
            return
        for pid, pre in self._txn_preimages.items():
            if pre is None:
                self._cache.pop(pid, None)
            else:
                self._cache[pid] = pre
        self._page_count = self._txn_base_page_count
        self._txn_modified = set()
        self._txn_preimages = {}
        self._txn_active = False

    def checkpoint(self) -> None:
        self._check_open()
        if not self._wal_enabled:
            return
        for pid in sorted(self._pending):
            self._file.seek(pid * self._page_size)
            self._file.write(self._cache[pid])
        self._maybe_crash("checkpoint_before_header")
        self._write_header()
        self._file.flush()
        os.fsync(self._file.fileno())
        self._maybe_crash("checkpoint_before_truncate")
        self._wal.truncate()
        self._pending = set()
        self._commits_since_checkpoint = 0

    # -- durability for wal=False ------------------------------------------

    def flush(self) -> None:
        """wal=False: write dirty pages + header to the file and fsync.
        wal=True: a checkpoint."""
        self._check_open()
        if self._wal_enabled:
            self.checkpoint()
            return
        self._write_header()
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
            if self._wal_enabled:
                if self._txn_active:
                    self.abort()
                self.checkpoint()
            else:
                self.flush()
        finally:
            if self._wal is not None:
                self._wal.close()
                self._wal = None
            if self._locked:
                with contextlib.suppress(OSError):
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                self._locked = False
            self._file.close()
            self._file = None

    def _abandon(self) -> None:
        """Test helper: drop handles with no checkpoint, truncate, or flush -
        i.e. simulate the process dying."""
        if self._wal is not None:
            with contextlib.suppress(OSError):
                self._wal.close()
            self._wal = None
        if self._file is not None:
            if self._locked:
                with contextlib.suppress(OSError):
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                self._locked = False
            self._file.close()
            self._file = None

    # -- setup / recovery ---------------------------------------------------

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
        self._page_count = 1
        self._write_header()
        self._file.flush()
        os.fsync(self._file.fileno())
        if self._wal_enabled and os.path.exists(self._wal_path):
            open(self._wal_path, "wb").close()  # clear any stale WAL

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
        self._page_size = page_size
        if self._wal_enabled:
            # The header's page_count is advisory in WAL mode; recovery (or a
            # later checkpoint) reconciles it with the file. A file longer than
            # the header claims is a mid-checkpoint crash, not corruption.
            self._page_count = pages_on_disk
        else:
            if pages_on_disk != page_count:
                raise MiniDBError(
                    f"{self._path!r} header says {page_count} pages but the "
                    f"file holds {pages_on_disk} (truncated or corrupt)"
                )
            self._page_count = pages_on_disk

    def _recover_if_needed(self) -> None:
        if (
            not os.path.exists(self._wal_path)
            or os.path.getsize(self._wal_path) == 0
        ):
            return

        def apply(page_id: int, image: bytes) -> None:
            self._file.seek(page_id * self._page_size)
            self._file.write(image)

        recover(self._wal_path, apply)
        self._file.flush()
        os.fsync(self._file.fileno())

        size = os.fstat(self._file.fileno()).st_size
        if size % self._page_size != 0:
            raise MiniDBError(
                f"{self._path!r}: {size} bytes after recovery is not a "
                f"multiple of page_size {self._page_size}"
            )
        self._page_count = size // self._page_size
        self._write_header()
        self._file.flush()
        os.fsync(self._file.fileno())
        open(self._wal_path, "wb").close()  # its contents are now in the file

    def _write_header(self) -> None:
        buf = bytearray(self._page_size)
        _HEADER.pack_into(
            buf, 0, MAGIC, FORMAT_VERSION, self._page_size, self._page_count
        )
        self._file.seek(0)
        self._file.write(buf)

    def _maybe_crash(self, point: str) -> None:
        if self.crash_at == point:
            self.crash_at = None
            raise SimulatedCrash(f"simulated crash at {point!r}")

    # -- validation ------------------------------------------------------

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
