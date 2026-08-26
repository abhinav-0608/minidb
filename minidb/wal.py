"""Stage 9 - the write-ahead log (redo-only).

The log file is a sequence of length-framed, CRC-checked records::

    [u32 payload_len][u8 type][payload][u32 crc32 of (type byte + payload)]

    type 1  PAGE   : [u64 txn_id][u32 page_id][page bytes]
    type 2  COMMIT : [u64 txn_id]

Recovery scans the file, notes which txn ids have a COMMIT, and replays the
PAGE images of those txns in log order. A frame that is short or fails its
CRC is a torn tail: it and everything after it are discarded. Replaying a
full-page image is idempotent, so recovery can itself be interrupted and
rerun.

This module knows nothing about page size or the database file - it only
frames bytes and hands page images back to a callback.
"""

from __future__ import annotations

import os
import struct
import zlib

_LEN = struct.Struct("<I")
_PAGE_HDR = struct.Struct("<QI")  # txn_id, page_id
_COMMIT_HDR = struct.Struct("<Q")  # txn_id
_TYPE_PAGE = 1
_TYPE_COMMIT = 2


class WriteAheadLog:
    """An append-only log file. One instance per open database."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._file = open(path, "ab", buffering=0)  # unbuffered binary append

    def append_page(self, txn_id: int, page_id: int, image: bytes) -> None:
        self._write(_TYPE_PAGE, _PAGE_HDR.pack(txn_id, page_id) + image)

    def append_commit(self, txn_id: int) -> None:
        self._write(_TYPE_COMMIT, _COMMIT_HDR.pack(txn_id))

    def fsync(self) -> None:
        os.fsync(self._file.fileno())

    def truncate(self) -> None:
        self._file.close()
        self._file = open(self._path, "wb", buffering=0)

    def close(self) -> None:
        self._file.close()

    def _write(self, rtype: int, payload: bytes) -> None:
        crc = zlib.crc32(bytes((rtype,)) + payload) & 0xFFFFFFFF
        self._file.write(
            _LEN.pack(len(payload)) + bytes((rtype,)) + payload + _LEN.pack(crc)
        )


def recover(path: str, apply_page) -> int:
    """Replay committed PAGE images from the WAL at ``path``.

    Calls ``apply_page(page_id, image)`` for each, in log order. Stops at the
    first short or CRC-bad frame. Returns the number of pages applied.
    """
    with open(path, "rb") as f:
        data = f.read()

    pages: list[tuple[int, int, bytes]] = []
    committed: set[int] = set()
    pos, n = 0, len(data)
    while pos + 5 <= n:
        (plen,) = _LEN.unpack_from(data, pos)
        rtype = data[pos + 4]
        end = pos + 5 + plen + 4
        if end > n:
            break  # torn: frame runs past end of file
        payload = data[pos + 5 : pos + 5 + plen]
        (crc,) = _LEN.unpack_from(data, pos + 5 + plen)
        if zlib.crc32(bytes((rtype,)) + payload) & 0xFFFFFFFF != crc:
            break  # torn/corrupt: CRC mismatch
        if rtype == _TYPE_PAGE:
            txn_id, page_id = _PAGE_HDR.unpack_from(payload, 0)
            pages.append((txn_id, page_id, payload[_PAGE_HDR.size :]))
        elif rtype == _TYPE_COMMIT:
            (txn_id,) = _COMMIT_HDR.unpack_from(payload, 0)
            committed.add(txn_id)
        pos = end

    applied = 0
    for txn_id, page_id, image in pages:
        if txn_id in committed:
            apply_page(page_id, image)
            applied += 1
    return applied
