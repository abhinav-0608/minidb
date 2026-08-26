"""Stage 9 tests - the write-ahead log module in isolation.

* TestFraming    - append + recover round-trips; winner/loser selection
* TestTornTail   - short frames, bad CRCs, garbage - detected, earlier data kept
* TestTruncate   - truncate() empties the file and drops prior records
"""

from __future__ import annotations

import os
import struct

import pytest

from minidb import wal
from minidb.wal import WriteAheadLog


@pytest.fixture
def wpath(tmp_path):
    return str(tmp_path / "x.wal")


def build(path, *ops):
    """ops: ("page", txn, page_id, image) | ("commit", txn)"""
    w = WriteAheadLog(path)
    for op in ops:
        if op[0] == "page":
            w.append_page(op[1], op[2], op[3])
        else:
            w.append_commit(op[1])
    w.fsync()
    w.close()


def replay(path):
    out = []
    n = wal.recover(path, lambda pid, img: out.append((pid, img)))
    return n, out


# ---------------------------------------------------------------------------
class TestFraming:
    def test_committed_page_is_applied(self, wpath):
        build(wpath, ("page", 1, 7, b"A" * 4096), ("commit", 1))
        assert replay(wpath) == (1, [(7, b"A" * 4096)])

    def test_uncommitted_page_is_not_applied(self, wpath):
        build(wpath, ("page", 1, 7, b"X" * 100))
        assert replay(wpath) == (0, [])

    def test_pages_replay_in_log_order_with_duplicates(self, wpath):
        build(
            wpath,
            ("page", 1, 3, b"a" * 10),
            ("page", 1, 9, b"b" * 10),
            ("page", 1, 3, b"c" * 10),
            ("commit", 1),
        )
        _, out = replay(wpath)
        assert out == [(3, b"a" * 10), (9, b"b" * 10), (3, b"c" * 10)]

    def test_only_committed_txn_is_applied(self, wpath):
        build(
            wpath,
            ("page", 1, 1, b"one"),
            ("commit", 1),
            ("page", 2, 2, b"two"),  # txn 2 never commits
        )
        assert replay(wpath) == (1, [(1, b"one")])

    def test_commit_can_follow_a_later_txns_pages(self, wpath):
        build(
            wpath,
            ("page", 1, 1, b"p1"),
            ("page", 2, 2, b"p2"),
            ("commit", 2),
            ("commit", 1),
        )
        n, out = replay(wpath)
        assert n == 2 and sorted(out) == [(1, b"p1"), (2, b"p2")]

    def test_full_page_every_byte_value_round_trips(self, wpath):
        img = (bytes(range(256)) * 16)[:4096]
        build(wpath, ("page", 1, 5, img), ("commit", 1))
        assert replay(wpath) == (1, [(5, img)])

    def test_empty_file(self, wpath):
        open(wpath, "wb").close()
        assert replay(wpath) == (0, [])

    def test_recover_returns_the_applied_count(self, wpath):
        build(
            wpath,
            ("page", 1, 1, b"x"),
            ("page", 1, 2, b"y"),
            ("commit", 1),
            ("page", 2, 3, b"z"),
            ("commit", 2),
        )
        n, _ = replay(wpath)
        assert n == 3


# ---------------------------------------------------------------------------
class TestTornTail:
    def test_garbage_appended_after_a_good_txn(self, wpath):
        build(wpath, ("page", 1, 1, b"good"), ("commit", 1))
        with open(wpath, "ab") as f:
            f.write(b"\x00\x01\x02\x03not-a-frame-at-all")
        assert replay(wpath) == (1, [(1, b"good")])

    def test_truncated_final_record(self, wpath):
        build(
            wpath,
            ("page", 1, 1, b"first"),
            ("commit", 1),
            ("page", 2, 2, b"second"),
            ("commit", 2),  # this record gets chopped
        )
        with open(wpath, "r+b") as f:
            f.truncate(os.path.getsize(wpath) - 3)
        assert replay(wpath) == (1, [(1, b"first")])  # txn 2 never fully commits

    def test_corrupt_payload_crc_stops_replay_there(self, wpath):
        build(
            wpath,
            ("page", 1, 1, b"aaaa"),
            ("commit", 1),
            ("page", 1, 2, b"bbbb"),
            ("commit", 1),
        )
        with open(wpath, "r+b") as f:
            f.seek(6)  # inside the first PAGE record's payload
            b = f.read(1)
            f.seek(6)
            f.write(bytes([b[0] ^ 0xFF]))
        assert replay(wpath) == (0, [])  # torn at record 0

    def test_corrupt_commit_crc_drops_that_txn(self, wpath):
        build(wpath, ("page", 1, 1, b"zzzz"), ("commit", 1))
        sz = os.path.getsize(wpath)
        with open(wpath, "r+b") as f:
            f.seek(sz - 1)  # last byte is part of the COMMIT's CRC
            b = f.read(1)
            f.seek(sz - 1)
            f.write(bytes([b[0] ^ 0xFF]))
        assert replay(wpath) == (0, [])  # PAGE seen, COMMIT rejected

    def test_bogus_length_prefix(self, wpath):
        with open(wpath, "wb") as f:
            f.write(struct.pack("<I", 10_000_000) + b"\x01" + b"short")
        assert replay(wpath) == (0, [])

    def test_good_txn_then_torn_second_txn(self, wpath):
        build(
            wpath,
            ("page", 1, 1, b"keep"),
            ("commit", 1),
            ("page", 2, 2, b"lost"),
            ("commit", 2),
        )
        # frames: PAGE(1,1,"keep")=25B, COMMIT(1)=17B -> the 2nd PAGE record
        # (txn 2) starts at offset 42. Corrupt a byte in its payload so its CRC
        # fails; recovery then stops before txn 2's COMMIT is ever seen.
        with open(wpath, "rb") as f:
            data = bytearray(f.read())
        data[42 + 8] ^= 0xFF
        with open(wpath, "wb") as f:
            f.write(bytes(data))
        assert replay(wpath) == (1, [(1, b"keep")])


# ---------------------------------------------------------------------------
class TestTruncate:
    def test_truncate_empties_and_drops_records(self, wpath):
        w = WriteAheadLog(wpath)
        w.append_page(1, 1, b"before")
        w.append_commit(1)
        w.fsync()
        w.truncate()
        assert os.path.getsize(wpath) == 0
        w.append_page(2, 2, b"after")
        w.append_commit(2)
        w.fsync()
        w.close()
        assert replay(wpath) == (1, [(2, b"after")])
