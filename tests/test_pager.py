"""Stage 1 tests - the pager.

Grouped by the four things Stage 1 has to get right:

* TestCreateAndReopen  - file creation, header, page_size, reopening
* TestPersistence      - written pages read back exactly after a restart
* TestPageBoundaries   - page-size validation, no bleed between pages, offsets
* TestInvalidPageIds   - reserved / out-of-range / wrong-type ids rejected
* TestLifecycleAndLock - advisory lock, close(), context manager
* TestCorruptionDetection - bad magic / version / truncated file caught on open

Every test writes to a pytest ``tmp_path`` so nothing touches the real repo.
"""

from __future__ import annotations

import random
import struct

import pytest

from minidb.constants import DEFAULT_PAGE_SIZE, FORMAT_VERSION, MAGIC
from minidb.pager import MiniDBError, Pager


def full(pager: Pager, byte: int) -> bytes:
    """A whole page filled with one byte value."""
    return bytes([byte]) * pager.page_size


def make_db(path, *, allocate: int = 0, page_size: int = DEFAULT_PAGE_SIZE):
    """Create a database, optionally pre-allocate some pages, close it."""
    with Pager(str(path), page_size=page_size) as p:
        for _ in range(allocate):
            p.allocate_page()
    return path


# ---------------------------------------------------------------------------
class TestCreateAndReopen:
    def test_opening_missing_path_creates_a_valid_file(self, tmp_path):
        path = tmp_path / "t.db"
        assert not path.exists()
        with Pager(str(path)) as p:
            assert p.page_size == DEFAULT_PAGE_SIZE
            assert p.page_count == 1  # just the header page
        assert path.exists()

    def test_fresh_file_on_disk_has_the_expected_header(self, tmp_path):
        path = tmp_path / "t.db"
        Pager(str(path)).close()
        raw = path.read_bytes()
        assert len(raw) == DEFAULT_PAGE_SIZE
        magic, version, page_size, page_count = struct.unpack("<8sIIQ", raw[:24])
        assert magic == MAGIC
        assert version == FORMAT_VERSION
        assert page_size == DEFAULT_PAGE_SIZE
        assert page_count == 1
        assert raw[24:] == b"\x00" * (DEFAULT_PAGE_SIZE - 24)  # reserved is zero

    def test_empty_existing_file_is_initialised(self, tmp_path):
        path = tmp_path / "t.db"
        path.write_bytes(b"")  # exists, zero length
        with Pager(str(path)) as p:
            assert p.page_count == 1
            assert p.page_size == DEFAULT_PAGE_SIZE

    @pytest.mark.parametrize("ps", [512, 1024, 8192])
    def test_custom_page_size_is_persisted(self, tmp_path, ps):
        path = tmp_path / "t.db"
        with Pager(str(path), page_size=ps) as p:
            assert p.page_size == ps
            pid = p.allocate_page()
            p.write_page(pid, full(p, 0x01))
        with Pager(str(path)) as p:  # reopen with the default arg
            assert p.page_size == ps
            assert p.read_page(1) == bytes([0x01]) * ps

    @pytest.mark.parametrize("ps", [0, 1, 511, (1 << 20) + 1, -4096])
    def test_implausible_page_size_rejected_on_create(self, tmp_path, ps):
        with pytest.raises(MiniDBError):
            Pager(str(tmp_path / "t.db"), page_size=ps)

    def test_reopen_with_conflicting_page_size_raises(self, tmp_path):
        path = tmp_path / "t.db"
        Pager(str(path), page_size=1024).close()
        with pytest.raises(MiniDBError):
            Pager(str(path), page_size=2048)

    def test_reopen_with_default_arg_accepts_any_page_size(self, tmp_path):
        path = tmp_path / "t.db"
        Pager(str(path), page_size=1024).close()
        Pager(str(path)).close()  # default arg means "don't care"

    def test_page_count_persists_across_restart(self, tmp_path):
        path = tmp_path / "t.db"
        with Pager(str(path)) as p:
            for _ in range(10):
                p.allocate_page()
        with Pager(str(path)) as p:
            assert p.page_count == 11
            assert p.read_page(10) == full(p, 0x00)


# ---------------------------------------------------------------------------
class TestPersistence:
    def test_written_page_survives_restart(self, tmp_path):
        path = tmp_path / "t.db"
        with Pager(str(path)) as p:
            for _ in range(3):
                p.allocate_page()
            payload = random.Random(7).randbytes(p.page_size)
            p.write_page(3, payload)
        with Pager(str(path)) as p:
            assert p.read_page(3) == payload
            assert p.read_page(1) == full(p, 0x00)  # untouched pages still zero
            assert p.read_page(2) == full(p, 0x00)

    def test_read_your_writes_before_flush(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as p:
            pid = p.allocate_page()
            p.write_page(pid, full(p, 0x7F))
            assert p.read_page(pid) == full(p, 0x7F)  # served from the cache

    def test_write_is_not_on_disk_until_flush(self, tmp_path):
        path = tmp_path / "t.db"
        p = Pager(str(path))
        try:
            pid = p.allocate_page()
            p.write_page(pid, full(p, 0xFF))
            assert len(path.read_bytes()) == p.page_size  # header only so far
            p.flush()
            raw = path.read_bytes()
            assert len(raw) == 2 * p.page_size
            assert raw[p.page_size : 2 * p.page_size] == full(p, 0xFF)
        finally:
            p.close()

    def test_overwrite_replaces_bytes(self, tmp_path):
        path = tmp_path / "t.db"
        with Pager(str(path)) as p:
            pid = p.allocate_page()
            p.write_page(pid, full(p, 0x01))
            p.write_page(pid, full(p, 0x02))
        with Pager(str(path)) as p:
            assert p.read_page(1) == full(p, 0x02)

    def test_all_byte_values_round_trip(self, tmp_path):
        path = tmp_path / "t.db"
        with Pager(str(path)) as p:
            ps = p.page_size
            page = (bytes(range(256)) * (ps // 256 + 1))[:ps]
            pid = p.allocate_page()
            p.write_page(pid, page)
        with Pager(str(path)) as p:
            assert p.read_page(1) == page

    def test_read_allocated_but_unwritten_page_is_zeros(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as p:
            pid = p.allocate_page()
            assert p.read_page(pid) == full(p, 0x00)

    def test_read_page_returns_immutable_copy(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as p:
            pid = p.allocate_page()
            p.write_page(pid, full(p, 0xAA))
            got = p.read_page(pid)
            assert isinstance(got, bytes)
            with pytest.raises(TypeError):
                got[0] = 1  # bytes cannot be mutated
            assert p.read_page(pid) == full(p, 0xAA)  # cache untouched

    def test_write_page_copies_the_caller_buffer(self, tmp_path):
        path = tmp_path / "t.db"
        with Pager(str(path)) as p:
            pid = p.allocate_page()
            buf = bytearray(full(p, 0x05))
            p.write_page(pid, buf)
            buf[0] = 0x99  # mutate after handing off
        with Pager(str(path)) as p:
            assert p.read_page(1)[0] == 0x05

    def test_context_manager_flushes_on_normal_exit(self, tmp_path):
        path = tmp_path / "t.db"
        with Pager(str(path)) as p:
            pid = p.allocate_page()
            p.write_page(pid, full(p, 0x42))
        with Pager(str(path)) as p:
            assert p.read_page(1) == full(p, 0x42)

    def test_randomised_round_trip_with_overwrites(self, tmp_path):
        path = tmp_path / "t.db"
        rng = random.Random(20240501)
        expected: dict[int, bytes] = {}
        n = 40
        with Pager(str(path)) as p:
            for _ in range(n):
                pid = p.allocate_page()
                data = rng.randbytes(p.page_size)
                expected[pid] = data
                p.write_page(pid, data)
            for pid in rng.sample(sorted(expected), 15):  # overwrite a subset
                data = rng.randbytes(p.page_size)
                expected[pid] = data
                p.write_page(pid, data)
        with Pager(str(path)) as p:
            assert p.page_count == n + 1
            for pid, data in expected.items():
                assert p.read_page(pid) == data


# ---------------------------------------------------------------------------
class TestPageBoundaries:
    @pytest.mark.parametrize("delta", [-1, 1, -100, 100])
    def test_write_page_rejects_wrong_size(self, tmp_path, delta):
        with Pager(str(tmp_path / "t.db")) as p:
            pid = p.allocate_page()
            with pytest.raises(MiniDBError):
                p.write_page(pid, b"\x00" * (p.page_size + delta))

    def test_write_page_accepts_exact_size(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as p:
            pid = p.allocate_page()
            p.write_page(pid, full(p, 0x00))  # no raise

    def test_pages_do_not_bleed_into_neighbours(self, tmp_path):
        path = tmp_path / "t.db"
        with Pager(str(path)) as p:
            for _ in range(3):
                p.allocate_page()
            p.write_page(1, full(p, 0x11))
            p.write_page(2, full(p, 0x22))
            p.write_page(3, full(p, 0x33))
        with Pager(str(path)) as p:
            assert p.read_page(1) == full(p, 0x11)
            assert p.read_page(2) == full(p, 0x22)
            assert p.read_page(3) == full(p, 0x33)

    def test_page_lands_at_the_right_file_offset(self, tmp_path):
        path = tmp_path / "t.db"
        ps = DEFAULT_PAGE_SIZE
        with Pager(str(path)) as p:
            for _ in range(5):
                p.allocate_page()
            p.write_page(4, full(p, 0xAB))
        raw = path.read_bytes()
        assert raw[4 * ps : 5 * ps] == bytes([0xAB]) * ps
        assert raw[3 * ps : 4 * ps] == b"\x00" * ps  # page 3 untouched

    def test_high_page_id_offset_arithmetic(self, tmp_path):
        path = tmp_path / "t.db"
        ps = DEFAULT_PAGE_SIZE
        with Pager(str(path)) as p:
            for _ in range(64):
                p.allocate_page()
            p.write_page(64, full(p, 0x5A))
        with Pager(str(path)) as p:
            assert p.read_page(64) == full(p, 0x5A)
            assert p.read_page(32) == full(p, 0x00)
        assert path.read_bytes()[64 * ps : 65 * ps] == bytes([0x5A]) * ps

    def test_allocate_returns_sequential_ids_from_one(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as p:
            assert [p.allocate_page() for _ in range(4)] == [1, 2, 3, 4]
            assert p.page_count == 5


# ---------------------------------------------------------------------------
class TestInvalidPageIds:
    @pytest.mark.parametrize("bad", [0, -1, 1, True, 1.0, "1", None])
    def test_read_page_rejects_bad_ids(self, tmp_path, bad):
        # fresh db: only page 0 exists, so id 1 is out of range and the rest
        # are reserved or the wrong type
        with Pager(str(tmp_path / "t.db")) as p:
            with pytest.raises(MiniDBError):
                p.read_page(bad)

    @pytest.mark.parametrize("bad", [0, -1, 2, True, 1.0, "1", None])
    def test_write_page_rejects_bad_ids(self, tmp_path, bad):
        with Pager(str(tmp_path / "t.db")) as p:
            p.allocate_page()  # now page 1 is valid, page 2 is not
            with pytest.raises(MiniDBError):
                p.write_page(bad, full(p, 0x00))

    def test_reading_one_past_the_last_page_raises(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as p:
            p.allocate_page()
            p.allocate_page()
            p.read_page(2)  # valid
            with pytest.raises(MiniDBError):
                p.read_page(3)


# ---------------------------------------------------------------------------
class TestLifecycleAndLock:
    def test_second_open_is_rejected_while_first_is_open(self, tmp_path):
        path = tmp_path / "t.db"
        p1 = Pager(str(path))
        try:
            with pytest.raises(MiniDBError):
                Pager(str(path))
        finally:
            p1.close()

    def test_lock_is_released_after_close(self, tmp_path):
        path = tmp_path / "t.db"
        Pager(str(path)).close()
        p2 = Pager(str(path))  # must succeed
        p2.close()

    def test_close_is_idempotent(self, tmp_path):
        p = Pager(str(tmp_path / "t.db"))
        p.close()
        p.close()  # no raise

    @pytest.mark.parametrize(
        "op",
        [
            lambda p: p.read_page(1),
            lambda p: p.write_page(1, b"\x00"),
            lambda p: p.allocate_page(),
            lambda p: p.flush(),
        ],
    )
    def test_closed_pager_rejects_operations(self, tmp_path, op):
        p = Pager(str(tmp_path / "t.db"))
        p.close()
        with pytest.raises(MiniDBError):
            op(p)

    def test_context_manager_closes_on_exception(self, tmp_path):
        path = tmp_path / "t.db"

        class Boom(Exception):
            pass

        with pytest.raises(Boom):
            with Pager(str(path)) as p:
                p.allocate_page()
                raise Boom()
        # lock released and the allocation was flushed on the way out
        with Pager(str(path)) as p:
            assert p.page_count == 2


# ---------------------------------------------------------------------------
class TestCorruptionDetection:
    def test_bad_magic_detected(self, tmp_path):
        path = make_db(tmp_path / "t.db")
        with open(path, "r+b") as f:
            f.write(b"XXXXXXXX")
        with pytest.raises(MiniDBError):
            Pager(str(path))

    def test_wrong_format_version_detected(self, tmp_path):
        path = make_db(tmp_path / "t.db")
        with open(path, "r+b") as f:
            f.seek(8)
            f.write(struct.pack("<I", 999))
        with pytest.raises(MiniDBError):
            Pager(str(path))

    def test_size_not_a_multiple_of_page_size_detected(self, tmp_path):
        path = make_db(tmp_path / "t.db", allocate=2)
        with open(path, "r+b") as f:
            f.truncate(3 * DEFAULT_PAGE_SIZE - 7)
        with pytest.raises(MiniDBError):
            Pager(str(path))

    def test_header_page_count_disagreeing_with_file_detected(self, tmp_path):
        path = make_db(tmp_path / "t.db", allocate=3)  # 4 pages, header says 4
        with open(path, "r+b") as f:
            f.truncate(3 * DEFAULT_PAGE_SIZE)  # 3 pages now, header still says 4
        with pytest.raises(MiniDBError):
            Pager(str(path))

    def test_file_smaller_than_header_detected(self, tmp_path):
        path = make_db(tmp_path / "t.db")
        with open(path, "r+b") as f:
            f.truncate(10)
        with pytest.raises(MiniDBError):
            Pager(str(path))
