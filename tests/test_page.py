"""Stage 2 tests - the slotted page.

* TestInitEmpty       - a fresh page's header and free space
* TestRecords         - add / read / iterate, and the exact byte layout
* TestPageFull        - can_fit and the boundary where add_record refuses
* TestPageTypeGuard   - loading the wrong kind of page fails loudly
* TestHeaderRoundTrip - state survives a .buffer -> re-wrap cycle (the pager path)
"""

from __future__ import annotations

import struct

import pytest

from minidb.errors import MiniDBError
from minidb.page import PageType, SlottedPage

PS = 512
H = SlottedPage.HEADER_SIZE
SLOT = SlottedPage.SLOT_SIZE


def wrap(page: SlottedPage, *, expected_type=PageType.HEAP) -> SlottedPage:
    """Simulate a pager round-trip: serialise and parse again."""
    return SlottedPage(bytearray(page.buffer), expected_type=expected_type)


# ---------------------------------------------------------------------------
class TestInitEmpty:
    def test_fresh_page_state(self):
        p = SlottedPage.init_empty(PS)
        assert p.num_slots == 0
        assert p.next_page_id == 0
        assert p.page_type == PageType.HEAP
        assert p.free_space == PS - H
        assert len(p.buffer) == PS

    def test_requires_a_bytearray(self):
        with pytest.raises(MiniDBError):
            SlottedPage(bytes(PS))  # immutable


# ---------------------------------------------------------------------------
class TestRecords:
    def test_add_then_read(self):
        p = SlottedPage.init_empty(PS)
        assert p.add_record(b"ABCD") == 0
        assert p.read_record(0) == b"ABCD"

    def test_slot_ids_increment(self):
        p = SlottedPage.init_empty(PS)
        assert [p.add_record(bytes([i]) * 3) for i in range(5)] == [0, 1, 2, 3, 4]
        assert p.num_slots == 5

    def test_iter_yields_all_records_in_order(self):
        p = SlottedPage.init_empty(PS)
        payloads = [b"one", b"two", b"three"]
        for x in payloads:
            p.add_record(x)
        assert list(p) == payloads

    def test_records_grow_from_the_end_slots_from_the_header(self):
        p = SlottedPage.init_empty(PS)
        p.add_record(b"WXYZ")
        buf = p.buffer
        assert buf[PS - 4 :] == b"WXYZ"  # record sits at the very end
        offset, length = struct.unpack("<HH", buf[H : H + SLOT])  # first slot entry
        assert (offset, length) == (PS - 4, 4)

    def test_free_space_shrinks_by_record_plus_slot(self):
        p = SlottedPage.init_empty(PS)
        before = p.free_space
        p.add_record(b"x" * 40)
        assert p.free_space == before - (40 + SLOT)

    def test_empty_record_is_allowed(self):
        p = SlottedPage.init_empty(PS)
        sid = p.add_record(b"")
        assert p.read_record(sid) == b""

    def test_read_record_on_empty_page_raises(self):
        p = SlottedPage.init_empty(PS)
        with pytest.raises(MiniDBError):
            p.read_record(0)

    @pytest.mark.parametrize("bad", [-1, 3, 99])
    def test_read_record_out_of_range(self, bad):
        p = SlottedPage.init_empty(PS)
        for _ in range(3):
            p.add_record(b"rec")  # slots 0, 1, 2 are valid
        with pytest.raises(MiniDBError):
            p.read_record(bad)


# ---------------------------------------------------------------------------
class TestPageFull:
    def test_can_fit_boundary(self):
        p = SlottedPage.init_empty(PS)
        biggest = PS - H - SLOT  # record that leaves exactly zero free
        assert p.can_fit(biggest) is True
        assert p.can_fit(biggest + 1) is False

    def test_fill_exactly_then_refuse(self):
        p = SlottedPage.init_empty(PS)
        p.add_record(b"x" * (PS - H - SLOT))
        assert p.free_space == 0
        with pytest.raises(MiniDBError):
            p.add_record(b"")  # even a 0-byte record needs 4 bytes of slot

    def test_add_record_refuses_when_too_big(self):
        p = SlottedPage.init_empty(PS)
        p.add_record(b"x" * 100)
        with pytest.raises(MiniDBError):
            p.add_record(b"y" * (PS - H - SLOT))  # would need more than remains

    def test_many_small_records_until_full(self):
        p = SlottedPage.init_empty(PS)
        n = 0
        while p.can_fit(4):
            p.add_record(b"abcd")
            n += 1
        with pytest.raises(MiniDBError):
            p.add_record(b"abcd")
        assert n == (PS - H) // (4 + SLOT)
        assert list(p) == [b"abcd"] * n


# ---------------------------------------------------------------------------
class TestPageTypeGuard:
    def test_all_zero_buffer_is_not_a_heap_page(self):
        with pytest.raises(MiniDBError):
            SlottedPage(bytearray(PS))  # page_type byte is 0, not HEAP

    def test_expected_type_none_skips_the_check(self):
        SlottedPage(bytearray(PS), expected_type=None)  # no raise

    def test_page_with_a_different_type_byte_raises(self):
        buf = bytearray(SlottedPage.init_empty(PS).buffer)
        buf[0] = 2  # pretend a later stage wrote a CATALOG page here
        with pytest.raises(MiniDBError):
            SlottedPage(buf)  # default expected_type is HEAP


# ---------------------------------------------------------------------------
class TestHeaderRoundTrip:
    def test_records_and_slots_survive_reparse(self):
        p = SlottedPage.init_empty(PS)
        for x in (b"alpha", b"beta", b"gamma"):
            p.add_record(x)
        again = wrap(p)
        assert again.num_slots == 3
        assert list(again) == [b"alpha", b"beta", b"gamma"]
        assert again.free_space == p.free_space

    def test_next_page_id_persists(self):
        p = SlottedPage.init_empty(PS)
        p.next_page_id = 42
        assert wrap(p).next_page_id == 42

    @pytest.mark.parametrize("bad", [-1, 1 << 32])
    def test_next_page_id_out_of_range(self, bad):
        p = SlottedPage.init_empty(PS)
        with pytest.raises(MiniDBError):
            p.next_page_id = bad
