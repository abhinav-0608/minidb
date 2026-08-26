"""Stage 2 tests - schemas and the row <-> bytes codec.

* TestSchema        - construction and validation
* TestRoundTrip     - encode then decode preserves the row, across edge cases
* TestEncodeRejects - malformed rows are refused
* TestDecodeRejects - corrupt / truncated bytes are refused
* TestNullBitmap    - decode honours NULL bits even though encode cannot set them
"""

from __future__ import annotations

import random
import struct

import pytest
from dataclasses import FrozenInstanceError

from minidb.constants import INT_MAX, INT_MIN, MAX_TEXT_BYTES
from minidb.errors import MiniDBError
from minidb.record import Column, ColumnType, Schema

USERS = Schema(
    (
        Column("id", ColumnType.INT),
        Column("name", ColumnType.TEXT),
        Column("age", ColumnType.INT),
    )
)


# ---------------------------------------------------------------------------
class TestSchema:
    def test_accepts_a_list_and_normalises_to_a_tuple(self):
        s = Schema([Column("a", ColumnType.INT)])
        assert isinstance(s.columns, tuple)

    def test_is_frozen(self):
        with pytest.raises(FrozenInstanceError):
            USERS.columns = ()

    def test_empty_column_list_rejected(self):
        with pytest.raises(MiniDBError):
            Schema(())

    def test_duplicate_column_names_rejected(self):
        with pytest.raises(MiniDBError):
            Schema((Column("x", ColumnType.INT), Column("x", ColumnType.TEXT)))

    def test_non_columntype_type_rejected(self):
        with pytest.raises(MiniDBError):
            Schema((Column("x", "INT"),))

    def test_column_names(self):
        assert USERS.column_names == ["id", "name", "age"]

    def test_column_index(self):
        assert USERS.column_index("name") == 1

    def test_column_index_missing_raises(self):
        with pytest.raises(MiniDBError):
            USERS.column_index("nope")


# ---------------------------------------------------------------------------
class TestRoundTrip:
    def _rt(self, schema, row):
        assert schema.decode_row(schema.encode_row(row)) == row

    def test_basic(self):
        self._rt(USERS, (1, "Alice", 24))

    def test_known_byte_layout(self):
        # 1 bitmap byte + 8 (id) + 2+5 (name) + 8 (age) = 24
        b = USERS.encode_row((1, "Alice", 24))
        assert b == bytes.fromhex("00" "0100000000000000" "0500" "416c696365" "1800000000000000")

    def test_empty_string(self):
        self._rt(USERS, (7, "", 0))

    def test_negative_and_boundary_ints(self):
        for v in (-1, 0, INT_MIN, INT_MAX, -(2**40)):
            self._rt(USERS, (v, "x", v))

    def test_text_length_limit_is_bytes_not_chars(self):
        # "e" with acute accent is 2 UTF-8 bytes
        self._rt(USERS, (1, "é" * (MAX_TEXT_BYTES // 2), 1))  # exactly 1024 bytes
        with pytest.raises(MiniDBError):
            USERS.encode_row((1, "é" * (MAX_TEXT_BYTES // 2 + 1), 1))  # 1026

    def test_text_at_exactly_the_limit(self):
        self._rt(USERS, (1, "a" * MAX_TEXT_BYTES, 1))

    def test_unicode(self):
        self._rt(USERS, (1, "café – \U0001f600", 2))

    def test_single_column_schema(self):
        s = Schema((Column("only", ColumnType.TEXT),))
        self._rt(s, ("hello",))

    def test_null_bitmap_spans_multiple_bytes(self):
        s = Schema(tuple(Column(f"c{i}", ColumnType.INT) for i in range(20)))
        self._rt(s, tuple(range(20)))

    def test_accepts_a_list_of_values(self):
        assert USERS.decode_row(USERS.encode_row([1, "a", 2])) == (1, "a", 2)

    def test_types_are_preserved(self):
        row = USERS.decode_row(USERS.encode_row((1, "a", 2)))
        assert isinstance(row[0], int) and isinstance(row[1], str)

    def test_randomised_round_trip(self):
        rng = random.Random(99)
        s = Schema(
            (
                Column("a", ColumnType.INT),
                Column("b", ColumnType.TEXT),
                Column("c", ColumnType.INT),
                Column("d", ColumnType.TEXT),
            )
        )
        for _ in range(500):
            row = (
                rng.randint(INT_MIN, INT_MAX),
                "".join(chr(rng.randint(32, 0xD7FF)) for _ in range(rng.randint(0, 40))),
                rng.randint(-1000, 1000),
                rng.randbytes(rng.randint(0, 30)).decode("latin-1"),
            )
            assert s.decode_row(s.encode_row(row)) == row


# ---------------------------------------------------------------------------
class TestEncodeRejects:
    @pytest.mark.parametrize(
        "row",
        [
            (1, "Alice"),  # too few
            (1, "Alice", 24, 9),  # too many
            ("1", "Alice", 24),  # str in INT
            (1.0, "Alice", 24),  # float in INT
            (True, "Alice", 24),  # bool in INT
            (1, 5, 24),  # int in TEXT
            (1, b"bytes", 24),  # bytes in TEXT
            (1, "Alice", None),  # NULL
            (2**63, "Alice", 24),  # INT_MAX + 1
            (-(2**63) - 1, "Alice", 24),  # INT_MIN - 1
            (1, "z" * (MAX_TEXT_BYTES + 1), 24),  # TEXT too long
        ],
    )
    def test_rejects(self, row):
        with pytest.raises(MiniDBError):
            USERS.encode_row(row)


# ---------------------------------------------------------------------------
class TestDecodeRejects:
    def test_missing_bitmap(self):
        with pytest.raises(MiniDBError):
            USERS.decode_row(b"")

    def test_truncated_int(self):
        good = USERS.encode_row((1, "Alice", 24))
        with pytest.raises(MiniDBError):
            USERS.decode_row(good[:-1])  # last INT byte gone

    def test_truncated_text_length(self):
        s = Schema((Column("t", ColumnType.TEXT),))
        with pytest.raises(MiniDBError):
            s.decode_row(b"\x00\x05")  # bitmap ok, only 1 of 2 length bytes

    def test_truncated_text_body(self):
        s = Schema((Column("t", ColumnType.TEXT),))
        with pytest.raises(MiniDBError):
            s.decode_row(b"\x00" + struct.pack("<H", 10) + b"abc")  # claims 10, has 3

    def test_trailing_bytes(self):
        good = USERS.encode_row((1, "Alice", 24))
        with pytest.raises(MiniDBError):
            USERS.decode_row(good + b"\x00")


# ---------------------------------------------------------------------------
class TestNullBitmap:
    def test_decode_honours_a_set_null_bit(self):
        # bit 1 set => the "name" column is NULL and contributes no bytes
        data = bytes([0b010]) + (1).to_bytes(8, "little") + (24).to_bytes(8, "little")
        assert USERS.decode_row(data) == (1, None, 24)

    def test_decode_honours_null_bits_in_the_second_bitmap_byte(self):
        s = Schema(tuple(Column(f"c{i}", ColumnType.INT) for i in range(10)))
        # columns 8 and 9 NULL (second bitmap byte = 0b11), rest present
        bitmap = bytes([0x00, 0b11])
        body = b"".join((i).to_bytes(8, "little") for i in range(8))
        assert s.decode_row(bitmap + body) == (0, 1, 2, 3, 4, 5, 6, 7, None, None)

    def test_encode_still_refuses_to_make_a_null(self):
        with pytest.raises(MiniDBError):
            USERS.encode_row((1, None, 24))
