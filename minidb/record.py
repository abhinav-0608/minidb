"""Stage 2 - schemas and the row <-> bytes codec.

A row is a Python ``tuple`` of values: ``int`` for an INT column, ``str`` for
a TEXT column, ``None`` for NULL (not accepted yet - see below). A schema is
an ordered list of typed columns and is the only thing needed to turn a row
into bytes or back.

On-disk row layout::

    [ null bitmap : ceil(ncols / 8) bytes, bit i set => column i is NULL ]
    [ for each column, in schema order, if not NULL: its value bytes ]
        INT   -> 8 bytes, signed, little-endian          (struct '<q')
        TEXT  -> 2-byte unsigned length + that many UTF-8 bytes

There is no length field inside the row; the slot that stores it carries the
length, and ``decode_row`` must consume the bytes exactly - anything left
over means corruption or a schema mismatch.

NULL is *designed in* (the bitmap is always written and always read) but not
*supported*: ``encode_row`` rejects ``None`` so we cannot create NULLs yet,
while ``decode_row`` honours a set bit so the format is ready and testable.

``Schema`` is a generic ordered list of typed columns. The rule that a
table's first column must be ``id INT`` (and unique) is enforced by
CREATE TABLE in Stage 3, not here.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass

from .constants import INT_MAX, INT_MIN, MAX_TEXT_BYTES
from .errors import MiniDBError

_INT = struct.Struct("<q")  # signed 64-bit little-endian
_TEXT_LEN = struct.Struct("<H")  # unsigned 16-bit little-endian


class ColumnType(enum.Enum):
    INT = "INT"
    TEXT = "TEXT"

    def __repr__(self) -> str:  # nicer test output
        return f"ColumnType.{self.name}"


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType


@dataclass(frozen=True)
class Schema:
    columns: tuple[Column, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        if not self.columns:
            raise MiniDBError("a schema needs at least one column")
        names = [c.name for c in self.columns]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise MiniDBError(f"duplicate column name(s): {', '.join(dupes)}")
        for c in self.columns:
            if not isinstance(c.type, ColumnType):
                raise MiniDBError(
                    f"column {c.name!r} has a non-ColumnType type {c.type!r}"
                )

    # -- shape queries ---------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column_index(self, name: str) -> int:
        for i, c in enumerate(self.columns):
            if c.name == name:
                return i
        raise MiniDBError(f"no column named {name!r} in this table")

    # -- codec ---------------------------------------------------------------

    def encode_row(self, values) -> bytes:
        values = tuple(values)
        if len(values) != len(self.columns):
            raise MiniDBError(
                f"table has {len(self.columns)} columns but "
                f"{len(values)} values were given"
            )

        null_bitmap = bytearray((len(self.columns) + 7) // 8)
        body: list[bytes] = []
        for i, (col, val) in enumerate(zip(self.columns, values)):
            if val is None:
                raise MiniDBError(
                    f"column {col.name!r}: NULL is not supported yet"
                )
            if col.type is ColumnType.INT:
                # bool is a subclass of int; reject it - it is almost always
                # a mistake to store True/False in an INT column.
                if type(val) is not int:
                    raise MiniDBError(
                        f"column {col.name!r} is INT but got a "
                        f"{type(val).__name__}"
                    )
                if not (INT_MIN <= val <= INT_MAX):
                    raise MiniDBError(
                        f"column {col.name!r}: {val} is out of range for a "
                        f"64-bit signed INT"
                    )
                body.append(_INT.pack(val))
            elif col.type is ColumnType.TEXT:
                if type(val) is not str:
                    raise MiniDBError(
                        f"column {col.name!r} is TEXT but got a "
                        f"{type(val).__name__}"
                    )
                raw = val.encode("utf-8")
                if len(raw) > MAX_TEXT_BYTES:
                    raise MiniDBError(
                        f"column {col.name!r}: TEXT is {len(raw)} bytes, the "
                        f"limit is {MAX_TEXT_BYTES}"
                    )
                body.append(_TEXT_LEN.pack(len(raw)))
                body.append(raw)
            else:  # pragma: no cover - guarded by Schema.__post_init__
                raise MiniDBError(f"column {col.name!r} has unknown type {col.type!r}")

        return bytes(null_bitmap) + b"".join(body)

    def decode_row(self, data: bytes) -> tuple:
        nbytes = (len(self.columns) + 7) // 8
        if len(data) < nbytes:
            raise MiniDBError("record is shorter than its null bitmap (corrupt)")
        null_bitmap = data[:nbytes]
        pos = nbytes
        values: list = []

        for i, col in enumerate(self.columns):
            if (null_bitmap[i // 8] >> (i % 8)) & 1:
                values.append(None)
                continue
            if col.type is ColumnType.INT:
                if pos + _INT.size > len(data):
                    raise MiniDBError(
                        f"record truncated reading INT column {col.name!r}"
                    )
                values.append(_INT.unpack_from(data, pos)[0])
                pos += _INT.size
            elif col.type is ColumnType.TEXT:
                if pos + _TEXT_LEN.size > len(data):
                    raise MiniDBError(
                        f"record truncated reading TEXT length for {col.name!r}"
                    )
                (n,) = _TEXT_LEN.unpack_from(data, pos)
                pos += _TEXT_LEN.size
                if pos + n > len(data):
                    raise MiniDBError(
                        f"record truncated reading TEXT body for {col.name!r}"
                    )
                values.append(data[pos : pos + n].decode("utf-8"))
                pos += n
            else:  # pragma: no cover
                raise MiniDBError(f"column {col.name!r} has unknown type {col.type!r}")

        if pos != len(data):
            raise MiniDBError(
                f"record has {len(data) - pos} trailing byte(s) "
                f"(corrupt or schema mismatch)"
            )
        return tuple(values)
