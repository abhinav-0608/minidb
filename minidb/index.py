"""Stage 6 - the persistent B+ tree index.

The Stage 5 algorithm, moved onto disk pages: one node per page, keys packed
in a fixed-width array. Routing, leaf/internal splits, and the copy-up /
move-up rules are unchanged - only "follow a Python reference" becomes "read
a page id".

Page layouts (little-endian, one node per page)::

    LEAF (PageType.BTREE_LEAF)              INTERNAL (PageType.BTREE_INTERNAL)
    header 8B: type, num_keys, next_leaf    header 8B: type, num_keys, (unused)
    keys[]  : num_keys x 8B (signed)        keys[]    : num_keys x 8B (signed)
    rids[]  : num_keys x 6B                 children[]: (num_keys+1) x 4B
              (page_id u32, slot_id u16)                (child page id u32)

    leaf capacity     = (page_size - 8) // 14
    internal capacity = (page_size - 12) // 12

Fixed root: the root page id never changes. On a root split the old root's
contents are copied to a fresh page and the root page is rewritten in place
as the new internal node. So the catalog stores the root page id once.

Nodes are fully parsed into Python lists on read and fully re-serialised on
write. A production engine would binary-search directly in the page bytes;
we deserialise for clarity (see design doc section 12).

Keys are unique: inserting an existing key raises.
"""

from __future__ import annotations

import struct
from bisect import bisect_left, bisect_right
from collections.abc import Iterator

from .errors import MiniDBError
from .page import PageType

# type u8 | pad | num_keys u16 | extra u32   (leaf uses `extra` as next_leaf_id)
_HEADER = struct.Struct("<BxHI")
_KEY = struct.Struct("<q")  # signed 64-bit, matches the INT column
_RID = struct.Struct("<IH")  # page_id u32, slot_id u16
_CHILD = struct.Struct("<I")  # child page id u32

RID = tuple  # a (page_id, slot_id) pair


class _Leaf:
    __slots__ = ("keys", "rids", "next_leaf_id")

    def __init__(self, keys=None, rids=None, next_leaf_id=0) -> None:
        self.keys: list[int] = keys if keys is not None else []
        self.rids: list[tuple[int, int]] = rids if rids is not None else []
        self.next_leaf_id: int = next_leaf_id

    @classmethod
    def parse(cls, buf: bytes) -> "_Leaf":
        ptype, num, nxt = _HEADER.unpack_from(buf, 0)
        if ptype != PageType.BTREE_LEAF:
            raise MiniDBError(f"expected a BTREE_LEAF page, found type {ptype}")
        koff = _HEADER.size
        roff = koff + num * _KEY.size
        keys = [_KEY.unpack_from(buf, koff + i * _KEY.size)[0] for i in range(num)]
        rids = [_RID.unpack_from(buf, roff + i * _RID.size) for i in range(num)]
        return cls(keys, rids, nxt)

    def to_bytes(self, page_size: int) -> bytes:
        buf = bytearray(page_size)
        _HEADER.pack_into(
            buf, 0, int(PageType.BTREE_LEAF), len(self.keys), self.next_leaf_id
        )
        koff = _HEADER.size
        roff = koff + len(self.keys) * _KEY.size
        for i, k in enumerate(self.keys):
            _KEY.pack_into(buf, koff + i * _KEY.size, k)
        for i, (pid, sid) in enumerate(self.rids):
            _RID.pack_into(buf, roff + i * _RID.size, pid, sid)
        return bytes(buf)


class _Internal:
    __slots__ = ("keys", "children")

    def __init__(self, keys=None, children=None) -> None:
        self.keys: list[int] = keys if keys is not None else []
        self.children: list[int] = children if children is not None else []

    @classmethod
    def parse(cls, buf: bytes) -> "_Internal":
        ptype, num, _ = _HEADER.unpack_from(buf, 0)
        if ptype != PageType.BTREE_INTERNAL:
            raise MiniDBError(
                f"expected a BTREE_INTERNAL page, found type {ptype}"
            )
        koff = _HEADER.size
        coff = koff + num * _KEY.size
        keys = [_KEY.unpack_from(buf, koff + i * _KEY.size)[0] for i in range(num)]
        children = [
            _CHILD.unpack_from(buf, coff + i * _CHILD.size)[0]
            for i in range(num + 1)
        ]
        return cls(keys, children)

    def to_bytes(self, page_size: int) -> bytes:
        buf = bytearray(page_size)
        _HEADER.pack_into(buf, 0, int(PageType.BTREE_INTERNAL), len(self.keys), 0)
        koff = _HEADER.size
        coff = koff + len(self.keys) * _KEY.size
        for i, k in enumerate(self.keys):
            _KEY.pack_into(buf, koff + i * _KEY.size, k)
        for i, c in enumerate(self.children):
            _CHILD.pack_into(buf, coff + i * _CHILD.size, c)
        return bytes(buf)


class BTreeIndex:
    def __init__(
        self,
        pager,
        root_page_id: int,
        *,
        leaf_capacity: int | None = None,
        internal_capacity: int | None = None,
    ) -> None:
        self._pager = pager
        self._root_page_id = root_page_id
        ps = pager.page_size
        self._leaf_cap = (
            leaf_capacity
            if leaf_capacity is not None
            else (ps - _HEADER.size) // (_KEY.size + _RID.size)
        )
        self._internal_cap = (
            internal_capacity
            if internal_capacity is not None
            else (ps - _HEADER.size - _CHILD.size) // (_KEY.size + _CHILD.size)
        )
        if self._leaf_cap < 2 or self._internal_cap < 2:
            raise MiniDBError("page too small to hold a B+ tree node")

    @classmethod
    def create(cls, pager, **caps) -> "BTreeIndex":
        root_id = pager.allocate_page()
        idx = cls(pager, root_id, **caps)
        pager.write_page(root_id, _Leaf().to_bytes(pager.page_size))
        return idx

    def __repr__(self) -> str:
        return (
            f"<BTreeIndex root={self._root_page_id} "
            f"leaf_cap={self._leaf_cap} internal_cap={self._internal_cap} "
            f"height={self.height()}>"
        )

    @property
    def root_page_id(self) -> int:
        return self._root_page_id

    @property
    def leaf_capacity(self) -> int:
        return self._leaf_cap

    @property
    def internal_capacity(self) -> int:
        return self._internal_cap

    # -- read ------------------------------------------------------------

    def search(self, key: int):
        node = self._read(self._root_page_id)
        while isinstance(node, _Internal):
            node = self._read(node.children[bisect_right(node.keys, key)])
        i = bisect_left(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            return node.rids[i]
        return None

    def __contains__(self, key: int) -> bool:
        return self.search(key) is not None

    def height(self) -> int:
        levels, node = 1, self._read(self._root_page_id)
        while isinstance(node, _Internal):
            levels += 1
            node = self._read(node.children[0])
        return levels

    def items(self) -> Iterator[tuple[int, tuple[int, int]]]:
        node = self._read(self._root_page_id)
        while isinstance(node, _Internal):
            node = self._read(node.children[0])
        while True:
            yield from zip(node.keys, node.rids)
            if node.next_leaf_id == 0:
                return
            node = self._read(node.next_leaf_id)

    # -- write ---------------------------------------------------------------

    def insert(self, key: int, rid: tuple[int, int]) -> None:
        split = self._insert(self._root_page_id, key, rid)
        if split is None:
            return
        sep, right_id = split
        # root split, fixed root: move the old root's contents to a new page,
        # then rebuild the root page in place as the new internal node.
        old_root = self._read(self._root_page_id)
        left_id = self._pager.allocate_page()
        self._write(left_id, old_root)
        self._write(
            self._root_page_id, _Internal(keys=[sep], children=[left_id, right_id])
        )

    def _insert(self, page_id: int, key: int, rid: tuple[int, int]):
        """Returns None, or ``(separator_key, new_right_page_id)`` on a split."""
        node = self._read(page_id)

        if isinstance(node, _Leaf):
            i = bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                raise MiniDBError(f"duplicate key: {key}")
            node.keys.insert(i, key)
            node.rids.insert(i, tuple(rid))
            if len(node.keys) > self._leaf_cap:
                return self._split_leaf(page_id, node)
            self._write(page_id, node)
            return None

        i = bisect_right(node.keys, key)
        child_split = self._insert(node.children[i], key, rid)
        if child_split is None:
            return None
        sep, right_id = child_split
        node.keys.insert(i, sep)
        node.children.insert(i + 1, right_id)
        if len(node.keys) > self._internal_cap:
            return self._split_internal(page_id, node)
        self._write(page_id, node)
        return None

    def _split_leaf(self, page_id: int, node: _Leaf):
        mid = (len(node.keys) + 1) // 2
        right_id = self._pager.allocate_page()
        right = _Leaf(node.keys[mid:], node.rids[mid:], node.next_leaf_id)
        node.keys = node.keys[:mid]
        node.rids = node.rids[:mid]
        node.next_leaf_id = right_id
        self._write(page_id, node)
        self._write(right_id, right)
        return (right.keys[0], right_id)  # separator COPIED up

    def _split_internal(self, page_id: int, node: _Internal):
        mid = len(node.keys) // 2
        up = node.keys[mid]  # MOVED up, not kept
        right_id = self._pager.allocate_page()
        right = _Internal(node.keys[mid + 1 :], node.children[mid + 1 :])
        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]
        self._write(page_id, node)
        self._write(right_id, right)
        return (up, right_id)

    # -- page access -------------------------------------------------------

    def _read(self, page_id: int):
        buf = self._pager.read_page(page_id)
        ptype = buf[0]
        if ptype == PageType.BTREE_LEAF:
            return _Leaf.parse(buf)
        if ptype == PageType.BTREE_INTERNAL:
            return _Internal.parse(buf)
        raise MiniDBError(
            f"page {page_id} is not a B+ tree node (page_type {ptype})"
        )

    def _write(self, page_id: int, node) -> None:
        self._pager.write_page(page_id, node.to_bytes(self._pager.page_size))
