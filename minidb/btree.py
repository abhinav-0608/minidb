"""Stage 5 - an in-memory B+ tree.

A balanced search tree mapping integer keys to opaque values, kept entirely
in RAM. Stage 6 moves the nodes onto disk pages; the algorithms here do not
change.

Shape
-----
* Internal nodes hold only separator keys and child links - a routing index.
* Leaf nodes hold every (key, value) pair, in sorted order.
* Leaves are singly linked left-to-right, so an in-order walk never revisits
  an internal node.
* Every leaf is at the same depth. The tree grows only at the root, by
  splitting - that is the only thing that increases its height.

Routing rule: in an internal node, child ``i`` covers keys ``k`` with
``keys[i-1] <= k < keys[i]`` (the separator key itself routes right, because
it was copied up from a leaf and the real entry still lives there). So the
child for key ``k`` is ``bisect_right(node.keys, k)``.

Capacity: one ``order`` for both node kinds (>= 3). A node that would hold
more than ``order`` keys splits. Real B+ trees size leaf and internal nodes
separately from the page layout - Stage 6 does that.

Keys are unique: inserting an existing key raises.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterator

from .errors import MiniDBError


class _Leaf:
    __slots__ = ("keys", "values", "next")

    def __init__(self) -> None:
        self.keys: list[int] = []
        self.values: list = []
        self.next: _Leaf | None = None


class _Internal:
    __slots__ = ("keys", "children")

    def __init__(self) -> None:
        self.keys: list[int] = []
        self.children: list = []  # len == len(keys) + 1


class BPlusTree:
    def __init__(self, order: int = 64) -> None:
        if order < 3:
            raise MiniDBError("B+ tree order must be at least 3")
        self._order = order
        self._root: _Leaf | _Internal = _Leaf()
        self._size = 0

    def __repr__(self) -> str:
        return (
            f"<BPlusTree order={self._order} size={self._size} "
            f"height={self.height}>"
        )

    # -- read --------------------------------------------------------------

    @property
    def order(self) -> int:
        return self._order

    @property
    def height(self) -> int:
        """Number of levels: 1 for a lone leaf root, +1 per root split."""
        levels, node = 1, self._root
        while isinstance(node, _Internal):
            levels += 1
            node = node.children[0]
        return levels

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key) -> bool:
        leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        return i < len(leaf.keys) and leaf.keys[i] == key

    def search(self, key):
        """Return the value for ``key``, or ``None`` if the key is absent.

        Stage 6 stores (page_id, slot_id) tuples, never ``None``, so ``None``
        unambiguously means 'not found'.
        """
        leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return leaf.values[i]
        return None

    def items(self) -> Iterator[tuple[int, object]]:
        """Yield every (key, value) in ascending key order.

        Walks the linked leaf list - the traversal a range scan will use.
        """
        leaf = self._leftmost_leaf()
        while leaf is not None:
            yield from zip(leaf.keys, leaf.values)
            leaf = leaf.next

    def keys(self) -> list[int]:
        return [k for k, _ in self.items()]

    # -- write -----------------------------------------------------------

    def insert(self, key: int, value) -> None:
        split = self._insert(self._root, key, value)
        if split is not None:
            sep, right = split
            new_root = _Internal()
            new_root.keys = [sep]
            new_root.children = [self._root, right]
            self._root = new_root
        self._size += 1

    # -- descent -------------------------------------------------------------

    def _find_leaf(self, key) -> _Leaf:
        node = self._root
        while isinstance(node, _Internal):
            node = node.children[bisect_right(node.keys, key)]
        return node

    def _leftmost_leaf(self) -> _Leaf:
        node = self._root
        while isinstance(node, _Internal):
            node = node.children[0]
        return node

    def _insert(self, node, key, value):
        """Insert into the subtree rooted at ``node``.

        Returns ``None``, or ``(separator_key, new_right_node)`` if ``node``
        split and the parent must absorb the new sibling.
        """
        if isinstance(node, _Leaf):
            i = bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                raise MiniDBError(f"duplicate key: {key}")
            node.keys.insert(i, key)
            node.values.insert(i, value)
            if len(node.keys) > self._order:
                return self._split_leaf(node)
            return None

        # internal node: recurse into the right child, then maybe absorb a split
        i = bisect_right(node.keys, key)
        split = self._insert(node.children[i], key, value)
        if split is None:
            return None
        sep, right = split
        node.keys.insert(i, sep)
        node.children.insert(i + 1, right)
        if len(node.keys) > self._order:
            return self._split_internal(node)
        return None

    def _split_leaf(self, leaf: _Leaf):
        mid = (len(leaf.keys) + 1) // 2
        right = _Leaf()
        right.keys = leaf.keys[mid:]
        right.values = leaf.values[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        right.next = leaf.next
        leaf.next = right
        return (right.keys[0], right)  # separator is COPIED up

    def _split_internal(self, node: _Internal):
        mid = len(node.keys) // 2
        up = node.keys[mid]  # this key moves UP; it is not kept here
        right = _Internal()
        right.keys = node.keys[mid + 1 :]
        right.children = node.children[mid + 1 :]
        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]
        return (up, right)

    # -- debugging ---------------------------------------------------------

    def debug_str(self) -> str:
        """One line per level: ``[..]`` internal nodes, ``(..)`` leaves."""
        lines, level = [], [self._root]
        while level:
            rendered, nxt = [], []
            for node in level:
                if isinstance(node, _Internal):
                    rendered.append("[" + " ".join(map(str, node.keys)) + "]")
                    nxt.extend(node.children)
                else:
                    rendered.append("(" + " ".join(map(str, node.keys)) + ")")
            lines.append("   ".join(rendered))
            level = nxt
        return "\n".join(lines)
