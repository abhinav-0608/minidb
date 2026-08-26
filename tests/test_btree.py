"""Stage 5 tests - the in-memory B+ tree.

The backbone is ``assert_btree_invariants``: a full structural check run after
operations. It verifies

  * every leaf is at the same depth, and ``height`` agrees with that depth;
  * keys are strictly sorted inside every node;
  * every internal node has ``len(children) == len(keys) + 1`` and >= 1 key;
  * each separator ``s_i`` is > all keys in the left subtree, <= all keys in
    the right subtree, and exactly equals the minimum key of the right
    subtree (the B+ tree "separator copied from a leaf" property);
  * the linked leaf chain is globally sorted, visits every leaf once, ends at
    None, and holds exactly the tree's keys;
  * ``len(tree)``, ``keys()`` and the chain all agree;
  * no node exceeds ``order`` keys.

* TestDesignTrace   - the doc's 10..60 @order3 walkthrough, pinned exactly
* TestEmptyAndOrder - empty tree behaviour, order validation
* TestInsertOrders  - ascending / descending / shuffled at tiny orders
* TestDuplicateKey  - the unique-key policy
* TestHeightGrowth  - height only ever rises, and only at a root split
* TestLeafChain     - the linked leaves, and range-scan-style iteration
* TestOracleFuzz    - random inserts vs a dict, many orders
* TestOpaqueValues  - the tree never inspects a value
"""

from __future__ import annotations

import random

import pytest

from minidb.btree import BPlusTree, _Internal, _Leaf
from minidb.errors import MiniDBError


# --- the invariant checker -------------------------------------------------

def _all_keys(node) -> list[int]:
    if isinstance(node, _Leaf):
        return list(node.keys)
    out: list[int] = []
    for c in node.children:
        out += _all_keys(c)
    return out


def _leaf_depths(node, depth: int = 0) -> list[int]:
    if isinstance(node, _Leaf):
        return [depth]
    out: list[int] = []
    for c in node.children:
        out += _leaf_depths(c, depth + 1)
    return out


def _min_key(node) -> int:
    while isinstance(node, _Internal):
        node = node.children[0]
    return node.keys[0]


def _check_node(node, order: int) -> None:
    assert node.keys == sorted(node.keys), f"keys not sorted: {node.keys}"
    assert len(set(node.keys)) == len(node.keys), f"dup keys in node: {node.keys}"
    assert len(node.keys) <= order, f"over capacity: {len(node.keys)} > {order}"
    if isinstance(node, _Internal):
        assert len(node.children) == len(node.keys) + 1
        assert len(node.keys) >= 1, "internal node with no separator"
        for i, sep in enumerate(node.keys):
            left = _all_keys(node.children[i])
            right = _all_keys(node.children[i + 1])
            assert all(k < sep for k in left), f"left subtree key >= sep {sep}"
            assert all(k >= sep for k in right), f"right subtree key < sep {sep}"
            assert _min_key(node.children[i + 1]) == sep, (
                f"separator {sep} != min right subtree {_min_key(node.children[i + 1])}"
            )
        for c in node.children:
            _check_node(c, order)


def assert_btree_invariants(tree: BPlusTree, expected_items=None) -> None:
    root = tree._root

    depths = _leaf_depths(root)
    assert len(set(depths)) == 1, f"leaves at differing depths: {sorted(set(depths))}"
    assert tree.height == depths[0] + 1, f"height {tree.height} != {depths[0] + 1}"

    _check_node(root, tree.order)

    leaf = tree._leftmost_leaf()
    chain_keys: list[int] = []
    chain_items: list[tuple] = []
    leaves_seen = 0
    while leaf is not None:
        leaves_seen += 1
        assert leaf.keys == sorted(leaf.keys)
        assert len(leaf.keys) == len(leaf.values)
        chain_keys += leaf.keys
        chain_items += list(zip(leaf.keys, leaf.values))
        leaf = leaf.next
    assert chain_keys == sorted(chain_keys), "leaf chain not globally sorted"
    assert len(set(chain_keys)) == len(chain_keys), "leaf chain has duplicate keys"
    assert leaves_seen == len(depths), "chain visits a different leaf count than the tree"
    assert sorted(_all_keys(root)) == chain_keys, "structural keys != chain keys"

    assert len(tree) == len(chain_keys), f"len {len(tree)} != {len(chain_keys)}"
    assert tree.keys() == chain_keys

    if expected_items is not None:
        assert chain_items == sorted(expected_items)


# ---------------------------------------------------------------------------
class TestDesignTrace:
    def test_heights_after_each_insert(self):
        t = BPlusTree(order=3)
        seen = []
        for k in (10, 20, 30, 40, 50, 60):
            t.insert(k, f"v{k}")
            seen.append(t.height)
            assert_btree_invariants(t)
        assert seen == [1, 1, 1, 2, 2, 2]

    def test_final_shape_is_pinned(self):
        t = BPlusTree(order=3)
        for k in (10, 20, 30, 40, 50, 60):
            t.insert(k, f"v{k}")
        assert t.debug_str() == "[30 50]\n(10 20)   (30 40)   (50 60)"

    def test_search_walks_to_the_right_leaf(self):
        t = BPlusTree(order=3)
        for k in (10, 20, 30, 40, 50, 60):
            t.insert(k, f"v{k}")
        for k in (10, 20, 30, 40, 50, 60):
            assert t.search(k) == f"v{k}"
        for miss in (5, 25, 35, 55, 100):
            assert t.search(miss) is None
            assert miss not in t


# ---------------------------------------------------------------------------
class TestEmptyAndOrder:
    def test_empty_tree(self):
        t = BPlusTree(order=4)
        assert len(t) == 0
        assert t.height == 1
        assert t.keys() == []
        assert list(t.items()) == []
        assert t.search(1) is None
        assert 1 not in t
        assert_btree_invariants(t, expected_items=[])

    @pytest.mark.parametrize("bad", [2, 1, 0, -5])
    def test_order_below_three_rejected(self, bad):
        with pytest.raises(MiniDBError):
            BPlusTree(order=bad)

    def test_order_three_is_allowed(self):
        BPlusTree(order=3)

    def test_negative_and_zero_keys(self):
        t = BPlusTree(order=3)
        for k in (0, -5, 5, -1, 1):
            t.insert(k, k)
        assert t.keys() == [-5, -1, 0, 1, 5]
        assert t.search(0) == 0 and t.search(-5) == -5
        assert_btree_invariants(t)


# ---------------------------------------------------------------------------
class TestInsertOrders:
    PATTERNS = {
        "ascending": lambda n, rng: list(range(n)),
        "descending": lambda n, rng: list(range(n - 1, -1, -1)),
        "shuffled": lambda n, rng: rng.sample(range(n), n),
    }

    @pytest.mark.parametrize("order", [3, 4, 5, 7])
    @pytest.mark.parametrize("pattern", list(PATTERNS))
    def test_invariants_after_every_insert(self, order, pattern):
        rng = random.Random(hash((order, pattern)) & 0xFFFF)
        keys = self.PATTERNS[pattern](60, rng)
        t = BPlusTree(order=order)
        for k in keys:
            t.insert(k, k * 3)
            assert_btree_invariants(t)
        assert t.keys() == sorted(keys)
        for k in keys:
            assert t.search(k) == k * 3

    @pytest.mark.parametrize("order", [3, 4, 8])
    @pytest.mark.parametrize("pattern", list(PATTERNS))
    def test_larger_run(self, order, pattern):
        rng = random.Random(hash((order, pattern, "big")) & 0xFFFF)
        keys = self.PATTERNS[pattern](2000, rng)
        t = BPlusTree(order=order)
        for k in keys:
            t.insert(k, -k)
        assert_btree_invariants(t)
        assert t.keys() == list(range(2000))
        assert all(t.search(k) == -k for k in keys)


# ---------------------------------------------------------------------------
class TestDuplicateKey:
    def test_reinserting_a_key_raises_and_changes_nothing(self):
        t = BPlusTree(order=3)
        for k in range(10):
            t.insert(k, k)
        with pytest.raises(MiniDBError):
            t.insert(4, 999)
        assert len(t) == 10
        assert t.search(4) == 4  # original value kept
        assert_btree_invariants(t)
        t.insert(10, 10)  # still usable
        assert t.search(10) == 10

    def test_duplicate_in_a_deep_tree(self):
        t = BPlusTree(order=3)
        for k in range(200):
            t.insert(k, k)
        for k in (0, 99, 150, 199):
            with pytest.raises(MiniDBError):
                t.insert(k, -1)
        assert len(t) == 200
        assert_btree_invariants(t)


# ---------------------------------------------------------------------------
class TestHeightGrowth:
    def test_height_is_monotonic_nondecreasing(self):
        t = BPlusTree(order=3)
        h = t.height
        for k in range(300):
            t.insert(k, k)
            assert t.height >= h
            h = t.height
        assert 4 <= t.height <= 9  # order 3, 300 keys

    def test_first_root_split_is_at_the_fourth_insert(self):
        t = BPlusTree(order=3)
        for k in (1, 2, 3):
            t.insert(k, k)
            assert t.height == 1
        t.insert(4, 4)
        assert t.height == 2

    def test_bigger_order_means_shorter_tree(self):
        def height_for(order):
            t = BPlusTree(order=order)
            for k in range(5000):
                t.insert(k, k)
            return t.height

        assert height_for(64) < height_for(8) < height_for(3)


# ---------------------------------------------------------------------------
class TestLeafChain:
    def test_chain_is_sorted_and_terminated(self):
        t = BPlusTree(order=4)
        for k in random.Random(1).sample(range(5000), 3000):
            t.insert(k, k)
        leaf = t._leftmost_leaf()
        prev = None
        count = 0
        while leaf is not None:
            count += 1
            if prev is not None:
                assert prev < leaf.keys[0]
            prev = leaf.keys[-1]
            leaf = leaf.next
        assert count == len(_leaf_depths(t._root))
        assert t.keys() == sorted(random.Random(1).sample(range(5000), 3000))

    def test_items_matches_sorted_pairs(self):
        t = BPlusTree(order=5)
        pairs = [(k, k * 2) for k in random.Random(2).sample(range(2000), 800)]
        for k, v in pairs:
            t.insert(k, v)
        assert list(t.items()) == sorted(pairs)

    def test_range_scan_style_iteration(self):
        t = BPlusTree(order=4)
        for k in range(1000):
            t.insert(k, k)
        lo, hi = 273, 641
        got = [v for k, v in t.items() if lo <= k < hi]
        assert got == list(range(lo, hi))


# ---------------------------------------------------------------------------
class TestOracleFuzz:
    @pytest.mark.parametrize("order", [3, 4, 5, 8, 16, 64])
    @pytest.mark.parametrize("trial", range(6))
    def test_random_inserts_match_a_dict(self, order, trial):
        rng = random.Random(order * 97 + trial)
        n = rng.choice([50, 500, 1500])
        keys = rng.sample(range(1_000_000), n)
        t = BPlusTree(order=order)
        oracle: dict[int, int] = {}
        for k in keys:
            v = rng.randrange(1 << 30)
            t.insert(k, v)
            oracle[k] = v
        assert_btree_invariants(t, expected_items=list(oracle.items()))
        assert len(t) == len(oracle)
        probes = list(oracle) + rng.sample(range(1_000_000), 400)
        for k in probes:
            assert t.search(k) == oracle.get(k)
            assert (k in t) == (k in oracle)

    def test_deep_tiny_order_tree(self):
        t = BPlusTree(order=3)
        keys = random.Random(0).sample(range(20000), 3000)
        for k in keys:
            t.insert(k, k)
        assert_btree_invariants(t)
        assert t.height >= 6


# ---------------------------------------------------------------------------
class TestOpaqueValues:
    def test_value_identity_is_preserved(self):
        t = BPlusTree(order=3)
        rid = (17, 4)
        marker = object()
        t.insert(1, rid)
        t.insert(2, marker)
        t.insert(3, "row")
        assert t.search(1) is rid
        assert t.search(2) is marker
        assert t.search(3) == "row"

    def test_tuple_values_survive_splits(self):
        t = BPlusTree(order=3)
        for k in range(100):
            t.insert(k, (k // 10, k % 10))
        for k in range(100):
            assert t.search(k) == (k // 10, k % 10)
        assert_btree_invariants(t)
