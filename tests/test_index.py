"""Stage 6 tests - the persistent B+ tree index.

``assert_index_invariants`` is the disk-tree counterpart of Stage 5's
checker: it reads every node through ``idx._read`` and verifies

  * all leaves at one depth; ``height()`` agrees;
  * keys strictly sorted, within capacity, in every node;
  * internal nodes: ``len(children) == len(keys)+1``, >= 1 separator;
  * each separator > all left-subtree keys, <= all right-subtree keys, and
    == the minimum key of the right subtree;
  * the leaf chain (via ``next_leaf_id``) is globally sorted, visits every
    leaf once, ends at 0, and holds exactly the tree's keys;
  * ``items()`` matches the chain.

* TestNodeCodec   - _Leaf / _Internal serialise/parse round-trips
* TestCreateEmpty - a fresh index; page-too-small guard
* TestPersistence - the tree survives a pager close/reopen
* TestSplits      - tiny capacity, invariants after every insert
* TestFixedRoot   - the root page id never moves, across splits and restart
* TestDuplicate   - the unique-key policy
* TestOracleFuzz  - random inserts vs a dict, several capacities
* TestPageType    - a non-B+tree page under the root is rejected
"""

from __future__ import annotations

import random

import pytest

from minidb.errors import MiniDBError
from minidb.heap import Heap
from minidb.index import BTreeIndex, _Internal, _Leaf
from minidb.page import PageType
from minidb.pager import Pager
from minidb.record import Column, ColumnType, Schema


# --- invariant checker ---------------------------------------------------

def _leaf_depths(idx, node, depth=0):
    if isinstance(node, _Leaf):
        return [depth]
    out = []
    for c in node.children:
        out += _leaf_depths(idx, idx._read(c), depth + 1)
    return out


def _check_subtree(idx, node) -> list[int]:
    """Validate node + subtree; return this subtree's keys in order."""
    assert node.keys == sorted(node.keys), f"keys not sorted: {node.keys}"
    assert len(set(node.keys)) == len(node.keys), f"dup keys: {node.keys}"
    if isinstance(node, _Leaf):
        assert len(node.keys) <= idx.leaf_capacity, "leaf over capacity"
        assert len(node.keys) == len(node.rids)
        return list(node.keys)

    assert len(node.keys) <= idx.internal_capacity, "internal over capacity"
    assert len(node.children) == len(node.keys) + 1
    assert len(node.keys) >= 1, "internal node with no separator"
    child_keys = [_check_subtree(idx, idx._read(c)) for c in node.children]
    for i, sep in enumerate(node.keys):
        assert all(k < sep for k in child_keys[i]), f"left subtree key >= sep {sep}"
        assert all(k >= sep for k in child_keys[i + 1]), f"right subtree key < sep {sep}"
        assert child_keys[i + 1][0] == sep, f"sep {sep} != min right subtree"
    out: list[int] = []
    for ck in child_keys:
        out += ck
    return out


def assert_index_invariants(idx: BTreeIndex, expected=None) -> None:
    root = idx._read(idx.root_page_id)

    depths = _leaf_depths(idx, root)
    assert len(set(depths)) == 1, f"leaf depths differ: {sorted(set(depths))}"
    assert idx.height() == depths[0] + 1

    struct_keys = _check_subtree(idx, root)
    assert struct_keys == sorted(struct_keys)

    node = root
    while isinstance(node, _Internal):
        node = idx._read(node.children[0])
    chain_items: list[tuple] = []
    leaves = 0
    while True:
        leaves += 1
        assert node.keys == sorted(node.keys)
        chain_items += list(zip(node.keys, node.rids))
        if node.next_leaf_id == 0:
            break
        node = idx._read(node.next_leaf_id)
    chain_keys = [k for k, _ in chain_items]
    assert chain_keys == sorted(chain_keys), "leaf chain not globally sorted"
    assert len(set(chain_keys)) == len(chain_keys), "leaf chain has dup keys"
    assert leaves == len(depths), "chain leaf count != structural leaf count"
    assert chain_keys == struct_keys
    assert list(idx.items()) == chain_items

    if expected is not None:
        assert chain_items == sorted((k, tuple(v)) for k, v in expected.items())


PATTERNS = {
    "ascending": lambda n: list(range(n)),
    "descending": lambda n: list(range(n - 1, -1, -1)),
    "shuffled": lambda n: random.Random(n).sample(range(n), n),
}


@pytest.fixture
def pager(tmp_path):
    with Pager(str(tmp_path / "i.db")) as pg:
        yield pg


# ---------------------------------------------------------------------------
class TestNodeCodec:
    def test_leaf_round_trip(self):
        n = _Leaf([1, 5, 9], [(2, 0), (3, 1), (4, 7)], next_leaf_id=42)
        m = _Leaf.parse(n.to_bytes(4096))
        assert (m.keys, m.rids, m.next_leaf_id) == ([1, 5, 9], [(2, 0), (3, 1), (4, 7)], 42)

    def test_internal_round_trip(self):
        n = _Internal([10, 20], [1, 2, 3])
        m = _Internal.parse(n.to_bytes(4096))
        assert (m.keys, m.children) == ([10, 20], [1, 2, 3])

    def test_empty_leaf_round_trip(self):
        m = _Leaf.parse(_Leaf().to_bytes(4096))
        assert m.keys == [] and m.rids == [] and m.next_leaf_id == 0

    def test_negative_keys_round_trip(self):
        n = _Leaf([-(2**40), -1, 0, 3], [(1, 1)] * 4)
        assert _Leaf.parse(n.to_bytes(4096)).keys == [-(2**40), -1, 0, 3]

    def test_full_leaf_fills_exactly_one_page(self):
        cap = (4096 - 8) // 14
        n = _Leaf(list(range(cap)), [(i, i % 7) for i in range(cap)])
        b = n.to_bytes(4096)
        assert len(b) == 4096
        assert _Leaf.parse(b).rids == [(i, i % 7) for i in range(cap)]

    def test_parse_rejects_wrong_page_type(self):
        with pytest.raises(MiniDBError):
            _Leaf.parse(bytearray(4096))  # type byte 0
        with pytest.raises(MiniDBError):
            _Internal.parse(_Leaf([1], [(2, 3)]).to_bytes(4096))


# ---------------------------------------------------------------------------
class TestCreateEmpty:
    def test_fresh_index(self, pager):
        idx = BTreeIndex.create(pager)
        assert idx.search(1) is None
        assert 1 not in idx
        assert list(idx.items()) == []
        assert idx.height() == 1
        assert_index_invariants(idx)

    @pytest.mark.parametrize("caps", [{"leaf_capacity": 1}, {"internal_capacity": 1}])
    def test_capacity_below_two_rejected(self, pager, caps):
        with pytest.raises(MiniDBError):
            BTreeIndex.create(pager, **caps)

    def test_default_capacities_from_page_size(self, pager):
        idx = BTreeIndex.create(pager)  # 4096
        assert idx.leaf_capacity == (4096 - 8) // 14
        assert idx.internal_capacity == (4096 - 12) // 12


# ---------------------------------------------------------------------------
class TestPersistence:
    def test_reopen_and_search(self, tmp_path):
        path = str(tmp_path / "i.db")
        keys = random.Random(1).sample(range(5000), 800)
        rid_of = lambda k: (k % 40 + 2, k % 20)
        with Pager(path) as pg:
            idx = BTreeIndex.create(pg, leaf_capacity=4, internal_capacity=4)
            root = idx.root_page_id
            for k in keys:
                idx.insert(k, rid_of(k))
        with Pager(path) as pg:
            idx = BTreeIndex(pg, root, leaf_capacity=4, internal_capacity=4)
            assert idx.root_page_id == root
            assert_index_invariants(idx, {k: rid_of(k) for k in keys})
            for k in keys:
                assert idx.search(k) == rid_of(k)
            for miss in range(5000, 5100):
                assert idx.search(miss) is None


# ---------------------------------------------------------------------------
class TestSplits:
    @pytest.mark.parametrize("cap", [3, 4, 5])
    @pytest.mark.parametrize("pattern", list(PATTERNS))
    def test_invariants_after_every_insert(self, tmp_path, cap, pattern):
        with Pager(str(tmp_path / "i.db")) as pg:
            idx = BTreeIndex.create(pg, leaf_capacity=cap, internal_capacity=cap)
            keys = PATTERNS[pattern](60)
            for k in keys:
                idx.insert(k, (k % 5 + 2, k))
                assert_index_invariants(idx)
            assert [k for k, _ in idx.items()] == sorted(keys)

    @pytest.mark.parametrize("cap", [3, 4])
    @pytest.mark.parametrize("pattern", list(PATTERNS))
    def test_larger_run(self, tmp_path, cap, pattern):
        with Pager(str(tmp_path / "i.db")) as pg:
            idx = BTreeIndex.create(pg, leaf_capacity=cap, internal_capacity=cap)
            keys = PATTERNS[pattern](1200)
            for k in keys:
                idx.insert(k, (k % 90 + 2, 0))
            assert_index_invariants(idx)
            assert [k for k, _ in idx.items()] == list(range(1200))

    def test_first_root_split_at_capacity_plus_one(self, tmp_path):
        with Pager(str(tmp_path / "i.db")) as pg:
            idx = BTreeIndex.create(pg, leaf_capacity=3, internal_capacity=3)
            for k in (1, 2, 3):
                idx.insert(k, (2, k))
                assert idx.height() == 1
            idx.insert(4, (2, 4))
            assert idx.height() == 2

    def test_rids_survive_many_splits(self, tmp_path):
        with Pager(str(tmp_path / "i.db")) as pg:
            idx = BTreeIndex.create(pg, leaf_capacity=3, internal_capacity=3)
            for k in range(200):
                idx.insert(k, (k // 10 + 2, k % 10))
            assert list(idx.items()) == [
                (k, (k // 10 + 2, k % 10)) for k in range(200)
            ]


# ---------------------------------------------------------------------------
class TestFixedRoot:
    def test_root_page_id_is_constant_across_splits_and_restart(self, tmp_path):
        path = str(tmp_path / "i.db")
        with Pager(path) as pg:
            idx = BTreeIndex.create(pg, leaf_capacity=3, internal_capacity=3)
            root = idx.root_page_id
            heights = set()
            for k in range(400):
                idx.insert(k, (2, k))
                assert idx.root_page_id == root
                heights.add(idx.height())
            assert max(heights) >= 4  # several root splits happened
        with Pager(path) as pg:
            idx = BTreeIndex(pg, root, leaf_capacity=3, internal_capacity=3)
            assert idx.root_page_id == root
            idx.insert(1000, (2, 0))
            assert idx.root_page_id == root
            assert_index_invariants(idx)


# ---------------------------------------------------------------------------
class TestDuplicate:
    def test_duplicate_raises_and_preserves_the_tree(self, tmp_path):
        with Pager(str(tmp_path / "i.db")) as pg:
            idx = BTreeIndex.create(pg, leaf_capacity=3, internal_capacity=3)
            for k in range(50):
                idx.insert(k, (2, k))
            before = list(idx.items())
            with pytest.raises(MiniDBError):
                idx.insert(25, (9, 9))
            assert list(idx.items()) == before
            assert idx.search(25) == (2, 25)
            assert_index_invariants(idx)


# ---------------------------------------------------------------------------
class TestOracleFuzz:
    @pytest.mark.parametrize("cap", [3, 4, 5, 8, None])
    @pytest.mark.parametrize("trial", range(4))
    def test_random_inserts_match_a_dict(self, tmp_path, cap, trial):
        rng = random.Random((cap or 99) * 100 + trial)
        n = rng.choice([40, 300, 900])
        keys = rng.sample(range(500_000), n)
        caps = {} if cap is None else {"leaf_capacity": cap, "internal_capacity": cap}
        with Pager(str(tmp_path / "i.db")) as pg:
            idx = BTreeIndex.create(pg, **caps)
            oracle: dict[int, tuple[int, int]] = {}
            for k in keys:
                rid = (rng.randrange(2, 1000), rng.randrange(0, 200))
                idx.insert(k, rid)
                oracle[k] = rid
            assert_index_invariants(idx, oracle)
            for k in list(oracle) + rng.sample(range(500_000), 300):
                assert idx.search(k) == oracle.get(k)
                assert (k in idx) == (k in oracle)

    def test_deep_tiny_order_tree(self, tmp_path):
        with Pager(str(tmp_path / "i.db")) as pg:
            idx = BTreeIndex.create(pg, leaf_capacity=3, internal_capacity=3)
            for k in random.Random(0).sample(range(30000), 3000):
                idx.insert(k, (2, k))
            assert_index_invariants(idx)
            assert idx.height() >= 6


# ---------------------------------------------------------------------------
class TestPageType:
    def test_heap_page_under_the_root_is_rejected(self, tmp_path):
        with Pager(str(tmp_path / "i.db")) as pg:
            h = Heap.create(
                pg, Schema((Column("id", ColumnType.INT),)), page_type=PageType.HEAP
            )
            idx = BTreeIndex(pg, h.first_page_id)
            with pytest.raises(MiniDBError):
                idx.search(1)

    def test_zeroed_page_under_the_root_is_rejected(self, tmp_path):
        with Pager(str(tmp_path / "i.db")) as pg:
            pid = pg.allocate_page()
            with pytest.raises(MiniDBError):
                BTreeIndex(pg, pid).search(1)
