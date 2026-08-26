"""Stage 3 tests - the catalog layer (below Database).

* TestColumnCodec   - encode_columns / decode_columns round-trip and rejections
* TestOpenFresh     - a new file gets a catalog page at page 1
* TestPersistence   - entries survive a pager close/reopen, in order
* TestCorrupt       - page 1 that is not a valid catalog is rejected
* TestChaining      - a catalog with many tables spans linked pages
"""

from __future__ import annotations

import pytest

from minidb.catalog import (
    CATALOG_SCHEMA,
    Catalog,
    decode_columns,
    encode_columns,
)
from minidb.errors import MiniDBError
from minidb.heap import Heap
from minidb.page import PageType, SlottedPage
from minidb.pager import Pager
from minidb.record import Column, ColumnType, Schema

USERS = Schema(
    (
        Column("id", ColumnType.INT),
        Column("name", ColumnType.TEXT),
        Column("age", ColumnType.INT),
    )
)


# ---------------------------------------------------------------------------
class TestColumnCodec:
    def test_exact_encoding(self):
        assert encode_columns(USERS) == "id INT, name TEXT, age INT"

    @pytest.mark.parametrize(
        "schema",
        [
            Schema((Column("id", ColumnType.INT),)),
            USERS,
            Schema(tuple(Column(f"c{i}", ColumnType.TEXT) for i in range(15))),
        ],
    )
    def test_round_trip(self, schema):
        assert decode_columns(encode_columns(schema)) == schema

    @pytest.mark.parametrize(
        "spec",
        [
            "",  # no tokens
            "id",  # one token
            "id INT extra",  # three tokens
            "id BLOB",  # unknown type
            "id INT, id INT",  # duplicate name -> Schema rejects
        ],
    )
    def test_decode_rejects_garbage(self, spec):
        with pytest.raises(MiniDBError):
            decode_columns(spec)


# ---------------------------------------------------------------------------
class TestOpenFresh:
    def test_new_file_gets_an_empty_catalog(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            cat = Catalog.open(pg)
            assert cat.table_names() == []
            assert pg.page_count == 2  # header + catalog page

    def test_catalog_page_is_typed_catalog(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            Catalog.open(pg)
            page = SlottedPage(
                bytearray(pg.read_page(1)), expected_type=PageType.CATALOG
            )
            assert page.page_type == PageType.CATALOG

    def test_reopening_an_empty_catalog_is_still_empty(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Pager(path) as pg:
            Catalog.open(pg)
        with Pager(path) as pg:
            assert Catalog.open(pg).table_names() == []


# ---------------------------------------------------------------------------
class TestPersistence:
    def test_entries_survive_reopen_in_insertion_order(self, tmp_path):
        path = str(tmp_path / "t.db")
        other = Schema((Column("id", ColumnType.INT), Column("v", ColumnType.TEXT)))
        with Pager(path) as pg:
            cat = Catalog.open(pg)
            cat.add("users", 2, 90, USERS)
            cat.add("logs", 3, 91, other)
        with Pager(path) as pg:
            cat = Catalog.open(pg)
            assert cat.table_names() == ["users", "logs"]
            assert cat.get("users").first_page_id == 2
            assert cat.get("users").index_root_page_id == 90
            assert cat.get("users").schema == USERS
            assert cat.get("logs").index_root_page_id == 91
            assert cat.get("logs").schema == other

    def test_add_is_visible_immediately(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            cat = Catalog.open(pg)
            cat.add("users", 2, 90, USERS)
            assert cat.has("users")
            assert cat.get("users").schema == USERS
            assert cat.get("users").index_root_page_id == 90

    def test_get_unknown_raises(self, tmp_path):
        with Pager(str(tmp_path / "t.db")) as pg:
            with pytest.raises(MiniDBError):
                Catalog.open(pg).get("nope")


# ---------------------------------------------------------------------------
class TestCorrupt:
    def test_heap_page_at_page_1_is_rejected(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Pager(path) as pg:
            Heap.create(pg, USERS)  # plain HEAP page becomes page 1
        with Pager(path) as pg:
            with pytest.raises(MiniDBError):
                Catalog.open(pg)

    def test_unparseable_column_defs_is_rejected_on_open(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Pager(path) as pg:
            Catalog.open(pg)  # make a real catalog page
            # write a row straight into the catalog heap, bypassing Catalog.add
            raw = Heap(pg, CATALOG_SCHEMA, 1, page_type=PageType.CATALOG)
            raw.insert(("bad", 9, "this is not a column list", 10))
        with Pager(path) as pg:
            with pytest.raises(MiniDBError):
                Catalog.open(pg)


# ---------------------------------------------------------------------------
class TestChaining:
    def test_many_tables_span_linked_pages(self, tmp_path):
        path = str(tmp_path / "t.db")
        names = [f"t{i:03d}" for i in range(150)]
        with Pager(path, page_size=512) as pg:
            cat = Catalog.open(pg)
            for i, n in enumerate(names):
                cat.add(n, 1000 + i, 5000 + i, USERS)
            assert pg.page_count > 3  # catalog definitely chained
        with Pager(path, page_size=512) as pg:
            cat = Catalog.open(pg)
            assert cat.table_names() == names
            assert cat.get("t075").first_page_id == 1075
            assert cat.get("t075").index_root_page_id == 5075
            assert cat.get("t149").schema == USERS
