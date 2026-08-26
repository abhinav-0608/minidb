"""Stage 3 tests - the Database.

* TestCreateAndRediscover - create, restart, tables + schemas + rows come back
* TestEmpty                - a database with no tables round-trips cleanly
* TestDuplicate            - a repeated table name is refused, nothing disturbed
* TestValidation           - bad names / missing 'id INT' first column refused
* TestIndependence         - tables in one file do not see each other's rows
* TestNoCorruptionOnError  - a refused create leaves the database fully usable
"""

from __future__ import annotations

import pytest

from minidb.database import Database
from minidb.errors import MiniDBError
from minidb.record import Column, ColumnType as CT, Schema

USERS = Schema((Column("id", CT.INT), Column("name", CT.TEXT), Column("age", CT.INT)))
PRODUCTS = Schema(
    (Column("id", CT.INT), Column("title", CT.TEXT), Column("price", CT.INT))
)


# ---------------------------------------------------------------------------
class TestCreateAndRediscover:
    def test_goal_create_restart_rediscover(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Database(path) as db:
            db.create_table("users", USERS)
            db.create_table("products", PRODUCTS)
            db.open_table("users").insert((1, "Alice", 24))
            db.open_table("users").insert((2, "Bob", 31))
            db.open_table("products").insert((10, "Widget", 999))
        with Database(path) as db:
            assert db.table_names() == ["users", "products"]
            assert db.schema_of("users") == USERS
            assert db.schema_of("products") == PRODUCTS
            assert list(db.open_table("users").scan()) == [
                (1, "Alice", 24),
                (2, "Bob", 31),
            ]
            assert list(db.open_table("products").scan()) == [(10, "Widget", 999)]

    def test_create_is_visible_before_restart(self, tmp_path):
        with Database(str(tmp_path / "t.db")) as db:
            db.create_table("users", USERS)
            assert db.table_names() == ["users"]
            assert db.schema_of("users") == USERS

    def test_many_tables_survive_restart(self, tmp_path):
        path = str(tmp_path / "t.db")
        schemas = {
            f"tab{i}": Schema(
                (Column("id", CT.INT), Column(f"c{i}", CT.TEXT))
            )
            for i in range(30)
        }
        with Database(path) as db:
            for name, s in schemas.items():
                db.create_table(name, s)
        with Database(path) as db:
            assert db.table_names() == list(schemas)
            for name, s in schemas.items():
                assert db.schema_of(name) == s

    def test_single_id_column_table_is_valid(self, tmp_path):
        with Database(str(tmp_path / "t.db")) as db:
            db.create_table("ids", Schema((Column("id", CT.INT),)))
            db.open_table("ids").insert((42,))
            assert list(db.open_table("ids").scan()) == [(42,)]

    def test_custom_page_size_persists_through_database(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Database(path, page_size=1024) as db:
            db.create_table("users", USERS)
            db.open_table("users").insert((1, "Alice", 24))
        with Database(path) as db:  # reopen with the default
            assert db._pager.page_size == 1024
            assert list(db.open_table("users").scan()) == [(1, "Alice", 24)]


# ---------------------------------------------------------------------------
class TestEmpty:
    def test_fresh_database_has_no_tables(self, tmp_path):
        with Database(str(tmp_path / "t.db")) as db:
            assert db.table_names() == []

    def test_empty_database_round_trips(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Database(path) as db:
            pass
        with Database(path) as db:
            assert db.table_names() == []


# ---------------------------------------------------------------------------
class TestDuplicate:
    def test_duplicate_name_is_refused_and_changes_nothing(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Database(path) as db:
            db.create_table("users", USERS)
            db.open_table("users").insert((1, "Alice", 24))
            with pytest.raises(MiniDBError):
                db.create_table("users", PRODUCTS)  # different schema, same name
            assert db.table_names() == ["users"]
            assert db.schema_of("users") == USERS
            assert list(db.open_table("users").scan()) == [(1, "Alice", 24)]
        with Database(path) as db:
            assert db.table_names() == ["users"]
            assert db.schema_of("users") == USERS


# ---------------------------------------------------------------------------
class TestValidation:
    @pytest.mark.parametrize("name", ["1bad", "has space", "", "dash-name", "a.b"])
    def test_bad_table_name_refused(self, tmp_path, name):
        with Database(str(tmp_path / "t.db")) as db:
            with pytest.raises(MiniDBError):
                db.create_table(name, USERS)

    def test_first_column_must_be_named_id(self, tmp_path):
        with Database(str(tmp_path / "t.db")) as db:
            with pytest.raises(MiniDBError):
                db.create_table(
                    "t", Schema((Column("uid", CT.INT), Column("x", CT.TEXT)))
                )

    def test_first_column_must_be_int(self, tmp_path):
        with Database(str(tmp_path / "t.db")) as db:
            with pytest.raises(MiniDBError):
                db.create_table(
                    "t", Schema((Column("id", CT.TEXT), Column("x", CT.TEXT)))
                )

    def test_bad_non_first_column_name_refused(self, tmp_path):
        with Database(str(tmp_path / "t.db")) as db:
            with pytest.raises(MiniDBError):
                db.create_table(
                    "t", Schema((Column("id", CT.INT), Column("bad name", CT.TEXT)))
                )

    def test_unknown_table_lookups_raise(self, tmp_path):
        with Database(str(tmp_path / "t.db")) as db:
            with pytest.raises(MiniDBError):
                db.schema_of("ghost")
            with pytest.raises(MiniDBError):
                db.open_table("ghost")


# ---------------------------------------------------------------------------
class TestIndependence:
    def test_interleaved_inserts_stay_separate(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Database(path) as db:
            db.create_table("users", USERS)
            db.create_table("products", PRODUCTS)
            db.open_table("users").insert((1, "Alice", 24))
            db.open_table("products").insert((10, "Widget", 999))
            db.open_table("users").insert((2, "Bob", 31))
            db.open_table("products").insert((11, "Gadget", 5))
        with Database(path) as db:
            assert list(db.open_table("users").scan()) == [
                (1, "Alice", 24),
                (2, "Bob", 31),
            ]
            assert list(db.open_table("products").scan()) == [
                (10, "Widget", 999),
                (11, "Gadget", 5),
            ]

    def test_creating_a_table_does_not_disturb_an_existing_one(self, tmp_path):
        with Database(str(tmp_path / "t.db")) as db:
            db.create_table("users", USERS)
            db.open_table("users").insert((1, "Alice", 24))
            db.create_table("products", PRODUCTS)
            assert list(db.open_table("users").scan()) == [(1, "Alice", 24)]
            assert db.schema_of("users") == USERS


# ---------------------------------------------------------------------------
class TestNoCorruptionOnError:
    def test_refused_create_leaves_database_usable(self, tmp_path):
        path = str(tmp_path / "t.db")
        with Database(path) as db:
            with pytest.raises(MiniDBError):
                db.create_table("t", Schema((Column("id", CT.TEXT),)))  # bad
            # database still works
            db.create_table("users", USERS)
            db.open_table("users").insert((1, "Alice", 24))
        with Database(path) as db:
            assert db.table_names() == ["users"]
            assert list(db.open_table("users").scan()) == [(1, "Alice", 24)]

    def test_validation_failure_allocates_no_page(self, tmp_path):
        with Database(str(tmp_path / "t.db")) as db:
            before = db._pager.page_count
            with pytest.raises(MiniDBError):
                db.create_table("t", Schema((Column("uid", CT.INT),)))  # no 'id'
            assert db._pager.page_count == before
