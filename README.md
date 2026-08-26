# MiniDB

A small relational database engine, written from scratch in Python to learn how
database internals actually work: pages on disk, a slotted-page record layout, a
persistent B+ tree index, a hand-written SQL parser, and a write-ahead log with
crash recovery.

It is **not** a production database and does not try to be. Python was chosen on
purpose — for readability, not throughput. Every important structure is
hand-built; nothing hides behind SQLite, an ORM, or a B-tree library.

~2,500 lines of engine code, 603 tests.

```sql
MiniDB> CREATE TABLE users (id INT, name TEXT, age INT);
OK
MiniDB> INSERT INTO users VALUES (1, 'Alice', 24);
OK
MiniDB> INSERT INTO users VALUES (2, 'Bob', 31);
OK
MiniDB> SELECT name FROM users WHERE id = 1;
name
Alice
(1 row)
```

The database persists to a single file and survives process restarts — and, via
the WAL, survives a crash *during* a write.

---

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'

# interactive shell
python -m minidb my.db

# run a script, then exit
python -m minidb my.db examples/demo.sql

# tests
.venv/bin/python -m pytest -q

# benchmarks (numbers below are produced by this)
python -m bench.benchmark
```

Programmatic use:

```python
from minidb.database import Database
from minidb.executor import execute_sql

with Database("my.db") as db:
    execute_sql(db, "CREATE TABLE users (id INT, name TEXT, age INT)")
    execute_sql(db, "INSERT INTO users VALUES (1, 'Alice', 24)")
    print(execute_sql(db, "SELECT * FROM users WHERE id = 1").rows)  # [(1, 'Alice', 24)]
```

---

## What it does

| Area | Implemented |
|---|---|
| **Storage** | Single self-describing file; fixed 4 KB pages; slotted-page layout (variable-length records, stable row ids); a page cache with write-back |
| **Records** | Typed columns: `INT` (64-bit signed), `TEXT` (≤ 1024 bytes, UTF-8); leading null bitmap (NULL is format-ready, not yet accepted by the API) |
| **Tables** | Heap file = a linked list of data pages; catalog stored *inside* the database file, as a table of its own |
| **Index** | A persistent B+ tree on the primary key `id`, one node per page, auto-maintained on every insert, used automatically for `WHERE id = k`; leaves linked for ordered iteration; fixed root page |
| **SQL** | `CREATE TABLE`, `INSERT INTO`, `SELECT` (`*` or a column list, optional `WHERE col = value`); `--` comments; `;`-separated scripts. Hand-written tokenizer + recursive-descent parser → typed query objects |
| **Durability** | Redo-only write-ahead log with full-page after-images; flush-before-data rule; autocommit per statement; crash recovery on open; idempotent replay; torn-tail detection |
| **Interface** | `MiniDB>` REPL, `.sql` file runner, `python -m minidb` |

---

## Architecture

```
  SQL text
     │
     ▼
  tokenizer  ──►  [tokens]  ──►  parser  ──►  typed query object
                                             (SelectQuery / InsertQuery / CreateTableQuery)
                                                        │
                                                        ▼
                                                     executor
                                       ┌────────────────┼───────────────┐
                                       ▼                ▼               ▼
                                    catalog          Table           (SELECT *: scan)
                                (table metadata)  heap + id index
                                       │                │
                                       ▼                ▼
                                     heap file      B+ tree index      WHERE id = k:
                              (linked page list)  (key → (page,slot))  tree → RID → heap
                                       └────────┬───────┘
                                                ▼
                                          pager  ──►  page cache
                                                │
                        write path:  begin ─► write_page(s) ─► commit
                                                │                 │
                                                ▼                 ▼
                                          database file        write-ahead log
                                       (array of 4 KB pages)   (fsync'd before the
                                                                data file is touched)
```

| Module | Responsibility |
|---|---|
| `pager.py` | The only code that touches the database file. Page cache, allocation, transactions, WAL, recovery |
| `wal.py` | Log record framing (CRC-checked), append, `recover()` |
| `page.py` | Slotted-page reads/writes; page-type tagging |
| `record.py` | `Schema`; row ↔ bytes codec |
| `heap.py` | A table as a linked list of slotted pages: scan, append |
| `index.py` | The persistent B+ tree: search, insert, node splits, fixed-root growth |
| `table.py` | Heap + index kept in step; unique-`id` enforcement |
| `catalog.py` | Table metadata, itself stored as a heap in the file |
| `database.py` | Ties the pager and catalog together; per-statement transactions |
| `sql/` | `tokenizer.py`, `parser.py` |
| `executor.py` | Runs a query object; picks the index fast-path or a scan |
| `repl.py` | Interactive shell and script runner |

---

## On-disk layout

The file is an array of fixed-size pages; page *N* lives at byte offset
`N * page_size`. Page 0 is a header (magic, format version, page size, page
count). Page 1 is the catalog root, by convention (the same choice SQLite
makes). Everything else is heap pages and B+ tree pages, allocated on demand
and tagged with a page-type byte so a mis-directed read fails loudly.

A **slotted page** grows records up from the bottom and a slot directory down
from the top:

```
+-----------------------------------+
| header (9 B)                      |
+-----------------------------------+
| slot dir → ↓   (offset, length)   |
+ - - - - - - - - - - - - - - - - - +
|             free space            |
+ - - - - - - - - - - - - - - - - - +
|   record bytes ↑                  |
+-----------------------------------+
```

Both ends grow toward the middle; they only meet when the page is full. The
`(page_id, slot_id)` pair is a stable row id even if a record's bytes move
within its page — which is what the index stores.

---

## The B+ tree index

Every table gets exactly one index, on its `id` column. A node is one page, with
keys packed in a fixed-width array — leaf entries are `key → (page_id,
slot_id)`; internal entries are separator keys + child page ids. Because the
entries are fixed-width, a 4 KB page holds ~290 leaf entries or ~340 internal
keys, so the tree is 2–3 levels deep for 100k rows and gains roughly one level
per 300× more rows.

A point lookup is `height` page reads to reach a leaf, plus one heap-page read to
fetch the row. Internal routing uses `bisect_right` (the separator key routes
*right*, because it was copied up from a leaf); the leaf match uses
`bisect_left`. Leaf split copies the separator up; internal split moves the
middle key up. The **root page id never changes**: on a root split the old root's
contents are copied to a fresh page and the root page is rewritten in place — so
the catalog stores the index root once, immutably.

The `SELECT ... WHERE id = k` fast-path is the only optimisation in the executor;
everything else is a full scan.

---

## The write-ahead log

Autocommit: each statement is one transaction. The model is deliberately the
simplest one that gives atomicity + durability:

* **redo-only**, with full-page **after-images** as log records;
* **no-steal** — uncommitted pages never reach the data file, so recovery needs
  no undo;
* **no-force** — committed pages trickle to the data file only at a checkpoint
  (every 128 commits, and on close), so recovery needs redo.

```
begin()   → snapshot page_count, start tracking modified pages
write(s)  → cache + this txn's set (NOT the data file)
commit()  → append PAGE records + COMMIT to the log → fsync the log → done
checkpoint→ write pending pages to the data file → fsync it → truncate the log
abort()   → restore each touched page from its pre-image; roll page_count back
```

On open, a non-empty log is replayed: collect the txn ids that have a `COMMIT`,
redo those transactions' page images in log order, `fsync`, rewrite the header,
truncate the log. A short or CRC-bad frame is a torn tail — it and everything
after it are discarded. Replaying a full-page image is idempotent, so recovery
can itself be interrupted and rerun.

**What it guarantees:** after any crash, the database equals "every
acknowledged statement applied, no un-acknowledged statement applied" — no
half-inserted rows, no dangling index entries, no half-built tables.
**What it does not:** multi-statement transactions, isolation, protection if
`fsync` lies (`os.fsync`, not `F_FULLFSYNC`), or bit-rot beyond the per-record
CRC and per-page type checks.

---

## Benchmarks

Real numbers from `python -m bench.benchmark` — Apple Silicon, Python 3.14,
`os.fsync`. Not tuned; Python and full deserialize-on-read are the point.

**Sequential scan vs. index** — point query `WHERE id = k`:

| rows | pages | tree height | insert/s | scan (ms) | index (ms) | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 17 | 2 | 9,887 | 1.14 | 0.053 | 21× |
| 10,000 | 148 | 2 | 8,358 | 11.23 | 0.100 | 112× |
| 100,000 | 1,471 | 3 | 5,556 | 110.87 | 0.697 | 159× |

The scan is dead linear (10× rows → 10× time). The index lookup stays
sub-millisecond and barely moves, because the tree gains only one level between
10k and 100k rows. The speedup widens with every order of magnitude.

**The WAL's cost** — insert throughput, 2,000 rows:

| mode | inserts/s |
|---|---:|
| `wal=off` (bulk load) | 9,471 |
| `wal=on` (fsync per commit) | 5,327 |

Durability costs an `fsync` per statement — here ~1.8×. The benchmark's build
phase uses `wal=off` for the same reason a real system offers a bulk-load mode.

**Page size** — 10,000 rows, index lookup:

| page size | pages | tree height | leaf/internal capacity | index (ms) |
|---:|---:|---:|---:|---:|
| 1 KB | 591 | 3 | 72 / 84 | 0.240 |
| 4 KB | 148 | 2 | 292 / 340 | 0.099 |
| 8 KB | 75 | 2 | 584 / 681 | 0.098 |

Bigger pages → higher fanout → shorter tree → fewer page reads per lookup. Past
the point where the tree is already flat (4 KB here), a larger page just moves
more bytes per node for no height win — which is why 4 KB is the common default.

---

## Engineering decisions

**Why a B+ tree and not a hash index?** A hash gives O(1) point lookups but no
ordered iteration and no range queries, and on disk it needs dynamic hashing
with painful rehashes. A B+ tree gives O(log n) point lookups *and* ordered
scans via its linked leaves, and grows gracefully by splitting. The whole
project's index need is "find one `id`, or walk `id`s in order" — B+ tree.

**Why fixed-size pages?** `offset = page_id * page_size` — no map of where things
are, just arithmetic. It matches the OS/SSD I/O unit, bounds the work of one
tree step to one page read, and makes cache slots uniform. The cost is internal
fragmentation and a hard "a row must fit in a page" rule (TEXT is capped at
1 KB).

**Why a slotted page instead of flat fixed records?** Flat records can't hold
variable-length TEXT without huge padding, and can't support delete/update
without shifting bytes. Slotted pages handle variable rows, give stable row ids
for the index to point at, and leave room for `DELETE`/`UPDATE` later.

**Why is the catalog a table inside the file?** One artifact to copy or back up,
one recovery story, and "the catalog is just another table" is both a clean
design and a good demonstration that the record/page/heap machinery is general.
A sidecar JSON file would be a second source of truth the WAL doesn't protect.

**Why redo-only, no-steal/no-force — not ARIES?** No-steal means uncommitted
pages never touch the data file, so recovery has *no undo path* at all —
~90 lines of `wal.py` instead of ARIES's analysis/redo/undo with compensation
records. It buys full atomicity and durability at this scale; the trade is
larger log records and no partial rollback (irrelevant under autocommit).

**Why single-writer, autocommit?** Concurrency is a deep topic that would crowd
out storage, indexing, and recovery — the things this project is *about*. The
model is one process, one writer, statements executed serially. An OS advisory
lock stops a second process from corrupting the file.

**Why a hand-written parser?** The grammar is ~5 productions. A recursive-descent
parser is ~120 lines and writing it is a stated goal; a parser library would
hide the interesting part. Typed query objects, not a general AST, because the
grammar has no nested expressions.

**Why deserialize every node/row on read?** Clarity. A production engine
binary-searches directly in the page bytes and mutates in place; MiniDB parses a
node into Python lists and re-serialises the whole node on write. This is the
dominant insert cost in the benchmark — and exactly the kind of thing a
C/C++/Rust implementation would do differently (fixed binary layouts, zero-copy
field access, arena allocation, `mmap`).

---

## Limitations

Deliberately out of scope:

* **SQL**: no `JOIN`, `GROUP BY`, aggregates, `ORDER BY`, `DISTINCT`, subqueries,
  `AND`/`OR`; `WHERE` supports only `col = value`.
* **DML**: no `DELETE` or `UPDATE`. The heap never reclaims space (no
  `VACUUM`), which is also why no-steal needs no undo log.
* **Indexes**: exactly one, on `id`. No `CREATE INDEX`, no secondary indexes,
  no composite keys.
* **Types**: `INT` and `TEXT` only. `TEXT` ≤ 1024 bytes. No `NULL` values (the
  on-disk format reserves a null bitmap; the API rejects `NULL`).
* **Transactions**: autocommit only. No `BEGIN`/`COMMIT`, no isolation levels,
  no MVCC, no concurrent access.
* **Operational**: unbounded page cache (no buffer-pool eviction); catalog root
  fixed at page 1 by convention rather than a stored pointer; `fsync`
  correctness is assumed.
* **Performance**: not optimised. Python, and deserialise-on-read, by design.

---

## Future improvements

* `DELETE` / `UPDATE` with in-page slot compaction and a free-space map.
* Range predicates (`>`, `<`, `BETWEEN`) served by walking the linked leaves.
* `AND` / `OR` in `WHERE` via a small predicate tree.
* Secondary indexes and `CREATE INDEX`.
* Multi-statement `BEGIN` / `COMMIT` — the pager's transaction machinery already
  supports it; only the grammar and a session flag are missing.
* A bounded buffer pool with LRU eviction.
* Zero-copy node access (binary-search in the page bytes, mutate in place).
* `page_lsn` on data pages so recovery can skip already-applied writes; group
  commit to amortise `fsync`.

---

## What I learned

* **Page-oriented thinking.** The unit of cost is a page read, not a CPU
  operation. That single idea explains fixed-size pages, why B+ tree nodes are
  sized to the page (fanout, not object count), why the tree stays shallow, and
  why "O(n) vs O(log n)" in a database is really "P page reads vs height page
  reads".
* **Indirection earns its keep.** The index stores a row id, not the row.
  That's one copy of the data, stable under in-page moves, and the thing that
  would let multiple indexes coexist.
* **Atomicity is a property you design in, not bolt on.** Making `CREATE TABLE`
  atomic in Stage 9 wasn't new recovery code — it was drawing the transaction
  boundary around work that already funnelled through one `write_page` path.
* **The WAL rule is small; the reasoning is the hard part.** "Log before data,
  fsync the log first" is one sentence. Understanding *why* no-steal removes the
  undo path, why full-page images make replay idempotent, and how a torn tail is
  distinguishable from real data — that's where the time went.
* **A clean layer boundary pays compound interest.** Because every write went
  through the pager, the entire WAL landed in `pager.py` + `wal.py` +
  a `wal=` flag, with `executor.py` untouched.
* **Tests as the spec.** The B+ tree invariant checker (all leaves same depth,
  separators equal the min of their right subtree, leaf chain globally sorted)
  caught more split bugs than any amount of staring at `_split_internal`.

---

## Project layout

```
minidb/
├── constants.py   errors.py
├── pager.py       wal.py           # file I/O, cache, transactions, recovery
├── page.py        record.py        # slotted pages, row codec
├── heap.py        index.py         # heap file, B+ tree
├── table.py       catalog.py       # table = heap + index; metadata
├── query.py       executor.py      # typed queries, execution
├── sql/           repl.py          # tokenizer + parser; shell
└── database.py    __main__.py

tests/    16 files, 603 tests
bench/    benchmark.py
examples/ demo.sql
```

## How it was built

Incrementally, in ten reviewed stages, each independently runnable and tested
before the next began:

1. Pager + persistent storage
2. Record serialization + heap storage
3. Catalog + multiple tables
4. Sequential `SELECT` (+ the pre-index baseline)
5. In-memory B+ tree
6. Persistent B+ tree index
7. SQL tokenizer + parser
8. Query executor + REPL
9. Write-ahead log + crash recovery
10. Benchmarking + this README
