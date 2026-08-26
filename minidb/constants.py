"""On-disk format constants for MiniDB.

Every value here describes bytes that end up in the database file. Changing
one changes the file format, which is why the file header carries a
``format_version`` (see ``pager.py``): a future change can then be *detected*
instead of silently misread.
"""

# The database file is an array of fixed-size pages. The page size is picked
# when a brand-new database is created and then recorded in the file header;
# from then on the stored value is authoritative and this default no longer
# matters for that file. 4096 matches the usual OS memory page and SSD I/O
# unit. Stage 10 re-runs the benchmarks with 1024 / 4096 / 8192.
DEFAULT_PAGE_SIZE = 4096

# Sanity bounds on page_size. Not a deep design decision - just a guard so an
# obviously wrong value (0, negative, gigabytes) fails at open() instead of
# producing a strange file. A page size is conventionally a power of two for
# alignment with OS pages and device sectors, but that is not enforced.
MIN_PAGE_SIZE = 512
# Upper bound is 2**15. Record offsets in a page's slot directory and the
# free-space pointers in a page header are unsigned 16-bit, so everything in
# a page must be addressable in 16 bits. Real page-oriented databases cap
# page size for the same reason (SQLite's limit is 65536). 32768 is far
# above anything MiniDB benchmarks (1K / 4K / 8K).
MAX_PAGE_SIZE = 1 << 15

# First 8 bytes of the file: ASCII "MINIDB" plus two zero pad bytes. Lets us
# reject "this is not a MiniDB file" before trusting any other byte.
MAGIC = b"MINIDB\x00\x00"

# Bumped whenever the *pager's* on-disk layout changes incompatibly (the file
# header or the page header). The catalog is a higher layer and needed no
# bump; the next is likely Stage 9 (a per-page LSN for the WAL).
FORMAT_VERSION = 1

# Page 0 is reserved for the file header and is owned by the pager. Every
# other component is handed page ids starting at 1.
HEADER_PAGE_ID = 0
FIRST_DATA_PAGE_ID = 1

# The catalog (table metadata) is rooted at a fixed page, found by convention
# rather than a stored pointer - the same choice SQLite makes for its page 1.
# Because Database.open() creates the catalog before any table can be made,
# the catalog always wins the first allocation, which is page 1. Stage 3.
CATALOG_ROOT_PAGE_ID = 1


# --- record encoding (Stage 2) -----------------------------------------------

# A stored INT is a signed 64-bit little-endian integer.
INT_MIN = -(1 << 63)
INT_MAX = (1 << 63) - 1

# A stored TEXT value is a 2-byte unsigned length prefix followed by that many
# UTF-8 bytes. The format allows up to 65535 bytes; MiniDB caps it far lower
# so a row stays small relative to a page. This is a policy limit, not a
# format limit - the hard "a row must fit in an empty page" guarantee is
# enforced by the heap layer against the real page size.
MAX_TEXT_BYTES = 1024
