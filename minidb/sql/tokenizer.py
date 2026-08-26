"""Stage 7 - the SQL tokenizer.

One linear pass over the source string, producing a flat list of ``Token``
objects ending in EOF. It has no grammar knowledge - it cannot tell a SELECT
from an INSERT, only how to split text into keywords, identifiers, integer
and string literals, and punctuation.

Rules:

* whitespace (space, tab, CR, LF) separates tokens and is otherwise ignored;
* keywords are case-insensitive (normalised to upper case);
* identifiers are case-sensitive: ``[A-Za-z_][A-Za-z0-9_]*``;
* integers: an optional ``-`` immediately followed by one or more digits;
* strings: single-quoted; ``''`` inside a string is one literal quote;
* punctuation: ``( ) , ; *`` ; comparison: ``= < >`` (only ``=`` is accepted
  by the parser, but ``< >`` tokenise so the parser can say so nicely);
* ``--`` starts a comment that runs to the end of the line.

No other operators, no double-quoted identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import MiniDBError

KEYWORDS = frozenset(
    {
        "SELECT", "FROM", "WHERE", "CREATE", "TABLE",
        "INSERT", "INTO", "VALUES", "INT", "TEXT",
    }
)

_PUNCT = frozenset("(),;*")
_DIGITS = frozenset("0123456789")
_IDENT_START = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
)
_IDENT_CONT = _IDENT_START | _DIGITS


@dataclass(frozen=True)
class Token:
    # kind is: a keyword ("SELECT", ...), "IDENT", "INTEGER", "STRING",
    # "OP" (= < >), a punctuation char ("(", ")", ",", ";", "*"), or "EOF".
    kind: str
    value: object
    pos: int

    def describe(self) -> str:
        if self.kind == "EOF":
            return "end of input"
        if self.kind == "STRING":
            return f"string {self.value!r}"
        if self.kind == "INTEGER":
            return f"number {self.value}"
        if self.kind == "IDENT":
            return f"identifier {self.value!r}"
        return repr(self.value)


def tokenize(sql: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":  # line comment
                i += 1
            continue
        start = i
        if c in _PUNCT:
            tokens.append(Token(c, c, start))
            i += 1
        elif c in "=<>":
            tokens.append(Token("OP", c, start))
            i += 1
        elif c == "'":
            value, i = _read_string(sql, i)
            tokens.append(Token("STRING", value, start))
        elif c in _DIGITS or (c == "-" and i + 1 < n and sql[i + 1] in _DIGITS):
            j = i + 1
            while j < n and sql[j] in _DIGITS:
                j += 1
            tokens.append(Token("INTEGER", int(sql[i:j]), start))
            i = j
        elif c in _IDENT_START:
            j = i + 1
            while j < n and sql[j] in _IDENT_CONT:
                j += 1
            word = sql[i:j]
            upper = word.upper()
            if upper in KEYWORDS:
                tokens.append(Token(upper, upper, start))
            else:
                tokens.append(Token("IDENT", word, start))
            i = j
        else:
            raise MiniDBError(
                f"SQL error at position {i}: unexpected character {c!r}"
            )
    tokens.append(Token("EOF", None, n))
    return tokens


def _read_string(sql: str, i: int) -> tuple[str, int]:
    """Read a single-quoted string starting at ``sql[i] == \"'\"``.

    Returns ``(value, next_index)``. ``''`` is an escaped single quote.
    """
    n = len(sql)
    j = i + 1
    out: list[str] = []
    while j < n:
        c = sql[j]
        if c == "'":
            if j + 1 < n and sql[j + 1] == "'":
                out.append("'")
                j += 2
                continue
            return "".join(out), j + 1
        out.append(c)
        j += 1
    raise MiniDBError(
        f"SQL error at position {i}: unterminated string literal"
    )
