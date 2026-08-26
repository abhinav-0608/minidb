"""Stage 7/8 - the SQL parser.

A hand-written recursive-descent parser over the token list: one method per
grammar rule, no backtracking. It emits the typed query objects from
``query.py`` - there is no general AST because the grammar has no nesting.

Grammar (the whole language)::

    script    := (statement ';')* statement? ';'?
    statement := select | insert | create
    select    := SELECT ('*' | ident (',' ident)*) FROM ident (WHERE cond)?
    cond      := ident '=' literal
    insert    := INSERT INTO ident VALUES '(' literal (',' literal)* ')'
    create    := CREATE TABLE ident '(' ident type (',' ident type)* ')'
    type      := INT | TEXT
    literal   := integer | string

The parser checks *syntax* only. Unknown table, wrong column count, the
first-column-must-be-id rule, type mismatches - all of that is the
executor's job (Stage 8).
"""

from __future__ import annotations

from typing import NoReturn

from ..errors import MiniDBError
from ..query import Condition, CreateTableQuery, InsertQuery, SelectQuery
from ..record import Column, ColumnType
from .tokenizer import Token, tokenize


def parse(sql: str):
    """Parse exactly one SQL statement into a typed query object."""
    p = _Parser(tokenize(sql))
    query = p._statement()
    if p._check(";"):
        p._advance()
    if not p._check("EOF"):
        p._error("unexpected trailing input", p._peek())
    return query


def parse_script(sql: str) -> list:
    """Parse a sequence of ';'-separated statements into query objects.

    Blank statements (stray or doubled ';') are ignored; a trailing ';' is
    optional. Returns [] for an empty script.
    """
    p = _Parser(tokenize(sql))
    out: list = []
    while True:
        while p._check(";"):
            p._advance()
        if p._check("EOF"):
            return out
        out.append(p._statement())
        if not p._check("EOF"):
            p._expect(";", "';' after a statement")


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # -- token helpers ---------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.kind != "EOF":
            self._pos += 1
        return tok

    def _check(self, kind: str) -> bool:
        return self._peek().kind == kind

    def _expect(self, kind: str, what: str | None = None) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            self._error(f"expected {what or repr(kind)}", tok)
        return self._advance()

    def _error(self, msg: str, tok: Token) -> NoReturn:
        raise MiniDBError(
            f"SQL error at position {tok.pos}: {msg}, got {tok.describe()}"
        )

    # -- grammar -------------------------------------------------------------

    def _statement(self):
        tok = self._peek()
        if tok.kind == "SELECT":
            return self._select()
        if tok.kind == "INSERT":
            return self._insert()
        if tok.kind == "CREATE":
            return self._create()
        if tok.kind == "EOF":
            self._error("empty statement", tok)
        self._error("expected SELECT, INSERT or CREATE", tok)

    def _select(self) -> SelectQuery:
        self._expect("SELECT")
        if self._check("*"):
            self._advance()
            columns: tuple[str, ...] | None = None
        else:
            names = [self._expect("IDENT", "a column name").value]
            while self._check(","):
                self._advance()
                names.append(self._expect("IDENT", "a column name").value)
            columns = tuple(names)

        self._expect("FROM")
        table = self._expect("IDENT", "a table name").value

        condition = None
        if self._check("WHERE"):
            self._advance()
            col = self._expect("IDENT", "a column name").value
            op = self._expect("OP", "'='")
            if op.value != "=":
                self._error(
                    f"only '=' is supported in WHERE (not {op.value!r})", op
                )
            condition = Condition(col, "=", self._literal())

        return SelectQuery(table, columns, condition)

    def _insert(self) -> InsertQuery:
        self._expect("INSERT")
        self._expect("INTO")
        table = self._expect("IDENT", "a table name").value
        self._expect("VALUES")
        self._expect("(")
        values = [self._literal()]
        while self._check(","):
            self._advance()
            values.append(self._literal())
        self._expect(")")
        return InsertQuery(table, tuple(values))

    def _create(self) -> CreateTableQuery:
        self._expect("CREATE")
        self._expect("TABLE")
        table = self._expect("IDENT", "a table name").value
        self._expect("(")
        columns = [self._column_def()]
        while self._check(","):
            self._advance()
            columns.append(self._column_def())
        self._expect(")")
        return CreateTableQuery(table, tuple(columns))

    def _column_def(self) -> Column:
        name = self._expect("IDENT", "a column name").value
        tok = self._peek()
        if tok.kind == "INT":
            self._advance()
            return Column(name, ColumnType.INT)
        if tok.kind == "TEXT":
            self._advance()
            return Column(name, ColumnType.TEXT)
        self._error("expected a column type (INT or TEXT)", tok)

    def _literal(self):
        tok = self._peek()
        if tok.kind == "INTEGER":
            return self._advance().value
        if tok.kind == "STRING":
            return self._advance().value
        if tok.kind == "IDENT" and str(tok.value).upper() == "NULL":
            self._error("NULL is not supported", tok)
        self._error("expected a value (a number or 'string')", tok)
