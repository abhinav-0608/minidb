"""Stage 7 tests - the SQL tokenizer.

* TestBasicTokens - keywords (case-insensitive), identifiers (case-kept),
                    punctuation, operators, EOF
* TestIntegers    - plain and negative integer literals
* TestStrings     - single quotes, '' escaping, unterminated
* TestWhitespace  - ignored between tokens, not required between them
* TestPositions   - every token's pos points at its start
* TestLexErrors   - stray characters, double-quoted identifiers
* TestDescribe    - Token.describe (used in parser error messages)
"""

from __future__ import annotations

import pytest

from minidb.errors import MiniDBError
from minidb.sql.tokenizer import Token, tokenize


def pairs(sql: str):
    return [(t.kind, t.value) for t in tokenize(sql)]


def kinds(sql: str):
    return [t.kind for t in tokenize(sql)]


EOF = ("EOF", None)


# ---------------------------------------------------------------------------
class TestBasicTokens:
    def test_empty_input_is_just_eof(self):
        assert pairs("") == [EOF]

    def test_whitespace_only_is_just_eof(self):
        assert pairs("  \t\r\n ") == [EOF]

    @pytest.mark.parametrize(
        "kw",
        ["SELECT", "FROM", "WHERE", "CREATE", "TABLE",
         "INSERT", "INTO", "VALUES", "INT", "TEXT"],
    )
    def test_keywords_are_case_insensitive(self, kw):
        assert pairs(kw) == [(kw, kw), EOF]
        assert pairs(kw.lower()) == [(kw, kw), EOF]
        assert pairs(kw.title()) == [(kw, kw), EOF]

    def test_identifier_keeps_its_case(self):
        assert pairs("Name") == [("IDENT", "Name"), EOF]
        assert pairs("_x1") == [("IDENT", "_x1"), EOF]
        assert pairs("USERS") == [("IDENT", "USERS"), EOF]

    def test_a_keyword_prefix_is_still_an_identifier(self):
        assert pairs("selected") == [("IDENT", "selected"), EOF]
        assert pairs("into_the") == [("IDENT", "into_the"), EOF]

    @pytest.mark.parametrize("ch", list("(),;*"))
    def test_punctuation(self, ch):
        assert pairs(ch) == [(ch, ch), EOF]

    @pytest.mark.parametrize("ch", list("=<>"))
    def test_operators(self, ch):
        assert pairs(ch) == [("OP", ch), EOF]

    def test_eof_is_always_last_at_end_position(self):
        toks = tokenize("SELECT x")
        assert toks[-1].kind == "EOF"
        assert toks[-1].pos == len("SELECT x")


# ---------------------------------------------------------------------------
class TestIntegers:
    @pytest.mark.parametrize(
        "src,val",
        [("42", 42), ("0", 0), ("007", 7), ("-7", -7), ("-0", 0),
         ("9223372036854775807", 2**63 - 1)],
    )
    def test_integer_values(self, src, val):
        assert pairs(src) == [("INTEGER", val), EOF]

    @pytest.mark.parametrize("src", ["-", "- 7", "-x", "-'a'"])
    def test_minus_not_followed_by_a_digit_is_an_error(self, src):
        with pytest.raises(MiniDBError):
            tokenize(src)

    def test_integer_adjacent_to_punctuation(self):
        assert kinds("(1,2)") == ["(", "INTEGER", ",", "INTEGER", ")", "EOF"]


# ---------------------------------------------------------------------------
class TestStrings:
    @pytest.mark.parametrize(
        "src,val",
        [
            ("'abc'", "abc"),
            ("''", ""),
            ("'it''s'", "it's"),
            ("'a''''b'", "a''b"),
            ("'has, punct; ()='", "has, punct; ()="),
            ("'line1\nline2'", "line1\nline2"),
            ("'  spaced  '", "  spaced  "),
        ],
    )
    def test_string_values(self, src, val):
        assert pairs(src) == [("STRING", val), EOF]

    def test_string_boundary_with_following_tokens(self):
        assert pairs("'x'y") == [("STRING", "x"), ("IDENT", "y"), EOF]
        assert pairs("'a' 'b'") == [("STRING", "a"), ("STRING", "b"), EOF]

    @pytest.mark.parametrize("src", ["'abc", "'", "'abc''", "'a''b"])
    def test_unterminated_string_is_an_error(self, src):
        with pytest.raises(MiniDBError):
            tokenize(src)


# ---------------------------------------------------------------------------
class TestWhitespace:
    def test_whitespace_between_tokens_is_ignored(self):
        assert kinds("  SELECT\n\t*\r\n FROM     t  ") == kinds("SELECT * FROM t")

    def test_tokens_need_no_separator(self):
        assert kinds("SELECT*FROM t") == ["SELECT", "*", "FROM", "IDENT", "EOF"]
        assert kinds("(1,'a',-2)") == [
            "(", "INTEGER", ",", "STRING", ",", "INTEGER", ")", "EOF"
        ]


# ---------------------------------------------------------------------------
class TestPositions:
    def test_positions_point_at_token_start(self):
        toks = tokenize("SELECT   name , x")
        assert [(t.kind, t.pos) for t in toks] == [
            ("SELECT", 0), ("IDENT", 9), (",", 14), ("IDENT", 16), ("EOF", 17)
        ]

    def test_string_position_is_the_opening_quote(self):
        toks = tokenize("  'hi'")
        assert (toks[0].kind, toks[0].pos) == ("STRING", 2)

    def test_negative_integer_position_is_the_minus(self):
        toks = tokenize("x = -5")
        assert (toks[2].kind, toks[2].value, toks[2].pos) == ("INTEGER", -5, 4)


# ---------------------------------------------------------------------------
class TestLexErrors:
    @pytest.mark.parametrize(
        "bad", ['"', "~", "@", "#", "&", "|", ".", "!", "%", "^", "?", ":", "/", "+"]
    )
    def test_unexpected_character(self, bad):
        with pytest.raises(MiniDBError) as exc:
            tokenize(f"SELECT {bad}")
        assert "position" in str(exc.value)

    def test_double_quoted_identifier_is_rejected(self):
        with pytest.raises(MiniDBError):
            tokenize('SELECT "col" FROM t')


# ---------------------------------------------------------------------------
class TestComments:
    def test_line_comment_is_skipped(self):
        assert kinds("SELECT * -- ignored\nFROM t") == kinds("SELECT * FROM t")

    def test_comment_running_to_end_of_input(self):
        assert pairs("SELECT -- trailing, no newline") == [("SELECT", "SELECT"), EOF]

    def test_whole_line_comment(self):
        assert pairs("-- nothing here\n") == [EOF]

    def test_tokens_resume_after_the_comment_line(self):
        assert kinds("a -- c1\nb -- c2\nc") == ["IDENT", "IDENT", "IDENT", "EOF"]

    def test_double_dash_inside_a_string_is_literal(self):
        assert pairs("'a -- b'") == [("STRING", "a -- b"), EOF]

    def test_a_lone_minus_still_errors(self):
        with pytest.raises(MiniDBError):
            tokenize("a - b")

    def test_negative_integer_still_tokenises(self):
        assert pairs("-5") == [("INTEGER", -5), EOF]


# ---------------------------------------------------------------------------
class TestDescribe:
    def test_describe(self):
        assert Token("EOF", None, 0).describe() == "end of input"
        assert Token("STRING", "hi", 0).describe() == "string 'hi'"
        assert Token("INTEGER", 5, 0).describe() == "number 5"
        assert Token("IDENT", "x", 0).describe() == "identifier 'x'"
        assert Token(";", ";", 0).describe() == "';'"
