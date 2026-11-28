"""Seeded test pools for the fault-localisation task (one pool per mutation subject).

The pools play the role of a project's regression test suite: calls with
structured, domain-specific inputs (texts with tabs, hyphens and sentence
ends; URLs with every component; fraction literals, operators and format
specs). They respect the argument kinds of software_pymod.MUTATION, so the
same calls are valid tests of the mutation tasks.
"""
from __future__ import annotations

import random
import zlib

POOL_SIZE = 240
WORDS = ["alpha", "be", "c", "delta-echo", "foxtrot.", "Golf!", "hotel?", "in", "juliet", "kilo-", "lima", "x" * 14,
         "Mr.", "end.", "tab\there", "two  spaces", "", "\n", "unié", "a-b-c"]
SCHEMES = ["http", "https", "ftp", "file", "mailto", "svn+ssh", "", "HTTP", "data", "urn"]
HOSTS = ["example.com", "user:pw@host:8080", "[::1]:80", "[fe80::1%25en0]", "h", "a.b.c:", "@x", "host:abc", "xn--n3h.net"]
PATHS = ["", "/", "/a/b/c", "/a/./b/../c", "a/b", "//x", "/%7Euser/file.txt", "/semi;param", "/sp ace", "/../.."]
QUERIES = ["", "?", "?a=1&b=2", "?a=1;b=2", "?x=%20y+z", "?k&=v&&", "?q=a%3Db", "?%zz=1"]
FRAGMENTS = ["", "#", "#frag", "#a#b", "#%41"]
RELATIVE = ["g", "./g", "g/", "/g", "//g", "?y", "g?y", "#s", "g#s", "..", "../g", "../../../g", ";x", "g;x?y#s", "", ".", "http:g"]
NUMBERS = ["0", "1", "-1", "3/4", "-7/2", "1.5", ".5", "5.", "1e3", "-2.5e-3", "1_000", " 22/7 ", "0/1", "1/0", "abc", "1/2/3",
           "+3", "-0", "1e-50", "3.14159", "007", "1__0", "2E+2", "1 / 2", "inf"]
SPECS = ["", "f", ".3f", "e", ".2e", "g", "%", ".1%", "10.3f", "<12", ">12.2f", "^9", "+", " .4g", "_", ",.2f", "x", "#.0f", "=+10", "08.3f"]


def _text(rng):
    words = [rng.choice(WORDS) for _ in range(rng.randint(0, 14))]
    joiner = rng.choice([" ", "  ", "\t", " \n "])
    return joiner.join(words)[:500]


def _url(rng):
    scheme = rng.choice(SCHEMES)
    netloc = "//" + rng.choice(HOSTS) if rng.random() < 0.8 else ""
    url = (scheme + ":" if scheme else "") + netloc + rng.choice(PATHS) + rng.choice(QUERIES) + rng.choice(FRAGMENTS)
    return url[:500]


def _query(rng):
    pairs = [rng.choice(["a", "b", "k+w", "%41", ""]) + rng.choice(["=", "", "=="]) + rng.choice(["1", "x+y", "%20", "", "a=b"])
             for _ in range(rng.randint(0, 5))]
    return rng.choice(["&", ";", "&&"]).join(pairs)


def textwrap_test(rng):
    call = rng.choice(["wrap", "wrap", "fill", "shorten", "dedent", "indent"])
    if call == "dedent":
        lines = [rng.choice(["", " ", "  ", "\t", "    "]) + rng.choice(WORDS) for _ in range(rng.randint(1, 5))]
        return {"call": call, "args": ["\n".join(lines)]}
    if call == "indent":
        return {"call": call, "args": ["\n".join(_text(rng) for _ in range(rng.randint(1, 3))), rng.choice(["  ", "> ", "", "\t"])]}
    options = {}
    for key, values in (("break_long_words", [True, False]), ("break_on_hyphens", [True, False]), ("drop_whitespace", [True, False]),
                        ("expand_tabs", [True, False]), ("replace_whitespace", [True, False]), ("fix_sentence_endings", [True, False]),
                        ("tabsize", [1, 4, 8]), ("initial_indent", ["", "* ", "    "]), ("subsequent_indent", ["", "  "]),
                        ("placeholder", [" [...]", "~", ""]), ("max_lines", [None, 1, 2, 3])):
        if rng.random() < 0.3 and not (call == "shorten" and key == "max_lines"):
            options[key] = rng.choice(values)
    return {"call": call, "args": [_text(rng), rng.choice([1, 2, 5, 8, 10, 12, 20, 40, 70]), options]}


def urllib_test(rng):
    call = rng.choice(["urlparse", "urlsplit", "urljoin", "urljoin", "urldefrag", "parse_qs", "parse_qsl", "quote", "quote_plus",
                       "unquote", "unquote_plus", "urlencode", "urlunsplit", "urlunparse", "unwrap"])
    if call in ("urlparse", "urlsplit"):
        return {"call": call, "args": [_url(rng), rng.choice(["", "http", "ftp"]), rng.random() < 0.7]}
    if call == "urljoin":
        base = rng.choice(["http://a/b/c/d;p?q", _url(rng), "", "file:///tmp/x"])
        return {"call": call, "args": [base, rng.choice(RELATIVE + [_url(rng)]), rng.random() < 0.8]}
    if call in ("parse_qs", "parse_qsl"):
        return {"call": call, "args": [_query(rng), rng.random() < 0.5, rng.random() < 0.2, rng.choice([None, None, 2, 10]), rng.choice(["&", ";", "&", ""])]}
    if call in ("quote", "quote_plus"):
        return {"call": call, "args": [rng.choice([_url(rng), "a b&c=d/e~f", "é€", "%41", ""]), rng.choice(["/", "", "&=", "~"])]}
    if call in ("unquote", "unquote_plus"):
        return {"call": call, "args": [rng.choice(["%41%42", "a+b%20c", "%e2%82%ac", "%zz", "%", "%4", "plain", _url(rng)])]}
    if call == "urlencode":
        pairs = [[rng.choice(["a", "b c", "é", "k=v"]), rng.choice(["1", "x y", "&", ""])] for _ in range(rng.randint(0, 4))]
        return {"call": call, "args": [pairs, rng.random() < 0.5, rng.choice(["", "/", " "])]}
    if call == "urlunsplit":
        return {"call": call, "args": [[rng.choice(SCHEMES), rng.choice(HOSTS + [""]), rng.choice(PATHS), rng.choice(["", "a=1"]), rng.choice(["", "f"])]]}
    if call == "urlunparse":
        return {"call": call, "args": [[rng.choice(SCHEMES), rng.choice(HOSTS + [""]), rng.choice(PATHS), rng.choice(["", "p"]), rng.choice(["", "a=1"]), rng.choice(["", "f"])]]}
    return {"call": call, "args": [rng.choice(["<URL:http://x/>", "URL:a", " <b> ", _url(rng)])]}


def fractions_test(rng):
    call = rng.choice(["parse", "parse", "make", "from_float", "binary", "binary", "binary", "limit_denominator", "round", "format", "format", "method"])
    number = lambda: rng.choice(NUMBERS) if rng.random() < 0.6 else rng.choice([rng.randint(-9, 9), rng.choice([0.5, -1.25, 1e-3, 3.0])])
    if call == "parse":
        return {"call": call, "args": [rng.choice(NUMBERS)]}
    if call == "make":
        return {"call": call, "args": [rng.randint(-50, 50), rng.choice([rng.randint(-12, 12), 0])]}
    if call == "from_float":
        return {"call": call, "args": [rng.choice([0.1, -2.5, 1e-7, 123456.75, 0.0, 1.0 / 3])]}
    if call == "binary":
        op = rng.choice(["+", "-", "*", "/", "//", "%", "**", "<", "<=", "==", "divmod", "max"])
        right = rng.choice([rng.randint(-3, 3), "2", "1/2", "-3/4", 0.5]) if op == "**" else number()
        return {"call": call, "args": [number(), op, right]}
    if call == "limit_denominator":
        return {"call": call, "args": [rng.choice(NUMBERS[:12] + ["3.14159", "0.333"]), rng.choice([1, 2, 7, 10, 100, 0])]}
    if call == "round":
        return {"call": call, "args": [rng.choice(NUMBERS[:12] + ["5/2", "-5/2", "7/2"]), rng.choice([None, 0, 1, 2, -1])]}
    if call == "format":
        return {"call": call, "args": [rng.choice(NUMBERS[:12] + ["2/3", "-1/7", "1e10"]), rng.choice(SPECS)]}
    return {"call": call, "args": [rng.choice(NUMBERS[:12] + ["7/3", "-7/3"]), rng.choice(["as_integer_ratio", "is_integer", "__floor__", "__ceil__",
                                                                                         "__trunc__", "__abs__", "__neg__", "__pos__", "__float__",
                                                                                         "__int__", "__bool__", "__hash__", "__str__"])]}


GENERATORS = {"textwrap": textwrap_test, "urllib.parse": urllib_test, "fractions": fractions_test}


def pool(subject, size=POOL_SIZE):
    rng = random.Random(zlib.crc32(subject.encode()))
    return [GENERATORS[subject](rng) for _ in range(size)]
