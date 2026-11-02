#!/usr/bin/env python3
"""E5's deterministic renamer: the same C or Rust program with every identifier it binds renamed.

The renamer reads a source text as a sequence of lexical items (identifiers,
numbers, string and character literals, comments, preprocessor lines and
Rust attributes) and replaces every identifier that the program binds by a
new name. It keeps (PROTOCOL.md, "Renamed sources"):

- the language's keywords, preprocessor directive names and built-in type
  names (KEEP_C, KEEP_RUST);
- the names of the program's interface: `run` (the kernel every driver
  calls) and `main`;
- names the program takes from a library: the library names of KEEP_C and
  KEEP_RUST, and every identifier that occurs, anywhere in the file, in a
  position that library_names() lists (after `.`; in Rust also next to
  `::`, before `!`, and captured by a format string), so that a method the
  file defines and calls by `.` keeps one name;
- the text of comments, literals, `#include` lines and Rust attributes, which
  are copied unchanged, except that comments are removed (a comment does not
  change what a program means).

Every other identifier is renamed. The new name of the k-th distinct
identifier in order of first appearance is `n<k>_` followed by the first 8
hexadecimal digits of the SHA-256 of the old name, so the renaming is a
function of the source text alone. A new name that already occurs in the
source is an error.

    python3 rename.py c|rust SOURCE
"""
import hashlib
import re
import sys

C_KEYWORDS = """auto break case char const continue default do double else enum extern float for goto if inline int
long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while
_Bool _Complex _Static_assert bool true false define undef ifdef ifndef elif endif pragma""".split()
C_LIBRARY = """uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t size_t uintptr_t intptr_t NULL
printf fprintf stderr stdout strtoul qsort exit puts malloc free memset memcpy""".split()
KEEP_C = set(C_KEYWORDS) | set(C_LIBRARY) | {"run", "main"}

RUST_KEYWORDS = """as break const continue crate else enum extern false fn for if impl in let loop match mod move mut
pub ref return self Self static struct super trait true type unsafe use where while dyn async await""".split()
RUST_LIBRARY = """u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64 bool char str String Vec Option Some None
Ok Err Result Box BTreeMap HashMap""".split()
KEEP_RUST = set(RUST_KEYWORDS) | set(RUST_LIBRARY) | {"run", "main", "_"}

ITEM = re.compile(r"""
    (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<pre>^[ \t]*\#[ \t]*include[^\n]*)
  | (?P<attr>\#!?\[(?:[^\[\]]|\[[^\[\]]*\])*\])
  | (?P<string>"(?:\\.|[^"\\])*")
  | (?P<char>'(?:\\.|[^'\\])')
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<number>[0-9][0-9A-Za-z_]*(?:\.[0-9][0-9A-Za-z_]*)?)
  | (?P<other>.)
""", re.S | re.M | re.X)


def lex(text):
    return [(m.lastgroup, m.group()) for m in ITEM.finditer(text)]


def library_names(lang, items):
    """Identifiers in a position that names something outside the program's own bindings.

    After `.` (or `->` in C): a field or method name. In Rust also: next to `::` (a path segment), before `!` (a
    macro), and inside `{...}` of a string literal (a name a format string captures). Such an identifier is kept
    everywhere in the file, so that a method defined in the file and called by `.` keeps one name.
    """
    sig = [(k, v) for k, v in items if not (k == "other" and v.isspace())]
    out = set()
    for i, (k, v) in enumerate(sig):
        if k == "string" and lang == "rust":
            out.update(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)", v))
        if k != "ident":
            continue
        p1 = sig[i - 1][1] if i >= 1 else ""
        p2 = sig[i - 2][1] if i >= 2 else ""
        n1 = sig[i + 1][1] if i + 1 < len(sig) else ""
        n2 = sig[i + 2][1] if i + 2 < len(sig) else ""
        if (p1 == "." and p2 != ".") or (lang == "c" and p1 == ">" and p2 == "-"):
            out.add(v)
        elif lang == "rust" and ((p1 == ":" and p2 == ":") or (n1 == ":" and n2 == ":") or n1 == "!"):
            out.add(v)
    return out


def rename(lang, text):
    """Return (renamed text, {old: new}) for a C ("c") or Rust ("rust") source text."""
    if lang not in ("c", "rust"):
        raise ValueError("language must be c or rust")
    items = lex(text)
    keep = (KEEP_C if lang == "c" else KEEP_RUST) | library_names(lang, items)
    present = {v for k, v in items if k == "ident"}
    mapping = {}
    out = []
    for kind, val in items:
        if kind == "comment":
            out.append(" " if val.startswith("/*") else "")
            continue
        if kind != "ident" or val in keep:
            out.append(val)
            continue
        if val not in mapping:
            new = "n%d_%s" % (len(mapping), hashlib.sha256(val.encode()).hexdigest()[:8])
            if new in present:
                raise ValueError("new name %s already occurs in the source" % new)
            mapping[val] = new
        out.append(mapping[val])
    return "".join(out), mapping


def main():
    a = sys.argv[1:]
    if len(a) != 2:
        sys.exit("usage: python3 rename.py c|rust SOURCE")
    text, mapping = rename(a[0], open(a[1], encoding="utf-8").read())
    sys.stdout.write(text)
    sys.stderr.write("".join("%s -> %s\n" % kv for kv in mapping.items()))


if __name__ == "__main__":
    main()
