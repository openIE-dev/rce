#!/usr/bin/env python3
"""The outcome vocabulary of the repository (COMP-19 of spec/proposals/compliance.md).

Every word for an input outcome, a run outcome, a static report, a claim
verdict or a record verdict is the compliance term of section 6 of the
proposal. The words of the search pattern below occur only in the text that
scripts/vocab-allow.tsv lists: third-party text, dated history, and, while the
migration of 11.1 step 2 is in progress, the lines a later group of that step
rewrites.

usage:
  scripts/vocab.py --check        list every hit outside the allow-list; status 1 if any
  scripts/vocab.py --stats        hits per repository area, allowed and outstanding
  scripts/vocab.py --inventory    hits per area and file (the line inventory)
  scripts/vocab.py --list REASON  the allowed hits whose reason starts with REASON
options:
  --pattern full                  also count `is an error`, the pattern of section 11.2
  --root DIR                      the repository (default: this script's parent)
  --allow FILE                    the allow-list (default: scripts/vocab-allow.tsv)
  --paths-only / --lines-only     restrict to path names or to text lines

A hit is one text line that matches the pattern, ignoring case, as
`git grep -I -i -E -c` counts; a tracked path whose name matches is a hit of
its own (line 0). Binary files are not searched.

The allow-list has one row per line: PATH-GLOB, TAB, LINE-REGEX, TAB, REASON.
PATH-GLOB is a glob over repository paths in which `**` matches any number of
directories and `*` matches within one. LINE-REGEX is a Python regular
expression searched in the line (`*` matches every line); `@path` matches the
path name itself. REASON starts with one of: `third-party`, `history`,
`domain` (a program's own data named in its domain), `migration`,
`pending g2`, `pending g3`, `pending g4`, `pending pr31`.
Lines starting with `#` and blank lines are ignored.
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys
from collections import defaultdict

WORDS = r"refus|reject|not accepted|turned away"
FULL = WORDS + r"|is an error"
REASONS = ("third-party", "history", "domain", "migration", "pending g2", "pending g3", "pending g4", "pending pr31")

# The areas of section 11.2 of the proposal, longest prefix first.
AREAS = [
    "conformance/reports", "conformance", "spec/checker", "spec/book", "spec/RCE-SPEC.md", "spec/lean",
    "spec/fuzz", "spec/proposals", "spec/examples-0.9", "spec/examples", "spec/traces", "spec/conc06b",
    "spec/conc06", "spec/LANGUAGE-REVIEW.md", "spec/wasm-test", "spec/README.md", "spec/lift",
    "spec/data", "spec/checker-wasm", "spec", "ide", "stdlib", "book", "programs", "docs",
    "tree-sitter/src", "tree-sitter", "scripts", "apps", "rcedb", "lib", "playground", "examples",
    ".githooks",
]


def area(path):
    for a in AREAS:
        if path == a or path.startswith(a + "/"):
            return a
    return path.split("/")[0] if "/" in path else path


def glob_re(g):
    """A path glob as a regular expression: `**/` any directories, `*` within one."""
    out, i = "", 0
    while i < len(g):
        if g.startswith("**/", i):
            out += r"(?:[^/]*/)*"
            i += 3
        elif g.startswith("**", i):
            out += r".*"
            i += 2
        elif g[i] == "*":
            out += r"[^/]*"
            i += 1
        elif g[i] == "?":
            out += r"[^/]"
            i += 1
        else:
            out += re.escape(g[i])
            i += 1
    return re.compile(out + r"\Z")


class Allow:
    def __init__(self, path):
        self.rows = []
        if not os.path.exists(path):
            return
        for n, line in enumerate(open(path, encoding="utf-8"), 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                sys.exit(f"{path}:{n}: expected PATH-GLOB, LINE-REGEX and REASON separated by tabs")
            g, rx, why = parts
            if not why.startswith(REASONS):
                sys.exit(f"{path}:{n}: the reason must start with one of {', '.join(REASONS)}")
            kind = "path" if rx == "@path" else ("any" if rx == "*" else "line")
            self.rows.append((glob_re(g), kind, re.compile(rx) if kind == "line" else None, why, n))

    def reason(self, path, text, is_path):
        """The reason of the first row that allows this hit, or None."""
        for g, kind, rx, why, _ in self.rows:
            if not g.match(path):
                continue
            if is_path:
                if kind == "path":
                    return why
            elif kind == "any" or (kind == "line" and rx.search(text)):
                return why
        return None

    def matches(self, path, text):
        """True when a line of PATH is allowed whatever words it holds (used by the migration)."""
        return self.reason(path, text, False) is not None


def hits(root, pattern, paths=True, lines=True):
    """(path, line number, text) for every hit; line 0 is the path name itself."""
    out = []
    if lines:
        p = subprocess.run(["git", "grep", "-n", "-I", "-i", "-E", "--no-color", pattern],
                           cwd=root, capture_output=True, text=True, errors="replace")
        if p.returncode not in (0, 1):
            sys.exit(p.stderr)
        for l in p.stdout.splitlines():
            f, n, t = l.split(":", 2)
            out.append((f, int(n), t))
    if paths:
        p = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True)
        rx = re.compile(pattern, re.I)
        for f in p.stdout.split("\0"):
            if f and rx.search(f) and os.path.exists(os.path.join(root, f)):
                out.append((f, 0, f))
    return out


def main(argv):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--check", action="store_true")
    m.add_argument("--stats", action="store_true")
    m.add_argument("--inventory", action="store_true")
    m.add_argument("--list", metavar="REASON")
    ap.add_argument("--pattern", choices=["words", "full"], default="words")
    ap.add_argument("--root", default=here)
    ap.add_argument("--allow")
    ap.add_argument("--paths-only", action="store_true")
    ap.add_argument("--lines-only", action="store_true")
    a = ap.parse_args(argv)
    allow = Allow(a.allow or os.path.join(a.root, "scripts", "vocab-allow.tsv"))
    pattern = FULL if a.pattern == "full" else WORDS
    hs = hits(a.root, pattern, paths=not a.lines_only, lines=not a.paths_only)
    rows = [(f, n, t, allow.reason(f, t, n == 0)) for f, n, t in hs]

    if a.check:
        bad = [(f, n, t) for f, n, t, why in rows if why is None]
        for f, n, t in bad:
            print(f"{f}: path name" if n == 0 else f"{f}:{n}: {t.strip()}")
        files = len({f for f, _, _ in bad})
        print(f"vocab: {len(bad)} hits outside scripts/vocab-allow.tsv in {files} files"
              if bad else "vocab: every hit is in scripts/vocab-allow.tsv", file=sys.stderr)
        return 1 if bad else 0

    if a.list is not None:
        for f, n, t, why in rows:
            if why is not None and why.startswith(a.list):
                print(f"{f}: path name   [{why}]" if n == 0 else f"{f}:{n}: {t.strip()}   [{why}]")
        return 0

    per = defaultdict(lambda: defaultdict(int))
    files = defaultdict(set)
    perfile = defaultdict(lambda: defaultdict(int))
    for f, n, t, why in rows:
        ar = area(f)
        key = "paths" if n == 0 else "lines"
        per[ar][key] += 1
        if n:
            files[ar].add(f)
        cls = "outstanding" if why is None else next(r for r in REASONS if why.startswith(r))
        per[ar][cls] += 1
        perfile[ar][f] += 1
    cols = ["lines", "paths", "outstanding"] + list(REASONS)
    if a.stats:
        print("area".ljust(26) + "".join(c.rjust(13) for c in ["lines", "files", "paths", "outstanding"] + list(REASONS)))
        tot = defaultdict(int)
        for ar in sorted(per, key=lambda k: (-per[k]["lines"], k)):
            vals = [per[ar]["lines"], len(files[ar]), per[ar]["paths"]] + [per[ar][c] for c in cols[2:]]
            for c, v in zip(["lines", "files", "paths"] + cols[2:], vals):
                tot[c] += v
            print(ar.ljust(26) + "".join(str(v).rjust(13) for v in vals))
        print("total".ljust(26) + "".join(str(tot[c]).rjust(13) for c in ["lines", "files", "paths"] + cols[2:]))
        print("(outstanding and the reason columns count lines and paths together)")
        return 0
    for ar in sorted(per, key=lambda k: (-per[k]["lines"], k)):
        print(f"{ar}: {per[ar]['lines']} lines in {len(files[ar])} files, {per[ar]['paths']} paths, "
              f"{per[ar]['outstanding']} outstanding")
        for f in sorted(perfile[ar], key=lambda k: (-perfile[ar][k], k)):
            print(f"  {perfile[ar][f]:6d}  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
