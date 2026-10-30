#!/usr/bin/env python3
"""Fail when a tracked text file names a path that git history removed.

A file deleted or renamed in an earlier commit is a retired path. Any
mention of one (`lib/<name>.rce`, `$here/lib/<name>.rce` after the root
lib/ was removed) points at a version of the repository that no longer
exists, so this check lists every such mention and exits 1. A mention that
resolves to a file that exists now, relative to the mentioning file's
directory or any directory above it (`lib/<name>.rce` inside stdlib/ is
stdlib/lib/<name>.rce), is not stale.

The ledger records past changes and may name what a change removed, so
spec/proposals/LEDGER.md is not checked. Needs the full history (CI checks
out with fetch-depth 0).

    python3 scripts/check-removed-paths.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_FILES = {"spec/proposals/LEDGER.md"}
SKIP_DIRS = ("ide/vendor/", "stdlib/vectors/")
TEXT = (".md", ".rce", ".rs", ".sh", ".py", ".yml", ".yaml", ".toml", ".lean", ".txt", ".js", ".ts")
PATH = re.compile(
    r"(?<![\w.-])(?:\$\{?\w+\}?/|\.\./|\./)*"
    r"((?:[\w-]+/)+[\w.-]+\.(?:rce|rs|md|json|sh|py|yml|toml|lean|txt))(?![\w/])"
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main():
    if git("rev-parse", "--is-shallow-repository").strip() == "true":
        sys.exit("check-removed-paths: shallow clone; fetch the full history first")
    files = [f for f in git("ls-files").split("\n") if f]
    tracked = set(files)
    dirs = set()
    for f in files:
        d = os.path.dirname(f)
        while d:
            dirs.add(d)
            d = os.path.dirname(d)
    retired = set()
    for line in git("log", "--diff-filter=DR", "--name-status", "--pretty=format:").split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0][:1] in "DR":
            retired.add(parts[1])
    retired -= tracked

    def exists_from(d, p):
        while True:
            c = os.path.normpath(os.path.join(d, p))
            if c in tracked or c in dirs:
                return True
            if not d:
                return False
            d = os.path.dirname(d)

    stale = []
    for f in files:
        if f in SKIP_FILES or f.startswith(SKIP_DIRS) or not f.endswith(TEXT):
            continue
        try:
            text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        d = os.path.dirname(f)
        for m in PATH.finditer(text):
            p = m.group(1)
            if p in retired and not exists_from(d, p):
                stale.append(f"{f}:{text.count(chr(10), 0, m.start()) + 1}: {p} was removed")
    for s in stale:
        print(s)
    if stale:
        print(f"{len(stale)} mention(s) of removed paths; name the current path instead", file=sys.stderr)
        sys.exit(1)
    print(f"check-removed-paths: {len(files)} files, no mention of the {len(retired)} removed paths")


if __name__ == "__main__":
    main()
