#!/usr/bin/env python3
"""Fail when a tracked file still gives an RCE command a flag.

spec/proposals/positional-cli.md removed every flag of `rce`, of the other
binaries of spec/checker, of the emitted programs, of the IDE binaries and
of the repository scripts: a command line is `PROGRAM COMMAND-PATH WORD...
OPERAND...` (PC-1) and no argument starts with `-` (PC-6). This scan
(section 4.3) lists every remaining line, with its path and line number,
that has

1. a command line of an RCE binary with an argument that starts with `-`
   (`rce`, `rce-book`, `rce-ide`, `rce-mcp`, ...), in shell, YAML,
   Markdown, Dockerfiles and `.cmd` listings;
2. in Rust, Python and JavaScript, a string literal that starts with `-`
   on a line that runs an RCE binary or an emitted program;
3. any of the removed flag names of sections 1.1 to 1.3, anywhere.

A line is exempt when `scripts/flag-allow.tsv` lists it (path glob, line
pattern, reason): uses by other programs (`lake env lean --run`, `unshare
--net`, cargo, `--version` of rustc, node and python3, ...), the ledger,
the proposal, dated release notes, and the tests and tables that check that
a removed flag exits 2. Exit status 1 when any line remains.

    python3 scripts/check-flags.py
"""
import fnmatch
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOW = os.path.join(ROOT, "scripts", "flag-allow.tsv")

# sections 1.1 to 1.3 of the proposal
REMOVED = (
    "json streams check nameless fuel control depth vectors advisories key log fs blob media wasm net "
    "chan tls peer pty http redact help version no-generate parse-only run trace fuel-trace fuel-batch "
    "link actor-log restart-log restart restart-replay restart-tree event"
).split()
NAMES = re.compile(r"(?<![\w-])--(" + "|".join(sorted(map(re.escape, REMOVED), key=len, reverse=True)) + r")(?![\w-])|(?<![\w-])-[hV](?![\w-])(?=[\s`'\"\])]|$)")
CMDLINE = re.compile(r"\brce(-[a-z-]+)?( [^ |;&]+)* --?[A-Za-z]")
SHELLISH = (".sh", ".yml", ".yaml", ".md", ".cmd", ".bash")
CODE = (".rs", ".py", ".js", ".mjs", ".cjs")
RUNS = re.compile(r"CARGO_BIN_EXE_|\brce\(&\[|\brun\(&\[|\[RCE\b|\"\$RCE\"|\bRCE\b.*\b(run|call|check_output|Popen)\(|cli::main")
DASH_LIT = re.compile(r"[\"']-{1,2}[A-Za-z][\w-]*")
SKIP_DIRS = ("ide/vendor/", "stdlib/vectors/", "lang/corpus/", "tree-sitter/src/")
TEXT = SHELLISH + CODE + (".toml", ".json", ".lean", ".txt", ".rce", ".scala", ".html", ".tsv", "Dockerfile")


def allow_rules():
    rules = []
    for line in open(ALLOW, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        glob, pat = parts[0], parts[1] if len(parts) > 1 else ""
        rules.append((glob, re.compile(pat) if pat else None))
    return rules


def allowed(rules, path, line):
    for glob, pat in rules:
        if fnmatch.fnmatch(path, glob) and (pat is None or pat.search(line)):
            return True
    return False


def main():
    rules = allow_rules()
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split("\n")
    hits = []
    for f in files:
        if not f or f.startswith(SKIP_DIRS) or not f.endswith(TEXT):
            continue
        try:
            text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        shellish = f.endswith(SHELLISH) or os.path.basename(f) == "Dockerfile"
        code = f.endswith(CODE)
        for n, line in enumerate(text.split("\n"), 1):
            found = NAMES.search(line) or (shellish and CMDLINE.search(line)) or (code and RUNS.search(line) and DASH_LIT.search(line))
            if found and not allowed(rules, f, line):
                hits.append(f"{f}:{n}: {line.strip()[:160]}")
    for h in hits:
        print(h)
    print(f"check-flags: {len(hits)} line(s) give an RCE command a flag")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
