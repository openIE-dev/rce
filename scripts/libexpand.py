#!/usr/bin/env python3
"""Expand the `#!lib crypto` directive for an rce that does not provide it.

Reads a module on stdin and writes it to stdout. With RCE_NO_LIB unset or
empty the text passes through unchanged (the rce binary expands the
directive itself). With RCE_NO_LIB=1 each `#!lib crypto` line is removed and
the text of lib/crypto.rce, without its `module` line and its `check` lines
(the binary appends crypto without crypto's own checks), is appended.
"""
import os, pathlib, sys

src = sys.stdin.read()
if os.environ.get("RCE_NO_LIB") != "1" or "#!lib crypto" not in src:
    sys.stdout.write(src)
    sys.exit(0)
crypto = pathlib.Path(__file__).resolve().parent.parent / "lib" / "crypto.rce"
body = [l for l in crypto.read_text().splitlines() if not l.startswith("module ") and not l.startswith("check ")]
out = [l for l in src.splitlines() if l.strip() != "#!lib crypto"]
sys.stdout.write("\n".join(out) + "\n\n" + "\n".join(body) + "\n")
