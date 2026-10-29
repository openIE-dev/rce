#!/usr/bin/env python3
"""Write `trace_lines`, the whole trace, into every `expect: trace` test, as
the reference adapter (rce-conform, $RCE_CONFORM) prints it for the test's
program and events. The head the reference gives must equal the test's
`trace_head`, and the lines must keep the hash chain of 6.1; a test where
either fails is reported and left as it is (exit 1).
  RCE_CONFORM=/path/to/rce-conform python3 tools/trace_lines.py
"""
import os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, os.path.dirname(__file__))
import rcetest
from refclient import Ref
from run import chain_law

ref, written, kept = Ref(), 0, []
for p in sorted((ROOT / "tests").rglob("*.rce")):
    if "_FIXTURE" in p.name:
        continue
    src, meta = rcetest.split(p.read_text(encoding="utf-8"))
    if meta.get("expect") != "trace":
        continue
    r = ref.trace(src, meta.get("events"))
    tr = r.get("trace") or {}
    head, lines = tr.get("head"), tr.get("lines")
    why = r.get("error") or (None if isinstance(lines, list) else "no lines") or chain_law(lines, head)
    if not why and head != meta.get("trace_head"):
        why = f"reference head {head} != trace_head {meta.get('trace_head')}"
    if why:
        kept.append(f"{meta.get('id')}: {why}")
        continue
    if meta.get("trace_lines") != lines:
        meta["trace_lines"] = lines
        rcetest.write(p, src, meta)
        written += 1
for k in kept:
    print(f"KEPT {k}")
print(f"trace_lines written in {written} tests; {len(kept)} kept unchanged")
sys.exit(1 if kept else 0)
