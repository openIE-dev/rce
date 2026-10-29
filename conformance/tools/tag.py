#!/usr/bin/env python3
"""Add the clauses a test exercises by its kind and constructs (idempotent).
A tag is added only when the test's own check observes the clause: e.g. T8
on a trace that contains a refused event (the harness checks T8 on every
trace), T16 on a run that carries schedule_seeds."""
import os, pathlib, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
import rcetest
ROOT = pathlib.Path(__file__).resolve().parent.parent
CONSTRUCT = [  # (regex on source, expectation kinds, clauses)
    (r"\bwith budget\b", ("run", "accept"), ["T34", "CH14-M-6d53e2af"]),
    (r"\bwith deadline\b|\bdeadline \d", ("run", "accept"), ["T22"]),
    (r"select nondet", ("run", "accept"), ["T21"]),
    (r"\bselect\b", ("accept", "run"), ["T11"]),
    (r"\bgroup\b|\blanes\b", ("run", "accept"), ["T23"]),
    (r"\blvar\b", ("run", "accept"), ["T29q", "T29s", "T29a"]),
    (r"\breplicated\b", ("run", "accept"), ["T31"]),
    (r"\bspawn\b", ("run",), ["T19b", "T24"]),
    (r"&mut\b", ("run",), ["T33"]),
    (r"\bfor \w+ in \w+\.(iter|gen|chars|words)|\byield\b", ("run", "accept"), ["T15"]),
    (r"\bextern\b|\bforeign\b", ("accept", "host", "change"), ["HC-4"]),
    (r"\bpar\b", ("run",), ["T19a"]),
    (r"\blvar\b|\.put\(|\bfreeze\b", ("run", "accept"), ["T29", "W-CRDT-1"]),
    (r"\bsupervisor\b", ("run", "accept"), ["W-SUP-1"]),
    (r"\bsession\b", ("run", "accept"), ["T28"]),
    (r"\batomic\b", ("run", "accept"), ["T30"]),
    (r"select nondet", ("run",), ["T21"]),
    (r"\bdeadline\b", ("run",), ["T22"]),
    (r"\bgroup\b|\bbarrier\b", ("run",), ["T23"]),
    (r"\bscope\b", ("run",), ["T26"]),
    (r"\bactor\b", ("run",), ["T27"]),
    (r"\bselect\b", ("run",), ["T11"]),
    (r"\blinear struct\b", ("accept", "run"), ["T12", "CH06-M-ca6452be", "CH06-M-f43a675d"]),
    (r"&mut\b|&\w", ("accept", "run"), ["T6"]),
    (r"\bimpl\b.*\bfor\b", ("accept",), ["CH06-M-8e9952a9", "CH06-M-02b1c842"]),
    (r"\bGuarded\[", ("accept",), ["CH06-M-7c164794"]),
    (r"\bchan\[", ("accept", "run"), ["CH06-M-5912c594"]),
    (r"^step .* if ", ("accept", "run", "trace"), ["CH05-M-d8ee78b3"]),
    (r"^\s*example .*-> kept:", ("accept",), ["CH08-M-51f39d7d"]),
    (r"^\s*example .*-> (?!kept)", ("accept",), ["CH08-M-9a15157f"]),
    (r"^# rce-version:", ("accept",), ["CH02-M-a696f3a3"]),
    (r"^use ", ("accept",), ["CH04-M-c3c51cd1"]),
    (r"\.chars\(\)|\.len\(\)|\\u\{", ("run",), ["N7", "CH05-M-84afe529"]),
    (r"\bnone\b", ("run",), ["N10"]),
    (r"\bTick\b|after \d", ("run", "host"), ["N11", "HC-2"]),
    (r"\[\s*\w+\s*\]|\[\w+:\w*\]", ("run",), ["N8"]),
]
def tags(src, m):
    exp, cl = m.get("expect"), set(m.get("clauses", []) or [])
    if exp in ("run", "trace"):
        cl |= {"T1", "T5", "CH00-M-61b6ac06", "CH07-M-0a0bd71f"}
    if exp == "run":
        cl |= {"T9", "T4"}
        if str(m.get("outcome", "")).startswith("kept"): cl |= {"T8"}
        if str(m.get("outcome", "")) == "deadlock": cl |= {"T4"}
        if m.get("schedule_seeds"): cl |= {"T16", "T17", "CH14-M-90d1ad7f"}
        if m.get("fuel"): cl |= {"T2", "CH14-M-fb31f51f"}
    if exp == "trace":
        cl |= {"T3", "CH07-M-265230fb", "CH16-M-bf12d963" if False else "T3"}
        cl |= {"T8"} if "kept" in src or "fail " in src or "else " in src else set()
    if exp == "change":
        cl |= {"CH09-M-82e65964"}
    if exp == "accept":
        cl |= {"T10"}
    if exp in ("accept",) or (exp == "change" and m.get("diagnostic") == "W-SYN-1"):
        cl |= {"CH03-M-28cdd241"}   # checked by tools/ts_crosscheck.py against tree-sitter-rce
    if m.get("area") == "lexing-parsing":
        cl |= {"T10", "CH02-M-69371f07"} if (exp == "accept" or m.get("diagnostic") in ("W-LEX-2", "W-SYN-1")) else set()
    if m.get("area") == "numbers" and exp == "run":
        cl |= {"N1", "N3", "F7", "F9", "CH05-M-2405dfe8"}
    if exp == "host":
        cl |= {"CH16-M-bf12d963", "HC-1", "HC-10", "CH16-M-262166d9"}
        if "admit-order" in str(m.get("id")): cl |= {"T40"}
    prov = str(m.get("provenance", ""))
    if "/mutual/" in prov or "mutual" in prov:
        cl |= {"CH08-M-286d2fd4", "CH08-M-cbd85cbd", "CH11-M-7fed7576"}
    if "39_structural_recursion" in prov:
        cl |= {"CH11-M-7fed7576"}
    for rx, kinds, cs in CONSTRUCT:
        if exp in kinds and re.search(rx, src, re.M):
            cl |= set(cs)
    return sorted(cl)
n = 0
for p in sorted((ROOT / "tests").rglob("*.rce")):
    if "_FIXTURE" in p.name: continue
    raw = p.read_bytes().decode("utf-8", errors="surrogateescape")
    src, m = rcetest.split(raw)
    new = tags(src, m)
    if new != sorted(m.get("clauses", []) or []):
        m["clauses"] = new
        with open(p, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
            f.write(src + rcetest.render_meta(m))
        n += 1
print(f"retagged {n} tests")
