#!/usr/bin/env python3
"""Static checks that need no implementation: every test parses, has a unique
id matching its path, a known area and expect, and names only clause ids
present in clauses.json; every language-profile clause has a test."""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import rcetest
clauses = {c["id"]: c for c in (json.load(open(ROOT / "clauses.json"))["clauses"] + json.load(open(ROOT / "clauses-pending.json"))["clauses"])}
EXPECT = {"accept", "reject", "run", "trace", "host", "replicate"}
errs, ids, hit = [], set(), set()
tests = [p for p in sorted((ROOT / "tests").rglob("*.rce")) if "_FIXTURE" not in p.name]
for p in tests:
    try:
        _, m = rcetest.split(p.read_text(encoding="utf-8"))
    except Exception as e:
        errs.append(f"{p}: {e}"); continue
    rel = p.relative_to(ROOT / "tests").with_suffix("").as_posix()
    if m.get("id") != rel: errs.append(f"{p}: id {m.get('id')} != path {rel}")
    if m.get("id") in ids: errs.append(f"{p}: duplicate id")
    ids.add(m.get("id"))
    if m.get("area") != rel.split("/")[0]: errs.append(f"{p}: area {m.get('area')}")
    if m.get("expect") not in EXPECT: errs.append(f"{p}: expect {m.get('expect')}")
    if m.get("expect") == "reject" and not m.get("diagnostic"): errs.append(f"{p}: reject without diagnostic")
    cl = m.get("clauses") or []
    if not cl: errs.append(f"{p}: no clauses")
    for c in cl:
        if c not in clauses: errs.append(f"{p}: unknown clause {c}")
        hit.add(c)
lang = [c for c, v in clauses.items() if v.get("normative") and v.get("profile") == "language"]
miss = [c for c in lang if c not in hit]
for c in miss: errs.append(f"untested language clause {c}")
for e in errs[:200]: print(e)
print(f"{len(tests)} tests, {len(lang) - len(miss)}/{len(lang)} language clauses covered, {len(errs)} problems")
sys.exit(1 if errs else 0)
