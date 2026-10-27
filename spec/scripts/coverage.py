#!/usr/bin/env python3
"""Totals of the library coverage ledger, spec/data/coverage-map.json.

  spec/scripts/coverage.py            print the totals by status, area, home,
                                      priority and wave
  spec/scripts/coverage.py --check    also verify the ledger's form: known
                                      fields and values, unique names, rows
                                      sorted by area then name

The ledger is the matrix of the proposal spec/proposals/the-library-covers-all-code.md.
A row's rce_status changes in the commit that lands its module, template,
adapter, tool or language item (rule LC-2 of that proposal), so these totals are the
coverage at the commit.
"""
import collections, json, pathlib, sys

PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "coverage-map.json"
FIELDS = ["name", "area", "sources", "prevalence", "rce_status", "rce_where", "home",
          "target_module", "priority", "wave", "depends_on", "note", "origin", "correction"]
STATUSES = ["have", "partial", "missing"]
HOMES = ["stdlib-module", "form-template", "example-form", "host-adapter", "language-growth", "tooling"]


def check(rows):
    errs = []
    for i, r in enumerate(rows):
        if list(r) != FIELDS:
            errs.append(f"row {i}: fields {list(r)}")
            continue
        if r["rce_status"] not in STATUSES:
            errs.append(f"row {i}: status {r['rce_status']!r}")
        if r["home"] not in HOMES:
            errs.append(f"row {i}: home {r['home']!r}")
        if r["priority"] not in (1, 2, 3):
            errs.append(f"row {i}: priority {r['priority']!r}")
        if r["wave"] is not None and r["wave"] not in range(1, 10):
            errs.append(f"row {i}: wave {r['wave']!r}")
        if r["wave"] is None and r["rce_status"] != "have":
            errs.append(f"row {i}: {r['name']!r} is {r['rce_status']} with no wave")
        if r["origin"] not in ("survey", "critique"):
            errs.append(f"row {i}: origin {r['origin']!r}")
    names = collections.Counter(r["name"] for r in rows)
    errs += [f"duplicate name {n!r}" for n, c in names.items() if c > 1]
    key = [(r["area"], r["name"].casefold(), r["name"]) for r in rows]
    if key != sorted(key):
        errs.append("rows are not sorted by area, then name")
    return errs


def table(title, rows, field, order=None):
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r[field]][r["rce_status"]] += 1
    keys = order or sorted(by, key=lambda k: (-sum(by[k].values()), str(k)))
    print(f"\n| {title} | rows | have | partial | missing |\n|---|---|---|---|---|")
    for k in keys:
        if k in by:
            c = by[k]
            print(f"| {'none' if k is None else k} | {sum(c.values())} | {c['have']} | {c['partial']} | {c['missing']} |")


def main():
    rows = json.loads(PATH.read_text())["rows"]
    if "--check" in sys.argv[1:]:
        errs = check(rows)
        for e in errs:
            print("coverage.py:", e, file=sys.stderr)
        if errs:
            sys.exit(1)
    s = collections.Counter(r["rce_status"] for r in rows)
    o = collections.Counter(r["origin"] for r in rows)
    print(f"{len(rows)} rows ({o['survey']} from the surveys, {o['critique']} added by the critique): "
          f"{s['have']} have, {s['partial']} partial, {s['missing']} missing")
    mods = {r["target_module"] for r in rows if r["home"] == "stdlib-module" and r["target_module"]}
    tpls = {r["target_module"] for r in rows if r["home"] == "form-template" and r["target_module"]}
    man = PATH.parent.parent.parent / "stdlib" / "lib" / "MANIFEST.json"
    have = {m["module"] for m in json.loads(man.read_text())["modules"]} if man.exists() else set()
    print(f"{len(mods)} target modules ({len(mods & have)} in stdlib/lib/MANIFEST.json), {len(tpls)} templates")
    table("Status", rows, "rce_status", STATUSES)
    table("Priority", rows, "priority", [1, 2, 3])
    table("Wave", rows, "wave", list(range(1, 10)) + [None])
    table("Home", rows, "home")
    table("Area", rows, "area")


if __name__ == "__main__":
    main()
