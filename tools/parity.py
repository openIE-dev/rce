#!/usr/bin/env python3
"""Emission parity: compare per-test status across every reports/results-*.json.

A test is in parity when every adapter that supports it reports the same
status as the reference. Writes reports/parity.json and reports/parity.md.
"""
import glob, json, os, sys
root = os.path.join(os.path.dirname(__file__), "..")
runs = {}
for f in sorted(glob.glob(os.path.join(root, "reports", "results-*.json"))):
    d = json.load(open(f))
    runs[d["adapter"]] = {r["id"]: r for r in d["results"]}
if "reference" not in runs:
    sys.exit("no reports/results-reference.json")
ref = runs["reference"]
table, split = {}, []
for name, res in runs.items():
    if name == "reference":
        continue
    t = {"agree": 0, "disagree": 0, "n/a": 0, "unsupported": 0, "missing": 0}
    for tid, r in ref.items():
        o = res.get(tid)
        if o is None: t["missing"] += 1
        elif o["status"] == "unsupported": t["unsupported"] += 1
        elif o["status"] == "n/a": t["n/a"] += 1   # the op belongs to another component (adapter not_applicable)
        elif o["status"] == r["status"]: t["agree"] += 1
        else:
            t["disagree"] += 1
            split.append({"id": tid, "adapter": name, "reference": r["status"], "adapter_status": o["status"], "detail": o.get("detail", "")[:300]})
    table[name] = t
json.dump({"adapters": table, "disagreements": split}, open(os.path.join(root, "reports", "parity.json"), "w"), indent=1)
with open(os.path.join(root, "reports", "parity.md"), "w") as w:
    w.write("# Emission parity against the reference\n\n| adapter | agree | disagree | n/a | unsupported | not run |\n|---|---|---|---|---|---|\n")
    for n, t in table.items():
        w.write(f"| {n} | {t['agree']} | {t['disagree']} | {t['n/a']} | {t['unsupported']} | {t['missing']} |\n")
    w.write("\n## Disagreements\n\n")
    for s in split:
        w.write(f"- `{s['id']}` {s['adapter']}: reference {s['reference']}, adapter {s['adapter_status']}\n")
print(json.dumps(table))
