#!/usr/bin/env python3
"""Join the results files of one adapter, each from a run over part of the
suite (`harness/run.py ADAPTER tests/AREA PART`, run from
`conformance/`), into one results file, as a machine too small for one
whole run makes it. The parts' results are put
together and sorted by path, and the tally is recounted. Every part must be
rce-results/2, name the same adapter and carry the same hello, and no test
may be in two parts; otherwise nothing is written (exit 1).
  tools/join_results.py OUT PART...
"""
import json, sys

if len(sys.argv) < 3:
    sys.exit("usage: tools/join_results.py OUT PART...")
out, parts = sys.argv[1], [json.load(open(p)) for p in sys.argv[2:]]
first = parts[0]
for p, name in zip(parts, sys.argv[2:]):
    if p.get("format") != "rce-results/2":
        sys.exit(f"{name}: format {p.get('format')}, not rce-results/2")
    if p["adapter"] != first["adapter"] or p.get("hello") != first.get("hello"):
        sys.exit(f"{name}: adapter or hello differs from {sys.argv[2]}")
results, seen = [], set()
for p, name in zip(parts, sys.argv[2:]):
    for r in p["results"]:
        if r["path"] in seen:
            sys.exit(f"{name}: {r['path']} is in two parts")
        seen.add(r["path"])
        results.append(r)
results.sort(key=lambda r: r["path"])
tally = {}
for r in results:
    tally[r["status"]] = tally.get(r["status"], 0) + 1
json.dump({"format": "rce-results/2", "adapter": first["adapter"], "hello": first.get("hello"), "tally": tally,
           "results": results}, open(out, "w"), indent=1)
print(f"{first['adapter']}: {len(results)} results from {len(parts)} parts: {tally}")
