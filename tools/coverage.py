#!/usr/bin/env python3
"""Clause coverage: which tests name each normative clause, and whether the
reference passes them.  coverage.py [results.json] -> reports/coverage.{json,md}
A clause counts as covered when at least one test names it. Informative
sentences (quotations, citations, tooling meta-statements) are listed apart
and do not count toward the total."""
import collections, json, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
import rcetest
ROOT = pathlib.Path(__file__).resolve().parent.parent
clauses = (json.load(open(ROOT / "clauses.json"))["clauses"] + json.load(open(ROOT / "clauses-pending.json"))["clauses"])
res = {}
if len(sys.argv) > 1:
    for r in json.load(open(sys.argv[1]))["results"]:
        res[r["path"]] = r["status"]
by = collections.defaultdict(list); areas = collections.Counter(); unknown = collections.Counter()
ids = {c["id"] for c in clauses}
for p in sorted((ROOT / "tests").rglob("*.rce")):
    if "_FIXTURE" in p.name: continue
    _, m = rcetest.split(p.read_text(encoding="utf-8", errors="surrogateescape"))
    rel = str(p.relative_to(ROOT)); areas[m.get("area")] += 1
    for c in m.get("clauses", []) or []:
        if c in ids: by[c].append(rel)
        else: unknown[c] += 1
allnorm = [c for c in clauses if c.get("normative", True)]
profiles = collections.Counter(c.get("profile", "language") for c in allnorm)
prof_cov = {pr: sum(1 for c in allnorm if c.get("profile", "language") == pr and by.get(c["id"])) for pr in profiles}
norm = [c for c in allnorm if c.get("profile", "language") == "language"]
rows, untested = [], []
for c in norm:
    ts = by.get(c["id"], [])
    st = collections.Counter(res.get(t, "not-run") for t in ts)
    rows.append({"id": c["id"], "kind": c["kind"], "chapter": c["chapter"], "line": c["line"], "tests": len(ts),
                 "reference": dict(st)})
    if not ts: untested.append(c)
cov = 100.0 * (len(norm) - len(untested)) / len(norm)
out = {"format": "rce-coverage/1", "clauses": len(norm), "covered": len(norm) - len(untested), "percent": round(cov, 2),
       "tests_by_area": dict(sorted(areas.items())), "tests_total": sum(areas.values()), "per_clause": rows,
       "untested": [c["id"] for c in untested], "unknown_clause_ids_in_tests": dict(unknown),
       "profiles": {pr: {"clauses": n, "covered": prof_cov[pr]} for pr, n in profiles.items()},
       "untested_other_profiles": [c["id"] for c in allnorm if c.get("profile", "language") != "language" and not by.get(c["id"])],
       "informative": [c["id"] for c in clauses if not c.get("normative", True)]}
(ROOT / "reports").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "reports" / "coverage.json", "w"), indent=1)
md = [f"# Clause coverage\n", f"Language profile: {out['covered']} of {out['clauses']} normative clauses have at least one test ({out['percent']}%). "
      f"{out['tests_total']} tests.\n", "## Tests by area\n", "| area | tests |", "|---|---|"]
md += [f"| {a} | {n} |" for a, n in sorted(areas.items())]
md += ["\n## Profiles\n", "| profile | clauses | covered |", "|---|---|---|"]
md += [f"| {pr} | {v['clauses']} | {v['covered']} |" for pr, v in out["profiles"].items()]
md += ["\n## Untested clauses (language profile)\n"] + ([f"- {c['id']} ({c['chapter']}:{c['line']}): {c['text'][:160]}" for c in untested] or ["none"])
md += ["\n## Per clause\n", "| clause | kind | chapter | tests | reference |", "|---|---|---|---|---|"]
md += [f"| {r['id']} | {r['kind']} | {r['chapter']}:{r['line']} | {r['tests']} | {r['reference']} |" for r in rows]
open(ROOT / "reports" / "coverage.md", "w").write("\n".join(md) + "\n")
print(json.dumps({k: out[k] for k in ("clauses", "covered", "percent", "tests_total")}), "untested:", len(untested))
