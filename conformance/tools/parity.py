#!/usr/bin/env python3
"""Emission parity: compare the bytes of each adapter's answers with the
reference's, test by test, across every reports/results-*.json.

Each result of a run records `observed`, the part of the answer the harness
compared (check: the verdict and the diagnostic codes; run: the outputs and
outcome of the source-order run; trace: the head and every line of the
whole trace). For a test both the reference and an adapter answered, both
answers are written as canonical JSON (keys sorted, no spaces, UTF-8) and
the two byte strings are compared. Statuses are not compared: a test counts
as `same` only when the bytes are identical.

A results file written before observations were recorded (format
rce-results/1) holds the statuses of an earlier run only: that
implementation was not run for this comparison, so every test is counted
as `not compared`, never as agreeing, and its row says so. Writes
reports/parity.json and reports/parity.md.
"""
import glob, json, os, sys

root = os.path.join(os.path.dirname(__file__), "..")


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def first_difference(ref, got):
    """Where two answers first differ, in words (for the report)."""
    for k in ("verdict", "codes", "head", "outcome"):
        if ref.get(k) != got.get(k):
            return f"{k}: reference {json.dumps(ref.get(k))[:160]}, adapter {json.dumps(got.get(k))[:160]}"
    for k in ("lines", "outputs"):
        a, b = ref.get(k) or [], got.get(k) or []
        if a != b:
            i = next(i for i in range(max(len(a), len(b))) if a[i:i + 1] != b[i:i + 1])
            ra = json.dumps(a[i])[:160] if i < len(a) else "(no line)"
            rb = json.dumps(b[i])[:160] if i < len(b) else "(no line)"
            return f"{k} line {i + 1}: reference {ra}, adapter {rb}"
    return "the canonical bytes differ"


def label(d):
    if d.get("format") != "rce-results/2":
        return "not run for this comparison: the file holds statuses of an earlier run (rce-results/1)"
    h = d.get("hello")
    if not h:
        return "not recorded"
    if "error" in h:
        return f"hello failed: {h['error'][:120]}"
    parts = [h.get("implementation", "?")]
    if h.get("commit"):
        # a report recorded before the reference stopped printing a typed
        # version still carries `version`; current builds name only the commit
        parts.append((f"version {h['version']}, " if h.get("version") else "") + f"commit {h['commit']}")
    if h.get("emitted_by"):
        parts.append(f"emitted by {h['emitted_by']}")
    if h.get("toolchain"):
        parts.append(h["toolchain"])
    return "; ".join(parts)


runs = {}
for f in sorted(glob.glob(os.path.join(root, "reports", "results-*.json"))):
    d = json.load(open(f))
    runs[d["adapter"]] = d
if "reference" not in runs:
    sys.exit("no reports/results-reference.json")
refrun = runs["reference"]
if refrun.get("format") != "rce-results/2":
    sys.exit("reports/results-reference.json records no observations (format rce-results/1): rerun the reference")
ref = {r["id"]: r for r in refrun["results"]}
table, different, unanswered = {}, [], []
for name, d in runs.items():
    if name == "reference":
        continue
    res = {r["id"]: r for r in d["results"]}
    observed = d.get("format") == "rce-results/2"
    t = {"same": 0, "different": 0, "no answer": 0, "n/a": 0, "unsupported": 0, "not run": 0, "not compared": 0}
    for tid, r in ref.items():
        o = res.get(tid)
        if not observed:   # statuses of an earlier run: nothing here was run for this comparison
            t["not compared"] += 1
        elif o is None:
            t["not run"] += 1
        elif o["status"] == "unsupported":
            t["unsupported"] += 1
        elif o["status"] == "n/a":   # the op belongs to another component (adapter not_applicable)
            t["n/a"] += 1
        elif o.get("observed") is None or r.get("observed") is None:
            t["no answer"] += 1
            unanswered.append({"id": tid, "adapter": name, "adapter_status": o["status"],
                               "reference_answered": r.get("observed") is not None,
                               "detail": (o.get("detail") or "")[:300]})
        elif canon(o["observed"]) == canon(r["observed"]):
            t["same"] += 1
        else:
            t["different"] += 1
            different.append({"id": tid, "adapter": name, "reference_status": r["status"], "adapter_status": o["status"],
                              "where": first_difference(r["observed"], o["observed"])})
    table[name] = dict(t, run=label(d), results_format=d.get("format", "rce-results/1"))
json.dump({"format": "rce-parity/2", "compared": "bytes of the canonical JSON of each answer", "reference": label(refrun),
           "adapters": table, "different": different, "no_answer": unanswered},
          open(os.path.join(root, "reports", "parity.json"), "w"), indent=1, ensure_ascii=False)
with open(os.path.join(root, "reports", "parity.md"), "w", encoding="utf-8") as w:
    w.write("# Emission parity against the reference, byte for byte\n\n")
    w.write("Generated by `tools/parity.py` from `reports/results-*.json`. For each test, the answer an adapter gave "
            "(check: verdict and diagnostic codes; run: outputs and outcome of the source-order run; trace: the head "
            "and every line of the whole trace) is written as canonical JSON and compared byte for byte with the "
            "reference's answer to the same test. Statuses are not compared. *No answer*: the adapter or the reference "
            "gave no answer to compare (an error). *Not compared*: the results file records the statuses of an earlier run only "
            "(`rce-results/1`, made before answers were recorded); that implementation was not run for this comparison, "
            "and none of its tests is counted as agreeing. *Not run*: the run has no result for a test the reference "
            "answered.\n\n")
    w.write(f"Reference: {label(refrun)}.\n\n")
    w.write("| adapter | run | same bytes | different bytes | no answer | n/a | unsupported | not run | not compared |\n")
    w.write("|---|---|---|---|---|---|---|---|---|\n")
    for n, t in table.items():
        w.write(f"| {n} | {t['run']} | {t['same']} | {t['different']} | {t['no answer']} | {t['n/a']} | {t['unsupported']} | {t['not run']} | {t['not compared']} |\n")
    w.write("\n## Different bytes\n\n")
    for s in different:
        w.write(f"- `{s['id']}` {s['adapter']} ({s['adapter_status']}; reference {s['reference_status']}): {s['where']}\n")
    if not different:
        w.write("None.\n")
    w.write("\n## No answer\n\n")
    for s in unanswered:
        who = "adapter" if s["reference_answered"] else "reference"
        w.write(f"- `{s['id']}` {s['adapter']}: no answer from the {who} ({s['adapter_status']}): {s['detail'][:200]}\n")
    if not unanswered:
        w.write("None.\n")
print(json.dumps({n: {k: v for k, v in t.items() if k not in ("run", "results_format")} for n, t in table.items()}))
