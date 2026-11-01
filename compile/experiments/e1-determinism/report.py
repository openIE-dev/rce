#!/usr/bin/env python3
"""Generate the E1 write-up from E1 results files; never edit its output by hand.

    python3 report.py write                      RESULTS.md from results/*.json
    python3 report.py check                      exit 1 if RESULTS.md is not what write would print
    python3 report.py write OUT_MD RESULTS_JSON...   the same for named files (used by CI)

Every number and every verdict in the output is computed here from the
results files that e1.py wrote.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARISONS = (
    ("same_dir", "Same directory twice"),
    ("cross_dir", "Two directories"),
    ("remapped", "Two directories, remapped"),
    ("same_dir_clock", "Same directory, clock free"),
)


def cell(rows, key):
    n = len(rows)
    st = [r[key]["status"] for r in rows]
    if all(s == "identical" for s in st):
        return "identical (%d of %d)" % (n, n)
    if all(s == "not tested" for s in st):
        return "not tested"
    parts = []
    diff = [r for r in rows if r[key]["status"] == "differs"]
    if diff:
        worst = max(sum(f["differing_bytes"] for f in r[key]["files"]) for r in diff)
        parts.append("differs (%d of %d; up to %s bytes)" % (len(diff), n, format(worst, ",")))
    fail = sum(1 for s in st if s == "build failed")
    if fail:
        parts.append("build failed (%d of %d)" % (fail, n))
    same = sum(1 for s in st if s == "identical")
    if same:
        parts.append("identical (%d of %d)" % (same, n))
    return "; ".join(parts)


def copies(rows, key):
    vals = [r.get(key) for r in rows]
    if all(v is None for v in vals):
        return "-"
    return "/".join("-" if v is None else str(v) for v in vals)


def shown(p):
    rel = os.path.relpath(os.path.abspath(p), HERE)
    return os.path.basename(p) if rel.startswith("..") else rel


def esc(s):
    return str(s).replace("|", "\\|")


def section(name, res):
    out = []
    m = res["machine"]
    out.append("## %s" % name)
    out.append("")
    out.append("Run of %s (UTC). Machine: %s %s, %s, %s CPUs, %s; Python %s." % (
        res["date_utc"], m["system"], m["machine"], m["cpu"] or "CPU model not reported", m["cpus"],
        m["os"] or "OS release not reported", m["python"]))
    env = res["environment"]
    out.append("Declared environment: %s. Harness: `e1.py` sha256 `%s`, `matrix.json` sha256 `%s`." % (
        ", ".join("%s=%s" % (k, env[k]) for k in sorted(env)),
        res["harness"]["e1.py"][:16], res["harness"]["matrix.json"][:16]))
    out.append("")
    out.append("| Tool | Version |")
    out.append("|---|---|")
    for t in sorted(res["toolchains"]):
        out.append("| `%s` | %s |" % (t, esc(res["toolchains"][t])))
    out.append("")
    order, groups = [], {}
    for r in res["rows"]:
        if r["config"] not in groups:
            order.append(r["config"])
            groups[r["config"]] = []
        groups[r["config"]].append(r)
    out.append("| Configuration | Language | Programs | %s | Path copies (two directories / remapped) |" % " | ".join(c[1] for c in COMPARISONS))
    out.append("|---|---|---|%s---|" % ("---|" * len(COMPARISONS)))
    for c in order:
        rows = groups[c]
        out.append("| `%s` | %s | %d | %s | %s / %s |" % (
            esc(c), rows[0]["language"], len(rows), " | ".join(cell(rows, k) for k, _ in COMPARISONS),
            copies(rows, "path_copies_cross"), copies(rows, "path_copies_remapped")))
    out.append("")
    if res["absent"]:
        out.append("Not run on this machine: " + "; ".join("`%s` (%s)" % (esc(a["config"]), a["reason"]) for a in res["absent"]) + ".")
        out.append("")
    detail = []
    for r in res["rows"]:
        for k, label in COMPARISONS:
            cmpn = r[k]
            if cmpn["status"] == "build failed":
                detail.append("| `%s` | `%s` | %s | | | | build failed: %s |" % (esc(r["config"]), r["program"], label, esc(cmpn["error"])))
            for f in cmpn.get("files", []):
                if f["differing_bytes"] or "missing_in" in f:
                    detail.append("| `%s` | `%s` | %s | `%s` | %s | %s | %s |" % (
                        esc(r["config"]), r["program"], label, esc(f["file"]),
                        "-" if f["size_a"] is None else format(f["size_a"], ","),
                        "-" if f["size_b"] is None else format(f["size_b"], ","),
                        format(f["differing_bytes"], ",")))
    if detail:
        out.append("Files that differ, per build pair (sizes in bytes; the first build of the pair is a, the second b):")
        out.append("")
        out.append("| Configuration | Program | Comparison | File | Size a | Size b | Differing bytes |")
        out.append("|---|---|---|---|---|---|---|")
        out.extend(detail)
        out.append("")
    survive = sorted({r["config"] for r in res["rows"] if r["remapped"]["status"] == "differs"})
    same = sorted({r["config"] for r in res["rows"] if r["same_dir"]["status"] != "identical"})
    clock = sorted({r["config"] for r in res["rows"] if r["same_dir_clock"]["status"] == "differs"})
    cross_path = sorted({r["config"] for r in res["rows"] if r["cross_dir"]["status"] == "differs" and (r.get("path_copies_cross") or 0) > 0})
    out.append("Computed from the rows above:")
    out.append("")
    out.append("- configurations with any difference between two builds in the same directory: %s;" % (", ".join("`%s`" % c for c in same) or "none"))
    out.append("- configurations whose two-directory difference comes with the build directory in the output: %s;" % (", ".join("`%s`" % c for c in cross_path) or "none"))
    out.append("- configurations with a difference that survives the remapped build: %s;" % (", ".join("`%s`" % c for c in survive) or "none"))
    out.append("- configurations whose output depends on the clock when SOURCE_DATE_EPOCH is unset: %s." % (", ".join("`%s`" % c for c in clock) or "none"))
    out.append("")
    return out


def across(pa, ra, pb, rb):
    """Compare output hashes of two results files where the tool version strings are equal."""
    na, nb = os.path.splitext(os.path.basename(pa))[0], os.path.splitext(os.path.basename(pb))[0]
    out = ["## Across machines: %s and %s" % (na, nb), ""]
    out.append("Only configurations whose tool version string is the same in both files are compared. "
               "Each cell compares the SHA-256 of every output of the first build in directory A on each machine; "
               "the directory A of the two runs is a different absolute path.")
    out.append("")
    ib = {(r["config"], r["program"]): r for r in rb["rows"]}
    tools = {}
    lines = []
    for r in ra["rows"]:
        key = (r["config"], r["program"])
        if key not in ib:
            continue
        s = ib[key]
        va = [ra["toolchains"][r["tool"]]] if r["tool"] in ra["toolchains"] else []
        vb = [rb["toolchains"][s["tool"]]] if s["tool"] in rb["toolchains"] else []
        if not va or va != vb:
            tools.setdefault(r["config"], (va[0] if va else "?", vb[0] if vb else "?"))
            continue

        def eq(x, y):
            if x is None or y is None:
                return "-"
            if x == y:
                return "identical"
            return "differs (%s)" % ", ".join("`%s`" % f for f in sorted(set(x) | set(y)) if x.get(f) != y.get(f))
        lines.append("| `%s` | `%s` | %s | %s | %s |" % (
            esc(r["config"]), r["program"], esc(va[0]), eq(r.get("sha256"), s.get("sha256")),
            eq(r.get("sha256_remapped"), s.get("sha256_remapped"))))
    if lines:
        out.append("| Configuration | Program | Version | Default build | Remapped build |")
        out.append("|---|---|---|---|---|")
        out.extend(lines)
        out.append("")
    else:
        out.append("No configuration has the same tool version string in both files.")
        out.append("")
    if tools:
        out.append("Not compared, because the tool version strings differ: " + "; ".join(
            "`%s` (%s; %s)" % (esc(c), esc(a), esc(b)) for c, (a, b) in sorted(tools.items())) + ".")
        out.append("")
    return out


def render(paths):
    out = [
        "# E1 results: determinism of lowering",
        "",
        "Generated by `report.py` from %s. Do not edit; run `python3 report.py write`." % ", ".join("`%s`" % shown(p) for p in paths),
        "The method is in [PROTOCOL.md](PROTOCOL.md). Each comparison is over every file a build leaves in its directory, other than the source.",
        "A cell counts programs: \"identical (2 of 2)\" means both programs of the configuration gave byte-identical outputs in that comparison.",
        "\"Path copies\" is the number of times the absolute build directory occurs in the second directory's outputs, per program.",
        "",
    ]
    loaded = []
    for p in paths:
        res = json.load(open(p, encoding="utf-8"))
        if res.get("schema") != "rce-compile-e1-results/1":
            sys.exit("%s: unknown schema %r" % (p, res.get("schema")))
        out.extend(section(os.path.splitext(os.path.basename(p))[0], res))
        loaded.append((p, res))
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            out.extend(across(loaded[i][0], loaded[i][1], loaded[j][0], loaded[j][1]))
    return "\n".join(out).rstrip("\n") + "\n"


def main():
    a = sys.argv[1:]
    if a in (["write"], ["check"]):
        md = os.path.join(HERE, "RESULTS.md")
        paths = sorted(glob.glob(os.path.join(HERE, "results", "*.json")))
    elif len(a) >= 3 and a[0] in ("write", "check"):
        md, paths = a[1], a[2:]
    else:
        sys.exit("usage: python3 report.py write|check [OUT_MD RESULTS_JSON...]")
    if not paths:
        sys.exit("report.py: no results files")
    text = render(paths)
    if a[0] == "write":
        with open(md, "w", encoding="utf-8") as f:
            f.write(text)
        print("wrote %s from %d results file(s)" % (md, len(paths)))
        return
    try:
        cur = open(md, encoding="utf-8").read()
    except OSError:
        cur = None
    if cur != text:
        sys.exit("report.py check: %s is not generated from %s; run python3 report.py write" % (md, ", ".join(paths)))
    print("report.py check: %s matches %d results file(s)" % (md, len(paths)))


if __name__ == "__main__":
    main()
