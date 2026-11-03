#!/usr/bin/env python3
"""Write or check RESULTS.md of E5 from the results files.

    python3 report.py write|check [OUT_MD RESULTS_JSON...]

With no further arguments the write-up is RESULTS.md and the results files
are results/*.json. `check` fails when the write-up is not what `write`
would print. No figure in RESULTS.md is typed: every number comes from a
results file.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "rce-compile-e5-results/1"
R_KEYS = ("machine-module", "machine-run", "ir-module", "ir-run")
BASELINES = ("output-bytes", "source-key")
DIAGNOSTIC = ("ir-module-path-removed",)
ORDER = BASELINES + R_KEYS + DIAGNOSTIC
OUTPUT_ORDER = ("native-object", "macho-arm64-object", "wasm-object", "wasm-module")
COMP_ORDER = ("two_dir", "renamed", "opt_levels", "c_vs_rust", "different_kernels")


def shown(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, HERE) if p.startswith(HERE) else os.path.basename(p)


def rate(t):
    if not t or not t["pairs"]:
        return "n/a"
    return "%d/%d (%.1f%%)" % (t["equal"], t["pairs"], 100.0 * t["equal"] / t["pairs"])


def outputs_of(comp):
    outs = set(comp["by_output"])
    return [o for o in OUTPUT_ORDER if o in outs]


def comparison_table(name, comp):
    out = ["", "#### `%s` (declared %s)" % (name, comp["declared"]), "",
           "Equal pairs out of pairs compared. `n/a`: the key does not apply (no function `run`, or not this kind "
           "of output).", "",
           "| Output | Programs | %s |" % " | ".join("`%s`" % k for k in ORDER),
           "|---|---|" + "---|" * len(ORDER)]
    for o in outputs_of(comp):
        for st in ("E4", "E1"):
            t = comp["by_output"][o].get(st)
            if t is None:
                continue
            out.append("| %s | %s | %s |" % (o, st, " | ".join(rate(t.get(k)) for k in ORDER)))
    return out


def misses(name, comp):
    """Pairs whose R hash disagrees with the declaration."""
    out = []
    for o in outputs_of(comp):
        for st, lst in sorted(comp["examples"].get(o, {}).items()):
            for e in lst:
                parts = e.get("parts_differing")
                out.append("| `%s` | %s | %s | %s | %s | %s |" % (
                    name, o, e["key"], e["a"], e["b"],
                    ("parts differing: " + ", ".join(parts) if parts is not None else "-") +
                    "; copies of the build directory in the outputs: %s" % "/".join(str(x) for x in e["dir_copies"])))
    return out


def held(cond):
    return "held" if cond else "**not held**"


def get(comp, out, st, key):
    return comp["by_output"].get(out, {}).get(st, {}).get(key)


def predictions(res):
    c = res["comparisons"]
    lines = ["", "### Registered predictions and conditions", "",
             "| Prediction (PROTOCOL.md) | Outcome | Measured |", "|---|---|---|"]

    def all_keys(comp, keys, want_equal):
        """Cells (output, programs, key) where the prediction fails, and the number of cells."""
        bad, tot = [], 0
        for o, by in comp["by_output"].items():
            for st, t in by.items():
                for k in keys:
                    x = t.get(k)
                    if not x or not x["pairs"]:
                        continue
                    tot += 1
                    if (x["equal"] < x["pairs"]) if want_equal else (x["equal"] > 0):
                        bad.append("%s %s `%s` %s" % (o, st, k, rate(x)))
        return bad, tot

    b, _ = all_keys(c["two_dir"], R_KEYS, True)
    lines.append("| every R hash equal on every `two_dir` pair | %s | %s |" % (held(not b), "; ".join(b) or "all equal"))
    b, _ = all_keys(c["renamed"], R_KEYS, True)
    lines.append("| every R hash equal on every `renamed` pair | %s | %s |" % (held(not b), "; ".join(b) or "all equal"))
    b, _ = all_keys(c["renamed"], ("source-key",), False)
    unchanged = sorted(p for p, v in res["renames"].items() if not v["mapping"])
    lines.append("| `source-key` unequal on every `renamed` pair | %s | %s%s |" % (
        held(not b), "; ".join(b) or "all unequal",
        "; programs the renamer left unchanged (they bind no identifier): %s" % ", ".join(unchanged) if unchanged else ""))
    ob = []
    for o, by in c["renamed"]["by_output"].items():
        for st, t in by.items():
            x = t.get("output-bytes")
            if x and x["pairs"] and x["equal"] < x["pairs"]:
                ob.append("%s %s %s" % (o, st, rate(x)))
    lines.append("| `output-bytes` unequal on some `renamed` pairs | %s | equal: %s |" % (held(bool(ob)), "; ".join(ob)))
    b, _ = all_keys(c["c_vs_rust"], ("machine-module", "machine-run"), False)
    lines.append("| machine-code R hashes unequal on every `c_vs_rust` pair | %s | %s |" % (held(not b), "; ".join(b) or "all unequal"))
    b, _ = all_keys(c["c_vs_rust"], ("ir-run",), False)
    lines.append("| `ir-run` unequal on every `c_vs_rust` pair | %s | %s |" % (held(not b), "; ".join(b) or "all unequal"))
    fb, _ = all_keys(c["different_kernels"], R_KEYS, False)
    lines.append("| no R hash equal on any `different_kernels` pair | %s | %s |" % (held(not fb), "; ".join(fb) or "none equal"))
    m = res["mutations"]
    lines.append("| `ir-module` changed by every mutation | %s | %d of %d mutations of %d modules left it unchanged |" % (
        held(m["unchanged"] == 0), m["unchanged"], m["mutations"], m["modules"]))

    # H5 by its registered conditions
    gains, nogain = [], []
    for o, by in c["renamed"]["by_output"].items():
        for st, t in by.items():
            base = max((t[k]["equal"] / t[k]["pairs"]) for k in BASELINES if t.get(k) and t[k]["pairs"])
            for k in R_KEYS:
                x = t.get(k)
                if not x or not x["pairs"]:
                    continue
                r = x["equal"] / x["pairs"]
                (gains if r > base else nogain).append("%s %s `%s`" % (o, st, k))
    false_hits = len(fb) + (1 if m["unchanged"] else 0)
    lines.append("")
    if false_hits:
        verdict = "**Falsified**: a false hit occurred (%s)." % "; ".join(fb + (["mutations"] if m["unchanged"] else []))
    elif not gains:
        verdict = "**Falsified**: no R hash has a higher hit rate than both baselines on the `renamed` pairs."
    elif nogain:
        verdict = ("Neither condition met in full: no false hit, and a gain for %s, but no gain for %s." %
                   (", ".join(gains), ", ".join(nogain)))
    else:
        verdict = ("**Supported** on this corpus: no false hit in %d R-hash comparisons of `different_kernels` pairs and %d mutations, "
                   "and every R hash has a higher hit rate on the `renamed` pairs than both baselines (%d output kind "
                   "and R hash combinations)." % (sum(x["pairs"] for by in c["different_kernels"]["by_output"].values()
                                            for t in by.values() for k, x in t.items() if k in R_KEYS),
                                        m["mutations"], len(gains)))
    lines.append("**H5 by its registered conditions.** " + verdict)
    return lines


def parts_table(comp, label):
    pe = comp.get("parts_equal", {}).get("E4")
    if not pe:
        return []
    out = ["", "Parts of the canonical form that are equal on the %s pairs of linked modules (`ir-module` form and "
           "`ir-run` form):" % label, "",
           "| Form | %s |" % " | ".join(sorted(pe["parts"], key=lambda x: x)), "|---|" + "---|" * len(pe["parts"])]
    for which in ("parts", "run_parts"):
        if which in pe:
            out.append("| %s | %s |" % ("ir-module" if which == "parts" else "ir-run",
                                        " | ".join(rate(pe[which][k]) for k in sorted(pe[which]))))
    return out


def section(name, res):
    m, rows = res["machine"], res["rows"]
    out = ["", "## %s" % name, "",
           "Machine: %s, %s (%s), %d CPUs, Python %s. Run %s to %s (UTC)." % (
               m["os"], m["machine"], m["cpu"] or "CPU not reported", m["cpus"] or 0, m["python"],
               res["start_utc"], res["date_utc"]), ""]
    for k, v in sorted(res["toolchains"].items()):
        out.append("- Toolchain `%s`: `%s` (%s)" % (k, v["tool"], v["version"]))
    for a in res["absent"]:
        out.append("- Toolchain `%s` absent: %s" % (a["toolchain"], a["reason"]))
    out.append("- Harness: %s" % ", ".join("`%s` %s" % (f, h[:12]) for f, h in sorted(res["harness"].items())))
    built = {}
    for r in rows:
        key = (r["output"], r["status"], r.get("hash_info", {}).get("status", "-"))
        built[key] = built.get(key, 0) + 1
    out += ["", "### Builds", "", "| Output | Status | Hash status | Builds |", "|---|---|---|---|"]
    for (o, s, h), n in sorted(built.items(), key=lambda x: (OUTPUT_ORDER.index(x[0][0]), x[0][1:])):
        out.append("| %s | %s | %s | %d |" % (o, s, h, n))
    dc = [r["decoder_check"] for r in rows if r.get("decoder_check")]
    if dc:
        out += ["", "x86-64 decoder against %s: instruction boundaries agree in %d of %d functions of the A host "
                "objects." % (res["llvm_objdump"], sum(x["agree"] for x in dc), sum(x["functions"] for x in dc))]
    ev = {}
    for r in rows:
        if "evidence" in r:
            k = (r["variant"], r["evidence"]["status"])
            ev[k] = ev.get(k, 0) + 1
    out += ["", "**Evidence that the renamed source is the same program.** Linked modules equal to the reference on "
            "every declared input, per variant: %s." % ", ".join("%s %s: %d" % (v, s, n) for (v, s), n in sorted(ev.items()))]
    for p, v in sorted(res["renames"].items()):
        e = v["evidence"]
        if not e["status"].startswith("checked per level"):
            out.append("- `%s`: %d identifiers renamed; %s." % (p, len(v["mapping"]), e["status"]))
    out.append("- E4 kernels: identifiers renamed per program: %s." % ", ".join(
        "`%s` %d" % (p, len(v["mapping"])) for p, v in sorted(res["renames"].items()) if not p.startswith("e1/")))
    out += ["", "### Comparisons"]
    for cname in COMP_ORDER:
        out += comparison_table(cname, res["comparisons"][cname])
        if cname == "c_vs_rust":
            out += parts_table(res["comparisons"][cname], "`c_vs_rust`")
    mm = res["mutations"]
    out += ["", "#### `mutations` (declared not equal)", "",
            "%d modules; %d mutations; `ir-module` unchanged by %d. By kind: %s." % (
                mm["modules"], mm["mutations"], mm["unchanged"],
                ", ".join("%s %d (unchanged %d)" % (k, v["mutations"], v["unchanged"])
                          for k, v in sorted(mm["by_kind"].items())))]
    ms = []
    for cname in COMP_ORDER:
        if cname != "opt_levels":  # equal pairs there are the same program and are counted above
            ms += misses(cname, res["comparisons"][cname])
    if ms:
        out += ["", "### Pairs whose R hash disagrees with the declaration", "",
                "For linked modules the parts of the canonical form that differ are named; the copies are the number of "
                "times each output contains its build directory's path. `ir-module-path-removed` is a diagnostic added "
                "after the first run.", "",
                "| Comparison | Output | Key | A | B | Detail |", "|---|---|---|---|---|---|"] + ms
    out += predictions(res)
    out.append("")
    return out


def render(paths):
    out = ["# E5 results: R, name-free identity across builds", "",
           "Generated by `report.py` from %s. Do not edit; run `python3 report.py write`." % ", ".join(
               "`%s`" % shown(p) for p in paths),
           "The method, the keys, the comparisons and the registered predictions are in [PROTOCOL.md](PROTOCOL.md). "
           "Keys: `output-bytes` and `source-key` are the baselines; `machine-module` and `machine-run` are the "
           "WARP-style IDs of unlinked objects; `ir-module` and `ir-run` are the alpha-equivalence hashes of linked "
           "WebAssembly modules; `ir-module-path-removed` is a diagnostic added after the first run and is not used "
           "for H5's verdict."]
    for p in paths:
        res = json.load(open(p, encoding="utf-8"))
        if res.get("schema") != SCHEMA:
            sys.exit("%s: unknown schema %r" % (p, res.get("schema")))
        out.extend(section(os.path.splitext(os.path.basename(p))[0], res))
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
