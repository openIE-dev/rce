#!/usr/bin/env python3
"""Generate the E4 write-up from E4 results files; never edit its output by hand.

    python3 report.py write                      RESULTS.md from results/*.json
    python3 report.py check                      exit 1 if RESULTS.md is not what write would print
    python3 report.py write OUT_MD RESULTS_JSON...   the same for named files (used by CI)

Every number, label and verdict in the output is computed here from the
results files that e4.py wrote, with the analysis declared in PROTOCOL.md
(stats.py).
"""
import glob
import json
import os
import sys

import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "rce-compile-e4-results/1"


def g(x, digits=4):
    if x is None:
        return "-"
    if isinstance(x, int):
        return format(x, ",")
    return "%.*g" % (digits, x)


def interval(iv, digits=3):
    return "-" if iv is None else "[%s, %s]" % (g(iv[0], digits), g(iv[1], digits))


def esc(s):
    return str(s).replace("|", "\\|")


def shown(p):
    rel = os.path.relpath(os.path.abspath(p), HERE)
    return os.path.basename(p) if rel.startswith("..") else rel


def energy_text(e):
    if e["kind"] == "unmeasured":
        return ["**Energy: unmeasured.** %s. This file holds ε and λ only; it contains no joules, and no "
                "energy analysis is computed from it." % e["reason"].rstrip(".")]
    out = ["**Energy: %s** (an on-chip or operating-system counter, not a meter on the supply), regime %s." % (e["kind"], e["regime"])]
    out.append("")
    out.append("| Provenance | Value |")
    out.append("|---|---|")
    out.append("| Sampler | %s |" % esc(e["sampler"]))
    if "macmon macmon" in str(e.get("sampler", "")):
        out.append("| Sampler note | labelling defect of commit 057b6739: the string repeats the name; the sampler is macmon and the version is 0.10.0 |")
    out.append("| Counter | %s |" % esc(e["counter"]))
    out.append("| Interval | %s |" % esc(e["interval"]))
    out.append("| Boundary | %s |" % esc(e["boundary"]))
    if e.get("zones"):
        out.append("| Zones | %s |" % "; ".join("`%s` %s, `max_energy_range_uj` %s, wraps %d" % (
            z["zone"], esc(z["domain"]), g(z["max_energy_range_uj"]), z["wraps"]) for z in e["zones"]))
        out.append("| Wraps | %d |" % sum(z["wraps"] for z in e["zones"]))
    if e.get("processor_keys_of_first_sample") is not None:
        out.append("| Keys of the first sample | %s |" % esc(", ".join(e["processor_keys_of_first_sample"])))
    out.append("| Idle baseline | %s W before (%s s), %s W after (%s s); %s W used |" % (
        g(e["idle_before"]["watts"]), g(e["idle_before"]["seconds"], 3), g(e["idle_after"]["watts"]),
        g(e["idle_after"]["seconds"], 3), g(e["idle_watts"])))
    out.append("| Net energy | %s |" % esc(e["net"]))
    out.append("| Start, end (UTC) | %s, %s |" % (e["start_utc"], e["end_utc"]))
    return out


def analysis_rows(res):
    rows = [r for r in res["rows"] if r["eps"]["status"] == "counted" and r["energy"].get("joules_net_per_call") is not None
            and res["lambda"]["primary"] in r["lambda"]]
    if not rows:
        return []
    total_j = sum(r["energy"]["joules_net_per_call"] for r in rows)
    total_s = sum(r["energy"]["seconds_per_call"] for r in rows)
    p = total_j / total_s
    return [{"program": r["program"], "joules": r["energy"]["joules_net_per_call"], "eps": r["eps"]["bits"],
             "lam": r["lambda"][res["lambda"]["primary"]], "base": r["energy"]["seconds_per_call"] * p} for r in rows], p


def analysis(res, acfg):
    out = ["### Declared analysis", ""]
    got = analysis_rows(res)
    if not got:
        out.append("Not computed: this file has no energy figures.")
        out.append("")
        return out
    rows, p = got
    try:
        a = stats.regime_b(rows, acfg["resamples"], acfg["seed"], acfg["level"])
    except stats.Singular as e:
        out.append("Not computed: %s." % e)
        out.append("")
        return out
    iv = a["interval"]
    out.append("Regime %s, %d rows from %d programs; response: net joules per call; ε and λ (`%s`) against the "
               "baseline, seconds per call times %s W (the file's mean net power). Share of variance is adjusted R²; "
               "intervals are %d%% percentiles of %d resamples of programs (seed %d; %d singular draws skipped)." % (
                   res["energy"]["regime"], a["rows"], a["programs"], res["lambda"]["primary"], g(p), round(100 * a["level"]),
                   a["resamples"], a["seed"], a["skipped"]))
    out.append("")
    out.append("| Model | R² | Adjusted R² | Interval |")
    out.append("|---|---|---|---|")
    out.append("| joules on ε and λ | %s | %s | %s |" % (g(a["point"]["eps_lam_r2"]), g(a["point"]["eps_lam"]), interval(iv["eps_lam"])))
    out.append("| joules on the baseline | %s | %s | %s |" % (g(a["point"]["baseline_r2"]), g(a["point"]["baseline"]), interval(iv["baseline"])))
    out.append("| baseline minus ε and λ | | %s | %s |" % (g(a["point"]["baseline_minus_eps_lam"]), interval(iv["baseline_minus_eps_lam"])))
    out.append("")

    def side(v):
        if v is None:
            return "undetermined (no interval)"
        if v[0] > 0:
            return "above zero"
        if v[1] < 0:
            return "below zero"
        return "includes zero"
    out.append("By the declared rule: the interval of the share explained by ε and λ is %s, so ε and λ %s; "
               "the interval of the baseline's share minus theirs is %s, so the baseline explains %s." % (
                   side(iv["eps_lam"]), "explain variance" if side(iv["eps_lam"]) == "above zero" else "do not explain variance by that rule",
                   side(iv["baseline_minus_eps_lam"]),
                   {"above zero": "more than ε and λ", "below zero": "less than ε and λ"}.get(side(iv["baseline_minus_eps_lam"]), "neither more nor less, by that rule")))
    out.append("")
    ratios = sorted(r["energy"]["joules_net_per_call"] / r["eps"]["landauer_joules"] for r in res["rows"]
                    if r["eps"]["status"] == "counted" and r["eps"]["bits"] and r["energy"].get("joules_net_per_call"))
    if ratios:
        out.append("Net joules per call over the Landauer bound of the call's ε: median %s, range %s to %s." % (
            g(ratios[len(ratios) // 2], 3), g(ratios[0], 3), g(ratios[-1], 3)))
        out.append("")
    return out


def section(name, res, acfg):
    m = res["machine"]
    out = ["## %s" % name, ""]
    out.append("Run of %s (UTC). Machine: %s %s, %s, %s CPUs, %s; Python %s; zlib %s." % (
        res["date_utc"], m["system"], m["machine"], m["cpu"] or "CPU model not reported", m["cpus"],
        m["os"] or "OS release not reported", m["python"], m["zlib"]))
    out.append("Harness: " + ", ".join("`%s` sha256 `%s`" % (f, h[:16]) for f, h in sorted(res["harness"].items())) + ".")
    out.append("")
    out.extend(energy_text(res["energy"]))
    out.append("")
    out.append("| Toolchain | Compiler | Version | ε counted |")
    out.append("|---|---|---|---|")
    for k in sorted(res["toolchains"]):
        t = res["toolchains"][k]
        out.append("| %s | `%s` | %s | %s |" % (k, esc(t["tool"]), esc(t["version"]), "yes" if t["wasm"] else "no: " + esc(t["wasm_reason"])))
    for a in res["absent"]:
        out.append("| %s | - | - | absent: %s |" % (a["toolchain"], esc(a["reason"])))
    out.append("")
    T = res["landauer"]["temperature_k"]
    measured = res["energy"]["kind"] != "unmeasured"
    head = ["Program", "Configuration", "Equivalence", "ε (bits per call)", "Landauer bound (J, modelled, %s K)" % g(T),
            "Executed wasm instructions", "λ: IR nodes", "wasm bytes", "wasm zlib bytes", "native bytes", "native zlib bytes",
            "Native build (s)"]
    if measured:
        head += ["Build energy, net (J, %s)" % res["energy"]["kind"], "Energy per call, net (J, %s)" % res["energy"]["kind"],
                 "95% interval", "Seconds per call", "Calls per run, runs"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for r in res["rows"]:
        e, lam, b = r["eps"], r["lambda"], r["build_native"]
        cells = ["`%s`" % r["program"], "`%s`" % esc(r["config"]), r["equivalence"]["status"],
                 g(e["bits"]) if e["status"] == "counted" else "not counted",
                 g(e.get("landauer_joules"), 3), g(e.get("ops")), g(lam.get("ir_nodes")), g(lam.get("wasm_bytes")),
                 g(lam.get("wasm_zlib_bytes")), g(lam.get("native_bytes")), g(lam.get("native_zlib_bytes")),
                 g(sum(b["seconds"]) / len(b["seconds"]), 3) if b["seconds"] else b["status"]]
        if measured:
            en = r["energy"]
            jn = b.get("joules_net")
            cells += [g(sum(jn) / len(jn), 3) if jn else "-"]
            if en.get("joules_net_per_call") is not None:
                cells += [g(en["joules_net_per_call"], 3), interval(en["joules_net_per_call_interval"]),
                          g(en["seconds_per_call"], 3), "%s, %d" % (g(en["calls_per_run"]), en["runs_with_reference_output"])]
            else:
                cells += [en.get("status", "-") + (": " + esc(en["reason"]) if en.get("reason") else ""), "-", "-", "-"]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    notes = [r for r in res["rows"] if r["eps"]["status"] != "counted" or r["build_native"]["status"] != "built"]
    if notes:
        out.append("Rows without ε or without a native build: " + "; ".join(
            "`%s` `%s` (%s)" % (r["program"], esc(r["config"]), esc(r["eps"].get("reason") or r["build_native"].get("error"))) for r in notes) + ".")
        out.append("")
    if measured:
        per = {}
        for r in res["rows"]:
            jn = r["build_native"].get("joules_net")
            if jn:
                per.setdefault(r["config"], []).append(sum(jn) / len(jn))
        if per:
            out.append("Compile-time energy per configuration, the sum over its programs of the mean net joules of one native build:")
            out.append("")
            out.append("| Configuration | Programs | Net joules (%s) |" % res["energy"]["kind"])
            out.append("|---|---|---|")
            for c in sorted(per):
                out.append("| `%s` | %d | %s |" % (esc(c), len(per[c]), g(sum(per[c]), 3)))
            out.append("")
    eq = {}
    for r in res["rows"]:
        eq.setdefault(r["program"], set()).add(r["equivalence"]["status"])
    same = sorted(p for p, s in eq.items() if s == {"equal to the reference"})
    out.append("Equivalence evidence: %d of %d programs give the reference value on every declared input under every "
               "configuration, natively and as WebAssembly%s." % (
                   len(same), len(eq), "" if len(same) == len(eq) else "; the others: " + ", ".join(
                       "`%s` (%s)" % (p, "; ".join(sorted(eq[p]))) for p in sorted(set(eq) - set(same)))))
    out.append("")
    out.extend(analysis(res, acfg))
    out.append("Regime A (counted operations on an in-order core or FPGA with a supply meter) and regime C "
               "(entropy production) are not measured in this file: it has no counter of either.")
    out.append("")
    return out


def across(pa, ra, pb, rb):
    na, nb = os.path.splitext(os.path.basename(pa))[0], os.path.splitext(os.path.basename(pb))[0]
    out = ["## Across files: %s and %s" % (na, nb), ""]
    same_tc = [k for k in ra["toolchains"] if k in rb["toolchains"] and ra["toolchains"][k]["version"] == rb["toolchains"][k]["version"]]
    ib = {(r["program"], r["config"]): r for r in rb["rows"]}
    n = eq = 0
    diff = []
    for r in ra["rows"]:
        s = ib.get((r["program"], r["config"]))
        if r["toolchain"] not in same_tc or s is None or r["eps"]["status"] != "counted" or s["eps"]["status"] != "counted":
            continue
        n += 1
        if r["eps"]["bits"] == s["eps"]["bits"] and r["lambda"].get("ir_nodes") == s["lambda"].get("ir_nodes"):
            eq += 1
        else:
            diff.append("`%s` `%s`" % (r["program"], esc(r["config"])))
    if not n:
        out.append("No row is compared: no toolchain has the same version string in both files.")
    else:
        out.append("Rows of toolchains with the same version string in both files (%s): %d compared, %d with equal ε and "
                   "equal IR node count%s." % (", ".join(same_tc), n, eq, "" if not diff else "; differing: " + ", ".join(diff)))
    out.append("")
    return out


def render(paths):
    matrix = json.load(open(os.path.join(HERE, "matrix.json"), encoding="utf-8"))
    out = [
        "# E4 results: erasure and description length against energy",
        "",
        "Generated by `report.py` from %s. Do not edit; run `python3 report.py write`." % ", ".join("`%s`" % shown(p) for p in paths),
        "The method, the definitions of ε and λ and the declared analysis are in [PROTOCOL.md](PROTOCOL.md).",
        "Kinds: *reported* is an on-chip or operating-system counter; *measured* is a meter on the supply; *modelled* is "
        "computed from a model or a declared constant; *unmeasured* means no counter was read.",
        "ε is counted on the WebAssembly lowering of each kernel at the configuration's optimisation level, not on the "
        "native machine code whose energy is read. The primary λ proxy is `%s`." % matrix["lambda"]["primary"],
        "",
    ]
    loaded = []
    for p in paths:
        res = json.load(open(p, encoding="utf-8"))
        if res.get("schema") != SCHEMA:
            sys.exit("%s: unknown schema %r" % (p, res.get("schema")))
        out.extend(section(os.path.splitext(os.path.basename(p))[0], res, matrix["analysis"]))
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
