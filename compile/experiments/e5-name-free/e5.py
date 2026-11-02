#!/usr/bin/env python3
"""E5, R: name-free identity across builds: build, hash, compare, write JSON.

For every E4 kernel (C and Rust, six levels each) and every C and Rust
program of E1 (six levels each), the harness

1. writes the source into directory A, the same bytes into directory B, and
   the renamed source (rename.py) into directory N;
2. builds each declared output in each of the three directories: for the
   E4 kernels an x86-64 (host) object, an arm64 Mach-O object (C only), a
   WebAssembly object and a linked WebAssembly module; for E1's programs a
   host object;
3. checks that the renaming kept the program: every renamed module gives
   reference.py's value for each of the kernel's declared inputs, and every
   renamed E1 program prints what the original prints;
4. computes six keys per output: the two baselines (output bytes, source
   key) and the four R hashes (machine-module, machine-run, ir-module,
   ir-run), as applicable to the output;
5. compares the keys for each declared comparison of matrix.json, and for
   the mutation comparison changes each A module one instruction at a time
   and hashes it again.

The results file holds the machine, the toolchain versions, the SHA-256 of
every harness file, one row per build and the comparisons; RESULTS.md is
generated from it by report.py.

    python3 e5.py run RESULTS_JSON [WORK_DIR]

WORK_DIR (default: a new directory under the system temporary directory,
named by its real path) holds the builds and is deleted at the end.
"""
import hashlib
import itertools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.join(HERE, "..", "e4-energy")
E1 = os.path.join(HERE, "..", "e1-determinism")
sys.path.insert(0, HERE)
sys.path.insert(0, E4)

import irhash  # noqa: E402
import machine  # noqa: E402
import reference  # noqa: E402
import rename  # noqa: E402
import wasm_eps  # noqa: E402

SCHEMA = "rce-compile-e5-results/1"
TIMEOUT = 900
VARIANTS = ("A", "B", "N")
KEYS = {
    "native-object": ("output-bytes", "source-key", "machine-module", "machine-run"),
    "macho-arm64-object": ("output-bytes", "source-key", "machine-module", "machine-run"),
    "wasm-object": ("output-bytes", "source-key", "machine-module", "machine-run"),
    "wasm-module": ("output-bytes", "source-key", "ir-module", "ir-run"),
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def first_line(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "unavailable: %s" % e.__class__.__name__
    out = (p.stdout or p.stderr).strip().splitlines()
    return out[0].strip() if out else "exit %d, no output" % p.returncode


def machine_info():
    cpu, osrel = "", ""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if not cpu and platform.system() == "Darwin":
        cpu = first_line(["sysctl", "-n", "machdep.cpu.brand_string"])
    try:
        for line in open("/etc/os-release", encoding="utf-8"):
            if line.startswith("PRETTY_NAME="):
                osrel = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    if not osrel and platform.system() == "Darwin":
        osrel = "macOS " + platform.mac_ver()[0]
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "cpu": cpu, "cpus": os.cpu_count(), "os": osrel, "python": platform.python_version()}


def subst(argv, **kw):
    out = []
    for a in argv:
        for k, v in kw.items():
            a = a.replace("{%s}" % k, v)
        out.append(a)
    return out


def execute(argv, cwd):
    env = dict(os.environ, LC_ALL="C", TZ="UTC", SOURCE_DATE_EPOCH="1735689600")
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "%s: %s" % (e.__class__.__name__, e), b""
    if p.returncode != 0:
        msg = (p.stderr or p.stdout).decode("utf-8", "replace").strip().splitlines()
        return "exit %d: %s" % (p.returncode, msg[-1] if msg else ""), p.stdout
    return None, p.stdout


# -- one output -------------------------------------------------------------------------

def keys_of(kind, data, src_bytes, shown):
    """The keys of one built output."""
    k = {"output-bytes": sha256_bytes(data),
         "source-key": sha256_bytes(src_bytes + b"\0" + "\0".join(shown).encode())}
    info = {}
    if kind == "wasm-module":
        try:
            m = irhash.Module(data)
            k["ir-module"], im = irhash.r_hash(m)
            try:
                k["ir-run"], ir = irhash.r_hash(m, "run")
            except irhash.Unsupported as e:
                k["ir-run"], ir = None, {"error": str(e)}
            info = {"status": "hashed", "rounds": im["rounds"], "tied_functions": im["tied_functions"],
                    "functions": im["functions"], "run_functions": ir.get("functions"), "ir_nodes": m.ir_nodes()}
        except irhash.Unsupported as e:
            k["ir-module"] = k["ir-run"] = None
            info = {"status": "not hashed: %s" % e}
    else:
        res = machine.object_ids(data)
        k["machine-module"] = machine.module_id(res)
        k["machine-run"] = machine.entry_id(res, "run")
        info = {"status": res["status"], "format": res["format"], "arch": res["arch"],
                "functions": len(res["functions"]), "blocks": sum(f["blocks"] for f in res["functions"]),
                "relocated_instructions": sum(f["relocs"] for f in res["functions"])}
    return k, info


def objdump_agreement(path, data):
    """Instruction starts of machine.x86_insn against llvm-objdump's, per function of an x86-64 ELF object."""
    tool = shutil.which("llvm-objdump")
    if tool is None:
        return None
    try:
        arch, funcs = machine.elf_functions(data)
    except machine.Undecoded:
        return None
    if arch != "x86_64" or not funcs:
        return None
    try:
        out = subprocess.run([tool, "-d", path], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    offs, cur, base = {}, None, 0
    for line in out.splitlines():
        mt = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if mt:
            cur, base = mt.group(2), int(mt.group(1), 16)
            offs[cur] = []
            continue
        mt = re.match(r"^\s+([0-9a-f]+):\s", line)
        if mt and cur is not None:
            offs[cur].append(int(mt.group(1), 16) - base)
    agree = 0
    for name, code, _r in funcs:
        mine, p = [], 0
        try:
            while p < len(code):
                n, _k, _t = machine.x86_insn(code, p)
                mine.append(p)
                p += n
        except machine.Undecoded:
            continue
        if mine == [o for o in offs.get(name, []) if o < len(code)]:
            agree += 1
    return {"functions": len(funcs), "agree": agree}


# -- the corpus ----------------------------------------------------------------------------

def corpus(matrix, e4m):
    out = []
    for p in e4m["programs"]:
        out.append({"id": p["id"], "set": "E4", "language": p["language"], "toolchain": p["toolchain"],
                    "kernel": p["kernel"], "source": os.path.join(E4, "programs", p["source"]),
                    "inputs": p["inputs"], "outputs": "outputs"})
    for p in matrix["e1_programs"]:
        out.append({"id": p["id"], "set": "E1", "language": p["language"], "toolchain": p["toolchain"],
                    "kernel": None, "source": os.path.join(E1, "programs", p["source"]),
                    "inputs": None, "outputs": "e1_outputs"})
    return out


def check_rename(prog, tc, tool, srcs, work):
    """Evidence that the renamed source is the same program; returns a dict."""
    if prog["set"] == "E4":
        return {"status": "checked per level on the WebAssembly module (see each N row)"}
    outs = {}
    for v in ("A", "N"):
        d = os.path.join(work, v, prog["id"].replace("/", "-"), "exe")
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, os.path.basename(prog["source"]))
        with open(src, "wb") as f:
            f.write(srcs[v])
        exe = os.path.join(d, "prog")
        err, _ = execute(subst(tc["executable"], tool=tool, src=src, out=exe), d)
        if err:
            return {"status": "not checked", "error": "%s build: %s" % (v, err)}
        err, out = execute([exe], d)
        if err:
            return {"status": "not checked", "error": "%s run: %s" % (v, err)}
        outs[v] = out.decode("utf-8", "replace")
    return {"status": "same output" if outs["A"] == outs["N"] else "output differs", "output_A": outs["A"],
            "output_N": outs["N"]}


def wasm_values(data, inputs):
    vals = []
    try:
        m = wasm_eps.Module(data)
        for n in inputs:
            vals.append(wasm_eps.Run(m).call_export("run", [n])[0])
    except (wasm_eps.Unsupported, wasm_eps.Trap, RecursionError) as e:
        return None, "%s: %s" % (e.__class__.__name__, e)
    return vals, None


def build_all(matrix, progs, tools, work):
    rows, renames = [], {}
    for prog in progs:
        tcname = prog["toolchain"]
        if tcname not in tools:
            continue
        tc, tool = matrix["toolchains"][tcname], tools[tcname]
        text = open(prog["source"], encoding="utf-8").read()
        rtext, mapping = rename.rename(tcname, text)
        srcs = {"A": text.encode(), "B": text.encode(), "N": rtext.encode()}
        renames[prog["id"]] = {"mapping": mapping, "renamed_sha256": sha256_bytes(srcs["N"]),
                               "source_sha256": sha256_bytes(srcs["A"]),
                               "evidence": check_rename(prog, tc, tool, srcs, work)}
        for level in tc["levels"]:
            for variant in VARIANTS:
                d = os.path.join(work, variant, prog["id"].replace("/", "-"), level.strip("-"))
                os.makedirs(d, exist_ok=True)
                src = os.path.join(d, os.path.basename(prog["source"]))
                with open(src, "wb") as f:
                    f.write(srcs[variant])
                for kind, argv in tc[prog["outputs"]].items():
                    out = os.path.join(d, kind + (".wasm" if kind == "wasm-module" else ".o"))
                    full = subst(argv, tool=tool, opt=level, src=src, out=out)
                    shown = [a.replace(d, "{dir}") for a in full]
                    row = {"program": prog["id"], "set": prog["set"], "language": prog["language"],
                           "kernel": prog["kernel"], "toolchain": tcname, "level": level,
                           "config": tc["config"].replace("{opt}", level), "variant": variant, "output": kind,
                           "argv": shown}
                    err, _ = execute(full, d)
                    if err:
                        row.update({"status": "not built", "error": err, "keys": {}})
                        rows.append(row)
                        continue
                    data = open(out, "rb").read()
                    row["status"] = "built"
                    row["bytes"] = len(data)
                    row["keys"], row["hash_info"] = keys_of(kind, data, srcs[variant], shown)
                    if kind == "native-object" and variant == "A":
                        row["decoder_check"] = objdump_agreement(out, data)
                    if kind == "wasm-module" and prog["inputs"]:
                        vals, verr = wasm_values(data, prog["inputs"])
                        exp = [reference.KERNELS[prog["kernel"]](n) for n in prog["inputs"]]
                        row["evidence"] = {"inputs": prog["inputs"], "expected": exp, "values": vals, "error": verr,
                                           "status": "equal to the reference" if vals == exp else "differs"}
                    if kind == "wasm-module" and variant == "A":
                        row["_data"] = data
                    rows.append(row)
                print("%-18s %-18s built %s" % (prog["id"], tc["config"].replace("{opt}", level),
                      ", ".join(sorted(set(r["output"] for r in rows if r["program"] == prog["id"] and r["level"] == level
                                           and r["status"] == "built")))), flush=True)
    return rows, renames


# -- comparisons -------------------------------------------------------------------------------

def pair(a, b):
    """For each key both rows have: True when equal."""
    if a["status"] != "built" or b["status"] != "built":
        return None
    out = {}
    for k in KEYS[a["output"]]:
        x, y = a["keys"].get(k), b["keys"].get(k)
        out[k] = None if x is None or y is None else x == y
    return out


def tally(pairs):
    """{key: {"pairs": n, "equal": e, "unhashed": u}} over a list of pair() results."""
    t = {}
    for p in pairs:
        if p is None:
            continue
        for k, v in p.items():
            s = t.setdefault(k, {"pairs": 0, "equal": 0, "unhashed": 0})
            if v is None:
                s["unhashed"] += 1
            else:
                s["pairs"] += 1
                s["equal"] += 1 if v else 0
    return t


def compare(rows, matrix):
    idx = {}
    for r in rows:
        idx[(r["program"], r["level"], r["variant"], r["output"])] = r
    outputs = sorted(set(r["output"] for r in rows))
    comp = {}

    def group(name, sets):
        comp[name] = {"declared": matrix["comparisons"][name]["declared"], "by_output": {}, "examples": {}}
        for (out, label), ps in sets.items():
            res = [pair(a, b) for a, b in ps]
            comp[name]["by_output"].setdefault(out, {})[label] = tally(res)
            ex = []
            for (a, b), p in zip(ps, res):
                if p is None:
                    continue
                declared_equal = matrix["comparisons"][name]["declared"] == "equal"
                for k, v in p.items():
                    if v is not None and v != declared_equal and k not in ("output-bytes", "source-key"):
                        ex.append({"key": k, "a": "%s %s %s" % (a["program"], a["config"], a["variant"]),
                                   "b": "%s %s %s" % (b["program"], b["config"], b["variant"])})
            comp[name]["examples"].setdefault(out, {})[label] = ex[:200]

    progs = sorted(set(r["program"] for r in rows))
    levels = {}
    for r in rows:
        levels.setdefault(r["program"], [])
        if r["level"] not in levels[r["program"]]:
            levels[r["program"]].append(r["level"])

    def sets_for(fn):
        s = {}
        for out in outputs:
            for st in ("E4", "E1"):
                lst = fn(out, st)
                if lst:
                    s[(out, st)] = lst
        return s

    def two(v):
        def fn(out, st):
            return [(idx[(p, lv, "A", out)], idx[(p, lv, v, out)]) for p in progs for lv in levels[p]
                    if (p, lv, "A", out) in idx and (p, lv, v, out) in idx and idx[(p, lv, "A", out)]["set"] == st]
        return fn
    group("two_dir", sets_for(two("B")))
    group("renamed", sets_for(two("N")))

    def opts(out, st):
        return [(idx[(p, l1, "A", out)], idx[(p, l2, "A", out)]) for p in progs
                for l1, l2 in itertools.combinations(levels[p], 2)
                if (p, l1, "A", out) in idx and (p, l2, "A", out) in idx and idx[(p, l1, "A", out)]["set"] == st]
    group("opt_levels", sets_for(opts))

    kern = sorted(set(r["kernel"] for r in rows if r["kernel"]))

    def cvr(out, st):
        if st != "E4":
            return []
        ps = []
        for k in kern:
            c, rs = "c/" + k, "rust/" + k
            if c not in levels or rs not in levels:
                continue
            for lc, lr in zip(levels[c], levels[rs]):
                if (c, lc, "A", out) in idx and (rs, lr, "A", out) in idx:
                    ps.append((idx[(c, lc, "A", out)], idx[(rs, lr, "A", out)]))
        return ps
    group("c_vs_rust", sets_for(cvr))

    def diffk(out, st):
        if st != "E4":
            return []
        a = [r for r in rows if r["output"] == out and r["variant"] == "A" and r["set"] == "E4"]
        return [(x, y) for x, y in itertools.combinations(a, 2) if x["kernel"] != y["kernel"]]
    group("different_kernels", sets_for(diffk))
    return comp


def mutation_runs(rows):
    """Change each A module one instruction at a time; count the hashes left unchanged."""
    out = {"modules": 0, "mutations": 0, "unchanged": 0, "by_kind": {}, "unchanged_examples": [], "not_hashed": 0}
    for r in rows:
        data = r.pop("_data", None)
        if data is None or r["keys"].get("ir-module") is None:
            continue
        m = irhash.Module(data)
        base = r["keys"]["ir-module"]
        out["modules"] += 1
        for mut in irhash.mutations(m):
            undo = irhash.apply_mutation(m, mut)
            try:
                h, _ = irhash.r_hash(m)
            except irhash.Unsupported:
                out["not_hashed"] += 1
                undo()
                continue
            undo()
            s = out["by_kind"].setdefault(mut[2], {"mutations": 0, "unchanged": 0})
            s["mutations"] += 1
            out["mutations"] += 1
            if h == base:
                s["unchanged"] += 1
                out["unchanged"] += 1
                if len(out["unchanged_examples"]) < 50:
                    out["unchanged_examples"].append("%s %s function %d instruction %d (%s)" % (
                        r["program"], r["config"], mut[0], mut[1], mut[2]))
        assert irhash.r_hash(m)[0] == base
    return out


def main():
    args = sys.argv[1:]
    if len(args) not in (2, 3) or args[0] != "run":
        sys.exit("usage: python3 e5.py run RESULTS_JSON [WORK_DIR]")
    sys.setrecursionlimit(100000)
    out = os.path.abspath(args[1])
    matrix = json.load(open(os.path.join(HERE, "matrix.json"), encoding="utf-8"))
    e4m = json.load(open(os.path.join(E4, "matrix.json"), encoding="utf-8"))
    own = len(args) == 2
    work = os.path.realpath(tempfile.mkdtemp(prefix="e5-") if own else os.path.abspath(args[2]))
    os.makedirs(work, exist_ok=True)
    start = utc()
    try:
        tools, absent = {}, []
        for name, tc in matrix["toolchains"].items():
            if shutil.which(tc["tool"]):
                tools[name] = tc["tool"]
            else:
                absent.append({"toolchain": name, "reason": "%s is not on PATH" % tc["tool"]})
        progs = corpus(matrix, e4m)
        rows, renames = build_all(matrix, progs, tools, work)
        comps = compare(rows, matrix)
        print("mutations ...", flush=True)
        muts = mutation_runs(rows)
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)
    for r in rows:
        r.pop("_data", None)
    files = ["e5.py", "irhash.py", "machine.py", "rename.py", "matrix.json"]
    result = {
        "schema": SCHEMA,
        "date_utc": utc(),
        "start_utc": start,
        "machine": machine_info(),
        "harness": dict({f: sha256_file(os.path.join(HERE, f)) for f in files},
                        **{"../e4-energy/wasm_eps.py": sha256_file(os.path.join(E4, "wasm_eps.py")),
                           "../e4-energy/reference.py": sha256_file(os.path.join(E4, "reference.py")),
                           "../e4-energy/matrix.json": sha256_file(os.path.join(E4, "matrix.json")),
                           "../e1-determinism/e1.py": sha256_file(os.path.join(E1, "e1.py"))}),
        "toolchains": {k: {"tool": v, "version": first_line(subst(matrix["toolchains"][k]["version"], tool=v))}
                       for k, v in tools.items()},
        "absent": absent,
        "llvm_objdump": first_line(["llvm-objdump", "--version"]) if shutil.which("llvm-objdump") else None,
        "nops": machine.NOPS,
        "renames": renames,
        "rows": rows,
        "comparisons": comps,
        "mutations": muts,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    print("wrote %s: %d builds" % (out, len(rows)))


if __name__ == "__main__":
    main()
