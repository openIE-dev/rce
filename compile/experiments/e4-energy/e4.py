#!/usr/bin/env python3
"""E4, erasure and description length against energy: build, count, measure, write JSON.

For every program of matrix.json and every optimisation level of its
toolchain, the harness

1. builds the native program (kernel and driver) and the WebAssembly module
   of the kernel, timing each native build and, where a counter exists,
   reading its energy over `compile_repeats` builds;
2. checks the equivalence evidence: the native program and the module give
   reference.py's value for every declared input;
3. counts ε on the module for the call run(eps_input) with wasm_eps.py, and
   computes λ (the IR node count of the module, its bytes and zlib-9 bytes,
   the native program's bytes and zlib-9 bytes);
4. where a counter exists, reads the idle power, then runs each native
   program `repeats` times, each run making R calls of run(eps_input) with R
   set so that a run takes about `target_seconds`, in an order shuffled per
   round by a declared seed, and reads the idle power again.

Energy comes from RAPL on Linux and macmon on macOS (meters.py). On a
machine with neither, every energy field is "unmeasured" and the rest is
recorded as usual. The results file holds the machine, the toolchain
versions, the counter's provenance and one row per program and level;
RESULTS.md is generated from it by report.py.

    python3 e4.py run RESULTS_JSON [WORK_DIR]

WORK_DIR (default: a new directory under the system temporary directory)
holds the builds and is deleted at the end. On macOS, macmon runs without
root; without macmon, powermetrics is used only when sudo is already
cached, and otherwise energy is recorded as unmeasured.
"""
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

import meters
import reference
import stats
import wasm_eps

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "rce-compile-e4-results/1"
K_B = 1.380649e-23  # J/K, exact in the 2019 SI
TIMEOUT = 900
PROBE_C = "unsigned run(unsigned n) { return n + 1u; }\n"
PROBE_RS = ('#![cfg_attr(target_arch = "wasm32", no_std)]\n'
            '#[no_mangle]\npub extern "C" fn run(n: u32) -> u32 { n.wrapping_add(1) }\n'
            '#[cfg(target_arch = "wasm32")]\n#[panic_handler]\nfn panic(_: &core::panic::PanicInfo) -> ! { loop {} }\n'
            '#[cfg(not(target_arch = "wasm32"))]\nfn main() {}\n')


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def first_line(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "unavailable: %s" % e.__class__.__name__
    out = (p.stdout or p.stderr).strip().splitlines()
    return out[0].strip() if out else "exit %d, no output" % p.returncode


def machine():
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
            "cpu": cpu, "cpus": os.cpu_count(), "os": osrel, "python": platform.python_version(),
            "zlib": zlib.ZLIB_RUNTIME_VERSION}


def subst(argv, **kw):
    out = []
    for a in argv:
        for k, v in kw.items():
            a = a.replace("{%s}" % k, v)
        out.append(a)
    return out


def execute(argv, cwd):
    """Run argv; return (seconds, error or None, stdout)."""
    env = dict(os.environ, LC_ALL="C", TZ="UTC")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        return time.perf_counter() - t0, "%s: %s" % (e.__class__.__name__, e), b""
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        msg = (p.stderr or p.stdout).decode("utf-8", "replace").strip().splitlines()
        return dt, "exit %d: %s" % (p.returncode, msg[-1] if msg else ""), p.stdout
    return dt, None, p.stdout


# -- the meter --------------------------------------------------------------------

def open_meter(cfg, work, argv_line):
    system = platform.system()
    if system == "Darwin":
        if meters.macmon_path():
            try:
                return meters.Macmon(work, cfg.get("macmon_interval_ms", 100))
            except LookupError as e:
                return meters.Unmeasured(str(e))
        if os.path.exists(meters.POWERMETRICS) and (os.geteuid() == 0 or meters.sudo_cached()):
            return meters.Powermetrics(work, cfg["powermetrics_interval_ms"])
        return meters.Unmeasured("macmon not found (brew install macmon)")
    if system == "Linux":
        try:
            return meters.Rapl()
        except meters.NeedsAccess as e:
            print("E4 reads energy from RAPL, whose counters only root can read since Linux 5.10.")
            print("This line makes them readable until the next boot; then run the same command again:")
            print("")
            print("    " + str(e))
            print("")
            sys.exit(2)
        except LookupError as e:
            return meters.Unmeasured("no RAPL counter: %s" % e)
    return meters.Unmeasured("no counter is read on %s" % system)


def idle(meter, seconds):
    m = meter.mark()
    t0 = time.perf_counter()
    time.sleep(seconds)
    j = meter.since(m)
    dt = time.perf_counter() - t0
    return {"seconds": dt, "joules": j, "watts": j / dt}


# -- toolchains -----------------------------------------------------------------------

def found(cand):
    return shutil.which(cand) or (os.path.isabs(cand) and os.access(cand, os.X_OK))


def pick_toolchain(name, tc, work):
    """Choose the compiler of a toolchain; return {tool, version, wasm: bool, wasm_reason}."""
    d = os.path.join(work, "probe-" + name)
    os.makedirs(d, exist_ok=True)
    if name == "c":
        src = os.path.join(d, "probe.c")
        open(src, "w").write(PROBE_C)
    else:
        src = os.path.join(d, "probe.rs")
        open(src, "w").write(PROBE_RS)
    driver = os.path.join(HERE, "programs", "c", "driver.c")
    native_ok, last = None, "no candidate on PATH: " + ", ".join(tc["candidates"])
    for cand in tc["candidates"]:
        if not found(cand):
            continue
        _, err, _ = execute(subst(tc["wasm"], tool=cand, opt=tc["levels"][0], src=src, out=os.path.join(d, "p.wasm")), d)
        if err is None:
            return {"tool": cand, "version": first_line(subst(tc["version"], tool=cand)), "wasm": True, "wasm_reason": None}
        last = "%s cannot build wasm32: %s" % (cand, err)
        if native_ok is None:
            _, nerr, _ = execute(subst(tc["native"], tool=cand, opt=tc["levels"][0], src=src, driver=driver,
                                       out=os.path.join(d, "p")), d)
            if nerr is None:
                native_ok = cand
    if native_ok:
        return {"tool": native_ok, "version": first_line(subst(tc["version"], tool=native_ok)), "wasm": False,
                "wasm_reason": last}
    return None


# -- λ ------------------------------------------------------------------------------------

def strip_custom(data):
    """The module without its custom sections (names, debug information, producers)."""
    out, p = bytearray(data[:8]), 8
    while p < len(data):
        sid = data[p]
        r = wasm_eps.Reader(data, p + 1)
        size = r.u()
        end = r.p + size
        if sid != 0:
            out += data[p:end]
        p = end
    return bytes(out)


def lambdas(module, wasm_bytes, native_bytes, level):
    out = {}
    if module is not None:
        s = strip_custom(wasm_bytes)
        out.update({"ir_nodes": module.ir_nodes(), "wasm_code_bytes": module.code_bytes, "wasm_bytes": len(s),
                    "wasm_zlib_bytes": len(zlib.compress(s, level))})
    if native_bytes is not None:
        out.update({"native_bytes": len(native_bytes), "native_zlib_bytes": len(zlib.compress(native_bytes, level))})
    return out


def shown_argv(argv, work):
    """argv with the harness directory written relative and the work directory as {work}."""
    return [os.path.relpath(a, HERE) if a.startswith(HERE) else a.replace(work, "{work}") for a in argv]


# -- one program under one level ---------------------------------------------------------

def build_row(prog, tcname, tc, chosen, level, work, meter, ecfg, lcfg, idle_w):
    k = prog["kernel"]
    d = os.path.join(work, "build", prog["id"].replace("/", "-"), level.strip("-") or "default")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(HERE, "programs", prog["source"])
    driver = os.path.join(HERE, "programs", "c", "driver.c")
    exe, wasm = os.path.join(d, k), os.path.join(d, k + ".wasm")
    row = {"program": prog["id"], "language": prog["language"], "kernel": k, "toolchain": tcname,
           "config": tc["config"].replace("{opt}", level), "level": level, "eps_input": prog["eps_input"]}

    # native build, repeated under the meter
    nargv = subst(tc["native"], tool=chosen["tool"], opt=level, src=src, driver=driver, out=exe)
    reps = ecfg["compile_repeats"] if meter.kind != "unmeasured" else 1
    secs, joules, err = [], [], None
    for _ in range(reps):
        m = meter.mark()
        dt, err, _ = execute(nargv, d)
        j = meter.since(m)
        if err:
            break
        secs.append(dt)
        joules.append(j)
    row["build_native"] = {"argv": shown_argv(nargv, work),
                           "status": "failed" if err else "built", "error": err, "seconds": secs}
    if meter.kind != "unmeasured" and not err:
        row["build_native"]["joules_gross"] = joules
        row["build_native"]["joules_net"] = [j - idle_w * s for j, s in zip(joules, secs)]
    native_bytes = open(exe, "rb").read() if not err else None

    # wasm build
    module, wasm_bytes = None, None
    if chosen["wasm"]:
        wargv = subst(tc["wasm"], tool=chosen["tool"], opt=level, src=src, out=wasm)
        _, werr, _ = execute(wargv, d)
        row["build_wasm"] = {"argv": shown_argv(wargv, work),
                             "status": "failed" if werr else "built", "error": werr}
        if not werr:
            wasm_bytes = open(wasm, "rb").read()
            try:
                module = wasm_eps.Module(wasm_bytes)
            except wasm_eps.Unsupported as e:
                row["build_wasm"]["status"] = "not interpretable"
                row["build_wasm"]["error"] = str(e)
    else:
        row["build_wasm"] = {"status": "not built", "error": chosen["wasm_reason"]}

    # equivalence evidence and ε
    expected = {n: reference.KERNELS[k](n) for n in prog["inputs"]}
    eq = {"inputs": prog["inputs"], "expected": [expected[n] for n in prog["inputs"]], "native": [], "wasm": []}
    if native_bytes is not None:
        for n in prog["inputs"]:
            _, e2, out = execute([exe, str(n), "1"], d)
            eq["native"].append(None if e2 else out.decode("ascii", "replace").split())
    eps = {"status": "not counted", "reason": row["build_wasm"].get("error")}
    if module is not None:
        for n in prog["inputs"]:
            try:
                run = wasm_eps.Run(module)
                t0 = time.perf_counter()
                val = run.call_export("run", [n])[0]
                dt = time.perf_counter() - t0
            except (wasm_eps.Unsupported, wasm_eps.Trap, RecursionError) as e:
                eq["wasm"].append(None)
                eps = {"status": "not counted", "reason": "%s: %s" % (e.__class__.__name__, e)}
                continue
            eq["wasm"].append(val)
            if n == prog["eps_input"]:
                eps = {"status": "counted", "bits": run.eps, "ops": run.ops, "seconds": dt}
    native_ok = len(eq["native"]) == len(prog["inputs"]) and all(
        o is not None and o == [str(expected[n]), str(expected[n])] for o, n in zip(eq["native"], prog["inputs"]))
    wasm_ok = len(eq["wasm"]) == len(prog["inputs"]) and all(
        v == expected[n] for v, n in zip(eq["wasm"], prog["inputs"]))
    if not eq["native"]:
        eq["status"] = "not checked"
    elif native_ok and (wasm_ok or module is None):
        eq["status"] = "equal to the reference" + ("" if module is not None else " (native only)")
    else:
        eq["status"] = "differs"
    row["equivalence"] = eq
    row["eps"] = eps
    row["lambda"] = lambdas(module, wasm_bytes, native_bytes, lcfg["compress_level"])
    row["exe"] = exe
    return row


def calibrate(exe, n, target, cwd):
    r = 1
    while True:
        dt, err, _ = execute([exe, str(n), str(r)], cwd)
        if err:
            return None, err
        if dt >= 0.2 or r >= 10 ** 9:
            return max(1, int(round(r * target / dt))), None
        r *= 10


def measure(rows, meter, ecfg, work):
    eligible = [r for r in rows if r["equivalence"]["status"].startswith("equal")]
    for r in rows:
        if r not in eligible:
            r["energy"] = {"status": "not measured", "reason": "equivalence evidence: " + r["equivalence"]["status"]}
    for r in eligible:
        calls, err = calibrate(r["exe"], r["eps_input"], ecfg["target_seconds"], work)
        r["energy"] = {"status": "failed", "reason": err} if err else {"status": "reported", "calls_per_run": calls, "runs": []}
    todo = [r for r in eligible if r["energy"]["status"] == "reported"]
    for rep in range(ecfg["repeats"]):
        order = sorted(todo, key=lambda r: (r["program"], r["level"]))
        random.Random(ecfg["order_seed"] + rep).shuffle(order)
        for r in order:
            e = r["energy"]
            m = meter.mark()
            dt, err, out = execute([r["exe"], str(r["eps_input"]), str(e["calls_per_run"])], work)
            j = meter.since(m)
            e["runs"].append({"round": rep, "seconds": dt, "joules_gross": j,
                              "output_first": (out.split() or [b""])[0].decode("ascii", "replace"), "error": err})
        print("energy round %d of %d done" % (rep + 1, ecfg["repeats"]), flush=True)


def summarise(rows, idle_w):
    for r in rows:
        e = r.get("energy")
        if not e or e.get("status") != "reported":
            continue
        want = str(reference.KERNELS[r["kernel"]](r["eps_input"]))
        good = [x for x in e["runs"] if x["error"] is None and x["output_first"] == want]
        e["runs_with_reference_output"] = len(good)
        if not good:
            e["status"] = "failed"
            e["reason"] = "no run gave the reference output"
            continue
        c = e["calls_per_run"]
        for x in good:
            x["joules_net"] = x["joules_gross"] - idle_w * x["seconds"]
        jm, ji = stats.mean_interval([x["joules_net"] / c for x in good])
        tm, ti = stats.mean_interval([x["seconds"] / c for x in good])
        gm, gi = stats.mean_interval([x["joules_gross"] / c for x in good])
        e.update({"joules_net_per_call": jm, "joules_net_per_call_interval": ji,
                  "joules_gross_per_call": gm, "joules_gross_per_call_interval": gi,
                  "seconds_per_call": tm, "seconds_per_call_interval": ti})


def main():
    args = sys.argv[1:]
    if len(args) not in (2, 3) or args[0] != "run":
        sys.exit("usage: python3 e4.py run RESULTS_JSON [WORK_DIR]")
    sys.setrecursionlimit(100000)
    out = os.path.abspath(args[1])
    matrix = json.load(open(os.path.join(HERE, "matrix.json"), encoding="utf-8"))
    ecfg, lcfg = matrix["energy"], matrix["lambda"]
    own_work = len(args) == 2
    work = tempfile.mkdtemp(prefix="e4-") if own_work else os.path.abspath(args[2])
    os.makedirs(work, exist_ok=True)
    line = "python3 " + " ".join([os.path.relpath(os.path.abspath(__file__))] + args)
    meter = None
    try:
        meter = open_meter(ecfg, work, line)
        start = utc()
        print("energy counter: %s" % (meter.kind if meter.kind != "unmeasured" else "none (%s)" % meter.reason), flush=True)
        idle_before = idle(meter, ecfg["idle_seconds"]) if meter.kind != "unmeasured" else None
        idle_w = idle_before["watts"] if idle_before else 0.0
        chosen, absent, rows = {}, [], []
        for name, tc in matrix["toolchains"].items():
            c = pick_toolchain(name, tc, work)
            if c is None:
                absent.append({"toolchain": name, "reason": "no candidate builds a native program: " + ", ".join(tc["candidates"])})
                print("absent  %s" % name, flush=True)
            else:
                chosen[name] = c
        for prog in matrix["programs"]:
            tcname = prog["toolchain"]
            if tcname not in chosen:
                continue
            tc = matrix["toolchains"][tcname]
            for level in tc["levels"]:
                row = build_row(prog, tcname, tc, chosen[tcname], level, work, meter, ecfg, lcfg, idle_w)
                rows.append(row)
                print("%-16s %-22s eq=%s eps=%s ir=%s" % (row["program"], row["config"], row["equivalence"]["status"],
                      row["eps"].get("bits", row["eps"]["status"]), row["lambda"].get("ir_nodes", "-")), flush=True)
        if meter.kind != "unmeasured":
            measure(rows, meter, ecfg, work)
            idle_after = idle(meter, ecfg["idle_seconds"])
            idle_w = (idle_before["watts"] + idle_after["watts"]) / 2
            summarise(rows, idle_w)
            energy = dict(meter.provenance(), regime=ecfg["regime"], idle_before=idle_before, idle_after=idle_after,
                          idle_watts=idle_w, start_utc=start, end_utc=utc(),
                          net="gross joules minus the mean idle power times the run's seconds")
        else:
            idle_after = None
            for r in rows:
                r["energy"] = {"status": "unmeasured"}
            energy = dict(meter.provenance(), regime=None, start_utc=start, end_utc=utc())
    finally:
        if meter is not None:
            meter.close()
        if own_work:
            shutil.rmtree(work, ignore_errors=True)
    for r in rows:
        r.pop("exe", None)
        e = r["eps"]
        if e["status"] == "counted":
            e["landauer_joules"] = e["bits"] * K_B * matrix["landauer"]["temperature_k"] * math.log(2)
    files = ["e4.py", "wasm_eps.py", "meters.py", "stats.py", "reference.py", "matrix.json"]
    result = {
        "schema": SCHEMA,
        "date_utc": utc(),
        "machine": machine(),
        "harness": {f: sha256_file(os.path.join(HERE, f)) for f in files},
        "toolchains": {k: {"tool": v["tool"], "version": v["version"], "wasm": v["wasm"], "wasm_reason": v["wasm_reason"]}
                       for k, v in chosen.items()},
        "absent": absent,
        "energy": energy,
        "landauer": matrix["landauer"],
        "lambda": lcfg,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    print("wrote %s: %d rows; energy %s" % (out, len(rows), energy["kind"]))


if __name__ == "__main__":
    main()
