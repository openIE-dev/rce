#!/usr/bin/env python3
"""E1, determinism of lowering: build each declared program several times and compare the bytes.

For every configuration of matrix.json whose tool is on PATH, and every
program it names, the harness makes these builds, each in a fresh directory
holding only the copied source:

    A1, A2   twice in directory A                       -> same_dir
    B1       once in directory B (another absolute path) -> cross_dir (A1 against B1)
    Ar, Br   in A and in B with the documented path remapping -> remapped
    C1, C2   twice in A with SOURCE_DATE_EPOCH unset, at least 1.1 s apart -> same_dir_clock

Every other build has the declared environment: PATH and HOME from the
caller, LC_ALL=C, TZ=UTC, SOURCE_DATE_EPOCH=1735689600, and the
configuration's own variables. Every file the build leaves in its directory,
other than the source, is an output. For each pair of builds the harness
records, per output file, the two sizes and the number of differing bytes
(byte positions below the longer length at which the files differ or one
file has ended). It also counts how many times the absolute build directory
appears in the outputs of B1 and of Br, and keeps the SHA-256 of every output
of A1 and of Ar, so that results files from two machines can be compared.

The results file is JSON and holds the machine, the toolchain versions and
one row per configuration and program. RESULTS.md is generated from it by
report.py; neither is written by hand.

    python3 e1.py run RESULTS_JSON [WORK_DIR]

WORK_DIR (default: a new directory under the system temporary directory)
holds the build directories and is deleted at the end.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "rce-compile-e1-results/1"
EPOCH = "1735689600"
TIMEOUT = 900


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def first_line(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "unavailable: %s" % e.__class__.__name__
    out = (p.stdout or p.stderr).strip().splitlines()
    return out[0].strip() if out else "exit %d, no output" % p.returncode


def machine():
    cpu = ""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    osrel = ""
    try:
        for line in open("/etc/os-release", encoding="utf-8"):
            if line.startswith("PRETTY_NAME="):
                osrel = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu,
        "cpus": os.cpu_count(),
        "os": osrel,
        "python": platform.python_version(),
    }


def subst(argv, src, stem, d):
    return [a.replace("{src}", src).replace("{stem}", stem).replace("{dir}", d) for a in argv]


def snapshot(d, src):
    out = {}
    for root, _dirs, files in os.walk(d):
        for name in files:
            p = os.path.join(root, name)
            rel = os.path.relpath(p, d)
            if rel == src or os.path.islink(p):
                continue
            with open(p, "rb") as f:
                out[rel] = f.read()
    return out


def build(cfg, argv, program, d, work, epoch=True, extra_env=None):
    """Build program in directory d with argv; return (outputs, error)."""
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    src = os.path.basename(program)
    stem = os.path.splitext(src)[0]
    shutil.copyfile(os.path.join(HERE, "programs", program), os.path.join(d, src))
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/"), "LC_ALL": "C", "TZ": "UTC"}
    for k in ("RUSTUP_HOME", "CARGO_HOME"):
        if k in os.environ:
            env[k] = os.environ[k]
    if epoch:
        env["SOURCE_DATE_EPOCH"] = EPOCH
    env.update({k: subst([v], src, stem, d)[0] for k, v in cfg.get("env", {}).items()})
    env.update({k: subst([v], src, stem, d)[0] for k, v in (extra_env or {}).items()})
    fresh = []
    for k in cfg.get("fresh", []):
        f = tempfile.mkdtemp(prefix="fresh-", dir=work)
        env[k] = f
        fresh.append(f)
    try:
        p = subprocess.run(subst(argv, src, stem, d), cwd=d, env=env, capture_output=True, timeout=TIMEOUT)
        if p.returncode != 0:
            msg = (p.stderr or p.stdout).decode("utf-8", "replace").strip().splitlines()
            return None, "exit %d: %s" % (p.returncode, msg[-1] if msg else "")
        return snapshot(d, src), None
    except subprocess.TimeoutExpired:
        return None, "timeout after %d s" % TIMEOUT
    finally:
        for f in fresh:
            shutil.rmtree(f, ignore_errors=True)


def differing_bytes(a, b):
    n = min(len(a), len(b))
    diff = abs(len(a) - len(b))
    if a[:n] == b[:n]:
        return diff
    # compare in blocks; count differing positions exactly
    i = 0
    step = 1 << 16
    while i < n:
        x, y = a[i:i + step], b[i:i + step]
        if x != y:
            diff += sum(1 for p, q in zip(x, y) if p != q)
        i += step
    return diff


def compare(x, y):
    if x[1] or y[1]:
        return {"status": "build failed", "error": x[1] or y[1]}
    a, b = x[0], y[0]
    files = []
    for name in sorted(set(a) | set(b)):
        fa, fb = a.get(name), b.get(name)
        row = {
            "file": name,
            "size_a": None if fa is None else len(fa),
            "size_b": None if fb is None else len(fb),
        }
        if fa is None or fb is None:
            row["differing_bytes"] = len(fa if fa is not None else fb)
            row["missing_in"] = "a" if fa is None else "b"
        else:
            row["differing_bytes"] = differing_bytes(fa, fb)
        files.append(row)
    same = all(r["differing_bytes"] == 0 and "missing_in" not in r for r in files)
    return {"status": "identical" if same else "differs", "files": files}


def hashes(x):
    if x[0] is None:
        return None
    return {k: hashlib.sha256(v).hexdigest() for k, v in sorted(x[0].items())}


def path_copies(outputs, d):
    if outputs is None:
        return None
    needle = d.encode()
    return sum(v.count(needle) for v in outputs.values())


def run_row(cfg, program, work):
    a = os.path.join(work, "a", "build")
    b = os.path.join(work, "another-directory", "build")
    argv = cfg["build"]
    a1 = build(cfg, argv, program, a, work)
    a2 = build(cfg, argv, program, a, work)
    b1 = build(cfg, argv, program, b, work)
    row = {
        "config": cfg["id"],
        "tool": cfg["tool"],
        "language": cfg["language"],
        "program": program,
        "same_dir": compare(a1, a2),
        "cross_dir": compare(a1, b1),
        "path_copies_cross": path_copies(b1[0], b),
        "sha256": hashes(a1),
    }
    if cfg.get("remap"):
        ar = build(cfg, cfg["remap"], program, a, work, extra_env=cfg.get("remap_env"))
        br = build(cfg, cfg["remap"], program, b, work, extra_env=cfg.get("remap_env"))
        row["remapped"] = compare(ar, br)
        row["path_copies_remapped"] = path_copies(br[0], b)
        row["sha256_remapped"] = hashes(ar)
    else:
        row["remapped"] = {"status": "not tested", "error": cfg.get("remap_source", "")}
        row["path_copies_remapped"] = None
        row["sha256_remapped"] = None
    t0 = time.monotonic()
    c1 = build(cfg, argv, program, a, work, epoch=False)
    wait = 1.1 - (time.monotonic() - t0)
    if wait > 0:
        time.sleep(wait)
    c2 = build(cfg, argv, program, a, work, epoch=False)
    row["same_dir_clock"] = compare(c1, c2)
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    return row


def main():
    args = sys.argv[1:]
    if len(args) not in (2, 3) or args[0] != "run":
        sys.exit("usage: python3 e1.py run RESULTS_JSON [WORK_DIR]")
    out = os.path.abspath(args[1])
    matrix = json.load(open(os.path.join(HERE, "matrix.json"), encoding="utf-8"))
    own_work = len(args) == 2
    work = tempfile.mkdtemp(prefix="e1-") if own_work else os.path.abspath(args[2])
    os.makedirs(work, exist_ok=True)
    toolchains, rows, absent = {}, [], []
    try:
        for cfg in matrix["configs"]:
            tool = cfg["tool"]
            if shutil.which(tool) is None:
                absent.append({"config": cfg["id"], "reason": "%s not on PATH" % tool})
                print("absent  %s (%s not on PATH)" % (cfg["id"], tool), flush=True)
                continue
            if "probe" in cfg and subprocess.run(cfg["probe"], capture_output=True).returncode != 0:
                absent.append({"config": cfg["id"], "reason": "probe failed: %s" % " ".join(cfg["probe"])})
                print("absent  %s (probe failed)" % cfg["id"], flush=True)
                continue
            toolchains.setdefault(tool, first_line(cfg["version"]))
            for program in cfg["programs"]:
                row = run_row(cfg, program, work)
                rows.append(row)
                print("%-34s %-22s same=%s cross=%s remap=%s clock=%s" % (
                    cfg["id"], program, row["same_dir"]["status"], row["cross_dir"]["status"],
                    row["remapped"]["status"], row["same_dir_clock"]["status"]), flush=True)
    finally:
        if own_work:
            shutil.rmtree(work, ignore_errors=True)
    result = {
        "schema": SCHEMA,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": machine(),
        "environment": {"LC_ALL": "C", "TZ": "UTC", "SOURCE_DATE_EPOCH": EPOCH},
        "harness": {
            "e1.py": sha256_file(os.path.abspath(__file__)),
            "matrix.json": sha256_file(os.path.join(HERE, "matrix.json")),
        },
        "toolchains": toolchains,
        "absent": absent,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote %s: %d rows, %d configurations absent" % (out, len(rows), len(absent)))


if __name__ == "__main__":
    main()
