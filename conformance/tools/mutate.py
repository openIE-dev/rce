#!/usr/bin/env python3
"""Mutation testing of the suite against the reference checker and
interpreter, by mutant schemata: every mutation site is compiled into one
binary behind a guard `crate::mutant::on(K)`, so one build serves every
mutant and the suite runs once per mutant with RCE_MUTANT=K.

Operators
  NEG   negate the condition of an `if` (interp, conc, parse, check)
  DEL   drop a diagnostic the checker would push (check, check06b)
A mutant is killed when at least one test fails (or the implementation
crashes or hangs). Survivors are listed for review: each is a missing test
or an equivalent mutant.

  mutate.py prepare SPEC_DIR WORK_DIR      copy the crate, insert schemata
  mutate.py run WORK_DIR SAMPLE SEED        build, run the suite per mutant
"""
import json, os, pathlib, random, re, shutil, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = {"src/interp.rs": "NEG", "src/conc.rs": "NEG", "src/parse.rs": "NEG", "src/check.rs": "NEG+DEL", "src/check06b.rs": "NEG+DEL"}
IF = re.compile(r"^(\s*(?:\} else )?if )(.+)( \{)$")
DEL = re.compile(r"^(\s*)((?:self\.e\(|self\.errs\.push\(|errs\.push\().*\);)$")

def prepare(spec, work):
    work = pathlib.Path(work); spec = pathlib.Path(spec)
    if work.exists(): shutil.rmtree(work)
    shutil.copytree(spec, work / "spec", ignore=shutil.ignore_patterns("target", "lean", "figures", "*.wasm"))
    if (spec.parent / "rcedb").exists():   # include_str!("../../../rcedb/rcedb.rce")
        shutil.copytree(spec.parent / "rcedb", work / "rcedb")
    ck = work / "spec" / "checker"
    (ck / "src" / "mutant.rs").write_text(
        "//! Mutant schemata guard (rce-conformance tools/mutate.py).\n"
        "static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new();\n"
        "#[inline]\npub fn on(k: u32) -> bool {\n"
        "    *K.get_or_init(|| std::env::var(\"RCE_MUTANT\").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) == k\n}\n")
    lib = (ck / "src" / "lib.rs").read_text()
    (ck / "src" / "lib.rs").write_text(lib.replace("pub mod adapters;", "pub mod adapters;\npub mod mutant;", 1))
    sites, k = [], 0
    for f, ops in FILES.items():
        p = ck / f; lines = p.read_text().split("\n"); depth_raw = 0
        for i, l in enumerate(lines):
            if l.lstrip().startswith("//") or "const " in l or "matches!(" in l and " let " in l: continue
            m = IF.match(l)
            bal = lambda c: c.count("(") == c.count(")") and c.count("[") == c.count("]") and "match " not in c and "|" not in c.replace("||", "")
            if "NEG" in ops and m and bal(m.group(2)) and " let " not in m.group(2) and not m.group(2).startswith("let ") and "{" not in m.group(2) and "=>" not in l:
                k += 1
                lines[i] = f"{m.group(1)}({m.group(2)}) != crate::mutant::on({k}){m.group(3)}"
                sites.append({"k": k, "op": "NEG", "file": f, "line": i + 1, "orig": l.strip()[:160]})
                continue
            d = DEL.match(l)
            if "DEL" in ops and d:
                k += 1
                lines[i] = f"{d.group(1)}if !crate::mutant::on({k}) {{ {d.group(2)} }}"
                sites.append({"k": k, "op": "DEL", "file": f, "line": i + 1, "orig": l.strip()[:160]})
        p.write_text("\n".join(lines))
    json.dump(sites, open(work / "sites.json", "w"), indent=1)
    print(f"{len(sites)} sites")

def run(work, sample, seed):
    work = pathlib.Path(work); ck = work / "spec" / "checker"
    env = dict(os.environ, CARGO_TARGET_DIR=str(work / "target"))
    b = subprocess.run(["cargo", "build", "--release", "-q", "--bin", "rce-conform"], cwd=ck, env=env, capture_output=True, text=True)
    if b.returncode != 0:
        print(b.stderr[-4000:]); sys.exit(1)
    exe = str(work / "target" / "release" / "rce-conform")
    sites = json.load(open(work / "sites.json"))
    rnd = random.Random(seed)
    pick = sorted(rnd.sample(sites, min(sample, len(sites))), key=lambda s: s["k"])
    spec = {"name": "mutant", "cmd": [exe], "ops": ["check", "run", "trace", "host", "replicate"], "features": ["fuel", "schedule_seed", "trace"]}
    sp = work / "adapter.json"; json.dump(spec, open(sp, "w"))
    # mutant 0 is the unmutated build: its failures are the baseline, not kills
    basef = work / "baseline.txt"
    def suite(k):
        e = dict(os.environ, RCE_MUTANT=str(k), RCE_TEST_TIMEOUT="20")
        out = work / f"res-{k}.json"
        # past the baseline, stop at the first failure the unmutated build does not have
        stop = ["stop-on-fail", "known-failures", str(basef)] if k else []
        # words, then ADAPTER TESTS OUT (positional-cli.md PC-1)
        subprocess.run([sys.executable, str(ROOT / "harness" / "run.py"), "quiet"] + stop + [str(sp), str(ROOT / "tests"), str(out)],
                       env=e, capture_output=True, text=True, cwd=ROOT)
        r = json.load(open(out)); out.unlink()
        return {x["path"] for x in r["results"] if x["status"] in ("fail", "error")}
    base = suite(0); results = []
    basef.write_text("".join(p + "\n" for p in sorted(base)))
    # resume: reuse verdicts already logged for this seed's sample
    done = {}
    for l in open("/tmp/rce-conformance-progress.log"):
        m = re.search(r"mutant (\d+) \S+ \S+ killed=(True|False)", l)
        if m: done[int(m.group(1))] = m.group(2) == "True"
    t0 = time.time()
    for s in pick:
        if s["k"] in done:
            results.append(dict(s, killed=done[s["k"]], by=[], n_failing=None)); continue
        failed = suite(s["k"]) - base
        s = dict(s, killed=bool(failed), by=sorted(failed)[:3], n_failing=len(failed)); results.append(s)
        with open("/tmp/rce-conformance-progress.log", "a") as lg:
            lg.write(f"{time.ctime()} mutant {s['k']} {s['op']} {s['file']}:{s['line']} killed={s['killed']}\n")
    killed = sum(1 for r in results if r["killed"])
    by_op = {}
    for r in results:
        o = by_op.setdefault(f"{r['op']} {r['file']}", [0, 0]); o[0] += r["killed"]; o[1] += 1
    rep = {"format": "rce-mutation/1", "sites": len(sites), "sampled": len(results), "seed": seed, "killed": killed,
           "kill_rate": round(100.0 * killed / max(1, len(results)), 1), "baseline_failures": sorted(base),
           "by_operator_file": {k: {"killed": v[0], "total": v[1]} for k, v in by_op.items()},
           "survivors": [r for r in results if not r["killed"]], "seconds": int(time.time() - t0)}
    json.dump(rep, open(ROOT / "reports" / "mutation.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("sites", "sampled", "killed", "kill_rate", "seconds")}))

if __name__ == "__main__":
    {"prepare": lambda: prepare(sys.argv[2], sys.argv[3]), "run": lambda: run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))}[sys.argv[1]]()
