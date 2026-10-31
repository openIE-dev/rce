#!/usr/bin/env python3
"""The cost of determinism (critique 7): wall and CPU time per step of the
same programs under the hash-chained log D, without it, as emitted Rust,
and as hand-written Rust with native floats and fused multiply-add (FMA).

    bench-determinism.py BENCH_BIN OUT_DIR [PROGRAM_ID ...]

BENCH_BIN is `rce-bench-det` (spec/checker/src/bin/bench_det.rs), built
from the same commit. For every program of PROGRAMS the script

1. draws CANDIDATES seeded candidate events (SplitMix64 as
   scripts/xarch-streams.py, seed SEED) from the program's templates and
   keeps the first STEPS whose step applies (`rce-bench-det filter`), so the
   stream has no kept step; the stream is a pure function of the seed;
2. times the checker modes of `rce-bench-det run` (delta, encode, chain,
   ckpt64, host, host1; see bench_det.rs);
3. emits the program with the stream as one example (`rce-bench-det emit`,
   the emitter of `rce emit rust`), compiles it with `rustc -O`, and times
   it four ways: the word `trace` (it computes and prints every D entry), plain
   (it prints the state after every step, its observations), quiet (a
   benchmark-only copy with the per-step state prints removed) and quiet
   with `trace` (the D entries computed and printed, no state prints).
   The emitted `main` is replaced by a benchmark-only one that reads the
   stream at start-up (see GLUE), since an emitted example holds one
   closure per event and a 1,000-event example does not compile in
   reasonable time and memory;
4. for the programs with a baseline in scripts/bench-determinism/baselines,
   compiles the hand-written idiomatic Rust with `-C opt-level=3 -C
   target-cpu=native`, twice: `fma` (f64::mul_add where the RCE program has
   a multiply then an add) and `nofma` (the same with two roundings), and
   times both, and a third build `nofma_log` that also keeps the log D
   natively (positional JSON and two SHA-256 per step, portable SHA-256 as
   in the emitted runtime; bench-determinism/baselines/common.rs).

Timing: each configuration runs REPEATS times with the stream applied R
times in the process, R the smallest power of 4 whose run takes MIN_RUN
seconds more than start-up, and REPEATS times with R = 0 (start-up:
reading and checking the program, parsing the stream). Per step = (run -
median start-up) / (R * STEPS), CPU (the primary measure) from the child's
rusage (user + system, wait4) and wall from time.perf_counter around the
process; the median over the REPEATS is reported.
Output: OUT_DIR/results.json, OUT_DIR/streams/*.events and a table on
stdout. est_j is CPU seconds times 10 W (the paper's estimate, section
6.3), never a measurement.
"""
import hashlib, json, os, platform, re, shutil, statistics, subprocess, sys, time
from pathlib import Path

SEED = 0x5EED_D3AD_0007
STEPS = int(os.environ.get("BENCH_STEPS", "1000"))
CANDIDATES = 4 * STEPS
REPEATS = int(os.environ.get("BENCH_REPEATS", "11"))
MIN_RUN = float(os.environ.get("BENCH_MIN_RUN", "0.5"))
EST_W = 10.0
ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "scripts" / "bench-determinism"


class Rng:
    def __init__(self, s):
        self.s = s & ((1 << 64) - 1)

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)

    def int(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)

    def real(self, lo, hi):
        # a float with 6 decimals, written so the literal is exact text
        k = self.int(0, 1_000_000)
        return f"{lo + (hi - lo) * k / 1_000_000:.6f}"

    def pick(self, xs):
        return xs[self.next() % len(xs)]


def w(*pairs):
    """Weighted choice table: [(weight, fn(rng) -> event text)]."""
    return list(pairs)


# id, path, class, candidate templates (weight, maker)
PROGRAMS = [
    ("bank", "spec/examples/04_bank_account.rce", "arithmetic", w(
        (6, lambda r: f"Deposit(amount: {r.int(1, 100)})"),
        (4, lambda r: f"Withdraw(amount: {r.int(1, 60)})"))),
    ("ring", "programs/systems/ring_buffer.rce", "arithmetic", w(
        (1, lambda r: f"Push(v: {r.int(-1000000, 1000000)})"),
        (1, lambda r: "Pop"))),
    ("automaton", "programs/world/cellular_automaton.rce", "arithmetic", w(
        (1, lambda r: "Step"),)),
    ("inventory", "spec/examples/31_inventory.rce", "collection", w(
        (6, lambda r: f"Receive(sku: \"s{r.int(0, 63)}\", qty: {r.int(1, 20)})"),
        (4, lambda r: f"Ship(sku: \"s{r.int(0, 63)}\", qty: {r.int(1, 10)})"))),
    ("rcedb", "spec/examples/55_rcedb.rce", "collection", w(
        (12, lambda r: f"Put(key: \"k{r.int(0, 63)}\", val: \"v{r.int(0, 9999)}\")"),
        (4, lambda r: f"Delete(key: \"k{r.int(0, 63)}\")"),
        (2, lambda r: f"Synced(upto: {r.int(0, 100000)})"),
        (1, lambda r: f"Scan(start: \"k{r.int(0, 63)}\")"))),
    ("cart", "spec/examples/16_multi_region.rce", "collection", w(
        (10, lambda r: f"AddItem(name: \"i{r.int(0, 31)}\")"),
        (6, lambda r: f"RemoveItem(name: \"i{r.int(0, 31)}\")"),
        (1, lambda r: "Reset"))),
    ("arm", "programs/body/arm.rce", "float", w(
        (1, lambda r: f"Sensor(reading: {r.real(-3.0, 3.0)})"),)),
    ("floatvec", "spec/examples/57_float_vectors.rce", "float", w(
        (2, lambda r: f"MulSub(a: {r.real(-100, 100)}, b: {r.real(-100, 100)}, c: {r.real(-100, 100)})"),
        (2, lambda r: f"Add(a: {r.real(-1000, 1000)}, b: {r.real(-1000, 1000)})"),
        (2, lambda r: f"Mul(a: {r.real(-1000, 1000)}, b: {r.real(-1000, 1000)})"),
        (1, lambda r: f"Div(a: {r.real(-1000, 1000)}, b: {r.real(1, 1000)})"),
        (1, lambda r: f"Sqrt(a: {r.real(0, 1000)})"),
        (1, lambda r: f"Round(a: {r.real(-1000, 1000)})"),
        (1, lambda r: f"Conv(n: {r.int(-(1 << 60), 1 << 60)})"),
        (1, lambda r: f"F32Mul(a: {r.real(-100, 100)}, b: {r.real(-100, 100)})"))),
    ("fir", "scripts/bench-determinism/programs/fir_filter.rce", "float", w(
        (1, lambda r: f"Sample(x: {r.real(-1.0, 1.0)})"),)),
    ("elevator", "programs/multi/elevator.rce", "state", w(
        (3, lambda r: "Up"), (3, lambda r: "Down"), (2, lambda r: "OpenDoor"),
        (2, lambda r: "CloseDoor"), (3, lambda r: "Arrive"))),
    ("lexer", "programs/text/word_count.rce", "state", w(
        (5, lambda r: f"Char(c: {r.int(97, 122)})"), (1, lambda r: "Char(c: 32)"))),
]

# programs with a hand-written baseline in HERE/baselines/<id>.rs
BASELINES = ["bank", "automaton", "inventory", "arm", "fir", "elevator"]

CHECKER_MODES = ["delta", "encode", "chain", "chainfast", "ckpt64", "host", "host1"]


def candidates(rng, table):
    total = sum(k for k, _ in table)
    out = []
    for _ in range(CANDIDATES):
        x = rng.next() % total
        for k, f in table:
            if x < k:
                out.append(f(rng))
                break
            x -= k
    return out


def timed(cmd, env, scratch):
    """(wall s, CPU s, last stdout line, stderr, status) of one run of cmd;
    CPU is user + system time of the child from wait4's rusage."""
    with open(scratch / "out.txt", "wb") as fo, open(scratch / "err.txt", "wb") as fe:
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=env)
        _, status, ru = os.wait4(p.pid, 0)
        t1 = time.perf_counter()
        p.returncode = os.waitstatus_to_exitcode(status)
    with open(scratch / "out.txt", "rb") as fo:
        fo.seek(0, 2)
        fo.seek(max(0, fo.tell() - 4096))
        tail = fo.read().decode(errors="replace").strip().splitlines()
    err = (scratch / "err.txt").read_text(errors="replace")
    return t1 - t0, ru.ru_utime + ru.ru_stime, tail[-1] if tail else "", err, p.returncode


def measure(make_cmd, steps, scratch):
    """make_cmd(reps) -> (argv, env). R is the smallest power of 4 whose run
    exceeds the start-up by MIN_RUN seconds of wall time; then REPEATS pairs
    (start-up run, R run) are timed. Returns per-step medians."""

    def one(reps):
        argv, env = make_cmd(reps)
        wall, cpu, out, err, rc = timed(argv, env, scratch)
        if rc != 0:
            raise SystemExit(f"failed ({rc}): {' '.join(map(str, argv))}\n{err[-2000:]}")
        return wall, cpu, out

    w0 = min(one(0)[0] for _ in range(3))
    reps = 1
    while one(reps)[0] - w0 < MIN_RUN and reps < 1 << 24:
        reps *= 4
    base_w, base_c, runs_w, runs_c = [], [], [], []
    out_last = ""
    for _ in range(REPEATS):
        a, b, _ = one(0)
        base_w.append(a)
        base_c.append(b)
        a, b, out_last = one(reps)
        runs_w.append(a)
        runs_c.append(b)
    bw, bc = statistics.median(base_w), statistics.median(base_c)
    per_w = [(x - bw) / (reps * steps) for x in runs_w]
    per_c = [(x - bc) / (reps * steps) for x in runs_c]
    return {
        "reps": reps,
        "ns_per_step_cpu": statistics.median(per_c) * 1e9,
        "ns_per_step_wall": statistics.median(per_w) * 1e9,
        "ns_per_step_cpu_min": min(per_c) * 1e9,
        "ns_per_step_cpu_max": max(per_c) * 1e9,
        "run_cpu_s_median": statistics.median(runs_c),
        "startup_wall_s": bw,
        "startup_cpu_s": bc,
        "est_j_per_step_estimate": statistics.median(per_c) * EST_W,
        "out": out_last,
    }


def rustc(src, exe, flags):
    r = subprocess.run(["rustc", "--edition", "2024", *flags, "-o", str(exe), str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"rustc {src}: {r.stderr[-3000:]}")


GLUE = r"""
// ---- benchmark-only main (scripts/bench-determinism.py): the stream is read
// from the file BENCH_STREAM at start-up and each event becomes the same
// thunk the emitter writes for an example event (`construct` with the
// event's declared fields, one `fuel()` per argument); the example runs
// BENCH_REPS times through run_example, as the emitted main runs examples.
fn bench_val(s: &str) -> V {
    if let Some(t) = s.strip_prefix('"') { return V::Text(t.strip_suffix('"').unwrap().to_string()); }
    if s.contains('.') || s.contains('e') { return V::F(s.parse::<f64>().unwrap()); }
    lit(s.parse::<i128>().unwrap())
}
fn bench_fields(n: &str) -> Vec<(&'static str, T, Init)> {
    match n {
FIELDS        _ => panic!("event {n} not in the template"),
    }
}
fn bench_events() -> Vec<Ev> {
    let text = std::fs::read_to_string(std::env::var("BENCH_STREAM").unwrap()).unwrap();
    let mut out: Vec<Ev> = vec![];
    for l in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let (name, rest) = l.split_once('(').unwrap_or((l, ")"));
        let name: &'static str = Box::leak(name.to_string().into_boxed_str());
        let args: Vec<(&'static str, V)> = rest.trim_end_matches(')').split(", ").filter(|kv| !kv.is_empty()).map(|kv| {
            let (k, v) = kv.split_once(": ").unwrap();
            (&*Box::leak(k.to_string().into_boxed_str()), bench_val(v))
        }).collect();
        out.push((name, Rc::new(move || -> R<Option<V>> {
            let mut a = vec![];
            for (k, v) in &args { fuel()?; a.push((Some(*k), v.clone())); }
            Ok(Some(construct(name, bench_fields(name), a)?))
        }) as Rc<dyn Fn() -> R<Option<V>> + Send + Sync>));
    }
    out
}
fn main() {
    let n: usize = std::env::var("BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let evs = bench_events();
    for _ in 0..n { run_example(0, evs.clone()); }
}
"""


def bench_main(src_text, quiet):
    """Benchmark-only copy of an emitted program (emitted with a template
    stream holding each event name once): its `main` is replaced by GLUE,
    which reads the stream at start-up and runs it BENCH_REPS times;
    `quiet` also drops the per-step prints of run_example1."""
    k = src_text.index("\nfn main() {")
    t, tail = src_text[:k], src_text[k:]
    fields = ""
    for name, sch in re.findall(r'construct\("(\w+)", (vec!\[[^\]]*\]), vec!\[', tail):
        fields += f'        "{name}" => {sch},\n'
    if not fields:
        raise SystemExit("emitted program: no event thunk in main")
    if quiet:
        for old in ('                println!("{k} {} {}", show(&ev), show_state(&s));\n',
                    '                for o in outs { println!("  emit {}", show(&o)); }\n'):
            if t.count(old) != 1:
                raise SystemExit(f"emitted program changed shape: {old.strip()}")
            t = t.replace(old, "                std::hint::black_box((&s, &outs));\n" if "show_state" in old else "")
    return t + GLUE.replace("FIELDS", fields)


def machine():
    cpu = ""
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                cpu = l.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip() != ""
    rv = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    return {
        "cpu": cpu, "logical_cpus": os.cpu_count(), "os": platform.platform(),
        "loadavg_at_start": open("/proc/loadavg").read().split()[:3] if os.path.exists("/proc/loadavg") else None,
        "rce_commit": commit, "rce_tree_dirty": dirty, "rustc": rv,
        "python": platform.python_version(), "steps": STEPS, "repeats": REPEATS,
        "min_run_s": MIN_RUN, "seed": hex(SEED),
    }


def main():
    bench = str(Path(sys.argv[1]).resolve())
    out = Path(sys.argv[2]).resolve()
    only = set(sys.argv[3:])
    (out / "streams").mkdir(parents=True, exist_ok=True)
    build = out / "build"
    build.mkdir(exist_ok=True)
    res = {"machine": machine(), "programs": {}}
    rng = Rng(SEED)
    for pid, path, cls, table in PROGRAMS:
        cands = candidates(rng, table)  # drawn for every program, so a subset run sees the same streams
        if only and pid not in only:
            continue
        f = ROOT / path
        cf = build / f"{pid}.candidates"
        cf.write_text("\n".join(cands) + "\n")
        r = subprocess.run([bench, "filter", str(f), str(cf), str(STEPS)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{pid}: filter: {r.stderr}")
        stream = out / "streams" / f"{pid}.events"
        stream.write_text(r.stdout)
        row = {"path": path, "class": cls, "steps": STEPS, "filter": r.stderr.strip(),
               "stream_sha256": hashlib.sha256(r.stdout.encode()).hexdigest(), "configs": {}}
        print(f"== {pid} ({cls}) {path}: {r.stderr.strip()}", flush=True)
        for mode in CHECKER_MODES:
            m = measure(lambda reps, mode=mode: ([bench, "run", mode, str(f), str(stream), str(reps)], None), STEPS, build)
            row["configs"][f"checker_{mode}"] = m
            print(f"  checker {mode:7} {m['ns_per_step_wall']:12.0f} ns/step wall {m['ns_per_step_cpu']:12.0f} cpu  (R={m['reps']}) {m['out'][:60]}", flush=True)
        # the template: the first event of each name in the stream
        seen, tmpl = set(), []
        for l in r.stdout.splitlines():
            n = l.split("(")[0]
            if n not in seen:
                seen.add(n)
                tmpl.append(l)
        tf = build / f"{pid}.template"
        tf.write_text("\n".join(tmpl) + "\n")
        em = subprocess.run([bench, "emit", str(f), str(tf)], capture_output=True, text=True)
        if em.returncode != 0:
            raise SystemExit(f"{pid}: emit: {em.stderr}")
        for quiet in (False, True):
            src = build / f"{pid}_emit{'_quiet' if quiet else ''}.rs"
            src.write_text(bench_main(em.stdout, quiet))
            exe = build / src.stem
            t = time.perf_counter()
            rustc(src, exe, ["-O"])
            row[f"compile_s{'_quiet' if quiet else ''}"] = round(time.perf_counter() - t, 1)
            variants = [("emit_quiet", []), ("emit_trace_quiet", ["trace"])] if quiet else [("emit_trace", ["trace"]), ("emit_plain", [])]
            for name, args in variants:
                def mk(reps, exe=exe, args=args):
                    env = dict(os.environ, BENCH_REPS=str(reps), BENCH_STREAM=str(stream))
                    return [str(exe), *args], env
                m = measure(mk, STEPS, build)
                row["configs"][name] = m
                print(f"  {name:15} {m['ns_per_step_wall']:12.0f} ns/step wall {m['ns_per_step_cpu']:12.0f} cpu  (R={m['reps']}) {m['out'][:60]}", flush=True)
        if pid in BASELINES:
            for fl, cfgs in (("fma", ["bench_fma"]), ("nofma", ["bench_nofma"]), ("nofma_log", ["bench_nofma", "bench_log"])):
                src = HERE / "baselines" / f"{pid}.rs"
                exe = build / f"{pid}_hand_{fl}"
                rustc(src, exe, ["-C", "opt-level=3", "-C", "target-cpu=native", *[x for c in cfgs for x in ("--cfg", c)]])
                m = measure(lambda reps, exe=exe: ([str(exe), str(stream), str(reps)], None), STEPS, build)
                row["configs"][f"hand_{fl}"] = m
                print(f"  hand {fl:10} {m['ns_per_step_wall']:12.1f} ns/step wall {m['ns_per_step_cpu']:12.1f} cpu  (R={m['reps']}) {m['out'][:60]}", flush=True)
        c = row["configs"]
        # the configurations reach the same D and the same last state
        dcheck = c["checker_delta"]["out"].split()[-1]
        chain = c["checker_chain"]["out"].split()[-1]
        row["parity"] = {
            "delta_state_hash": dcheck,
            "chain_head": chain,
            "emit_trace_head_equal": c["emit_trace"]["out"] == f"trace {chain}",
            "emit_trace_quiet_head_equal": c["emit_trace_quiet"]["out"] == f"trace {chain}",
            "ckpt64_differs_from_chain": c["checker_ckpt64"]["out"].split()[-1] != chain,
        }
        if "hand_nofma_log" in c:
            row["parity"]["hand_nofma_log_head_equal"] = c["hand_nofma_log"]["out"] == f"head {chain}"
        row["parity"]["chainfast_head_equal"] = c["checker_chainfast"]["out"].split()[-1] == chain
        for fl in ("fma", "nofma"):
            if f"hand_{fl}" in c:
                st = c[f"hand_{fl}"]["out"].split(" ", 1)[1]
                row["parity"][f"hand_{fl}_state_hash_equal"] = hashlib.sha256(st.encode()).hexdigest() == dcheck
        print(f"  parity {row['parity']}", flush=True)
        ratio = lambda a, b: c[a]["ns_per_step_cpu"] / c[b]["ns_per_step_cpu"] if a in c and b in c and c[b]["ns_per_step_cpu"] > 0 else None
        row["ratios_cpu"] = {
            "log_cost_chain_over_delta": ratio("checker_chain", "checker_delta"),
            "log_cost_host_over_delta": ratio("checker_host", "checker_delta"),
            "encode_over_delta": ratio("checker_encode", "checker_delta"),
            "ckpt64_over_delta": ratio("checker_ckpt64", "checker_delta"),
            "emit_trace_over_quiet": ratio("emit_trace", "emit_quiet"),
            "emit_log_cost_trace_quiet_over_quiet": ratio("emit_trace_quiet", "emit_quiet"),
            "emit_trace_quiet_over_hand_fma": ratio("emit_trace_quiet", "hand_fma"),
            "emit_plain_over_quiet": ratio("emit_plain", "emit_quiet"),
            "emit_quiet_over_hand_fma": ratio("emit_quiet", "hand_fma"),
            "emit_trace_over_hand_fma": ratio("emit_trace", "hand_fma"),
            "hand_nofma_over_hand_fma": ratio("hand_nofma", "hand_fma"),
            "hand_log_cost_nofma_log_over_nofma": ratio("hand_nofma_log", "hand_nofma"),
            "hand_nofma_log_over_hand_fma": ratio("hand_nofma_log", "hand_fma"),
            "chainfast_over_delta": ratio("checker_chainfast", "checker_delta"),
            "checker_delta_over_emit_quiet": ratio("checker_delta", "emit_quiet"),
        }
        res["programs"][pid] = row
        (out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    shutil.rmtree(build)
    print(json.dumps({k: v["ratios_cpu"] for k, v in res["programs"].items()}, indent=1))


if __name__ == "__main__":
    main()
