#!/usr/bin/env python3
"""Run rce-conformance against one implementation through its adapter.

  harness/run.py --adapter adapters/reference.json [--tests tests] [--filter AREA]
                 [--out reports/results-NAME.json] [--jobs N]

An adapter is any program that speaks rce-adapter/1 (see ADAPTERS.md) on
stdin/stdout. The adapter file names its command line and which ops it
claims; tests that need an op it does not claim are reported as
`unsupported`, never as passes. An adapter may instead list an op under
`not_applicable` with the reason it belongs to another component (an
emitter does not admit modules or run the host); those tests are `n/a`.

Each result records `observed`, the part of the answer that was compared
(check: verdict and diagnostic codes; run: outputs and outcome of the
source-order run; trace: head and lines), so that tools/parity.py can
compare the bytes of every adapter's answer with the reference's. The
results file also records the adapter's `hello` answer, which names the
implementation and, for an emitter, the checker that emitted and the
target's toolchain.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, pathlib, select, shlex, subprocess, sys, time
TIMEOUT = float(os.environ.get("RCE_TEST_TIMEOUT", "120"))
sys.path.insert(0, os.path.dirname(__file__))
import rcetest

# Entries whose content describes the machine (HC-11) and the store; their
# presence is checked by the host tests that name them, not by position.
MACHINE_ENTRIES = {"host.FabricCaps", "host.StorageCaps"}
ROOT = pathlib.Path(__file__).resolve().parent.parent

class Adapter:
    def __init__(self, spec):
        self.spec = spec
        cmd = [os.path.expandvars(os.path.expanduser(x)) for x in spec["cmd"]]
        # RCE_RUNNER prefixes the adapter command (wasmtime, qemu-user)
        cmd = shlex.split(os.environ.get("RCE_RUNNER", "")) + cmd
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  encoding="utf-8", cwd=ROOT, bufsize=1)
        self.n = 0
    def call(self, req, timeout=None):
        self.n += 1
        req = dict(req, id=self.n)
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        if os.name != "nt":
            ready, _, _ = select.select([self.p.stdout], [], [], timeout or TIMEOUT)
            if not ready:
                self.p.kill()
                raise RuntimeError(f"no answer within {timeout or TIMEOUT}s")
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("adapter closed its output")
        return json.loads(line)
    def close(self):
        try: self.p.stdin.close(); self.p.wait(5)
        except Exception: self.p.kill()

def load_tests(root, filt=None):
    out = []
    for p in sorted(pathlib.Path(root).rglob("*.rce")):
        if "_FIXTURE" in p.name:
            continue
        raw = p.read_bytes().decode("utf-8", errors="surrogateescape")
        try:
            src, meta = rcetest.split(raw)
        except ValueError as e:
            out.append({"path": str(p), "error": str(e), "meta": {}}); continue
        if filt and not str(meta.get("area", "")).startswith(filt):
            continue
        fixtures = {}
        for fx in meta.get("fixtures", []) or []:
            fixtures[fx] = (p.parent / fx).read_text(encoding="utf-8")
        out.append({"path": str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p), "src": src, "meta": meta, "fixtures": fixtures})
    return out

GENESIS = "0" * 64

def chain_law(lines, head):
    """6.1 and T8 on a whole trace: every entry line is `d applied|kept HASH
    BODY` with HASH the SHA-256 of BODY, `seq` counts from 0, each
    `prev_hash` is the hash before it (64 zeros first), a kept event leaves
    the state hash as it was, and the head is the last entry's hash. Returns
    the first law broken, or None."""
    prev, prev_state, seq = GENESIS, None, 0
    for l in lines:
        if not l.startswith("d "):
            continue
        parts = l.split(" ", 3)
        if len(parts) != 4 or parts[1] not in ("applied", "kept"):
            return f"entry line {l[:120]!r} is not `d applied|kept HASH BODY`"
        _, outcome, h, body = parts
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != h:
            return f"entry {seq}: {h} is not the SHA-256 of its body"
        try:
            b = json.loads(body)
        except ValueError as e:
            return f"entry {seq}: body is not JSON ({e})"
        if b.get("seq") != seq:
            return f"entry {seq}: seq is {b.get('seq')}"
        if b.get("prev_hash") != prev:
            return f"entry {seq}: prev_hash {b.get('prev_hash')} is not the hash before it, {prev}"
        if outcome == "kept" and prev_state is not None and b.get("state_hash") != prev_state:
            return f"T8: kept entry {seq} changed the state hash"
        prev, prev_state, seq = h, b.get("state_hash"), seq + 1
    if head != prev:
        return f"head {head} is not the hash of the last entry, {prev}"
    return None

def judge(t, a, ops):
    """(status, detail, observed): `observed` is the part of the answer that
    is compared, or None when the adapter gave no answer to compare."""
    m, src = t["meta"], t["src"]
    exp = m.get("expect")
    need = {"accept": "check", "reject": "check", "run": "run", "trace": "trace", "host": "host", "replicate": "replicate", "cli": "cli"}.get(exp)
    if need is None:
        return "error", f"unknown expect `{exp}`", None
    if need not in ops:
        na = (a.spec.get("not_applicable") or {}).get(need)
        if na:   # the op belongs to another component, by the law the adapter cites
            return "n/a", na, None
        return "unsupported", f"adapter does not claim `{need}`", None
    if exp in ("accept", "reject"):
        files = {"main.rce": src, **t.get("fixtures", {})}
        r = a.call({"op": "check", "files": files, "source": src})
        obs = {"verdict": r["verdict"], "codes": [d.get("code") for d in r.get("diagnostics", []) or []]} if "verdict" in r else None
        if r.get("verdict") != exp:
            return "fail", f"expected {exp}, got {r.get('verdict')}: {json.dumps(r.get('diagnostics'))[:300]}", obs
        if exp == "reject" and m.get("diagnostic") not in (None, "any"):
            codes = [d.get("code") for d in r.get("diagnostics", [])]
            if m["diagnostic"] not in codes:
                return "fail", f"expected diagnostic {m['diagnostic']}, got {codes}", obs
        return "pass", "", obs
    if exp == "run":
        seeds = [None] + list(m.get("schedule_seeds", []) or [])
        first = None
        for s in seeds:
            if s is not None and "schedule_seed" not in a.spec.get("features", []):
                continue
            r = a.call({"op": "run", "source": src, "events": m.get("events"), "fuel": m.get("fuel"), "schedule_seed": s})
            obs = first and {"outputs": first[0], "outcome": first[1]}
            if "unsupported" in r:
                return "unsupported", r["unsupported"], obs
            if "error" in r:
                return "fail", f"run error: {r['error'][:300]}", obs
            if r.get("verdict") == "reject":
                return "fail", f"refused at entry: {json.dumps(r.get('diagnostics'))[:300]}", obs
            got = (r.get("outputs", []), r.get("outcome"))
            if first is None: first = got
            obs = {"outputs": first[0], "outcome": first[1]}
            if got != first:
                return "fail", f"schedule {s} differs from source order: {got} vs {first}", obs
            if "outputs" in m and r.get("outputs", []) != m["outputs"]:
                return "fail", f"outputs differ (seed {s}): got {r.get('outputs')} expected {m['outputs']}", obs
            if "outcome" in m and r.get("outcome") != m["outcome"]:
                return "fail", f"outcome {r.get('outcome')} != {m['outcome']} (seed {s})", obs
        return "pass", "", first and {"outputs": first[0], "outcome": first[1]}
    if exp == "host":
        r = a.call({"op": "host", "source": src, "steps": m.get("steps", []), "tear": m.get("tear")})
        if "error" in r:
            return "fail", f"host error: {r['error'][:300]}", None
        h = r.get("host") or {}
        # HC-8 and HC-17: replay and reopen re-derive the live state
        if not (h.get("state_hash") == h.get("replay_state_hash") == h.get("reopen_state_hash")):
            return "fail", f"replay/reopen differ from the live run: {h.get('state_hash')} {h.get('replay_state_hash')} {h.get('reopen_state_hash')}", None
        want = m.get("host_expect", {})
        got = dict(h, entries=[e for e in h.get("entries", []) if e not in MACHINE_ENTRIES],
                   entries_all=h.get("entries", []),
                   torn_reopen_state_hash=(h.get("torn") or {}).get("reopen_state_hash"))
        if "fabric_caps_first" in want:   # HC-11: fabric discovery is logged at start
            ok = "host.FabricCaps" in h.get("entries", [])[:3]
            if not ok: return "fail", "host.FabricCaps is not among the first entries of D", None
            want = {k: v for k, v in want.items() if k != "fabric_caps_first"}
        for k, v in want.items():
            if got.get(k) != v:
                return "fail", f"host {k}: got {json.dumps(got.get(k))[:300]} expected {json.dumps(v)[:300]}", None
        return "pass", "", None
    if exp == "replicate":
        r = a.call({"op": "replicate", "nodes": m.get("nodes", 3), "script": m.get("script", [])})
        if "error" in r:
            return "fail", f"replicate error: {r['error'][:300]}", None
        reps = [x for x in (r.get("replicate") or {}).get("replicas", []) if x]
        logs = [x["committed"] for x in reps]
        # HC-18 / T41: committed logs never disagree; every one is a prefix of the longest
        longest = max(logs, key=len) if logs else []
        for l in logs:
            if l != longest[: len(l)]:
                return "fail", f"committed logs disagree: {l} vs {longest}", None
        want = m.get("replicate_expect", {})
        if "committed_all" in want and any(l != want["committed_all"] for l in logs):
            return "fail", f"committed {logs} expected {want['committed_all']} on every replica", None
        evs = (r.get("replicate") or {}).get("events", [])
        if want.get("deposed_frozen"):
            iso = [e for e in evs if "isolated" in e]
            snap = [e for e in evs if "old_commit" in e and "isolated" not in e]
            if not iso or not snap or snap[0]["old_commit"] != iso[0]["old_commit"]:
                return "fail", f"a deposed leader without a quorum committed: {evs}", None
            if not snap[0].get("term") or snap[0]["term"] <= iso[0]["old_term"]:
                return "fail", f"the new term does not exceed the deposed leader's: {evs}", None
        return "pass", "", None
    if exp == "cli":
        st, why = judge_cli(t, a.spec)
        return st, why, None
    if exp == "trace":
        # the whole trace, compared line for line with the reference's
        # (`trace_lines`, written by tools/trace_lines.py from rce-conform)
        r = a.call({"op": "trace", "source": src, "events": m.get("events")})
        if "error" in r:
            return "fail", f"trace error: {r['error'][:300]}", None
        tr = r.get("trace") or {}
        head, lines = tr.get("head"), tr.get("lines")
        if not isinstance(lines, list) or not all(isinstance(l, str) for l in lines):
            return "fail", "the answer has no trace lines", None
        obs = {"head": head, "lines": lines}
        law = chain_law(lines, head)
        if law:
            return "fail", law, obs
        if head != m.get("trace_head"):
            return "fail", f"trace head {head} != {m.get('trace_head')}", obs
        want = m.get("trace_lines")
        if not isinstance(want, list):
            return "fail", "the test records no trace_lines to compare the whole trace with", obs
        if lines != want:
            i = next(i for i in range(max(len(lines), len(want))) if lines[i:i + 1] != want[i:i + 1])
            got_l = lines[i] if i < len(lines) else "(no line)"
            want_l = want[i] if i < len(want) else "(no line)"
            return "fail", f"trace line {i + 1}: got {got_l[:200]!r}, expected {want_l[:200]!r}", obs
        return "pass", "", obs

def judge_cli(t, spec):
    """Tooling and profile tests (chapter 12, appendix D): run one command of
    the implementation's own tools, named by the adapter's `cli` map, in a
    scratch directory holding the test source as main.rce plus its fixtures,
    and compare the exit status and required/forbidden output text."""
    import tempfile
    m = t["meta"]
    tool = m.get("tool", "checker")
    base = (spec.get("cli") or {}).get(tool)
    if not base:
        return "unsupported", f"adapter names no `{tool}` command"
    base = [os.path.expandvars(os.path.expanduser(x)) for x in base]
    if any(x.startswith("$") for x in base):
        return "unsupported", f"`{tool}` command not configured ({' '.join(base)})"
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "main.rce").write_text(t["src"], encoding="utf-8")
        for name, body in (t.get("fixtures") or {}).items():
            pathlib.Path(d, name).write_text(body, encoding="utf-8")
        try:
            p = subprocess.run(base + [str(x) for x in m.get("argv", [])], cwd=d, capture_output=True, text=True,
                               timeout=TIMEOUT, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "fail", "timeout"
    out = (p.stdout + p.stderr).replace(d + os.sep, "").replace(d, ".")
    if "status" in m and p.returncode != m["status"]:
        return "fail", f"exit status {p.returncode} != {m['status']}: {out[-300:]}"
    for x in m.get("output_has", []) or []:
        if x not in out:
            return "fail", f"output lacks {x!r}: {out[-300:]}"
    for x in m.get("output_lacks", []) or []:
        if x in out:
            return "fail", f"output has {x!r}: {out[-300:]}"
    if "lines" in m:
        n = len([l for l in p.stdout.splitlines() if l.strip()])
        if n != m["lines"]:
            return "fail", f"{n} stdout lines != {m['lines']}: {out[-300:]}"
    return "pass", ""

STOP = None  # set when --stop-on-fail: (threading.Event, ids to ignore)

def run_chunk(spec, tests, ops):
    a = Adapter(spec); res = []
    try:
        for t in tests:
            if STOP and STOP[0].is_set():
                break
            if "error" in t:
                res.append({"path": t["path"], "status": "error", "detail": t["error"]}); continue
            t0 = time.time()
            try:
                st, why, obs = judge(t, a, ops)
            except Exception as e:  # adapter crash counts against the implementation
                st, why, obs = "fail", f"adapter: {e}", None
                a.close(); a = Adapter(spec)
            if STOP and st in ("fail", "error") and t["meta"].get("id") not in STOP[1] and t["path"] not in STOP[1]:
                STOP[0].set()
            res.append({"path": t["path"], "id": t["meta"].get("id"), "area": t["meta"].get("area"),
                        "clauses": t["meta"].get("clauses", []), "status": st, "detail": why,
                        "observed": obs, "ms": int((time.time() - t0) * 1000)})
    finally:
        a.close()
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tests", default=str(ROOT / "tests"))
    ap.add_argument("--filter")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--known-failures", help="file of test ids expected to fail (findings); exit 0 iff the failing set equals it")
    ap.add_argument("--stop-on-fail", action="store_true", help="mutation runs: stop at the first failing test")
    a = ap.parse_args()
    if a.stop_on_fail:
        import threading
        global STOP
        known = set()
        if a.known_failures:
            known = {l.split("#")[0].strip() for l in open(a.known_failures)} - {""}
        STOP = (threading.Event(), known)
    spec = json.load(open(a.adapter))
    ops = set(spec.get("ops", ["check", "run", "trace"]))
    h = Adapter(spec)
    try:   # the implementation and, for an emitter, its checker and toolchain
        hello = h.call({"op": "hello"})
        hello.pop("id", None)
    except Exception as e:
        hello = {"error": f"hello: {e}"}
    finally:
        h.close()
    tests = load_tests(a.tests, a.filter)
    chunks = [tests[i::a.jobs] for i in range(a.jobs)]
    results = []
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for r in ex.map(lambda c: run_chunk(spec, c, ops), chunks):
            results.extend(r)
    results.sort(key=lambda r: r["path"])
    tally = {}
    for r in results:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    import tempfile
    tmp = tempfile.gettempdir()
    for r in results:  # keep machine-local paths out of published reports
        r["detail"] = r.get("detail", "").replace(tmp, "$TMP").replace(str(pathlib.Path.home()), "~")
    summary = {"format": "rce-results/2", "adapter": spec["name"], "hello": hello, "tally": tally, "results": results}
    out = a.out or str(ROOT / "reports" / f"results-{spec['name']}.json")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=1)
    if not a.quiet:
        for r in results:
            if r["status"] in ("fail", "error"):
                print(f"{r['status'].upper()} {r['path']}: {r['detail']}")
    print(f"{spec['name']}: {tally}")
    if a.known_failures:
        known = {l.split("#")[0].strip() for l in open(a.known_failures)} - {""}
        bad = {r["id"] for r in results if r["status"] in ("fail", "error")}
        ran = {r["id"] for r in results if r["status"] == "pass"}
        new, fixed = sorted(bad - known), sorted((known & ran) - bad)
        det = {r["id"]: r.get("detail", "") for r in results}
        for t in new: print(f"NEW FAILURE {t}: {det.get(t, '')[:400]}")
        for t in fixed: print(f"NOW PASSES (update the baseline and FINDINGS.md) {t}")
        sys.exit(1 if new or fixed else 0)
    sys.exit(1 if tally.get("fail") or tally.get("error") else 0)

if __name__ == "__main__":
    main()
