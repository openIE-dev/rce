#!/usr/bin/env python3
"""Run rce-conformance against one implementation through its adapter.

  harness/run.py --adapter adapters/reference.json [--tests tests] [--filter AREA]
                 [--out reports/results-NAME.json] [--jobs N]

An adapter is any program that speaks rce-adapter/1 (see ADAPTERS.md) on
stdin/stdout. The adapter file names its command line and which ops it
claims; tests that need an op it does not claim are reported as
`unsupported`, never as passes.
"""
import argparse, concurrent.futures as cf, json, os, pathlib, select, shlex, subprocess, sys, time
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

def judge(t, a, ops):
    m, src = t["meta"], t["src"]
    exp = m.get("expect")
    need = {"accept": "check", "reject": "check", "run": "run", "trace": "trace", "host": "host", "replicate": "replicate"}.get(exp)
    if need is None:
        return "error", f"unknown expect `{exp}`"
    if need not in ops:
        return "unsupported", f"adapter does not claim `{need}`"
    if exp in ("accept", "reject"):
        files = {"main.rce": src, **t.get("fixtures", {})}
        r = a.call({"op": "check", "files": files, "source": src})
        if r.get("verdict") != exp:
            return "fail", f"expected {exp}, got {r.get('verdict')}: {json.dumps(r.get('diagnostics'))[:300]}"
        if exp == "reject" and m.get("diagnostic") not in (None, "any"):
            codes = [d.get("code") for d in r.get("diagnostics", [])]
            if m["diagnostic"] not in codes:
                return "fail", f"expected diagnostic {m['diagnostic']}, got {codes}"
        return "pass", ""
    if exp == "run":
        seeds = [None] + list(m.get("schedule_seeds", []) or [])
        first = None
        for s in seeds:
            if s is not None and "schedule_seed" not in a.spec.get("features", []):
                continue
            r = a.call({"op": "run", "source": src, "events": m.get("events"), "fuel": m.get("fuel"), "schedule_seed": s})
            if "unsupported" in r:
                return "unsupported", r["unsupported"]
            if "error" in r:
                return "fail", f"run error: {r['error'][:300]}"
            if r.get("verdict") == "reject":
                return "fail", f"refused at entry: {json.dumps(r.get('diagnostics'))[:300]}"
            got = (r.get("outputs", []), r.get("outcome"))
            if first is None: first = got
            elif got != first:
                return "fail", f"schedule {s} differs from source order: {got} vs {first}"
            if "outputs" in m and r.get("outputs", []) != m["outputs"]:
                return "fail", f"outputs differ (seed {s}): got {r.get('outputs')} expected {m['outputs']}"
            if "outcome" in m and r.get("outcome") != m["outcome"]:
                return "fail", f"outcome {r.get('outcome')} != {m['outcome']} (seed {s})"
        return "pass", ""
    if exp == "host":
        r = a.call({"op": "host", "source": src, "steps": m.get("steps", []), "tear": m.get("tear")})
        if "error" in r:
            return "fail", f"host error: {r['error'][:300]}"
        h = r.get("host") or {}
        # HC-8 and HC-17: replay and reopen re-derive the live state
        if not (h.get("state_hash") == h.get("replay_state_hash") == h.get("reopen_state_hash")):
            return "fail", f"replay/reopen differ from the live run: {h.get('state_hash')} {h.get('replay_state_hash')} {h.get('reopen_state_hash')}"
        want = m.get("host_expect", {})
        got = dict(h, entries=[e for e in h.get("entries", []) if e not in MACHINE_ENTRIES],
                   entries_all=h.get("entries", []),
                   torn_reopen_state_hash=(h.get("torn") or {}).get("reopen_state_hash"))
        if "fabric_caps_first" in want:   # HC-11: fabric discovery is logged at start
            ok = "host.FabricCaps" in h.get("entries", [])[:3]
            if not ok: return "fail", "host.FabricCaps is not among the first entries of D"
            want = {k: v for k, v in want.items() if k != "fabric_caps_first"}
        for k, v in want.items():
            if got.get(k) != v:
                return "fail", f"host {k}: got {json.dumps(got.get(k))[:300]} expected {json.dumps(v)[:300]}"
        return "pass", ""
    if exp == "replicate":
        r = a.call({"op": "replicate", "nodes": m.get("nodes", 3), "script": m.get("script", [])})
        if "error" in r:
            return "fail", f"replicate error: {r['error'][:300]}"
        reps = [x for x in (r.get("replicate") or {}).get("replicas", []) if x]
        logs = [x["committed"] for x in reps]
        # HC-18 / T41: committed logs never disagree; every one is a prefix of the longest
        longest = max(logs, key=len) if logs else []
        for l in logs:
            if l != longest[: len(l)]:
                return "fail", f"committed logs disagree: {l} vs {longest}"
        want = m.get("replicate_expect", {})
        if "committed_all" in want and any(l != want["committed_all"] for l in logs):
            return "fail", f"committed {logs} expected {want['committed_all']} on every replica"
        evs = (r.get("replicate") or {}).get("events", [])
        if want.get("deposed_frozen"):
            iso = [e for e in evs if "isolated" in e]
            snap = [e for e in evs if "old_commit" in e and "isolated" not in e]
            if not iso or not snap or snap[0]["old_commit"] != iso[0]["old_commit"]:
                return "fail", f"a deposed leader without a quorum committed: {evs}"
            if not snap[0].get("term") or snap[0]["term"] <= iso[0]["old_term"]:
                return "fail", f"the new term does not exceed the deposed leader's: {evs}"
        return "pass", ""
    if exp == "trace":
        r = a.call({"op": "trace", "source": src, "events": m.get("events")})
        if "error" in r:
            return "fail", f"trace error: {r['error'][:300]}"
        steps = (r.get("trace") or {}).get("steps") or []
        # T8 on every trace: a refused event leaves the state exactly as it was
        prev = None
        for st in steps:
            if st.get("outcome") not in (None, "applied") and prev is not None and st.get("state_hash") != prev:
                return "fail", f"T8: refused step {st.get('seq')} changed the state hash"
            prev = st.get("state_hash")
        head = (r.get("trace") or {}).get("head")
        if head != m.get("trace_head"):
            return "fail", f"trace head {head} != {m.get('trace_head')}"
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
                st, why = judge(t, a, ops)
            except Exception as e:  # adapter crash counts against the implementation
                st, why = "fail", f"adapter: {e}"
                a.close(); a = Adapter(spec)
            if STOP and st in ("fail", "error") and t["meta"].get("id") not in STOP[1] and t["path"] not in STOP[1]:
                STOP[0].set()
            res.append({"path": t["path"], "id": t["meta"].get("id"), "area": t["meta"].get("area"),
                        "clauses": t["meta"].get("clauses", []), "status": st, "detail": why,
                        "ms": int((time.time() - t0) * 1000)})
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
    summary = {"format": "rce-results/1", "adapter": spec["name"], "tally": tally, "results": results}
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
