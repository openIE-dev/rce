#!/usr/bin/env python3
"""rce-adapter/1 for the program emitters of spec 10.5 (rust, python, js, scala).

An emitter is a back end: it translates an admitted module. So this adapter
answers `run` only. Entry (parse, check, admission) is the reference
front end's job and is certified through the reference adapter.

  run, no events   emit TARGET, run the program with `--run [--fuel=N]`
                   (fn main on the emitted tick runtime)
  run, events      add the line `example EVENTS -> true` (or `-> refused: E`
                   when the last event is refused) so the module has an
                   observation for exactly these events, emit, run, and read
                   that example's block. The example's expected value is used
                   only to admit the module; the lines compared are printed
                   by the emitted program.
Env: RCE (the rce binary), RCE_RUSTC, RCE_PYTHON, RCE_NODE, RCE_SCALA_CLI.
"""
import hashlib, json, os, re, subprocess, sys, tempfile

LANG = sys.argv[1]
RCE = os.environ.get("RCE", "rce")
TMP = os.path.join(tempfile.gettempdir(), f"rce-conformance-{LANG}")
os.makedirs(TMP, exist_ok=True)
EXT = {"rust": "rs", "python": "py", "js": "js", "scala": "scala"}[LANG]

def sh(cmd, inp=None, timeout=300):
    p = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr

def emit(src):
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    f = os.path.join(TMP, f"m{h}.rce"); open(f, "w").write(src)
    rc, out, err = sh([RCE, "emit", LANG, f])
    if rc != 0:
        return None, err.strip() or out.strip()
    prog = os.path.join(TMP, f"m{h}.{EXT}"); open(prog, "w").write(out)
    if LANG == "rust":
        exe = os.path.join(TMP, f"m{h}")
        if not os.path.exists(exe):
            rc, o, e = sh([os.environ.get("RCE_RUSTC", "rustc"), "-O", "--edition", "2024", "-o", exe, prog], timeout=600)
            if rc != 0:
                return None, "rustc: " + e[-800:]
        return [exe], None
    if LANG == "python":
        return [os.environ.get("RCE_PYTHON", "python3"), prog], None
    if LANG == "js":
        return [os.environ.get("RCE_NODE", "node"), prog], None
    return [os.environ.get("RCE_SCALA_CLI", "scala-cli"), "run", "--jvm", "21", "--server=false", "-q", prog, "--"], None

def outcome_of(r):
    """One spelling for an outcome (see ADAPTERS.md): trap:KIND or refused:E."""
    if r.startswith("trap: "): return "trap:" + r[6:]
    if r.startswith("trap(") and r.endswith(")"): return "trap:" + r[5:-1]
    return "refused:" + r

def run_main(src, fuel):
    cmd, err = emit(src)
    if cmd is None:
        return {"error": f"emit: {err}"}
    args = ["--run"] + ([f"--fuel={fuel}"] if fuel else [])
    rc, out, e = sh(cmd + args)
    lines = [l for l in out.splitlines() if l and not l.startswith("actorlog ")]
    outcome = "done"
    if lines and lines[-1].startswith("outcome "):
        o = lines.pop()[8:]
        if o == "Done": outcome = "done"
        elif o == "Deadlock": outcome = "deadlock"
        elif o.startswith("Refused: "):
            outcome = outcome_of(o[9:])
    elif rc != 0:
        return {"error": f"exit {rc}: {e[-500:]}"}
    return {"verdict": "accept", "outputs": lines, "outcome": outcome}

def run_events(src, events):
    base = src if src.endswith("\n") else src + "\n"
    n = base.count("\n") + 1
    cand = [base + f"example {events} -> true\n"]
    cmd, err = emit(cand[0])
    if cmd is None:
        m = re.search(r"refused[:(]\s*([A-Za-z_][\w]*)", err or "")
        trap = re.search(r"refused: trap\(([^)]*)\)", err or "")
        if m and not trap:
            cmd, err2 = emit(base + f"example {events} -> refused: {m.group(1)}\n")
            if cmd is None:
                return {"error": f"emit: {err2}"}
        elif trap:
            # An `example` can name only a declared error after `refused:`, so an
            # event list whose last event traps has no observation in an
            # emitted program (finding EXAMPLE-TRAP). Say so rather than guess.
            return {"unsupported": "the last event traps; no `example` form observes a trap"}
        else:
            return {"error": f"emit: {err}"}
    rc, out, e = sh(cmd)
    blocks, cur = {}, None
    for l in out.splitlines():
        mm = re.match(r"example (\d+)$", l)
        if mm:
            cur = int(mm.group(1)); blocks[cur] = []; continue
        if cur is not None:
            blocks[cur].append(l)
    if n not in blocks:
        return {"error": f"no observation block for line {n}; exit {rc}; {e[-300:]}"}
    lines = blocks[n]
    outcome = "done"
    if lines and " refused: " in lines[-1]:
        outcome = outcome_of(lines.pop().split(" refused: ", 1)[1])
    return {"verdict": "accept", "outputs": lines, "outcome": outcome}

for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    try:
        op = req.get("op")
        if op == "hello":
            r = {"protocol": "rce-adapter/1", "implementation": f"rce emit {LANG}", "ops": ["run"], "features": ["fuel"]}
        elif op == "run":
            r = run_events(req["source"], req["events"]) if req.get("events") else run_main(req["source"], req.get("fuel"))
        else:
            r = {"error": f"op `{op}` is not served by an emitter"}
    except Exception as ex:
        r = {"error": f"adapter: {ex}"}
    r["id"] = req.get("id")
    sys.stdout.write(json.dumps(r) + "\n"); sys.stdout.flush()
