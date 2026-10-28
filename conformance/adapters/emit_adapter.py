#!/usr/bin/env python3
"""rce-adapter/1 for the program emitters of spec 10.5 (rust, python, js, scala).

An emitter is a back end: it translates an admitted module. So this adapter
answers `run` and `trace`. Entry (parse, check, admission) is the reference
front end's job and is certified through the reference adapter.

  run, no events   emit TARGET, run the program with `--run [--fuel=N]`
                   (fn main on the emitted tick runtime)
  run, events      add the line `example EVENTS -> true` (or `-> kept: E`
                   when the last event has the kept outcome) so the module has an
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
        # keyed by the emitted program, so a changed runtime is never served from a stale binary
        exe = os.path.join(TMP, "x" + hashlib.sha256(out.encode()).hexdigest()[:20])
        if not os.path.exists(exe):
            # a private output name, then an atomic rename: parallel jobs may build the same program
            tmp_exe = f"{exe}_{os.getpid()}"
            rc, o, e = sh([os.environ.get("RCE_RUSTC", "rustc"), "-O", "--edition", "2024", "-o", tmp_exe, prog], timeout=600)
            if rc != 0:
                return None, "rustc: " + e[-800:]
            os.replace(tmp_exe, exe)
        return [exe], None
    if LANG == "python":
        return [os.environ.get("RCE_PYTHON", "python3"), prog], None
    if LANG == "js":
        return [os.environ.get("RCE_NODE", "node"), prog], None
    return [os.environ.get("RCE_SCALA_CLI", "scala-cli"), "run", "--jvm", "21", "--server=false", "-q", prog, "--"], None

def outcome_of(r, word="kept"):
    """One spelling for an outcome (see ADAPTERS.md): trap:KIND, kept:E for an input, stopped:E for a run."""
    if r.startswith("trap: "): return "trap:" + r[6:]
    if r.startswith("trap(") and r.endswith(")"): return "trap:" + r[5:-1]
    return word + ":" + r

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
        elif o.startswith("Stopped: "):
            outcome = outcome_of(o[9:], "stopped")
    elif rc != 0:
        return {"error": f"exit {rc}: {e[-500:]}"}
    return {"verdict": "ok", "outputs": lines, "outcome": outcome}

def run_events(src, events, extra=()):
    base = src if src.endswith("\n") else src + "\n"
    n = base.count("\n") + 1
    cand = [base + f"example {events} -> true\n"]
    cmd, err = emit(cand[0])
    if cmd is None:
        m = re.search(r"kept[:(]\s*([A-Za-z_][\w]*)", err or "")
        trap = re.search(r"kept: trap\(([^)]*)\)", err or "")
        if m and not trap:
            cmd, err2 = emit(base + f"example {events} -> kept: {m.group(1)}\n")
            if cmd is None:
                return {"error": f"emit: {err2}"}
        elif trap:
            # `kept: trap("K")` (book 3.2/6.9): the checker's verdict only
            # makes the example well formed; the outcome compared is the one
            # the emitted program prints for it
            cmd, err2 = emit(base + f"example {events} -> kept: trap({json.dumps(trap.group(1))})\n")
            if cmd is None:
                return {"error": f"emit: {err2}"}
        else:
            return {"error": f"emit: {err}"}
    rc, out, e = sh(cmd + list(extra))
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
    if extra:
        return {"block": lines}
    outcome = "done"
    if lines and " kept: " in lines[-1]:
        rest = lines.pop().split(" kept: ", 1)[1]
        outcome = outcome_of(rest[:rest.find(") ") + 1] if rest.startswith("trap(") and ") " in rest else rest.split(" ", 1)[0])
    return {"verdict": "ok", "outputs": lines, "outcome": outcome}

GENESIS = "0" * 64

def trace(src, events):
    """6.1 D: the emitted program logs each example's D chain with `--trace`
    (canonical JSON, N3 floats, SHA-256 entries, as the reference writes it)
    and prints `trace HEAD`; the head of the last log is the trace head."""
    if events:
        r = run_events(src, events, ("--trace",))
        if "error" in r: return r
        heads = [l[6:] for l in r["block"] if l.startswith("trace ")]
    else:
        cmd, err = emit(src)
        if cmd is None:
            return {"error": f"emit: {err}"}
        rc, out, e = sh(cmd + ["--trace"])
        if rc != 0 and "trace " not in out:
            return {"error": f"exit {rc}: {e[-300:]}"}
        heads = [l[6:] for l in out.splitlines() if l.startswith("trace ")]
    return {"verdict": "ok", "trace": {"head": heads[-1] if heads else GENESIS, "steps": []}}

for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    try:
        op = req.get("op")
        if op == "hello":
            r = {"protocol": "rce-adapter/1", "implementation": f"rce emit {LANG}", "ops": ["run", "trace"], "features": ["fuel", "trace"]}
        elif op == "run":
            r = run_events(req["source"], req["events"]) if req.get("events") else run_main(req["source"], req.get("fuel"))
        elif op == "trace":
            r = trace(req["source"], req.get("events"))
        else:
            r = {"error": f"op `{op}` is not served by an emitter"}
    except Exception as ex:
        r = {"error": f"adapter: {ex}"}
    r["id"] = req.get("id")
    sys.stdout.write(json.dumps(r) + "\n"); sys.stdout.flush()
