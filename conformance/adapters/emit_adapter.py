#!/usr/bin/env python3
"""rce-adapter/1 for the program emitters of spec 10.5 (rust, python, js, scala).

An emitter is a back end: it translates an admitted module. So this adapter
answers `run` and `trace`. Entry (parse, check, admission) is the reference
front end's job and is certified through the reference adapter.

  run, no events   emit TARGET, run the program with `--run [--fuel=N]`
                   (fn main on the emitted tick runtime)
  run, events      add the line `example EVENTS -> true` (or `-> refused: E`
                   when the last event is refused) so the module has an
                   observation for exactly these events, emit, run, and read
                   that example's block. The example's expected value is used
                   only to admit the module; the lines compared are printed
                   by the emitted program.
  trace            the same example, run with `--trace --fuel-trace`: the
                   answer is the whole trace, the example's block without
                   its `example LINE` header (every entry of D as
                   `d applied|kept HASH BODY`, the state after each event,
                   the outputs, the reason an event was kept, `trace HEAD`
                   and `fuel N`), and its head.
The target is the one positional argument (rust, python, js or scala).
Env: RCE (the rce binary), RCE_RUSTC, RCE_PYTHON, RCE_NODE, RCE_SCALA_CLI.
"""
import hashlib, json, os, re, subprocess, sys, tempfile

LANG = sys.argv[1]
RCE = os.environ.get("RCE", "rce")
TMP = os.path.join(tempfile.gettempdir(), f"rce-conformance-{LANG}")
os.makedirs(TMP, exist_ok=True)
EXT = {"rust": "rs", "python": "py", "js": "js", "scala": "scala"}[LANG]
LAST_EXE = [None]   # rust: the one executable this process keeps

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
        # keyed by the emitted program, so a changed runtime is never served
        # from a stale binary, and private to this process; the process keeps
        # only its last executable (a run and a trace of the same events
        # share it), so a whole run's disk use stays bounded (an executable
        # is a few MB, and the suite has well over a thousand programs)
        exe = os.path.join(TMP, f"x{hashlib.sha256(out.encode()).hexdigest()[:20]}_{os.getpid()}")
        if not os.path.exists(exe):
            rc, o, e = sh([os.environ.get("RCE_RUSTC", "rustc"), "-O", "--edition", "2024", "-o", exe, prog], timeout=600)
            if rc != 0:
                return None, "rustc: " + e[-800:]
        if LAST_EXE[0] not in (None, exe) and os.path.exists(LAST_EXE[0]):
            os.remove(LAST_EXE[0])
        LAST_EXE[0] = exe
        return [exe], None
    if LANG == "python":
        return [os.environ.get("RCE_PYTHON", "python3"), prog], None
    if LANG == "js":
        return [os.environ.get("RCE_NODE", "node"), prog], None
    return [os.environ.get("RCE_SCALA_CLI", "scala-cli"), "run", "--jvm", "21", "--server=false", "-q", prog, "--"], None

def blocks_of(out):
    """The example blocks of an emitted program's output, in order, as
    (LINE, lines). A block starts at its `example LINE` header and ends at
    the next header or at `main`, the first line of fn main's observation,
    which comes after every example."""
    blocks, cur = [], None
    for l in out.splitlines():
        mm = re.match(r"example (\d+)$", l)
        if mm:
            cur = (int(mm.group(1)), []); blocks.append(cur); continue
        if l == "main":
            cur = None; continue
        if cur is not None:
            cur[1].append(l)
    return blocks

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

def run_events(src, events, extra=()):
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
            # `refused: trap("K")` (book 3.2/6.9): the checker's verdict only
            # makes the example well formed; the outcome compared is the one
            # the emitted program prints for it
            cmd, err2 = emit(base + f"example {events} -> refused: trap({json.dumps(trap.group(1))})\n")
            if cmd is None:
                return {"error": f"emit: {err2}"}
        else:
            return {"error": f"emit: {err}"}
    rc, out, e = sh(cmd + list(extra))
    blocks = dict(blocks_of(out))
    if n not in blocks:
        return {"error": f"no observation block for line {n}; exit {rc}; {e[-300:]}"}
    lines = blocks[n]
    if extra:
        return {"block": lines}
    outcome = "done"
    if lines and " refused: " in lines[-1]:
        outcome = outcome_of(lines.pop().split(" refused: ", 1)[1])
    return {"verdict": "accept", "outputs": lines, "outcome": outcome}

TRACE_ARGS = ("--trace", "--fuel-trace")

def trace(src, events):
    """6.1 D and its derived lines: with `--trace` the emitted program prints
    each entry of an example's D (canonical JSON, N3 floats, SHA-256, as the
    reference writes it) before the event's line and `trace HEAD` after the
    last event; with `--fuel-trace` it prints the example's `fuel N`. The
    whole block is returned; nothing in it is dropped or rewritten. With no
    events the module's last example is traced."""
    if events:
        r = run_events(src, events, TRACE_ARGS)
        if "error" in r: return r
        lines = r["block"]
    else:
        cmd, err = emit(src)
        if cmd is None:
            return {"error": f"emit: {err}"}
        rc, out, e = sh(cmd + list(TRACE_ARGS))
        blocks = blocks_of(out)
        if not blocks:
            return {"error": f"no example block; exit {rc}; {e[-300:]}"}
        lines = blocks[-1][1]
    heads = [l[6:] for l in lines if l.startswith("trace ")]
    return {"verdict": "accept", "trace": {"head": heads[-1] if heads else None, "lines": lines}}

def toolchain():
    """The version of the target's toolchain, as it reports it."""
    cmd = {"rust": [os.environ.get("RCE_RUSTC", "rustc"), "--version"],
           "python": [os.environ.get("RCE_PYTHON", "python3"), "--version"],
           "js": [os.environ.get("RCE_NODE", "node"), "--version"],
           "scala": [os.environ.get("RCE_SCALA_CLI", "scala-cli"), "--version"]}[LANG]
    try:
        rc, out, err = sh(cmd, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as ex:
        return f"{os.path.basename(cmd[0])} does not run: {ex}"
    name = os.path.basename(cmd[0])   # never a machine-local path in a report
    if rc != 0:
        return f"{name} exited {rc}"
    v = (out.strip() or err.strip()).splitlines()
    line = v[0] if v else ""
    # node prints only `vX.Y.Z`; the others name themselves
    return line if line.lower().startswith(name.lower()[:4]) else f"{name} {line}"

def emitted_by():
    """The checker that emits, from the provenance line of an emitted file
    (`emitted by rce VERSION (COMMIT)`)."""
    f = os.path.join(TMP, "hello.rce")
    open(f, "w").write("const K: i64 = 1\n")
    rc, out, err = sh([RCE, "emit", LANG, f])
    m = re.search(r"emitted by (rce \S+ \([^)]*\))", out.splitlines()[0] if out else "")
    return m.group(1) if m else f"unknown: {(err or out).strip()[:200]}"

for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    try:
        op = req.get("op")
        if op == "hello":
            r = {"protocol": "rce-adapter/1", "implementation": f"rce emit {LANG}", "emitted_by": emitted_by(),
                 "toolchain": toolchain(), "ops": ["run", "trace"], "features": ["fuel", "trace"]}
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
