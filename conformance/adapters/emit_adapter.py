#!/usr/bin/env python3
"""rce-adapter/1 for the program emitters of spec 10.5 (rust, python, js, scala).

An emitter is a back end: it translates an admitted module. So this adapter
answers `run` and `trace`. Entry (parse, check, admission) is the reference
front end's job and is certified through the reference adapter.

  run, no events   emit TARGET, run the program with `--run [--fuel=N]`
                   (fn main on the emitted tick runtime)
  run, events      add the line `example EVENTS -> true` (or `-> kept: E`
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
Env: RCE (the rce binary), RCE_RUSTC, RCE_PYTHON, RCE_NODE, RCE_SCALA_CLI,
RCE_SCALA_TRY (seconds before a scala-cli call is retried without its server).
"""
import hashlib, json, os, re, subprocess, sys, tempfile

LANG = sys.argv[1]
RCE = os.environ.get("RCE", "rce")
TMP = os.path.join(tempfile.gettempdir(), f"rce-conformance-{LANG}")
os.makedirs(TMP, exist_ok=True)
EXT = {"rust": "rs", "python": "py", "js": "js", "scala": "scala"}[LANG]
SCALA_CLI = os.environ.get("RCE_SCALA_CLI", "scala-cli")
SCALA_VERSION = "3.10.0"  # pinned, as in spec/checker/tests/emit*.rs
LAST_EXE = [None]   # rust: the one executable this process keeps

def sh(cmd, inp=None, timeout=300):
    # its own process group, so a timeout also ends the processes it started
    # (scala-cli's JVM), which would otherwise hold its output open
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if inp is not None else None, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, start_new_session=(os.name != "nt"))
    try:
        out, err = p.communicate(inp, timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(p.pid, 9)
        else:
            p.kill()
        p.communicate()
        raise
    return p.returncode, out, err

def write(path, text):
    """Write a whole file under its name at once: parallel jobs write the
    same names (a run and a trace of the same events emit the same
    program), and a reader must never see a file another job is writing."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as w:
        w.write(text)
    os.replace(tmp, path)

def emit(src):
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    f = os.path.join(TMP, f"m{h}.rce"); write(f, src)
    rc, out, err = sh([RCE, "emit", LANG, f])
    if rc != 0:
        return None, err.strip() or out.strip()
    prog = os.path.join(TMP, f"m{h}.{EXT}"); write(prog, out)
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
    # scala-cli's compile server (Bloop) stays warm across the suite's
    # programs. It keeps every project it has built, and one project per
    # program made it fail after about 1,400 of them, so each adapter process
    # builds its programs, one at a time, from one file of its own
    wd = os.path.join(TMP, f"w{os.getpid()}"); os.makedirs(wd, exist_ok=True)
    prog = os.path.join(wd, "main.scala"); open(prog, "w").write(out)
    return [SCALA_CLI, "run", "--jvm", "21", "--scala", SCALA_VERSION, "--scalac-option", "-nowarn", "-q", prog, "--"], None

def toolchain():
    """The tool that compiles or runs the emitted program, as it names itself."""
    cmd = {"rust": [os.environ.get("RCE_RUSTC", "rustc"), "--version"],
           "python": [os.environ.get("RCE_PYTHON", "python3"), "--version"],
           "js": [os.environ.get("RCE_NODE", "node"), "--version"],
           "scala": [SCALA_CLI, "--version"]}[LANG]
    try:
        rc, out, err = sh(cmd, timeout=120)
    except Exception as ex:
        return f"{cmd[0]}: {ex}"
    v = "; ".join(l.strip() for l in (out + err).splitlines() if l.strip() and not l.startswith("Picked up "))
    return v + (f"; Scala {SCALA_VERSION}, JVM 21" if LANG == "scala" else "")

SCALA_TRY = float(os.environ.get("RCE_SCALA_TRY", "150"))

def run_prog(cmd):
    """Run an emitted program. scala-cli builds it on its compile server
    first; when the call does not end within SCALA_TRY seconds, or scala-cli
    itself fails (its own stack trace, `at scala.cli.`, on standard error),
    the server is at fault, not the program, and the program is run once
    more with `--server=false`, a compiler of its own. The lines compared are
    those of that second run; a program that hangs times out both times."""
    if LANG != "scala":
        return sh(cmd)
    i = cmd.index("run") + 1
    alone = cmd[:i] + ["--server=false"] + cmd[i:]
    try:
        rc, out, err = sh(cmd, timeout=SCALA_TRY)
    except subprocess.TimeoutExpired:
        return sh(alone, timeout=SCALA_TRY)
    if rc != 0 and "at scala.cli." in err:
        return sh(alone, timeout=SCALA_TRY)
    return rc, out, err

def blocks_of(out):
    """The example blocks of an emitted program's output, in order, as
    (LINE, lines). A block starts at its `example LINE` header and ends at
    the next header, at `main` (the first line of fn main's observation,
    which comes after every example), or at the observation of an item that
    follows the example: `const NAME`, `check LINE` or `fn NAME` (with its
    `fuel N`). An example's own lines never start so: they are `initial`,
    `K ...` for event K, `  emit`, `d`, `trace` and `fuel`. So a block holds
    the example's own lines only, as emit::trace_example returns them."""
    blocks, cur = [], None
    for l in out.splitlines():
        mm = re.match(r"example (\d+)$", l)
        if mm:
            cur = (int(mm.group(1)), []); blocks.append(cur); continue
        if l == "main" or re.match(r"(const|check|fn) ", l):
            cur = None; continue
        if cur is not None:
            cur[1].append(l)
    return blocks

def outcome_of(r):
    """One spelling for an outcome (see ADAPTERS.md): trap:KIND or refused:E."""
    if r.startswith("trap: "): return "trap:" + r[6:]
    if r.startswith("trap(") and r.endswith(")"): return "trap:" + r[5:-1]
    return "kept:" + r

def run_main(src, fuel):
    cmd, err = emit(src)
    if cmd is None:
        return {"error": f"emit: {err}"}
    args = ["--run"] + ([f"--fuel={fuel}"] if fuel else [])
    rc, out, e = run_prog(cmd + args)
    lines = [l for l in out.splitlines() if l and not l.startswith("actorlog ")]
    outcome = "done"
    if lines and lines[-1].startswith("outcome "):
        o = lines.pop()[8:]
        if o == "Done": outcome = "done"
        elif o == "Deadlock": outcome = "deadlock"
        elif o.startswith("Stopped: "):
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
    rc, out, e = run_prog(cmd + list(extra))
    blocks = dict(blocks_of(out))
    if n not in blocks:
        return {"error": f"no observation block for line {n}; exit {rc}; {e[-300:]}"}
    lines = blocks[n]
    if extra:
        return {"block": lines}
    outcome = "done"
    if lines and " kept: " in lines[-1]:
        outcome = outcome_of(lines.pop().split(" kept: ", 1)[1])
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
        rc, out, e = run_prog(cmd + list(TRACE_ARGS))
        blocks = blocks_of(out)
        if not blocks:
            return {"error": f"no example block; exit {rc}; {e[-300:]}"}
        lines = blocks[-1][1]
    heads = [l[6:] for l in lines if l.startswith("trace ")]
    return {"verdict": "accept", "trace": {"head": heads[-1] if heads else None, "lines": lines}}


def emitted_by():
    """The checker that emits, from the provenance line of an emitted file
    (`emitted by rce VERSION (COMMIT)`)."""
    f = os.path.join(TMP, "hello.rce")
    write(f, "const K: i64 = 1\n")
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
