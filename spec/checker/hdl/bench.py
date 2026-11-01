#!/usr/bin/env python3
"""bench.py: run a module printed by `rce emit verilog` (spec 10.5,
spec/proposals/emit-verilog.md EV-6 and EV-7) and print what it did.

    python3 bench.py iverilog MODULE.v WORKDIR
    python3 bench.py verilator MODULE.v WORKDIR
    python3 bench.py replay MODULE.v DUMPFILE
    python3 bench.py yosys MODULE.v WORKDIR

`iverilog` and `verilator` write the example events that the module's
`rce-hdl/1` line lists as one event stream (rce-hdl-stream/1, described
in rce_tb.v), simulate rce_tb.v with the module on that stream, and print
the trace of every example from the register dumps: a line `example LINE`,
then the lines an emitted program prints for that example under
`trace fuel-trace` (spec 10.5, `emit::trace_example`) except its last line
`fuel N`, which counts the reference evaluator's steps and has no
counterpart in a circuit. The state hash of every entry of D is computed
here, outside the hardware, from the dumped state registers: the
positional form of the state (6.1) as canonical JSON (6.1.1), and its
SHA-256. `replay` prints the same lines from a dump file written earlier
(the lines `R ...` and `S ...` that rce_tb.v prints). `yosys` synthesizes
the module with Yosys's generic `synth` and prints its cell counts.

The tools are found on PATH: iverilog and vvp, verilator (with make and a
C++ compiler, as `--binary` needs), yosys. The words are positional; the
flags below are the tools' own.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.path.join(HERE, "rce_tb.v")
GENESIS = "0" * 64


def interface(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("// rce-hdl/1 "):
                return json.loads(line[len("// rce-hdl/1 "):])
    sys.exit(f"{path}: no rce-hdl/1 line; is it a module of `rce emit verilog`?")


def ty_width(t, kinds):
    if t == "bool":
        return 1
    if isinstance(t, str):
        return {"i8": 8, "u8": 8, "i16": 16, "u16": 16, "i32": 32, "u32": 32}.get(t, 64)
    if "kind" in t:
        n = len(kinds[t["kind"]])
        b = 1
        while (1 << b) < n:
            b += 1
        return b
    return ty_width(t["array"], kinds) * t["len"]


def encode(t, v, kinds):
    """The bits of value v of type t, as an int (least significant first)."""
    if t == "bool":
        return 1 if v else 0
    if isinstance(t, str):
        return v & ((1 << ty_width(t, kinds)) - 1)
    if "kind" in t:
        return kinds[t["kind"]].index(v)
    w = ty_width(t["array"], kinds)
    out = 0
    for j, x in enumerate(v):
        out |= encode(t["array"], x, kinds) << (j * w)
    return out


def decode(t, bits, kinds):
    """The value of type t held in bits (an int)."""
    w = ty_width(t, kinds)
    bits &= (1 << w) - 1
    if t == "bool":
        return bits == 1
    if isinstance(t, str):
        if t.startswith("i") and bits >> (w - 1):
            return bits - (1 << w)
        return bits
    if "kind" in t:
        return ("kind", t["kind"], bits)
    ew = ty_width(t["array"], kinds)
    return [decode(t["array"], bits >> (j * ew), kinds) for j in range(t["len"])]


def show(v, kinds):
    """A value as `emit` lines and states show it (interp V::show)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, tuple):
        names = kinds[v[1]]
        return names[v[2]] if v[2] < len(names) else f"<variant {v[2]} of {v[1]}>"
    return "[" + ", ".join(show(x, kinds) for x in v) + "]"


def positional(v, kinds):
    """The positional form of a value (6.1, T7): a variant is [k, []]."""
    if isinstance(v, tuple):
        return [v[2], []]
    if isinstance(v, list):
        return [positional(x, kinds) for x in v]
    return v


def canonical(x):
    """Canonical JSON (6.1.1) of the values a circuit holds: integers,
    bools, lists, and objects with string keys in sorted order."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, list):
        return "[" + ",".join(canonical(y) for y in x) + "]"
    return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + canonical(x[k]) for k in sorted(x)) + "}"


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def state_of(I, bits):
    kinds = I["kinds"]
    return [[decode(f["type"], bits >> f["offset"], kinds) for f in r["fields"]] for r in I["regions"]]


def show_state(I, s):
    kinds = I["kinds"]
    parts = []
    for r, vs in zip(I["regions"], s):
        fs = ", ".join(f"{f['name']} = {show(v, kinds)}" for f, v in zip(r["fields"], vs))
        parts.append(f"{r['name']} {{ {fs} }}")
    return "; ".join(parts)


def event_value(I, ev):
    """The example event as values of its declared field types."""
    kinds = I["kinds"]
    decl = I["events"][ev["tag"]]
    vals = []
    for f, v in zip(decl["fields"], ev["values"]):
        vals.append(decode(f["type"], encode(f["type"], v, kinds), kinds))
    return decl, vals


def show_event(I, ev):
    decl, vals = event_value(I, ev)
    if not decl["fields"]:
        return decl["name"]
    kinds = I["kinds"]
    return decl["name"] + "(" + ", ".join(f"{f['name']}: {show(v, kinds)}" for f, v in zip(decl["fields"], vals)) + ")"


def signed(bits, w):
    bits &= (1 << w) - 1
    return bits - (1 << w) if bits >> (w - 1) else bits


def reason(I, err, trap, tval):
    if err:
        return I["errors"][err - 1]
    if trap:
        t = I["traps"][trap - 1]
        v = signed(tval, I["trap_value_width"])
        if t["kind"] == "overflow":
            return f"trap(overflow: {v} is outside {t['type']})"
        if t["kind"] == "index":
            return f"trap(index {v} out of range)"
        return f"trap({t['text']})"
    return "kept without a reason (the module is wrong)"


def widths(I):
    slots = max(1, I["emit_slots"])
    return {
        "TW": I["tag_width"],
        "PW": I["payload_width"],
        "TB": (I["tag_width"] + 7) // 8,
        "PB": (I["payload_width"] + 7) // 8,
        "ERW": I["error_width"],
        "TRW": I["trap_width"],
        "TVW": I["trap_value_width"],
        "NCW": I["emit_count_width"],
        "ETW": slots * I["emit_type_width"],
        "EVW": slots * I["emit_value_width"],
        "SW": I["state_width"],
        "RCE_TOP": I["module"],
    }


def stream(I):
    """The examples as one rce-hdl-stream/1 byte stream."""
    kinds = I["kinds"]
    w = widths(I)
    out = bytearray()
    for x in I["examples"]:
        out.append(0)
        for ev in x["events"]:
            out.append(1)
            out += ev["tag"].to_bytes(w["TB"], "little")
            decl = I["events"][ev["tag"]]
            p = 0
            for f, v in zip(decl["fields"], ev["values"]):
                p |= encode(f["type"], v, kinds) << f["offset"]
            out += p.to_bytes(w["PB"], "little")
    return bytes(out)


def trace(I, dump):
    """The trace lines of every example from the bench's dump lines."""
    kinds = I["kinds"]
    recs = [l.split() for l in dump if l.startswith(("R ", "S "))]
    out = []
    pos = 0
    for x in I["examples"]:
        if pos >= len(recs) or recs[pos][0] != "R":
            raise SystemExit(f"the dump has no reset for example {x['line']}")
        s = state_of(I, int(recs[pos][1], 16))
        pos += 1
        out.append(f"example {x['line']}")
        out.append(f"0 (initial) {show_state(I, s)}")
        prev, seq = GENESIS, 0
        for k, ev in enumerate(x["events"], 1):
            if pos >= len(recs) or recs[pos][0] != "S":
                raise SystemExit(f"the dump has no step for event {k} of example {x['line']}")
            _, applied, err, trap, tval, ecnt, etys, evals, st = recs[pos]
            pos += 1
            decl, vals = event_value(I, ev)
            evp = canonical({"payload": [positional(v, kinds) for v in vals], "type": ev["tag"]})
            ok = int(applied, 16) == 1
            if ok:
                s = state_of(I, int(st, 16))
            sh = sha(canonical(positional(s, kinds)))
            body = canonical({"event": "@", "prev_hash": prev, "seq": seq, "state_hash": sh}).replace('"@"', evp)
            h = sha(body)
            out.append(f"d {'applied' if ok else 'kept'} {h} {body}")
            prev, seq = h, seq + 1
            if ok:
                out.append(f"{k} {show_event(I, ev)} {show_state(I, s)}")
                n = int(ecnt, 16)
                ety, evw = I["emit_type_width"], I["emit_value_width"]
                tb, vb = int(etys, 16), int(evals, 16)
                for j in range(n):
                    t = I["emit_types"][(tb >> (j * ety)) & ((1 << ety) - 1)]
                    raw = (vb >> (j * evw)) & ((1 << evw) - 1)
                    if t == "int":
                        v = signed(raw, evw)
                    elif t == "bool":
                        v = raw == 1
                    else:
                        v = ("kind", t, raw)
                    out.append(f"  emit {show(v, kinds)}")
            else:
                out.append(f"{k} {show_event(I, ev)} kept: {reason(I, int(err, 16), int(trap, 16), int(tval, 16))}")
        out.append(f"trace {prev}")
    return out


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit(f"{cmd[0]} exited with status {p.returncode}")
    return p.stdout


def simulate(sim, module, work):
    I = interface(module)
    os.makedirs(work, exist_ok=True)
    sp = os.path.join(work, "stream.bin")
    with open(sp, "wb") as f:
        f.write(stream(I))
    defs = [f"-D{k}={v}" for k, v in widths(I).items()]
    if sim == "iverilog":
        vvp = os.path.join(work, "sim.vvp")
        run(["iverilog", "-g2005", "-s", "rce_tb", "-o", vvp] + defs + [TB, module])
        out = run(["vvp", "-n", vvp, f"+stream={sp}"])
    else:
        obj = os.path.join(work, "obj")
        run(["verilator", "--binary", "-Wno-fatal", "-Wno-lint", "-Wno-style", "--top-module", "rce_tb", "-Mdir", obj] + defs + [TB, module])
        out = run([os.path.join(obj, "Vrce_tb"), f"+stream={sp}"])
    lines = out.splitlines()
    with open(os.path.join(work, "dump.txt"), "w") as f:
        f.write(out)
    bad = [l for l in lines if l.startswith("E ")]
    if bad:
        raise SystemExit(bad[0])
    return I, lines


def synth(module, work):
    I = interface(module)
    os.makedirs(work, exist_ok=True)
    out = run(["yosys", "-p", f"read_verilog {module}; synth -top {I['module']}; stat"])
    with open(os.path.join(work, "yosys.log"), "w") as f:
        f.write(out)
    # the last `stat` block: the total and the count of each cell type
    i = out.rfind("Printing statistics")
    block = out[i:] if i >= 0 else out
    total = None
    cells = []
    for l in block.splitlines():
        m = re.match(r"\s+Number of cells:\s+(\d+)", l) or re.match(r"\s+(\d+)\s+cells$", l)
        if m and total is None:
            total = int(m.group(1))
            continue
        m = re.match(r"\s+(\$_\w+_)\s+(\d+)$", l) or re.match(r"\s+(\d+)\s+(\$_\w+_)$", l)
        if m:
            a, b = m.groups()
            name, n = (a, int(b)) if a.startswith("$") else (b, int(a))
            cells.append((name, n))
    print(f"module {I['module']}")
    print(f"cells {total}")
    for name, n in cells:
        print(f"  {name} {n}")


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("iverilog", "verilator", "replay", "yosys"):
        sys.exit("usage: bench.py iverilog|verilator|yosys MODULE.v WORKDIR | bench.py replay MODULE.v DUMPFILE")
    word, module, third = sys.argv[1:]
    if word == "yosys":
        synth(module, third)
        return
    if word == "replay":
        I = interface(module)
        with open(third, encoding="utf-8") as f:
            lines = f.read().splitlines()
    else:
        I, lines = simulate(word, module, third)
    for l in trace(I, lines):
        print(l)


if __name__ == "__main__":
    main()
