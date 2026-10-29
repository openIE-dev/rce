"""Run a hand-written program on a stream and print what `rce run` prints.
The display rules here are the part of RCE's display (5.15) these eight
programs use: integers, booleans, text, lists, maps in key order, floats in
shortest form without a trailing .0, records and kinds.

  python3 drive.py NAME STREAM.json
"""
import dataclasses
import enum
import json
import math
import sys

from programs import PROGRAMS, Kept


def show_float(x):
    r = repr(x)
    if r.endswith(".0"):
        r = r[:-2]
    return r


def show(v):
    if isinstance(v, enum.Enum):  # a kind's variant displays as its name
        return v.name
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return show_float(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], float):  # a float key (programs.fkey)
        return show_float(math.copysign(abs(v[0]), v[1]))
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(show(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(show(k) + ": " + show(x) for k, x in sorted(v.items())) + "}"
    raise TypeError(type(v))


def show_region(r):
    fs = ", ".join(f.name + " = " + show(getattr(r, f.name)) for f in dataclasses.fields(r))
    return type(r).__name__ + " { " + fs + " }"


def show_rec(name, fs):
    return name + ("(" + ", ".join(k + ": " + show(v) for k, v in fs.items()) + ")" if fs else "")


def main():
    init, step = PROGRAMS[sys.argv[1]]
    s = init()
    out = ["0 (initial) " + "; ".join(show_region(r) for r in s)]
    for k, ev in enumerate(json.load(open(sys.argv[2])), 1):
        evs = show_rec(ev["type"], {f: v for f, v in ev.items() if f != "type"})
        try:
            s2, emits = step(s, ev)
        except Kept as r:
            out.append(f"{k} {evs} kept: {r}")
            continue
        s = list(s2) if isinstance(s2, tuple) else [s2]
        out.append(f"{k} {evs} " + "; ".join(show_region(r) for r in s))
        out += ["  emit " + show_rec(n, fs) for n, fs in emits]
    sys.stdout.write("".join(x + "\n" for x in out))


if __name__ == "__main__":
    main()
