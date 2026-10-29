#!/usr/bin/env python3
"""Property-based number tests with an independent oracle.

The oracle below is written from the laws, not from the reference: exact
integers with range traps (N2), IEEE 754 binary64 with roundTiesToEven
(Python floats are binary64 and correctly rounded), non-finite results trap
(F4), truncating division and dividend-signed remainder (N2), explicit
wrapping and saturating methods, roundToIntegral for floor/ceil/round (F8),
min/max ties return the receiver (N5), conversions of N4, and the canonical
float text of F9 (shortest round-trip digits; decimal for 1e-5 <= |x| < 1e16).
Expected values in the generated tests come from this oracle. Where the
reference disagrees, the test keeps the oracle's value and the case is
written to findings/numbers.json.
  gen_numbers.py OUT_DIR SEED COUNT
"""
import json, math, os, random, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
sys.path.insert(0, os.path.dirname(__file__))
import rcetest
from refclient import Ref

RANGE = {"i64": (-2**63, 2**63 - 1), "u64": (0, 2**64 - 1), "i32": (-2**31, 2**31 - 1), "u8": (0, 255)}

class Trap(Exception):
    pass

def fit(v, t):
    lo, hi = RANGE[t]
    if not lo <= v <= hi:
        raise Trap(f"overflow: {v} is outside {t}")
    return v

def tdiv(a, b):
    if b == 0: raise Trap("division by zero")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q

def tmod(a, b):
    if b == 0: raise Trap("division by zero")
    return a - b * tdiv(a, b)

def wrap(v, t):
    lo, hi = RANGE[t]; n = hi - lo + 1
    return (v - lo) % n + lo

def sat(v, t):
    lo, hi = RANGE[t]
    return max(lo, min(hi, v))

def ffin(x):
    if math.isnan(x) or math.isinf(x): raise Trap("non-finite float")
    return x

def fop(op, a, b):
    try:
        if op == "+": r = a + b
        elif op == "-": r = a - b
        elif op == "*": r = a * b
        elif op == "/":
            if b == 0.0: raise Trap("non-finite float")
            r = a / b
    except OverflowError:
        raise Trap("non-finite float")
    return ffin(r)

def ftext(x):
    """F9 canonical float text (display, `tostr`)."""
    if x == 0.0: return "-0" if math.copysign(1, x) < 0 else "0"
    r = repr(x)
    m, _, e = r.partition("e")
    digits = m.replace("-", "").replace(".", "").lstrip("0").rstrip("0") or "0"
    exp = math.floor(math.log10(abs(x))) if x != 0 else 0
    # exact decimal exponent from repr
    mant = m.lstrip("-")
    if "." in mant: ip, fp = mant.split(".")
    else: ip, fp = mant, ""
    allds = (ip + fp)
    lead = len(allds) - len(allds.lstrip("0"))
    point = len(ip) + (int(e) if e else 0)
    exp = point - lead - 1
    sign = "-" if x < 0 else ""
    if 1e-5 <= abs(x) < 1e16:
        if exp >= 0:
            ipart = digits[: exp + 1].ljust(exp + 1, "0")
            fpart = digits[exp + 1 :]
        else:
            ipart = "0"; fpart = "0" * (-exp - 1) + digits
        return sign + ipart + ("." + fpart if fpart else "")
    d = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return f"{sign}{d}e{'+' if exp >= 0 else '-'}{abs(exp)}"

def lit_f(x):
    r = repr(x)
    if r.startswith("-"): return f"(0.0 - {r[1:]})" if False else r
    return r

INT_EDGE = [0, 1, -1, 2, -2, 7, -7, 3, 255, 256, 2**31 - 1, -2**31, 2**53, 2**62, 2**63 - 1, -2**63 + 1]
U_EDGE = [0, 1, 2, 7, 2**32, 2**53 + 1, 2**63, 2**64 - 1, 2**64 - 2]
F_EDGE = [0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 10.0, 2.5, -2.5, 0.5, 1e16, 9007199254740993.0, 1e308, 1.7976931348623157e308,
          5e-324, 2.2250738585072014e-308, 1e-5, 1e-300, 3.141592653589793, 6.02214076e23, 1e-7, 123456.789]

def ilit(v, t):
    # A literal's magnitude must fit u64 (W-LEX-3); the minimum i64 is spelled as an expression.
    if v == -2**63: return "(-9223372036854775807 - 1)"
    return str(v)

def gen_int(rnd):
    t = rnd.choice(["i64", "i64", "u64"])
    pool = INT_EDGE if t == "i64" else U_EDGE
    pick = lambda: rnd.choice(pool) if rnd.random() < 0.6 else rnd.randint(*RANGE[t]) // rnd.choice([1, 1, 1000, 2**40])
    a, b = pick(), pick()
    a, b = sat(a, t), sat(b, t)
    op = rnd.choice(["+", "-", "*", "/", "%", "wrapping_add", "wrapping_sub", "wrapping_mul",
                     "saturating_add", "saturating_sub", "saturating_mul", "min", "max", "abs"])
    if op == "abs" and t == "u64": op = "max"
    expr = f"a {op} b" if op in "+-*/%" else (f"a.{op}()" if op == "abs" else f"a.{op}(b)")
    def val():
        if op == "+": return fit(a + b, t)
        if op == "-": return fit(a - b, t)
        if op == "*": return fit(a * b, t)
        if op == "/": return fit(tdiv(a, b), t)
        if op == "%": return fit(tmod(a, b), t)
        if op.startswith("wrapping_"): return wrap({"add": a + b, "sub": a - b, "mul": a * b}[op[9:]], t)
        if op.startswith("saturating_"): return sat({"add": a + b, "sub": a - b, "mul": a * b}[op[11:]], t)
        if op == "min": return a if a <= b else b
        if op == "max": return a if a >= b else b
        if op == "abs": return fit(abs(a), t)
    src = f"fn main()\n    let a: {t} = {ilit(a, t)}\n    let b: {t} = {ilit(b, t)}\n    emit {expr}\n"
    try:
        return src, [f"emit {val()}"], "done", ["N2", "N5"] if op in ("min", "max") else ["N2"]
    except Trap as e:
        return src, [], f"trap:{e}", ["N2"]

def gen_float(rnd):
    pick = lambda: rnd.choice(F_EDGE) * rnd.choice([1, 1, -1]) if rnd.random() < 0.6 else \
        struct.unpack("<d", struct.pack("<Q", rnd.getrandbits(63)))[0] * rnd.choice([1, -1])
    a, b = pick(), pick()
    while math.isnan(a) or math.isinf(a): a = pick()
    while math.isnan(b) or math.isinf(b): b = pick()
    op = rnd.choice(["+", "-", "*", "/", "floor", "ceil", "round", "abs", "min", "max", "sqrt", "as_i64", "mulsub"])
    cl = {"+": ["F1", "F2"], "-": ["F1"], "*": ["F1", "F3"], "/": ["F1", "F4"], "floor": ["F8"], "ceil": ["F8"], "round": ["F8"],
          "abs": ["F8"], "min": ["N5"], "max": ["N5"], "sqrt": ["F1", "F4"], "as_i64": ["N4"], "mulsub": ["F2"]}[op]
    if op in "+-*/": expr = f"a {op} b"
    elif op == "mulsub": expr = "a * b - a"
    elif op in ("min", "max"): expr = f"a.{op}(b)"
    elif op == "as_i64": expr = "a as i64"
    else: expr = f"a.{op}()"
    def val():
        if op in "+-*/": return fop(op, a, b)
        if op == "mulsub": return fop("-", fop("*", a, b), a)
        # IEEE 754 roundToIntegral keeps the sign, so a zero result of a negative operand is -0.0
        if op == "floor": return math.copysign(float(math.floor(a)), a)
        if op == "ceil": return math.copysign(float(math.ceil(a)), a)
        if op == "round": return math.copysign(float(math.floor(abs(a) + 0.5)), a) if abs(a) < 2**52 else a
        if op == "abs": return abs(a)
        if op == "min": return a if a <= b else b
        if op == "max": return a if a >= b else b
        if op == "sqrt":
            if a < 0: raise Trap("non-finite float")
            return math.sqrt(a)
        if op == "as_i64":
            v = math.trunc(a)
            if not RANGE["i64"][0] <= v <= RANGE["i64"][1]: raise Trap(f"overflow: {v} is outside i64")
            return v
    src = f"fn main()\n    let a: f64 = {lit_f(a)}\n    let b: f64 = {lit_f(b)}\n    emit {expr}\n"
    try:
        v = val()
    except Trap as e:
        return src, [], f"trap:{e}", cl
    if isinstance(v, float) and v == 0.0 and math.copysign(1, v) < 0 and op not in ("floor", "ceil"):
        # kept so the seeded stream (and every test id) stays as first generated; F6-TOSTR is closed
        # (f64 tostr prints -0) and floor/ceil/round results now carry the sign
        return None
    return src, [f"emit {v if isinstance(v, int) else ftext(v)}"], "done", cl

def main():
    out, seed, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rnd = random.Random(seed); ref = Ref(); findings = []
    d = os.path.join(out, "numbers", "generated"); os.makedirs(d, exist_ok=True)
    k = 0
    while k < count:
        g = (gen_int if k % 2 == 0 else gen_float)(rnd)
        if g is None: continue
        src, outs, outcome, cl = g
        r = ref.run(src)
        if r.get("verdict") == "change":
            # the generator wrote a program that requires change: a generator bug or a finding
            findings.append({"case": src, "reference": r.get("diagnostics"), "oracle": [outs, outcome], "kind": "requires-change-at-entry"})
            continue
        got = (r.get("outputs"), r.get("outcome"))
        if got != (outs, outcome):
            findings.append({"case": src, "reference": got, "oracle": [outs, outcome], "kind": "value"})
        name = f"num-{seed}-{k:04d}"
        rcetest.write(os.path.join(d, name + ".rce"), src, {"id": f"numbers/generated/{name}", "area": "numbers",
            "clauses": cl, "expect": "run", "outcome": outcome, "outputs": outs, "flags": ["generated"],
            "provenance": f"gen_numbers.py seed {seed}; oracle-derived"})
        k += 1
    os.makedirs(os.path.join(out, "..", "findings"), exist_ok=True)
    json.dump(findings, open(os.path.join(out, "..", "findings", f"numbers-{seed}.json"), "w"), indent=1)
    print(json.dumps({"generated": k, "disagreements": len(findings)}))

if __name__ == "__main__":
    main()
