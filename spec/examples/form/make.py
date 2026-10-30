#!/usr/bin/env python3
"""Write the fixtures of the gate on rce-form/1 (spec/proposals/rce-form.md, §10
and §11) into this directory, and `expect.json`, the outcome each fixture must
have.

Each fixture is built here from the rules as the proposal writes them, with
an encoder written from the text of 6.1.1 and §10 alone: group keys, module
keys and Merkle digests (C2) are computed here, and so are the orders that C3
(symmetric operands), C5 (numbering), C7 (captures) and C9 (recursive groups)
fix. A fixture that requires change breaks exactly one rule and is built canonical in every
other respect. The expected outcomes in `expect.json` are written by hand
from the rules, in the table at the end of this file; the checker's code is
never run to make them.

    python3 spec/examples/form/make.py           write the fixtures
    python3 spec/examples/form/make.py --check   fail when a written file differs
"""
import hashlib
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAT_TEXT = open(os.path.join(HERE, "..", "..", "data", "form-catalogue.json"), "rb").read()
CAT_KEY = hashlib.sha256(CAT_TEXT).hexdigest()
CAT = {e["id"]: e for e in json.loads(CAT_TEXT)["entries"]}


# ---- canonical JSON (6.1.1) and SHA-256

def jstr(s):
    o = ['"']
    for c in s:
        if c == '"':
            o.append('\\"')
        elif c == "\\":
            o.append("\\\\")
        elif c == "\n":
            o.append("\\n")
        elif c == "\r":
            o.append("\\r")
        elif c == "\t":
            o.append("\\t")
        elif ord(c) < 0x20:
            o.append("\\u%04x" % ord(c))
        else:
            o.append(c)
    o.append('"')
    return "".join(o)


def canon(v):
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # 6.1.1 rule 3 for the values the fixtures hold: 1e-4 <= |x| < 1e16
        r = repr(v)
        assert "e" not in r and "." in r and 1e-4 <= abs(v) < 1e16, v
        return r
    if isinstance(v, str):
        return jstr(v)
    if isinstance(v, list):
        return "[" + ",".join(canon(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(jstr(k) + ":" + canon(v[k]) for k in sorted(v)) + "}"
    raise TypeError(v)


def sha(v):
    return hashlib.sha256(canon(v).encode()).hexdigest()


# ---- types

U64 = ["int", "0", "18446744073709551615"]
U8 = ["int", "0", "255"]
BOOL = ["bool"]
ORDER = ["order"]
BYTES = ["bytes"]
HALT = ["sum", BYTES, ["tup", BYTES, BYTES]]


def lit_int(lo, hi=None):
    return ["int", str(lo), str(lo if hi is None else hi)]


# ---- regions, built node by node with their Merkle digests (C2)

class Region:
    def __init__(self, nin):
        self.nin = nin
        self.nodes = []

    def inputs(self):
        return [("in", j) for j in range(self.nin)]

    def md(self, s):
        if s[0] == "in":
            return sha(["in", s[1]])
        return sha(["out", self.nodes[s[1]]["md"], s[2]])

    def add(self, k, attrs, ins, regions=(), nout=1, sym="sorted"):
        """Add a node; its symmetric operands (C3) in ascending digest, or
        in descending digest when `sym` is "reversed"."""
        ins = list(ins)
        if k == 1 and attrs[0] == "prim":
            s = CAT.get(attrs[1], {}).get("symmetric")
            if s:
                a, b = s
                lo_first = self.md(ins[a]) <= self.md(ins[b])
                if lo_first != (sym == "sorted"):
                    ins[a], ins[b] = ins[b], ins[a]
        rj = [r[0] for r in regions]
        rmd = [r[1] for r in regions]
        md = sha([k, attrs, [self.md(s) for s in ins], rmd])
        self.nodes.append({"k": k, "attrs": attrs, "ins": ins, "regions": rj, "md": md})
        i = len(self.nodes) - 1
        return [("out", i, p) for p in range(nout)]

    def lit(self, t, v):
        return self.add(0, ["lit", t, v], [])[0]

    def prim(self, name, targs, ins, nout=1, cell=None, sym="sorted"):
        attrs = ["prim", name, targs] + ([cell] if cell is not None else [])
        return self.add(1, attrs, ins, nout=nout, sym=sym)

    def call(self, ref, ins, nout=1):
        return self.add(1, ["call", ref], ins, nout=nout)

    def halt(self, types, value, omega):
        return self.add(13, [types], [value, omega], nout=len(types))

    def branch(self, sel, caps, arms, types, order="sorted"):
        """A Branch capturing `caps`, in ascending digest (C7), or in
        descending digest when `order` is "reversed". Each arm is a function
        of a region and the arm's handles of the captures, in the order given."""
        idx = sorted(range(len(caps)), key=lambda j: self.md(caps[j]), reverse=(order == "reversed"))
        pos = {j: idx.index(j) for j in range(len(caps))}
        regs = []
        for f in arms:
            r = Region(len(caps))
            outs = f(r, [("in", pos[j]) for j in range(len(caps))])
            regs.append(r.finish(outs))
        return self.add(3, [len(arms), types], [sel] + [caps[j] for j in idx], regs, nout=len(types))

    def iterate(self, attrs, ins, test, body, ncarried):
        regs = []
        for f in (test, body):
            r = Region(ncarried)
            regs.append(r.finish(f(r, r.inputs())))
        return self.add(4, attrs, ins, regs, nout=ncarried)

    def finish(self, outs, numbering="dfs"):
        """The region's JSON with nodes numbered as C5 states (first
        completion of a depth-first walk from the outputs, sources in port
        order), or in the order they were added; and its digest."""
        md = sha(["region", [self.md(s) for s in outs]])
        n = len(self.nodes)
        if numbering == "dfs":
            order, done = [], [False] * n
            for s in outs:
                if s[0] != "out" or done[s[1]]:
                    continue
                stack = [(s[1], 0)]
                while stack:
                    i, k = stack[-1]
                    ins = self.nodes[i]["ins"]
                    if k < len(ins):
                        stack[-1] = (i, k + 1)
                        if ins[k][0] == "out" and not done[ins[k][1]] and all(x[0] != ins[k][1] for x in stack):
                            stack.append((ins[k][1], 0))
                        continue
                    stack.pop()
                    if not done[i]:
                        done[i] = True
                        order.append(i)
            order += [i for i in range(n) if not done[i]]
        else:
            order = list(range(n))
        pos = {old: new for new, old in enumerate(order)}

        def src(s):
            return [-1, s[1]] if s[0] == "in" else [pos[s[1]], s[2]]

        nodes = [[self.nodes[i]["k"], self.nodes[i]["attrs"], [src(s) for s in self.nodes[i]["ins"]], self.nodes[i]["regions"]] for i in order]
        return {"nodes": nodes, "out": [src(s) for s in outs]}, md


def fn(ins, outs, eff, build, bound=None, numbering="dfs"):
    r = Region(len(ins))
    body, _ = r.finish(build(r, r.inputs()), numbering)
    return {"sig": [ins, outs, eff], "bound": bound, "body": body}


def group(*fns):
    g = {"fns": list(fns)}
    return sha(g), g


def module(defs, regions=(), events=(), steps=(), types=(), errors=(), main=None, caps=()):
    return {
        "format": "rce-form/1",
        "catalogue": CAT_KEY,
        "types": list(types),
        "regions": list(regions),
        "events": list(events),
        "errors": list(errors),
        "steps": list(steps),
        "main": main,
        "caps": list(caps),
        "defs": {k: g for k, g in defs},
    }


def text_hex(s):
    return s.encode().hex()


# ---- the runtime definition that writes a trap text

def trap_group(text="overflow", numbering="dfs", extra=False):
    def build(r, ins):
        if numbering == "added":
            # the text first, then the index: not the order C5 fixes
            t = r.lit(BYTES, text_hex(text))
            i = r.lit(lit_int(0), "0")
        else:
            i = r.lit(lit_int(0), "0")
            t = r.lit(BYTES, text_hex(text))
        if extra:
            r.lit(BOOL, True)  # reached from no output (C6)
        return r.prim("sum.mk", [HALT], [i, t])
    return group(fn([], [HALT], 0, build, numbering=numbering))


TRAP_KEY, TRAP = trap_group()


# ---- the counter of §10's worked example

def guarded_step(bad, op, capture_order="sorted", sym="sorted", dup_literal=False, inline=True, omega_twice=False, unguarded=False):
    """A step body n -> n op 1, guarded inline by its bad test."""
    def build(r, ins):
        n, om = ins
        one = r.lit(lit_int(1), "1")
        if unguarded:
            s = r.prim(op, [U64], [n, one])[0]
            return [s, om]
        one2 = r.lit(lit_int(1), "1") if dup_literal else one
        b = r.prim(bad, [U64], [n, one], sym=sym)[0]

        def arm0(a, h):
            hom, hn, hone = h
            return [a.prim(op, [U64], [hn, hone])[0], hom]

        def arm1(a, h):
            hom, hn, hone = h
            t = a.call([TRAP_KEY, 0], [])[0]
            x = a.halt([U64, ORDER], t, hom)
            return [x[0], hom if omega_twice else x[1]]
        out = r.branch(b, [om, n, one2], [arm0, arm1], [U64, ORDER], order=capture_order)
        return [out[0], out[1]]
    return build


def counter(variant=None):
    v = variant or ""
    eff = 1 if v == "fg7" else 2
    inc_kw = {}
    if v == "c3":
        inc_kw["sym"] = "reversed"
    if v == "c4":
        inc_kw["dup_literal"] = True
    if v == "c7":
        inc_kw["capture_order"] = "reversed"
    if v == "fg6":
        inc_kw["omega_twice"] = True
    if v == "fg8":
        inc_kw["unguarded"] = True
    defs = []
    trap_key, trap = TRAP_KEY, TRAP
    if v == "c5":
        trap_key, trap = trap_group(numbering="added")
    if v == "c6":
        trap_key, trap = trap_group(extra=True)
    defs.append((trap_key, trap))

    def with_trap(build):
        # the step bodies call the trap group this variant writes
        global TRAP_KEY
        saved = TRAP_KEY
        TRAP_KEY = trap_key
        try:
            return fn([U64, ORDER], [U64, ORDER], eff, build)
        finally:
            TRAP_KEY = saved

    if v == "c11":
        # the guarded addition as a function of its own, called by the step
        def guard_body(r, ins):
            n, one, om = ins
            b = r.prim("int.bad.add", [U64], [n, one])[0]

            def arm0(a, h):
                hom, hn, hone = h
                return [a.prim("int.add", [U64], [hn, hone])[0], hom]

            def arm1(a, h):
                hom, hn, hone = h
                t = a.call([trap_key, 0], [])[0]
                x = a.halt([U64, ORDER], t, hom)
                return [x[0], x[1]]
            out = r.branch(b, [om, n, one], [arm0, arm1], [U64, ORDER])
            return [out[0], out[1]]
        gk, g = group(fn([U64, lit_int(1), ORDER], [U64, ORDER], 1, guard_body))
        defs.append((gk, g))

        def inc_call(r, ins):
            n, om = ins
            one = r.lit(lit_int(1), "1")
            out = r.call([gk, 0], [n, one, om], nout=2)
            return [out[0], out[1]]
        inc = fn([U64, ORDER], [U64, ORDER], eff, inc_call)
    else:
        inc = with_trap(guarded_step("int.bad.add", "int.add", **inc_kw))
    dec = with_trap(guarded_step("int.bad.sub", "int.sub"))
    k_inc, g_inc = group(inc)
    k_dec, g_dec = group(dec)

    def pos(r, ins):
        (n,) = ins
        if v == "fg5":
            z = r.lit(BOOL, False)
        else:
            z = r.lit(lit_int(0), "0")
        name = "int.less" if v == "fg1" else "int.lt"
        extra = [n] if v == "fg4" else []
        return r.prim(name, [U64], [z, n] + extra)
    k_pos, g_pos = group(fn([U64], [BOOL], 0, pos))
    defs += [(k_inc, g_inc), (k_dec, g_dec), (k_pos, g_pos)]
    init = "abc" if v == "fg10" else 0
    ev0 = 7 if v == "fg11" else 0
    caps = ["ab" * 32] if v == "fg13" else []
    if v == "fg2":
        # the guard's group under a key that is not the SHA-256 of its bytes
        bad_key = sha("not this group")
        defs = [(bad_key if k == k_pos else k, g) for k, g in defs]
        k_pos = bad_key
    return module(
        defs,
        regions=[["state", [[U64, init]]]],
        events=[[], []],
        steps=[[0, ev0, None, [k_inc, 0], None], [0, 1, [k_pos, 0], [k_dec, 0], None]],
        caps=caps,
    )


# ---- declared types, an option field, a payload and an emitted float

def types_module(variant=None):
    v = variant or ""
    rec = ["record", [U8, ["text"]]]
    kind = ["kind", [[], [U8]]]
    if v == "c8":
        decls = [rec, rec]
        rec_i, kind_payload = 0, [["decl", 1]]
    else:
        decls = sorted([rec, kind], key=sha)
        rec_i = decls.index(rec)
        kind_payload = [["decl", decls.index(kind)]]
    if v == "fg14":
        kind_payload = [["opt", ["unit"]]]
    opt = ["sum", ["tup"], ["f64"]]
    lit = "7ff8000000000000" if v == "fg12" else "3ff8000000000000"

    def body(r, ins):
        rv, old, x, om = ins
        one = r.lit(lit_int(1), "1")
        some = r.prim("sum.mk", [opt], [one, x])[0]
        f = r.lit(["f64"], lit)
        om2 = r.prim("emit", [["f64"]], [f, om])[0]
        return [rv, some, om2]
    k, g = group(fn([["decl", rec_i], opt, ["f64"], ORDER], [["decl", rec_i], opt, ORDER], 2, body))
    return module(
        [(k, g)],
        types=decls,
        regions=[["state", [[["decl", rec_i], [0, "a"]], [["opt", ["f64"]], None]]]],
        events=[[["f64"]], kind_payload],
        steps=[[0, 0, None, [k, 0], None]],
    )


# ---- a recursive group with declared bounds (C9, FG8)

def parity_fn(answer, other, bound):
    def build(r, ins):
        n, om = ins
        z = r.lit(lit_int(0), "0")
        isz = r.prim("int.eq", [U64], [n, z])[0]

        def nonzero(a, h):
            hn, hom = h
            one = a.lit(lit_int(1), "1")
            bad = a.prim("int.bad.sub", [U64], [hn, one])[0]

            def ok(b, k):
                kn, kone, kom = k
                m = b.prim("int.sub", [U64], [kn, kone])[0]
                out = b.call(["rec", other], [m, kom], nout=2)
                return [out[0], out[1]]

            def over(b, k):
                kn, kone, kom = k
                t = b.call([TRAP_KEY, 0], [])[0]
                x = b.halt([BOOL, ORDER], t, kom)
                return [x[0], x[1]]
            out = a.branch(bad, [hn, one, hom], [ok, over], [BOOL, ORDER])
            return [out[0], out[1]]

        def zero(a, h):
            hn, hom = h
            return [a.lit(BOOL, answer), hom]
        out = r.branch(isz, [n, om], [nonzero, zero], [BOOL, ORDER])
        return [out[0], out[1]]
    return fn([U64, ORDER], [BOOL, ORDER], 1, build, bound=bound)


def rec_digest(f, d):
    def sub(j):
        if isinstance(j, list):
            if len(j) == 2 and j[0] == "rec" and isinstance(j[1], int):
                return ["rec", d(j[1])]
            return [sub(x) for x in j]
        if isinstance(j, dict):
            return {k: sub(x) for k, x in j.items()}
        return j
    return sha(sub(f))


def c9_order(fns):
    """C9: the iterated digest of each member, rounds 0 to n - 1."""
    n = len(fns)
    d = [rec_digest(f, lambda _: "?") for f in fns]
    for _ in range(1, n):
        prev = d
        d = [rec_digest(f, lambda j: prev[j]) for f in fns]
    return d


def parity_group(variant=None):
    """even at one position and odd at the other: the canonical order is the
    one whose members are already in ascending iterated digest; the variant
    "c9" writes the other order."""
    bound = None if variant == "fg8-rec" else "1000"
    for even_at in (0, 1):
        odd_at = 1 - even_at
        fns = [None, None]
        fns[even_at] = parity_fn(True, odd_at, bound)
        fns[odd_at] = parity_fn(False, even_at, bound)
        d = c9_order(fns)
        canonical = d == sorted(d) and len(set(d)) == 2
        if canonical == (variant != "c9"):
            return group(*fns), even_at
    raise SystemExit("no order of the parity group is canonical")


def group_module(variant=None):
    (gk, g), even_at = parity_group(variant)

    def body(r, ins):
        n, om = ins
        out = r.call([gk, even_at], [n, om], nout=2)
        om2 = r.prim("emit", [BOOL], [out[0], out[1]])[0]
        return [n, om2]
    k, s = group(fn([U64, ORDER], [U64, ORDER], 2, body))
    return module([(TRAP_KEY, TRAP), (gk, g), (k, s)], regions=[["state", [[U64, 4]]]], events=[[]], steps=[[0, 0, None, [k, 0], None]])


# ---- loops, indices, tags and a memo cell (FG6, FG8)

def loops_module():
    i100 = ["int", "0", "100"]

    def raise_i(b, k):
        """The body region of a loop over (ω, i, j): i + 1, guarded."""
        om, i, j = k
        one = b.lit(lit_int(1), "1")
        bad = b.prim("int.bad.add", [i100], [i, one])[0]

        def ok(a, h):
            hom, hi, hone = h
            return [a.prim("int.add", [i100], [hi, hone])[0], hom]

        def over(a, h):
            hom, hi, hone = h
            t = a.call([TRAP_KEY, 0], [])[0]
            x = a.halt([i100, ORDER], t, hom)
            return [x[0], x[1]]
        out = b.branch(bad, [om, i, one], [ok, over], [i100, ORDER])
        return [b.lit(BOOL, True), out[1], out[0], j]

    def measured(r, ins):
        i, j, om = ins

        def test(t, k):
            tom, ti, tj = k
            return [t.prim("int.lt", [i100], [ti, tj])[0], tom, ti, tj]
        out = r.iterate(["measure", [1, 2], [ORDER, i100, i100]], [om, i, j], test, raise_i, 3)
        return [out[1], out[0]]
    k_measure, g_measure = group(fn([i100, i100, ORDER], [i100, ORDER], 1, measured))

    def counted(r, ins):
        i, om = ins
        bound = r.lit(lit_int(10), "10")
        three = r.lit(lit_int(3), "3")

        def test(t, k):
            tom, ti, tj = k
            return [t.prim("int.lt", [i100], [ti, tj])[0], tom, ti, tj]
        out = r.iterate(["count", [ORDER, i100, lit_int(3)]], [bound, om, i, three], test, raise_i, 3)
        return [out[1], out[0]]
    k_count, g_count = group(fn([i100, ORDER], [i100, ORDER], 1, counted))

    def first(r, ins):
        xs, om = ins
        ln = r.prim("list.len", [U8], [xs], nout=2)
        z = r.lit(lit_int(0), "0")
        c = r.prim("int.lt", [U64], [z, ln[0]])[0]

        def empty(a, h):
            hxs, hz, hom = h
            t = a.call([TRAP_KEY, 0], [])[0]
            x = a.halt([U8, ORDER], t, hom)
            return [x[0], x[1]]

        def some(a, h):
            hxs, hz, hom = h
            e = a.prim("list.get", [U8], [hxs, hz], nout=2)
            return [e[0], hom]
        out = r.branch(c, [ln[1], z, om], [empty, some], [U8, ORDER])
        return [out[0], out[1]]
    k_first, g_first = group(fn([["list", U8], ORDER], [U8, ORDER], 1, first))

    opt = ["sum", ["tup"], U8]

    def payload(r, ins):
        (v,) = ins
        t = r.prim("sum.tag", [opt], [v], nout=2)

        def none(a, h):
            return [a.lit(U8, "0")]

        def some(a, h):
            (hv,) = h
            one = a.lit(lit_int(1), "1")
            return [a.prim("sum.get", [opt], [hv, one])[0]]
        return [r.branch(t[0], [t[1]], [none, some], [U8])[0]]
    k_payload, g_payload = group(fn([opt], [U8], 0, payload))

    k_c, g_c = group(fn([], [U64], 0, lambda r, ins: [r.lit(lit_int(42), "42")]))
    cell = ["memo", [k_c, 0]]
    mopt = ["sum", ["tup"], U64]

    def memo(r, ins):
        (om,) = ins
        got = r.prim("memo.get", [U64], [om], nout=2, cell=cell)
        t = r.prim("sum.tag", [mopt], [got[0]], nout=2)

        def empty(a, h):
            hs, hom = h
            x = a.call([k_c, 0], [])[0]
            return [x, a.prim("memo.put", [U64], [x, hom], cell=cell)[0]]

        def full(a, h):
            hs, hom = h
            one = a.lit(lit_int(1), "1")
            x = a.prim("sum.get", [mopt], [hs, one])[0]
            return [x, a.prim("memo.put", [U64], [x, hom], cell=cell)[0]]
        out = r.branch(t[0], [t[1], got[1]], [empty, full], [U64, ORDER])
        return [out[0], out[1]]
    k_memo, g_memo = group(fn([ORDER], [U64, ORDER], 1, memo))
    return module([(TRAP_KEY, TRAP), (k_measure, g_measure), (k_count, g_count), (k_first, g_first), (k_payload, g_payload), (k_c, g_c), (k_memo, g_memo)])


# ---- a region with a cycle, which FG4 keeps before FG9 is reached

def cycle_module():
    m = counter()
    k_pos = m["steps"][1][2][0]
    g = m["defs"].pop(k_pos)
    # node 1 (int.lt) reads node 1 itself
    g["fns"][0]["body"]["nodes"][1][2] = [[1, 0], [-1, 0]]
    nk = sha(g)
    m["defs"][nk] = g
    m["steps"][1][2] = [nk, 0]
    return m


# ---- admission vectors (FG15): cases whose outcomes follow from δ (§9, 6.4)

def vectors(m, cases):
    key = hashlib.sha256(canon(m).encode()).hexdigest()
    out = []
    for inputs, outputs, outcome, state in cases:
        case = {"inputs": [{"type": t, "payload": p} for t, p in inputs]}
        out.append({
            "inputs": case["inputs"],
            "expect": {"outputs": outputs, "outcome": outcome, "state": None if state is None else sha(state)},
            "source": "oracle:reference",
            "case": sha(case),
        })
    return {"format": "rce-vectors/1", "module": key, "observations": [], "cases": out}


# ---- the fixtures and their outcomes, from the rules as written

def fixtures():
    c = counter()
    t = types_module()
    yield "counter", c, None
    # Inc, Inc, Dec: n = 1; Dec at 0: the guard is false, no step matches, the event is applied unchanged (6.4)
    yield "counter.vectors", c, vectors(c, [
        ([(0, []), (0, []), (1, [])], [], "done", [[1]]),
        ([(1, [])], [], "done", [[0]]),
    ])
    yield "counter.wrong-vectors", c, vectors(c, [([(0, []), (0, []), (1, [])], [], "done", [[3]])])
    yield "types", t, None
    yield "types.vectors", t, vectors(t, [([(0, [2.5])], [1.5], "done", [[[0, "a"], 2.5]])])
    g = group_module()
    yield "group", g, None
    yield "group.vectors", g, vectors(g, [([(0, [])], [True], "done", [[4]])])
    yield "loops", loops_module(), None
    for v in ["c3", "c4", "c5", "c6", "c7", "c11", "fg1", "fg2", "fg4", "fg5", "fg6", "fg7", "fg8", "fg10", "fg11", "fg13"]:
        yield f"counter-{v}", counter(v), None
    yield "types-c8", types_module("c8"), None
    yield "types-fg12", types_module("fg12"), None
    yield "types-fg14", types_module("fg14"), None
    yield "group-c9", group_module("c9"), None
    yield "group-fg8", group_module("fg8-rec"), None
    yield "cycle", cycle_module(), None


# raw variants: bytes that are not the canonical encoding of a decodable module
def raw_fixtures(written):
    c = written["counter"]
    # C1: an integer literal that is not canonical decimal text
    yield "counter-c1", c.replace('["lit",["int","0","0"],"0"]', '["lit",["int","0","0"],"00"]', 1)
    # C10: a field that holds names
    yield "counter-c10", c[:-1] + ',"names":{"n":"count"}}'
    # FG3 by 6.1.1: the same module with a space after the first comma
    yield "counter-fg3", c.replace(",", ", ", 1)


# fixture -> (outcome, rule the outcome names, rule the reason names)
EXPECT = {
    "counter": ("applied", None, None),
    "counter.vectors": ("applied", None, None),
    "counter.wrong-vectors": ("requires change", "FG15", None),
    "types": ("applied", None, None),
    "types.vectors": ("applied", None, None),
    "group": ("applied", None, None),
    "group.vectors": ("applied", None, None),
    "loops": ("applied", None, None),
    "counter-c1": ("requires change", "FG1", "C1"),
    "counter-c10": ("requires change", "FG1", "C10"),
    "counter-c3": ("requires change", "FG3", "C3"),
    "counter-c4": ("requires change", "FG3", "C4"),
    "counter-c5": ("requires change", "FG3", "C5"),
    "counter-c6": ("requires change", "FG3", "C6"),
    "counter-c7": ("requires change", "FG3", "C7"),
    "types-c8": ("requires change", "FG3", "C8"),
    "group-c9": ("requires change", "FG3", "C9"),
    "counter-c11": ("requires change", "FG3", "C11"),
    "counter-fg1": ("requires change", "FG1", None),
    "counter-fg2": ("requires change", "FG2", None),
    "counter-fg3": ("requires change", "FG3", "6.1.1"),
    "counter-fg4": ("requires change", "FG4", None),
    "counter-fg5": ("requires change", "FG5", None),
    "counter-fg6": ("requires change", "FG6", None),
    "counter-fg7": ("requires change", "FG7", None),
    "counter-fg8": ("requires change", "FG8", None),
    "group-fg8": ("requires change", "FG8", None),
    "cycle": ("requires change", "FG4", None),
    "counter-fg10": ("requires change", "FG10", None),
    "counter-fg11": ("requires change", "FG11", None),
    "types-fg12": ("requires change", "FG12", None),
    "counter-fg13": ("requires change", "FG13", None),
    "types-fg14": ("requires change", "FG14", None),
}

# rule -> (a fixture the rule applies, a fixture the rule keeps); C2 is the
# digest the other rules use and FG9 is implied by FG4, so neither can keep a
# module on its own (README.md)
RULES = {
    "C1": ("counter", "counter-c1"),
    "C2": ("counter", None),
    "C3": ("counter", "counter-c3"),
    "C4": ("counter", "counter-c4"),
    "C5": ("counter", "counter-c5"),
    "C6": ("counter", "counter-c6"),
    "C7": ("counter", "counter-c7"),
    "C8": ("types", "types-c8"),
    "C9": ("group", "group-c9"),
    "C10": ("counter", "counter-c10"),
    "C11": ("counter", "counter-c11"),
    "FG1": ("counter", "counter-fg1"),
    "FG2": ("counter", "counter-fg2"),
    "FG3": ("counter", "counter-fg3"),
    "FG4": ("counter", "counter-fg4"),
    "FG5": ("counter", "counter-fg5"),
    "FG6": ("loops", "counter-fg6"),
    "FG7": ("counter", "counter-fg7"),
    "FG8": ("loops", "counter-fg8"),
    "FG9": ("loops", None),
    "FG10": ("types", "counter-fg10"),
    "FG11": ("types", "counter-fg11"),
    "FG12": ("types", "types-fg12"),
    "FG13": ("counter", "counter-fg13"),
    "FG14": ("types", "types-fg14"),
    "FG15": ("counter.vectors", "counter.wrong-vectors"),
}


def main():
    check = "--check" in sys.argv[1:]
    files = {}
    written = {}
    for name, m, vec in fixtures():
        base = name.split(".")[0]
        text = canon(m)
        if base not in written:
            written[base] = text
            files[base + ".json"] = text
        elif written[base] != text:
            raise SystemExit(f"{name}: two modules under one name")
        if vec is not None:
            assert "." in name, name
            files[name + ".json"] = canon(vec)
    for name, text in raw_fixtures(written):
        files[name + ".json"] = text
    entries = []
    for name, (outcome, rule, why) in sorted(EXPECT.items()):
        base, _, vec = name.partition(".")
        entries.append({"form": base + ".json", "vectors": name + ".json" if vec else None, "outcome": outcome, "rule": rule, "names": why})
    files["expect.json"] = canon({"format": "rce-form-fixtures/1", "fixtures": entries, "rules": {r: {"applied": a, "requires_change": k} for r, (a, k) in RULES.items()}})
    bad = 0
    for name, text in sorted(files.items()):
        p = os.path.join(HERE, name)
        if check:
            if not os.path.exists(p) or open(p, encoding="utf-8").read() != text:
                print(f"make.py: {name} differs from what make.py writes", file=sys.stderr)
                bad += 1
        else:
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(text)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
