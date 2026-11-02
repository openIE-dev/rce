#!/usr/bin/env python3
"""E5's IR-level R hash: a hash of a WebAssembly module modulo alpha-equivalence.

The declared IR is WebAssembly 1.0 in its binary format, with the
sign-extension, multi-value, bulk-memory and reference-type additions that
LLVM emits for wasm32 (PROTOCOL.md, "The IR-level hash"). A module binds
names of six kinds, all as indices: types, functions, tables, globals, data
and element segments, and the locals of each function. Labels are already
relative depths (de Bruijn indices) and stay as they are. The free names of a
module are its import and export names: they are its linking interface and
are hashed as text. Custom sections (the names section, producers, target
features, debug information) are left out.

The canonical form replaces every bound index by a position that depends
only on the module's structure:

1. *Types* are written out in place: a type index becomes the signature it
   names.
2. *Locals*: a parameter keeps its position (it is the calling convention);
   every other local is numbered by the order in which the body first
   refers to it, and its declaration follows that order. Locals declared and
   never used are written as a sorted count by type.
3. *Functions* are ordered by call-graph shape. Each function starts with a
   hash of its signature, its export names and its body with every callee
   left blank; each round replaces a function's hash by a hash of its
   previous hash and its body with every callee written as the callee's
   previous hash (colour refinement, as in Weisfeiler and Lehman). Rounds
   stop when a round splits no class. Imported functions are anchored by
   their import names. Defined functions are sorted by their final hash;
   functions with equal final hashes keep their relative order in the
   module, and the number of such ties is reported.
4. *Globals* are numbered by first reference, in the canonical function
   order; unreferenced globals follow, sorted by type, mutability and
   initial value. In the refinement rounds a global is written as its type,
   mutability and initial value.
5. *Data and element segments*: active segments are sorted by offset and
   contents; passive data segments are numbered by first reference.
6. *Exports* are sorted by name.

The R hash is SHA-256 of a byte encoding of the canonical form. Two modules
with the same R hash have the same canonical form (barring a SHA-256
collision), so one is the other with its bound indices permuted and its
custom sections changed: the same module up to renaming.

    python3 irhash.py MODULE.wasm [EXPORT]
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e4-energy"))

import wasm_eps  # noqa: E402  (its Reader, LEB128 decoding and MEMOPS table)

Reader = wasm_eps.Reader
MEMOPS = wasm_eps.MEMOPS
Unsupported = wasm_eps.Unsupported
SCHEMA = "e5-ir-alpha/1"
VALTYPES = (0x7F, 0x7E, 0x7D, 0x7C, 0x7B, 0x70, 0x6F)

# immediate kinds of each opcode that carries a bound index
LOCAL_OPS = (0x20, 0x21, 0x22)
GLOBAL_OPS = (0x23, 0x24)
FUNC_OPS = (0x10, 0x12, 0xD2)          # call, return_call, ref.func
DATA_OPS = (0xFC08, 0xFC09)            # memory.init, data.drop
ELEM_OPS = (0xFC0C, 0xFC0D)            # table.init, elem.drop
BLOCK_OPS = (0x02, 0x03, 0x04)


# -- byte encoding of a canonical form ---------------------------------------------

def enc(x, out=None):
    """A prefix-free byte encoding of nested tuples and lists of ints, str, bytes and None."""
    top = out is None
    if top:
        out = bytearray()
    if x is None:
        out += b"n"
    elif isinstance(x, bool):
        out += b"t" if x else b"f"
    elif isinstance(x, int):
        s = str(x).encode()
        out += b"i%d:" % len(s) + s
    elif isinstance(x, str):
        s = x.encode("utf-8")
        out += b"s%d:" % len(s) + s
    elif isinstance(x, (bytes, bytearray)):
        out += b"b%d:" % len(x) + bytes(x)
    elif isinstance(x, (tuple, list)):
        out += b"(%d:" % len(x)
        for y in x:
            enc(y, out)
        out += b")"
    else:
        raise TypeError("cannot encode %r" % type(x))
    return bytes(out) if top else None


def sha(x):
    return hashlib.sha256(enc(x)).hexdigest()


# -- decoding -------------------------------------------------------------------------

class Instr:
    """One instruction: opcode (0xFCnn for the 0xFC prefix), immediates, and its byte range in the module."""
    __slots__ = ("op", "imm", "start", "end")

    def __init__(self, op, imm, start, end):
        self.op, self.imm, self.start, self.end = op, imm, start, end

    def __repr__(self):
        return "Instr(0x%x, %r)" % (self.op, self.imm)


def _blocktype(s):
    b = s.d[s.p]
    if b == 0x40:
        s.p += 1
        return ("empty",)
    if b in VALTYPES:
        s.p += 1
        return ("val", b)
    return ("type", s.s(33))


def decode_expr(s, until_end=True):
    """Decode instructions from s until the end of the reader (a body) or a final `end` (an expression)."""
    code, depth = [], 0
    while s.p < s.end:
        start = s.p
        op = s.byte()
        imm = ()
        if op in BLOCK_OPS:
            imm = (_blocktype(s),)
            depth += 1
        elif op == 0x0B:
            depth -= 1
        elif op in (0x0C, 0x0D) or op in LOCAL_OPS or op in GLOBAL_OPS or op in FUNC_OPS or op in (0x25, 0x26):
            imm = (s.u(),)
        elif op == 0x0E:
            imm = (tuple(s.u() for _ in range(s.u())), s.u())
        elif op in (0x11, 0x13):
            imm = (s.u(), s.u())
        elif op == 0x1C:
            imm = (tuple(s.bytes(s.u())),)
        elif op in MEMOPS:
            imm = (s.u(), s.u())
        elif op in (0x3F, 0x40):
            imm = (s.byte(),)
        elif op == 0x41:
            imm = (s.s(32) & wasm_eps.M32,)
        elif op == 0x42:
            imm = (s.s(64) & wasm_eps.M64,)
        elif op == 0x43:
            imm = (bytes(s.bytes(4)),)
        elif op == 0x44:
            imm = (bytes(s.bytes(8)),)
        elif op == 0xD0:
            imm = (s.byte(),)
        elif op == 0xFC:
            sub = s.u()
            op = 0xFC00 | sub
            if sub <= 7:
                imm = ()
            elif sub == 8:
                imm = (s.u(), s.byte())
            elif sub in (9, 13, 15, 16, 17):
                imm = (s.u(),)
            elif sub == 10:
                imm = (s.byte(), s.byte())
            elif sub == 11:
                imm = (s.byte(),)
            elif sub in (12, 14):
                imm = (s.u(), s.u())
            else:
                raise Unsupported("instruction 0xfc %d" % sub)
        elif op == 0xFD:
            raise Unsupported("SIMD (0xfd prefix)")
        elif op in (0x06, 0x07, 0x08, 0x09, 0x18, 0x19, 0x1F):
            raise Unsupported("exception-handling instruction 0x%02x" % op)
        elif op in (0x00, 0x01, 0x05, 0x0F, 0x1A, 0x1B, 0xD1) or 0x45 <= op <= 0xC4:
            imm = ()
        else:
            raise Unsupported("opcode 0x%02x" % op)
        code.append(Instr(op, imm, start, s.p))
        if op == 0x0B and depth < 0 and not until_end:
            return code
    return code


def _limits(s):
    flag = s.u()
    mn = s.u()
    mx = s.u() if flag & 1 else None
    return (flag & ~1, mn, mx)


class Module:
    """A WebAssembly module parsed with every immediate kept, for hashing."""

    def __init__(self, data):
        data = bytes(data)
        if data[:8] != b"\0asm\x01\0\0\0":
            raise Unsupported("not a WebAssembly 1.0 binary module")
        self.data = data
        self.types, self.imports, self.ftypes, self.tables, self.mems = [], [], [], [], []
        self.globals, self.exports, self.start, self.elems, self.datas = [], [], None, [], []
        self.bodies, self.customs = [], []
        self.code_start = None  # offset of the code section's payload
        self.sections = []      # (id, payload start, payload end)
        r = Reader(data, 8)
        while r.p < len(data):
            sid = r.byte()
            size = r.u()
            end = r.p + size
            self.sections.append((sid, r.p, end))
            s = Reader(data, r.p, end)
            if sid == 0:
                self.customs.append((s.name(), s.p, end))
            elif sid == 1:
                for _ in range(s.u()):
                    if s.byte() != 0x60:
                        raise Unsupported("type that is not a function type")
                    self.types.append((tuple(s.bytes(s.u())), tuple(s.bytes(s.u()))))
            elif sid == 2:
                for _ in range(s.u()):
                    mod, nm, kind = s.name(), s.name(), s.byte()
                    if kind == 0:
                        desc = s.u()
                    elif kind == 1:
                        desc = (s.byte(), _limits(s))
                    elif kind == 2:
                        desc = _limits(s)
                    elif kind == 3:
                        desc = (s.byte(), s.byte())
                    else:
                        raise Unsupported("import kind %d" % kind)
                    self.imports.append((mod, nm, kind, desc))
            elif sid == 3:
                self.ftypes = [s.u() for _ in range(s.u())]
            elif sid == 4:
                for _ in range(s.u()):
                    self.tables.append((s.byte(), _limits(s)))
            elif sid == 5:
                for _ in range(s.u()):
                    self.mems.append(_limits(s))
            elif sid == 6:
                for _ in range(s.u()):
                    t, mut = s.byte(), s.byte()
                    self.globals.append((t, mut, decode_expr(s, until_end=False)))
            elif sid == 7:
                for _ in range(s.u()):
                    nm = s.name()
                    self.exports.append((nm, s.byte(), s.u()))
            elif sid == 8:
                self.start = s.u()
            elif sid == 9:
                for _ in range(s.u()):
                    self.elems.append(self._elem(s))
            elif sid == 10:
                self.code_start = s.p
                n = s.u()
                for _ in range(n):
                    fsize = s.u()
                    fstart = s.p
                    fend = s.p + fsize
                    b = Reader(data, s.p, fend)
                    locals_ = []
                    for _ in range(b.u()):
                        cnt = b.u()
                        locals_.append((cnt, b.byte()))
                    body_start = b.p
                    self.bodies.append({"start": fstart, "end": fend, "body_start": body_start,
                                        "locals": locals_, "code": decode_expr(b)})
                    s.p = fend
            elif sid == 11:
                for _ in range(s.u()):
                    flags = s.u()
                    if flags == 1:
                        self.datas.append(("passive", None, None, bytes(s.bytes(s.u()))))
                        continue
                    mem = s.u() if flags == 2 else 0
                    off = decode_expr(s, until_end=False)
                    self.datas.append(("active", mem, off, bytes(s.bytes(s.u()))))
            elif sid == 12:
                s.u()
            r.p = end
        if len(self.ftypes) != len(self.bodies):
            raise Unsupported("function and code sections disagree")

    @staticmethod
    def _elem(s):
        flags = s.u()
        if flags == 0:
            off = decode_expr(s, until_end=False)
            return ("active", 0, off, ("funcs", tuple(s.u() for _ in range(s.u()))))
        if flags == 2:
            tab = s.u()
            off = decode_expr(s, until_end=False)
            s.byte()
            return ("active", tab, off, ("funcs", tuple(s.u() for _ in range(s.u()))))
        if flags in (1, 3):
            s.byte()
            return ("passive" if flags == 1 else "declarative", None, None, ("funcs", tuple(s.u() for _ in range(s.u()))))
        raise Unsupported("element segment kind %d" % flags)

    # index spaces
    def imported(self, kind):
        return [i for i in self.imports if i[2] == kind]

    def n_imported_funcs(self):
        return len(self.imported(0))

    def func_sig(self, fi):
        nf = self.n_imported_funcs()
        t = self.imported(0)[fi][3] if fi < nf else self.ftypes[fi - nf]
        return self.types[t]

    def ir_nodes(self):
        """Instructions in all bodies without each body's closing end (E4's `ir_nodes`)."""
        return sum(len(b["code"]) - 1 for b in self.bodies)


# -- canonical form ----------------------------------------------------------------------

def _blk(m, bt):
    if bt[0] == "type":
        return ("sig",) + tuple(m.types[bt[1]])
    return bt


def _global_desc(m, gi, depth=0):
    """A global's type, mutability and initial value, for the refinement rounds."""
    ng = len(m.imported(3))
    if gi < ng:
        mod, nm, _k, desc = m.imported(3)[gi]
        return ("gimport", mod, nm) + tuple(desc)
    t, mut, init = m.globals[gi - ng]
    return ("g", t, mut, tuple(_const_expr(m, init, lambda g: _global_desc(m, g, depth + 1) if depth < 8 else "?",
                                           lambda f: "?")))


def _const_expr(m, code, gref, fref):
    out = []
    for ins in code:
        if ins.op in GLOBAL_OPS:
            out.append((ins.op, gref(ins.imm[0])))
        elif ins.op in FUNC_OPS:
            out.append((ins.op, fref(ins.imm[0])))
        else:
            out.append((ins.op,) + tuple(ins.imm))
    return out


def _body(m, fi, fref, gref, dref, eref):
    """The canonical body of defined function fi: (signature, locals, used-local types, instructions)."""
    nf = m.n_imported_funcs()
    b = m.bodies[fi - nf]
    params, results = m.func_sig(fi)
    declared = []
    for cnt, t in b["locals"]:
        declared.extend([t] * cnt)
    np = len(params)
    renum, order = {}, []
    code = []
    for ins in b["code"]:
        op, imm = ins.op, ins.imm
        if op in LOCAL_OPS:
            x = imm[0]
            if x >= np:
                if x not in renum:
                    renum[x] = np + len(order)
                    order.append(x)
                x = renum[x]
            code.append((op, x))
        elif op in GLOBAL_OPS:
            code.append((op, gref(imm[0])))
        elif op in FUNC_OPS:
            code.append((op, fref(imm[0])))
        elif op in BLOCK_OPS:
            code.append((op, _blk(m, imm[0])))
        elif op in (0x11, 0x13):
            code.append((op, ("sig",) + tuple(m.types[imm[0]]), imm[1]))
        elif op in DATA_OPS:
            code.append((op, dref(imm[0])) + tuple(imm[1:]))
        elif op in ELEM_OPS:
            code.append((op, eref(imm[0])) + tuple(imm[1:]))
        else:
            code.append((op,) + tuple(imm))
    used = tuple(declared[x - np] for x in order)
    unused = sorted(declared[k] for k in range(len(declared)) if k + np not in renum)
    return (("sig", params, results), used, tuple(unused), tuple(code))


def _refs(m, fi):
    """Functions, globals and data segments that defined function fi refers to, in body order."""
    nf = m.n_imported_funcs()
    fs, gs, ds, ind = [], [], [], False
    for ins in m.bodies[fi - nf]["code"]:
        if ins.op in FUNC_OPS:
            fs.append(ins.imm[0])
        elif ins.op in GLOBAL_OPS:
            gs.append(ins.imm[0])
        elif ins.op in DATA_OPS:
            ds.append(ins.imm[0])
        elif ins.op in (0x11, 0x13):
            ind = True
    return fs, gs, ds, ind


def reachable(m, roots):
    """Defined and imported functions reachable from the function indices roots by calls, ref.func and tables."""
    seen, todo = set(), list(roots)
    table_funcs = [f for e in m.elems for f in e[3][1]]
    nf = m.n_imported_funcs()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        if f < nf:
            continue
        fs, _gs, _ds, ind = _refs(m, f)
        todo.extend(fs)
        if ind:
            todo.extend(table_funcs)
    return seen


def canonical(m, entry=None):
    """The canonical form of module m and facts about it; with entry, only what export `entry` reaches.

    Returns (form, info) where info holds the refinement rounds and the number of tied functions.
    """
    nf = m.n_imported_funcs()
    nfun = nf + len(m.bodies)
    exports = list(m.exports)
    if entry is not None:
        exports = [e for e in exports if e[0] == entry and e[1] == 0]
        if not exports:
            raise Unsupported("no exported function %r" % entry)
        keep = reachable(m, [exports[0][2]])
    else:
        keep = set(range(nfun))
    funcs = sorted(keep)
    exp_names = {}
    for nm, kind, idx in exports:
        if kind == 0:
            exp_names.setdefault(idx, []).append(nm)
    imp_funcs = m.imported(0)

    # refinement rounds
    blank = lambda f: "?"  # noqa: E731
    gdesc = lambda g: _global_desc(m, g)  # noqa: E731
    dblank = lambda d: ("d", m.datas[d][0], m.datas[d][3])  # noqa: E731
    eblank = lambda e: "e"  # noqa: E731
    h = {}
    for f in funcs:
        if f < nf:
            mod, nm, _k, t = imp_funcs[f]
            h[f] = sha(("import", mod, nm, m.types[t]))
        else:
            h[f] = sha(("f", tuple(sorted(exp_names.get(f, []))), _body(m, f, blank, gdesc, dblank, eblank)))
    classes = len(set(h.values()))
    rounds = 0
    while True:
        rounds += 1
        nh = {}
        for f in funcs:
            if f < nf:
                nh[f] = h[f]
            else:
                nh[f] = sha((h[f], _body(m, f, lambda g, h=h: h.get(g, "?"), gdesc, dblank, eblank)))
        nclasses = len(set(nh.values()))
        h = nh
        if nclasses == classes or rounds > nfun:
            break
        classes = nclasses

    # canonical positions: imports by name, defined functions by final hash
    imp_order = sorted((f for f in funcs if f < nf), key=lambda f: imp_funcs[f][:2])
    def_order = sorted((f for f in funcs if f >= nf), key=lambda f: (h[f], f))
    order = imp_order + def_order
    fpos = {f: k for k, f in enumerate(order)}
    hs = [h[f] for f in def_order]
    ties = sum(1 for k in range(len(hs)) if (k > 0 and hs[k - 1] == hs[k]) or (k + 1 < len(hs) and hs[k + 1] == hs[k]))

    # globals by first reference in canonical order
    ng = len(m.imported(3))
    gorder, dorder = [], []
    for f in def_order:
        _fs, gs, ds, _ind = _refs(m, f)
        for g in gs:
            if g not in gorder:
                gorder.append(g)
        for d in ds:
            if d not in dorder:
                dorder.append(d)
    for nm, kind, idx in sorted(exports):
        if kind == 3 and idx not in gorder:
            gorder.append(idx)
    if entry is None:
        rest = [g for g in range(ng + len(m.globals)) if g not in gorder]
        rest.sort(key=lambda g: enc(_global_desc(m, g)))
        gorder += rest
    # globals referred to by other globals' initial values
    k = 0
    while k < len(gorder):
        g = gorder[k]
        if g >= ng:
            for ins in m.globals[g - ng][2]:
                if ins.op in GLOBAL_OPS and ins.imm[0] not in gorder:
                    gorder.append(ins.imm[0])
        k += 1
    gpos = {g: k for k, g in enumerate(gorder)}

    # data segments: passive by first reference, then active by offset and bytes
    def seg_key(d):
        kind, mem, off, b = m.datas[d]
        return enc((kind, mem, _const_expr(m, off, lambda g: "g", lambda f: "f") if off else None, b))
    passive = [d for d in dorder if m.datas[d][0] == "passive"]
    others = sorted((d for d in range(len(m.datas)) if d not in passive), key=seg_key)
    dpos = {d: k for k, d in enumerate(passive + others)}
    epos = {e: e for e in range(len(m.elems))}

    fref = lambda f: ("F", fpos[f])  # noqa: E731
    gref = lambda g: ("G", gpos[g])  # noqa: E731
    dref = lambda d: ("D", dpos[d])  # noqa: E731
    eref = lambda e: ("E", epos[e])  # noqa: E731

    form_imports = []
    for f in imp_order:
        mod, nm, _k, t = imp_funcs[f]
        form_imports.append(("func", mod, nm, m.types[t]))
    for mod, nm, kind, desc in sorted(i for i in m.imports if i[2] != 0):
        form_imports.append((kind, mod, nm, desc))
    form_globals = []
    for g in gorder:
        if g < ng:
            form_globals.append(("import",) + m.imported(3)[g][:2])
        else:
            t, mut, init = m.globals[g - ng]
            form_globals.append((t, mut, tuple(_const_expr(m, init, gref, lambda f: fref(f) if f in fpos else "?"))))
    form_funcs = [_body(m, f, fref, gref, dref, eref) for f in def_order]
    form_elems = []
    for kind, tab, off, (_x, fs) in m.elems:
        form_elems.append((kind, tab, tuple(_const_expr(m, off, gref, fref)) if off else None,
                           tuple(fref(f) if f in fpos else "?" for f in fs)))
    form_elems.sort(key=enc)
    form_datas = []
    for d in passive + others:
        kind, mem, off, b = m.datas[d]
        form_datas.append((kind, mem, tuple(_const_expr(m, off, gref, fref)) if off else None, b))
    form_exports = []
    for nm, kind, idx in sorted(exports):
        ref = fpos.get(idx) if kind == 0 else gpos.get(idx) if kind == 3 else idx
        form_exports.append((nm, kind, ref))
    form = (SCHEMA, "entry" if entry is not None else "module", tuple(form_imports), tuple(m.tables), tuple(m.mems),
            tuple(form_globals), tuple(form_funcs), tuple(form_elems), tuple(form_datas), tuple(form_exports),
            fref(m.start) if m.start is not None and m.start in fpos else None)
    return form, {"rounds": rounds, "tied_functions": ties, "functions": len(def_order)}


def r_hash(m, entry=None):
    form, info = canonical(m, entry)
    return sha(form), info


# -- mutations: the property that any change of an opcode or a constant changes the hash ----------

# classes of opcodes that take and give the same types, so swapping within one keeps a module valid
OP_CLASSES = [
    list(range(0x46, 0x50)),   # i32 comparisons
    list(range(0x51, 0x5B)),   # i64 comparisons
    list(range(0x67, 0x6A)),   # i32 clz, ctz, popcnt
    list(range(0x6A, 0x79)),   # i32 binary operators
    list(range(0x79, 0x7C)),   # i64 clz, ctz, popcnt
    list(range(0x7C, 0x8B)),   # i64 binary operators
    [0x28, 0x2C, 0x2D, 0x2E, 0x2F],   # i32 loads
    [0x29, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35],  # i64 loads
    [0x36, 0x3A, 0x3B],        # i32 stores
    [0x37, 0x3C, 0x3D, 0x3E],  # i64 stores
]
CLASS_OF = {op: c for c in OP_CLASSES for op in c}


def mutations(m):
    """Every single change of one opcode within its class or of one constant: [(function, instruction, kind)]."""
    out = []
    for bi, b in enumerate(m.bodies):
        for ii, ins in enumerate(b["code"]):
            if ins.op in CLASS_OF:
                out.append((bi, ii, "opcode"))
            if ins.op in (0x41, 0x42, 0x43, 0x44):
                out.append((bi, ii, "constant"))
            if ins.op in MEMOPS:
                out.append((bi, ii, "offset"))
    return out


def apply_mutation(m, mut):
    """Change module m in place by one mutation; return a function that undoes it."""
    bi, ii, kind = mut
    code = m.bodies[bi]["code"]
    old = code[ii]
    if kind == "opcode":
        c = CLASS_OF[old.op]
        new = Instr(c[(c.index(old.op) + 1) % len(c)], old.imm, old.start, old.end)
    elif kind == "constant":
        if old.op == 0x41:
            v = ((old.imm[0] + 1) & wasm_eps.M32,)
        elif old.op == 0x42:
            v = ((old.imm[0] + 1) & wasm_eps.M64,)
        else:
            b = bytearray(old.imm[0])
            b[0] ^= 1
            v = (bytes(b),)
        new = Instr(old.op, v, old.start, old.end)
    else:
        new = Instr(old.op, (old.imm[0], old.imm[1] + 1), old.start, old.end)
    code[ii] = new

    def undo():
        code[ii] = old
    return undo


def main():
    a = sys.argv[1:]
    if len(a) not in (1, 2):
        sys.exit("usage: python3 irhash.py MODULE.wasm [EXPORT]")
    m = Module(open(a[0], "rb").read())
    h, info = r_hash(m, a[1] if len(a) == 2 else None)
    print(h, info)


if __name__ == "__main__":
    main()
