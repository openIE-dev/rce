#!/usr/bin/env python3
"""E4's pinned instrumentation for ε: a WebAssembly interpreter that counts erased bits.

The interpreter runs one exported function of a WebAssembly module (the
kernel built for wasm32 under the same optimisation configuration as the
native program) and counts ε, the bits a run erases, by this definition
(PROTOCOL.md, "The ε count"):

- A *location* is an operand-stack slot, a local of a live call frame, a
  global, or a byte of linear memory. Every value is held as a sequence of
  bytes (4 for i32, 8 for i64), and every byte carries an *instance*: the
  identity of the byte value that the instruction which made it produced.
- An instruction that moves or copies a value (local.get, local.set,
  local.tee, global.get, global.set, the loads and stores, select,
  i32.wrap_i64, i64.extend_i32_u, memory.copy, memory.fill, a call's
  arguments and results) gives its destination the same instances. Any other
  instruction gives its result fresh instances.
- Bytes whose value the program text fixes are *permanent* and never
  counted: constants, the zero bytes of zero extension and of comparison
  results, the initial memory (data segments and zeros), initial globals,
  zero-initialised locals, and the kernel's argument, which its caller
  keeps.
- A byte instance is *copied out* while another location still holds it.
  When a location is overwritten or released (a stack slot popped by an
  instruction that does not move it, a frame's locals and leftover stack
  slots at return, the values a branch discards) and it held the last copy
  of an instance, the instance's 8 bits are erased and ε grows by 8.
- The value the kernel returns is copied out to the caller. Locations still
  live when the run ends (memory, globals) are not counted.

Sign-extension bytes are fresh, so ε is an upper bound on the information a
run discards at byte granularity, not a minimum.

Supported: the integer instructions of WebAssembly 1.0, sign-extension
operators, multi-value block types, memory.copy and memory.fill. Floating
point, SIMD, imports and table instructions other than call_indirect are
not: a module that uses one gives an Unsupported error naming it, and the
harness records that ε was not counted.

    python3 wasm_eps.py MODULE.wasm EXPORT N
"""
import sys

PAGE = 65536
M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
Z4 = (0, 0, 0, 0)
Z8 = (0,) * 8
I32, I64, F32, F64 = 0x7F, 0x7E, 0x7D, 0x7C
WIDTH = {I32: 4, I64: 8, F32: 4, F64: 8}


class Unsupported(Exception):
    pass


class Trap(Exception):
    pass


class Reader:
    def __init__(self, data, pos=0, end=None):
        self.d, self.p = data, pos
        self.end = len(data) if end is None else end

    def byte(self):
        if self.p >= self.end:
            raise Unsupported("truncated module")
        b = self.d[self.p]
        self.p += 1
        return b

    def u(self):
        r = s = 0
        while True:
            b = self.byte()
            r |= (b & 0x7F) << s
            s += 7
            if not b & 0x80:
                return r

    def s(self, bits):
        r = s = 0
        while True:
            b = self.byte()
            r |= (b & 0x7F) << s
            s += 7
            if not b & 0x80:
                break
        if b & 0x40:
            r -= 1 << s
        return r

    def bytes(self, n):
        if self.p + n > self.end:
            raise Unsupported("truncated module")
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def name(self):
        return self.bytes(self.u()).decode("utf-8")


# memory instructions: opcode -> (width in bytes, value type, extension: "u", "s" or None, is_store)
MEMOPS = {
    0x28: (4, I32, None, False), 0x29: (8, I64, None, False), 0x2A: (4, F32, None, False), 0x2B: (8, F64, None, False),
    0x2C: (1, I32, "s", False), 0x2D: (1, I32, "u", False), 0x2E: (2, I32, "s", False), 0x2F: (2, I32, "u", False),
    0x30: (1, I64, "s", False), 0x31: (1, I64, "u", False), 0x32: (2, I64, "s", False), 0x33: (2, I64, "u", False),
    0x34: (4, I64, "s", False), 0x35: (4, I64, "u", False),
    0x36: (4, I32, None, True), 0x37: (8, I64, None, True), 0x38: (4, F32, None, True), 0x39: (8, F64, None, True),
    0x3A: (1, I32, None, True), 0x3B: (2, I32, None, True), 0x3C: (1, I64, None, True), 0x3D: (2, I64, None, True),
    0x3E: (4, I64, None, True),
}


def _sx(v, bits):
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


def _div_s(a, b, bits):
    if b == 0:
        raise Trap("integer divide by zero")
    a, b = _sx(a, bits), _sx(b, bits)
    if a == -(1 << (bits - 1)) and b == -1:
        raise Trap("integer overflow")
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _rem_s(a, b, bits):
    if b == 0:
        raise Trap("integer divide by zero")
    a, b = _sx(a, bits), _sx(b, bits)
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def _clz(a, bits):
    return bits - a.bit_length()


def _ctz(a, bits):
    return bits if a == 0 else (a & -a).bit_length() - 1


def _rotl(a, b, bits):
    b %= bits
    m = (1 << bits) - 1
    return ((a << b) | (a >> (bits - b))) & m


def _binops(bits):
    m = (1 << bits) - 1
    return [
        lambda a, b: (a + b) & m,
        lambda a, b: (a - b) & m,
        lambda a, b: (a * b) & m,
        lambda a, b: _div_s(a, b, bits) & m,
        lambda a, b: a // b if b else _trap("integer divide by zero"),
        lambda a, b: _rem_s(a, b, bits) & m,
        lambda a, b: a % b if b else _trap("integer divide by zero"),
        lambda a, b: a & b,
        lambda a, b: a | b,
        lambda a, b: a ^ b,
        lambda a, b: (a << (b % bits)) & m,
        lambda a, b: (_sx(a, bits) >> (b % bits)) & m,
        lambda a, b: a >> (b % bits),
        lambda a, b: _rotl(a, b, bits),
        lambda a, b: _rotl(a, bits - (b % bits), bits),
    ]


def _cmpops(bits):
    return [
        lambda a, b: a == b,
        lambda a, b: a != b,
        lambda a, b: _sx(a, bits) < _sx(b, bits),
        lambda a, b: a < b,
        lambda a, b: _sx(a, bits) > _sx(b, bits),
        lambda a, b: a > b,
        lambda a, b: _sx(a, bits) <= _sx(b, bits),
        lambda a, b: a <= b,
        lambda a, b: _sx(a, bits) >= _sx(b, bits),
        lambda a, b: a >= b,
    ]


def _trap(msg):
    raise Trap(msg)


# opcode -> ("bin", width, fn) | ("cmp", width, fn) | ("un", width in, width out, fn)
NUMERIC = {}
for _i, _f in enumerate(_cmpops(32)):
    NUMERIC[0x46 + _i] = ("cmp", 4, _f)
for _i, _f in enumerate(_cmpops(64)):
    NUMERIC[0x51 + _i] = ("cmp", 8, _f)
for _i, _f in enumerate(_binops(32)):
    NUMERIC[0x6A + _i] = ("bin", 4, _f)
for _i, _f in enumerate(_binops(64)):
    NUMERIC[0x7C + _i] = ("bin", 8, _f)
NUMERIC[0x45] = ("test", 4, lambda a: a == 0)
NUMERIC[0x50] = ("test", 8, lambda a: a == 0)
NUMERIC[0x67] = ("un", 4, 4, lambda a: _clz(a, 32))
NUMERIC[0x68] = ("un", 4, 4, lambda a: _ctz(a, 32))
NUMERIC[0x69] = ("un", 4, 4, lambda a: bin(a).count("1"))
NUMERIC[0x79] = ("un", 8, 8, lambda a: _clz(a, 64))
NUMERIC[0x7A] = ("un", 8, 8, lambda a: _ctz(a, 64))
NUMERIC[0x7B] = ("un", 8, 8, lambda a: bin(a).count("1"))
# conversions that keep some input bytes: opcode -> (width in, width out, kept low bytes, fill: "zero" | "fresh", fn)
CONVERT = {
    0xA7: (8, 4, 4, None, lambda a: a & M32),                      # i32.wrap_i64
    0xAC: (4, 8, 4, "fresh", lambda a: _sx(a, 32) & M64),          # i64.extend_i32_s
    0xAD: (4, 8, 4, "zero", lambda a: a),                          # i64.extend_i32_u
    0xC0: (4, 4, 1, "fresh", lambda a: _sx(a & 0xFF, 8) & M32),    # i32.extend8_s
    0xC1: (4, 4, 2, "fresh", lambda a: _sx(a & 0xFFFF, 16) & M32),  # i32.extend16_s
    0xC2: (8, 8, 1, "fresh", lambda a: _sx(a & 0xFF, 8) & M64),    # i64.extend8_s
    0xC3: (8, 8, 2, "fresh", lambda a: _sx(a & 0xFFFF, 16) & M64),  # i64.extend16_s
    0xC4: (8, 8, 4, "fresh", lambda a: _sx(a & M32, 32) & M64),    # i64.extend32_s
}

# control opcodes and their names, for the unsupported-instruction message
NAMES = {0x43: "f32.const", 0x44: "f64.const", 0xFD: "SIMD (0xfd prefix)", 0x25: "table.get", 0x26: "table.set",
         0x1C: "select with type", 0xD0: "ref.null", 0xD1: "ref.is_null", 0xD2: "ref.func"}


class Func:
    __slots__ = ("type", "locals", "code", "ends", "elses")


class Module:
    def __init__(self, data):
        if data[:8] != b"\0asm\x01\0\0\0":
            raise Unsupported("not a WebAssembly 1.0 binary module")
        self.types, self.funcs, self.imports, self.exports = [], [], [], {}
        self.globals, self.mem_min, self.table, self.data = [], 0, [], []
        self.elems, self.start = [], None
        self.code_bytes = 0
        r = Reader(data, 8)
        ftypes = []
        while r.p < len(data):
            sid = r.byte()
            size = r.u()
            end = r.p + size
            s = Reader(data, r.p, end)
            if sid == 1:
                for _ in range(s.u()):
                    if s.byte() != 0x60:
                        raise Unsupported("type that is not a function type")
                    params = list(s.bytes(s.u()))
                    results = list(s.bytes(s.u()))
                    self.types.append((params, results))
            elif sid == 2:
                for _ in range(s.u()):
                    self.imports.append((s.name(), s.name()))
                    kind = s.byte()
                    if kind == 0:
                        s.u()
                    elif kind == 1:
                        s.byte()
                        self._limits(s)
                    elif kind == 2:
                        self._limits(s)
                    else:
                        s.bytes(2)
            elif sid == 3:
                ftypes = [s.u() for _ in range(s.u())]
            elif sid == 4:
                for _ in range(s.u()):
                    s.byte()
                    self.table = [None] * self._limits(s)[0]
            elif sid == 5:
                n = s.u()
                if n:
                    self.mem_min = self._limits(s)[0]
            elif sid == 6:
                for _ in range(s.u()):
                    t = s.byte()
                    mut = s.byte()
                    self.globals.append((t, mut, self._const(s)))
            elif sid == 7:
                for _ in range(s.u()):
                    nm = s.name()
                    self.exports[nm] = (s.byte(), s.u())
            elif sid == 8:
                self.start = s.u()
            elif sid == 9:
                for _ in range(s.u()):
                    flags = s.u()
                    if flags != 0:
                        raise Unsupported("element segment kind %d" % flags)
                    off = self._const(s)
                    self.elems.append((off, [s.u() for _ in range(s.u())]))
            elif sid == 10:
                self.code_bytes = size
                n = s.u()
                for i in range(n):
                    fsize = s.u()
                    fend = s.p + fsize
                    f = Func()
                    f.type = ftypes[i]
                    f.locals = []
                    for _ in range(s.u()):
                        cnt = s.u()
                        f.locals.extend([s.byte()] * cnt)
                    f.code, f.ends, f.elses = self._decode(Reader(data, s.p, fend))
                    s.p = fend
                    self.funcs.append(f)
            elif sid == 11:
                for _ in range(s.u()):
                    flags = s.u()
                    if flags == 1:
                        s.bytes(s.u())
                        continue
                    if flags == 2:
                        s.u()
                    off = self._const(s)
                    self.data.append((off, s.bytes(s.u())))
            r.p = end
        if self.imports:
            raise Unsupported("imports (%s)" % ", ".join("%s.%s" % i for i in self.imports))

    @staticmethod
    def _limits(s):
        flag = s.u()
        mn = s.u()
        mx = s.u() if flag & 1 else None
        return mn, mx

    @staticmethod
    def _const(s):
        op = s.byte()
        if op == 0x41:
            v = s.s(32) & M32
        elif op == 0x42:
            v = s.s(64) & M64
        else:
            raise Unsupported("constant expression opcode 0x%02x" % op)
        if s.byte() != 0x0B:
            raise Unsupported("constant expression longer than one instruction")
        return v

    def _blocktype(self, s):
        b = s.d[s.p]
        if b == 0x40:
            s.p += 1
            return 0, 0
        if b in WIDTH:
            s.p += 1
            return 0, 1
        params, results = self.types[s.s(33)]
        return len(params), len(results)

    def _decode(self, s):
        """Decode a body into (code, ends, elses); code is a list of (op, a, b)."""
        code, ends, elses, open_ = [], {}, {}, []
        while s.p < s.end:
            op = s.byte()
            a = b = None
            if op in (0x02, 0x03, 0x04):
                a = self._blocktype(s)
                open_.append(len(code))
            elif op == 0x05:
                elses[open_[-1]] = len(code)
            elif op == 0x0B:
                if open_:
                    ends[open_.pop()] = len(code)
            elif op in (0x0C, 0x0D, 0x10, 0x20, 0x21, 0x22, 0x23, 0x24):
                a = s.u()
            elif op == 0x0E:
                a = [s.u() for _ in range(s.u())]
                b = s.u()
            elif op == 0x11:
                a = s.u()
                b = s.u()
            elif op in MEMOPS:
                s.u()
                a = s.u()
            elif op in (0x3F, 0x40):
                s.byte()
            elif op == 0x41:
                a = s.s(32) & M32
            elif op == 0x42:
                a = s.s(64) & M64
            elif op == 0x43:
                s.bytes(4)
            elif op == 0x44:
                s.bytes(8)
            elif op == 0x1C:
                s.bytes(s.u())
            elif op == 0xFC:
                a = s.u()
                if a == 8:
                    s.u()
                    s.byte()
                elif a in (9, 13, 15, 16, 17):
                    s.u()
                elif a == 10:
                    s.bytes(2)
                elif a == 11:
                    s.byte()
                elif a in (12, 14):
                    s.u()
                    s.u()
                op = 0xFC00 | a
            elif op == 0xFD:
                raise Unsupported(NAMES[0xFD])
            elif op in (0x25, 0x26, 0xD0, 0xD2):
                s.u()
            code.append((op, a, b))
        return code, ends, elses

    def ir_nodes(self):
        """Instructions in all function bodies, not counting the closing end of each body."""
        return sum(len(f.code) - 1 for f in self.funcs)


class Run:
    """One instance of a module and the ε count of the calls made on it."""

    def __init__(self, module):
        self.m = module
        self.mem = bytearray(module.mem_min * PAGE)
        self.mids = [0] * len(self.mem)
        for off, b in module.data:
            if off + len(b) > len(self.mem):
                raise Trap("data segment out of bounds")
            self.mem[off:off + len(b)] = b
        self.gv = [g[2] for g in module.globals]
        self.gi = [Z8 if g[0] in (I64, F64) else Z4 for g in module.globals]
        self.table = list(module.table)
        for off, fs in module.elems:
            for k, fi in enumerate(fs):
                self.table[off + k] = fi
        self.rc = {}
        self.next = 1
        self.eps = 0
        self.ops = 0
        self.depth = 0

    # -- instance bookkeeping ------------------------------------------------
    def fresh(self, n):
        k = self.next
        self.next = k + n
        rc = self.rc
        for i in range(k, k + n):
            rc[i] = 1
        return tuple(range(k, k + n))

    def copy(self, ids):
        rc = self.rc
        for i in ids:
            if i:
                rc[i] += 1

    def release(self, ids):
        rc = self.rc
        for i in ids:
            if i:
                c = rc[i] - 1
                if c:
                    rc[i] = c
                else:
                    del rc[i]
                    self.eps += 8

    def live_instances(self):
        return len(self.rc)

    # -- execution -------------------------------------------------------------
    def call_export(self, name, args):
        kind, idx = self.m.exports[name]
        if kind != 0:
            raise Unsupported("export %s is not a function" % name)
        params, _ = self.m.types[self.m.funcs[idx].type]
        ids = [Z8 if t in (I64, F64) else Z4 for t in params]
        vals, _ = self.invoke(idx, list(args), ids)
        return vals

    def invoke(self, fidx, lv, li):
        self.depth += 1
        if self.depth > 20000:
            raise Trap("call stack exhausted")
        f = self.m.funcs[fidx]
        params, results = self.m.types[f.type]
        for t in f.locals:
            lv.append(0)
            li.append(Z8 if t in (I64, F64) else Z4)
        code, ends, elses = f.code, f.ends, f.elses
        nres = len(results)
        sv, si = [], []
        labels = [(0, nres, len(code) - 1, False)]
        mem, mids = self.mem, self.mids
        rc = self.rc
        release, copy, fresh = self.release, self.copy, self.fresh
        pc = 0
        ops = 0
        try:
            while True:
                op, a, b = code[pc]
                ops += 1
                if op == 0x20:  # local.get
                    sv.append(lv[a])
                    ids = li[a]
                    for i in ids:
                        if i:
                            rc[i] += 1
                    si.append(ids)
                elif op == 0x21:  # local.set
                    old = li[a]
                    lv[a] = sv.pop()
                    li[a] = si.pop()
                    release(old)
                elif op == 0x22:  # local.tee
                    old = li[a]
                    ids = si[-1]
                    copy(ids)
                    lv[a] = sv[-1]
                    li[a] = ids
                    release(old)
                elif op == 0x41 or op == 0x42:  # const
                    sv.append(a)
                    si.append(Z4 if op == 0x41 else Z8)
                elif op in NUMERIC:
                    kind = NUMERIC[op]
                    if kind[0] == "bin":
                        y = sv.pop()
                        x = sv.pop()
                        release(si.pop())
                        release(si.pop())
                        sv.append(kind[2](x, y))
                        si.append(fresh(kind[1]))
                    elif kind[0] == "cmp":
                        y = sv.pop()
                        x = sv.pop()
                        release(si.pop())
                        release(si.pop())
                        sv.append(1 if kind[2](x, y) else 0)
                        si.append(fresh(1) + (0, 0, 0))
                    elif kind[0] == "test":
                        x = sv.pop()
                        release(si.pop())
                        sv.append(1 if kind[2](x) else 0)
                        si.append(fresh(1) + (0, 0, 0))
                    else:
                        x = sv.pop()
                        release(si.pop())
                        sv.append(kind[3](x))
                        si.append(fresh(kind[2]))
                elif op in MEMOPS:
                    w, t, ext, store = MEMOPS[op]
                    if store:
                        v = sv.pop()
                        ids = si.pop()
                        addr = sv.pop() + a
                        release(si.pop())
                        if addr + w > len(mem):
                            raise Trap("out of bounds memory access")
                        mem[addr:addr + w] = (v & ((1 << (8 * w)) - 1)).to_bytes(w, "little")
                        old = mids[addr:addr + w]
                        mids[addr:addr + w] = ids[:w]
                        release(ids[w:])
                        release(old)
                    else:
                        addr = sv.pop() + a
                        release(si.pop())
                        if addr + w > len(mem):
                            raise Trap("out of bounds memory access")
                        if t in (F32, F64):
                            raise Unsupported("floating-point load")
                        v = int.from_bytes(mem[addr:addr + w], "little")
                        ids = tuple(mids[addr:addr + w])
                        copy(ids)
                        full = WIDTH[t]
                        if ext == "s":
                            v = _sx(v, 8 * w) & ((1 << (8 * full)) - 1)
                            ids = ids + fresh(full - w)
                        elif ext == "u":
                            ids = ids + (0,) * (full - w)
                        sv.append(v)
                        si.append(ids)
                elif op == 0x0D:  # br_if
                    c = sv.pop()
                    release(si.pop())
                    if c:
                        pc = self._branch(labels, a, sv, si)
                        if pc is None:
                            return self._ret(nres, sv, si, li)
                        continue
                elif op == 0x0C:  # br
                    pc = self._branch(labels, a, sv, si)
                    if pc is None:
                        return self._ret(nres, sv, si, li)
                    continue
                elif op == 0x02:  # block
                    p, r = a
                    labels.append((len(sv) - p, r, ends[pc], False))
                elif op == 0x03:  # loop
                    p, r = a
                    labels.append((len(sv) - p, p, pc, True))
                elif op == 0x04:  # if
                    c = sv.pop()
                    release(si.pop())
                    p, r = a
                    if c:
                        labels.append((len(sv) - p, r, ends[pc], False))
                    elif pc in elses:
                        labels.append((len(sv) - p, r, ends[pc], False))
                        pc = elses[pc]
                    else:
                        pc = ends[pc]
                elif op == 0x05:  # else reached from the then-branch
                    pc = labels.pop()[2]
                elif op == 0x0B:  # end
                    if len(labels) == 1:
                        return self._ret(nres, sv, si, li)
                    labels.pop()
                elif op == 0x10:  # call
                    pc = self._call(a, sv, si, pc)
                elif op == 0x1A:  # drop
                    sv.pop()
                    release(si.pop())
                elif op == 0x1B:  # select
                    c = sv.pop()
                    release(si.pop())
                    y = sv.pop()
                    yi = si.pop()
                    if c:
                        release(yi)
                    else:
                        sv[-1] = y
                        release(si[-1])
                        si[-1] = yi
                elif op == 0x23:  # global.get
                    sv.append(self.gv[a])
                    ids = self.gi[a]
                    copy(ids)
                    si.append(ids)
                elif op == 0x24:  # global.set
                    old = self.gi[a]
                    self.gv[a] = sv.pop()
                    self.gi[a] = si.pop()
                    release(old)
                elif op in CONVERT:
                    win, wout, keep, fill, fn = CONVERT[op]
                    x = sv.pop()
                    ids = si.pop()
                    release(ids[keep:])
                    if fill == "zero":
                        out = ids[:keep] + (0,) * (wout - keep)
                    elif fill == "fresh":
                        out = ids[:keep] + fresh(wout - keep)
                    else:
                        out = ids[:keep]
                    sv.append(fn(x))
                    si.append(out)
                elif op == 0x0E:  # br_table
                    i = sv.pop()
                    release(si.pop())
                    pc = self._branch(labels, a[i] if i < len(a) else b, sv, si)
                    if pc is None:
                        return self._ret(nres, sv, si, li)
                    continue
                elif op == 0x0F:  # return
                    return self._ret(nres, sv, si, li)
                elif op == 0x11:  # call_indirect
                    i = sv.pop()
                    release(si.pop())
                    if i >= len(self.table) or self.table[i] is None:
                        raise Trap("undefined table element")
                    fi = self.table[i]
                    if self.m.types[self.m.funcs[fi].type] != self.m.types[a]:
                        raise Trap("indirect call type mismatch")
                    pc = self._call(fi, sv, si, pc)
                elif op == 0x01:  # nop
                    pass
                elif op == 0x3F:  # memory.size
                    sv.append(len(mem) // PAGE)
                    si.append(fresh(4))
                elif op == 0x40:  # memory.grow
                    d = sv.pop()
                    release(si.pop())
                    old = len(mem) // PAGE
                    mem.extend(bytes(d * PAGE))
                    mids.extend([0] * (d * PAGE))
                    sv.append(old)
                    si.append(fresh(4))
                elif op == 0xFC0A:  # memory.copy
                    n = sv.pop()
                    s = sv.pop()
                    d = sv.pop()
                    for _ in range(3):
                        release(si.pop())
                    if s + n > len(mem) or d + n > len(mem):
                        raise Trap("out of bounds memory access")
                    src = mids[s:s + n]
                    copy(src)
                    old = mids[d:d + n]
                    mem[d:d + n] = mem[s:s + n]
                    mids[d:d + n] = src
                    release(old)
                elif op == 0xFC0B:  # memory.fill
                    n = sv.pop()
                    v = sv.pop()
                    vi = si.pop()
                    d = sv.pop()
                    release(si.pop())
                    release(si.pop())
                    if d + n > len(mem):
                        raise Trap("out of bounds memory access")
                    byte = vi[0]
                    if byte:
                        rc[byte] += n
                    release(vi)
                    old = mids[d:d + n]
                    mem[d:d + n] = bytes([v & 0xFF]) * n
                    mids[d:d + n] = [byte] * n
                    release(old)
                elif op == 0x00:
                    raise Trap("unreachable executed")
                else:
                    name = NAMES.get(op) or ("0xfc %d" % (op & 0xFF) if op >= 0xFC00 else "0x%02x" % op)
                    raise Unsupported("instruction %s" % name)
                pc += 1
        finally:
            self.ops += ops
            self.depth -= 1

    def _call(self, fi, sv, si, pc):
        params, _ = self.m.types[self.m.funcs[fi].type]
        n = len(params)
        if n:
            args, aids = sv[-n:], si[-n:]
            del sv[-n:]
            del si[-n:]
        else:
            args, aids = [], []
        vals, ids = self.invoke(fi, args, aids)
        sv.extend(vals)
        si.extend(ids)
        return pc

    def _branch(self, labels, depth, sv, si):
        """Branch to the label depth levels out; None means the function's own label (a return)."""
        h, arity, target, is_loop = labels[-1 - depth]
        keep = len(sv) - arity
        for ids in si[h:keep]:
            self.release(ids)
        del sv[h:keep]
        del si[h:keep]
        if depth == len(labels) - 1:
            return None
        if is_loop:
            del labels[len(labels) - depth:]
            return target + 1
        del labels[len(labels) - 1 - depth:]
        return target + 1

    def _ret(self, nres, sv, si, li):
        vals, ids = sv[len(sv) - nres:], si[len(si) - nres:]
        for x in si[:len(si) - nres]:
            self.release(x)
        for x in li:
            self.release(x)
        return vals, ids


def count(path_or_bytes, export, args):
    """Run export(args) on a fresh instance; return dict with result, eps_bits, ops, ir_nodes."""
    data = path_or_bytes
    if isinstance(data, str):
        with open(data, "rb") as f:
            data = f.read()
    m = Module(data)
    r = Run(m)
    vals = r.call_export(export, args)
    return {"result": vals[0] if vals else None, "eps_bits": r.eps, "ops": r.ops, "ir_nodes": m.ir_nodes()}


def main():
    a = sys.argv[1:]
    if len(a) != 3:
        sys.exit("usage: python3 wasm_eps.py MODULE.wasm EXPORT N")
    sys.setrecursionlimit(100000)
    print(count(a[0], a[1], [int(a[2]) & M32]))


if __name__ == "__main__":
    main()
