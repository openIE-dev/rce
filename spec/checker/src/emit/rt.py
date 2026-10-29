# RCE runtime for emitted Python (spec 10.5). It mirrors the reference
# evaluator value by value: typed integers with range traps, f32 rounded
# once from f64, values copied on assignment, traps as exceptions.
import math, struct, sys, threading, queue, hashlib
sys.setrecursionlimit(1000000)

class Trap(Exception):
    def __init__(s, m): s.m = m
class Fail(Exception):
    def __init__(s, v): s.v = v
class Stop(Exception): pass
class Ret(Exception):
    def __init__(s, v): s.v = v

class Cancel(Exception):
    def __init__(s, f): s.f = f

def trap(m): raise Trap(m)

# Fuel (6.7, 10.5): one unit at each point where the reference evaluator
# ticks (every expression, every statement, every user function call, every
# comprehension element and every element a map, filter or fold visits).
# Each observation starts with full fuel and empty constant caches, as the
# reference starts each observation with a new evaluator.
FUEL_LIMIT = int(next((a[7:] for a in sys.argv if a.startswith("--fuel=")), "50000000"))
FUEL_TRACE = "--fuel-trace" in sys.argv
# The pool and the constant cache are per thread (TLS), as the reference
# gives every task its own evaluator; a par branch starts from its parent's
# remaining fuel and is charged back in source order (par_resolve).
def fuel():
    t = TLS
    if t.fuel == 0: trap("comptime fuel exhausted")
    t.fuel -= 1
# `bound n` (6.7): active calls of each bounded function, per thread
def depth_enter(k, n):
    d = TLS.depth.get(k, 0)
    if d >= n: trap("recursion bound")
    TLS.depth[k] = d + 1
def depth_exit(k):
    d = TLS.depth.get(k, 0)
    if d > 0: TLS.depth[k] = d - 1
def fuel_reset():
    TLS.depth = {}
    TLS.fuel = FUEL_LIMIT
    TLS.kc = {}
def fuel_report():
    if FUEL_TRACE: print("fuel %d" % (FUEL_LIMIT - TLS.fuel))
# `--fuel-batch=n1,n2,..`: every observation once per limit in this process,
# each pass headed by `#fuel-batch n` (an observation starts from full fuel
# and empty caches, so a pass equals a fresh run)
def observe_all(f):
    global FUEL_LIMIT
    b = next((a[13:] for a in sys.argv if a.startswith("--fuel-batch=")), None)
    if b is None: f(); return
    for n in b.split(","):
        FUEL_LIMIT = int(n); print("#fuel-batch " + n); f()

# per-thread runtime state: each task and par branch runs on its own thread
class _TLS(threading.local):
    def __init__(s):
        s.yields = []; s.out = []; s.bud = []; s.puts = [0]; s.sched = [0]
        s.task = None; s.atomic = None; s.hsends = None; s.parcx = None
        s.fuel = FUEL_LIMIT; s.kc = {}; s.depth = {}
TLS = _TLS()

RANGES = {"i8": (-128, 127), "i16": (-32768, 32767), "i32": (-2**31, 2**31 - 1),
          "i64": (-2**63, 2**63 - 1), "u8": (0, 255), "u16": (0, 65535),
          "u32": (0, 2**32 - 1), "u64": (0, 2**64 - 1), "usize": (0, 2**64 - 1)}
def irange(t): return RANGES.get(t, RANGES["i64"])
I128 = (-2**127, 2**127 - 1)

class Int:
    __slots__ = ("v", "t")
    def __init__(s, v, t): s.v = v; s.t = t
class F:
    __slots__ = ("v",)
    def __init__(s, v): s.v = v
class F32:
    __slots__ = ("v",)
    def __init__(s, v): s.v = v
class UnitT:
    __slots__ = ()
UNIT = UnitT()
class Tup:
    __slots__ = ("xs",)
    def __init__(s, xs): s.xs = xs
class Lst:
    __slots__ = ("xs",)
    def __init__(s, xs): s.xs = xs
class St:
    # a Set[τ]: elements in canonical order without duplicates (7.1)
    __slots__ = ("xs",)
    def __init__(s, xs): s.xs = xs
class Mp:
    __slots__ = ("ps",)
    def __init__(s, ps): s.ps = ps
class Opt:
    __slots__ = ("x",)
    def __init__(s, x): s.x = x
NONE = Opt(None)
class Rec:
    __slots__ = ("n", "fs")
    def __init__(s, n, fs): s.n = n; s.fs = fs
class Err:
    __slots__ = ("e",)
    def __init__(s, e): s.e = e
class FnV:
    __slots__ = ("n",)
    def __init__(s, n): s.n = n
class Clo:
    __slots__ = ("f",)
    def __init__(s, f): s.f = f
class Rng:
    __slots__ = ("a", "b", "t")
    def __init__(s, a, b, t): s.a = a; s.b = b; s.t = t
class Gen:
    __slots__ = ("run",)
    def __init__(s, run): s.run = run

def some(x): return Opt(x)
def lit(v): return Int(v, "lit")

# ---- formatting (Rust Display / Debug) ----
def _digits(s):
    # s: a decimal in Python repr or %e form, positive. Returns (digits, point).
    if "e" in s: m, e = s.split("e"); e = int(e)
    else: m, e = s, 0
    if "." in m: i = m.index("."); d = m[:i] + m[i + 1:]; p = i + e
    else: d = m; p = len(m) + e
    k = 0
    while k < len(d) - 1 and d[k] == "0": k += 1
    d = d[k:]; p -= k
    d = d.rstrip("0") or "0"
    return d, p
def _plain(neg, d, p):
    if d == "0": r = "0"
    elif p <= 0: r = "0." + "0" * (-p) + d
    elif p >= len(d): r = d + "0" * (p - len(d))
    else: r = d[:p] + "." + d[p:]
    return ("-" if neg else "") + r
def _neg(x): return math.copysign(1.0, x) < 0
def fdisplay(x):
    if x == 0: return "-0" if _neg(x) else "0"
    a = abs(x); d, p = _digits(repr(a))
    if 1e-5 <= a < 1e16: return _plain(x < 0, d, p)
    m = d[0] + ("." + d[1:] if len(d) > 1 else ""); e = p - 1
    return ("-" if x < 0 else "") + m + "e" + ("+" if e >= 0 else "") + str(e)
def fdebug(x):
    if x == 0: return "-0.0" if _neg(x) else "0.0"
    a = abs(x); d, p = _digits(repr(a))
    if 1e-4 <= a < 1e16:
        s = _plain(x < 0, d, p)
        return s if "." in s else s + ".0"
    m = d[0] + ("." + d[1:] if len(d) > 1 else "")
    return ("-" if x < 0 else "") + m + "e" + str(p - 1)
def f32r(x):
    try: return struct.unpack("f", struct.pack("f", x))[0]
    except OverflowError: return math.copysign(math.inf, x)
def f32digits(x):
    a = abs(x)
    for q in range(1, 10):
        s = "%.*e" % (q - 1, a)
        if f32r(float(s)) == a: return _digits(s)
    return _digits(repr(a))
def f32display(x):
    if x == 0: return "-0" if _neg(x) else "0"
    d, p = f32digits(x)
    return _plain(x < 0, d, p)
def int_to_f32(n):
    a = abs(n)
    if a < 2**24: return float(n)
    b = a.bit_length(); sh = b - 24
    m = a >> sh; rem = a - (m << sh); half = 1 << (sh - 1)
    if rem > half or (rem == half and (m & 1)): m += 1
    r = float(m) * 2.0**sh if sh < 1000 else math.inf
    r = f32r(r)
    return -r if n < 0 else r
def display_escapes(k):
    # 6.3 N7: k lies in a range of DISPLAY_ESCAPE, the set of
    # spec/data/display-escape.json that the emitter writes after this runtime
    lo, hi = 0, len(DISPLAY_ESCAPE) // 2
    while lo < hi:
        m = (lo + hi) // 2
        if DISPLAY_ESCAPE[2 * m + 1] < k: lo = m + 1
        else: hi = m
    return lo < len(DISPLAY_ESCAPE) // 2 and DISPLAY_ESCAPE[2 * lo] <= k
def rdebug(s):
    o = ['"']
    for c in s:
        k = ord(c)
        if c == "\t": o.append("\\t")
        elif c == "\r": o.append("\\r")
        elif c == "\n": o.append("\\n")
        elif c == "\\": o.append("\\\\")
        elif c == '"': o.append('\\"')
        elif c == "\0": o.append("\\0")
        elif display_escapes(k): o.append("\\u{%x}" % k)
        else: o.append(c)
    o.append('"')
    return "".join(o)

def show(v):
    if v is True: return "true"
    if v is False: return "false"
    c = type(v)
    if c is Int: return str(v.v)
    if c is LVar: return "lvar(" + show(v.v) + ")"
    if c is F: return fdisplay(v.v)
    if c is str: return rdebug(v)
    if c is UnitT: return "()"
    if c is Tup: return "(" + ", ".join(show(x) for x in v.xs) + ")"
    if c is Lst or c is St: return "[" + ", ".join(show(x) for x in v.xs) + "]"
    if c is Mp: return "{" + ", ".join(show(a) + ": " + show(b) for a, b in v.ps) + "}"
    if c is Opt: return "none" if v.x is None else "some(" + show(v.x) + ")"
    if c is Rec:
        if not v.fs: return v.n
        return v.n + "(" + ", ".join(a + ": " + show(b) for a, b in v.fs) + ")"
    if c is Err: return "error " + show(v.e)
    if c is FnV: return v.n
    if c is Clo: return "<fn>"
    if c is Rng: return "%d..%d" % (v.a, v.b)
    if c is F32: return f32display(v.v)
    if c is Gen: return "<iter>"
    if c is Chan: return "<chan>"
    if c is Task: return "<task>"
    if c is Actor: return "<actor " + v.ty + ">"
    if c is Session: return "<session>"
    return "?"

def eq(a, b):
    ca, cb = type(a), type(b)
    if ca is bool or cb is bool: return ca is cb and a == b
    if ca is not cb: return False
    if ca is Int or ca is F or ca is F32: return a.v == b.v
    if ca is str: return a == b
    if ca is UnitT: return True
    if ca is Tup or ca is Lst: return len(a.xs) == len(b.xs) and all(eq(x, y) for x, y in zip(a.xs, b.xs))
    if ca is St: return len(a.xs) == len(b.xs) and all(keq(x, y) for x, y in zip(a.xs, b.xs))
    if ca is Mp:
        return len(a.ps) == len(b.ps) and all(any(keq(k, k2) and eq(v, v2) for k2, v2 in b.ps) for k, v in a.ps)
    if ca is Opt:
        if a.x is None or b.x is None: return a.x is None and b.x is None
        return eq(a.x, b.x)
    if ca is Rec: return a.n == b.n and len(a.fs) == len(b.fs) and all(p == q and eq(x, y) for (p, x), (q, y) in zip(a.fs, b.fs))
    if ca is Err: return eq(a.e, b.e)
    if ca is Chan or ca is Task or ca is Actor: return a.id == b.id
    if ca is Session: return a.out == b.out and a.inn == b.inn
    if ca is LVar: return a is b
    return False

def _c(x, y): return (x > y) - (x < y)
def cmp(a, b):
    ca, cb = type(a), type(b)
    if ca is Int and cb is Int: return _c(a.v, b.v)
    if (ca is F and cb is F) or (ca is F32 and cb is F32):
        if math.isnan(a.v) or math.isnan(b.v): return 0
        return _c(a.v, b.v)
    if ca is str and cb is str: return _c(a, b)
    if ca is bool and cb is bool: return _c(a, b)
    if (ca is Tup and cb is Tup) or (ca is Lst and cb is Lst) or (ca is St and cb is St):
        for x, y in zip(a.xs, b.xs):
            r = cmp(x, y)
            if r: return r
        return _c(len(a.xs), len(b.xs))
    trap("values are not ordered")

def truthy(v):
    if v is True or v is False: return v
    trap("expected bool, found " + show(v))

def chk(x, t):
    tt = "i64" if t == "lit" else t
    lo, hi = irange(tt)
    if x < lo or x > hi: trap("overflow: %d is outside %s" % (x, tt))
    return Int(x, t)

def tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q
def trem(a, b): return a - b * tdiv(a, b)

def finite(r, k):
    if math.isinf(r) or math.isnan(r): trap("non-finite float")
    return k(r)

def arith(op, a, b): return size_ok(arith_raw(op, a, b))
def arith_raw(op, a, b):
    ca, cb = type(a), type(b)
    if ca is Int and cb is Int:
        x, y = a.v, b.v
        ty = b.t if a.t == "lit" else a.t
        bits = 8 if ty in ("i8", "u8") else 16 if ty in ("i16", "u16") else 32 if ty in ("i32", "u32") else 64
        if op == "+": return chk(x + y, ty)
        if op == "-": return chk(x - y, ty)
        if op == "*": return chk(x * y, ty)
        if op == "/":
            if y == 0: trap("division by zero")
            return chk(tdiv(x, y), ty)
        if op == "%":
            if y == 0: trap("division by zero")
            chk(tdiv(x, y), ty)
            return chk(trem(x, y), ty)
        if op == "&": return chk(x & y, ty)
        if op == "|": return chk(x | y, ty)
        if op == "^": return chk(x ^ y, ty)
        if op == "<<":
            if y < 0 or y >= bits: trap("shift amount out of range")
            r = (x << y) & ((1 << bits) - 1)
            return chk(r - (1 << bits) if ty.startswith("i") and r >= 1 << (bits - 1) else r, ty)
        if op == ">>":
            if y < 0 or y >= bits: trap("shift amount out of range")
            return chk(x >> y, ty)
        trap("bad operator " + op)
    if (ca is F and cb is F) or (ca is F32 and cb is F32):
        x, y = a.v, b.v
        if op == "+": r = x + y
        elif op == "-": r = x - y
        elif op == "*": r = x * y
        elif op == "/":
            if y == 0:
                r = math.nan if x == 0 or math.isnan(x) else math.copysign(math.inf, x) * math.copysign(1.0, y)
            else: r = x / y
        else: trap("bad float operator")
        if ca is F32: r = f32r(r); return finite(r, F32)
        return finite(r, F)
    if ca is Lst and cb is Lst and op in ("+", "-", "*", "/", "%"):
        if len(a.xs) != len(b.xs): trap("lane count mismatch")
        return Lst([arith(op, x, y) for x, y in zip(a.xs, b.xs)])
    if ca is F and cb is Int: return arith(op, a, F(float(b.v)))
    if ca is Int and cb is F: return arith(op, F(float(a.v)), b)
    if ca is str and cb is str and op == "+": return a + b
    trap("cannot apply %s to %s and %s" % (op, show(a), show(b)))

def contains(c, x):
    t = type(c)
    if t is Lst or t is Tup: return any(eq(y, x) for y in c.xs)
    if t is St: return any(keq(y, x) for y in c.xs)
    if t is Mp: return any(keq(k, x) for k, _ in c.ps)
    if t is str: return type(x) is str and x in c
    if t is Rng: return type(x) is Int and c.a <= x.v < c.b
    trap("`in` needs a collection")

def binop(op, x, y):
    if op == "==": return eq(x, y)
    if op == "!=": return not eq(x, y)
    if op == "<": return cmp(x, y) < 0
    if op == "<=": return cmp(x, y) <= 0
    if op == ">": return cmp(x, y) > 0
    if op == ">=": return cmp(x, y) >= 0
    if op == "in": return contains(y, x)
    if op == "not in": return not contains(y, x)
    if op == ".." or op == "..=":
        if type(x) is Int and type(y) is Int:
            return Rng(x.v, y.v if op == ".." else y.v + 1, y.t if x.t == "lit" else x.t)
        trap("bad range")
    return arith(op, x, y)

def map_set(ps, k, v):
    out = list(ps)
    for i, (a, _) in enumerate(out):
        if keq(a, k): out[i] = (a, v); return out
    i = 0
    while i < len(out) and canon_cmp(out[i][0], k) < 0: i += 1
    out.insert(i, (k, v)); return out
def mk_map(pairs):
    ps = []
    for k, v in pairs: ps = map_set(ps, k, v)
    return Mp(ps)
def list_repeat(x, n):
    if type(n) is Int and 0 <= n.v <= 10000000:
        charge(4 + n.v)
        return Lst([x] * n.v)
    trap("bad repeat count")

def u8s(s): return s.encode("utf-8")
def index(c, i):
    tc = type(c)
    if tc is Lst and type(i) is Int:
        k = i.v
        if 0 <= k < len(c.xs): return c.xs[k]
        trap("index %d out of range" % k)
    if tc is str and type(i) is Int:
        b = u8s(c); k = i.v
        if 0 <= k < len(b): return Int(b[k], "u8")
        trap("index %d out of range" % k)
    if tc is Mp:
        for a, v in c.ps:
            if keq(a, i): return v
        trap("missing key " + show(i))
    trap("cannot index " + show(c))

def slice_(c, ta, tb):
    charge(1)
    tc = type(c)
    if tc is Lst: xs = c.xs; text = None
    elif tc is str: text = u8s(c); xs = text
    else: trap("cannot slice " + show(c))
    n = len(xs)
    av = ta() if ta else None
    bv = tb() if tb else None
    def norm(v, d):
        if v is None: return d
        if type(v) is Int:
            k = v.v
            if k < 0: k = n + k
            if k < 0 or k > n: trap("slice index out of range")
            return k
        trap("bad slice bound")
    lo = norm(av, 0); hi = norm(bv, n)
    if lo > hi: trap("slice bounds reversed")
    if text is not None:
        def bd(k): return k == 0 or k == n or (text[k] & 0xC0) != 0x80
        if not bd(lo) or not bd(hi): trap("not a char boundary")
        return text[lo:hi].decode("utf-8")
    return Lst(xs[lo:hi])

def field(v, m):
    if type(v) is Rec:
        for n, x in v.fs:
            if n == m: return x
        trap("no field " + m)
    trap("%s has no field %s" % (show(v), m))
def tmember(v, i):
    if type(v) is Tup and i < len(v.xs): return v.xs[i]
    trap("bad tuple index")
def neg(v):
    t = type(v)
    if t is Int: return chk(-v.v, v.t)
    if t is F: return F(-v.v)
    if t is F32: return F32(-v.v)
    trap("bad negation")
def bitnot(v):
    if type(v) is Int:
        lo, hi = irange("i64" if v.t == "lit" else v.t)
        return Int(hi - v.v if lo == 0 else ~v.v, v.t)
    trap("bad ~")
def not_(v): return not truthy(v)

def sat128(x):
    return max(I128[0], min(I128[1], x))
def cast(v, T):
    to_f32 = T[0] == "f32"
    n = T[1] if T[0] == "int" else None
    t = type(v)
    if t is Int and n is None and to_f32: return F32(int_to_f32(v.v))
    if t is F and n is None and to_f32:
        r = f32r(v.v)
        if math.isinf(r) or math.isnan(r): trap("non-finite float")
        return F32(r)
    if t is F32 and n is None and not to_f32: return F(v.v)
    if t is F32 and n is not None: return chk(int(v.v) if math.isfinite(v.v) else 0, n)
    if t is Int and n is not None: return chk(v.v, n)
    if t is F and n is not None:
        if not math.isfinite(v.v): trap("non-finite float")
        return chk(int(v.v), n)
    if (v is True or v is False) and n is not None: return chk(int(v), n)
    if t is Int and n is None: return F(float(v.v))
    return v

def prompt_encode(s):
    # F.5.2: SHA-256 derived delimiter, rehashed until the text does not contain it
    h = hashlib.sha256(s.encode("utf-8")).digest()
    while True:
        d = "data-" + h.hex()
        if d not in s: return "<<" + d + "\n" + s + "\n" + d + ">>"
        h = hashlib.sha256(h).digest()
def ctx_lit(s):
    charge(4)
    return s
def ctx_encode(c, s):
    charge(4 + size_of(s))
    if type(s) is not str: trap("context text")
    if c == "Html": return "".join({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}.get(ch, ch) for ch in s)
    if c == "Url": return "".join(chr(b) if (48 <= b <= 57 or 65 <= b <= 90 or 97 <= b <= 122 or b in (45, 46, 95, 126)) else "%%%02X" % b for b in s.encode("utf-8"))
    if c == "Sql": return "'" + s.replace("'", "''") + "'"
    if c == "ShellArg": return s
    if c == "Prompt": return prompt_encode(s)
    if c == "Header":
        if "\r" in s or "\n" in s: trap("Header.encode: a header value contains CR or LF (F.5.7)")
        return s
    o = ['"']
    for ch in s:
        if ch == '"': o.append('\\"')
        elif ch == "\\": o.append("\\\\")
        elif ch == "\n": o.append("\\n")
        elif ch == "\r": o.append("\\r")
        elif ch == "\t": o.append("\\t")
        elif ord(ch) < 0x20: o.append("\\u%04x" % ord(ch))
        else: o.append(ch)
    o.append('"')
    return "".join(o)
def ptm(t, v):
    if t[0] == "lit": return t[1]
    if t[0] == "self": return v.v if type(v) is Int else None
    if type(v) is str: return len(v.encode("utf-8"))
    if type(v) is Lst or type(v) is St: return len(v.xs)
    if type(v) is Mp: return len(v.ps)
    return None
def pholds(p, v):
    k = p[0]
    if k == "cmp":
        x, y = ptm(p[2], v), ptm(p[3], v)
        if x is None or y is None: return False
        o = p[1]
        return x < y if o == "<" else x <= y if o == "<=" else x > y if o == ">" else x >= y if o == ">=" else x == y if o == "==" else x != y
    if k == "cls":
        if type(v) is not str: return False
        c = p[1]
        if c == "isdigit": return v != "" and all("0" <= ch <= "9" for ch in v)
        if c == "isalpha": return v != "" and all("a" <= ch <= "z" or "A" <= ch <= "Z" for ch in v)
        if c == "isspace": return v != "" and all(ch in " \t\n\x0c\r" for ch in v)
        return v == ""
    if k == "and": return pholds(p[1], v) and pholds(p[2], v)
    if k == "or": return pholds(p[1], v) or pholds(p[2], v)
    return not pholds(p[1], v)
def fit(v, T):
    if T[0] == "refn":
        v = fit(v, T[1])
        if not pholds(T[3], v): trap("refinement: value is outside %s" % T[2])
        return v
    k = T[0]; t = type(v)
    if t is Int:
        if k == "int":
            lo, hi = irange(T[1])
            if v.v < lo or v.v > hi: trap("overflow: %d is outside %s" % (v.v, T[1]))
            return Int(v.v, T[1])
        if k == "f64": return F(float(v.v))
        if k == "f32": return F32(int_to_f32(v.v))
    if t is Opt and v.x is not None and k == "opt": return Opt(fit(v.x, T[1]))
    if t is Lst and (k == "list" or k == "arr"): return Lst([fit(x, T[1]) for x in v.xs])
    if t is Mp and k == "map": return Mp([(fit(a, T[1]), fit(b, T[2])) for a, b in v.ps])
    if t is Tup and k == "tup" and len(v.xs) == len(T[1]): return Tup([fit(x, y) for x, y in zip(v.xs, T[1])])
    if t is Err: return v
    if k == "union" or k == "ref": return fit(v, T[1])
    return v
def fitall(a, Ts): return [fit(x, t) for x, t in zip(a, Ts)]

def wrapsat(m, x, t, y):
    if type(y) is not Int: trap(m)
    y = y.v
    r = x + y if m.endswith("add") else x - y if m.endswith("sub") else x * y
    lo, hi = irange(t)
    if m.startswith("wrapping"): return Int((r - lo) % (hi - lo + 1) + lo, t)
    return Int(max(lo, min(hi, r)), t)

WS = set("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000")
def rtrim(s):
    i, j = 0, len(s)
    while i < j and s[i] in WS: i += 1
    while j > i and s[j - 1] in WS: j -= 1
    return s[i:j]
def rwords(s):
    out, cur = [], []
    for c in s:
        if c in WS:
            if cur: out.append("".join(cur)); cur = []
        else: cur.append(c)
    if cur: out.append("".join(cur))
    return out
def rlines(s):
    if s == "": return []
    parts = s.split("\n")
    if parts[-1] == "": parts.pop()
    return [p[:-1] if p.endswith("\r") else p for p in parts]
def aupper(s): return "".join(chr(ord(c) - 32) if "a" <= c <= "z" else c for c in s)
def alower(s): return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)
def toint(s):
    t = rtrim(s); u = t[1:] if t[:1] in ("+", "-") else t
    if u == "" or not all("0" <= c <= "9" for c in u): trap("not an integer: " + rdebug(s))
    n = int(t)
    if n < I128[0] or n > I128[1]: trap("not an integer: " + rdebug(s))
    return chk(n, "i64")
def fround(x):
    r = float(math.trunc(x)) or math.copysign(0.0, x)
    if abs(x - r) >= 0.5: r += math.copysign(1.0, x)
    return r

def sort_list(xs):
    import functools
    err = []
    def k(p, q):
        try: return cmp(p, q)
        except Trap as e:
            if not err: err.append(e)
            return 0
    r = sorted(xs, key=functools.cmp_to_key(k))
    if err: raise err[0]
    return r

def apply(f, vals):
    if type(f) is Clo: return f.f(vals)
    if type(f) is FnV: return RAW[f.n](vals)
    trap(show(f) + " is not callable")
def call_value(f, vals):
    if type(f) is FnV:
        fuel(); return FNS[f.n](None, vals)[0]
    if type(f) is Clo: return f.f(vals)
    trap(show(f) + " is not callable")

def builtin(r, m, a):
    v, w = builtin_raw(r, m, a)
    return size_ok(v), (size_ok(w) if w is not None else None)
def builtin_raw(r, m, a):
    t = type(r)
    if t is Chan and m == "close": comm(("close", r)); return UNIT, None
    if t is Task and m == "join":
        rp = comm(("join", r.id))
        return (rp[1] if rp[0] == "val" else UNIT), None
    if t is Session and m == "offer":
        charge(8)
        rp = comm(("recv", Chan(r.inn, r.cap)))
        if rp[0] == "val" and type(rp[1]) is Opt and rp[1].x is not None:
            v = rp[1].x
            if type(v) is Rec and not v.fs: return Rec("$label:" + v.n, []), None
            return v, None
        trap("session closed")
    if t is Session and m == "close": comm(("close", Chan(r.out, r.cap))); return UNIT, None
    if t is LVar:
        if m == "put": lvar_put(r, a[0] if a else UNIT); return UNIT, None
        if m == "wait": return lvar_wait(r, a[0] if a else UNIT), None
        if m == "freeze": return lvar_freeze(r), None
    if t is Int:
        x, ty = r.v, r.t
        if m in ("checked_add", "checked_sub", "checked_mul", "checked_div"):
            op = {"checked_add": "+", "checked_sub": "-", "checked_mul": "*"}.get(m, "/")
            try: return Opt(arith(op, r, a[0])), None
            except Trap: return NONE, None
        if m in ("wrapping_add", "wrapping_sub", "wrapping_mul", "saturating_add", "saturating_sub", "saturating_mul"):
            return wrapsat(m, x, ty, a[0]), None
        if m == "tostr": return str(x), None
        if m == "abs": return chk(abs(x), ty), None
        if m in ("min", "max"):
            if type(a[0]) is not Int: trap(m)
            return Int(min(x, a[0].v) if m == "min" else max(x, a[0].v), ty), None
    if t is F:
        x = r.v
        if m == "sqrt":
            if x < 0: trap("non-finite float")
            return F(math.sqrt(x)), None
        if m == "abs": return F(abs(x)), None
        if m in ("min", "max"):
            if type(a[0]) is not F: trap(m)
            y = a[0].v
            return F((x if x <= y else y) if m == "min" else (x if x >= y else y)), None
        # F7: a zero result keeps the operand's sign (ceil(-0.3) is -0.0), as IEEE 754 does
        if m == "floor": return F(float(math.floor(x)) or math.copysign(0.0, x)), None
        if m == "ceil": return F(float(math.ceil(x)) or math.copysign(0.0, x)), None
        if m == "round": return F(fround(x)), None
        if m == "tostr": return fdisplay(x), None
    if (r is True or r is False) and m == "tostr": return ("true" if r else "false"), None
    if t is str and m in ("len", "len_bytes"): return Int(len(u8s(r)), "usize"), None
    if t is str and m == "len_chars": return Int(len(r), "usize"), None
    if t is F32:
        if m == "abs": return F32(abs(r.v)), None
        if m == "tostr": return f32display(r.v), None
    if t is Gen and m == "collect":
        items, _, err = r.run(1 << 62)
        if err is not None: raise err
        return Lst(items), None
    if t is str:
        if m == "strip": return rtrim(r), None
        if m == "upper": return aupper(r), None
        if m == "sha256": return hashlib.sha256(r.encode("utf-8")).hexdigest(), None
        if m == "lower": return alower(r), None
        if m == "reverse": return r[::-1], None
        if m == "tostr": return r, None
        if m == "raw": return r, None
        if m == "chars": return Lst(list(r)), None
        if m == "words": return Lst(rwords(r)), None
        if m == "lines": return Lst(rlines(r)), None
        if m == "split":
            if type(a[0]) is not str: trap("split")
            if a[0] == "": return Lst([""] + list(r) + [""]), None
            return Lst(r.split(a[0])), None
        if m == "isdigit": return (r != "" and all("0" <= c <= "9" for c in r)), None
        if m == "isalpha": return (r != "" and all("a" <= c <= "z" or "A" <= c <= "Z" for c in r)), None
        if m == "isspace": return (r != "" and all(c in " \t\n\x0c\r" for c in r)), None
        if m == "isempty": return r == "", None
        if m in ("contains", "startswith", "endswith"):
            if type(a[0]) is not str: trap(m)
            return (a[0] in r if m == "contains" else r.startswith(a[0]) if m == "startswith" else r.endswith(a[0])), None
        if m == "toint": return toint(r), None
        if m == "join":
            if type(a[0]) is not Lst: trap("join")
            return r.join(x if type(x) is str else show(x) for x in a[0].xs), None
    if t is Lst:
        xs = r.xs
        if m == "len": return Int(len(xs), "usize"), None
        if m == "isempty": return len(xs) == 0, None
        if m in ("append", "push"): return UNIT, Lst(xs + [a[0]])
        if m == "pop":
            if not xs: return NONE, Lst([])
            return Opt(xs[-1]), Lst(xs[:-1])
        if m == "clear": return UNIT, Lst([])
        if m == "sort": return UNIT, Lst(sort_list(xs))
        if m == "with": return Lst(xs + [a[0]]), None
        if m == "reverse": return Lst(xs[::-1]), None
        if m == "contains": return any(eq(x, a[0]) for x in xs), None
        if m == "get":
            if type(a[0]) is not Int: trap("get")
            k = a[0].v
            return (Opt(xs[k]) if 0 <= k < len(xs) else NONE), None
        if m == "utf8":
            if not all(type(x) is Int and 0 <= x.v <= 255 for x in xs): trap("utf8: byte outside 0..255")
            try: return bytes(x.v for x in xs).decode("utf-8", "strict"), None
            except UnicodeDecodeError: trap("utf8: invalid UTF-8")
        if m == "sum":
            acc = Int(0, "lit")
            for x in xs: acc = arith("+", acc, x)
            return acc, None
        if m == "map": return Lst([(fuel(), apply(a[0], [x]))[1] for x in xs]), None
        if m == "filter": return Lst([x for x in xs if (fuel(), apply(a[0], [x]))[1] is True]), None
        if m == "pscan": return pscan(xs, a[0], a[1], False), None
        if m == "pscan_incl": return pscan(xs, a[0], a[1], True), None
        if m == "fold":
            acc = a[0]
            for x in xs: fuel(); acc = apply(a[1], [acc, x])
            return acc, None
    if t is St:
        if m == "union": return merge_builtin(r, a[0]), None
        if m == "inter": return St([x for x in r.xs if any(keq(x, y) for y in a[0].xs)]), None
        if m == "insert": return UNIT, merge_builtin(r, St([a[0]]))
        if m == "tolist": return Lst(list(r.xs)), None
        if m == "len": return Int(len(r.xs), "usize"), None
        if m == "contains": return any(keq(x, a[0]) for x in r.xs), None
    if t is Mp:
        ps = r.ps
        if m == "merge": return map_merge(ps, a[0].ps), None
        if m == "len": return Int(len(ps), "usize"), None
        if m == "get":
            for k, v in ps:
                if keq(k, a[0]): return v, None
            return Err(Rec("Missing", [])), None
        if m == "contains": return any(keq(k, a[0]) for k, _ in ps), None
        if m == "insert": return UNIT, Mp(map_set(ps, a[0], a[1]))
        if m == "remove":
            for i, (k, v) in enumerate(ps):
                if keq(k, a[0]): return Opt(v), Mp(ps[:i] + ps[i + 1:])
            return NONE, Mp(list(ps))
        if m == "keys": return Lst([k for k, _ in ps]), None
        if m == "values": return Lst([v for _, v in ps]), None
    if t is Opt:
        if m == "isnone": return r.x is None, None
        if m == "issome": return r.x is not None, None
        if m == "or": return (a[0] if r.x is None else r.x), None
    trap("%s has no method %s at comptime" % (show(r), m))

# C.6.4 method work, as conc::method_work in the reference evaluator.
LIN_TEXT = frozenset(("strip", "upper", "lower", "reverse", "tostr", "chars", "words", "lines", "split", "isdigit", "isalpha", "isspace", "contains", "startswith", "endswith", "toint", "join", "sha256",))
LIN_LIST = frozenset(("reverse", "contains", "sum", "map", "filter", "fold", "utf8",))
LIN_SET = frozenset(("union", "inter", "insert", "contains", "tolist",))
LIN_MAP = frozenset(("get", "contains", "insert", "remove", "keys", "values", "merge",))
def size_of(v):
    t = type(v)
    if t is str: return len(u8s(v))
    if t is Lst or t is St: return len(v.xs)
    if t is Mp: return len(v.ps)
    return 0
def ceil_log2(n): return 0 if n <= 1 else (n - 1).bit_length()
def method_work(recv, m, args):
    n = size_of(recv)
    k = size_of(args[0]) if args else 0
    t = type(recv)
    if (t is str and m in LIN_TEXT) or (t is Lst and m in LIN_LIST) or (t is St and m in LIN_SET) or (t is Mp and m in LIN_MAP): return n + k
    if t is Lst and m == "sort": return n * ceil_log2(n)
    return 0

def call_method(recv, m, args, sized):
    """(value, per-argument copy-out list, new receiver or None)"""
    t = type(recv)
    head = KIND_OF.get(recv.n, recv.n) if t is Rec else "List" if t is Lst else "Set" if t is St else None
    if head is not None and (head, m) in METHODS:
        fuel()
        v, outs, s = METHODS[(head, m)](recv, args)
        return v, outs, s
    charge(4 + (method_work(recv, m, args) if sized else 0))
    v, s = builtin(recv, m, args)
    return v, [], s

def construct(name, fields, args):
    n = len(fields); vals = [None] * n
    for i, (lb, v) in enumerate(args):
        k = i
        if lb is not None:
            for j, f in enumerate(fields):
                if f[0] == lb: k = j; break
        if k < n: vals[k] = fit(v, fields[k][1])
    out = []
    for k, (f, T, init) in enumerate(fields):
        v = vals[k]
        if v is None:
            if init is None: trap("missing field " + f)
            v = fit(init(), T)
        out.append((f, v))
    return Rec(name, out)

def pmatch(P, v, b):
    k = P[0]
    if k == "wild": return True
    if (k == "bind" or k == "tag") and type(v) is Rec and not v.fs and v.n.startswith("$label:"):
        return v.n[7:] == P[1].split("::")[-1]
    if k == "bind": b[P[1]] = v; return True
    if k == "tag": return type(v) is Rec and v.n == P[1]
    if k == "some":
        if type(v) is Opt and v.x is not None: return pmatch(P[1], v.x, b)
        return False
    if k == "ctor":
        if type(v) is not Rec or v.n != P[1]: return False
        for i, (lb, q) in enumerate(P[2]):
            fv = None
            if lb is not None:
                for n, x in v.fs:
                    if n == lb: fv = x; break
            elif i < len(v.fs): fv = v.fs[i][1]
            if fv is None or not pmatch(q, fv, b): return False
        return True
    if k == "lit": return eq(P[1](), v)
    if k == "range":
        x, y = P[1](), P[3]()
        lo = cmp(v, x) >= 0
        hi = cmp(v, y) <= 0 if P[2] else cmp(v, y) < 0
        return lo and hi
    if k == "tup":
        if type(v) is not Tup or len(v.xs) != len(P[1]): return False
        for q, x in zip(P[1], v.xs):
            if not pmatch(q, x, b): return False
        return True
    if k == "alt":
        for q in P[1]:
            if pmatch(q, v, b): return True
        return False
    return False
def pm(P, v):
    b = {}
    return b if pmatch(P, v, b) else None

def yield_(v):
    if not TLS.yields: trap("yield outside an iterator")
    buf, cap = TLS.yields[-1]
    if len(buf) >= LOOP_LIMIT: trap("loop bound")
    buf.append(v)
    if len(buf) >= cap: raise Stop()
def gen_run(body, cap):
    TLS.yields.append(([], cap))
    try:
        body(); fin, err = True, None
    except Stop: fin, err = False, None
    except (Trap, Fail) as e: fin, err = True, e
    finally: items = TLS.yields.pop()[0]
    return items, fin, err

def iter_items(v):
    t = type(v)
    if t is Lst or t is St: return v.xs
    if t is Rng:
        if v.b - v.a > LOOP_LIMIT * 10: trap("loop bound")
        return [Int(i, v.t) for i in range(v.a, max(v.a, v.b))]
    if t is str: return list(v)
    if t is Mp: return [Tup([k, x]) for k, x in v.ps]
    if t is Gen:
        items, _, err = v.run(1 << 62)
        if err is not None: raise err
        return items
    trap("cannot iterate over " + show(v))
def iterate(v):
    if type(v) is not Gen:
        yield from iter_items(v); return
    done, cap = 0, 64
    while True:
        items, fin, err = v.run(cap)
        for x in items[done:]:
            done += 1
            yield x
        if err is not None: raise err
        if fin: return
        cap *= 2

def listcomp(itv, P, names, body, cond):
    out = []
    for x in iter_items(itv):
        fuel()
        b = {}
        if not pmatch(P, x, b): continue
        args = [b.get(n, UNIT) for n in names]
        if cond is not None and not truthy(cond(*args)): continue
        out.append(body(*args))
    charge(4 + len(out))
    return Lst(out)

def catch_(x, h): return h(x.e) if type(x) is Err else x
def try_(x, els):
    t = type(x)
    if t is Err: raise Fail(els() if els else x.e)
    if t is Opt and not els: return x  # ok value of `T? ! E` [E-Try-V]
    if t is Opt:
        if x.x is None:
            if els: raise Fail(els())
            trap("try on none")
        return x.x
    return x

def try_eval(th):
    try: return th()
    except (Trap, Fail, Stop, Ret): return None
def compound(op, cur, r):
    charge(op_cost(op, cur))
    return compound_uncharged(op, cur, r)
def compound_uncharged(op, cur, r):
    t = cur.t if type(cur) is Int else "lit"
    x = arith(op, cur, r)
    return chk(x.v, t) if type(x) is Int else x
def fixtag(cur, v):
    if type(cur) is Int and cur.t != "lit" and type(v) is Int: return chk(v.v, cur.t)
    return v
def set_field(rec, m, v):
    if type(rec) is not Rec: trap("member assignment on a non-record")
    fs = list(rec.fs)
    for i, (n, _) in enumerate(fs):
        if n == m: fs[i] = (n, v); return Rec(rec.n, fs)
    trap("no field")
def set_tuple(t, i, v):
    if type(t) is Tup and i < len(t.xs):
        xs = list(t.xs); xs[i] = v; return Tup(xs)
    trap("bad tuple assignment")
def set_index(c, k, v):
    if type(c) is Lst and type(k) is Int:
        i = k.v
        if i < 0 or i >= len(c.xs): trap("index %d out of range" % i)
        xs = list(c.xs); xs[i] = v; return Lst(xs)
    if type(c) is Mp: return Mp(map_set(c.ps, k, v))
    trap("bad indexed assignment")
def loop_bound(v): return v.v if type(v) is Int else LOOP_LIMIT
def arena_bound(v):
    if type(v) is Int: return v.v
    trap("bad arena bound")
def check_(v):
    if not truthy(v): trap("check failed")
def rec_name(v): return v.n if type(v) is Rec else None

def jesc(s):
    n = 2
    for c in s:
        if c in '"\\\n\r\t': n += 2
        elif ord(c) < 0x20: n += 6
        else: n += len(c.encode("utf-8"))
    return n
def json_size(v):
    t = type(v)
    if v is True: return 4
    if v is False: return 5
    if t is Int: return len(str(v.v))
    if t is F or t is F32: return len(fdebug(v.v))
    if t is str: return jesc(v)
    if t is UnitT: return 4
    if t is Opt: return 4 if v.x is None else json_size(v.x)
    if t is Err: return json_size(v.e)
    def seq(xs, n): return 2 + sum(xs) + max(n - 1, 0)
    if t is Tup or t is Lst or t is St: return seq([json_size(x) for x in v.xs], len(v.xs))
    if t is Mp: return seq([3 + json_size(a) + json_size(b) for a, b in v.ps], len(v.ps))
    if t is Rec: return 2 + jesc(v.n) + 1 + seq([jesc(a) + 1 + json_size(b) for a, b in v.fs], len(v.fs))
    if t is Rng: return 3 + len(str(v.a)) + len(str(v.b))
    if t is FnV: return jesc(v.n)
    return 2
def arena_check(bound, vals):
    if sum(json_size(v) for v in vals) > bound: trap("size limit")

def why(e):
    if isinstance(e, Trap): return "trap: " + e.m
    if isinstance(e, Fail): return "failed with " + show(e.v)
    if isinstance(e, Stop): return "Stop"
    return "Ret"
def dbg_v(v):
    """Rust Debug of the interpreter's V (refusal text); closures,
    iterators and lvars print a placeholder"""
    if v is True or v is False: return "Bool(%s)" % ("true" if v else "false")
    c = type(v)
    if c is Int: return "Int(%d, %s)" % (v.v, rdebug(v.t))
    if c is F: return "F(" + fdebug_full(v.v) + ")"
    if c is F32: return "F32(" + fdebug_full(v.v) + ")"
    if c is str: return "Text(" + rdebug(v) + ")"
    if c is UnitT: return "Unit"
    if c is Tup: return "Tuple([" + ", ".join(dbg_v(x) for x in v.xs) + "])"
    if c is Lst: return "List([" + ", ".join(dbg_v(x) for x in v.xs) + "])"
    if c is St: return "Set([" + ", ".join(dbg_v(x) for x in v.xs) + "])"
    if c is Mp: return "Map([" + ", ".join("(%s, %s)" % (dbg_v(a), dbg_v(b)) for a, b in v.ps) + "])"
    if c is Opt: return "Opt(None)" if v.x is None else "Opt(Some(" + dbg_v(v.x) + "))"
    if c is Rec: return "Rec(%s, [%s])" % (rdebug(v.n), ", ".join("(%s, %s)" % (rdebug(a), dbg_v(b)) for a, b in v.fs))
    if c is Err: return "Err(" + dbg_v(v.e) + ")"
    if c is FnV: return "Fn(" + rdebug(v.n) + ")"
    if c is Rng: return "Range(%d, %d, %s)" % (v.a, v.b, rdebug(v.t))
    if c is Chan: return "Chan(%s, %d)" % (rdebug(v.id), v.cap)
    if c is Task: return "Task(" + rdebug(v.id) + ")"
    if c is Actor: return "Actor(%s, %s)" % (rdebug(v.id), rdebug(v.ty))
    if c is Session: return "Session(%s, %s, %d)" % (rdebug(v.out), rdebug(v.inn), v.cap)
    if c is Clo: return "Closure(..)"
    if c is Gen: return "Gen(..)"
    return "LVar(..)"
def fdebug_full(x):
    if x != x: return "NaN"
    if x in (math.inf, -math.inf): return "inf" if x > 0 else "-inf"
    return fdebug(x)
def dbg_flow(e):
    if isinstance(e, Ret): return "Ret(" + dbg_v(e.v) + ")"
    if isinstance(e, Fail): return "Fail(" + dbg_v(e.v) + ")"
    if isinstance(e, Trap): return "Trap(" + rdebug(e.m) + ")"
    if isinstance(e, Stop): return "Stop"
    if isinstance(e, Cancel): return "Cancel(%d)" % e.f
    return "?"
def refusal(e):
    """machine step refusal (interp.rs)"""
    if isinstance(e, Fail): return v_name(e.v)
    if isinstance(e, Trap): return "trap(" + e.m + ")"
    return "trap(" + dbg_flow(e) + ")"
def actor_refusal(e):
    """actor handler refusal (conc.rs)"""
    if isinstance(e, Fail) and type(e.v) is Rec: return e.v.n
    if isinstance(e, Trap): return "trap(" + e.m + ")"
    return "trap(" + dbg_flow(e) + ")"
def v_name(v): return v.n if type(v) is Rec else "?"

def emit(v):
    if TLS.task is not None: comm(("emit", v)); return
    TLS.out.append(v)

def obs(label, th):
    fuel_reset()
    try:
        v = th()
        print(label + " = " + show(v))
    except (Trap, Fail, Stop, Ret) as e:
        print(label + " " + why(e))
    fuel_report()

def show_state(s):
    return "; ".join(r + " { " + ", ".join(n + " = " + show(v) for n, v in fs) + " }" for r, fs in s)

def run_example(line, events):
    fuel_reset()
    try: run_example1(line, events)
    finally: fuel_report()
def run_example1(line, events):
    print("example %d" % line)
    try: s = initial_state()
    except (Trap, Fail, Stop, Ret) as e:
        print("initial " + why(e)); return
    print("0 (initial) " + show_state(s))
    d = DLog()
    try: run_events(s, events, d)
    finally:
        if TRACE: print("trace " + d.prev)
def run_events(s, events, d):
    for k, (name, th) in enumerate(events, 1):
        try: ev = th()
        except (Trap, Fail, Stop, Ret) as e:
            print("%d %s %s" % (k, name, why(e))); return
        if ev is None:
            print("%d %s not an event" % (k, name)); return
        try: s2, outs = delta(s, ev)
        except Refuse as r:
            if TRACE: d.add(name, ev, s, True)
            print("%d %s refused: %s" % (k, show(ev), r.r)); return
        s = s2
        if TRACE: d.add(name, ev, s, False)
        print("%d %s %s" % (k, show(ev), show_state(s)))
        for o in outs: print("  emit " + show(o))

# ---- 6.1 the log D: canonical JSON (N3) and the SHA-256 entry chain, as interp.rs writes it
TRACE = "--trace" in sys.argv
D_GENESIS = "0" * 64
class FRec:
    __slots__ = ("fs",)
    def __init__(s, fs): s.fs = sorted(fs, key=lambda p: p[0])
class FFl:
    __slots__ = ("x",)
    def __init__(s, x): s.x = x
def form_value(v):
    if v is True or v is False: return v
    c = type(v)
    if c is Int: return v.v
    if c is F or c is F32: return FFl(v.v)
    if c is str: return v
    if c is UnitT: return None
    if c is Opt: return None if v.x is None else form_value(v.x)
    if c is Tup or c is Lst or c is St: return [form_value(x) for x in v.xs]
    if c is Mp:
        l = [(form_value(a), form_value(b)) for a, b in v.ps]
        l.sort(key=functools.cmp_to_key(lambda a, b: fv_cmp(a[0], b[0])))
        return [[a, b] for a, b in l]
    if c is Rec:
        f = [(k, form_value(x)) for k, x in v.fs]
        if v.n in VARIANTS: f.append(("$v", v.n))
        return FRec(f)
    if c is Err: return FRec([("$err", form_value(v.e))])
    if c is Rng: return list(range(v.a, v.b))
    if c is FnV: return FRec([("$fn", v.n)])
    return None
def form_value_pos(v):
    # 6.1, T7: the positional embedding hashed by D (names replaced by positions)
    c = type(v)
    if c is Opt: return None if v.x is None else form_value_pos(v.x)
    if c is Tup or c is Lst or c is St: return [form_value_pos(x) for x in v.xs]
    if c is Mp:
        l = [(form_value(a), form_value_pos(a), form_value_pos(b)) for a, b in v.ps]
        l.sort(key=functools.cmp_to_key(lambda a, b: fv_cmp(a[0], b[0])))
        return [[a, b] for _, a, b in l]
    if c is Rec:
        at = lambda f: next((form_value_pos(x) for k, x in v.fs if k == f), None)
        if v.n in DECL_FIELDS: return [at(f) for f in DECL_FIELDS[v.n]]
        if v.n in VARIANT_INDEX:
            k, fs = VARIANT_INDEX[v.n]
            return [k, [at(f) for f in fs]]
        return form_value(v)
    if c is Err: return FRec([("$err", form_value_pos(v.e))])
    return form_value(v)
def fv_rank(x):
    if x is None: return 0
    if x is True or x is False: return 1
    c = type(x)
    if c is int or c is FFl: return 2
    if c is str: return 3
    if c is list: return 4
    return 5
def fv_cmp(a, b):
    ra, rb = fv_rank(a), fv_rank(b)
    if ra != rb: return -1 if ra < rb else 1
    if ra == 1 or (ra == 2 and type(a) is int and type(b) is int) or ra == 3:
        return (a > b) - (a < b)
    if ra == 2:
        x = float(a.x if type(a) is FFl else a); y = float(b.x if type(b) is FFl else b)
        return (x > y) - (x < y)
    if ra == 4:
        for x, y in zip(a, b):
            c = fv_cmp(x, y)
            if c: return c
        return (len(a) > len(b)) - (len(a) < len(b))
    if ra == 5:
        for (ka, va), (kb, vb) in zip(a.fs, b.fs):
            if ka != kb: return -1 if ka < kb else 1
            c = fv_cmp(va, vb)
            if c: return c
        return (len(a.fs) > len(b.fs)) - (len(a.fs) < len(b.fs))
    return 0
def float_json(x):
    if x == 0.0: return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"
    import decimal
    t = decimal.Decimal(repr(abs(x))).normalize().as_tuple()
    digits = "".join(map(str, t.digits))
    ex = t.exponent + len(digits) - 1
    sign = "-" if x < 0 else ""
    a = abs(x)
    if 1e-4 <= a < 1e16:
        p = ex + 1
        if p <= 0: body = "0." + "0" * (-p) + digits
        elif p >= len(digits): body = digits + "0" * (p - len(digits)) + ".0"
        else: body = digits[:p] + "." + digits[p:]
        return sign + body
    m = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return "%s%se%d" % (sign, m, ex)
def jstr(s):
    o = ['"']
    for ch in s:
        if ch == '"': o.append('\\"')
        elif ch == "\\": o.append("\\\\")
        elif ch == "\n": o.append("\\n")
        elif ch == "\r": o.append("\\r")
        elif ch == "\t": o.append("\\t")
        elif ord(ch) < 0x20: o.append("\\u%04x" % ord(ch))
        else: o.append(ch)
    o.append('"')
    return "".join(o)
def fv_json(x):
    if x is None: return "null"
    if x is True: return "true"
    if x is False: return "false"
    c = type(x)
    if c is int: return str(x)
    if c is FFl: return float_json(x.x)
    if c is str: return jstr(x)
    if c is list: return "[" + ",".join(fv_json(y) for y in x) + "]"
    return "{" + ",".join(jstr(k) + ":" + fv_json(v) for k, v in x.fs) + "}"
def d_sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
class DLog:
    def __init__(s): s.prev = D_GENESIS; s.seq = 0
    def add(s, name, ev, st, kept):
        # one entry, printed with --trace: d applied|kept HASH BODY (HASH = SHA-256 of BODY)
        ty = str(EVENT_INDEX[name]) if name in EVENT_INDEX else jstr(name)
        evj = '{"payload":' + fv_json(form_value_pos(ev)) + ',"type":' + ty + "}"
        sj = fv_json([[form_value_pos(v) for n, v in fs] for r, fs in st])
        body = '{"event":' + evj + ',"prev_hash":"' + s.prev + '","seq":' + str(s.seq) + ',"state_hash":"' + d_sha(sj) + '"}'
        h = d_sha(body)
        print("d " + ("kept" if kept else "applied") + " " + h + " " + body)
        s.prev = h; s.seq += 1

class Refuse(Exception):
    def __init__(s, r): s.r = r

def delta(s, ev):
    nxt, outs = [], []
    for name, fs in s:
        got = None
        for st in STEPS.get(name, []):
            TLS.out.clear()
            try: r = st(fs, ev)
            except (Trap, Fail, Stop, Ret) as e: raise Refuse(refusal(e))
            if r is not None:
                got = (r, list(TLS.out)); break
        if got is None: nxt.append((name, fs))
        else:
            nxt.append((name, got[0])); outs.extend(got[1])
    return nxt, outs

def run_main(f):
    fuel_reset()
    try: run_main1(f)
    finally: fuel_report()
def run_main1(f):
    print("main")
    TLS.out.clear()
    try:
        fuel()
        v = f(None, [])[0]
        for o in TLS.out: print("emit " + show(o))
        print("main = " + show(v))
    except (Trap, Fail, Stop, Ret) as e:
        for o in TLS.out: print("emit " + show(o))
        print("main " + why(e))

# ---- draft 0.6 (the draft 0.6 concurrency chapter, not published here C.5.2, C.11.2)
def par_ident(op, b):
    if type(b) is Int:
        if op in ("+", "|", "^"): return Int(0, b.t)
        if op == "*": return Int(1, b.t)
        if op == "&":
            lo, hi = irange("i64" if b.t == "lit" else b.t)
            return Int(hi if lo == 0 else -1, b.t)
    if type(b) is F:
        if op == "+": return F(-0.0)
        if op == "*": return F(1.0)
    if type(b) is F32:
        if op == "+": return F32(-0.0)
        if op == "*": return F32(1.0)
    if b is True or b is False:
        if op == "&": return True
        if op in ("|", "^"): return False
    trap("no identity for " + op + "= on " + show(b))
def par_combine(op, a, b):
    charge(op_cost(op, a))
    if (a is True or a is False) and (b is True or b is False):
        if op == "&": return a and b
        if op == "|": return a or b
        if op == "^": return a != b
    return compound_uncharged(op, a, b)
def par_tree(op, c, i):
    if len(c) == 0: return i
    if len(c) == 1: return c[0]
    h = 1 << ((len(c) - 1).bit_length() - 1)
    l = par_tree(op, c[:h], i)
    r = par_tree(op, c[h:], i)
    return par_combine(op, l, r)
def pscan(xs, f, e, inclusive):
    n = len(xs)
    if n == 0: return Lst([])
    sums = {}
    def up(lo, hi):
        if hi - lo == 1: v = xs[lo]
        else:
            h = lo + (1 << ((hi - lo - 1).bit_length() - 1))
            l = up(lo, h); r = up(h, hi); v = apply(f, [l, r])
        sums[(lo, hi)] = v
        return v
    out = [None] * n
    def down(lo, hi, off):
        if hi - lo == 1:
            out[lo] = off; return
        h = lo + (1 << ((hi - lo - 1).bit_length() - 1))
        ls = sums[(lo, h)]
        roff = ls if off is None else apply(f, [off, ls])
        down(lo, h, off)
        down(h, hi, roff)
    up(0, n)
    down(0, n, None)
    r = []
    for i, o in enumerate(out):
        if inclusive: r.append(xs[i] if o is None else apply(f, [o, xs[i]]))
        else: r.append(e if o is None else o)
    return Lst(r)


# ---- cost accounting for `with budget` (C.6.1, C.6.4), as interp `charge`
def charge(u):
    if TLS.bud:
        for b in TLS.bud:
            if b[0] + u > b[1]: trap("budget")
        for b in TLS.bud: b[0] += u
class budget:
    def __init__(s, n):
        s.n = marker_int(n)
    def __enter__(s): TLS.bud.append([0, s.n])
    def __exit__(s, *a):
        TLS.budlast = TLS.bud.pop()[0]
        return False
def budget_last():
    return Int(getattr(TLS, "budlast", 0), "u64")
def op_cost(op, x):
    if type(x) is F or type(x) is F32:
        return 2 if op in ("+", "-", "*") else 8 if op in ("/", "%") else 1
    return 1
def cindex(c, i): charge(1); return index(c, i)
def cfield(v, m): charge(1); return field(v, m)
def ctmember(v, i): charge(1); return tmember(v, i)
def cneg(v): charge(1); return neg(v)
def cbitnot(v): charge(1); return bitnot(v)
def cnot(v): charge(1); return not_(v)
def cbinop(op, x, y): charge(op_cost(op, x)); return binop(op, x, y)
def pre_method(recv, m):
    t = type(recv)
    head = KIND_OF.get(recv.n, recv.n) if t is Rec else "List" if t is Lst else "Set" if t is St else None
    if head is not None and (head, m) in METHODS: charge(4)
    return recv
def pre_value(f):
    if type(f) is FnV: charge(4)
    return f

def marker_int(v):
    if type(v) is Int: return v.v
    trap("expected an integer, got " + show(v))

# ---- LVars (C.11.8) and the built-in Merge instances (C.11.5)
class LVar:
    __slots__ = ("v", "top", "frz")
    def __init__(s, v, top): s.v = v; s.top = top; s.frz = False
def lvar_new(a):
    return LVar(a[0] if a else UNIT, a[1] if len(a) > 1 else None)
def _crank(v):
    t = type(v)
    if t is UnitT or (t is Opt and v.x is None): return 0
    if v is True or v is False: return 1
    if t is Int or t is F or t is F32: return 2
    if t is str: return 3
    if t is Lst or t is St or t is Tup: return 4
    return 5
def _cmp(a, b): return (a > b) - (a < b)
def keq(a, b): return canon_cmp(a, b) == 0  # map/set key identity (6.1)
def canon_cmp(a, b):
    ra, rb = _crank(a), _crank(b)
    if ra != rb: return _cmp(ra, rb)
    ta, tb = type(a), type(b)
    if ra == 1: return _cmp(a, b)
    if ta is Int and tb is Int: return _cmp(a.v, b.v)
    if ra == 2:
        # int before float (never equal keys); floats by IEEE 754 totalOrder (6.1)
        if ta is Int: return -1
        if tb is Int: return 1
        x, y = float(a.v), float(b.v)
        if x == y == 0.0: return _cmp(math.copysign(1.0, x), math.copysign(1.0, y))
        return _cmp(x, y)
    if ra == 3: return _cmp(a.encode("utf-8"), b.encode("utf-8"))
    if ra == 4:
        for p, q in zip(a.xs, b.xs):
            c = canon_cmp(p, q)
            if c != 0: return c
        return _cmp(len(a.xs), len(b.xs))
    if ta is Opt and tb is Opt and a.x is not None and b.x is not None: return canon_cmp(a.x, b.x)
    if ta is Rec and tb is Rec:
        if a.n != b.n: return _cmp(a.n.encode("utf-8"), b.n.encode("utf-8"))
        for (k1, v1), (k2, v2) in zip(a.fs, b.fs):
            c = _cmp(k1.encode("utf-8"), k2.encode("utf-8")) or canon_cmp(v1, v2)
            if c != 0: return c
        return _cmp(len(a.fs), len(b.fs))
    return _cmp(show(a).encode("utf-8"), show(b).encode("utf-8"))
import functools
def merge_builtin(a, b):
    ta, tb = type(a), type(b)
    if ta is Int and tb is Int: return Int(max(a.v, b.v), b.t if a.t == "lit" else a.t)
    if (a is True or a is False) and (b is True or b is False): return a or b
    if ta is Lst and tb is Lst:
        v = sorted(a.xs + b.xs, key=functools.cmp_to_key(canon_cmp))
        out = []
        for x in v:
            if not out or canon_cmp(out[-1], x) != 0: out.append(x)
        return Lst(out)
    if ta is St and tb is St:
        v = sorted(a.xs + b.xs, key=functools.cmp_to_key(canon_cmp))
        out = []
        for x in v:
            if not out or canon_cmp(out[-1], x) != 0: out.append(x)
        return St(out)
    if ta is Mp and tb is Mp:
        out = list(a.ps)
        for k, v in b.ps:
            for i, (k2, v2) in enumerate(out):
                if keq(k2, k): out[i] = (k2, merge_builtin(v2, v)); break
            else: out.append((k, v))
        out.sort(key=functools.cmp_to_key(lambda p, q: canon_cmp(p[0], q[0])))
        return Mp(out)
    if ta is Tup and tb is Tup and len(a.xs) == len(b.xs): return Tup([merge_builtin(p, q) for p, q in zip(a.xs, b.xs)])
    if ta is Rec and tb is Rec and a.n == b.n and len(a.fs) == len(b.fs):
        return Rec(a.n, [(k, merge_builtin(p, q)) for (k, p), (_, q) in zip(a.fs, b.fs)])
    if ta is Opt and a.x is None: return b
    if tb is Opt and b.x is None: return a
    if ta is Opt and tb is Opt: return Opt(merge_builtin(a.x, b.x))
    trap("no Merge instance for %s and %s" % (show(a), show(b)))
def map_merge(xs, ys):
    out = list(xs)
    for k, v in ys:
        for i, (k2, v2) in enumerate(out):
            if keq(k2, k): out[i] = (k2, merge_values(v2, v)); break
        else: out.append((k, v))
    out.sort(key=functools.cmp_to_key(lambda p, q: canon_cmp(p[0], q[0])))
    return Mp(out)
def set_of(v): return St(merge_builtin(v, Lst([])).xs)
def merge_values(a, b):
    if type(a) is Rec:
        h = KIND_OF.get(a.n, a.n)
        if (h, "merge") in METHODS:
            charge(4)
            return METHODS[(h, "merge")](a, [b])[0]
    return merge_builtin(a, b)
def lvar_put(c, v):
    if TLS.parcx is not None: return lvar_put_t(c, v)
    nv = merge_values(c.v, v)
    if c.top is not None and eq(nv, c.top): trap("lvar conflict")
    if not eq(nv, c.v):
        if c.frz: trap("lvar put after freeze")
        TLS.puts[0] += 1
    c.v = nv
def lvar_freeze(c):
    with _LVLOCK:
        c.frz = True
        return c.v
# merge for a threshold test: the program's Merge, charged nothing (interp `merge_free`)
def merge_free(a, b):
    bud, f = TLS.bud, TLS.fuel
    TLS.bud = []; TLS.fuel = FUEL_LIMIT
    try: return merge_values(a, b)
    finally: TLS.bud = bud; TLS.fuel = f
def lvar_above(ths, cur):
    for t in ths:
        if eq(merge_free(t, cur), cur): return t
    return None
def lvar_incomp(c, ths):
    for i, a in enumerate(ths):
        for b in ths[i + 1:]:
            if eq(a, b): continue
            if c.top is None or not eq(merge_free(a, b), c.top): trap("lvar thresholds are not pairwise incompatible")
def puts(): return TLS.puts[0]
def lvar_wait(c, q):
    if type(q) is not Lst: trap("wait needs a list of thresholds")
    lvar_incomp(c, q.xs)
    if TLS.parcx is not None: return lvar_wait_t(c, q)
    t = lvar_above(q.xs, c.v)
    if t is not None: return t
    if TLS.sched[0] > 0: trap("$block")
    trap("lvar wait cannot be satisfied")
def par_enter(): TLS.sched[0] += 1
def par_exit(): TLS.sched[0] -= 1
def bud_snap(): return [list(b) for b in TLS.bud]
def bud_restore(s): TLS.bud[:] = s
# a restarted par attempt (C.14) restores both the budgets and the fuel, so an
# abandoned attempt costs nothing, as in the reference's threaded run
def par_snap(): return (bud_snap(), TLS.fuel)
def par_restore(s):
    bud_restore(s[0])
    TLS.fuel = s[1]

# ---- workgroups (C.11.2)
def lanes_of(v):
    n = marker_int(v)
    return max(n, 0), n
def group_rec(lane, n): return Rec("$group", [("lane", Int(lane, "usize")), ("lanes", Int(n, "usize"))])


# ---- par on OS threads (interp `par_exec`, Threads mode); with the
# free-threaded CPython (3.13t / 3.14t) the branches run in parallel
class ParCx:
    def __init__(s, k, parent):
        s.cv = threading.Condition(); s.running = k; s.idle = 0; s.ver = 0; s.stuck = False; s.parent = parent
    def bump(s):
        with s.cv:
            s.ver += 1; s.idle = 0; s.cv.notify_all()
        if s.parent is not None: s.parent.bump()
_LVLOCK = threading.Lock()
def lvar_put_t(c, v):
    while True:
        cur = c.v
        nv = merge_values(cur, v)
        with _LVLOCK:
            if c.v is not cur: continue
            if c.top is not None and eq(nv, c.top): trap("lvar conflict")
            ch = not eq(nv, cur)
            if ch and c.frz: trap("lvar put after freeze")
            c.v = nv
        break
    if ch:
        TLS.puts[0] += 1
        if TLS.parcx is not None: TLS.parcx.bump()
def lvar_wait_t(c, q):
    cx = TLS.parcx
    ths = q.xs
    def above(cur): return lvar_above(ths, cur)
    with cx.cv:
        seen = None
        while True:
            if cx.stuck: trap("lvar wait cannot be satisfied")
            if seen != cx.ver:
                t = above(c.v)
                if t is not None: return t
                seen = cx.ver; cx.idle += 1
            if cx.idle >= cx.running:
                cx.stuck = True; cx.cv.notify_all(); trap("lvar wait cannot be satisfied")
            cx.cv.wait()
def par_threads(k, branches):
    """run k branch functions on threads; returns [(exc or None, cost, writes)]"""
    cx = ParCx(k, TLS.parcx)
    bud = [list(b) for b in TLS.bud]
    f0 = TLS.fuel
    out = [None] * k
    def run(i):
        TLS.parcx = cx
        TLS.fuel = f0
        TLS.bud = [list(b) for b in bud]
        s0 = bud[0][0] if bud else 0
        e, w = None, None
        try: w = branches[i]()
        except (Trap, Fail, Stop, Ret, Cancel) as x: e = x
        s1 = TLS.bud[0][0] if TLS.bud else 0
        with cx.cv:
            cx.running -= 1; cx.cv.notify_all()
        out[i] = (e, s1 - s0, w, f0 - TLS.fuel, f0)
    ts = [threading.Thread(target=run, args=(i,)) for i in range(k)]
    for t in ts: t.start()
    for t in ts: t.join()
    return out
def par_chunks(n):
    """`for ... par`: how many chunks (threads) for n items"""
    import os
    w = int(os.environ.get("RCE_PAR_THREADS") or (os.cpu_count() or 4))
    return max(1, min(n, max(1, w)))
def par_split(items, k):
    n = len(items)
    return [items[i * n // k:(i + 1) * n // k] for i in range(k)]
def par_resolve(outs):
    rem = min((l - s for s, l in TLS.bud), default=None)
    cum = 0
    fcum = 0
    for e, c, w, u, f0 in outs:
        fcum += u
        TLS.fuel = max(0, f0 - fcum)
        if fcum > f0: trap("comptime fuel exhausted")
        if rem is not None and ((type(e) is Trap and e.m == "budget") or cum + c > rem): trap("budget")
        if e is not None: raise e
        cum += c
    for b in TLS.bud: b[0] += cum
    return [o[2] for o in outs]
def merge3(base, cur, new):
    if eq(new, base): return cur
    if eq(cur, base): return new
    if type(base) is Rec and type(cur) is Rec and type(new) is Rec and base.n == cur.n == new.n and len(base.fs) == len(cur.fs) == len(new.fs):
        return Rec(new.n, [(k, merge3(b, c, n)) for (k, b), (_, c), (_, n) in zip(base.fs, cur.fs, new.fs)])
    for T in (Tup, Lst):
        if type(base) is T and type(cur) is T and type(new) is T and len(base.xs) == len(cur.xs) == len(new.xs):
            return T([merge3(b, c, n) for b, c, n in zip(base.xs, cur.xs, new.xs)])
    return new

# ---- the tick runtime for `--run` (6.8, chapter C), as the Rust runtime:
# one thread per task, a scheduler on the main thread admitting moves tick
# by tick (staged delivery, held sends, select rotation, seeded nondet,
# deadlines, joins, scopes, actors, atomic, partitions, supervisors).
class Chan:
    __slots__ = ("id", "cap")
    def __init__(s, i, c): s.id = i; s.cap = c
class Task:
    __slots__ = ("id",)
    def __init__(s, i): s.id = i
class Actor:
    __slots__ = ("id", "ty")
    def __init__(s, i, t): s.id = i; s.ty = t
class Session:
    __slots__ = ("out", "inn", "cap")
    def __init__(s, o, i, c): s.out = o; s.inn = i; s.cap = c
CANCEL_TASK = 2**64 - 1
def hid(key): return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
class TaskCx:
    def __init__(s, tid, req, rep):
        s.id = tid; s.req = req; s.rep = rep; s.created = 0; s.ords = {}; s.now = 0; s.dl = []; s.next_frame = 1; s.scopes = [0]; s.sel_k = {}
def comm(op):
    cx = TLS.task
    if cx is None: trap("concurrency is not available at comptime")
    cx.req.put((op, list(cx.dl)))
    tick, rep = cx.rep.get()
    cx.now = tick
    if rep[0] == "cancel": raise Cancel(rep[1])
    if rep[0] == "failed": raise Fail(rep[1])
    return rep
def _fresh(kind, line):
    cx = TLS.task; idx = cx.created; cx.created += 1
    return hid("%s|%d|%s|%d" % (kind, line, cx.id, idx))
def chan_new(cap, line):
    if TLS.task is None: trap("channels are not available at comptime")
    return Chan(_fresh("chan", line), marker_int(cap()))
def session_new(cap, line):
    if TLS.task is None: trap("channels are not available at comptime")
    n = marker_int(cap())
    a = _fresh("chan", line); b = _fresh("chan", line)
    return Tup([Session(a, b, n), Session(b, a, n)])
def _ord(cx, key):
    k = cx.ords.get(key, 0); cx.ords[key] = k + 1
    return k
def actor_new(ty, pos):
    # stable path of C.11.11: supervisor position (or type), creating task, ordinal
    if TLS.task is None: trap("cannot call %s at comptime" % ty)
    cx = TLS.task; cx.created += 1
    return Actor(hid("actor|%s|%s|%d" % (pos, cx.id, _ord(cx, "actor|" + pos))), ty)
def send(t, v):
    charge(8)
    if type(t) is Session: t = Chan(t.out, t.cap)
    if TLS.atomic is not None: TLS.atomic.append((t, v)); return
    if TLS.hsends is not None: TLS.hsends.append((t, v)); return
    comm(("send", t, v))
def recv(c):
    charge(8)
    session = type(c) is Session
    if session: c = Chan(c.inn, c.cap)
    rp = comm(("recv", c))
    if rp[0] != "val": trap("bad receive")
    v = rp[1]
    if session:
        if type(v) is Opt and v.x is not None: return v.x
        trap("session closed")
    return v
def session_choose(r, label):
    if type(r) is not Session: trap("%s has no method choose at comptime" % show(r))
    charge(8)
    comm(("send", Chan(r.out, r.cap), Rec(label, [])))
    return UNIT
def spawn_(f, args, prio, site):
    charge(32)
    cx = TLS.task
    if cx is None: trap("concurrency is not available at comptime")
    cx.created += 1; idx = _ord(cx, "spawn")
    rp = comm(("spawn", f, args, max(0, min(255, prio)), site, idx, cx.scopes[-1] if cx.scopes else 0))
    return Task(rp[1])
def select_(arms, line, nondet):
    cx = TLS.task
    k = 0
    if cx is not None:
        k = cx.sel_k.get(line, 0); cx.sel_k[line] = k + 1
    a2 = []
    for a in arms:
        if a[0] == "recv" and type(a[1]) is Session: a = ("recv", Chan(a[1].inn, a[1].cap), a[2])
        elif a[0] == "send" and type(a[1]) is Session: a = ("send", Chan(a[1].out, a[1].cap), a[2])
        a2.append(a)
    rp = comm(("select", a2, (line, k), nondet))
    return rp[1], (NONE if rp[2] is None else rp[2])
def sarm_after(n): return ("after", max(marker_int(n), 0))
def dl_enter(n):
    cx = TLS.task
    if cx is None: return CANCEL_TASK - 1
    f = cx.next_frame; cx.next_frame += 1
    cx.dl.append((f, cx.now + max(n, 0)))
    return f
def dl_exit():
    if TLS.task is not None: TLS.task.dl.pop()
def scope_enter():
    cx = TLS.task
    if cx is None: return 0
    f = cx.next_frame; cx.next_frame += 1; cx.scopes.append(f); return f
def scope_exit():
    if TLS.task is not None: TLS.task.scopes.pop()
def await_scope(sid):
    if TLS.task is None: return
    comm(("await", sid))
def atomic_enter():
    o = TLS.atomic; TLS.atomic = []; return o
def atomic_restore(o):
    buf = TLS.atomic or []; TLS.atomic = o; return buf
def atomic_commit(buf):
    if TLS.atomic is not None: TLS.atomic.extend(buf); return
    if TLS.task is not None: comm(("atomic", buf)); return
    trap("concurrency is not available at comptime")

class _ChanSt:
    def __init__(s, cap): s.cap = cap; s.buf = []; s.held = []; s.closed = False
class _Slot:
    def __init__(s, tid, prio, parent, scope):
        s.id = tid; s.prio = prio; s.req = queue.Queue(); s.rep = queue.Queue(); s.st = ("ready", ("go",)); s.parent = parent; s.scope = scope
        s.result = None; s.joined = False; s.cancel = False; s.holding = None; s.released = False; s.since = 0
class _ActorSt:
    def __init__(s, ty, fields, node): s.ty = ty; s.fields = fields; s.mailbox = []; s.node = node
def _task_body(slot, f, args):
    first = slot.rep.get()
    TLS.task = TaskCx(slot.id, slot.req, slot.rep)
    r = None
    if first[1][0] == "go":
        TLS.task.now = first[0]
        try:
            for _ in range(len(args) + 1): fuel()
            v = FNS[f](None, args)[0]
            await_scope(0)
            r = ("ok", v)
        except (Trap, Fail, Stop, Ret, Cancel) as e: r = ("err", e)
    else: r = ("err", Cancel(CANCEL_TASK))
    slot.req.put((("done", r), []))
class Sched:
    def __init__(s, links):
        s.slots = []; s.chans = {}; s.staged = []; s.outputs = []; s.rng = 0x5eed | 1; s.tick = 0; s.stop = None; s.max_ticks = 100000
        s.actors = {}; s.mail_staged = []; s.txns = []; s.txn_staged = []; s.held_links = []; s.parted = []; s.links = links; s.restarts = {}; s.alog = []; s.rlog = []; s.rrep = restart_inputs(); s.events = event_inputs()
    def spawn_task(s, f, args, tid, prio, parent, scope):
        sl = _Slot(tid, prio, parent, scope)
        th = threading.Thread(target=_task_body, args=(sl, f, args), daemon=True)
        th.start()
        s.slots.append(sl)
    def order(s):
        v = [i for i, sl in enumerate(s.slots) if sl.st[0] != "done"]
        v.sort(key=lambda i: (-s.slots[i].prio, s.slots[i].id))
        return v
    def chan(s, v):
        if type(v) is Chan:
            if v.id not in s.chans: s.chans[v.id] = _ChanSt(v.cap)
            return v.id
        return None
    def space(s, c):
        st = s.chans[c]
        return len(st.buf) + sum(1 for x, _ in s.staged if x == c) < st.cap
    def take(s, c):
        st = s.chans.get(c)
        if st is None: return None
        if st.buf:
            m = st.buf.pop(0)
            if st.held:
                h, o = st.held.pop(0); st.buf.append(h)
                if o is not None: s.slots[o].released = True
            return (m,)
        if st.held:
            h, o = st.held.pop(0)
            if o is not None: s.slots[o].released = True
            return (h,)
        return None
    def readable(s, c):
        st = s.chans[c]; has = bool(st.buf) or bool(st.held)
        return has, (not has and st.closed)
    def route(s, frm, target, m):
        to = NODE_OF.get(target.ty, 0) if type(target) is Actor else 0
        key = (min(frm, to), max(frm, to))
        if frm != to and key in s.parted: s.held_links.append((frm, to, target, m)); return
        if type(target) is Actor: s.mail_staged.append((target.id, target.ty, m))
        elif type(target) is Chan:
            c = s.chan(target)
            if s.space(c): s.staged.append((c, m))
            else: s.chans[c].held.append((m, None))
    def ensure_actor(s, aid, ty):
        if aid not in s.actors: s.actors[aid] = _ActorSt(ty, actor_init(ty), NODE_OF.get(ty, 0))
    def handler(s, aid, msg):
        st = s.actors[aid]
        for h in ACTOR_HANDLERS.get(st.ty, []):
            TLS.out = []; TLS.hsends = []
            try: r = h(st.fields, msg)
            except (Trap, Fail, Stop, Ret, Cancel) as e:
                TLS.hsends = None; TLS.out = []
                return ("err", actor_refusal(e))
            sends = TLS.hsends; TLS.hsends = None
            if r is None: continue
            outs = TLS.out; TLS.out = []
            return ("ok", (r, outs, sends))
        return ("ok", None)
    def on_crash(s, aid, reason):
        # restart as input (C.11.11)
        s.alog.append("Refused(%s)" % reason)
        ty = s.actors[aid].ty
        sv = SUPERVISORS.get(ty)
        if sv is None: return
        sup, strat, tys, child, bound = sv
        tick = s.tick
        if s.rrep is not None:
            if s.rrep and (s.rrep[0][1] + "/" + s.rrep[0][2]) not in SUP_TREE:
                s.stop = ("refused", "trap: upgrade removed supervisor position %s/%s" % (s.rrep[0][1], s.rrep[0][2])); return
            if not s.rrep or s.rrep[0][3] != aid or s.rrep[0][1] != sup:
                s.stop = ("refused", "replay diverged"); return
            dec = s.rrep.pop(0)
        else:
            h = [t for t in s.restarts.get((sup, child), []) if RESTART_WINDOW == 0 or tick - t < RESTART_WINDOW]
            h.append(tick); s.restarts[(sup, child)] = h
            stop = len(h) > bound
            pos = tys.index(ty) if ty in tys else 0
            reset = []
            if not stop:
                for a2, a in sorted(s.actors.items()):
                    if strat == "one_for_all": hit = a.ty in tys
                    elif strat == "rest_for_one": hit = a.ty in tys and tys.index(a.ty) >= pos
                    else: hit = a2 == aid
                    if hit: reset.append(a2)
            dec = (tick, sup, child, aid, stop, reset)
        s.rlog.append(dec)
        c = sum(1 for r in s.rlog if r[1] == sup and r[2] == child and (RESTART_WINDOW == 0 or tick - r[0] < RESTART_WINDOW))
        if dec[4]:
            s.alog.append("Stopped(supervisor %s)" % sup)
            s.stop = ("refused", "supervisor %s stopped: child %s restarted more than %d times" % (sup, child, bound)); return
        s.alog.append("Restart(%s, %s, %s, tick %d, %d)" % (child, aid, reason, tick, c))
        for a2 in dec[5]:
            if a2 in s.actors: s.actors[a2].fields = actor_init(s.actors[a2].ty)
    def commit_handler(s, aid, r):
        if r is None: return
        fs, outs, sends = r
        node = s.actors[aid].node
        s.actors[aid].fields = fs
        s.outputs.extend(outs)
        for t, v in sends: s.route(node, t, v)
    def actors_phase(s):
        progress = False; busy = set()
        txns, s.txns = s.txns, []
        for txn in txns:
            progress = True; results = []; refused = None
            for aid, ty, m in txn:
                s.ensure_actor(aid, ty); busy.add(aid)
                r = s.handler(aid, m)
                if r[0] == "err": refused = r[1]; break
                results.append((aid, r[1]))
            if refused is not None:
                failed = len(results)
                for k, (aid, _, _) in enumerate(txn):
                    if k == failed: s.on_crash(aid, refused)
                    else: s.alog.append("Refused(%s)" % refused)
                if s.stop is not None: break
                continue
            for aid, r in results: s.commit_handler(aid, r)
        for aid in sorted(s.actors):
            if aid in busy: continue
            a = s.actors[aid]
            if not a.mailbox: continue
            m = a.mailbox.pop(0); progress = True
            r = s.handler(aid, m)
            if r[0] == "ok": s.commit_handler(aid, r[1])
            else: s.on_crash(aid, r[1])
            if s.stop is not None: break
        return progress
    def events_phase(s):
        # HC-6: an event at this tick is logged as Input and staged to every
        # live actor that handles it, in actor id order; else Unhandled
        evs = [e[1] for e in s.events if e[0] == s.tick]
        for v in evs:
            s.alog.append("Input(%s)" % show(v))
            to = [(aid, a.ty) for aid, a in sorted(s.actors.items()) if v.n in ACTOR_EVENTS.get(a.ty, [])]
            if not to: s.alog.append("Unhandled(%s)" % show(v))
            for aid, ty in to: s.mail_staged.append((aid, ty, v))
        return bool(evs)
    def links_phase(s):
        anyl = False
        for _, part, a, b in [e for e in s.links if e[0] == s.tick]:
            if a not in NODE_NAMES or b not in NODE_NAMES: continue
            x, y = NODE_NAMES.index(a), NODE_NAMES.index(b)
            anyl = True; key = (min(x, y), max(x, y))
            if part:
                if key not in s.parted: s.parted.append(key)
            else:
                s.parted = [k for k in s.parted if k != key]
                held, s.held_links = s.held_links, []
                for f, t, target, m in held:
                    if (min(f, t), max(f, t)) == key: s.route(f, target, m)
                    else: s.held_links.append((f, t, target, m))
        return anyl
    def failed_value(s, i):
        r = s.slots[i].result
        reason = r[1] if r is not None and r[0] == "err" else ""
        return Rec("TaskFailed", [("task", s.slots[i].id), ("reason", reason)])
    def pick_select(s, arms, key, nondet):
        ready = []
        for k, a in enumerate(arms):
            ok = False
            if a[0] == "recv":
                c = s.chan(a[1])
                if c is not None:
                    has, closed = s.readable(c)
                    ok = (has and a[2] != 2) or (closed and a[2] != 1)
            elif a[0] == "send":
                c = s.chan(a[1])
                if c is not None: ok = s.space(c) and not s.chans[c].closed
            if ok: ready.append(k)
        if not ready: return None
        if not nondet:
            n = len(arms); st = key[1] % n
            for j in range(n):
                if (st + j) % n in ready: return (st + j) % n
        M = 2**64 - 1
        s.rng ^= (s.rng << 13) & M; s.rng ^= s.rng >> 7; s.rng ^= (s.rng << 17) & M
        return ready[s.rng % len(ready)]
    def handle(s, i, req):
        op, dl = req
        sl = s.slots[i]
        for fid, e in dl:
            if s.tick > e:
                if sl.holding is not None:
                    c = sl.holding; sl.holding = None
                    if c in s.chans: s.chans[c].held = [x for x in s.chans[c].held if x[1] != i]
                sl.released = False; sl.since = 0
                return ("cancel", fid)
        if sl.cancel: return ("cancel", CANCEL_TASK)
        k = op[0]
        if k == "emit": s.outputs.append(op[1]); return ("val", UNIT)
        if k == "send":
            target, v = op[1], op[2]
            if type(target) is Actor: s.route(0, target, v); return ("val", UNIT)
            c = s.chan(target)
            if c is None: return None
            if sl.holding is not None:
                if sl.released: sl.released = False; sl.holding = None; return ("val", UNIT)
                return None
            if s.chans[c].closed:
                s.stop = ("refused", "trap: send on a closed channel"); return ("cancel", CANCEL_TASK)
            if s.space(c): s.staged.append((c, v)); return ("val", UNIT)
            s.chans[c].held.append((v, i)); sl.holding = c
            return None
        if k == "recv":
            c = s.chan(op[1])
            if c is None: return None
            has, closed = s.readable(c)
            if has: return ("val", Opt(s.take(c)[0]))
            if closed: return ("val", NONE)
            return None
        if k == "close":
            c = s.chan(op[1])
            if c is None: return None
            s.chans[c].closed = True; return ("val", UNIT)
        if k == "select":
            arms, key, nondet = op[1], op[2], op[3]
            a = s.pick_select(arms, key, nondet)
            if a is not None:
                sl.since = 0
                arm = arms[a]
                if arm[0] == "recv":
                    m = s.take(s.chan(arm[1]))
                    return ("arm", a, NONE if m is None else Opt(m[0]))
                if arm[0] == "send":
                    s.staged.append((s.chan(arm[1]), arm[2])); return ("arm", a, None)
                return ("arm", a, None)
            for j, arm in enumerate(arms):
                if arm[0] == "default": return ("arm", j, None)
            for j, arm in enumerate(arms):
                if arm[0] == "after":
                    if sl.since == 0: sl.since = s.tick
                    if s.tick - sl.since >= arm[1]: sl.since = 0; return ("arm", j, None)
                    break
            return None
        if k == "spawn": return ("spawned", hid("spawn|%s|%d" % (sl.id, op[5])))
        if k == "join":
            j = next((x for x, t in enumerate(s.slots) if t.id == op[1]), None)
            if j is None or s.slots[j].st[0] != "done": return None
            s.slots[j].joined = True
            r = s.slots[j].result
            return ("val", r[1] if r[0] == "ok" else Err(s.failed_value(j)))
        if k == "await":
            kids = [x for x, t in enumerate(s.slots) if t.parent == i and t.scope == op[1]]
            if any(s.slots[x].st[0] != "done" for x in kids): return None
            for x in kids:
                if s.slots[x].result[0] == "err" and not s.slots[x].joined:
                    s.slots[x].joined = True
                    return ("failed", s.failed_value(x))
            return ("val", UNIT)
        if k == "atomic":
            sends = op[1]
            chans = [c for c in (s.chan(t) for t, _ in sends) if c is not None]
            for c in chans:
                need = chans.count(c); st = s.chans[c]
                staged = sum(1 for x, _ in s.staged if x == c)
                if st.closed or len(st.buf) + staged + need > st.cap: return None
            txn = []
            for t, v in sends:
                if type(t) is Actor: txn.append((t.id, t.ty, v))
                else: s.staged.append((s.chan(t), v))
            if txn: s.txn_staged.append(txn)
            return ("val", UNIT)
        return None
    def finish_task(s, i, r):
        if r[0] == "ok": res = r
        elif type(r[1]) is Cancel: res = ("err", "cancelled")
        else: res = ("err", why(r[1]))
        failed = r[0] == "err" and type(r[1]) is not Cancel
        sl = s.slots[i]; sl.result = res
        if i == 0:
            if res[0] == "err": return ("refused", res[1])
        elif failed:
            for k2, t in enumerate(s.slots):
                if k2 != i and t.parent == sl.parent and t.scope == sl.scope: t.cancel = True
        for t in s.slots:
            if t.parent == i: t.cancel = True
        return None
    def run(s):
        if s.stop is not None: return s.stop
        while True:
            s.tick += 1
            if s.tick > s.max_ticks: return ("refused", "trap: tick limit")
            progress = s.links_phase()
            progress = s.events_phase() or progress
            order = s.order()
            moved = []
            for i in order:
                sl = s.slots[i]
                if sl.st[0] == "ready":
                    rep = sl.st[1]; sl.st = ("done",)
                    sl.rep.put((s.tick, rep)); moved.append(i)
            for i in moved:
                progress = True
                s.slots[i].st = ("pending", s.slots[i].req.get())
            progress = s.actors_phase() or progress
            if s.stop is not None: return s.stop
            for i in order:
                sl = s.slots[i]
                if sl.st[0] != "pending": continue
                req = sl.st[1]; sl.st = ("done",)
                if req[0][0] == "done":
                    progress = True
                    o = s.finish_task(i, req[0][1])
                    if o is not None: return o
                    continue
                rep = s.handle(i, req)
                if s.stop is not None: return s.stop
                if rep is None: sl.st = ("pending", req); continue
                progress = True
                if req[0][0] == "spawn" and rep[0] == "spawned":
                    _, f, args, prio, site, idx, scope = req[0]
                    if f not in FNS: return ("refused", "trap: no task function " + f)
                    s.spawn_task(f, args, rep[1], prio, i, scope)
                sl.st = ("ready", rep)
            for c, m in s.staged: s.chans[c].buf.append(m)
            s.staged = []
            for aid, ty, m in s.mail_staged:
                s.ensure_actor(aid, ty); s.actors[aid].mailbox.append(m)
            s.mail_staged = []
            s.txns.extend(s.txn_staged); s.txn_staged = []
            all_done = all(sl.st[0] == "done" for sl in s.slots)
            idle = all(not a.mailbox for a in s.actors.values()) and not s.txns and not s.mail_staged
            fev = any(e[0] > s.tick for e in s.events)
            future = any(e[0] > s.tick for e in s.links) or fev
            if all_done and idle and not fev and (not s.held_links or not future): return ("done",)
            if not progress and idle and not future:
                timers = any(sl.st[0] == "pending" and (sl.st[1][1] or (sl.st[1][0][0] == "select" and any(a[0] == "after" for a in sl.st[1][0][1]))) for sl in s.slots)
                if not timers: return ("deadlock",)
def actor_init(ty):
    f = ACTOR_INIT.get(ty)
    return f() if f is not None else []
def run_tasks(main):
    links = []
    for a in sys.argv:
        if a.startswith("--link="):
            p = a[7:].split(",")
            links.append((int(p[0]), p[1] == "partition", p[2], p[3]))
    s = Sched(links)
    rec = restart_tree_arg()
    if s.rrep is not None and rec is not None:
        e = tree_migrates(rec, SUP_TREE)
        if e is not None: s.stop = ("refused", e)
    s.spawn_task(main, [], hid("main"), 0, None, 0)
    o = s.run()
    for sl in s.slots:
        while sl.st[0] != "done":
            if sl.st[0] == "pending" and sl.st[1][0][0] == "done": sl.st = ("done",); continue
            sl.rep.put((s.tick, ("cancel", CANCEL_TASK)))
            r = sl.req.get()
            sl.st = ("done",) if r[0][0] == "done" else ("pending", r)
    for v in s.outputs: print("emit " + show(v))
    if "--actor-log" in sys.argv:
        for e in s.alog: print("actorlog " + e)
    if "--restart-log" in sys.argv:
        print("restartlog v1 tree=" + ",".join(SUP_TREE))
        for r in s.rlog: print("restartin %d|%s|%s|%s|%d|%s" % (r[0], r[1], r[2], r[3], 1 if r[4] else 0, ",".join(r[5])))
    if o[0] == "done": print("outcome Done")
    elif o[0] == "deadlock": print("outcome Deadlock")
    else: print("outcome Refused: " + o[1])


def restart_inputs():
    # logged restart inputs (C.11.11): --restart=TICK|SUP|CHILD|ID|STOP|RESET,...
    v = []
    for a in sys.argv:
        if a.startswith("--restart="):
            p = a[len("--restart="):].split("|")
            v.append((int(p[0]), p[1], p[2], p[3], p[4] == "1", [x for x in (p[5] if len(p) > 5 else "").split(",") if x]))
    if not v and "--restart-replay" not in sys.argv: return None
    return v

def restart_tree_arg():
    # --restart-tree=Sup/child,...: the schema of the replayed restart log
    for a in sys.argv:
        if a.startswith("--restart-tree="): return [x for x in a[len("--restart-tree="):].split(",") if x]
    return None

def _group(v):
    g = []
    for p in v:
        sup, _, c = p.partition("/")
        if g and g[-1][0] == sup: g[-1][1].append(c)
        else: g.append((sup, [c]))
    return g
def tree_migrates(rec, cur):
    # migration rule (C.11.11): every recorded position keeps its supervisor
    # and child; new ones only after the recorded ones (conc::tree_migrates)
    r, c = _group(rec), _group(cur)
    for k, (sup, kids) in enumerate(r):
        for j, kid in enumerate(kids):
            if not (k < len(c) and c[k][0] == sup and j < len(c[k][1]) and c[k][1][j] == kid):
                p = sup + "/" + kid
                return ("trap: upgrade reordered supervisor position " if p in cur else "trap: upgrade removed supervisor position ") + p
    return None

INT_TYS = ("i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "usize")
def event_inputs():
    # program events (HC-6): --event=TICK|Name|field=value,...
    v = []
    for a in sys.argv:
        if not a.startswith("--event="): continue
        r = a[len("--event="):]
        def bad(m):
            sys.stderr.write("bad --event %s: %s\n" % (r, m)); sys.exit(2)
        p = r.split("|", 2)
        try: tick = int(p[0])
        except ValueError: bad("tick")
        if len(p) < 2: bad("name")
        name = p[1]
        if name not in EVENT_FIELDS: bad("no event " + name)
        given = dict(y.split("=", 1) for y in (p[2] if len(p) > 2 else "").split(",") if "=" in y)
        fs = []
        for f, t in EVENT_FIELDS[name]:
            if f not in given: bad("missing field " + f)
            x = given[f]
            if t == "bool": val = x == "true"
            elif t == "text": val = x
            elif t in INT_TYS:
                try: val = Int(int(x), t)
                except ValueError: bad("field " + f)
            else: bad("field %s has type %s, not an integer, bool or text" % (f, t))
            fs.append((f, val))
        v.append((tick, Rec(name, fs)))
    return v

def size_ok(v):
    """N9: a list or text built by any operation has at most 1,000,000 elements."""
    if isinstance(v, Lst) and len(v.xs) > 1000000: trap("size limit")
    if isinstance(v, str) and len(v) > 1000000: trap("size limit")
    return v
