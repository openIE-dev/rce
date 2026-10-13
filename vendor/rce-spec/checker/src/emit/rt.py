# RCE runtime for emitted Python (spec 10.5). It mirrors the reference
# evaluator value by value: typed integers with range traps, f32 rounded
# once from f64, values copied on assignment, traps as exceptions.
import math, struct, sys
sys.setrecursionlimit(1000000)

class Trap(Exception):
    def __init__(s, m): s.m = m
class Fail(Exception):
    def __init__(s, v): s.v = v
class Stop(Exception): pass
class Ret(Exception):
    def __init__(s, v): s.v = v

def trap(m): raise Trap(m)

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
    d, p = _digits(repr(abs(x)))
    return _plain(x < 0, d, p)
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
        elif k < 0x20 or 0x7f <= k <= 0x9f or 0x300 <= k <= 0x36f: o.append("\\u{%x}" % k)
        else: o.append(c)
    o.append('"')
    return "".join(o)

def show(v):
    if v is True: return "true"
    if v is False: return "false"
    c = type(v)
    if c is Int: return str(v.v)
    if c is F: return fdisplay(v.v)
    if c is str: return rdebug(v)
    if c is UnitT: return "()"
    if c is Tup: return "(" + ", ".join(show(x) for x in v.xs) + ")"
    if c is Lst: return "[" + ", ".join(show(x) for x in v.xs) + "]"
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
    return "?"

def eq(a, b):
    ca, cb = type(a), type(b)
    if ca is bool or cb is bool: return ca is cb and a == b
    if ca is not cb: return False
    if ca is Int or ca is F or ca is F32: return a.v == b.v
    if ca is str: return a == b
    if ca is UnitT: return True
    if ca is Tup or ca is Lst: return len(a.xs) == len(b.xs) and all(eq(x, y) for x, y in zip(a.xs, b.xs))
    if ca is Mp:
        return len(a.ps) == len(b.ps) and all(any(eq(k, k2) and eq(v, v2) for k2, v2 in b.ps) for k, v in a.ps)
    if ca is Opt:
        if a.x is None or b.x is None: return a.x is None and b.x is None
        return eq(a.x, b.x)
    if ca is Rec: return a.n == b.n and len(a.fs) == len(b.fs) and all(p == q and eq(x, y) for (p, x), (q, y) in zip(a.fs, b.fs))
    if ca is Err: return eq(a.e, b.e)
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
    if (ca is Tup and cb is Tup) or (ca is Lst and cb is Lst):
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

def arith(op, a, b):
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
    if t is Mp: return any(eq(k, x) for k, _ in c.ps)
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
        if eq(a, k): out[i] = (a, v); return out
    out.append((k, v)); return out
def mk_map(pairs):
    ps = []
    for k, v in pairs: ps = map_set(ps, k, v)
    return Mp(ps)
def list_repeat(x, n):
    if type(n) is Int and 0 <= n.v <= 10000000: return Lst([x] * n.v)
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
            if eq(a, i): return v
        trap("missing key " + show(i))
    trap("cannot index " + show(c))

def slice_(c, ta, tb):
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
    if t is F32 and n is not None: return chk(sat128(int(v.v)) if math.isfinite(v.v) else 0, n)
    if t is Int and n is not None: return chk(v.v, n)
    if t is F and n is not None:
        if not math.isfinite(v.v): trap("non-finite float")
        return chk(sat128(int(v.v)), n)
    if (v is True or v is False) and n is not None: return chk(int(v), n)
    if t is Int and n is None: return F(float(v.v))
    return v

def fit(v, T):
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
    return Int(n, "i64")
def fround(x):
    r = float(math.trunc(x))
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
    if type(f) is FnV: return FNS[f.n](None, vals)[0]
    if type(f) is Clo: return f.f(vals)
    trap(show(f) + " is not callable")

def builtin(r, m, a):
    t = type(r)
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
        if m == "floor": return F(float(math.floor(x))), None
        if m == "ceil": return F(float(math.ceil(x))), None
        if m == "round": return F(fround(x)), None
        if m == "tostr": return fdisplay(x), None
    if (r is True or r is False) and m == "tostr": return ("true" if r else "false"), None
    if t is str and m == "len": return Int(len(u8s(r)), "usize"), None
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
        if m == "lower": return alower(r), None
        if m == "reverse": return r[::-1], None
        if m == "tostr": return r, None
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
        if m == "sum":
            acc = Int(0, "lit")
            for x in xs: acc = arith("+", acc, x)
            return acc, None
        if m == "map": return Lst([apply(a[0], [x]) for x in xs]), None
        if m == "filter": return Lst([x for x in xs if apply(a[0], [x]) is True]), None
        if m == "fold":
            acc = a[0]
            for x in xs: acc = apply(a[1], [acc, x])
            return acc, None
    if t is Mp:
        ps = r.ps
        if m == "len": return Int(len(ps), "usize"), None
        if m == "get":
            for k, v in ps:
                if eq(k, a[0]): return v, None
            return Err(Rec("Missing", [])), None
        if m == "contains": return any(eq(k, a[0]) for k, _ in ps), None
        if m == "insert": return UNIT, Mp(map_set(ps, a[0], a[1]))
        if m == "remove":
            for i, (k, v) in enumerate(ps):
                if eq(k, a[0]): return Opt(v), Mp(ps[:i] + ps[i + 1:])
            return NONE, Mp(list(ps))
        if m == "keys": return Lst([k for k, _ in ps]), None
        if m == "values": return Lst([v for _, v in ps]), None
    if t is Opt:
        if m == "isnone": return r.x is None, None
        if m == "issome": return r.x is not None, None
        if m == "or": return (a[0] if r.x is None else r.x), None
    trap("%s has no method %s at comptime" % (show(r), m))

def call_method(recv, m, args):
    """(value, per-argument copy-out list, new receiver or None)"""
    t = type(recv)
    head = KIND_OF.get(recv.n, recv.n) if t is Rec else "List" if t is Lst else None
    if head is not None and (head, m) in METHODS:
        v, outs, s = METHODS[(head, m)](recv, args)
        return v, outs, s
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

YIELDS = []
def yield_(v):
    if not YIELDS: trap("yield outside an iterator")
    buf, cap = YIELDS[-1]
    if len(buf) >= LOOP_LIMIT: trap("loop bound")
    buf.append(v)
    if len(buf) >= cap: raise Stop()
def gen_run(body, cap):
    YIELDS.append(([], cap))
    try:
        body(); fin, err = True, None
    except Stop: fin, err = False, None
    except (Trap, Fail) as e: fin, err = True, e
    finally: items = YIELDS.pop()[0]
    return items, fin, err

def iter_items(v):
    t = type(v)
    if t is Lst: return v.xs
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
        b = {}
        if not pmatch(P, x, b): continue
        args = [b.get(n, UNIT) for n in names]
        if cond is not None and not truthy(cond(*args)): continue
        out.append(body(*args))
    return Lst(out)

def catch_(x, h): return h(x.e) if type(x) is Err else x
def try_(x, els):
    t = type(x)
    if t is Err: raise Fail(els() if els else x.e)
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
    if t is Tup or t is Lst: return seq([json_size(x) for x in v.xs], len(v.xs))
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
def refusal(e):
    if isinstance(e, Fail): return v_name(e.v)
    if isinstance(e, Trap): return "trap(" + e.m + ")"
    if isinstance(e, Stop): return "trap(Stop)"
    return "trap(Ret)"
def v_name(v): return v.n if type(v) is Rec else "?"

OUT = []
def emit(v): OUT.append(v)

def obs(label, th):
    try:
        v = th()
        print(label + " = " + show(v))
    except (Trap, Fail, Stop, Ret) as e:
        print(label + " " + why(e))

def show_state(s):
    return "; ".join(r + " { " + ", ".join(n + " = " + show(v) for n, v in fs) + " }" for r, fs in s)

def run_example(line, events):
    print("example %d" % line)
    try: s = initial_state()
    except (Trap, Fail, Stop, Ret) as e:
        print("initial " + why(e)); return
    print("0 (initial) " + show_state(s))
    for k, (name, th) in enumerate(events, 1):
        try: ev = th()
        except (Trap, Fail, Stop, Ret) as e:
            print("%d %s %s" % (k, name, why(e))); return
        if ev is None:
            print("%d %s not an event" % (k, name)); return
        try: s2, outs = delta(s, ev)
        except Refuse as r:
            print("%d %s refused: %s" % (k, show(ev), r.r)); return
        s = s2
        print("%d %s %s" % (k, show(ev), show_state(s)))
        for o in outs: print("  emit " + show(o))

class Refuse(Exception):
    def __init__(s, r): s.r = r

def delta(s, ev):
    nxt, outs = [], []
    for name, fs in s:
        got = None
        for st in STEPS.get(name, []):
            OUT.clear()
            try: r = st(fs, ev)
            except (Trap, Fail, Stop, Ret) as e: raise Refuse(refusal(e))
            if r is not None:
                got = (r, list(OUT)); break
        if got is None: nxt.append((name, fs))
        else:
            nxt.append((name, got[0])); outs.extend(got[1])
    return nxt, outs

def run_main(f):
    print("main")
    OUT.clear()
    try:
        v = f(None, [])[0]
        for o in OUT: print("emit " + show(o))
        print("main = " + show(v))
    except (Trap, Fail, Stop, Ret) as e:
        for o in OUT: print("emit " + show(o))
        print("main " + why(e))
