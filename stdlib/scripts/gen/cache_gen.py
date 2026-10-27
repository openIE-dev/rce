#!/usr/bin/env python3
# Vectors for lib/col/cache.rce (col.cache). Derived vectors, not official
# ones, from three independent references:
#   - CPython 3.13: functools.lru_cache(typed=True) for memo hit and miss
#     sequences, and cachetools 7.2.6 (pip install cachetools==7.2.6):
#     LRUCache, LFUCache and TTLCache (with a logical timer) for generated
#     access traces;
#   - the canonical JSON of spec 6.1.1: the 77 golden vectors of
#     spec/data/canonical-json-vectors.json, and random values encoded by
#     the spec's own Python encoder (scripts/canonical-json-vectors.py,
#     written from 6.1.1 alone);
#   - the rce engine: state hashes from `rce trace --json` of generated
#     programs whose state holds typed values (Set, Map, tuples, structs,
#     kinds, options, floats), which are SHA-256 of the canonical JSON of
#     the state's positional form.
# cachetools.LFUCache picks an arbitrary key among those with the fewest
# uses (it iterates a Python set). The LFU traces replace that set by an
# insertion-ordered one, which pins the choice to the key that reached the
# count first (col.cache's rule) and changes nothing else; the generator
# counts the evictions where more than one key was a candidate.
# Usage, from stdlib/:
#   RCE=<rce binary> python3 scripts/gen/cache_gen.py
# It rewrites tests/col/cache_checks_2.rce .. cache_checks_9.rce.
import functools, hashlib, importlib.util, json, os, pathlib, random, struct, subprocess, sys, tempfile

import cachetools

HERE = pathlib.Path(__file__).resolve().parent
STD = HERE.parent.parent
REPO = STD.parent
OUT = STD / "tests" / "col"
RCE = os.environ.get("RCE", "rce")
assert cachetools.__version__ == "7.2.6", cachetools.__version__

spec = importlib.util.spec_from_file_location("cjv", REPO / "scripts" / "canonical-json-vectors.py")
cjv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cjv)

I64_MIN, I64_MAX, U64_MAX = -2**63, 2**63 - 1, 2**64 - 1


# ---------------- RCE literals ----------------

def q(s):
    o = []
    for ch in s:
        c = ord(ch)
        if ch == "\\": o.append("\\\\")
        elif ch == '"': o.append('\\"')
        elif 32 <= c < 127: o.append(ch)
        else: o.append("\\u{%x}" % c)
    return '"' + "".join(o) + '"'


def ilit(n):
    if n == I64_MIN: return "0 - 9223372036854775807 - 1"
    return str(n)


def flit(x):
    if x == 0.0: return "-0.0" if struct.pack(">d", x)[0] & 0x80 else "0.0"
    r = repr(x)
    if "e" not in r and "." not in r: r += ".0"
    if r.startswith("-"): return "-" + r[1:]
    return r


def fbits(h):
    return struct.unpack(">d", struct.pack(">Q", int(h, 16)))[0]


# A tagged value (the format of canonical-json-vectors.json, plus
# ["set", [v...]]) as a CVal expression.
def cv(v):
    t = v[0]
    if t == "null": return "CvNull"
    if t == "bool": return "CvBool(b: %s)" % ("true" if v[1] else "false")
    if t == "int":
        n = int(v[1])
        return "CvInt(v: %s)" % ilit(n) if n <= I64_MAX else "CvU64(v: %d)" % n
    if t == "float": return "CvFloat(f: %s)" % flit(fbits(v[1]))
    if t == "text": return "CvText(t: %s)" % q(v[1])
    if t == "list": return "CvList(items: [%s])" % ", ".join(cv(x) for x in v[1])
    if t == "rec": return "CvRec(keys: [%s], vals: [%s])" % (", ".join(q(k) for k, _ in v[1]), ", ".join(cv(x) for _, x in v[1]))
    if t == "map": return "t_map([%s], [%s])" % (", ".join(cv(k) for k, _ in v[1]), ", ".join(cv(x) for _, x in v[1]))
    if t == "set": return "t_set([%s])" % ", ".join(cv(x) for x in v[1])
    raise ValueError(t)


def fl(x): return ["float", "0x%016x" % struct.unpack(">Q", struct.pack(">d", x))[0]]


HEAD_CV = """
fn t_map(ks: List[CVal], vs: List[CVal]) -> CVal = cv_map(&ks, &vs)

fn t_set(xs: List[CVal]) -> CVal = cv_set(&xs)

fn t_canon(v: CVal, want: text) -> bool = cv_canon(&v) == want

fn t_hash(v: CVal, want: text) -> bool = cv_hash(&v) == want
"""


def write(n, header, body):
    p = OUT / ("cache_checks_%d.rce" % n)
    p.write_text(header.rstrip("\n") + "\n" + body)
    return p


# ---------------- shard 2: canonical JSON ----------------

TEXT_POOL = (["a", "b", "z", "A", " ", '"', "\\", "/", "\n", "\r", "\t", "\b", "\f", "\x00", "\x01", "\x1f", "\x7f",
              "é", "é", "€", " ", " ", "דּ", "﻿", "\U0001F600", "\U0010FFFF", "$v", "0", "-"])


def rand_text(r):
    return "".join(r.choice(TEXT_POOL) for _ in range(r.randint(0, 6)))


def rand_float(r):
    k = r.randrange(6)
    if k == 0: return r.choice([0.0, -0.0, 1.0, -1.0, 0.1, 1e-4, 1e16, 9999999999999998.0, 5e-324, 1.7976931348623157e308, 2.0**53, 1e-5, 123.5])
    if k == 1: return r.uniform(-1e6, 1e6)
    if k == 2: return float(r.randint(-10**17, 10**17))
    while True:
        x = struct.unpack(">d", struct.pack(">Q", r.getrandbits(64)))[0]
        if x == x and abs(x) != float("inf"): return x


def rand_int(r):
    k = r.randrange(5)
    if k == 0: return r.choice([0, 1, -1, I64_MIN, I64_MAX, U64_MAX, 2**63, 2**53 + 1])
    if k == 1: return r.randint(-1000, 1000)
    if k == 2: return r.randint(0, U64_MAX)
    return r.randint(I64_MIN, I64_MAX)


def rand_val(r, d):
    k = r.randrange(9 if d < 3 else 6)
    if k == 0: return ["null"]
    if k == 1: return ["bool", r.random() < 0.5]
    if k == 2: return ["int", str(rand_int(r))]
    if k == 3: return fl(rand_float(r))
    if k in (4, 5): return ["text", rand_text(r)]
    if k == 6: return ["list", [rand_val(r, d + 1) for _ in range(r.randint(0, 4))]]
    if k == 7:
        keys = []
        for _ in range(r.randint(0, 4)):
            t = rand_text(r)
            if t not in keys: keys.append(t)
        return ["rec", [[t, rand_val(r, d + 1)] for t in keys]]
    # a map with int or text keys (the key types the spec encoder orders)
    if r.random() < 0.5:
        ks = sorted(set(r.randint(-50, 50) for _ in range(r.randint(0, 5))))
        r.shuffle(ks)
        return ["map", [[["int", str(x)], rand_val(r, d + 1)] for x in ks]]
    ks = list(dict.fromkeys(rand_text(r) for _ in range(r.randint(0, 5))))
    return ["map", [[["text", x], rand_val(r, d + 1)] for x in ks]]


def shard_canon():
    golden = json.loads((REPO / "spec" / "data" / "canonical-json-vectors.json").read_text())["vectors"]
    lines = []
    for g in golden:
        assert cjv.enc(g["value"]) == g["canonical"]
        assert hashlib.sha256(g["canonical"].encode()).hexdigest() == g["sha256"]
        lines.append("check t_canon(%s, %s) and t_hash(%s, %s)" % (cv(g["value"]), q(g["canonical"]), cv(g["value"]), q(g["sha256"])))
    r = random.Random(20261027)
    n_rand = 0
    for _ in range(160):
        v = rand_val(r, 0)
        c = cjv.enc(v)
        lines.append("check t_canon(%s, %s) and t_hash(%s, %s)" % (cv(v), q(c), cv(v), q(hashlib.sha256(c.encode()).hexdigest())))
        n_rand += 1
    head = ("# Generated by scripts/gen/cache_gen.py: canonical JSON (spec 6.1.1).\n"
            "# %d golden vectors of spec/data/canonical-json-vectors.json, then %d\n"
            "# random values (seed 20261027) encoded by scripts/canonical-json-vectors.py.\n" % (len(golden), n_rand)) + HEAD_CV
    return head, "\n".join(lines) + "\n", len(golden), n_rand


# ---------------- shard 13: floats ----------------

def shard_floats():
    import math
    r = random.Random(20261038)
    xs = []
    for b in [1e-4, 1e16, 1e-5, 1e15, 1e21, 2.0**53, 2.0**63, 1.0, 0.1]:
        lo = hi = b
        for _ in range(3):
            lo = math.nextafter(lo, 0.0); hi = math.nextafter(hi, math.inf)
            xs += [lo, hi, -lo, -hi]
        xs += [b, -b]
    xs += [5e-324, 1e-323, 2.2250738585072014e-308, 2.225073858507201e-308, 1.7976931348623157e308]
    for _ in range(250): xs.append(rand_float(r))
    for _ in range(120): xs.append(round(r.uniform(-1e5, 1e5), r.randint(0, 8)))
    for _ in range(60): xs.append(float(r.randint(1, 10**9)) * 10.0 ** r.randint(-30, 30))
    lines = []
    for x in xs:
        bits = struct.unpack(">Q", struct.pack(">d", x))[0]
        lines.append("check cv_float_text(%s) == %s" % (flit(x), q(cjv.float_text(bits))))
    head = ("# Generated by scripts/gen/cache_gen.py: cv_float_text against float_text of\n"
            "# scripts/canonical-json-vectors.py (rule 3 of spec 6.1.1), %d binary64\n"
            "# values: the neighbours of the notation boundaries, then random ones\n"
            "# (seed 20261038).\n" % len(xs))
    return head, "\n".join(lines) + "\n", len(xs)


# ---------------- shard 3: engine state hashes ----------------
# Each case is (RCE type, RCE expression, tagged value of the positional
# form). The engine hashes [[f0, f1, ...]] for one state of fields f0...

def eng_cases(r):
    cs = []
    # floats, including both zeros and the decimal/exponent boundaries
    fs = [0.0, -0.0, 1.0, 0.1, 1e-4, 9.999999999999999e-5, 1e16, 9999999999999998.0, 5e-324, -1.7976931348623157e308, 1.5e-7, 123456789.125] + [rand_float(r) for _ in range(12)]
    cs.append(("List[f64]", "[%s]" % ", ".join(flit(x) for x in fs), ["list", [fl(x) for x in fs]]))
    # sets: text, ints, floats (totalOrder), tuples; input unsorted with repeats
    ts = [rand_text(r) for _ in range(10)] + ["b", "a", "b", "", "é", "\U0001F600", "דּ", "Z"]
    cs.append(("Set[text]", "set_of([%s])" % ", ".join(q(t) for t in ts), ["set", [["text", t] for t in ts]]))
    ns = [r.randint(-20, 20) for _ in range(15)] + [I64_MIN, I64_MAX]
    cs.append(("Set[i64]", "set_of([%s])" % ", ".join(ilit(n) for n in ns), ["set", [["int", str(n)] for n in ns]]))
    xs = [0.0, -0.0, 2.5, -2.5, 1e300, -1e-300, 2.5]
    cs.append(("Set[f64]", "set_of([%s])" % ", ".join(flit(x) for x in xs), ["set", [fl(x) for x in xs]]))
    tp = [(r.randint(0, 3), r.choice(["a", "b", "", "ab"])) for _ in range(10)]
    cs.append(("Set[(i64, text)]", "set_of([%s])" % ", ".join("(%d, %s)" % (a, q(b)) for a, b in tp), ["set", [["list", [["int", str(a)], ["text", b]]] for a, b in tp]]))
    ls = [[r.randint(0, 2) for _ in range(r.randint(0, 3))] for _ in range(10)]
    # an empty element list is written copy(t_empty) (a typed let in the step)
    cs.append(("Set[List[i64]]", "set_of([%s])" % ", ".join(("[%s]" % ", ".join(str(x) for x in l)) if l else "copy(t_empty)" for l in ls), ["set", [["list", [["int", str(x)] for x in l]] for l in ls]]))
    # maps: text keys, int keys, tuple keys, float keys, nested values
    m1 = {rand_text(r): r.randint(-9, 9) for _ in range(10)}
    cs.append(("Map[text, i64]", "{%s}" % ", ".join("%s: %d" % (q(k), v) for k, v in m1.items()), ["map", [[["text", k], ["int", str(v)]] for k, v in m1.items()]]))
    m2 = {r.randint(-100, 100): rand_text(r) for _ in range(10)}
    cs.append(("Map[i64, text]", "{%s}" % ", ".join("%d: %s" % (k, q(v)) for k, v in m2.items()), ["map", [[["int", str(k)], ["text", v]] for k, v in m2.items()]]))
    m3 = {(r.randint(0, 3), r.randint(0, 3)): rand_text(r) for _ in range(8)}
    cs.append(("Map[(i64, i64), text]", "{%s}" % ", ".join("(%d, %d): %s" % (a, b, q(v)) for (a, b), v in m3.items()), ["map", [[["list", [["int", str(a)], ["int", str(b)]]], ["text", v]] for (a, b), v in m3.items()]]))
    m4 = {x: i for i, x in enumerate([0.5, -0.5, 3.0, 1e20, -0.0])}
    cs.append(("Map[f64, i64]", "{%s}" % ", ".join("%s: %d" % (flit(k), v) for k, v in m4.items()), ["map", [[fl(k), ["int", str(v)]] for k, v in m4.items()]]))
    m5 = {rand_text(r): [r.randint(0, 9) for _ in range(r.randint(0, 3))] for _ in range(6)}
    cs.append(("Map[text, List[i64]]", "{%s}" % ", ".join("%s: %s" % (q(k), ("[%s]" % ", ".join(str(x) for x in v)) if v else "copy(t_empty)") for k, v in m5.items()), ["map", [[["text", k], ["list", [["int", str(x)] for x in v]]] for k, v in m5.items()]]))
    # scalars and products
    for t in [rand_text(r) for _ in range(4)] + ["\x00\x08\x0c\x1f\x7f\"\\/\n\r\t", "  \U0010FFFF"]:
        cs.append(("text", q(t), ["text", t]))
    cs.append(("u64", "18446744073709551615", ["int", str(U64_MAX)]))
    cs.append(("u64", "9223372036854775808", ["int", str(2**63)]))
    cs.append(("i64", ilit(I64_MIN), ["int", str(I64_MIN)]))
    cs.append(("bool", "true", ["bool", True]))
    cs.append(("i64?", "none", ["null"]))
    cs.append(("i64?", "some(42)", ["int", "42"]))
    cs.append(("(i64, text, bool)", "(7, %s, false)" % q("t\x01"), ["list", [["int", "7"], ["text", "t\x01"], ["bool", False]]]))
    cs.append(("TPt", "TPt(y: -3, x: %s)" % q("q"), ["list", [["int", "-3"], ["text", "q"]]]))
    cs.append(("List[TSh]", "[TDot, TBox(w: 1, h: 2), TDot]", ["list", [["list", [["int", "0"], ["list", []]]], ["list", [["int", "1"], ["list", [["int", "1"], ["int", "2"]]]]], ["list", [["int", "0"], ["list", []]]]]]))
    cs.append(("List[List[text]]", "[copy(t_etext), [%s], [%s, %s]]" % (q("a"), q(""), q("é")), ["list", [["list", []], ["list", [["text", "a"]]], ["list", [["text", ""], ["text", "é"]]]]]))
    return cs


def default_of(ty):
    if ty.startswith("Set["): return "set_of([])"
    if ty.startswith("Map["): return "{}"
    if ty.startswith("List["): return "[]"
    return {"text": '""', "u64": "0", "i64": "0", "bool": "false", "i64?": "none", "(i64, text, bool)": '(0, "", false)', "TPt": 'TPt(y: 0, x: "")'}[ty]


def engine_hash(fields):
    src = ["module eng_case", "", "event Go", "", "struct TPt", "    y: i64", "    x: text", "", "kind TSh", "    TDot", "    TBox(w: i64, h: i64)", "", "state S"]
    for i, (ty, _, _) in enumerate(fields):
        src.append("    f%d: %s = %s" % (i, ty, default_of(ty)))
    src += ["", "step S on Go"]
    if any("t_empty" in ex for _, ex, _ in fields):
        src.append("    let t_empty: List[i64] = []")
    if any("t_etext" in ex for _, ex, _ in fields):
        src.append("    let t_etext: List[text] = []")
    for i, (_, ex, _) in enumerate(fields):
        src.append("    f%d = %s" % (i, ex))
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "eng_case.rce"
        p.write_text("\n".join(src) + "\n")
        out = subprocess.run([RCE, "trace", "--json", str(p), "Go"], capture_output=True, text=True)
    if out.returncode != 0: raise SystemExit("rce trace failed:\n" + "\n".join(src) + "\n" + out.stdout + out.stderr)
    st = json.loads(out.stdout)["traces"][0]["steps"][0]
    assert st["outcome"] == "applied", st
    return st["state_hash"]


def shard_engine():
    r = random.Random(20261028)
    cases = eng_cases(r)
    lines = []
    # each case alone, then the whole state at once
    for c in cases:
        h = engine_hash([c])
        lines.append("check t_hash(CvList(items: [CvList(items: [%s])]), %s)" % (cv(c[2]), q(h)))
    h = engine_hash(cases)
    lines.append("check t_hash(CvList(items: [CvList(items: [%s])]), %s)" % (", ".join(cv(c[2]) for c in cases), q(h)))
    head = ("# Generated by scripts/gen/cache_gen.py: cv_hash against the rce engine. Each\n"
            "# expected hash is the state_hash `rce trace --json` printed for a program\n"
            "# whose one state holds the value (seed 20261028): SHA-256 of the canonical\n"
            "# JSON of the state's positional form [[f0, ...]] (spec 6.1). Sets and maps\n"
            "# are given unsorted, with repeats, and built by cv_set and cv_map.\n") + HEAD_CV
    return head, "\n".join(lines) + "\n", len(lines)


# ---------------- trace drivers (RCE) ----------------

HEAD_TR = """
fn t_kt(i: i64) -> text = "k" + i.tostr()

fn t_kid(k: &text) -> i64
    var v: i64 = 0
    var i: usize = 1
    while i < k.len_bytes() bound 32
        v = v * 10 + ((k[i] as i64) - 48)
        i += 1
    return v

fn t_oi(o: i64?) -> i64
    match o
        some(x) -> return x
        none -> return -1

fn t_ids(ks: &[text]) -> List[i64]
    var o: List[i64] = []
    for k in ks
        o.push(t_kid(&k))
    return o

fn t_lst(ks: &[text]) -> List[i64]
    var o: List[i64] = [ks.len() as i64]
    for k in ks
        o.push(t_kid(&k))
    return o

fn t_kv(p: (text, i64)?) -> List[i64]
    match p
        some((k, v)) -> return [t_kid(&k), v]
        none -> return [-1, -1]

# whether got equals want[j .. j + got.len()]
fn t_at(got: &[i64], want: &[i64], j: usize) -> bool
    if j + got.len() > want.len()
        return false
    var i: usize = 0
    while i < got.len() bound 1000000
        if got[i] != want[j + i]
            return false
        i += 1
    return true
"""

LRU_DRV = """
# ops: (op, key, value) triples; outs: the expected outputs in order.
# Returns -1 when every output and the final order agree, else the index
# of the first op that disagrees (or -2 for the final order).
fn t_lru(cap: i64, ops: List[i64], outs: List[i64], fin: List[i64]) -> i64
    var c: LruCache[i64] = lru_new(cap)
    var j: usize = 0
    var i: usize = 0
    while i + 2 < ops.len() bound 1000000
        let op = ops[i]
        let k = t_kt(ops[i + 1])
        let v = ops[i + 2]
        var got: List[i64] = []
        if op == 0
            got = [t_oi(lru_get(&mut c, &k))]
        elif op == 1
            let ev = lru_put_evict(&mut c, copy(k), v)
            got = t_lst(&ev)
        elif op == 2
            got = [if lru_contains(&c, &k) then 1 else 0]
        elif op == 3
            got = [t_oi(lru_remove(&mut c, &k))]
        elif op == 4
            got = [t_oi(lru_peek(&c, &k))]
        elif op == 5
            got = t_kv(lru_pop(&mut c))
        else
            got = [lru_len(&c)]
        if not t_at(&got, &outs, j)
            return (i / 3) as i64
        j += got.len()
        i += 3
    let ks = lru_keys(&c)
    if t_ids(&ks) != fin or j != outs.len()
        return -2
    return -1
"""

LFU_DRV = """
fn t_lfu(cap: i64, ops: List[i64], outs: List[i64], fin: List[i64], cnt: List[i64]) -> i64
    var c: LfuCache[i64] = lfu_new(cap)
    var j: usize = 0
    var i: usize = 0
    while i + 2 < ops.len() bound 1000000
        let op = ops[i]
        let k = t_kt(ops[i + 1])
        let v = ops[i + 2]
        var got: List[i64] = []
        if op == 0
            got = [t_oi(lfu_get(&mut c, &k))]
        elif op == 1
            let ev = lfu_put_evict(&mut c, copy(k), v)
            got = t_lst(&ev)
        elif op == 2
            got = [if lfu_contains(&c, &k) then 1 else 0]
        elif op == 3
            got = [t_oi(lfu_remove(&mut c, &k))]
        elif op == 4
            got = [t_oi(lfu_peek(&c, &k))]
        elif op == 5
            got = t_kv(lfu_pop(&mut c))
        elif op == 6
            got = [lfu_len(&c)]
        else
            got = [lfu_count(&c, &k)]
        if not t_at(&got, &outs, j)
            return (i / 3) as i64
        j += got.len()
        i += 3
    let ks = lfu_keys(&c)
    if t_ids(&ks) != fin or j != outs.len()
        return -2
    var n: usize = 0
    while n < ks.len() bound 1000000
        if lfu_count(&c, &ks[n]) != cnt[n]
            return -3
        n += 1
    return -1
"""

TTL_DRV = """
# ops: (op, key, value, now) quadruples.
fn t_ttl(cap: i64, ttl: i64, ops: List[i64], outs: List[i64], fin: List[i64], tfin: i64) -> i64
    var c: TtlCache[i64] = ttl_new(cap, ttl)
    var j: usize = 0
    var i: usize = 0
    while i + 3 < ops.len() bound 1000000
        let op = ops[i]
        let k = t_kt(ops[i + 1])
        let v = ops[i + 2]
        let now = ops[i + 3]
        var got: List[i64] = []
        if op == 0
            got = [t_oi(ttl_get(&mut c, &k, now))]
        elif op == 1
            let ev = ttl_put_evict(&mut c, copy(k), v, now)
            got = t_lst(&ev)
        elif op == 2
            got = [if ttl_contains(&c, &k, now) then 1 else 0]
        elif op == 3
            got = [t_oi(ttl_remove(&mut c, &k, now))]
        elif op == 4
            got = [t_oi(ttl_peek(&c, &k, now))]
        elif op == 5
            got = t_kv(ttl_pop(&mut c, now))
        elif op == 6
            got = [ttl_len(&mut c, now)]
        else
            let ex = ttl_expire(&mut c, now)
            got = t_lst(&ex)
        if not t_at(&got, &outs, j)
            return (i / 4) as i64
        j += got.len()
        i += 4
    let ks = ttl_keys(&c, tfin)
    if t_ids(&ks) != fin or j != outs.len()
        return -2
    return -1
"""


def ilist(xs):
    return "[" + ", ".join(ilit(x) for x in xs) + "]"


# ---------------- shard 4..5: LRU (cachetools.LRUCache) ----------------

def kid(k): return k


def lru_trace(r, cap, nops, nkeys):
    c = cachetools.LRUCache(maxsize=cap)
    ops, outs = [], []
    for _ in range(nops):
        x = r.random()
        k = r.randrange(nkeys)
        v = r.randrange(1000)
        if x < 0.33:
            op = 0; o = [c.get(k, -1)]
        elif x < 0.70:
            op = 1
            before = list(c.keys())
            try:
                c[k] = v
                o = [e for e in before if e not in c]
            except ValueError:
                o = [k]
            o = [len(o)] + o
        elif x < 0.78:
            op = 2; o = [1 if k in c else 0]
        elif x < 0.85:
            op = 3; o = [c.pop(k, -1)]
        elif x < 0.91:
            op = 4; o = [cachetools.Cache.__getitem__(c, k) if k in c else -1]
        elif x < 0.95:
            op = 5
            try: kk, vv = c.popitem(); o = [kk, vv]
            except KeyError: o = [-1, -1]
        else:
            op = 6; o = [len(c)]
        ops += [op, k, v]
        outs += o
    fin = list(c._LRUCache__order)
    return ops, outs, fin


def shard_lru(seed, caps, reps=3, nops=(150, 260), keys=2.0):
    r = random.Random(seed)
    lines, n = [], 0
    for cap in caps:
        for rep in range(reps):
            nk = max(2, int(keys * cap) + r.randint(0, 3))
            ops, outs, fin = lru_trace(r, cap, r.randint(*nops), nk)
            lines.append("check t_lru(%d, %s, %s, %s) == -1" % (cap, ilist(ops), ilist(outs), ilist(fin)))
            n += len(ops) // 3
    return lines, n


# ---------------- shard 6..7: LFU (cachetools.LFUCache) ----------------

class OSet:
    """An insertion-ordered set: the LFU bucket order col.cache uses."""
    __slots__ = ("d",)
    def __init__(self): self.d = {}
    def add(self, k): self.d[k] = None
    def remove(self, k): del self.d[k]
    def __iter__(self): return iter(self.d)
    def __len__(self): return len(self.d)


class PinnedLFU(cachetools.LFUCache):
    pass


_orig_link_init = cachetools.LFUCache._Link.__init__


def _ordered_link_init(self, count):
    _orig_link_init(self, count)
    self.keys = OSet()


TIES = [0, 0]  # evictions, evictions with more than one candidate


def lfu_buckets(c):
    root = c._LFUCache__root
    out, b = [], root.next
    while b is not root:
        out.append((b.count, list(b.keys)))
        b = b.next
    return out


def lfu_trace(r, cap, nops, nkeys):
    cachetools.LFUCache._Link.__init__ = _ordered_link_init
    try:
        c = cachetools.LFUCache(maxsize=cap)
        ops, outs = [], []
        for _ in range(nops):
            x = r.random()
            k = r.randrange(nkeys)
            v = r.randrange(1000)
            if x < 0.33:
                op = 0; o = [c.get(k, -1)]
            elif x < 0.68:
                op = 1
                before = list(c.keys())
                bk = lfu_buckets(c)
                try:
                    c[k] = v
                    o = [e for e in before if e not in c]
                except ValueError:
                    o = [k]
                if k not in before and cap > 0 and len(before) >= cap:
                    TIES[0] += 1
                    if len(bk[0][1]) > 1: TIES[1] += 1
                o = [len(o)] + o
            elif x < 0.74:
                op = 2; o = [1 if k in c else 0]
            elif x < 0.80:
                op = 3; o = [c.pop(k, -1)]
            elif x < 0.85:
                op = 4; o = [cachetools.Cache.__getitem__(c, k) if k in c else -1]
            elif x < 0.89:
                op = 5
                bk = lfu_buckets(c)
                try:
                    kk, vv = c.popitem(); o = [kk, vv]
                    TIES[0] += 1
                    if len(bk[0][1]) > 1: TIES[1] += 1
                except KeyError: o = [-1, -1]
            elif x < 0.93:
                op = 6; o = [len(c)]
            else:
                op = 7; o = [c._LFUCache__links[k].count if k in c else 0]
            ops += [op, k, v]
            outs += o
        fin, cnt = [], []
        for count, keys in lfu_buckets(c):
            for k in keys:
                fin.append(k); cnt.append(count)
        return ops, outs, fin, cnt
    finally:
        cachetools.LFUCache._Link.__init__ = _orig_link_init


def shard_lfu(seed, caps, reps=3, nops=(150, 260), keys=2.0):
    r = random.Random(seed)
    lines, n = [], 0
    for cap in caps:
        for rep in range(reps):
            nk = max(2, int(keys * cap) + r.randint(0, 3))
            ops, outs, fin, cnt = lfu_trace(r, cap, r.randint(*nops), nk)
            lines.append("check t_lfu(%d, %s, %s, %s, %s) == -1" % (cap, ilist(ops), ilist(outs), ilist(fin), ilist(cnt)))
            n += len(ops) // 3
    return lines, n


# ---------------- shard 8: TTL (cachetools.TTLCache) ----------------

class LoggedTTL(cachetools.TTLCache):
    """TTLCache that logs the keys removed by expire() and popitem()."""
    def __init__(self, *a, **kw):
        self.log = []
        super().__init__(*a, **kw)

    def expire(self, time=None):
        r = super().expire(time)
        self.log.extend(k for k, _ in r)
        return r

    def popitem(self):
        k, v = super().popitem()
        self.log.append(k)
        return (k, v)


def ttl_trace(r, cap, ttl, nops, nkeys, t0):
    clock = [t0]
    c = LoggedTTL(maxsize=cap, ttl=ttl, timer=lambda: clock[0])
    ops, outs = [], []
    seen = None  # the latest instant an op that records it has passed
    for _ in range(nops):
        clock[0] += r.choice([0, 0, 0, 1, 1, 2, 3, ttl // 2 if ttl > 1 else 1])
        now = clock[0]
        x = r.random()
        # an earlier instant, once the cache has recorded now: the cache
        # must read it as the latest instant it has seen
        sent = now - r.randint(1, 50) if (seen == now and r.random() < 0.15) else now
        if not (0.62 <= x < 0.70 or 0.77 <= x < 0.83):
            seen = now
        k = r.randrange(nkeys)
        v = r.randrange(1000)
        c.log = []
        if x < 0.30:
            op = 0; o = [c.get(k, -1)]
        elif x < 0.62:
            op = 1
            try:
                c[k] = v
                o = list(c.log)
            except ValueError:
                o = list(c.log) + [k]
            o = [len(o)] + o
        elif x < 0.70:
            op = 2; o = [1 if k in c else 0]
        elif x < 0.77:
            op = 3; o = [c.pop(k, -1)]
        elif x < 0.83:
            op = 4; o = [cachetools.Cache.__getitem__(c, k) if k in c else -1]
        elif x < 0.87:
            op = 5
            try: kk, vv = c.popitem(); o = [kk, vv]
            except KeyError: o = [-1, -1]
        elif x < 0.92:
            op = 6; o = [len(c)]
        else:
            op = 7; c.expire(); o = [len(c.log)] + list(c.log)
        ops += [op, k, v, sent]
        outs += o
    fin = [k for k in c._TTLCache__links if k in c]
    return ops, outs, fin, clock[0]


TTL_SMALL = [(0, 5), (1, 3), (2, 4), (3, 1), (3, 10), (4, 0), (5, 7), (8, 6), (8, 20), (16, 12), (3, -2), (30, 9)]


def shard_ttl(seed, params=TTL_SMALL, reps=2, nops=(150, 240), keys=2.0):
    r = random.Random(seed)
    lines, n = [], 0
    for cap, ttl in params:
        for rep in range(reps):
            nk = max(2, int(keys * cap) + r.randint(0, 3))
            m = r.randint(*nops)
            t0 = r.choice([0, 1000, -500, 10**12, I64_MAX - 3000])
            # time stays inside i64 and below the saturation of t + ttl
            # (that boundary has its own checks in cache_checks.rce)
            if t0 + m * max(3, ttl // 2) + abs(ttl) >= I64_MAX:
                t0 = 10**12
            ops, outs, fin, tf = ttl_trace(r, cap, ttl, m, nk, t0)
            lines.append("check t_ttl(%d, %d, %s, %s, %s, %d) == -1" % (cap, ttl, ilist(ops), ilist(outs), ilist(fin), tf))
            n += len(ops) // 4
    return lines, n


# ---------------- shard 9: memo (functools.lru_cache) ----------------

def py_val(r, d=0):
    """A Python argument value whose lru_cache(typed=True) key equality is
    canonical-JSON equality: bools and floats only at the top level (inside
    a tuple Python compares 1 == 1.0 == True), no -0.0, ints in the Form
    range."""
    k = r.randrange(7 if d == 0 else 4)
    if k == 0: return r.randint(-5, 5)
    if k == 1: return r.choice([I64_MIN, I64_MAX, U64_MAX, 2**63, 10**18, -7])
    if k == 2: return rand_text(r)
    if k == 3: return tuple(py_val(r, d + 1) for _ in range(r.randint(0, 3)))
    if k == 4: return r.choice([True, False])
    if k == 5:
        x = r.choice([0.0, 1.0, 0.5, -2.5, 1e16, 1e-7, 3.0, 5e-324])
        return x
    return None


def tagged(x):
    if x is None: return ["null"]
    if isinstance(x, bool): return ["bool", x]
    if isinstance(x, int): return ["int", str(x)]
    if isinstance(x, float): return fl(x)
    if isinstance(x, str): return ["text", x]
    if isinstance(x, tuple): return ["list", [tagged(y) for y in x]]
    raise TypeError(x)


def memo_pool(r, n):
    pool = []
    while len(pool) < n:
        args = tuple(py_val(r) for _ in range(r.randint(0, 3)))
        pool.append(args)
    # equal arguments built separately, and 1 / 1.0 / True / "1"
    pool += [(1,), (1.0,), (True,), ("1",), (1,), ((1, 2),), (1, 2), ((1, 2),), (None,), (), ((),)]
    mk = lambda a: functools._make_key(a, {}, True)
    for a in pool:
        for b in pool:
            same_py = mk(a) == mk(b)
            same_cj = cjv.enc(tagged(a)) == cjv.enc(tagged(b))
            assert same_py == same_cj, (a, b)
    return pool


def memo_trace(r, pool, maxsize, ncalls):
    step = [0]

    @functools.lru_cache(maxsize=maxsize, typed=True)
    def f(*args):
        return step[0]

    calls, want = [], []
    # a working set that drifts, so that both hits and evictions occur
    for s in range(ncalls):
        i = r.randrange(len(pool)) if r.random() < 0.3 else min(len(pool) - 1, (s // 7) % len(pool) + r.randrange(6))
        step[0] = s
        calls.append(i)
        want.append(f(*pool[i]))
    ci = f.cache_info()
    return calls, want, [ci.hits, ci.misses, -1 if ci.maxsize is None else ci.maxsize, ci.currsize]


MEMO_DRV = """
# calls: pool indexes; want: the value lru_cache returned for each call
# (the index of the call that computed it); info: hits, misses, maxsize
# (-1 for None), currsize. maxsize -1 builds an unbounded memo.
fn t_memo(maxsize: i64, calls: List[i64], want: List[i64], info: List[i64]) -> i64
    let pool = t_pool()
    var m: Memo[i64] = if maxsize < 0 then memo_unbounded() else memo_new(maxsize)
    var s: usize = 0
    while s < calls.len() bound 1000000
        let a = copy(pool[calls[s] as usize])
        var got: i64 = -1
        match memo_lookup(&mut m, &a)
            some(x) -> got = x
            none -> got = -1
        if got < 0
            memo_store(&mut m, &a, s as i64)
            got = s as i64
        if got != want[s]
            return s as i64
        s += 1
    let i = memo_info(&m)
    if [i.hits, i.misses, i.maxsize, i.currsize] != info
        return -2
    return -1
"""


def shard_memo():
    r = random.Random(20261029)
    pool = memo_pool(r, 45)
    body = ["fn t_pool() -> List[CVal] = [%s]" % ", ".join("CvList(items: [%s])" % ", ".join(cv(tagged(x)) for x in a) for a in pool), ""]
    lines = []
    for i, a in enumerate(pool):
        c = cjv.enc(tagged(a))
        lines.append("check t_canon(copy(t_pool()[%d]), %s) and t_hash(copy(t_pool()[%d]), %s)" % (i, q(c), i, q(hashlib.sha256(c.encode()).hexdigest())))
    nt = ncall = 0
    for maxsize in [0, 1, 2, 3, 5, 8, 16, 32, None]:
        for rep in range(2):
            calls, want, info = memo_trace(r, pool, maxsize, r.randint(120, 200))
            lines.append("check t_memo(%d, %s, %s, %s) == -1" % (-1 if maxsize is None else maxsize, ilist(calls), ilist(want), ilist(info)))
            nt += 1; ncall += len(calls)
    head = ("# Generated by scripts/gen/cache_gen.py: Memo against CPython %s\n"
            "# functools.lru_cache(maxsize, typed=True) (seed 20261029). The pool holds %d\n"
            "# argument tuples whose lru_cache key equality is canonical JSON equality\n"
            "# (checked by the generator for every pair); each pool entry's canonical\n"
            "# JSON and SHA-256 come from scripts/canonical-json-vectors.py.\n" % (sys.version.split()[0], len(pool))) + HEAD_CV + HEAD_TR + MEMO_DRV
    return head, "\n".join(body) + "\n" + "\n".join(lines) + "\n", len(pool), nt, ncall


def main():
    stats = {}
    h, b, ng, nr = shard_canon()
    write(2, h, b); stats["canon"] = (ng, nr)
    h, b, ne = shard_engine()
    write(3, h, b); stats["engine"] = ne
    lru_head = ("# Generated by scripts/gen/cache_gen.py: LruCache against cachetools %s\n"
                "# LRUCache (get, put with the keys it evicts, in, pop, the base\n"
                "# __getitem__ as peek, popitem, len; the final order is LRUCache's).\n" % cachetools.__version__) + HEAD_TR + LRU_DRV
    l1, n1 = shard_lru(20261030, [0, 1, 2, 3, 4])
    l2, n2 = shard_lru(20261031, [5, 8, 13, 32])
    write(4, lru_head, "\n".join(l1) + "\n"); write(5, lru_head, "\n".join(l2) + "\n")
    stats["lru"] = (len(l1) + len(l2), n1 + n2)
    lfu_head = ("# Generated by scripts/gen/cache_gen.py: LfuCache against cachetools %s\n"
                "# LFUCache, with its per-count key set made insertion-ordered (the\n"
                "# choice among keys of equal count that cachetools leaves open). The\n"
                "# final keys and counts are the bucket list of LFUCache.\n" % cachetools.__version__) + HEAD_TR + LFU_DRV
    l1, n1 = shard_lfu(20261032, [0, 1, 2, 3, 4])
    l2, n2 = shard_lfu(20261033, [5, 8, 13, 32])
    write(6, lfu_head, "\n".join(l1) + "\n"); write(7, lfu_head, "\n".join(l2) + "\n")
    stats["lfu"] = (len(l1) + len(l2), n1 + n2, tuple(TIES))
    ttl_head = ("# Generated by scripts/gen/cache_gen.py: TtlCache against cachetools %s\n"
                "# TTLCache with a logical timer; the keys a put, popitem or expire removes\n"
                "# are logged from TTLCache.expire and popitem. Some ops pass an instant\n"
                "# earlier than one already passed; the cache reads it as the latest.\n" % cachetools.__version__) + HEAD_TR + TTL_DRV
    l1, n1 = shard_ttl(20261034)
    write(8, ttl_head, "\n".join(l1) + "\n")
    stats["ttl"] = (len(l1), n1)
    h, b, npool, nt, nc = shard_memo()
    write(9, h, b)
    stats["memo"] = (npool, nt, nc)
    # larger caches: the slot table grows, wraps and deletes under load
    l1, n1 = shard_lru(20261035, [64, 200, 700], reps=1, nops=(2500, 3000), keys=1.4)
    write(10, lru_head, "\n".join(l1) + "\n")
    stats["lru_large"] = (len(l1), n1)
    l1, n1 = shard_lfu(20261036, [64, 200, 700], reps=1, nops=(2500, 3000), keys=1.4)
    write(11, lfu_head, "\n".join(l1) + "\n")
    stats["lfu_large"] = (len(l1), n1, tuple(TIES))
    l1, n1 = shard_ttl(20261037, [(64, 40), (200, 150), (700, 500)], reps=1, nops=(2500, 3000), keys=1.4)
    write(12, ttl_head, "\n".join(l1) + "\n")
    stats["ttl_large"] = (len(l1), n1)
    h, b, nf = shard_floats()
    write(13, h, b)
    stats["floats"] = nf
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
