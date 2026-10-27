#!/usr/bin/env python3
# Generates tests/data/json_checks*.rce for lib/data/json.rce.
#
# Three steps, run from stdlib/ with CPython 3.13 (a venv holding
# jsonpointer 3.0.0 and jsonpatch 1.33), Go 1.24 and Node 22 (canonicalize
# 2.0.0 from npm):
#   python3 scripts/gen/json_gen.py corpus   writes vectors/data/json/syntax_corpus.json
#   go run scripts/gen/json_go.go > vectors/data/json/go1.24_syntax.json
#   node scripts/gen/json_node.js            writes vectors/data/json/node22_es.json
#   python3 scripts/gen/json_gen.py          writes the checks
#
# official: JSONTestSuite (nst/JSONTestSuite test_parsing, every y_, n_
#   and i_ file), the RFC 8785 test data (cyberphone/json-canonicalization
#   testdata input/output and the head of es6testfile100m.txt), the
#   examples of RFC 6901 sections 5 and 6.
# derived: CPython 3.13 json (values, compact and indented text, with
#   duplicate members and number lexemes kept by hooks), float() and
#   decimal (number conversions), jsonpointer 3.0.0 (pointers), Go 1.24
#   encoding/json (syntax verdicts and error offsets), Node 22 (ECMAScript
#   number text, canonicalize 2.0.0 for RFC 8785).
import decimal, json, os, random, re, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcelit import lit as _lit

# W-LEX-4 refuses these characters in source text; \u{...} writes them
def lex4(u):
    return (0x202A <= u <= 0x202E or 0x2066 <= u <= 0x2069 or u in (0x061C, 0x200E, 0x200F) or 0x200B <= u <= 0x200D
            or 0x2060 <= u <= 0x2064 or u in (0xFEFF, 0x00AD, 0x034F, 0x180E, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x3164, 0xFFA0)
            or 0xFFF9 <= u <= 0xFFFB or 0xE0000 <= u <= 0xE007F or u < 32 or 0x7F <= u <= 0x9F or 0xFDD0 <= u <= 0xFDEF
            or (u & 0xFFFE) == 0xFFFE or 0x40000 <= u <= 0xDFFFF or u in (0x2028, 0x2029))
def lit(s):
    o = _lit(s)
    return ''.join('\\u{%x}' % ord(c) if lex4(ord(c)) and c not in '\n\t\r' else c for c in o)
def lits(xs):
    return '[' + ', '.join(lit(x) for x in xs) + ']'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
V = os.path.join(ROOT, 'vectors/data/json')
SUITE = os.path.join(V, 'JSONTestSuite_test_parsing')
SEED = 20261025

out = []
# Each check carries an estimate of its comptime fuel (ticks of the
# reference evaluator, measured with rce emit python --fuel-trace): about
# 120 per character of the check line, 2500 per character for the
# number conversions (big-integer paths), or an explicit figure.
HEAVY = ('ck_fv', 'ck_fz', 'ck_fe', 'ck_nc', 'ck_cn')
def chk(cat, line, w=None):
    if w is None:
        h = line[6:].split('(')[0]
        w = 2000 + len(line) * (2500 if h in HEAVY else 120)
    out.append((cat, line, w))

# ---------------- CPython reference with duplicates and lexemes kept ----------------
class Pairs(dict):
    # json.dumps of a dict subclass iterates items(): every member, in order
    def __init__(self, ps):
        super().__init__(ps)
        self.ps = ps
    def items(self):
        return self.ps
MARK = '\x00\x01#'
def lexeme(s):
    return MARK + s + '\x00'
def py_load(s):
    return json.loads(s, object_pairs_hook=Pairs, parse_int=lexeme, parse_float=lexeme)
MRE = re.compile(r'"\\u0000\\u0001#([-+.eE0-9]+)\\u0000"')
def py_unmark(t):
    return MRE.sub(lambda m: m.group(1), t)
# strings and numbers boxed, so that jsonpointer sees them as scalars
class Box:
    def __init__(self, x):
        self.x = x
def boxed(v):
    if isinstance(v, str):
        return Box(v)
    if isinstance(v, Pairs):
        return Pairs([(k, boxed(x)) for k, x in v.ps])
    if isinstance(v, list):
        return [boxed(x) for x in v]
    return v
def py_compact(v):
    return py_unmark(json.dumps(v, ensure_ascii=False, separators=(',', ':'), default=lambda o: o.x))
def py_pretty(v, ind):
    return py_unmark(json.dumps(v, ensure_ascii=False, indent=ind))
def has_lone(v):
    if isinstance(v, str):
        return any(0xd800 <= ord(c) <= 0xdfff for c in v)
    if isinstance(v, Pairs):
        return any(has_lone(k) or has_lone(x) for k, x in v.ps)
    if isinstance(v, list):
        return any(has_lone(x) for x in v)
    return False

# the start of the first ill-formed UTF-8 sequence (CPython's strict decoder)
def first_bad(b):
    try:
        b.decode('utf-8')
        return len(b)
    except UnicodeDecodeError as e:
        return e.start
def blit(b):
    return '[' + ', '.join(str(x) for x in b) + ']'
def ilist(xs):
    return '[' + ', '.join(str(x) for x in xs) + ']'

# position (line, col) of byte offset `at` in UTF-8 bytes b
def linecol(b, at):
    pre = b[:at].decode('utf-8')
    ls = pre.split('\n')
    return len(ls), len(ls[-1]) + 1
def errtext(b, at):
    l, c = linecol(b, at)
    return 'ERR %d %d:%d' % (at, l, c)
# my offset from Go's: Go's Offset counts the offending byte; at the end of
# input Go steps one virtual space, so an error found then has Offset equal
# to the length and is about ' ' (when the last real byte is not a space)
def go_at(v, b):
    n = len(b)
    if v['msg'] == 'unexpected end of JSON input':
        return v['offset']
    if v['offset'] == n and "invalid character ' '" in v['msg'] and not b.endswith(b' '):
        return n
    return v['offset'] - 1
def hexval(c):
    return int(c, 16) if c in '0123456789abcdefABCDEF' else -1
# The offset of the first character after which a string escape can only
# be a lone surrogate (Go accepts those; data.json refuses them): after a
# high half, anything but \u, D and C..F; a D then C..F with no high half
# before it. None when there is none (or the grammar fails first).
def lone_at(b):
    s = b.decode('utf-8')
    offs = [0]
    for c in s:
        offs.append(offs[-1] + len(c.encode('utf-8')))
    i, n, ins, hi = 0, len(s), False, False
    while i < n:
        c = s[i]
        if not ins:
            if c == '"':
                ins = True
            i += 1
            continue
        if hi:
            if c != '\\':
                return offs[i]
            if i + 1 >= n:
                return None
            if s[i + 1] != 'u':
                return offs[i + 1]
            for k in range(4):
                j = i + 2 + k
                if j >= n or hexval(s[j]) < 0:
                    return None
                h = hexval(s[j])
                if (k == 0 and h != 13) or (k == 1 and h < 12):
                    return offs[j]
            hi = False
            i += 6
            continue
        if c == '"':
            ins = False
            i += 1
        elif c == '\\':
            if i + 1 >= n:
                return None
            if s[i + 1] == 'u':
                hs = []
                for k in range(4):
                    j = i + 2 + k
                    if j >= n or hexval(s[j]) < 0:
                        return None
                    h = hexval(s[j])
                    if k == 1 and hs[0] == 13 and h >= 12:
                        return offs[j]
                    hs.append(h)
                u = (hs[0] << 12) | (hs[1] << 8) | (hs[2] << 4) | hs[3]
                hi = 0xd800 <= u <= 0xdbff
                i += 6
            else:
                i += 2
        else:
            i += 1
    return None
def want_at(v, b):
    # data.json's offset: Go's, or an earlier lone surrogate
    g = None if v['valid'] else go_at(v, b)
    try:
        l = lone_at(b)
    except UnicodeDecodeError:
        l = None
    if l is not None and (g is None or l < g):
        return l
    return g

# ---------------- random values and texts ----------------
CTL = [chr(i) for i in range(32)]
CHARS = (list('abcXYZ019 ~!@#$%^&*()_+-=[]{}|;:,.<>?/') + CTL + ['"', '\\', '/', '\x7f', '\x80', '\xa0', '\xe9', '\xff',
         'Ā', '߿', 'ࠀ', ' ', ' ', '﻿', '�', '￾', '￿', '\U00010000',
         '\U0001f600', '\U0010ffff', '中', ''])
def rstr(rng, n=None):
    if n is None:
        n = rng.choice([0, 1, 2, 3, 5, 8, 13, 30])
    return ''.join(rng.choice(CHARS) for _ in range(n))
def rnum(rng):
    k = rng.randrange(12)
    if k == 0:
        return str(rng.randrange(-1000, 1000))
    if k == 1:
        return str(rng.randrange(-2**70, 2**70))
    if k == 2:
        return repr(rng.uniform(-1e6, 1e6))
    if k == 3:
        return repr(struct.unpack('<d', struct.pack('<Q', rng.getrandbits(64) & 0x7fefffffffffffff))[0])
    if k == 4:
        return '-0'
    if k == 5:
        return rng.choice(['0', '-0.0', '0.0e0', '1E10', '1e-10', '-1.5E+3', '0.5e-2', '1.000', '123e+4'])
    if k == 6:
        return '%s%d.%de%s%d' % (rng.choice(['', '-']), rng.randrange(10), rng.randrange(10**6), rng.choice(['', '+', '-']), rng.randrange(400))
    if k == 7:
        return str(rng.randrange(10**30)) + '.' + str(rng.randrange(10**20))
    if k == 8:
        return rng.choice(['9223372036854775807', '-9223372036854775808', '9223372036854775808', '1e308', '2e-308', '5e-324'])
    return str(rng.randrange(0, 10**rng.randrange(1, 18)))
def rval(rng, d=0, dups=True):
    k = rng.randrange(10 if d < 4 else 6)
    if k == 0:
        return None
    if k == 1:
        return rng.choice([True, False])
    if k in (2, 3):
        return ('num', rnum(rng))
    if k in (4, 5):
        return rstr(rng)
    if k in (6, 7):
        return [rval(rng, d + 1, dups) for _ in range(rng.choice([0, 1, 2, 3, 5]))]
    ks = [rstr(rng, rng.choice([0, 1, 2, 3, 6])) for _ in range(rng.choice([0, 1, 2, 3, 5]))]
    if dups and len(ks) > 1 and rng.random() < 0.2:
        ks.append(ks[0])
    if not dups:
        ks = list(dict.fromkeys(ks))
    return ('obj', [(x, rval(rng, d + 1, dups)) for x in ks])
WS = [' ', '\t', '\n', '\r', '  ', '\r\n']
def rws(rng):
    return '' if rng.random() < 0.6 else ''.join(rng.choice(WS) for _ in range(rng.randrange(1, 3)))
def esc_char(rng, c):
    o = ord(c)
    if c == '"':
        return rng.choice(['\\"', '\\u0022'])
    if c == '\\':
        return rng.choice(['\\\\', '\\u005c', '\\u005C'])
    short = {'\b': '\\b', '\f': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
    if o < 32:
        if c in short and rng.random() < 0.6:
            return short[c]
        h = '%04x' % o
        return '\\u' + (h.upper() if rng.random() < 0.5 else h)
    if c == '/' and rng.random() < 0.5:
        return '\\/'
    if rng.random() < 0.15:
        if o >= 0x10000:
            u = o - 0x10000
            return '\\u%04x\\u%04X' % (0xd800 + (u >> 10), 0xdc00 + (u & 1023))
        return '\\u%04x' % o if rng.random() < 0.5 else '\\u%04X' % o
    return c
def rtext_str(rng, s):
    return '"' + ''.join(esc_char(rng, c) for c in s) + '"'
def rtext(rng, v):
    # a JSON text of v with random whitespace and random escapes
    if v is None:
        return 'null'
    if v is True:
        return 'true'
    if v is False:
        return 'false'
    if isinstance(v, str):
        return rtext_str(rng, v)
    if isinstance(v, tuple) and v[0] == 'num':
        return v[1]
    if isinstance(v, list):
        if not v:
            return '[' + rws(rng) + ']'
        return '[' + ','.join(rws(rng) + rtext(rng, x) + rws(rng) for x in v) + ']'
    ms = v[1]
    if not ms:
        return '{' + rws(rng) + '}'
    return '{' + ','.join(rws(rng) + rtext_str(rng, k) + rws(rng) + ':' + rws(rng) + rtext(rng, x) + rws(rng) for k, x in ms) + '}'
def rdoc(rng, dups=True):
    v = rval(rng, 0, dups)
    return rws(rng) + rtext(rng, v) + rws(rng)

# ---------------- corpus for the syntax reference ----------------
# string escapes around lone surrogates (Go's verdicts give the grammar's)
LONE = ['"\\uD800"', '["\\ud800x"]', '["a\\uDBFF\\n"]', '"\\uDC00"', '"\\ud83d\\u0041"', '"x\\uD83D\\uDE00\\uDE00"', '{"\\uDFFF":1}',
        '"\\uD800\\uD800\\uDC00"', '"\\uD83D', '"\\uD83D\\', '"\\uD83D\\u', '"\\uD83D\\uD', '"\\uD83D\\uDB', '"\\uD83D\\uE000"', '"\\udbff\\udfff"',
        '"\\uDBFF\\uDFFF\\uD800"', '"\\u00e9\\uDc"', '"\\ud8"', '"\\uD800\\uDC0g"']

MUT = list('{}[]:,"\\ 0123456789eE.+-tfnulr') + ['\x00', '\x1f', '\t', '\n', 'x', 'é', '﻿', '\\u', '\\uD83D']
def mutate(rng, s):
    k = rng.randrange(5)
    if not s:
        return rng.choice(MUT)
    i = rng.randrange(len(s) + 1)
    if k == 0:
        return s[:i] + s[i + 1:]
    if k == 1:
        return s[:i] + rng.choice(MUT) + s[i:]
    if k == 2:
        return s[:i] + rng.choice(MUT) + s[i + 1:]
    if k == 3:
        return s[:i]
    return s[:i] + s[i:i + rng.randrange(1, 4)] + s[i:]
def make_corpus():
    rng = random.Random(SEED + 1)
    xs = []
    seen = set()
    hand = ['', ' ', '[', ']', '{', '}', '[1,]', '{"a":1,}', '[01]', '[-]', '[1.]', '[1e]', '[1e+]', '[.5]', '[+1]', '[0x1]',
            '"\\x"', '"\\u12"', '"\\u12G4"', '"a\tb"', 'nul', 'nulll', 'True', '[1 2]', '{"a" "b"}', '{"a":}', '{1:2}',
            '[] []', '"a" "b"', '1 2', '-01', '00', '1.e5', '1e5.', '[1,2', '{"a":[{"b":', '"\\', '"\\u', '﻿1',
            '/* c */ 1', '1 // c', "['a']", '[NaN]', '[Infinity]', '[-Infinity]', '{"a":1 "b":2}', '[,1]', '[1,,2]',
            '{,}', '{"a":1,,"b":2}', ' 1', '1 ', '[ ]', '"\x00"', '"\x1f"', '"\x7f"']
    for s in hand:
        xs.append(s)
    while len(xs) < 1400:
        s = rdoc(rng)
        for _ in range(rng.randrange(1, 3)):
            s = mutate(rng, s)
        if len(s) > 400 or s in seen:
            continue
        seen.add(s)
        xs.append(s)
    out = []
    for s in xs:
        # lone surrogate escapes: Go accepts them, data.json refuses them;
        # those inputs are checked by the hand vectors instead
        try:
            v = json.loads(s, object_pairs_hook=Pairs)
            if has_lone(v):
                continue
        except Exception:
            if re.search(r'\\u[dD][89a-fA-F]', s):
                continue
        out.append(s.encode('utf-8').hex())
    json.dump({'mutations': out, 'lone': [x.encode('utf-8').hex() for x in LONE]}, open(os.path.join(V, 'syntax_corpus.json'), 'w'), indent=0)
    print('corpus', len(out))

if len(sys.argv) > 1 and sys.argv[1] == 'corpus':
    make_corpus()
    sys.exit(0)

GO = json.load(open(os.path.join(V, 'go1.24_syntax.json')))
NODE = json.load(open(os.path.join(V, 'node22_es.json')))
assert GO['go'].startswith('go1.24'), GO['go']

# ---------------- JSONTestSuite ----------------
# i_ files: what data.json does with each (the suite leaves them open).
I_ACCEPT = {'i_number_double_huge_neg_exp.json', 'i_number_huge_exp.json', 'i_number_neg_int_huge_exp.json',
            'i_number_pos_double_huge_exp.json', 'i_number_real_neg_overflow.json', 'i_number_real_pos_overflow.json',
            'i_number_real_underflow.json', 'i_number_too_big_neg_int.json', 'i_number_too_big_pos_int.json',
            'i_number_very_big_negative_int.json', 'i_structure_500_nested_arrays.json'}
BIG = {'n_structure_100000_opening_arrays.json': (b'[', 100000), 'n_structure_open_array_object.json': (b'[{"":', 50000)}
counts = {'y': 0, 'n': 0, 'i': 0}
i_notes = []
for f in sorted(os.listdir(SUITE)):
    b = open(os.path.join(SUITE, f), 'rb').read()
    g = GO['suite'][f]
    cat = f[0]
    counts[cat] += 1
    try:
        s = b.decode('utf-8')
        call = None
    except UnicodeDecodeError:
        s = None
        call = 'ck_pb(%s)' % blit(b)
    if f in BIG:
        unit, n = BIG[f]
        tail = b[len(unit) * n:]
        assert b == unit * n + tail and tail in (b'', b'\n'), f
        call = 'ck_p(ck_rep(%s, %d)%s)' % (lit(unit.decode()), n, ' + "\\n"' if tail else '')
    elif call is None:
        call = 'ck_p(%s)' % lit(s)
    if cat == 'y':
        assert g['valid'], f
        want = py_compact(py_load(s))
        chk('official', 'check %s == %s' % (call, lit(want)))
    elif cat == 'n':
        assert not g['valid'], f
        at = want_at(g, b) if s is not None else go_at(g, b)
        # data.json validates UTF-8 before the grammar sees it: the first
        # ill-formed byte is the error unless the grammar fails earlier
        if s is None:
            bad = first_bad(b)
            at = min(at, bad) if at >= 0 else bad
            pre = b[:at].decode('utf-8', 'replace')
            ls = pre.split('\n')
            want = 'ERR %d %d:%d' % (at, len(ls), len(ls[-1]) + 1)
        else:
            want = errtext(b, at)
        chk('official', 'check %s == %s' % (call, lit(want)), 3000000 if f in BIG else None)
    else:
        if f in I_ACCEPT:
            want = py_compact(py_load(s))
            chk('official', 'check %s == %s' % (call, lit(want)))
            i_notes.append((f, 'accepted'))
        else:
            # refused: lone surrogate escapes, ill-formed UTF-8, UTF-16, BOM
            if s is not None:
                at = 0 if s.startswith('\ufeff') else want_at(g, b)
                assert at is not None, f
                want = errtext(b, at)
            else:
                bad = first_bad(b)
                # a grammar error before the first ill-formed byte comes first
                gv = GO['suite'][f]
                at = bad
                if not gv['valid'] and go_at(gv, b) < bad:
                    at = go_at(gv, b)
                ls = b[:at].decode('utf-8').split('\n')
                want = 'ERR %d %d:%d' % (at, len(ls), len(ls[-1]) + 1)
            chk('official', 'check %s == %s' % (call, lit(want)))
            i_notes.append((f, 'refused at %s' % want))
print('suite', counts)

# ---------------- syntax corpus against Go 1.24 ----------------
CORPUS = json.load(open(os.path.join(V, 'syntax_corpus.json')))
corpus = CORPUS['mutations']
gv = GO['mutations']
assert len(gv) == len(corpus)
rng = random.Random(SEED + 2)
nc = 0
for h, g in zip(corpus, gv):
    b = bytes.fromhex(h)
    s = b.decode('utf-8')
    if g['valid']:
        v = py_load(s)
        want = py_compact(v)
    else:
        if g['offset'] < 0:
            raise SystemExit('Go gave a non-syntax error for %r' % s)
        want = errtext(b, go_at(g, b))
    chk('derived', 'check ck_p(%s) == %s' % (lit(s), lit(want)))
    nc += 1
    # the same input in chunks
    cs = list(s)
    if len(cs) > 2 and rng.random() < 0.5:
        cuts = sorted(rng.sample(range(1, len(cs)), min(len(cs) - 1, rng.randrange(1, 4))))
        chk('derived', 'check ck_ch(%s, %s) == %s' % (lit(s), ilist(cuts), lit(want)))
print('corpus checks', nc)

# lone surrogates (Go accepts them; data.json refuses them where the
# string can no longer be valid Unicode)
for s, g in zip(LONE, GO['lone']):
    b = s.encode()
    at = want_at(g, b)
    want = errtext(b, at) if at is not None else py_compact(py_load(s))
    chk('derived', 'check ck_p(%s) == %s' % (lit(s), lit(want)))

# ---------------- round trips: CPython parses the text, writes the expected ----------------
rng = random.Random(SEED + 3)
nrt = 0
for k in range(500):
    s = rdoc(rng)
    v = py_load(s)
    assert not has_lone(v)
    c = py_compact(v)
    chk('derived', 'check ck_p(%s) == %s' % (lit(s), lit(c)))
    if k % 2 == 0:
        ind = rng.choice([0, 1, 2, 4])
        chk('derived', 'check ck_pp(%s, %d) == %s' % (lit(s), ind, lit(py_pretty(v, ind))))
    if k % 5 == 0:
        chk('derived', 'check ck_pu(%s, "\\t") == %s' % (lit(s), lit(py_pretty(v, '\t'))))
    if k % 3 == 0:
        # chunked text and chunked UTF-8 bytes (cut inside characters too)
        cs = list(s)
        if len(cs) > 1:
            cuts = sorted(rng.sample(range(1, len(cs)), min(len(cs) - 1, rng.randrange(1, 5))))
            chk('derived', 'check ck_ch(%s, %s) == %s' % (lit(s), ilist(cuts), lit(c)))
        b = s.encode('utf-8')
        if len(b) > 1:
            cuts = sorted(rng.sample(range(1, len(b)), min(len(b) - 1, rng.randrange(1, 6))))
            chk('derived', 'check ck_bch(%s, %s) == %s' % (blit(b), ilist(cuts), lit(c)))
    if k % 4 == 1:
        n = rng.choice([-2, -1, 2])
        mx = rng.choice([1, 2, 3, 7, 50])
        want = py_compact(v) if n == -1 else py_pretty(v, '\t' if n == -2 else n)
        chk('derived', 'check ck_w(%s, %d, %d) == %s' % (lit(s), n, mx, lit(want)))
    # the compact text parses back to itself
    if k % 7 == 0:
        chk('derived', 'check ck_p(%s) == %s' % (lit(c), lit(c)))
    nrt += 1
print('round trips', nrt)

# string literals: every control character, NUL included (CPython dumps)
for i in range(32):
    chk('derived', 'check ck_lit(%s) == %s' % (lit('a' + chr(i) + 'b'), lit(json.dumps('a' + chr(i) + 'b', ensure_ascii=False))))
for s in ['', '"', '\\', '/', '\x7f', ' ', '\U0001f600', '\x00\x00', '"\\\x00\x1f\n']:
    chk('derived', 'check ck_lit(%s) == %s' % (lit(s), lit(json.dumps(s, ensure_ascii=False))))
# long strings: written in pieces (the writer's segment is 16384 characters)
long1 = ''.join(chr(rng.choice([0, 9, 34, 92, 97, 233, 0x1f600])) for _ in range(40000))
chk('derived', 'check ck_lit(ck_rep(%s, 1000)).len_chars() == %d' % (lit(long1[:40]), len(json.dumps(long1[:40] * 1000, ensure_ascii=False))), 3000000)
doc = json.dumps({'k' * 20000: long1[:20000], 'x': [long1[:17000]]}, ensure_ascii=False, separators=(',', ':'))
chk('derived', 'check ck_w(%s, -1, 5000) == %s' % (lit(doc), lit(doc)), 12000000)

# ---------------- numbers ----------------
ctx = decimal.Context(prec=2000, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)
def py_i64(s):
    d = ctx.create_decimal(s)
    if d.is_zero():
        return 0
    if d != d.to_integral_value(context=ctx):
        return None
    if d < -2**63 or d >= 2**63:
        return None
    return int(d)
def flit(x):
    r = repr(x)
    if r.startswith('-'):
        return '(0.0 - ' + r[1:] + ')'
    return r
def num_lexemes(rng):
    xs = ['0', '-0', '0.0', '-0.0', '0e0', '-0E-0', '1', '-1', '1.5', '0.1', '0.2', '0.3', '1e1', '1E+1', '1e-1', '123.456e-7',
          '9007199254740993', '9007199254740992.5', '18014398509481985', '1e22', '1e23', '8.98846567431158e307',
          '1.7976931348623157e308', '1.7976931348623158e308', '1.7976931348623159e308', '1e308', '1e309', '-1e309',
          '2.2250738585072011e-308', '2.2250738585072014e-308', '4.9406564584124654e-324', '2.4703282292062327e-324',
          '2.4703282292062328e-324', '2.4703282292062327208e-324', '1e-324', '1e-400', '-1e-400', '1e-999999999999',
          '1e999999999999', '0e999999999999', '123456789012345678901234567890', '0.000000000000000000000000000001',
          '3.141592653589793238462643383279502884197169399375105820974944592307816406286',
          '7.2057594037927933e16', '1448997445238699', '5e-324', '9.007199254740993e15', '1.00000000000000011102230246251565404236316680908203125',
          '1.000000000000000111022302462515654042363166809082031250000000000000001', '1.00000000000000011102230246251565404236316680908203124',
          '9223372036854775808', '9223372036854775807', '-9223372036854775809', '9.223372036854775808e18', '92233720368547758080e-1',
          '-92233720368547758080e-1', '9.223372036854775807e18']
    # a midpoint between neighbouring doubles (the lower one even), its
    # digits padded beyond 800 and then a 1: only the sticky digit says the
    # value is above the midpoint
    for _ in range(4):
        while True:
            q = rng.getrandbits(62) % 0x7fe0000000000000
            if q % 2 == 0:
                break
        x = struct.unpack('<d', struct.pack('<Q', q))[0]
        y = struct.unpack('<d', struct.pack('<Q', q + 1))[0]
        m = ctx.divide(ctx.add(decimal.Decimal(x), decimal.Decimal(y)), 2)
        sg, dg, ex = m.as_tuple()
        ds = ''.join(map(str, dg))
        pad = 850 - len(ds)
        xs.append(ds[0] + '.' + ds[1:] + '0' * pad + '1' + 'e' + str(ex + len(ds) - 1))
    for _ in range(300):
        k = rng.randrange(6)
        if k == 0:
            xs.append(rnum(rng))
        elif k == 1:
            # exact midpoints between neighbouring doubles: ties to even
            x = struct.unpack('<d', struct.pack('<Q', rng.getrandbits(63) % 0x7fe0000000000000))[0]
            y = struct.unpack('<d', struct.pack('<Q', struct.unpack('<Q', struct.pack('<d', x))[0] + 1))[0]
            m = ctx.divide(ctx.add(decimal.Decimal(x), decimal.Decimal(y)), 2)
            xs.append(format(ctx.plus(m), 'e') if rng.random() < 0.5 else str(ctx.plus(m)))
        elif k == 2:
            # long digit strings
            xs.append(str(rng.randrange(1, 10)) + ''.join(rng.choice('0123456789') for _ in range(rng.randrange(17, 900))) + 'e' + str(rng.randrange(-1200, 400)))
        elif k == 3:
            # subnormal and near overflow
            e = rng.choice([-323, -320, -310, -308, -307, 305, 307, 308])
            xs.append('%d.%de%d' % (rng.randrange(1, 10), rng.randrange(10**15), e))
        elif k == 4:
            xs.append('%s%d' % (rng.choice(['', '-']), rng.randrange(2**63 - 3000, 2**63 + 3000)))
        else:
            xs.append(repr(rng.random() * 10**rng.randrange(-30, 30)))
    return xs
rng = random.Random(SEED + 4)
nn = 0
for s in num_lexemes(rng):
    assert re.fullmatch(r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?', s), s
    x = float(s)
    if x in (float('inf'), float('-inf')):
        chk('derived', 'check ck_fe(%s) == true' % lit(s))
    elif x == 0.0:
        chk('derived', 'check ck_fz(%s) == %s' % (lit(s), lit('-0' if str(x).startswith('-') else '0')))
    else:
        chk('derived', 'check ck_fv(%s) == %s' % (lit(s), flit(x)))
    iv = py_i64(s)
    chk('derived', 'check ck_iv(%s) == %s' % (lit(s), lit('ERR' if iv is None else str(iv))))
    nn += 1
print('numbers', nn)
# lexeme validity
for s, ok in [('1', True), ('-', False), ('01', False), ('1.', False), ('.1', False), ('1e', False), ('+1', False), ('-0', True),
              ('1E+2', True), ('1e+', False), ('0x10', False), (' 1', False), ('1 ', False), ('', False), ('Infinity', False), ('NaN', False)]:
    chk('derived', 'check ck_ok(%s) == %s' % (lit(s), 'true' if ok else 'false'))

# ECMAScript number text: the head of the RFC 8785 es6 test file (official)
ES = open(os.path.join(V, 'rfc8785/es6testfile100m_head2000.txt')).read().split('\n')
ne = 0
for line in ES[:2000]:
    if not line:
        continue
    hx, want = line.split(',')
    x = struct.unpack('>d', bytes.fromhex(hx.rjust(16, '0')))[0]
    chk('official', 'check f64_text(%s) == %s' % (flit(x), lit(want)))
    ne += 1
# Node 22 String(x) of random doubles and RFC 8785 number forms of lexemes (derived)
for hx, want in NODE['es']:
    x = struct.unpack('>d', bytes.fromhex(hx))[0]
    chk('derived', 'check f64_text(%s) == %s' % (flit(x), lit(want)))
for s, want in NODE['canon_num']:
    chk('derived', 'check ck_nc(%s) == %s' % (lit(s), lit(want if want is not None else 'ERR')))
print('es', ne, len(NODE['es']), len(NODE['canon_num']))

# ---------------- canonical form (RFC 8785) ----------------
for f in sorted(os.listdir(os.path.join(V, 'rfc8785/input'))):
    s = open(os.path.join(V, 'rfc8785/input', f), encoding='utf-8').read()
    want = open(os.path.join(V, 'rfc8785/output', f), encoding='utf-8').read()
    chk('official', 'check ck_cn(%s) == %s' % (lit(s), lit(want)))
for s, want in NODE['canon']:
    chk('derived', 'check ck_cn(%s) == %s' % (lit(s), lit(want)))
for s, want in [('{"a":1,"a":2}', 'ERR duplicate member name at /a'), ('[{"b":{"x":1,"y":[1e999],"x":2}}]', 'ERR duplicate member name at /0/b/x'),
                ('[1,{"a":[2,1e400]}]', 'ERR number outside the f64 range: 1e400 at /1/a/1'), ('{"~/":-1e999}', 'ERR number outside the f64 range: -1e999 at /~0~1'),
                ('-0', '0'), ('[-0.0,1E-7,1e21,1e+20,0.000001]', '[0,1e-7,1e+21,100000000000000000000,0.000001]')]:
    chk('derived', 'check ck_cn(%s) == %s' % (lit(s), lit(want)))
# enc.jcs (integer numbers) agrees with to_canonical
rng = random.Random(SEED + 5)
for _ in range(40):
    def ival(d=0):
        k = rng.randrange(6 if d < 3 else 4)
        if k == 0:
            return None
        if k == 1:
            return rng.choice([True, False])
        if k == 2:
            return rng.randrange(-2**53, 2**53)
        if k == 3:
            return rstr(rng)
        if k == 4:
            return [ival(d + 1) for _ in range(rng.randrange(4))]
        return {rstr(rng, rng.randrange(4)): ival(d + 1) for _ in range(rng.randrange(4))}
    chk('derived', 'check ck_jcs(%s) == true' % lit(json.dumps(ival(), ensure_ascii=False)))

# ---------------- JSON Pointer (RFC 6901) ----------------
import jsonpointer
RFC = json.load(open(os.path.join(V, 'rfc6901_examples.json')))
rdoc_t = json.dumps(RFC['document'], ensure_ascii=False)
for p, want in RFC['pointers']:
    chk('official', 'check ck_ptr(%s, %s) == %s' % (lit(rdoc_t), lit(p), lit(json.dumps(want, ensure_ascii=False, separators=(',', ':')))))
for fr, want in RFC['fragments']:
    chk('official', 'check ck_frag(%s) == %s' % (lit(fr), lit(want)))
for fr in ['#/%', '#/%4', '#/%zz', '#/%C0%A0', 'foo', '#/%E2%82%AC', '#/%F0%9F%98%80']:
    try:
        want = __import__('urllib.parse').parse.unquote(fr[1:], errors='strict') if fr.startswith('#') else None
        if fr.startswith('#') and re.search(r'%(?![0-9A-Fa-f]{2})', fr):
            want = None
    except UnicodeDecodeError:
        want = None
    chk('derived', 'check ck_frag(%s) == %s' % (lit(fr), lit(want if want is not None else '\x00ERR')))
def paths(v, pre):
    yield pre
    if isinstance(v, Pairs):
        for k, x in v.ps:
            yield from paths(x, pre + [k])
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from paths(x, pre + [str(i)])
def ptr(toks):
    return ''.join('/' + t.replace('~', '~0').replace('/', '~1') for t in toks)
rng = random.Random(SEED + 6)
npt = 0
for _ in range(150):
    s = rdoc(rng)
    v = boxed(py_load(s))
    ps = list(paths(v, []))
    for _ in range(4):
        toks = list(rng.choice(ps))
        k = rng.randrange(6)
        if k == 1 and toks:
            toks[-1] = toks[-1] + rng.choice(['x', '~', '/'])
        elif k == 2:
            toks.append(rng.choice(['0', '5', '-', 'a', '', '~']))
        elif k == 3 and toks:
            toks = toks[:-1] + ['99']
        p = ptr(toks)
        if rng.random() < 0.05:
            p = p + '~2'
        if re.search(r'/0[0-9]', p):
            continue  # jsonpointer 3.0.0 accepts leading zeros; RFC 6901 does not (hand vectors below)
        try:
            r = jsonpointer.resolve_pointer(v, p)
            if isinstance(r, jsonpointer.EndOfList):
                raise jsonpointer.JsonPointerException('end of list')
            want = py_compact(r)
        except (jsonpointer.JsonPointerException, KeyError, IndexError, TypeError, ValueError, AssertionError):
            want = None
        if want is None:
            chk('derived', 'check ck_ptr(%s, %s).startswith("\\"\\\\u0000ERR") == true' % (lit(s), lit(p)))
        else:
            chk('derived', 'check ck_ptr(%s, %s) == %s' % (lit(s), lit(p), lit(want)))
        npt += 1
print('pointers', npt)
for d, p, want in [('{"a":[1,2]}', '/a/01', '"\\u0000ERR /a/01"'), ('{"a":[1,2]}', '/a/-', '"\\u0000ERR /a/-"'), ('{"a":[1,2]}', '/a/2', '"\\u0000ERR /a/2"'),
                   ('{"a":[1,2]}', 'a', '"\\u0000ERR a"'), ('{"a":1,"a":2}', '/a', '2'), ('{"a":{"b":3}}', '/a/b/c', '"\\u0000ERR /a/b/c"'),
                   ('[[0,[1,[2]]]]', '/0/1/1/0', '2'), ('{"":{"":5}}', '//', '5'), ('{"a":[1,2]}', '/a/+1', '"\\u0000ERR /a/+1"'),
                   ('{"a":[1,2]}', '/a/1e0', '"\\u0000ERR /a/1e0"'), ('{"a":[1,2]}', '/a/99999999999999999999', '"\\u0000ERR /a/99999999999999999999"')]:
    chk('derived', 'check ck_ptr(%s, %s) == %s' % (lit(d), lit(p), lit(want)))
# set and remove against jsonpointer.set_pointer and del on the parent
rng = random.Random(SEED + 7)
import copy as pycopy
nset = 0
for _ in range(120):
    s = rdoc(rng, dups=False)
    v = boxed(py_load(s))
    ps = [x for x in paths(v, []) if x]
    if not ps:
        continue
    toks = list(rng.choice(ps))
    newv = rval(rng, 2, False)
    nt = rtext(rng, newv)
    nv = boxed(py_load(nt))
    k = rng.randrange(3)
    if k == 1:
        toks = toks[:-1] + [rng.choice(['-', 'zz'])]
    p = ptr(toks)
    try:
        par = jsonpointer.JsonPointer(ptr(toks[:-1])).resolve(v)
        if isinstance(par, list) and toks[-1] not in ('-',) and not re.fullmatch(r'0|[1-9][0-9]*', toks[-1]):
            raise TypeError
        if isinstance(par, Pairs):
            # set on a Pairs value: keep member order (replace in place or append)
            v2 = pycopy.deepcopy(v)
            par2 = jsonpointer.JsonPointer(ptr(toks[:-1])).resolve(v2)
            ps2 = par2.ps
            hit = [i for i, (a, _b) in enumerate(ps2) if a == toks[-1]]
            if hit:
                ps2[hit[-1]] = (toks[-1], nv)
            else:
                ps2.append((toks[-1], nv))
            dict.__setitem__(par2, toks[-1], nv)
            want = py_compact(v2)
        elif isinstance(par, list):
            v2 = jsonpointer.set_pointer(pycopy.deepcopy(v), p, nv, inplace=False)
            want = py_compact(v2)
        else:
            raise TypeError
    except (jsonpointer.JsonPointerException, KeyError, IndexError, TypeError, AssertionError):
        want = None
    if want is not None:
        chk('derived', 'check ck_set(%s, %s, %s) == %s' % (lit(s), lit(p), lit(nt), lit(want)))
        nset += 1
    # remove the location the path names
    p = ptr(list(rng.choice(ps)))
    v3 = pycopy.deepcopy(v)
    try:
        pp, last = jsonpointer.JsonPointer(p).to_last(v3)
        if isinstance(pp, Pairs):
            pp.ps[:] = [(a, b) for a, b in pp.ps if a != last]
            dict.__delitem__(pp, last)
        else:
            del pp[last]
        want = py_compact(v3)
    except (jsonpointer.JsonPointerException, KeyError, IndexError, TypeError, AssertionError):
        want = None
    if want is not None:
        chk('derived', 'check ck_del(%s, %s) == %s' % (lit(s), lit(p), lit(want)))
        nset += 1
print('set/del', nset)
for d, p, v, want in [('{"a":1}', '', '[2]', '[2]'), ('[1]', '/1', '2', '[1,2]'), ('[1]', '/2', '2', '"\\u0000ERR /2"'), ('{"a":1}', '/b/c', '2', '"\\u0000ERR /b"'),
                      ('{"a":1,"a":2}', '/a', '3', '{"a":1,"a":3}'), ('5', '/a', '1', '"\\u0000ERR /a"')]:
    chk('derived', 'check ck_set(%s, %s, %s) == %s' % (lit(d), lit(p), lit(v), lit(want)))
for d, p, want in [('{"a":1}', '', '"\\u0000ERR "'), ('[1,2,3]', '/1', '[1,3]'), ('[1]', '/1', '"\\u0000ERR /1"'), ('{"a":1,"b":2,"a":3}', '/a', '{"a":1,"b":2}'),
                   ('{"a":{"b":[0,{"c":1}]}}', '/a/b/1/c', '{"a":{"b":[0,{}]}}')]:
    chk('derived', 'check ck_del(%s, %s) == %s' % (lit(d), lit(p), lit(want)))

# typed getters
G = '{"s":"x\\u00e9","b":true,"i":-9223372036854775808,"f":1e-7,"a":[1,[2],{}],"o":{"k":1,"k2":null},"big":1e400,"frac":2.5,"n":null}'
for p, k, want in [('/s', 'text', 'xé'), ('/b', 'text', 'ERR expected a string, found boolean at /b'), ('/b', 'bool', 'true'), ('/i', 'i64', '-9223372036854775808'),
                   ('/frac', 'i64', '-1'), ('/f', 'f64', '1e-7'), ('/big', 'f64', '-1'), ('/a', 'items', '3'), ('/o', 'keys', 'k,k2'), ('/n', 'keys', 'ERR'),
                   ('/zz', 'text', 'ERR no member named "zz" at /zz'), ('/a/1/0', 'i64', '2'), ('/a/1/0', 'f64', '2'), ('/s/0', 'text', 'ERR a string has no members at /s/0')]:
    chk('derived', 'check ck_get(%s, %s, %s) == %s' % (lit(G), lit(p), lit(k), lit(want)))

# ---------------- streaming: partial results and first values ----------------
for s, want in [('', 'null'), ('{"ab', '{}'), ('{"a":1,"b', '{"a":1}'), ('[{"x', '[{}]'), ('{"a":[1,2', '{"a":[1,2]}'), ('{"a":[1,2.', '{"a":[1]}'), ('{"a":"he', '{"a":"he"}'), ('{"a"', '{}'), ('{"a":', '{}'),
                ('[tr', '[]'), ('[1,{"b":[{"c":"x\\u00', '[1,{"b":[{"c":"x"}]}]'), ('"abc', '"abc"'), ('12', '12'), ('-', 'null'), ('[1,2]', '[1,2]'),
                ('[[[[', '[[[[]]]]'), ('{"k":{"j":null', '{"k":{"j":null}}'), ('{"k":{"j":nul', '{"k":{}}')]:
    chk('derived', 'check ck_partial(%s) == %s' % (lit(s), lit(want)))
for s, at, want in [('["é"]x', 0, '["é"] 6'), ('1 ["\U0001f600"]', 1, '["\U0001f600"] 10'), ('1 2', 0, '1 1'), ('1 2', 1, '2 3'), ('[1]{"a":2}', 3, '{"a":2} 10'), ('  "x"  ', 0, '"x" 5'), (' 12', 0, '12 3'), ('é[]', 2, '[] 4'),
                    ('é[]', 1, '"ERR start offset inside a character" 1'), ('[1', 0, '"ERR unexpected end of input" 2'), ('', 0, '"ERR unexpected end of input" 0'),
                    ('1', 5, '"ERR start offset outside the text" 5'), ('\n\n x', 0, '"ERR expected a value" 3')]:
    chk('derived', 'check ck_first(%s, %d) == %s' % (lit(s), at, lit(want)))

# ---------------- JSON Lines ----------------
rng = random.Random(SEED + 8)
JL_BAD = {'[1,': None, '{"a"}': 4, 'nul': 3, '1 2': 2, '"\\x"': 2, ']': 0}
njl = 0
for _ in range(60):
    lines = []
    for _ in range(rng.randrange(1, 7)):
        k = rng.randrange(6)
        if k == 0:
            lines.append(rng.choice(['', ' ', '\t ', '\r']))
        elif k == 1:
            lines.append(rng.choice(sorted(JL_BAD)))
        else:
            lines.append(rtext(rng, rval(rng, 1)).replace('\n', ' ').replace('\r', ' '))
    t = '\n'.join(lines) + rng.choice(['', '\n', '\r\n'])
    want = []
    at = 0
    for n, l in enumerate(t.split('\n'), 1):
        lb = l.encode('utf-8')
        if l.strip(' \t\r') != '':
            try:
                want.append('%d %s' % (n, py_compact(py_load(l))))
            except ValueError:
                # the bad lines are fixed texts; the first offending character
                # (the end of the line for an unfinished array)
                bad = l.rstrip('\r')
                off = JL_BAD[bad]
                if off is None:
                    off = len(lb)
                want.append('%d ERR %d %d:%d' % (n, at + off, n, len(lb[:off].decode('utf-8')) + 1))
        at += len(lb) + 1
    chk('derived', 'check ck_jl(%s) == %s' % (lit(t), lits(want)))
    cs = list(t)
    if len(cs) > 1:
        cuts = sorted(rng.sample(range(1, len(cs)), min(len(cs) - 1, rng.randrange(1, 5))))
        chk('derived', 'check ck_jlc(%s, %s) == %s' % (lit(t), ilist(cuts), lits(want)))
    njl += 1
print('jsonl', njl)
chk('derived', 'check ck_jlt() == "null\\n[1]\\n"')
chk('derived', 'check ck_jl("") == []')

# ---------------- every cut of a few tricky texts ----------------
TRICKY = ['{"a\\u00e9\\ud83d\\ude00":[-12.5e+3,true,null,"\\\\\\"\\/"]}', '[0,-0,1E2,0.25,false,"x\\u0000y"]',
          ' {"\u00e9\U0001f600":{"":[]}} ', '[1,2', '["a\\uD83Dx"]', '[nul]', '[1.e2]', '{"a" :1 ,"b":[ ]}']
for t in TRICKY:
    b = t.encode('utf-8')
    try:
        v = py_load(t)
        want = None if has_lone(v) else py_compact(v)
    except ValueError:
        want = None
    if want is None:
        k = {'[1,2': len(b), '["a\\uD83Dx"]': lone_at(b), '[nul]': 4, '[1.e2]': 3}[t]
        want = errtext(b, k)
    for k in range(1, len(t)):
        chk('derived', 'check ck_ch(%s, [%d]) == %s' % (lit(t), k, lit(want)))
    for k in range(1, len(b)):
        chk('derived', 'check ck_bch(%s, [%d]) == %s' % (blit(b), k, lit(want)))

# ---------------- nesting depth ----------------
chk('derived', 'check ck_deep(10000) == "text canon [] writer partial"', 12000000)
chk('derived', 'check ck_p(ck_rep("[", 10001) + ck_rep("]", 10001)) == "ERR 10000 1:10001"', 3000000)
chk('derived', 'check ck_p(ck_rep("{\\"a\\":", 10000) + "1" + ck_rep("}", 10000)).len_chars() == 60001', 6000000)
for t, d, want in [('[[]]', 2, '[[]]'), ('[[[]]]', 2, '"ERR 2"'), ('{"a":{"b":1}}', 1, '"ERR 5"'), ('1', 1, '1'), ('[]', 0, '[]'), ('[[]]', 0, '"ERR 1"')]:
    chk('derived', 'check ck_depth(%s, %d) == %s' % (lit(t), d, lit(want)))

# ---------------- the elements of a large array, taken while reading ----------------
rng = random.Random(SEED + 9)
for _ in range(40):
    s = rdoc(rng)
    if rng.random() < 0.7:
        s = '[' + ','.join(rdoc(rng) for _ in range(rng.randrange(0, 6))) + ']'
    v = py_load(s)
    if isinstance(v, list):
        want = ';'.join(py_compact(x) for x in v) + ' | []'
    else:
        want = ' | ' + py_compact(v)
    cs = list(s)
    cuts = sorted(rng.sample(range(1, len(cs)), min(len(cs) - 1, rng.randrange(0, 5)))) if len(cs) > 1 else []
    chk('derived', 'check ck_take(%s, %s) == %s' % (lit(s), ilist(cuts), lit(want)))
chk('derived', 'check ck_take("[1,[2],3", [3]) == %s' % lit('1;[2];3 | "ERR 8" !'))

# ---------------- builder ----------------
chk('derived', 'check ck_build() == %s' % lit('{"a":-5,"b":[null,true,0.1,1e+21,"x\\u0000"]} "member names and values differ in number"'))
chk('derived', 'check ck_with() == %s' % lit('{"a":2,"b":"a"} "expected an array, found object" [null,false]'))
for t, want in [('1.5e3', '1.5e3'), ('-0', '-0'), ('01', '"not a JSON number: 01"'), ('1.', '"not a JSON number: 1."'), ('', '"not a JSON number: "')]:
    chk('derived', 'check ck_num(%s) == %s' % (lit(t), lit(want)))

# ---------------- shards ----------------
HEAD = open(os.path.join(ROOT, 'scripts/gen/json_head.rce')).read()
# one comptime fuel budget (50,000,000) per file; shards are filled to
# an estimated 25,000,000
BUDGET = int(os.environ.get('BUDGET', '25000000'))
shards = []
cur = []
wsum = 0
for cat, line, w in out:
    if wsum + w > BUDGET and cur:
        shards.append(cur)
        cur = []
        wsum = 0
    cur.append(line)
    wsum += w
if cur:
    shards.append(cur)
for f in os.listdir(os.path.join(ROOT, 'tests/data')):
    if f.startswith('json_checks'):
        os.remove(os.path.join(ROOT, 'tests/data', f))
for k, sh in enumerate(shards):
    name = 'json_checks.rce' if k == 0 else 'json_checks_%d.rce' % (k + 1)
    open(os.path.join(ROOT, 'tests/data', name), 'w').write(HEAD + '\n'.join(sh) + '\n')
official = sum(1 for c, _l, _w in out if c == 'official')
print('official', official, 'derived', len(out) - official, 'shards', len(shards))
json.dump([[f, n] for f, n in i_notes], open(os.path.join(V, 'i_decisions.json'), 'w'), indent=1)
