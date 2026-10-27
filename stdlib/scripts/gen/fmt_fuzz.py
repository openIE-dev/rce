# Random differential tests of lib/fmt/fmt.rce against CPython 3.13 (not
# part of check.sh; the checks it writes are not kept).
# Usage: RCE=path/to/rce python3 -I scripts/gen/fmt_fuzz.py MODE SEED COUNT [PNF_DIR]
# MODE: regex (search, match and fullmatch spans of random patterns),
# regex-api (finditer, sub, split, search from an offset, chunked
# search), float-fmt (random binary64 values and format specs),
# float-parse (random decimal text: long digit strings, midpoints between
# neighbouring values and points just beside them), corpus (COUNT random
# lines of each data file of a parse-number-fxx-test-data checkout).
# Each run writes the module, scripts/gen/fmt_ext_helpers.rce and the
# checks to one file in a temporary directory, runs `rce test` on it and
# prints the failing checks. A pattern on which CPython's backtracking
# takes more than a second is skipped.
import decimal, math, os, random, re, signal, struct, subprocess, sys, tempfile
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
RCE = os.environ.get('RCE', 'rce')
mode, seed, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = random.Random(seed * 7919 + len(mode))
D = decimal.Decimal
decimal.getcontext().prec = 1200

def q(s):
    o = ''
    for ch in s:
        if ch == '\\': o += '\\\\'
        elif ch == '"': o += '\\"'
        elif not ch.isprintable() or (ord(ch) & 0xFFFE) == 0xFFFE: o += '\\u{%x}' % ord(ch)
        else: o += ch
    return '"' + o + '"'

def lst(xs): return '[' + ', '.join(map(str, xs)) + ']'
def tl(xs): return '[' + ', '.join(q(x) for x in xs) + ']'
def bo(s, i): return len(s[:i].encode())
def fb(b): return struct.unpack('<d', struct.pack('<Q', b))[0]

def slots(m, s):
    if m is None: return []
    o = []
    for k in range(m.re.groups + 1):
        a, b = m.span(k)
        o += [bo(s, a), bo(s, b)] if a >= 0 else [-1, -1]
    return o

class Slow(Exception): pass
def alarm(*_): raise Slow()
signal.signal(signal.SIGALRM, alarm)

def atom(d):
    k = r.randrange(13 if d < 3 else 6)
    if k < 3: return r.choice(['a', 'b', 'é'])
    if k == 3: return '.'
    if k == 4: return r.choice(['[ab]', '[^a]', '\\w', '\\s', '[a-ÿ]', '\\W', '\\d'])
    if k == 5: return r.choice(['', 'a', 'b'])
    if k in (6, 7, 10): return '(' + alt(d + 1) + ')'
    if k == 8: return '(?:' + alt(d + 1) + ')'
    if k == 9: return r.choice(['^', '$', '\\b', '\\B'])
    return '(?P<g%d>' % r.randrange(10 ** 6) + alt(d + 1) + ')'

def quant(a):
    if a in ('^', '$', '\\b', '\\B', ''): return a
    qq = r.choice(['', '', '', '*', '+', '?', '{0,2}', '{1,3}', '{2}', '{,2}', '{2,}'])
    if qq and r.random() < 0.35: qq += '?'
    return a + qq

def seq(d): return ''.join(quant(atom(d)) for _ in range(r.randint(1, 3)))
def alt(d): return '|'.join(seq(d) for _ in range(r.randint(1, 2 if d < 2 else 1)))

def rand_double():
    k = r.randrange(8)
    if k == 0: return fb(r.getrandbits(52))
    if k == 1: return r.randint(-10 ** 6, 10 ** 6) / 10 ** r.randint(0, 8)
    if k == 2: return float(r.randint(-2 ** 62, 2 ** 62))
    if k == 3: return (r.randint(0, 2 ** 20) + 0.5) / 2 ** r.randint(0, 30)
    x = fb((r.randint(1, 2046) << 52) | r.getrandbits(52))
    return -x if r.random() < 0.4 else x

def rand_spec():
    o = ''
    if r.random() < 0.3: o += r.choice(['', '*', '0', 'é']) + r.choice('<>=^')
    if r.random() < 0.3: o += r.choice('+- ')
    if r.random() < 0.1: o += 'z'
    if r.random() < 0.2: o += '#'
    if r.random() < 0.2: o += '0'
    if r.random() < 0.4: o += str(r.randint(0, 30))
    if r.random() < 0.2: o += r.choice(',_')
    if r.random() < 0.7: o += '.' + str(r.choice([0, 1, 2, 3, 5, 10, 16, 17, 20, 30, r.randint(0, 60)]))
    if r.random() < 0.85: o += r.choice(['e', 'E', 'f', 'F', 'g', 'G', 'n', '%', ''])
    return o

def regex_lines():
    p = alt(0)
    try:
        signal.alarm(1)
        py = re.compile(p.replace('$', '\\Z'), re.ASCII)
        out = []
        for _ in range(3):
            s = ''.join(r.choice('abé ') for _ in range(r.randint(0, 7)))
            out.append('check ck_rx(%s, %s, %s, %s, %s)' % (q(p), q(s), lst(slots(py.search(s), s)), lst(slots(py.match(s), s)), lst(slots(py.fullmatch(s), s))))
        if mode == 'regex':
            return out
        out = []
        s = ''.join(r.choice('abé ') for _ in range(r.randint(0, 7)))
        ms = list(py.finditer(s))
        flat = []
        for m in ms: flat += slots(m, s)
        out.append('check ck_rall(%s, %s, %s)' % (q(p), q(s), lst(flat)))
        t = r.choice(['<\\g<0>>', '[\\1]', '\\1\\1-', '{\\g<1>}x', '\\\\\\n'])
        if py.groups < 1: t = '<\\g<0>>'
        c = r.choice([0, 0, 1, 2])
        out.append('check ck_rsub(%s, %s, %s, %d, %s)' % (q(p), q(s), q(t), c, q(py.sub(t, s, count=c))))
        mx = r.choice([0, 0, 1, 2])
        pieces, last, n = [], 0, 0
        for m in ms:
            if mx and n >= mx: break
            pieces.append(s[last:m.start()]); last = m.end(); n += 1
        pieces.append(s[last:])
        out.append('check ck_rsplit(%s, %s, %d, %s)' % (q(p), q(s), mx, tl(pieces)))
        pos = r.randint(0, len(s))
        out.append('check ck_rat(%s, %s, %d, %s)' % (q(p), q(s), bo(s, pos), lst(slots(py.search(s, pos), s))))
        out.append('check ck_rscan(%s, %s, %d, %s)' % (q(p), q(s), r.randint(1, 3), lst(slots(py.search(s), s))))
        return out
    except (re.error, Slow):
        return []
    finally:
        signal.alarm(0)

def float_fmt_lines():
    x, s = rand_double(), rand_spec()
    try:
        t = format(x, s)
    except ValueError:
        return ['check ck_ff_bad(%r, %s)' % (x, q(s))]
    return ['check ck_ff(%r, %s, %s)' % (x, q(s), q(t))] if len(t) <= 300 else []

def float_parse_lines():
    k = r.randrange(6)
    if k == 0:
        t = str(r.randint(1, 10 ** r.randint(1, 25))) + 'e' + str(r.randint(-340, 320))
    elif k == 1:
        t = '0.' + '0' * r.randint(0, 20) + str(r.randint(1, 10 ** r.randint(1, 40)))
    elif k == 2:
        x = abs(rand_double()); y = math.nextafter(x, math.inf)
        if math.isinf(y) or x == 0: return []
        mid = (D(x) + D(y)) / 2
        j = r.randrange(3)
        if j == 1: mid += D(10) ** (mid.adjusted() - r.randint(17, 780))
        if j == 2: mid -= D(10) ** (mid.adjusted() - r.randint(17, 780))
        t = format(mid, 'e')
    elif k == 3:
        t = repr(rand_double())
    elif k == 4:
        t = ''.join(r.choice('0123456789') for _ in range(r.randint(1, 900))) + 'e' + str(r.randint(-1200, 400))
    else:
        t = r.choice(['', '-', '+']) + str(r.randint(0, 10 ** 18)) + '.' + str(r.randint(0, 10 ** 18)) + r.choice(['', 'e' + str(r.randint(-30, 30))])
    v = float(t)
    return ['check ck_fpo(%s)' % q(t)] if math.isinf(v) else ['check ck_fp(%s, %r)' % (q(t), v)]

lines = []
if mode == 'corpus':
    d = os.path.join(sys.argv[4], 'data')
    for name in sorted(os.listdir(d)):
        ls = [l for l in open(os.path.join(d, name)).read().split('\n') if l]
        for i in r.sample(range(len(ls)), min(N, len(ls))):
            h16, h32, h64, s = ls[i].split()
            x = fb(int(h64, 16))
            lines.append('check ck_fpo(%s)' % q(s) if math.isinf(x) else 'check ck_fp(%s, %r)' % (q(s), x))
else:
    gen = {'regex': regex_lines, 'regex-api': regex_lines, 'float-fmt': float_fmt_lines, 'float-parse': float_parse_lines}[mode]
    while len(lines) < N:
        lines += gen()
helpers = open(os.path.join(root, 'scripts/gen/fmt_ext_helpers.rce')).read()
mod = open(os.path.join(root, 'lib/fmt/fmt.rce')).read()
L = mod.count('\n') + 1 + helpers.count('\n') + 1
with tempfile.TemporaryDirectory() as tmp:
    f = os.path.join(tmp, 'fuzz.rce')
    open(f, 'w').write(mod + '\n' + helpers + '\n' + '\n'.join(lines) + '\n')
    out = subprocess.run([RCE, 'test', f], capture_output=True, text=True).stdout
bad = re.findall(r'line (\d+): ([^;\n]*)', out)
print(mode, 'seed', seed, len(lines), 'checks,', len(bad), 'failing')
for ln, why in bad[:20]:
    k = int(ln) - L - 1
    print('  ', why, lines[k][:400] if 0 <= k < len(lines) else ln)
sys.exit(1 if bad else 0)
