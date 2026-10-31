#!/usr/bin/env python3
"""Vectors for lib/math: expected values are MPFR (gmpy2, ieee(64) context,
subnormalized, round to nearest even), so each is the correctly rounded
binary64 result. Inputs: CORE-MATH worst cases (<fn>.wc, MIT), random and
edge cases. Usage: math_vec.py [cm DIR] [sample N] [out DIR] [shard S] FN
Reads FN.wc from vectors/math/ (cbrt, hypot: copied in) or from a CORE-MATH
checkout (cm DIR = core-math/src/binary64, commit dae5dd8c, MIT; the
larger files are not copied), keeps every k-th worst case so about N remain
(sample N; the repo's push-CI shards use it), and writes
OUT/tests/math/math_checks_FN_N.rce (OUT defaults to the repo).
Needs gmpy2 (pip install gmpy2; MPFR 4.2.2 was used)."""
import sys, os, random, struct, gmpy2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../scripts'))
from positional import read  # noqa: E402  (positional-cli.md PC-1)
from gmpy2 import mpfr

CTX = gmpy2.ieee(64); CTX.subnormalize = True

def trigpi(kind):
    # x mod 2 is exact; pi r at 4400 bits, then one rounding in CTX
    def g(x):
        with gmpy2.context(precision=4400, emin=-10**6, emax=10**6, subnormalize=False):
            r = x - 2 * gmpy2.floor(x / 2)
            if kind == 'tan' and gmpy2.is_integer(2 * r) and not gmpy2.is_integer(r): return mpfr('inf')
            v = {'sin': gmpy2.sin, 'cos': gmpy2.cos, 'tan': gmpy2.tan}[kind](gmpy2.const_pi() * r)
        return +v
    return g

FNS = {
 'cbrt': (1, lambda x: gmpy2.cbrt(x)),
 'hypot': (2, lambda x, y: gmpy2.hypot(x, y)),
 'exp': (1, lambda x: gmpy2.exp(x)),
 'log': (1, lambda x: gmpy2.log(x)),
 'log2': (1, lambda x: gmpy2.log2(x)),
 'log10': (1, lambda x: gmpy2.log10(x)),
 'sin': (1, lambda x: gmpy2.sin(x)),
 'cos': (1, lambda x: gmpy2.cos(x)),
 'tan': (1, lambda x: gmpy2.tan(x)),
 'atan2': (2, lambda y, x: gmpy2.atan2(y, x)),
 'pow': (2, lambda x, y: x ** y),
 'atan': (1, lambda x: gmpy2.atan(x)),
 'asin': (1, lambda x: gmpy2.asin(x)),
 'acos': (1, lambda x: gmpy2.acos(x)),
 'sinh': (1, lambda x: gmpy2.sinh(x)),
 'cosh': (1, lambda x: gmpy2.cosh(x)),
 'tanh': (1, lambda x: gmpy2.tanh(x)),
 'asinh': (1, lambda x: gmpy2.asinh(x)),
 'acosh': (1, lambda x: gmpy2.acosh(x)),
 'atanh': (1, lambda x: gmpy2.atanh(x)),
 'expm1': (1, lambda x: gmpy2.expm1(x)),
 'log1p': (1, lambda x: gmpy2.log1p(x)),
 'exp2': (1, lambda x: gmpy2.exp2(x)),
 'exp10': (1, lambda x: gmpy2.exp10(x)),
 'sinpi': (1, trigpi('sin')),
 'cospi': (1, trigpi('cos')),
 'tanpi': (1, trigpi('tan')),
 'erf': (1, lambda x: gmpy2.erf(x)),
 'erfc': (1, lambda x: gmpy2.erfc(x)),
 'tgamma': (1, lambda x: gmpy2.gamma(x)),
 'lgamma': (1, lambda x: gmpy2.lgamma(x)[0]),
}

def fin(x): return x == x and abs(x) != float('inf')

def rnd_double(r):
    b = r.getrandbits(64); x = struct.unpack('<d', struct.pack('<Q', b))[0]
    return x if fin(x) else 1.0

def lit(x):
    if x == 0 and str(x).startswith('-'): return '(0.0 * (0.0 - 1.0))'
    s = repr(x)
    return s if not s.startswith('-') else '(0.0 - %s)' % s[1:]

def main():
    from types import SimpleNamespace
    w, (fn,) = read('math_vec.py', {'cm': 1, 'sample': 1, 'out': 1, 'shard': 1}, 'FN', 1, 1)
    A = SimpleNamespace(cm=w.get('cm'), sample=int(w.get('sample', 0)), out=w.get('out'), shard=int(w.get('shard', 1000)))
    shard = A.shard
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
    out = A.out or repo
    wcf = os.path.join(A.cm, fn, fn + '.wc') if A.cm else os.path.join(repo, 'vectors/math', fn + '.wc')
    ar, f = FNS[fn]
    cases, seen = [], set()
    def add(args, src):
        if not all(fin(a) for a in args) or args in seen: return
        with gmpy2.context(CTX):
            y = f(*[mpfr(a) for a in args])
        y = float(y)
        if not fin(y): return  # overflow traps; tested separately
        seen.add(args); cases.append((args, y, src))
    lines = [l.strip() for l in open(wcf)]
    lines = [l for l in lines if l and l[0] != '#']
    step = max(1, len(lines) // A.sample) if A.sample else 1
    for line in lines[::step]:
        try: args = tuple(float.fromhex(t) for t in line.split(','))
        except ValueError: continue
        if len(args) == ar: add(args, 'wc')
    r = random.Random(20261015 + ar)
    for _ in range(2000):
        add(tuple(rnd_double(r) for _ in range(ar)), 'rand')
    for _ in range(1000):
        add(tuple(r.uniform(-4, 4) for _ in range(ar)), 'rand')
    edge = [0.0, -0.0, 5e-324, 1e-323, 2.2250738585072014e-308, 2.225073858507201e-308,
            1.7976931348623157e308, 8.98846567431158e307, 1.0, -1.0, 8.0, 27.0, 0.5, 2.0, 3.0]
    import itertools
    nx = lambda x, n: struct.unpack('<d', struct.pack('<q', struct.unpack('<q', struct.pack('<d', x))[0] + n))[0]
    extra = {
     'exp': [709.782712893384, 709.7827128933841, -708.3964185322641, -708.3964185322642,
             -745.1332191019411, -745.1332191019412, -744.44007192138, 5.551115123125783e-17,
             5.551115123125784e-17, -5.551115123125783e-17, 2.0**-60, 1e-300, 0.6931471805599453],
     'log': [nx(1.0, 1), nx(1.0, -1), nx(1.0, 7), nx(1.0, -9), 1.5, nx(1.5, -1), 0.75, nx(0.75, -1),
             2.0**-1074, 2.0**-1022, nx(2.0**-1022, -1), 2.718281828459045, 10.0],
     'sin': [6381956970095103 * 2.0**797, 5.319372648326541e+255, 1.5707963267948966, 3.141592653589793,
             2.0**1023, 1e22, 7.450580596923828e-09, nx(7.450580596923828e-09, -1), 2.0**-1074, 0.7853981633974483],
    }
    extra['cos'] = extra['sin']
    extra['tan'] = extra['sin'] + [1.5707963267948966 * 3, 4.71238898038469]
    extra['log2'] = extra['log'] + [2.0**k for k in (-1074, -1022, -1, 1, 52, 1023)]
    extra['log10'] = extra['log'] + [10.0**k for k in range(0, 23)] + [1e23, 1e-5, 0.1]
    if fn == 'pow':
        # exact results and midpoints: integer y, and y = n/2^k with x's odd
        # part a perfect 2^k-th power (t^2 to the 3/2 is a 54-bit midpoint)
        for t in [3, 5, 7, 131071, 134217727, 94906265, 2**26 + 1]:
            for yv in range(2, 65):
                add((float(t), float(yv)), 'edge'); add((-float(t), float(yv)), 'edge'); add((float(t) * 2.0**-600, float(yv)), 'edge')
        for t in [3, 5, 7, 262143, 177147, 129140163, 2**17 + 1, 2**18 - 1]:
            for k in range(1, 6):
                xv = float(t) ** (2 ** k)
                if xv >= 2**53: continue
                for n in range(1, 35, 2):
                    add((xv, n / 2**k), 'edge'); add((xv * 2.0**(-4 * 2**k), n / 2**k), 'edge')
        for xv in [2.0, 0.5, 10.0, 1.0000000000000002, 0.9999999999999999, -1.0, -2.0, 5e-324, 1.7976931348623157e308]:
            for yv in [0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 1e10, -1e10, 1023.0, -1074.0, 1e-300, 2.0**62, 0.1]:
                add((xv, yv), 'edge'); add((-xv, yv), 'edge')
        r2 = random.Random(77)
        for _ in range(2000):
            add((r2.uniform(0, 4), r2.uniform(-300, 300)), 'rand')
            add((2.0 ** r2.uniform(-1074, 1023), r2.uniform(-1.5, 1.5)), 'rand')
    if fn in ('asin', 'acos'):
        r2 = random.Random(78)
        for _ in range(3000):
            add((r2.uniform(-1, 1),), 'rand'); add((r2.choice([-1, 1]) * 2.0 ** r2.uniform(-40, 0),), 'rand')
            add((r2.choice([-1, 1]) * (1 - 2.0 ** r2.uniform(-53, -1)),), 'rand')
    extra['atan'] = [1.0, 2.0**-28, 2.0**28, 2.0**54, 2.0**-1074, 0.5, 2.0]
    extra['asin'] = [1.0, 0.5, 2.0**-28, 0.7071067811865476, 2.0**-1074, 0.9999999999999999]
    extra['acos'] = extra['asin']
    if fn in ('sinh', 'cosh', 'tanh'):
        r2 = random.Random(79)
        for _ in range(2500):
            add((r2.uniform(-30, 30),), 'rand'); add((r2.uniform(-711, 711),), 'rand')
            add((r2.choice([-1, 1]) * 2.0 ** r2.uniform(-30, 3),), 'rand')
    extra['sinh'] = [7.450580596923828e-09, 710.4758600739439, 1.0, 0.5, 2.0**-1074, 22.0, 19.0]
    extra['cosh'] = extra['sinh']; extra['tanh'] = extra['sinh']
    if fn in ('asinh', 'acosh', 'atanh'):
        r2 = random.Random(80)
        for _ in range(2500):
            add((r2.uniform(-1, 1),), 'rand'); add((r2.choice([-1, 1]) * 2.0 ** r2.uniform(-30, 1023),), 'rand')
            add((1 + 2.0 ** r2.uniform(-52, 2),), 'rand'); add((r2.choice([-1, 1]) * (1 - 2.0 ** r2.uniform(-53, -1)),), 'rand')
    extra['asinh'] = [7.450580596923828e-09, 1.0, 0.5, 2.0**-1074, 1.7976931348623157e308, 2.0**26, 0.75]
    extra['acosh'] = [1.0, 1.0000000000000002, 1.5, 2.0, 1.7976931348623157e308, 2.0**26]
    extra['atanh'] = [7.450580596923828e-09, 0.5, 0.9999999999999999, 2.0**-1074, 0.75, 0.2]
    if fn in ('expm1', 'log1p', 'exp2', 'exp10'):
        r2 = random.Random(81)
        hi = {'expm1': 709, 'log1p': 1e300, 'exp2': 1023, 'exp10': 308}[fn]
        for _ in range(2500):
            add((r2.uniform(-1, 1),), 'rand'); add((r2.choice([-1, 1]) * 2.0 ** r2.uniform(-60, 0),), 'rand')
            add((r2.uniform(-hi, hi) if fn != 'log1p' else 2.0 ** r2.uniform(-60, 1023),), 'rand')
            if fn == 'exp2': add((float(r2.randint(-1100, 1023)) + r2.choice([0, 0.5, 0.25]),), 'rand')
    extra['expm1'] = [5.551115123125783e-17, 709.7827128933841, -38.0, 1e-10, 0.6931471805599453, 2.0**-1074]
    extra['log1p'] = [5.551115123125783e-17, -0.5, 1.0, 1.7976931348623157e308, 2.0**-1074, 1e-10, 2.0**53]
    extra['exp2'] = [-1075.0, -1074.0, -1022.0, 1023.0, 0.5, 2.7755575615628914e-17, -1074.5]
    extra['exp10'] = [float(k) for k in range(-5, 30)] + [308.0, 308.25, -323.0, 6.938893903907228e-18, 0.5]
    if fn in ('sinpi', 'cospi', 'tanpi'):
        r2 = random.Random(82)
        for _ in range(2500):
            add((r2.uniform(-4, 4),), 'rand'); add((r2.choice([-1, 1]) * 2.0 ** r2.uniform(-1074, 53),), 'rand')
            add((r2.randint(-4000, 4000) / 12.0,), 'rand'); add((r2.randint(-2**20, 2**20) / 2.0**r2.randint(0, 3),), 'rand')
    extra['sinpi'] = [0.25, 0.5, 1.0 / 6, 1.0 / 3, 2.0**-1074, 2.0**-28, 4503599627370495.5, 2.0**52, 2.0**53, 0.75, 1e300]
    extra['cospi'] = extra['sinpi']; extra['tanpi'] = extra['sinpi']
    if fn in ('erf', 'erfc'):
        r2 = random.Random(83)
        for _ in range(2500):
            add((r2.uniform(-6, 6),), 'rand'); add((r2.choice([-1, 1]) * 2.0 ** r2.uniform(-1074, 2.6),), 'rand')
            add((r2.uniform(-6, 27.3),), 'rand')
    extra['erf'] = [6.0, 5.9, 2.0**-1074, 1e-300, 0.5, 1.0, 27.0, 26.5, 27.2, 27.3]
    extra['erfc'] = extra['erf']
    if fn in ('tgamma', 'lgamma'):
        r2 = random.Random(84)
        hi = 172 if fn == 'tgamma' else 1e5
        for _ in range(2500):
            add((r2.uniform(-200, 172),), 'rand'); add((r2.choice([-1, 1]) * 2.0 ** r2.uniform(-1074, 7.4),), 'rand')
            add((r2.uniform(0, 3),), 'rand')
            if fn == 'lgamma': add((2.0 ** r2.uniform(0, 1014),), 'rand')
    extra['tgamma'] = [float(n) for n in range(1, 30)] + [171.0, 171.6243769563027, 0.5, 1.5, -0.5, -1.5, 2.0**-1074, 2.0**-1022, -190.5, -184.3, 1e-300]
    extra['lgamma'] = extra['tgamma'] + [2.5e305, 1e300, 1e10, -2.4570247382208006, -2.7476826467274127, 2.0**53]
    for e in extra.get(fn, []):
        for k in range(-3, 4):
            add((nx(e, k),), 'edge'); add((-nx(e, k),), 'edge')
    for t in itertools.product(edge + [-e for e in edge], repeat=ar): add(t, 'edge')
    os.makedirs(os.path.join(out, 'tests/math'), exist_ok=True)
    for p in os.listdir(os.path.join(out, 'tests/math')):
        if p.startswith('math_checks_%s_' % fn): os.remove(os.path.join(out, 'tests/math', p))
    for i in range(0, len(cases), shard):
        with open(os.path.join(out, 'tests/math', 'math_checks_%s_%d.rce' % (fn, i // shard)), 'w') as o:
            o.write('# generated by scripts/gen/math_vec.py %s: MPFR-checked, exact bits\n' % fn)
            for a, y, s in cases[i:i + shard]:
                o.write('check %s(%s) == %s\n' % (fn, ', '.join(lit(x) for x in a), lit(y)))
    print(fn, len(cases), {k: sum(1 for c in cases if c[2] == k) for k in ('wc', 'rand', 'edge')})
main()
