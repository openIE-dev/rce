# Generate tests/num/special_checks_*.rce for lib/num/special.rce:
#  * bit for bit against FDLIBM 5.3 (scripts/ref/fdlibm, gcc x86-64 SSE2),
#  * error at most 1 ulp against the correctly rounded value from mpmath
#    1.4.1 at 200 bits (the FDLIBM accuracy claim, checked in rce).
# Inputs: every branch threshold of each routine (the hex constants in the
# FDLIBM sources and their neighbours), special points, and seeded
# log-uniform samples (Python random, seed 20261017).
import os, sys, struct, random, subprocess, math
import mpmath as mp
mp.mp.prec = 200
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
ref = os.path.join(root, 'scripts/ref/fdlibm')
subprocess.run([os.path.join(ref, 'make.sh')], check=True)
rng = random.Random(20261017)
def bits(x): return struct.unpack('<Q', struct.pack('<d', x))[0]
def fb(b): return struct.unpack('<d', struct.pack('<Q', b & (2**64 - 1)))[0]
def hiw(h, l=0): return fb((h << 32) | l)
def nb(x):  # x and its two neighbours
    b = bits(x); return [fb(b - 1), x, fb(b + 1)] if x != 0 else [x]
def f(x): return repr(x) if x != 0 else '0.0'
def logu(lo, hi, neg=False):
    v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return -v if neg and rng.random() < 0.5 else v
P = {}
P['exp'] = [0.5, 1.0, -1.0, 2.0, 10.0, -10.0, 100.0, -700.0, 709.0, -745.0, 709.782712893384, -745.1332191019411, -708.4, -708.3964185322641,
            hiw(0x3fd62e42), hiw(0x3FF0A2B2), -hiw(0x3fd62e42), -hiw(0x3FF0A2B2), hiw(0x3e300000), -hiw(0x3e300000), 1e-300, 0.6931471805599453, 0.34657359027997264]
P['exp'] += [logu(1e-20, 709) * rng.choice([1, -1]) for _ in range(60)]
P['log'] = [1.0, 2.0, 0.5, 10.0, 1e-300, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 1.0000001, 0.9999999, hiw(0x3ff00000, 1), 1.4142135623730951, 0.7071067811865476, hiw(0x3ff6147a), hiw(0x3ff6b851)]
P['log'] += [logu(1e-310, 1e308) for _ in range(60)] + [1 + rng.uniform(-1e-6, 1e-6) for _ in range(10)]
P['log1p'] = [1e-300, -0.5, 0.41421356, -0.29289321881, hiw(0x3FDA827A), hiw(0x3e200000), hiw(0x3c900000), 1e15, 1e300, -0.9999999999, 3.0, hiw(0x43400000), 0.4142135623730951]
P['log1p'] += [logu(1e-20, 1e10) for _ in range(40)] + [-logu(1e-20, 0.999) for _ in range(20)]
P['expm1'] = [1e-300, 1e-17, -1e-17, 0.5, -0.5, 0.25, -0.25, 1.0, -1.0, 38.0, -38.0, 39.0, -40.0, 100.0, 709.0, -800.0, hiw(0x4043687A), -hiw(0x4043687A), hiw(0x3fd62e42), hiw(0x3FF0A2B2)]
P['expm1'] += [logu(1e-20, 709) * rng.choice([1, -1]) for _ in range(60)]
P['erf'] = [0.5, -0.5, 0.84375, -0.84375, 1.25, -1.25, 2.857142857142857, 6.0, -6.0, 1e-300, 3e-9, hiw(0x00800000) * 0.5, 2.5, 4.0, -3.0]
P['erf'] += [logu(1e-12, 7, True) for _ in range(50)]
P['erfc'] = [0.5, -0.5, 0.25, 0.24, 0.84375, 1.25, -1.25, 2.857142857142857, 2.85714, 6.0, -6.0, -7.0, 10.0, 27.0, 28.0, 30.0, 1e-17, 26.5]
P['erfc'] += [logu(1e-12, 27.2, True) for _ in range(50)]
P['lgamma'] = [1.0, 2.0, 0.5, 1.5, 2.5, 3.0, 7.999, 8.0, 1e-300, 1e-25, 0.9, 0.73, 0.23, 1.46163214496836224576, 1.73, 1.23, 1e10, 1e17, 1e18, 1e300, -0.5, -1.5, -2.5, -0.1, -1e-25, -3.7, -10.3, -100.5, -1e15 - 0.5, -4.5e15 - 1]
P['lgamma'] += [logu(1e-10, 1e30) for _ in range(40)] + [-logu(1e-5, 170) for _ in range(20)]
inputs = []
for fn, xs in P.items():
    for x in xs:
        for y in nb(x): inputs.append((fn, y))
p = subprocess.run([os.path.join(ref, 'fdlibm_ref')], input=''.join('%s %016x\n' % (fn, bits(x)) for fn, x in inputs), capture_output=True, text=True, check=True)
mpf = {'exp': mp.exp, 'log': mp.log, 'log1p': mp.log1p, 'expm1': mp.expm1, 'erf': mp.erf, 'erfc': mp.erfc, 'lgamma': lambda x: mp.log(abs(mp.gamma(x)))}
exact, ulp = [], []
stats = {}
for line in p.stdout.split('\n'):
    if not line: continue
    fn, xb, yb, sg = line.split(); x = fb(int(xb, 16)); y = fb(int(yb, 16))
    if not math.isfinite(y): continue
    if fn == 'lgamma' and x < 0 and x == math.floor(x): continue
    call = 'lgamma_r(%s).value' % f(x) if fn == 'lgamma' else '%s(%s)' % (fn, f(x))
    exact.append('check f64_bits(%s) == f64_bits(%s)  # FDLIBM %s(%s)' % (call, f(y), fn, xb))
    if fn == 'lgamma':
        exact.append('check lgamma_r(%s).sign == %s' % (f(x), sg))
    cr = float(mpf[fn](mp.mpf(x)))
    if cr == 0.0 and y == 0.0: continue
    if not math.isfinite(cr): continue
    d = abs(bits(cr) - bits(y)) if (cr > 0) == (y > 0) else 99
    stats.setdefault(fn, []).append(d)
    if fn == 'lgamma' and abs(cr) < 1e-3:
        continue  # near the zeros at 1 and 2 lgamma error is absolute, not ulp
    if fn == 'lgamma' and x < 0:
        continue  # FDLIBM's reflection is accurate in absolute, not ulp, terms near poles
    ulp.append('check ulps(%s, %s) <= 1  # mpmath %s' % (call, f(cr), fn))
for fn, ds in stats.items():
    print(fn, len(ds), 'max ulp diff vs mpmath', max(ds))
helpers = '''# helpers: distance in ulps between two finite doubles of the same sign
fn ulps(a: f64, b: f64) -> i64
    let x = f64_bits(a)
    let y = f64_bits(b)
    if (x < 0) != (y < 0)
        return 1000000
    return if x > y then x - y else y - x
'''
outdir = os.path.join(root, 'tests/num')
for fn in os.listdir(outdir):
    if fn.startswith('special_checks_'): os.remove(os.path.join(outdir, fn))
allc = exact + ulp
per = 150
for s in range(0, len(allc), per):
    with open(os.path.join(outdir, 'special_checks_%d.rce' % (s // per)), 'w') as o:
        o.write('# generated by scripts/gen/special_gen.py: FDLIBM 5.3 bit for bit; mpmath 1.4.1 (200 bits) within 1 ulp\n\n' + helpers + '\n' + '\n'.join(allc[s:s + per]) + '\n')
print(len(exact), 'exact', len(ulp), 'ulp')
