# Mutation test for lib/math/math.rce: each mutant must fail the comptime
# checks (tests/math/math_checks*.rce; shards run in parallel, first
# failure kills). Usage: RCE=path/to/rce python3 scripts/mut/math.py [fn...]
# Equivalent mutants removed: SPLITTER 2^27 instead of 2^27 + 1 (that
# variant of Veltkamp's split is also exact; 300,000 random products
# checked); the cbrt up-loop midpoint moved to a quarter step (the down
# loop, which compares at the true midpoint, steps back).
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
SRC = open(os.path.join(root, 'lib/math/math.rce')).read()
M = [('core',) + m for m in [
    ('    let e = (a - (s - bb)) + (b - bb)', '    let e = (a - (s - bb))'),
    ('    let h = c - (c - a)', '    let h = c - (c + a)'),
    ('    let e = ((x[0] * y[0] - p) + x[0] * y[1] + x[1] * y[0]) + x[1] * y[1]', '    let e = ((x[0] * y[0] - p) + x[0] * y[1] + x[1] * y[0])'),
    ('        if r[1] != 0.0\n            o.push(r[1])', '        if false\n            o.push(r[1])'),
    ('        o = exp_grow(&o, p[1])\n', ''),
    ('        o.push(0.0 - t)', '        o.push(t)'),
    ('        if e[i] < 0.0\n            return -1', '        if e[i] < 0.0\n            return 1'),
    ('const SPLITTER: f64 = 134217729.0', 'const SPLITTER: f64 = 67108865.0'),
    ('        return pow2n(k + 1000) * pow2n(-1000)', '        return pow2n(k + 1001) * pow2n(-1000)'),
    ('    while n < 0 bound 64\n        r = r * 0.5', '    while n < -1 bound 64\n        r = r * 0.5'),
    ('    while m >= 2.0 bound 64', '    while m > 2.0 bound 64'),
    ('    while m < 1.0 bound 64', '    while m < 0.5 bound 64'),
]] + [('cbrt',) + m for m in [
    ('    if r < 0\n        r += 3', '    if r < 0\n        r += 2'),
    ('    let s = pow2(k / 3)', '    let s = pow2(k / 3 + 1)'),
    ('        if cube_cmp(m, &hi) > 0', '        if cube_cmp(m, &hi) > 1'),
    ('        let lo = [y, u * -0.5]', '        let lo = [y, u * -0.25]'),
    ('        if cube_cmp(m, &lo) < 0', '        if cube_cmp(m, &lo) < -1'),
    ('    return if neg then 0.0 - y * s else y * s', '    return y * s'),
    ('    let c = exp_mul(&y2, y)', '    let c = y2'),
]]
SH = sorted(glob.glob(os.path.join(root, 'tests/math/math_checks*.rce')))

def fails(src, sh):
    with tempfile.NamedTemporaryFile('w', suffix='.rce', delete=False) as f:
        f.write(src + '\n' + open(sh).read()); n = f.name
    try:
        r = subprocess.run([RCE, 'test', n], capture_output=True, text=True)
        return r.returncode != 0
    finally:
        os.remove(n)

def killed(src):
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        for k in ex.map(lambda s: fails(src, s), SH):
            if k:
                ex.shutdown(cancel_futures=True); return True
    return False

want = sys.argv[1:]
assert not fails(SRC, SH[0]), 'unmutated source fails'
k = n = 0
for g, a, b in M:
    if want and g not in want: continue
    assert SRC.count(a) >= 1, a
    n += 1
    if killed(SRC.replace(a, b, 1)): k += 1
    else: print('SURVIVED', g, repr(a), '->', repr(b))
print('%d/%d killed' % (k, n))
sys.exit(0 if k == n else 1)
