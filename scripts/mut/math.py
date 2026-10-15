# Mutation test for lib/math/math.rce: each mutant must fail the comptime
# checks (tests/math/math_checks*.rce; shards run in parallel, first
# failure kills). Usage: RCE=path/to/rce python3 scripts/mut/math.py [fn...]
# Equivalent mutants removed: SPLITTER 2^27 instead of 2^27 + 1 (that
# variant of Veltkamp's split is also exact; 300,000 random products
# checked); the cbrt up-loop midpoint moved to a quarter step (the down
# loop, which compares at the true midpoint, steps back); the hypot
# up-loop tie rule ("c > 0" or "c >= 0" alone): an exact tie
# a^2 + b^2 = ((2R + 1) u / 2)^2 reduces to an odd square that is a sum of
# two squares, so 2R + 1 = 1 mod 4 and R is always even (the rule never
# steps up; a step up is undone by the down loop's tie rule, which is
# tested by tests/math/math_checks_ties.rce); n_round's overflow test at
# e > 1024 instead of 1023 (the product q 2^(sh + drop) overflows and traps
# all the same); n_add's final carry limb (every sum is below 1024 with
# 200, 500, 1000 or 2000 fraction bits, so it never carries past the top
# limb: 2^(f + 10) and 2^(f + 20) are limb boundaries, nothing reaches them).
# Exact results (log10(10^k) = k, log2(2^k) = k) need no special case: the
# bracket around a representable value rounds to it at once (only a midpoint
# could stall Ziv's loop).
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
]] + [('hypot',) + m for m in [
    ('    if b > a\n', '    if false\n'),
    ('    if b < a / 268435456.0', '    if b < a / 16384.0'),
    ('    let s2 = pow2(k / 2 - k)', '    let s2 = pow2(k / 2 - k + 1)'),
    ('    let us = if k < -1000 then pow2(-1074 - k) else 0.0', '    let us = 0.0'),
    ('        g = 2.0 / P52', '        g = 1.0 / P52'),
    ('    return q - 2.0 * h != 0.0', '    return q - 2.0 * h == 0.0'),
    ('        if (r == 2.0 or r == 1.0) and d * 0.5 >= us', '        if r == 1.0 and d * 0.5 >= us'),
    ('        if c < 0 or (c == 0 and odd_on(r, grid(r, us)))', '        if c < 0'),
    ('    let b2 = two_prod(bb, bb)', '    let b2 = [bb * bb]'),
    ('    return r * pow2(k)', '    return r * pow2(k + 1)'),
]] + [('exp',) + m for m in [
    ('    if fabs(x) < 5.551115123125783e-17', '    if fabs(x) < 1e-10'),
    ('    if x < -745.2\n        return 0.0', '    if x < -740.0\n        return 0.0'),
    ('        t = n_divs(&ps, k)', '        t = n_divs(&ps, k + 1)'),
    ('        let kk = if x >= 0.0 then k else 0 - k', '        let kk = if x >= 0.0 then k else 1 - k'),
    ('        o[i + b.len()] = c', '        o[i + b.len()] = 0'),
    ('            s += LM + 1', '            s += LM'),
    ('            v = v | ((a[i + 1] << (LB - b)) & LM)', '            v = v'),
    ('        c = if b > 0 then x >> (LB - b) else 0', '        c = 0'),
    ('    return ((a.len() - 1) as i64) * LB + n', '    return ((a.len() - 1) as i64) * LB + n + 1'),
    ('        r = t % s', '        r = 0'),
    ('        drop = -1074 - sh', '        drop = -1075 - sh'),
    ('    return n_shr(c, bits - f)', '    return n_shr(c, bits - f + 1)'),
]] + [('log',) + m for m in [
    ('        m = m * 0.5\n        e += 1', '        m = m * 0.5\n        e += 2'),
    ('        let d = n_divs(&t, k)', '        let d = n_divs(&t, k + 2)'),
    ('    return n_muls(&s, 2)', '    return n_muls(&s, 3)'),
    ('            rneg = a_neg', '            rneg = neg'),
    ('        digits.push(t / den)', '        digits.push((t / den) & 31)'),
    ('        p += 6', '        p += 5'),
    ('    if x == 1.0\n        return 0.0', '    if x == 1.0\n        return 1.0'),
]] + [('log2',) + m for m in [
    ('            kc = n_pow2(2100)', '            kc = n_pow2(2101)'),
    ('            let sc = c_log2e()\n            let s1 = cst(&sc, 2100, f)', '            let sc = c_log2e()\n            let s1 = cst(&sc, 2101, f)'),
    ('        if which == 1\n', '        if which == 3\n'),
]] + [('log10',) + m for m in [
    ('        elif which == 2\n', '        elif which == 3\n'),
    ('            kc = c_log10_2()', '            kc = c_ln2()'),
]] + [('sin',) + m for m in [
    ('            o.push(a[i] & ((1 << b) - 1))', '            o.push(a[i])'),
    ('    if ax < 7.450580596923828e-09', '    if ax < 1e-6'),
    ('            q = (q + 1) % 4', '            q = q % 4'),
    ('            neg = q >= 2', '            neg = q >= 3'),
    ('            useS = q % 2 == 0', '            useS = q % 2 == 1'),
    ('then not neg else neg', 'then neg else neg'),
    ('        let ws = n_shr(&t, 4200 - me[1] - f)', '        let ws = n_shr(&t, 4201 - me[1] - f)'),
    ('        t = n_divs(&ps, (k + 1) * (k + 2))', '        t = n_divs(&ps, (k + 1) * (k + 3))'),
    ('            if useS and minus\n                neg = not neg\n        else', '            if false\n                neg = not neg\n        else'),
]] + [('cos',) + m for m in [
    ('            neg = q == 1 or q == 2', '            neg = q == 2'),
    ('            useS = q % 2 == 1', '            useS = q % 2 == 0'),
    ('    var cp = n_pow2(f)', '    var cp = n_pow2(f + 1)'),
    ('    k = 0\n    plus = false', '    k = 1\n    plus = false'),
    ('        return 1.0\n    let me = mant(ax)', '        return 0.0\n    let me = mant(ax)'),
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
    if g in ('exp', 'log', 'log2', 'log10', 'sin', 'cos') and 'pub fn %s(' % g not in SRC: continue
    assert SRC.count(a) >= 1, a
    n += 1
    if killed(SRC.replace(a, b, 1)): k += 1
    else: print('SURVIVED', g, repr(a), '->', repr(b))
print('%d/%d killed' % (k, n))
sys.exit(0 if k == n else 1)
