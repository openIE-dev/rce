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
# could stall Ziv's loop). Error-bound margins (EB, tan's (1 + Q^2) term)
# are soundness slack, not observable on vectors, so not mutated; n_div's
# remainder == divisor step (never hit: the dividends are transcendental);
# n_div's start window lb - 52 instead of lb - 53 (the same Y0, one bit
# fewer read from b).
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
]] + [('tan',) + m for m in [
    ('            var s = if even then minus else not minus', '            var s = if even then minus else minus'),
    ('                s = not s\n                return if s', '                s = s\n                return if s'),
    ('    let qa = n_mul(&num, &y)\n    var q = n_shr(&qa, k)', '    let qa = n_mul(&num, &y)\n    var q = n_shr(&qa, k + 1)'),
    ('            let tq = if even then n_div(&s0, &s1, f) else n_div(&s1, &s0, f)', '            let tq = n_div(&s0, &s1, f)'),
]] + [('atan2',) + m for m in [
    ('    let dd = if f == 200 then d else 3 * d', '    let dd = d'),
    ('    if y == 0.0\n        if xneg', '    if y == 0.0\n        if false'),
    ('    let swap = ay > ax', '    let swap = false'),
    ('        r = n_sub(&pc, &r)\n    if xneg', '        r = n_add(&pc, &r)\n    if xneg'),
    ('        let pi = n_muls(&pc, 2)', '        let pi = n_muls(&pc, 3)'),
    ('        u = n_div(&num, &den, fe)', '        u = copy(num)'),
    ('        uneg = n_cmp(t, &c) < 0', '        uneg = n_cmp(t, &c) > 0'),
    ('        let d = n_divs(&p, k)', '        let d = n_divs(&p, k + 2)'),
    ('(x as f32).tostr() == "-0"', '(x as f32).tostr() == "0"'),
    ('        return [(x * pow2(600) * pow2(474)) as i64, -1074]\n    return mant(x)', '        return [(x * pow2(600) * pow2(475)) as i64, -1074]\n    return mant(x)'),
]] + [('pow',) + m for m in [
    ('    if yint and y >= 2.0 and y <= 64.0', '    if false'),
    ('        let v = n_round(&p, xo[1] * yi)', '        let v = n_round(&p, xo[1] * yi + 1)'),
    ('    if y > 0.0 and ym[1] < 0 and ym[1] >= -5 and ym[0] <= 34', '    if false'),
    ('        yodd = ay - 2.0 * h == 1.0', '        yodd = false'),
    ('        return if yodd then x else 0.0', '        return 0.0'),
    ('        let zneg = (lb[0][0] == 1) != (y < 0.0)', '        let zneg = lb[0][0] == 1'),
    ('        let z = n_shr(&lp, fl2 - ey - f)', '        let z = n_shr(&lp, fl2 - ey - f + 1)'),
    ('        if rneg and r.len() > 0\n            k -= 1', '        if false\n            k -= 1'),
    ('    let neg = x < 0.0 and yodd', '    let neg = x < 0.0'),
    ('    while o % 2 == 0 bound 64', '    while o % 4 == 0 bound 64'),
    ('        if n_cmp(&rk, &mo) == 0 and es % (1 << k) == 0', '        if n_cmp(&rk, &mo) == 0'),
    ('    if zd < -746.0', '    if zd < -700.0'),
    ('            let v = n_round(&one, ex)', '            let v = n_round(&one, ex - 1)'),
]] + [('cos',) + m for m in [
    ('            neg = q == 1 or q == 2', '            neg = q == 2'),
    ('            useS = q % 2 == 1', '            useS = q % 2 == 0'),
    ('    var cp = n_pow2(f)', '    var cp = n_pow2(f + 1)'),
    ('    k = 0\n    plus = false', '    k = 1\n    plus = false'),
    ('        return 1.0\n    let me = mant(ax)', '        return 0.0\n    let me = mant(ax)'),
]] + [('asin',) + m for m in [
    ('    if ax < 3.725290298461914e-09\n        return x\n    return asin_acos(x, false)', '    if ax < 3.725290298461914e-07\n        return x\n    return asin_acos(x, false)'),
    ('        let g = f + 160', '        let g = f + 2'),
    ('    let st = (n_to_i64(&tn) as f64).sqrt() * 1.000000001 + 2.0', '    let st = (n_to_i64(&tn) as f64).sqrt() * 0.999999 + 2.0'),
    ('            return if not ac and x < 0.0 then chs(z[0]) else z[0]', '            return z[0]'),
]] + [('acos',) + m for m in [
    ('        let xn = ac and x < 0.0', '        let xn = false'),
    ('    if ax < 2.7755575615628914e-17\n        return 1.5707963267948966', '    if ax < 2.7755575615628914e-15\n        return 1.5707963267948966'),
    ('    if x == -1.0\n        return 3.141592653589793', '    if x == -1.0\n        return 3.1415926535897927'),
    ('            let tq = n_muls(&tp, (2 * k - 1) * (2 * k - 1))', '            let tq = n_muls(&tp, (2 * k - 1) * (2 * k + 1))'),
    ('        let v = if x < 0.0 then n_add(&pc, &s) else n_sub(&pc, &s)', '        let v = n_sub(&pc, &s)'),
    ('    if ax < 0.0009765625\n        return acos_small(x)', '    if ax < 0.0625\n        return acos_small(x)'),
]] + [('atan',) + m for m in [
    ('    return atan2(x, 1.0)', '    return atan2(x, 1.0000000000000002)'),
]] + [('sinh',) + m for m in [
    ('        let g = f + ex + 64\n        let h = hyp_parts(ax, g)', '        let g = f + 64\n        let h = hyp_parts(ax, g)'),
    ('    return [[k1], n_sub(&p[1], &n2), n_add(&p[1], &n2)]', '    return [[k1], n_sub(&p[1], &n2), n_add(&p[1], &p[1])]'),
    ('            return if which == 0 and x < 0.0 then chs(z[0]) else z[0]', '            return z[0]'),
]] + [('cosh',) + m for m in [
    ('    if fabs(x) < 7.450580596923828e-09\n        return 1.0', '    if fabs(x) < 7.450580596923828e-07\n        return 1.0'),
]] + [('tanh',) + m for m in [
    ('    if ax > 22.0\n', '    if ax > 18.0\n'),
    ('        let t = n_div(&h[1], &h[2], fe)', '        let t = n_div(&h[1], &h[2], fe + 1)'),
]] + [('asinh',) + m for m in [
    ('    let one = if 2 * gg - 2 * t >= 0 then n_pow2(2 * gg - 2 * t) else []', '    let one = if 2 * gg - 2 * t >= 0 then n_pow2(2 * gg - 2 * t + 1) else []'),
    ('    return [n_add(&xg, &s), [gg - t]]', '    return [n_add(&xg, &s), [gg]]'),
    ('    if n_to_i64(&t2) == 3\n        e += 1', '    if false\n        e += 1'),
]] + [('acosh',) + m for m in [
    ('        let gg = f + 96', '        let gg = f + 20'),
]] + [('atanh',) + m for m in [
    ('        let z = ziv(&r[1], 0 - gg - 1)', '        let z = ziv(&r[1], 0 - gg)'),
    ('    let b = n_sub(&p, &mm)', '    let b = n_add(&p, &mm)'),
]] + [('expm1',) + m for m in [
    ('    if x < -38.0\n        return -1.0', '    if x < -30.0\n        return -1.0'),
    ('        let g = f + ex + 64\n        let xf = fix(ax, g)', '        let g = f + 64\n        let xf = fix(ax, g)'),
]] + [('log1p',) + m for m in [
    ('        let u = if x < 0.0 then n_sub(&one, &xg) else n_add(&one, &xg)', '        let u = n_add(&one, &xg)'),
    ('    let t = if m[1] + 53 > 0 then m[1] + 53 else 0\n    let mm = n_of(m[0])\n    for f in ZIV_F\n        let g = f + ex + 64', '    let t = if m[1] + 53 > 0 then m[1] + 54 else 0\n    let mm = n_of(m[0])\n    for f in ZIV_F\n        let g = f + ex + 64'),
]] + [('exp2',) + m for m in [
    ('    if k == x\n        return n_round(&one, ki)', '    if k == x\n        return pow2(ki)'),
    ('        let rf = if x < 0.0 then n_sub(&kk, &xf) else n_sub(&xf, &kk)', '        let rf = n_sub(&xf, &kk)'),
]] + [('exp10',) + m for m in [
    ('        return n_round(&p, n)', '        return n_round(&p, n) * 1.0000000000000002'),
    ('    if x == x.floor() and x > 0.0', '    if false'),
]] + [('sinpi',) + m for m in [
    ('    if n_cmp(&rn, &quarter) > 0\n        let half = n_pow2(fx - 1)', '    if n_cmp(&rn, &quarter) > 1\n        let half = n_pow2(fx - 1)'),
    ('    let s_neg = q >= 2', '    let s_neg = q >= 3'),
    ('        let ff = f + ex + 8\n        let c = c_pi()', '        let ff = f + 8\n        let c = c_pi()'),
]] + [('cospi',) + m for m in [
    ('    let c_neg = q == 1 or q == 2', '    let c_neg = q == 2'),
]] + [('tanpi',) + m for m in [
    ('            neg = s_neg != c_neg', '            neg = s_neg'),
]] + [('erf',) + m for m in [
    ('    if ax >= 6.0\n', '    if ax >= 5.0\n'),
    ('        t = n_divs(&ps, 2 * n + 1)', '        t = n_divs(&ps, 2 * n + 3)'),
    ('    return n_shr(&p, 2 * gg - e[0][0])', '    return n_shr(&p, 2 * gg - e[0][0] + 1)'),
]] + [('erfc',) + m for m in [
    ('        if x > 0.0 and lose > 120', '        if x > 0.0 and lose > 4000'),
    ('            err = n_shr(&t, 0)', '            err = []'),
    ('            if n % 2 == 1\n                neg = n_add(&neg, &t2)\n            else\n                pos = n_add(&pos, &t2)', '            if n % 2 == 1\n                neg = n_add(&neg, &t2)\n            else\n                pos = n_add(&pos, &t)'),
    ('            if n % 2 == 1\n                neg = n_add(&neg, &t2)', '            if n % 2 == 0\n                neg = n_add(&neg, &t2)'),
    ('        let gg = f + 72 + ex + (if x > 0.0 then lose else 0)', '        let gg = f + 72 + ex'),
]] + [('tgamma',) + m for m in [
    ('        for i in 1..n\n            p = n_muls(&p, i)', '        for i in 2..n\n            p = n_muls(&p, i)'),
    ('                return if h == h.floor() then z[0] else chs(z[0])', '                return z[0]'),
    ('            let q = n_div(&g[1], &fr[0], s + 64)', '            let q = n_div(&g[1], &fr[0], s + 63)'),
    ('    return 3 * hyp_ex(fabs(x)) >= gg + 2', '    return 2 * hyp_ex(fabs(x)) >= gg + 2'),
    ('    let w = n_add(&one, &cx)', '    let w = one'),
    ('    return if x > 0.0 then n_sub(&w, &gx) else n_add(&w, &gx)', '    return n_sub(&w, &gx)'),
    ('    return if x < 0.0 then [chs(z[1]), chs(z[0])] else z', '    return z'),
    ('            let o = one_plus(&fr[0], s, gg + 16)\n            let g = gamma_big(&o[0], o[1][0], 1.0 + ax, gg)\n            let q', '            let o = one_plus(&fr[0], s, 40)\n            let g = gamma_big(&o[0], o[1][0], 1.0 + ax, gg)\n            let q'),
]] + [('lgamma',) + m for m in [
    ('    let hf = n_pow2(g - 1)\n    let xh = n_sub(&xf, &hf)', '    let hf = n_pow2(g - 2)\n    let xh = n_sub(&xf, &hf)'),
    ('        if k % 2 == 0\n            pos = n_add(&pos, &t)', '        if k % 2 == 1\n            pos = n_add(&pos, &t)'),
    ('    if x > 2.5599833278516383e305', '    if x > 2.5e305'),
    ('    return pow2((gg + 54) / 42 + 1)', '    return pow2((gg + 54) / 42 - 4)'),
    ('                r = sadd(a[0][0] == 1, &a[1], lx[0][0] == 0, &lx[1])', '                r = sadd(a[0][0] == 1, &a[1], lx[0][0] == 1, &lx[1])'),
    ('            if short > 0 and gg + short <= 4100 and tries < 4', '            if false'),
    ('    let wneg = n_cmp(&w, &one) < 0', '    let wneg = false'),
    ('    let r = sadd(lx[0][0] == 0, &lx[1], wneg, &d)', '    let r = sadd(lx[0][0] == 1, &lx[1], wneg, &d)'),
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

def killed(src, g='core'):
    own = [s for s in SH if os.path.basename(s).startswith('math_checks_%s_' % g)]
    order = own + [s for s in SH if s not in own]
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        for k in ex.map(lambda s: fails(src, s), order):
            if k:
                ex.shutdown(cancel_futures=True); return True
    return False

want = sys.argv[1:]
assert not fails(SRC, SH[0]), 'unmutated source fails'
k = n = 0
for g, a, b in M:
    if want and g not in want: continue
    if g != 'core' and 'pub fn %s(' % g not in SRC: continue
    assert SRC.count(a) >= 1, a
    n += 1
    if killed(SRC.replace(a, b, 1), g): k += 1
    else: print('SURVIVED', g, repr(a), '->', repr(b))
print('%d/%d killed' % (k, n))
sys.exit(0 if k == n else 1)
