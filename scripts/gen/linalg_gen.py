# Generate tests/num/linalg_checks_ref_*.rce: num.linalg against
# Reference BLAS/LAPACK 3.12.1 (scripts/ref/reflapack.py), bit for bit.
# Inputs: classic test matrices (Higham, "Accuracy and Stability of
# Numerical Algorithms", 2nd ed., and MATLAB gallery definitions: Hilbert,
# Pascal, Wilkinson W+, Kahan, Frank, Moler, magic(4)) and seeded random
# matrices (Python random, seed 20261015), including scaled ones that
# drive the DNRM2 small/big accumulators and the DGETF2 SFMIN branch.
# Usage: python3 scripts/gen/linalg_gen.py
import os, sys, random, math
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'ref'))
import reflapack as R
root = os.path.join(here, '..', '..')
rng = random.Random(20261015)

def f(x):
    assert math.isfinite(x)
    if x == 0.0: return '0.0'
    return repr(float(x))
def L(xs): return '[' + ', '.join(f(x) for x in xs) + ']'
def B(b): return 'true' if b else 'false'
def colmajor(rows):
    m, n = len(rows), len(rows[0])
    return [rows[i][j] for j in range(n) for i in range(m)]
def rnd(m, n, s=1.0): return [rng.uniform(-1, 1) * s for _ in range(m * n)]

def named():
    out = []
    for n in (4, 6):
        out.append(('hilbert%d' % n, n, n, colmajor([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])))
    out.append(('pascal5', 5, 5, colmajor([[float(math.comb(i + j, i)) for j in range(5)] for i in range(5)])))
    w = 7; out.append(('wilkinson7', w, w, colmajor([[abs(3 - i) * 1.0 if i == j else (1.0 if abs(i - j) == 1 else 0.0) for j in range(w)] for i in range(w)])))
    th = 1.2; s, c = math.sin(th), math.cos(th)
    out.append(('kahan5', 5, 5, colmajor([[0.0 if j < i else (s ** i if j == i else -c * s ** i) for j in range(5)] for i in range(5)])))
    out.append(('frank5', 5, 5, colmajor([[float(5 - max(i, j)) if j >= i - 1 else 0.0 for j in range(5)] for i in range(5)])))
    out.append(('moler5', 5, 5, colmajor([[float(i + 1) if i == j else float(min(i, j) - 1) for j in range(5)] for i in range(5)])))
    out.append(('magic4', 4, 4, colmajor([[16., 2., 3., 13.], [5., 11., 10., 8.], [9., 7., 6., 12.], [4., 14., 15., 1.]])))
    out.append(('zero_pivot', 3, 3, colmajor([[0., 1., 2.], [0., 3., 4.], [0., 5., 7.]])))
    for k in range(10):
        m = rng.randint(1, 7); n = rng.randint(1, 7)
        out.append(('rand%d' % k, m, n, rnd(m, n)))
    out.append(('tiny', 4, 4, rnd(4, 4, 1e-310)))
    out.append(('small', 5, 3, rnd(5, 3, 1e-160)))
    out.append(('huge', 4, 3, rnd(4, 3, 1e300)))
    out.append(('mixed', 4, 4, [v * 10.0 ** rng.randint(-200, 200) for v in rnd(4, 4)]))
    return out

def spd(n):
    a = rnd(n, n)
    return [sum(a[k + i * n] * a[k + j * n] for k in range(n)) + (n if i == j else 0.0) for j in range(n) for i in range(n)]

def main():
    global checks
    checks = []
    def fin(xs): return all(math.isfinite(x) for x in xs)
    def C(tag, expr, val): checks.append('check %s == %s  # %s' % (expr, val, tag))

    for tag, m, n, a in named():
        lu, piv, info = R.dgetf2(m, n, a)
        C('dgetf2 ' + tag, 't_lu(%d, %d, %s)' % (m, n, L(a)), L(lu + [float(p) for p in piv] + [float(info)]))
        qa, tau = R.dgeqr2(m, n, a)
        C('dgeqr2 ' + tag, 't_qr(%d, %d, %s)' % (m, n, L(a)), L(qa + tau))
        k = min(m, n)
        q = R.dorg2r(m, k, k, qa[:m * k], tau)
        C('dorg2r ' + tag, 't_q(%d, %d, %s)' % (m, n, L(a)), L(q))
        if m == n and info == 0:
            for trans in (False, True):
                nr = 2; b = rnd(n, nr)
                x = R.dgetrs(trans, n, lu, piv, nr, b)
                if fin(x):
                    C('dgetrs %s %s' % ('T' if trans else 'N', tag), 't_lus(%d, %s, %s, %d, %s)' % (n, L(a), B(trans), nr, L(b)), L(x))
        if m >= n and all(qa[i + i * m] != 0.0 for i in range(n)) and tag not in ('tiny', 'small', 'huge', 'mixed'):
            nr = 2; b = rnd(m, nr)
            C('dgels ' + tag, 't_ls(%d, %d, %s, %d, %s)' % (m, n, L(a), nr, L(b)), L(R.dgels(m, n, nr, a, b)))

    for n in range(1, 8):
        a = spd(n)
        l, info = R.dpotf2(n, a)
        assert info == 0
        lz = [0.0 if i < j else l[i + j * n] for j in range(n) for i in range(n)]
        C('dpotf2 spd%d' % n, 't_chol(%d, %s)' % (n, L(a)), L(lz))
        b = rnd(n, 2)
        C('dpotrs spd%d' % n, 't_chols(%d, %s, 2, %s)' % (n, L(a), L(b)), L(R.dpotrs(n, lz, 2, b)))
    for tag, m, n, a in named():
        if m == n:
            l, info = R.dpotf2(n, a)
            if info > 0:
                C('dpotf2 info ' + tag, 't_chol_info(%d, %s)' % (n, L(a)), str(info - 1))
            else:
                lz = [0.0 if i < j else l[i + j * n] for j in range(n) for i in range(n)]
                C('dpotf2 ' + tag, 't_chol(%d, %s)' % (n, L(a)), L(lz))

    for ta in (False, True):
        for tb in (False, True):
            for alpha, beta in ((1.0, 0.0), (-0.75, 1.0), (2.5, -0.5)):
                m, n, k = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5)
                a = rnd(m * k, 1); b = rnd(k * n, 1); c = rnd(m * n, 1)
                lda = k if ta else m; ldb = n if tb else k
                C('dgemm %d%d' % (ta, tb), 't_gemm(%s, %s, %d, %d, %d, %s, %s, %s, %s, %s)' % (B(ta), B(tb), m, n, k, f(alpha), L(a), L(b), f(beta), L(c)),
                  L(R.dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c)))
    for t in (False, True):
        for alpha, beta in ((1.0, 0.0), (0.3, 1.0), (-2.0, 0.25)):
            m, n = rng.randint(1, 6), rng.randint(1, 6)
            a = rnd(m, n); x = rnd(m if t else n, 1); y = rnd(n if t else m, 1)
            C('dgemv %d' % t, 't_gemv(%s, %d, %d, %s, %s, %s, %s, %s)' % (B(t), m, n, f(alpha), L(a), L(x), f(beta), L(y)), L(R.dgemv(t, m, n, alpha, a, x, beta, y)))
    for _ in range(4):
        m, n = rng.randint(1, 6), rng.randint(1, 6); x = rnd(m, 1); y = rnd(n, 1); a = rnd(m, n); al = rng.uniform(-2, 2)
        C('dger', 't_ger(%d, %d, %s, %s, %s, %s)' % (m, n, f(al), L(x), L(y), L(a)), L(R.dger(m, n, al, x, y, a)))
    for up in (False, True):
        for tr in (False, True):
            for un in (False, True):
                m, n = rng.randint(1, 6), rng.randint(1, 3)
                a = rnd(m, m)
                for i in range(m): a[i + i * m] = rng.choice([-1, 1]) * rng.uniform(0.5, 2)
                b = rnd(m, n); al = rng.choice([1.0, -1.5])
                C('dtrsm L%s%s%s' % ('UL'[not up], 'NT'[tr], 'NU'[un]), 't_trsm(%s, %s, %s, %d, %d, %s, %s, %s)' % (B(up), B(tr), B(un), m, n, f(al), L(a), L(b)), L(R.dtrsm(up, tr, un, m, n, al, a, b)))
    for s in (1.0, 1e-170, 1e-300, 1e-320, 1e160, 1e300):
        for n in (1, 3, 9):
            x = rnd(n, 1, s); y = rnd(n, 1)
            C('dnrm2 %g' % s, 't_nrm2(%s)' % L(x), f(R.dnrm2(x)))
            C('dasum %g' % s, 't_asum(%s)' % L(x), f(R.dasum(x)))
            C('idamax %g' % s, 't_amax(%s)' % L(x), str(R.idamax(x)))
            if s < 1e100:
                C('ddot %g' % s, 't_dot(%s, %s)' % (L(x), L(y)), f(R.ddot(x, y)))
                al = rng.uniform(-3, 3)
                C('daxpy %g' % s, 't_axpy(%s, %s, %s)' % (f(al), L(x), L(y)), L(R.daxpy(al, x, y)))
    # mixed-scale vector: all three DNRM2 accumulators at once
    x = [1e-300, 0.5, 1e300, 3e-200, 2e290]
    C('dnrm2 three-accumulator', 't_nrm2(%s)' % L(x), f(R.dnrm2(x)))
    x = [1e-300, 0.5, 3e-200]
    C('dnrm2 small+medium', 't_nrm2(%s)' % L(x), f(R.dnrm2(x)))
    x = [3.0, 1e-300]
    C('ddot strided', 't_dots(2, [1.0, 9.0, 2.0, 9.0], 0, 2, [3.0, 4.0, 9.0, 5.0], 1, 2)', f(R.ddot_s(2, [1.0, 9.0, 2.0, 9.0], 2, [4.0, 9.0, 5.0], 2)))

    helpers = open(os.path.join(root, 'tests/num/linalg_helpers.rce')).read()
    per = 40
    outdir = os.path.join(root, 'tests/num')
    for fn in os.listdir(outdir):
        if fn.startswith('linalg_checks_') and fn.endswith('.rce') and '_ref_' in fn:
            os.remove(os.path.join(outdir, fn))
    for s in range(0, len(checks), per):
        with open(os.path.join(outdir, 'linalg_checks_ref_%d.rce' % (s // per)), 'w') as o:
            o.write('# generated by scripts/gen/linalg_gen.py: Reference BLAS/LAPACK 3.12.1, bit for bit\n\n')
            o.write(helpers + '\n')
            o.write('\n'.join(checks[s:s + per]) + '\n')
    print(len(checks), 'checks')


if __name__ == '__main__':
    main()
