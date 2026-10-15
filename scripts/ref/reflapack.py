# Reference BLAS/LAPACK 3.12.1 through ctypes (Debian libblas3 and
# liblapack3 3.12.1-6, the netlib reference implementation compiled by
# gfortran; the shared objects contain no FMA instructions). Column-major
# lists of Python floats in and out. Used by scripts/gen/linalg_gen.py.
import ctypes as C, hashlib
BLAS = '/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.12.1'
LAPACK = '/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.12.1'
SHA = {BLAS: '2b3fdbf8e9daa85ce62e449b9738cad4c8b50ba220342f51501c9ade19670895',
       LAPACK: '7efae004d2cafae3d37c7690bbf3c1ad9e4cd01db01c125755d2db74f7a565b1'}
for p, h in SHA.items():
    assert hashlib.sha256(open(p, 'rb').read()).hexdigest() == h, p
blas = C.CDLL(BLAS, mode=C.RTLD_GLOBAL)
lap = C.CDLL(LAPACK)
D, I = C.c_double, C.c_int
def da(xs): return (D * max(1, len(xs)))(*xs)
def ia(n): return (I * max(1, n))()
def i(v): return C.byref(I(v))
def d(v): return C.byref(D(v))
def ch(c): return C.c_char_p(c.encode())
L1 = C.c_size_t(1)

def ddot(x, y):
    f = blas.ddot_; f.restype = D
    return f(i(len(x)), da(x), i(1), da(y), i(1))
def ddot_s(n, x, incx, y, incy):
    f = blas.ddot_; f.restype = D
    return f(i(n), da(x), i(incx), da(y), i(incy))
def dnrm2(x):
    f = blas.dnrm2_; f.restype = D
    return f(i(len(x)), da(x), i(1))
def dasum(x):
    f = blas.dasum_; f.restype = D
    return f(i(len(x)), da(x), i(1))
def idamax(x):
    f = blas.idamax_; f.restype = I
    return f(i(len(x)), da(x), i(1)) - 1
def daxpy(a, x, y):
    yy = da(y); blas.daxpy_(i(len(x)), d(a), da(x), i(1), yy, i(1)); return list(yy)[:len(y)]
def dgemv(t, m, n, alpha, a, x, beta, y):
    yy = da(y)
    blas.dgemv_(ch('T' if t else 'N'), i(m), i(n), d(alpha), da(a), i(max(1, m)), da(x), i(1), d(beta), yy, i(1), L1)
    return list(yy)[:len(y)]
def dger(m, n, alpha, x, y, a):
    aa = da(a); blas.dger_(i(m), i(n), d(alpha), da(x), i(1), da(y), i(1), aa, i(max(1, m))); return list(aa)[:len(a)]
def dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c):
    cc = da(c)
    blas.dgemm_(ch('T' if ta else 'N'), ch('T' if tb else 'N'), i(m), i(n), i(k), d(alpha), da(a), i(lda), da(b), i(ldb), d(beta), cc, i(max(1, m)), L1, L1)
    return list(cc)[:len(c)]
def dtrsm(upper, trans, unit, m, n, alpha, a, b):
    bb = da(b)
    blas.dtrsm_(ch('L'), ch('U' if upper else 'L'), ch('T' if trans else 'N'), ch('U' if unit else 'N'), i(m), i(n), d(alpha), da(a), i(m), bb, i(m), L1, L1, L1, L1)
    return list(bb)[:len(b)]
def dgetf2(m, n, a):
    aa = da(a); p = ia(min(m, n)); info = I(0)
    lap.dgetf2_(i(m), i(n), aa, i(m), p, C.byref(info))
    return list(aa)[:m * n], [v - 1 for v in list(p)[:min(m, n)]], info.value
def dgetrs(trans, n, lu, piv, nrhs, b):
    bb = da(b); p = (I * n)(*[v + 1 for v in piv]); info = I(0)
    lap.dgetrs_(ch('T' if trans else 'N'), i(n), i(nrhs), da(lu), i(n), p, bb, i(n), C.byref(info), L1)
    assert info.value == 0
    return list(bb)[:n * nrhs]
def dpotf2(n, a):
    aa = da(a); info = I(0)
    lap.dpotf2_(ch('L'), i(n), aa, i(n), C.byref(info), L1)
    return list(aa)[:n * n], info.value
def dpotrs(n, l, nrhs, b):
    bb = da(b); info = I(0)
    lap.dpotrs_(ch('L'), i(n), i(nrhs), da(l), i(n), bb, i(n), C.byref(info), L1)
    return list(bb)[:n * nrhs]
def dgeqr2(m, n, a):
    aa = da(a); k = min(m, n); t = da([0.0] * k); w = da([0.0] * n); info = I(0)
    lap.dgeqr2_(i(m), i(n), aa, i(m), t, w, C.byref(info))
    return list(aa)[:m * n], list(t)[:k]
def dorg2r(m, n, k, a, tau):
    aa = da(a); w = da([0.0] * n); info = I(0)
    lap.dorg2r_(i(m), i(n), i(k), aa, i(m), da(tau), w, C.byref(info))
    return list(aa)[:m * n]
def dgels(m, n, nrhs, a, b):
    aa = da(a); bb = da(b); lw = 4096; w = da([0.0] * lw); info = I(0)
    lap.dgels_(ch('N'), i(m), i(n), i(nrhs), aa, i(m), bb, i(m), w, i(lw), C.byref(info), L1)
    assert info.value == 0, info.value
    return [bb[r + c * m] for c in range(nrhs) for r in range(n)]
def dsyev(n, a):
    aa = da(a); w = da([0.0] * n); lw = 4096; wk = da([0.0] * lw); info = I(0)
    lap.dsyev_(ch('V'), ch('L'), i(n), aa, i(n), w, wk, i(lw), C.byref(info), L1, L1)
    assert info.value == 0
    return list(w)[:n], list(aa)[:n * n]
def dgesvd(m, n, a):
    aa = da(a); k = min(m, n); s = da([0.0] * k); u = da([0.0] * m * k); vt = da([0.0] * k * n)
    lw = 8192; wk = da([0.0] * lw); info = I(0)
    lap.dgesvd_(ch('S'), ch('S'), i(m), i(n), aa, i(m), s, u, i(m), vt, i(k), wk, i(lw), C.byref(info), L1, L1)
    assert info.value == 0
    return list(s)[:k]
def dgeev(n, a):
    aa = da(a); wr = da([0.0] * n); wi = da([0.0] * n); dum = da([0.0]); lw = 4096; wk = da([0.0] * lw); info = I(0)
    lap.dgeev_(ch('N'), ch('N'), i(n), aa, i(n), wr, wi, dum, i(1), dum, i(1), wk, i(lw), C.byref(info), L1, L1)
    assert info.value == 0
    return list(wr)[:n], list(wi)[:n]
def dgehd2(n, a):
    aa = da(a); t = da([0.0] * max(1, n - 1)); w = da([0.0] * n); info = I(0)
    lap.dgehd2_(i(n), i(1), i(n), aa, i(n), t, w, C.byref(info))
    return list(aa)[:n * n], list(t)[:max(0, n - 1)]
