#!/usr/bin/env python3
"""Write tests/crmath_oracle.txt: exp, tanh and erf of a fixed sample of
binary64 arguments, correctly rounded, from an oracle independent of
`crmath` and of `math.rce`: Python's `decimal` at 120 to 200 digits.

Each value V is computed with a stated relative error bound; the result
is float(V - err) when it equals float(V + err) (CPython's float of a
Decimal is correctly rounded), and the script stops otherwise, so every
line it writes is the correctly rounded value. The arguments are edge
cases (subnormals, the limits of each function's range) and values of a
fixed linear congruential sequence, so the file is the same on every run.

    python3 tests/crmath_oracle.py > tests/crmath_oracle.txt
"""
import struct
import sys
from decimal import Decimal, getcontext, localcontext


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def rounded(v, rel):
    err = abs(v) * rel
    lo, hi = float(v - err), float(v + err)
    if lo != hi and not (lo == hi == 0.0):
        raise SystemExit(f"not decided: {v} +- {err}")
    return lo


def pi(prec):
    with localcontext() as c:
        c.prec = prec + 10
        three = Decimal(3)
        lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
        return +s


def d_exp(x):
    with localcontext() as c:
        c.prec = 120
        return rounded(Decimal(x).exp(), Decimal(10) ** -110)


def d_tanh(x):
    with localcontext() as c:
        c.prec = 150
        a = abs(Decimal(x))
        e = (2 * a).exp()
        t = (e - 1) / (e + 1)
        if x < 0:
            t = -t
        return rounded(t, Decimal(10) ** -125)


C2SP = None


def d_erf(x):
    global C2SP
    with localcontext() as c:
        c.prec = 200
        if C2SP is None:
            C2SP = 2 / pi(200).sqrt()
        a = Decimal(x)
        a2 = a * a
        term = a
        s = Decimal(0)
        n = 0
        while True:
            t = term / (2 * n + 1)
            s += t
            if t == 0 or abs(t) < abs(s) * Decimal(10) ** -195:
                break
            n += 1
            term = -term * a2 / n
        return rounded(C2SP * s, Decimal(10) ** -150)


state = 0x2545F4914F6CDD1D


def unit():
    global state
    state = (6364136223846793005 * state + 1442695040888963407) % (1 << 64)
    return (state >> 11) / float(1 << 53)


def sample(lo, hi, n):
    return [lo + (hi - lo) * unit() for _ in range(n)]


def logs(lo_exp, hi_exp, n):
    out = []
    for _ in range(n):
        m = 2.0 ** (lo_exp + (hi_exp - lo_exp) * unit())
        out.append(m if unit() < 0.5 else -m)
    return out


CASES = {
    "exp": (
        d_exp,
        [5e-324, -5e-324, 1e-300, 2.0**-53, -(2.0**-53), 2.0**-54, 5.551115123125783e-17,
         -5.551115123125783e-17, 0.5, 1.0, -1.0, 0.6931471805599453, 100.5,
         709.782712893384, 709.7827128933840, 709.78, -708.3964185322641, -708.4,
         -744.44, -745.1332191019411, -745.13, -745.1, -745.2]
        + sample(-745.2, 709.78, 400) + logs(-60, 9, 300),
    ),
    "tanh": (
        d_tanh,
        [1e-7, -1e-7, 2.0**-26, 2.0**-25, 0.5, 1.0, -1.0, 19.06, 20.0, 21.99, 22.0, -22.0]
        + sample(-23.0, 23.0, 400) + logs(-23, 4.4, 300),
    ),
    "erf": (
        d_erf,
        [5e-324, -5e-324, 1e-310, 2.2250738585072014e-308, 1e-300, 2.0**-27, 1e-8, 0.5,
         1.0, -1.0, 2.0, 3.0, 5.9, 5.921, 5.999999999999999, -5.999999999999999]
        + sample(-6.0, 6.0, 300) + logs(-80, 2.58, 200),
    ),
}

out = sys.stdout
out.write("# exp, tanh and erf, correctly rounded: FUNCTION ARG_BITS RESULT_BITS (hex)\n")
out.write("# written by tests/crmath_oracle.py (Python decimal, 120 to 200 digits)\n")
for name, (f, xs) in CASES.items():
    for x in xs:
        out.write(f"{name} {bits(x):016x} {bits(f(x)):016x}\n")
