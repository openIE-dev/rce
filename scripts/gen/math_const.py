#!/usr/bin/env python3
"""Constants of lib/math/math.rce: floor(c * 2^b) as base-2^30 limbs,
little-endian, computed with MPFR at 9000 bits (gmpy2). Prints the RCE
functions c_two_opi (2/pi, b = 4200), c_pio2 (pi/2, 2100), c_ln2 (ln 2,
2100), c_log2e, c_log10e, c_log10_2 (2100); math.rce must contain those it defines verbatim (--check FILE)."""
import sys, gmpy2
gmpy2.get_context().precision = 9000
pi = gmpy2.const_pi()
ln2 = gmpy2.const_log2(); ln10 = gmpy2.log(10)
C = [('c_two_opi', 2 / pi, 4200), ('c_pio2', pi / 2, 2100), ('c_ln2', ln2, 2100),
     ('c_log2e', 1 / ln2, 2100), ('c_log10e', 1 / ln10, 2100), ('c_log10_2', ln2 / ln10, 2100)]
def limbs(n):
    o = []
    while n: o.append(n & (2**30 - 1)); n >>= 30
    return o
def text(name, v, b):
    L = limbs(int(gmpy2.floor(v * gmpy2.mpfr(2)**b)))
    rows = [', '.join(str(x) for x in L[i:i + 8]) for i in range(0, len(L), 8)]
    return '# floor(%s * 2^%d), base 2^30 limbs, little-endian\nfn %s() -> List[i64]\n    return [%s]\n' % (name[2:], b, name, ',\n        '.join(rows))
out = [text(*c) for c in C]
if '--check' in sys.argv:
    src = open(sys.argv[-1]).read()
    bad = [c[0] for c, t in zip(C, out) if 'fn %s(' % c[0] in src and t not in src]
    print('ok' if not bad else 'MISMATCH ' + ' '.join(bad)); sys.exit(1 if bad else 0)
print('\n'.join(out))
