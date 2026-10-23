#!/usr/bin/env python3
"""Live reference for lib/ui/text.rce: ICU 76.1 (Unicode 16.0) through
ctypes on libicuuc.so.76. Seeded random sequences drawn from every value
of the relevant property (plus the special characters the rules name) are
segmented by ICU's character, word, sentence and line break iterators and
resolved by ubidi; the results are saved to vectors/ui/icu76_live.tsv.
Pools leave out characters ICU tailors (line break SA, and for words
Han, Hiragana, Katakana and Hangul syllables, plus the colons CLDR root
removes from MidLetter), which UAX 14 and 29 leave
to tailoring. ubidi is not used as a reference: in ICU 76 it resolves an
AN directly after L in an LTR paragraph to level 0 (UAX 9 I1 gives 2;
BidiTest agrees with UAX 9), so the bidi rows here were dropped. Usage: python3 scripts/gen/text_icu.py (needs libicuuc.so.76)."""
import ctypes, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_data import tables, ROOT
uc = ctypes.CDLL('libicuuc.so.76')
def f(n, res=None, args=None):
    g = getattr(uc, n + '_76')
    if res is not None: g.restype = res
    if args is not None: g.argtypes = args
    return g
ver = (ctypes.c_uint8 * 4)(); f('u_getUnicodeVersion')(ver)
assert list(ver)[:2] == [16, 0], list(ver)
def u16(cps):
    s = ''.join(map(chr, cps)).encode('utf-16-le')
    m, o = {}, 0
    for i, c in enumerate(cps):
        m[o] = i; o += 2 if c > 0xFFFF else 1
    m[o] = len(cps)
    return ctypes.create_string_buffer(s, len(s) + 2), len(s) // 2, m
op = f('ubrk_open', ctypes.c_void_p); nx = f('ubrk_next', ctypes.c_int, [ctypes.c_void_p]); cl = f('ubrk_close', None, [ctypes.c_void_p])
def breaks(kind, cps):
    buf, n, m = u16(cps); err = ctypes.c_int(0)
    bi = op(kind, b'root', buf, n, ctypes.byref(err)); assert err.value <= 0
    out = []
    while True:
        r = nx(bi)
        if r < 0: break
        out.append(m[r])
    cl(bi); return out
bo = f('ubidi_open', ctypes.c_void_p); bs = f('ubidi_setPara', None, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint8, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)])
bl = f('ubidi_getLevels', ctypes.POINTER(ctypes.c_uint8), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)])
bp = f('ubidi_getParaLevel', ctypes.c_uint8, [ctypes.c_void_p]); bc = f('ubidi_close', None, [ctypes.c_void_p])
def bidi(cps, d):
    buf, n, m = u16(cps); err = ctypes.c_int(0)
    h = bo(); bs(h, buf, n, {0: 0, 1: 1, 2: 0xfe}[d], None, ctypes.byref(err)); assert err.value <= 0
    lv = bl(h, ctypes.byref(err)); assert err.value <= 0
    inv = {v: k for k, v in m.items()}
    out = [lv[inv[i]] for i in range(len(cps))]
    p = bp(h); bc(h); return p, out
T = tables()
R = random.Random(47)
def pool(prop, skip=lambda c: False, extra=()):
    by = {}
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF or skip(c): continue
        by.setdefault(T[prop][c], []).append(c)
    p = []
    for v, cs in sorted(by.items()):
        p += R.sample(cs, min(4, len(cs)))
    return p + list(extra)
han = lambda c: 0x3040 <= c <= 0x30FF or 0x3400 <= c <= 0x9FFF or 0x20000 <= c <= 0x3FFFF or 0xF900 <= c <= 0xFAFF or 0x31F0 <= c <= 0x31FF or 0xFF66 <= c <= 0xFF9F
SA = 42
sa = lambda c: T['lb'][c] == SA
EP = [c for c in range(0x110000) if T['ep'][c]]
EPCN = [c for c in EP if T['gc'][c] == 0]
INCB = [c for c in range(0x110000) if T['incb'][c]]
SPEC = [0x2010, 0x25CC, 0x201C, 0x201D, 0x2018, 0x2019, 0xAB, 0xBB, 0x22, 0x27, 0x200D, 0xFE0F, 0x20, 0x2E, 0x2C, 0x3002, 0x300C, 0x300D, 0x28, 0x29, 0xFF08, 0xFF09, 0x5D0, 0x2D, 0x24, 0x25, 0x30, 0x41, 0x61] + R.sample(EP, 12) + R.sample(EPCN, 4) + R.sample(INCB, 16) + [0x1F3FB, 0x1F466, 0x1F1E6, 0x1F1E8]
P = {
 'g': pool('gcb', extra=SPEC),
 'w': pool('wb', skip=lambda c: han(c) or sa(c) or 0xAC00 <= c <= 0xD7A3 or T['wb'][c] == 8 or c in (0x3A, 0xFE13, 0xFE55, 0xFF1A), extra=[c for c in SPEC if T['wb'][c] != 8]),
 's': pool('sb', skip=lambda c: han(c) or sa(c), extra=SPEC),
 'l': pool('lb', skip=sa, extra=SPEC + [0x2329, 0x3008]),
 'b': pool('bc', extra=[0x28, 0x29, 0x5B, 0x5D, 0x7B, 0x7D, 0x2329, 0x232A, 0x3008, 0x3009, 0x300C, 0x300D, 0x5D0, 0x627, 0x661, 0x31, 0x41, 0x20, 0x300]),
}
KIND = {'g': 0, 'w': 1, 'l': 2, 's': 3}
N = {'g': 3000, 'w': 3000, 's': 3000, 'l': 6000, 'b': 6000}
rows = []
for k in 'gwsl':
    for _ in range(N[k]):
        cps = [R.choice(P[k]) for _ in range(R.randint(1, 8))]
        if k == 'b':
            d = R.choice([0, 1, 2])
            p, lv = bidi(cps, d)
            rows.append('b\t%s\t%d\t%d\t%s' % (' '.join('%X' % c for c in cps), d, p, ' '.join(map(str, lv))))
        else:
            rows.append('%s\t%s\t%s' % (k, ' '.join('%X' % c for c in cps), ' '.join(map(str, breaks(KIND[k], cps)))))
open(os.path.join(ROOT, 'vectors/ui/icu76_live.tsv'), 'w').write('\n'.join(rows) + '\n')
sys.stderr.write('%d rows\n' % len(rows))
