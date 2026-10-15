# Usage: python3 scripts/gen/rand_gen.py > /tmp/rand_vec.rce, then replace
# the lines after "# ---- generated" in tests/rand/det_checks.rce.
# Extracts the ChaCha20 vectors of RFC 8439 (sections 2.1.1, 2.3.2, 2.4.2,
# Appendix A.1 #1-#5, A.2 #1-#3) from vectors/rand/rfc8439.txt, checks
# them against the pure-Python ChaCha20 below, and prints them as RCE
# checks. Then prints derived generator vectors (seeded streams, bits,
# below, range, bytes, shuffle, split) from the same Python model of
# lib/rand/det.rce; those are derived, not official.
import os, re, sys
M = 0xffffffff
def rotl(x, n): return ((x << n) | (x >> (32 - n))) & M
def qr(v, a, b, c, d):
    v[a] = (v[a] + v[b]) & M; v[d] = rotl(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & M; v[b] = rotl(v[b] ^ v[c], 12)
    v[a] = (v[a] + v[b]) & M; v[d] = rotl(v[d] ^ v[a], 8)
    v[c] = (v[c] + v[d]) & M; v[b] = rotl(v[b] ^ v[c], 7)
def block_words(kw, ctr, nw):
    s = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574] + list(kw) + [ctr & M] + list(nw)
    x = s[:]
    for _ in range(10):
        for q in [(0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15),(0,5,10,15),(1,6,11,12),(2,7,8,13),(3,4,9,14)]: qr(x, *q)
    return [(x[i] + s[i]) & M for i in range(16)]
def wle(b): return [int.from_bytes(bytes(b[i:i+4]), 'little') for i in range(0, len(b), 4)]
def ble(w): return [x for v in w for x in v.to_bytes(4, 'little')]
def block(k, c, n): return ble(block_words(wle(k), c, wle(n)))
def chacha(k, c, n, d):
    o = []
    for i, x in enumerate(d):
        if i % 64 == 0: ks = block(k, c + i // 64, n)
        o.append(x ^ ks[i % 64])
    return o
here = os.path.dirname(os.path.abspath(__file__))
L = open(os.path.join(here, '../../vectors/rand/rfc8439.txt')).read().split('\n')
def dump(start):  # hex dump lines "  000  aa bb ..." from line index start
    out = []; i = start
    while i < len(L):
        m = re.match(r'^\s+(\d{3})  ((?:[0-9a-f]{2} ?)+)', L[i])
        if m: out += [int(h, 16) for h in m.group(2).split()]
        elif out and not re.match(r'^(\s*$|Nir & Langley|RFC 8439)', L[i]): break
        i += 1
    return out
def find(pat, frm=0):
    for i in range(frm, len(L)):
        if re.search(pat, L[i]): return i
    raise SystemExit('not found: ' + pat)
def hx(b): return ''.join('%02x' % x for x in b)
# ---- RFC vectors
v = [0x11111111, 0x01020304, 0x9b8d6f43, 0x01234567]; qr(v, 0, 1, 2, 3)
assert v == [0xea2a92f4, 0xcb1cf8ce, 0x4581472e, 0x5881c4bb]
print('# RFC 8439 2.1.1 quarter round')
print('check qr_ok([%d, %d, %d, %d], [%d, %d, %d, %d])' % (0x11111111, 0x01020304, 0x9b8d6f43, 0x01234567, *v))
key = list(range(32))
n232 = bytes.fromhex('000000090000004a00000000')
i = find(r'^  Serialized Block:', find(r'^2\.3\.2\.  Test Vector'))
ser = dump(i + 1); assert block(key, 1, n232) == ser
print('# RFC 8439 2.3.2 block function')
print('check block_ok("%s", 1, "%s", "%s")' % (hx(key), n232.hex(), hx(ser)))
s242 = find(r'^2\.4\.2\.  Example')
pt = dump(find(r'Plaintext Sunscreen:', s242) + 1)
ct = dump(find(r'^  Ciphertext Sunscreen:', s242) + 1)
n242 = bytes.fromhex('000000000000004a00000000')
assert chacha(key, 1, n242, pt) == ct and len(ct) == 114
print('# RFC 8439 2.4.2 encryption (sunscreen)')
print('check enc_ok("%s", 1, "%s", "%s", "%s")' % (hx(key), n242.hex(), hx(pt), hx(ct)))
a1 = find(r'^A\.1\.  The ChaCha20 Block'); a2 = find(r'^A\.2\.  ChaCha20 Encryption'); a3 = find(r'^A\.3\.  Poly1305')
print('# RFC 8439 A.1 block vectors #1-#5')
p = a1; nb = 0
for t in range(5):
    p = find(r'Test Vector #%d:' % (t + 1), p)
    k = dump(find(r'^  Key:', p) + 1); n = dump(find(r'^  Nonce:', p) + 1)
    c = int(re.search(r'Block Counter = (\d+)', L[find(r'Block Counter = ', p)]).group(1))
    ks = dump(find(r'^  Keystream:', p) + 1)
    assert block(k, c, n) == ks; nb += 1
    print('check block_ok("%s", %d, "%s", "%s")' % (hx(k), c, hx(n), hx(ks)))
print('# RFC 8439 A.2 encryption vectors #1-#3')
p = a2
for t in range(3):
    p = find(r'Test Vector #%d:' % (t + 1), p)
    k = dump(find(r'^  Key:', p) + 1); n = dump(find(r'^  Nonce:', p) + 1)
    c = int(re.search(r'Counter = (\d+)', L[find(r'Initial Block Counter = ', p)]).group(1))
    q = dump(find(r'^  Plaintext:', p) + 1); e = dump(find(r'^  Ciphertext:', p) + 1)
    assert p < a3 and chacha(k, c, n, q) == e and len(q) == len(e)
    print('check enc_ok("%s", %d, "%s", "%s", "%s")' % (hx(k), c, hx(n), hx(q), hx(e)))
# ---- model of lib/rand/det.rce
class Rng:
    def __init__(s, seed, stream): s.key = wle(seed); s.stream = stream; s.block = 0; s.buf = []; s.pos = 16
    def u32(s):
        if s.pos >= 16:
            s.buf = block_words(s.key, s.block & M, [s.stream & M, s.stream >> 32, s.block >> 32]); s.block += 1; s.pos = 0
        v = s.buf[s.pos]; s.pos += 1; return v
    def bits(s, k):
        if k <= 0: return 0
        if k <= 32: return s.u32() >> (32 - k)
        hi = s.u32() >> (64 - k); lo = s.u32(); return (hi << 32) | lo
    def below(s, n):
        k = (n - 1).bit_length(); v = s.bits(k)
        while v >= n: v = s.bits(k)
        return v
    def range(s, lo, hi): return lo + s.below(hi - lo + 1)
    def bytes(s, n):
        o = []
        for i in range(n):
            if i % 4 == 0: w = s.u32()
            o.append((w >> (8 * (i % 4))) & 255)
        return o
    def shuffle(s, xs):
        for i in range(len(xs) - 1, 0, -1):
            j = s.below(i + 1); xs[i], xs[j] = xs[j], xs[i]
    def split(s, stream): return Rng(s.bytes(32), stream)
print('# derived: generator streams from the Python model (scripts/gen/rand_gen.py)')
z = [0] * 32
r = Rng(z, 0); w = [r.u32() for _ in range(32)]
assert ble(w[:16]) == block(z, 0, [0] * 12) and ble(w[16:]) == block(z, 1, [0] * 12)
print('check words_ok("%s", 0, %s)' % (hx(z), w))
seeds = [list(range(32)), [255] * 32, list(bytes.fromhex('1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0'))]
for sd, st in [(seeds[0], 1), (seeds[1], 7), (seeds[2], 4294967297)]:
    r = Rng(sd, st); print('check words_ok("%s", %d, %s)' % (hx(sd), st, [r.u32() for _ in range(20)]))
for k in [1, 7, 31, 32, 33, 48, 62]:
    r = Rng(seeds[0], 3); print('check bits_ok("%s", 3, %d, %s)' % (hx(seeds[0]), k, [r.bits(k) for _ in range(12)]))
for n in [1, 2, 3, 6, 10, 1000, 2**31 + 1, 2**32, 2**32 + 1, 10**15 + 37, 2**62]:
    r = Rng(seeds[2], 5); print('check below_ok("%s", 5, %d, %s)' % (hx(seeds[2]), n, [r.below(n) for _ in range(12)]))
for lo, hi in [(-5, 5), (1, 6), (-2**61, 2**61 - 1), (100, 100)]:
    r = Rng(seeds[1], 9); print('check range_ok("%s", 9, %d, %d, %s)' % (hx(seeds[1]), lo, hi, [r.range(lo, hi) for _ in range(12)]))
r = Rng(seeds[2], 2); print('check bytes_ok("%s", 2, 70, "%s")' % (hx(seeds[2]), hx(r.bytes(70))))
r = Rng(seeds[0], 0); xs = list(range(20)); r.shuffle(xs); print('check shuffle_ok("%s", 0, 20, %s)' % (hx(seeds[0]), xs))
r = Rng(seeds[0], 0); c = r.split(11); print('check split_ok("%s", 0, 11, %s, %d)' % (hx(seeds[0]), [c.u32() for _ in range(8)], r.u32()))
print('# derived: block counter carry past 2^32 into the third nonce word')
r = Rng(seeds[0], 0); r.block = 2**32 + 3; print('check carry_ok("%s", %d, %s)' % (hx(seeds[0]), 2**32 + 3, [r.u32() for _ in range(4)]))
