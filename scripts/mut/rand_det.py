# Mutation test for lib/rand/det.rce: apply each mutant to a copy of the
# module, append tests/rand/det_checks.rce, and require `rce test` to fail.
# Usage: RCE=path/to/rce python3 scripts/mut/rand_det.py
import os, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
src = open(os.path.join(root, 'lib/rand/det.rce')).read()
chk = open(os.path.join(root, 'tests/rand/det_checks.rce')).read()
M = [
    ('rotl32(v[d] ^ v[a], 16)', 'rotl32(v[d] ^ v[a], 15)'),
    ('rotl32(v[b] ^ v[c], 7)', 'rotl32(v[b] ^ v[c], 8)'),
    ('[1634760805, ', '[1634760806, '),
    ('s.push(counter & M32)', 's.push((counter + 1) & M32)'),
    ('out.push((x[i] + s[i]) & M32)', 'out.push(x[i])'),
    ('quarter(&mut x, 1, 6, 11, 12)', 'quarter(&mut x, 1, 6, 11, 13)'),
    ('    for _ in 0..10\n', '    for _ in 0..9\n'),
    ('    r.block += 1', '    r.block += 2'),
    ('    while v >= n bound 256', '    while v > n bound 256'),
    ('let k = bitlen(n - 1)', 'let k = bitlen(n)'),
    ('    return lo + v', '    return lo + v + 1'),
    ('return next_u32(r) >> (32 - k)', 'return next_u32(r) & ((1 << k) - 1)'),
    ('below(r, i + 1) catch 0', 'below(r, i) catch 0'),
    ('let seed = bytes(r, 32)', 'let seed = bytes(r, 31)'),
    ('if seed.len() != 32 or stream < 0', 'if seed.len() < 32 or stream < 0'),
    ('r.stream >> 32, r.block >> 32]', '0, r.block >> 32]'),
    ('r.stream >> 32, r.block >> 32]', 'r.stream >> 32, 0]'),
    ('if b < 0 or b > 255', 'if b < 0 or b > 256'),
    ('((x << n) | (x >> (32 - n))) & M32', '((x << n) | (x >> (31 - n))) & M32'),
    ('o.push((w >> (8 * (i % 4))) & 255)', 'o.push((w >> (8 * (3 - i % 4))) & 255)'),
    ('if n < 1 or n > 4611686018427387904', 'if n < 0 or n > 4611686018427387904'),
]
killed = 0
with tempfile.TemporaryDirectory() as d:
    for i, (a, b) in enumerate(M):
        assert src.count(a) == 1, a
        f = os.path.join(d, 'm%d.rce' % i)
        open(f, 'w').write(src.replace(a, b) + '\n' + chk)
        p = subprocess.run([RCE, 'test', f], capture_output=True, text=True)
        dead = p.returncode != 0
        killed += dead
        print('%s %2d %s -> %s' % ('killed' if dead else 'SURVIVED', i, a.strip(), b.strip()))
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
