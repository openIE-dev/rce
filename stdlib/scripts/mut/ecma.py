# Mutation test for lib/ui/ecma.rce: each mutant must fail at least one
# shard of tests/ui/ecma_checks*.rce. (Node V8 agreement vectors.) Usage:
# RCE=path/to/rce python3 scripts/mut/ecma.py
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
SRC = open(os.path.join(root, 'lib/ui/ecma.rce')).read()
DEPS = {n: open(os.path.join(root, 'lib', p + '.rce')).read() for n, p in
        [('enc', 'enc/enc'), ('events', 'data/events'), ('xml', 'text/xml')]}
SH = sorted(glob.glob(os.path.join(root, 'tests/ui/ecma_checks*.rce')))
M = [
 # truthiness, equality
 ('        St(s: s) -> return s != ""', '        St(s: s) -> return true'),
 ('        Nm(n: n) -> return n != 0', '        Nm(n: n) -> return true'),
 ('    if e_nullish(a) or e_nullish(b)\n        return e_nullish(a) and e_nullish(b)', '    if e_nullish(a) and e_nullish(b)\n        return true'),
 ('    if e_is_str(a) and e_is_str(b)\n        return strict_eq(a, b)\n    let x', '    if e_is_str(a) or e_is_str(b)\n        return strict_eq(a, b)\n    let x'),
 ('    if t != vtag(b) or t == 4\n', '    if t != vtag(b)\n'),
 # precedence
 ('    if t == "&&"\n        return 1', '    if t == "&&"\n        return 0'),
 ('    if t == "*" or t == "/" or t == "%"\n        return 8', '    if t == "*" or t == "/" or t == "%"\n        return 7'),
 ('    if t == "<" or t == ">" or t == "<=" or t == ">=" or t == "instanceof" or t == "in"\n        return 6', '    if t == "<" or t == ">" or t == "<=" or t == ">=" or t == "instanceof" or t == "in"\n        return 5'),
 # string comparison by code unit, then length
 ('            lt = xs.len() < ys.len()', '            lt = xs.len() <= ys.len()'),
 ('    return ((num_of(v) % m) + m) % m', '    return num_of(v) % m'),
]
def run(src, f):
    with tempfile.TemporaryDirectory() as d:
        for n, t in DEPS.items():
            open(os.path.join(d, n + '.rce'), 'w').write(t)
        open(os.path.join(d, 'ecma.rce'), 'w').write(src + '\n' + open(f).read())
        return subprocess.run([RCE, 'test', d], capture_output=True).returncode != 0
def one(ab):
    i, m = ab
    src = SRC
    for a, b in (m if isinstance(m, list) else [m]):
        assert SRC.count(a) == 1, a
        src = src.replace(a, b)
    a = (m if isinstance(m, list) else [m])[0][0]
    for f in SH:
        if run(src, f):
            return i, a, True
    return i, a, False
killed = 0
with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
    for i, a, dead in ex.map(one, enumerate(M)):
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
