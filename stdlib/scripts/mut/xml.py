# Mutation test for lib/text/xml.rce: each mutant must fail at least one
# shard of tests/text/xml_checks*.rce.  Usage:
# RCE=path/to/rce python3 scripts/mut/xml.py
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
SRC = open(os.path.join(root, 'lib/text/xml.rce')).read()
DEPS = {n: open(os.path.join(root, 'lib', p + '.rce')).read() for n, p in
        [('enc', 'enc/enc'), ('events', 'data/events')]}
SH = sorted(glob.glob(os.path.join(root, 'tests/text/xml_checks*.rce')))
M = [
 # references
 ('        if nm == "lt"\n            o = o + "<"', '        if nm == "lt"\n            o = o + ">"'),
 ('            if ds.len() > 1 and ds[1] == "x"', '            if ds.len() > 1 and ds[1] == "X"'),
 ('                    v = v * 16 + h', '                    v = v * 10 + h'),
 ('            o = o + (if k > 1 then ";" else "") + q[k]', '            o = o + q[k]'),
 ('            if not ok or v <= 0 or v > 1114111', '            if not ok or v > 1114111'),
 # namespaces: the nearest declaration wins, up to the root element
 ('        for k in 0..names.len()\n            if names[k] == key\n                return copy(x.av[p as usize][k])\n        p = x.par[p as usize]', '        p = x.par[p as usize]'),
 # adjacent text runs merge into one node
 ('        x.txt[l] = x.txt[l] + t', '        x.txt[l] = t'),
 # serialization escapes
 ('        elif c == "&"\n            o = o + "&amp;"', '        elif false\n            o = o + "&amp;"'),
]
def run(src, f):
    with tempfile.TemporaryDirectory() as d:
        for n, t in DEPS.items():
            open(os.path.join(d, n + '.rce'), 'w').write(t)
        open(os.path.join(d, 'xml.rce'), 'w').write(src + '\n' + open(f).read())
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
