# Mutation test for lib/net/idna.rce: each mutant must fail at least one
# of the tests/net/idna_checks*.rce files (shards run in parallel).
# Usage: RCE=path/to/rce python3 scripts/mut/idna.py
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
src = open(os.path.join(root, 'lib/net/idna.rce')).read()
chks = [open(f).read() for f in sorted(glob.glob(os.path.join(root, 'tests/net/idna_checks*.rce')))]
M = [
    ('        o.push(V_BASE + (s % N_COUNT) / T_COUNT)', '        o.push(V_BASE + (s % N_COUNT) % T_COUNT)'),
    ('        if s % T_COUNT != 0\n', '        if true\n'),
    ('        while j > 0 and cc[j] != 0 and cc[j - 1] > cc[j] bound', '        while j > 0 and cc[j] != 0 and cc[j - 1] >= cc[j] + 1000 bound'),
    ('        return S_BASE + ((a - L_BASE) * 21 + (b - V_BASE)) * T_COUNT', '        return -1'),
    ('and (a - S_BASE) % T_COUNT == 0 and b > T_BASE', 'and b > T_BASE'),
    ('if starter and (last < cl or last == 0) and comp >= 0', 'if starter and (last <= cl or last == 0) and comp >= 0'),
    ('    var delta = if first then delta0 / 700 else delta0 / 2', '    var delta = delta0 / 2'),
    ('    delta += delta / numpoints', '    delta += delta / (numpoints + 1)'),
    ('    return k + (P_BASE - P_TMIN + 1) * delta / (delta + 38)', '    return k + (P_BASE - P_TMIN + 1) * delta / (delta + 39)'),
    ('    if c >= 65 and c <= 90\n        return c - 65', '    if false\n        return c - 65'),
    ('    if k >= bias + P_TMAX\n', '    if k > bias + P_TMAX\n'),
    ('        if input[k] >= 128\n            return Cps(cps: o, err: true)', '        if false\n            return Cps(cps: o, err: true)'),
    ('        if n > 1114111 or (n >= 55296 and n <= 57343)', '        if n > 1114111'),
    ('    if basic > 0\n        o.push(45)', '    if true\n        o.push(45)'),
    ('                bias = adapt(delta, h + 1, h == basic)', '                bias = adapt(delta, h + 1, false)'),
    ('            if b != 0 and b != 4 and b != 5 and b != 6 and b != 7 and b != 8 and b != 9 and b != 10', '            if b != 0 and b != 4 and b != 5 and b != 6 and b != 7 and b != 8 and b != 9 and b != 10 and b != 3'),
    ('        return last == 0 or last == 4', '        return last == 0 or last == 4 or last == 8'),
    ('    if en and an\n', '    if false\n'),
    ('    return last == 1 or last == 2 or last == 4 or last == 3', '    return last == 1 or last == 2 or last == 4 or last == 3 or last == 8'),
    ('    if f != 0 and f != 1 and f != 2\n', '    if f != 0 and f != 1 and f != 2 and f != 4\n'),
    ('            let virama = i > 0 and ccc(l[i - 1]) == 9', '            let virama = false'),
    ('                    left = t == 2 or t == 4', '                    left = t == 2'),
    ('                    right = t == 2 or t == 3', '                    right = t == 3 or t == 2 or t == 4'),
    ('                if c == 8205\n                    return false', '                if false\n                    return false'),
    ('        if n != copy(l)\n', '        if false\n'),
    ('        if l.len() >= 4 and l[2] == 45 and l[3] == 45', '        if l.len() >= 4 and l[2] == 45'),
    ('        if l[0] == 45 or l[l.len() - 1] == 45', '        if l[0] == 45'),
    ('    if is_mark(l[0])\n', '    if false\n'),
    ('        if not (s == 0 or (s == 2 and not transitional))', '        if not (s == 0 or s == 2)'),
    ('        if f.std3 and c < 128 and not ((c >= 97 and c <= 122) or (c >= 48 and c <= 57) or c == 45)', '        if f.std3 and c < 128 and not ((c >= 97 and c <= 122) or (c >= 48 and c <= 57) or c == 45 or c == 95)'),
    ('            if f.transitional and c == 7838', '            if false'),
    ('        elif s == 2 and f.transitional', '        elif s == 2'),
    ('            if d.cps.len() == 0 or all_ascii(&d.cps)', '            if d.cps.len() == 0'),
    ('            if not valid_label(&d.cps, f, false, true)', '            if not valid_label(&d.cps, f, f.transitional, true)'),
    ('            if not all_ascii(&l)\n                err = true', '            if false\n                err = true'),
    ('        if o.len() < 1 or o.len() > 253', '        if o.len() > 253'),
    ('            if l.len() < 1 or l.len() > 63', '            if l.len() > 63'),
    ('        if b == 1 or b == 2 or b == 3\n            return true', '        if b == 1 or b == 2\n            return true'),
    ('    while e > 0 and bidi_class(l[e - 1]) == 10 bound', '    while false bound'),
    ('        if c < 0 or c > 1114111 or (c >= 55296 and c <= 57343)\n            return Cps(cps: o, err: true)\n        if c < 128', '        if false\n            return Cps(cps: o, err: true)\n        if c < 128'),
    ('                err = true\n                ls.push(copy(l))\n                continue', '                err = true'),
]
ONLY = [int(x) for x in os.environ.get('ONLY', '').split(',') if x]
def run(f): return subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0
killed = 0
with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(os.cpu_count() or 4) as ex:
    for i, (a, b) in enumerate(M):
        if ONLY and i not in ONLY: continue
        assert src.count(a) == 1, a
        m = src.replace(a, b)
        fs = []
        for j, c in enumerate(chks):
            f = os.path.join(d, 'm%d_%d.rce' % (i, j)); open(f, 'w').write(m + '\n' + c); fs.append(f)
        dead = any(ex.map(run, fs))
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
n = len(ONLY) if ONLY else len(M)
print('%d of %d mutants killed' % (killed, n))
sys.exit(0 if killed == n else 1)
