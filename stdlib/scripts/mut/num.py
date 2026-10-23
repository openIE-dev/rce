# Mutation test for lib/num/{linalg,special,stats}.rce: apply each mutant
# to a copy of its module, then run the module's tests/num/<name>_checks_*
# shards as a package (with its "# uses:" dependencies) until one fails.
# Every mutant must be killed. Left out as equivalent or not
# distinguished by any vector: moving the erfc tiny-x cut (1 - x is also
# what the next branch rounds to there) and dropping the (sum d)^2 / n
# correction in sum_sq_dev (with the double-double mean, sum d is far
# below the last bit of the result on these sets).
# Usage: RCE=path/to/rce python3 scripts/mut/num.py
import glob, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.insert(0, os.path.join(root, 'scripts'))
import uses
M = [
    ('linalg', 'let tau = (beta - alpha) / beta', 'let tau = (beta - alpha) / alpha'),
    ('linalg', 'let jp = j + idamax(m - j, w, oa + j + j * lda, 1)', 'let jp = j'),
    ('linalg', 'if j + 1 < mn\n            ger_b', 'if j + 2 < mn\n            ger_b'),
    ('special', 'r = r + (x - 0.5)', 'r = r + (x - 0.4999999999999999)'),
    ('special', 'const EX_P1: f64 = 0.16666666666666602', 'const EX_P1: f64 = 0.1666666666666'),
    ('stats', 'let q = a * pre * h', 'let q = a * pre / x * h'),
    ('stats', 'return st_fsum(xs) / (xs.len() as f64)', 'return st_fsum(xs) / ((xs.len() + 1) as f64)'),
    ('stats', '    for _ in 0..4\n        var fv', '    for _ in 0..0\n        var fv'),
    ('stats', '            rh[i] = rh[i] + st_fsum(&t)', '            rh[i] = st_fsum(&t)'),
    ('stats', 'return [q1, st_fsum(&r) / n]', 'return [q1, 0.0]'),
    ('stats', 'let df = (a + b) * (a + b)', 'let df = (a + b) * (a + b) * 1.01'),
    ('stats', 'let sh = ad.min(0.5)', 'let sh = ad.min(0.25)'),
]

def run(i):
    mod, a, b = M[i]
    lib = os.path.join(root, 'lib/num/%s.rce' % mod)
    src = open(lib).read()
    assert src.count(a) == 1, a
    with tempfile.TemporaryDirectory() as d:
        for u in uses.deps(lib):
            shutil.copy(os.path.join(root, 'lib', u + '.rce'), d)
        for c in sorted(glob.glob(os.path.join(root, 'tests/num/%s_checks_*.rce' % mod))):
            open(os.path.join(d, mod + '.rce'), 'w').write(src.replace(a, b) + '\n' + open(c).read())
            p = subprocess.run([RCE, 'test', d], capture_output=True, text=True)
            if p.returncode != 0:
                return True, os.path.basename(c)
    return False, ''

with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
    res = list(ex.map(run, range(len(M))))
killed = 0
for (mod, a, b), (dead, by) in zip(M, res):
    killed += dead
    print('%s %s: %s -> %s %s' % ('killed' if dead else 'SURVIVED', mod, a.strip().split('\n')[0], b.strip().split('\n')[0], by))
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
