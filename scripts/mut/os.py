# Mutation test for lib/os/path.rce: each mutant must fail the comptime
# checks (three equivalent mutants removed: a dot test when there is
# neither dot nor separator, a UNC share without a second separator, and
# keeping an empty drive). Usage: RCE=path/to/rce python3 scripts/mut/os.py
import os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
src = open(os.path.join(root, 'lib/os/path.rce')).read()
chk = open(os.path.join(root, 'tests/os/path_checks.rce')).read()
M = [
    ('            if cs[k] != "."\n', '            if true\n'),
    ('    if cs.len() < 2 or cs[1] != "/" or (cs.len() > 2 and cs[2] == "/")', '    if cs.len() < 2 or cs[1] != "/"'),
    ('        elif b.startswith("/") or path == ""', '        elif b.startswith("/")'),
    ('        elif path.endswith("/")\n            path = path + b', '        elif false\n            path = path + b'),
    ('    if head != "" and not all\n        head = rstrip_seps(&head, false)', '    if head != ""\n        head = rstrip_seps(&head, false)'),
    ('        if c != ".." or (init == "" and out.len() == 0) or (out.len() > 0 and out[out.len() - 1] == "..")', '        if c != ".." or (out.len() > 0 and out[out.len() - 1] == "..")'),
    ('    return if o == "" then "." else o', '    return o'),
    ('    while i < sl.len() and i < tl.len() and sl[i] == tl[i] bound', '    while false bound'),
    ('            let start: usize = if up == "\\\\\\\\?\\\\UNC\\\\" then 8 else 2', '            let start: usize = 2'),
    ('        if n.len() > 2 and n[2] == "\\\\"\n', '        if n.len() > 2 and n[2] == "/"\n'),
    ('    return (s.len() >= 3 and s[1] == ":" and s[2] == "\\\\") or', '    return (s.len() >= 2 and s[1] == ":") or'),
    ('            if q.drive.lower() != rd.lower()\n', '            if q.drive != rd\n'),
    ('    if rp != "" and rr == "" and dc.len() > 0 and not (dc[dc.len() - 1] == ":" or is_sep(&dc[dc.len() - 1], true))', '    if false'),
    ('            elif i == 0 and r.root != ""\n', '            elif false\n'),
    ('    if prefix == "" and comps.len() == 0\n', '    if comps.len() == 0\n'),
    ('    if win_normcase(&sa.drive) != win_normcase(&pa.drive)\n', '    if false\n'),
    ('    while i < sl.len() and i < tl.len() and win_normcase(&sl[i]) == win_normcase(&tl[i]) bound', '    while i < sl.len() and i < tl.len() and sl[i] == tl[i] bound'),
    ('        if r.root != r0.root\n', '        if false\n'),
    ('        if p.startswith("/") != abs\n', '        if false\n'),
    ('            return text_lt(&a[i], &b[i])', '            return false'),
    ('        return n != "." and n != ".."', '        return true'),
    ('        if c.len() == 1 and (c[0] < 32 or c == "\\"" or', '        if c.len() == 1 and (c == "\\"" or'),
    ('    let s = rstrip_spaces(&stem).upper()', '    let s = stem.upper()'),
    ('COM9 COM\\u{b9}', 'COM9'),
    ('            if out.len() == 0\n                fail BadPath(why: "escapes the directory")', '            if false\n                fail BadPath(why: "escapes the directory")'),
    ('        if ok and qc.len() as i64 >= blen\n', '        if ok and qc.len() as i64 > blen\n'),
    ('    var ok = preopens[i].startswith("/") and qc.len() <= pc.len()', '    var ok = qc.len() <= pc.len()'),
]
def one(ab):
    i, (a, b) = ab
    assert src.count(a) == 1, a
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'm.rce'); open(f, 'w').write(src.replace(a, b) + '\n' + chk)
        return i, a, subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0
killed = 0
with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
    for i, a, dead in ex.map(one, enumerate(M)):
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
