# Mutation test for lib/fmt/fmt.rce: apply each mutant to a copy of the
# module, append each tests/fmt/fmt_checks*.rce, and require at least one
# `rce test` to fail. Usage: RCE=path/to/rce python3 scripts/mut/fmt.py
import glob, os, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
src = open(os.path.join(root, 'lib/fmt/fmt.rce')).read()
chks = [open(f).read() for f in sorted(glob.glob(os.path.join(root, 'tests/fmt/fmt_checks*.rce')))]
M = [
    ('        let a = if d < 0 then 0 - d else d', '        let a = d'),
    ('    if c >= 65 and c <= 90\n        return c - 55', '    if c >= 65 and c <= 90\n        return c - 56'),
    ('        if d >= r\n', '        if d > r\n'),
    ('        if acc == lim\n', '        if acc == lim + 1\n'),
    ('    if i >= n\n        fail BadInt(at: i as i64)', '    if i > n\n        fail BadInt(at: i as i64)'),
    ('        if k > 0 and (nd - k) % 3 == 0', '        if k > 0 and (nd - k) % 3 == 1'),
    ('            if i + 1 < n and cs[i + 1] == "{"', '            if i + 1 < n and cs[i + 1] == "}"'),
    ('                if idx < 0\n                    next += 1', '                if idx < 0\n                    next += 2'),
    ('    return cat(cat(ins(OP_SPLIT, 1, n + 2), f), ins(OP_JMP, 0 - (n + 1), 0))', '    return cat(cat(ins(OP_SPLIT, 1, n + 2), f), ins(OP_JMP, 0 - n, 0))'),
    ('    return cat(f, ins(OP_SPLIT, 0 - n, 1))', '    return cat(f, ins(OP_SPLIT, 1, 1))'),
    ('    return cat(ins(OP_SPLIT, 1, n + 1), f)', '    return cat(ins(OP_SPLIT, 1, n), f)'),
    ('    return cat(cat(cat(ins(OP_SPLIT, 1, na + 2), a), ins(OP_JMP, nb + 1, 0)), b)', '    return cat(cat(cat(ins(OP_SPLIT, 1, na + 1), a), ins(OP_JMP, nb + 1, 0)), b)'),
    ('    for _ in 0..(n - m)\n', '    for _ in 0..(n - m + 1)\n'),
    ('    return hit != (cl[at] == 1)', '    return hit'),
    ('        return [48, 57, 65, 90, 95, 95, 97, 122]', '        return [48, 57, 65, 90, 97, 122]'),
    ('        return [9, 13, 32, 32]', '        return [9, 10, 32, 32]'),
    ('        if op == OP_BOL and i == 0', '        if op == OP_BOL'),
    ('        if op == OP_EOL and i == n', '        if op == OP_EOL'),
    ('(op == OP_ANY and sl == 1 and c != 10)', '(op == OP_ANY and sl == 1)'),
    ('        if not full and (i + 1 >= n', '        if full and (i + 1 >= n'),
    ('or b[i + 1] < 128 or b[i + 1] > 191)', 'or true)'),
    ('    if c >= 224\n        return 3', '    if c >= 224\n        return 2'),
    ('        if c == 93 and not first', '        if c == 93'),
    ('        if hi < lo\n', '        if hi < lo - 1\n'),
    ("    if p.err < 0 and p.pos < p.pat.len()\n        p.err = p.pos as i64\n", ''),
]
killed = 0
with tempfile.TemporaryDirectory() as d:
    for i, (a, b) in enumerate(M):
        assert src.count(a) == 1, a
        dead = False
        for j, c in enumerate(chks):
            f = os.path.join(d, 'm%d_%d.rce' % (i, j))
            open(f, 'w').write(src.replace(a, b) + '\n' + c)
            if subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0:
                dead = True; break
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
