# Mutation test for lib/ui/view.rce: each mutant must fail at least one
# shard of tests/ui/view_checks*.rce (official and derived first, then
# minimality, live Vue logs, trees). Usage: RCE=path/to/rce python3 scripts/mut/view.py
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
SRC = open(os.path.join(root, 'lib/ui/view.rce')).read()
DEPS = {n: open(os.path.join(root, 'lib', p)).read() for n, p in
        [('enc.rce', 'enc/enc.rce'), ('blake3.rce', 'hash/blake3.rce'), ('utf8.rce', 'text/utf8.rce')]}
T = os.path.join(root, 'tests/ui')
SH = [os.path.join(T, 'view_checks.rce')] + sorted(glob.glob(os.path.join(T, 'view_checks_min*.rce'))) + sorted(glob.glob(os.path.join(T, 'view_checks_opt*.rce'))) \
    + sorted(glob.glob(os.path.join(T, 'view_checks_live*.rce'))) + sorted(glob.glob(os.path.join(T, 'view_checks_tree*.rce')))
# Equivalent mutant removed: attach's `c == before and not put` without
# `not put` (sibling ids are distinct, so before occurs at most once).
M = [
 # name order
 ('        return a.len() < b.len()\n', '        return a.len() > b.len()\n'),
 ('            return a[i] < b[i]', '            return a[i] > b[i]'),
 ('        if not name_lt(&v.names[u][j], &c)\n            return', '        if false\n            return'),
 ('            v.values[u][i] = value\n            return', '            return'),
 ('        if v.names[u][i] != name\n', '        if true\n'),
 # well-formedness
 ('    if n == 0 or v.parent[0] != -1', '    if n == 0'),
 ('        if v.tag[i] == "" and (v.key[i] != "" or v.names[i].len() > 0 or v.kids[i].len() > 0)', '        if v.tag[i] == "" and (v.key[i] != "" or v.names[i].len() > 0)'),
 ('or v.parent[c as usize] != i as i64 or', 'or'),
 ('                if keys.contains(v.key[c as usize])\n                    return false', '                if false\n                    return false'),
 ('    return reach.len() == n\n', '    return true\n'),
 # lis
 ('            if a[tails[m] as usize] < x', '            if a[tails[m] as usize] <= x'),
 ('            lo = m + 1', '            lo = m'),
 ('        if lo > 0\n            prev.push(tails[lo - 1])', '        if lo > 1\n            prev.push(tails[lo - 1])'),
 ('            tails[lo] = i as i64', '            tails[lo] = tails[lo]'),
 # matching
 ('fn same(a: &View, x: i64, b: &View, y: i64) -> bool = a.tag[x as usize] == b.tag[y as usize] and a.key[x as usize] == b.key[y as usize]',
  'fn same(a: &View, x: i64, b: &View, y: i64) -> bool = a.key[x as usize] == b.key[y as usize]'),
 ('    while i < na and i < nb and same(a, ak[i as usize], b, bk[i as usize])\n        m[i as usize] = i\n        i += 1',
  '    while false and i < na\n        m[i as usize] = i\n        i += 1'),
 ('    while e1 >= i and e2 >= i and same(a, ak[e1 as usize], b, bk[e2 as usize]) bound 16777216\n        m[e2 as usize] = e1',
  '    while false and e1 >= i bound 16777216\n        m[e2 as usize] = e1'),
 ('            if m[j as usize] < 0 and same(a, x, b, y)', '            if same(a, x, b, y)'),
 ('            if m[j as usize] < 0 and same(a, x, b, y)\n                m[j as usize] = o\n                done = true', '            if m[j as usize] < 0 and same(a, x, b, y)\n                m[j as usize] = o'),
 # child ops
 ('    while i < na and i < nb and m[i as usize] == i\n        i += 1', '    while false and i < na\n        i += 1'),
 ('        if e2 + 1 < nb\n            anc = live[(e2 + 1) as usize]', '        if false\n            anc = live[(e2 + 1) as usize]'),
 ('            if used[o] == 0\n', '            if used[o] != 0\n'),
 ('            elif keep[j as usize] == 0\n', '            elif true\n'),
 ('            if j + 1 < nb\n                anc = live[(j + 1) as usize]', '            if j + 2 < nb\n                anc = live[(j + 1) as usize]'),
 ('            live.push(off + bk[j])', '            live.push(bk[j])'),
 ('            qa.push(ak[m[j] as usize])\n            qb.push(bk[j])', '            qa.push(ak[m[j] as usize])\n            qb.push(bk[m[j] as usize])'),
 # node ops
 ('        if a.text[u] != b.text[w]\n', '        if false\n'),
 ('            if a.values[u][i] != b.values[w][j]\n', '            if true\n'),
 ('            ops.push(mk(OP_UNSET, x, -1, -1, copy(a.names[u][i]), ""))', '            ops.push(mk(OP_SET, x, -1, -1, copy(a.names[u][i]), ""))'),
 ('    if not same(a, 0, b, 0)\n', '    if a.tag[0] != b.tag[0]\n'),
 # apply
 ('    if not put\n        ks.push(n)\n    v.kids[p as usize] = ks', '    v.kids[p as usize] = ks'),
 ('            if o.parent < 0\n                root = n', '            if false\n                root = n'),
 ('        elif o.op == OP_MOVE\n            detach(&mut v, n)\n', '        elif o.op == OP_MOVE\n'),
 ('                v.values[l] = copy(b.values[d])', '                v.values[l] = copy(b.names[d])'),
 # encoding
 ('        put_head(&mut o, 4, 5)\n', '        put_head(&mut o, 4, 4)\n'),
 ('            put_text(&mut o, &v.values[u][i])\n', ''),
 ('        put_head(&mut o, 4, v.kids[u].len() as i64)', '        put_head(&mut o, 5, v.kids[u].len() as i64)'),
 ('    put_head(&mut o, 0, seq)', '    put_head(&mut o, 0, seq + 1)'),
 ('    let tag = "rce.ui.frame/1"', '    let tag = "rce.ui.frame/0"'),
 ('    put_head(&mut o, 2, 32)', '    put_head(&mut o, 3, 32)'),
 ('    for x in enc.cbor_head(3, bs.len() as i64)', '    for x in enc.cbor_head(2, bs.len() as i64)'),
  # optimal matching
 ('    while i < nb and i < na and same(a, ak[i as usize], b, bk[i as usize])', '    while i < nb and i < na and false'),
 ('                t[at] = t[((p + 1) * (c + 1) + q + 1) as usize] + 1', '                t[at] = t[((p + 1) * (c + 1) + q + 1) as usize]'),
 ('                t[at] = if d >= e then d else e', '                t[at] = d'),
 ('        elif t[((p + 1) * (c + 1) + q) as usize] >= t[(p * (c + 1) + q + 1) as usize]\n            p += 1', '        elif true\n            p += 1'),
 ('            used[(i + p) as usize] = 1\n', ''),
 ('                if m[j as usize] < 0 and same(a, ak[o as usize], b, bk[j as usize])', '                if same(a, ak[o as usize], b, bk[j as usize])'),
 ('    let m = if as_vue then match_kids(a, &ak, b, &bk) else match_opt(a, &ak, b, &bk)', '    let m = match_kids(a, &ak, b, &bk)'),
]
def run(src, f):
    with tempfile.TemporaryDirectory() as d:
        for n, s in DEPS.items(): open(os.path.join(d, n), 'w').write(s)
        open(os.path.join(d, 'view.rce'), 'w').write(src + '\n' + open(f).read())
        return subprocess.run([RCE, 'test', d], capture_output=True).returncode != 0
def one(ab):
    i, (a, b) = ab
    assert SRC.count(a) == 1, a
    src = SRC.replace(a, b)
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
