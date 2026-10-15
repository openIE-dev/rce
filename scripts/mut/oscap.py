# Mutation test for the capability-side os modules (lib/os/{fs,...}.rce,
# survey M38 to M45): each mutant must fail the module's comptime checks
# (tests/os/<name>_checks*.rce). Usage: RCE=path/to/rce python3 scripts/mut/oscap.py [module...]
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
M = {}
M['fs'] = [
    ('    if cr and dr\n        fail bad("invalid")', '    if false\n        fail bad("invalid")'),
    ('    if cr and w.slash and named(&w)', '    if cr and w.slash'),
    ('    elif cr and ex\n', '    elif ex\n'),
    ('        if wr or cr or tr\n', '        if wr or cr\n'),
    ('        if dr or w.slash\n            fail bad("not-directory")', '        if dr\n            fail bad("not-directory")'),
    ('        if tr\n            fs.nodes[n as usize].data = []', '        if tr and wr\n            fs.nodes[n as usize].data = []'),
    ('rd: rd, wr: wr', 'rd: true, wr: wr'),
    ('    if not named(&w) or w.node >= 0\n        fail bad("exist")', '    if w.node >= 0\n        fail bad("exist")'),
    ('        if follow\n', '        if true\n'),
    ('    if path.startswith("/")\n        fail bad("not-permitted")', '    if false\n        fail bad("not-permitted")'),
    ('        if p < 0\n            fail bad("not-permitted")', '        if p < -1\n            fail bad("not-permitted")'),
    ('    if not fs.nodes[cur as usize].dir\n        fail bad("not-directory")\n    if c == "."', '    if c == "."'),
    ('    if not fs.nodes[cur as usize].dir\n        fail bad("not-directory")\n    let last', '    let last'),
    ('    if path == ""\n        fail bad("no-entry")', '    if false\n        fail bad("no-entry")'),
    ('    if not d.rd\n        fail bad("bad-descriptor")', '    if false\n        fail bad("bad-descriptor")'),
    ('    if fs.nodes[d.node as usize].dir\n        fail bad("is-directory")\n    let data', '    let data'),
    ('    if not d.wr\n        fail bad("bad-descriptor")', '    if false\n        fail bad("bad-descriptor")'),
    ('    var p = if d.app then data.len() as i64 else d.pos', '    var p = d.pos'),
    ('    fs.descs[h as usize].pos = p\n    return true\n\npub fn seek', '    return true\n\npub fn seek'),
    ('    if not d.wr or len < 0\n', '    if len < 0\n'),
    ('    fs.descs[h as usize].live = false', '    fs.descs[h as usize].live = true'),
    ('    if not named(&w)\n        fail bad("is-directory")', '    if not named(&w)\n        fail bad("invalid")'),
    ('    if w.slash\n        fail bad("not-directory")\n    drop_ent', '    drop_ent'),
    ('    if w.last == "."\n        fail bad("invalid")', '    if w.last == "."\n        fail bad("not-empty")'),
    ('    if not is_empty(fs, w.node)\n        fail bad("not-empty")\n    drop_ent', '    drop_ent'),
    ('    if not named(&o) or not named(&n)\n', '    if not named(&o)\n'),
    ('    if not isdir and (o.slash or n.slash)\n', '    if not isdir and o.slash\n'),
    ('child_towards(fs, o.dir, n.dir) == o.node\n            fail bad("invalid")', 'child_towards(fs, o.dir, n.dir) == o.node and false\n            fail bad("invalid")'),
    ('child_towards(fs, n.dir, o.dir) == n.node\n            fail bad("not-empty")', 'child_towards(fs, n.dir, o.dir) == n.node and false\n            fail bad("not-empty")'),
    ('    if o.node == n.node\n        return true', '    if false\n        return true'),
    ('        if isdir and not tdir\n', '        if false\n'),
    ('        if not isdir and tdir\n', '        if false\n'),
    ('        if tdir and not is_empty(fs, n.node)\n', '        if false\n'),
    ('    if isdir\n        fs.nodes[o.node as usize].parent = n.dir', '    if false\n        fs.nodes[o.node as usize].parent = n.dir'),
    ('        return "d"\n    if w.slash', '        return "d"\n    if false'),
    ('            return a[i] < b[i]', '            return a[i] > b[i]'),
    ('    return o\n\n# ---------------- scripts', '    return []\n\n# ---------------- scripts'),
    ('        o.push(if i < data.len() as i64 then copy(data[i as usize]) else "\\u{0}")', '        o.push(if i < data.len() as i64 then copy(data[i as usize]) else "")'),
    ('    while p < data.len() as i64 and p < d.pos + n bound', '    while p < data.len() as i64 bound'),
    ('access would-block already', 'access already would-block'),
]
def one(job):
    name, a, b = job
    src = open(os.path.join(root, 'lib/os/%s.rce' % name)).read()
    assert src.count(a) == 1, a
    chks = sorted(glob.glob(os.path.join(root, 'tests/os/%s_checks*.rce' % name)))
    with tempfile.TemporaryDirectory() as d:
        for c in chks:
            f = os.path.join(d, 'm.rce'); open(f, 'w').write(src.replace(a, b) + '\n' + open(c).read())
            if subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0:
                return name, a, True
        return name, a, False
mods = sys.argv[1:] or list(M)
jobs = [(n, a, b) for n in mods for a, b in M[n]]
killed = 0
with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
    for n, a, dead in ex.map(one, jobs):
        killed += dead
        print('%s %s %s' % ('killed' if dead else 'SURVIVED', n, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(jobs)))
sys.exit(0 if killed == len(jobs) else 1)
