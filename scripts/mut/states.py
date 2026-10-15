# Mutation test for lib/ui/states.rce: each mutant must fail at least one
# shard of tests/ui/states_checks*.rce (IRP shards first). Usage:
# RCE=path/to/rce python3 scripts/mut/states.py
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
SRC = open(os.path.join(root, 'lib/ui/states.rce')).read()
DEPS = {n: open(os.path.join(root, 'lib', p + '.rce')).read() for n, p in
        [('enc', 'enc/enc'), ('events', 'data/events'), ('xml', 'text/xml'), ('ecma', 'ui/ecma')]}
SH = sorted(glob.glob(os.path.join(root, 'tests/ui/states_checks_*.rce'))) + [os.path.join(root, 'tests/ui/states_checks.rce')]
M = [
 # event descriptors (3.12.1)
 ('        if d == "*" or d == e\n            return true', '        if d == e\n            return true'),
 ('        let pre = d + "."\n', '        let pre = d\n'),
 # transition domain / LCCA (App. D)
 ('        if c.xs or c.skind[a as usize] == COMPOUND or c.skind[a as usize] == ROOT', '        if c.xs or c.skind[a as usize] != ATOMIC'),
 ('    if c.tint[t as usize] and (c.xs or', '    if false and (c.xs or'),
 # history
 ('                if r.hset[ses as usize][s as usize] == 1\n', '                if false\n'),
 ('                            if is_atomic(c, q as i64) and is_desc(c, q as i64, x)', '                            if is_desc(c, q as i64, x)'),
 # conflicts: a descendant source preempts
 ('                        rm.push(t2)\n                    else\n                        pre = true', '                        pre = true\n                    else\n                        pre = true'),
 # selection: first enabled per atomic state chain
 ('                                    done = true\n    return remove_conflicts', '                                    done = false\n    return remove_conflicts'),
 # exit order: reverse document order
 ('    var order: List[i64] = []\n    var s = c.sid.len()\n    while s > 0 bound 16777216\n        s -= 1\n        if xs.contains(s as i64)\n            order.push(s as i64)', '    var order: List[i64] = []\n    for s in 0..c.sid.len()\n        if xs.contains(s as i64)\n            order.push(s as i64)'),
 # done events
 ('                            if not is_hist(c, k2) and not in_final(c, r, ses, k2)', '                            if not is_hist(c, k2) and false'),
 ('                    raise_ev(r, ses, k)\n', '\n'),
 # executable content order: transition content between exit and entry
 ('fn microstep(c: &Chart, r: &mut Run, ses: i64, ts: &[i64])\n    exit_states(c, r, ses, ts)\n    for t in ts\n        let ok = exec_block(c, r, ses, c.tblk[t as usize])\n    enter_states(c, r, ses, ts)',
  'fn microstep(c: &Chart, r: &mut Run, ses: i64, ts: &[i64])\n    for t in ts\n        let ok = exec_block(c, r, ses, c.tblk[t as usize])\n    exit_states(c, r, ses, ts)\n    enter_states(c, r, ses, ts)'),
 # done.invoke only for a child that reached a top-level final
 ('    if p >= 0 and r.srunning[p as usize] and r.sfinal[ses as usize]', '    if p >= 0 and r.srunning[p as usize]'),
 # cancelled invocations stop the child
 ('            r.srunning[k] = false\n            finish(c, r, k as i64)', '            finish(c, r, k as i64)'),
 # late binding
 ('    if not c.mlate[m as usize] or not is_ecma(c, r, ses)\n        return', '    if true\n        return'),
 # invocations start at the end of the macrostep, only for states still active
 # (two places guard it: both removed)
 [('        if ti.contains(s as i64) and r.conf[ses as usize][s] == 1', '        if ti.contains(s as i64)'),
  ('            if y != x\n                keep.push(y)', '            if true\n                keep.push(y)')],
]
def run(src, f):
    with tempfile.TemporaryDirectory() as d:
        for n, t in DEPS.items():
            open(os.path.join(d, n + '.rce'), 'w').write(t)
        open(os.path.join(d, 'states.rce'), 'w').write(src + '\n' + open(f).read())
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
