# Mutation test for lib/effect/effect.rce: each mutant must fail the
# comptime checks (package with enc). Usage: RCE=path/to/rce python3 scripts/mut/effect.py
import os, shutil, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
L = lambda p: open(os.path.join(root, p)).read()
src = L('lib/effect/effect.rce'); chk = L('tests/effect/effect_checks.rce')
M = [
    ('    if not known(&CONTEXTS, ctx) or not known(&EFFECTS, eff)\n        return false\n    if eff == "fail"', '    if not known(&CONTEXTS, ctx)\n        return false\n    if eff == "fail"'),
    ('    if eff == "fail"\n        return true', '    if eff == "fail"\n        return false'),
    ('    if ctx == "fn" or ctx == "comptime"', '    if ctx == "fn"'),
    ('        return eff != "state"', '        return true'),
    ('    return eff == "emit" or eff == "send" or eff == "state"', '    return eff == "emit" or eff == "state"'),
    ('    return eff == "emit" or eff == "send" or eff == "state"', '    return eff == "emit" or eff == "send"'),
    ('        return caller == "main" or caller == "task"', '        return caller == "main"'),
    ('    if callee == "fn"\n        return true', '    if callee == "fn"\n        return caller != "comptime"'),
    ('    if not known(&CONTEXTS, caller)\n', '    if false\n'),
    ('    return callee == "task" and allows(caller, &sp)', '    return allows(caller, &sp)'),
    ('    p.cmds.push(c)', '    p.cmds.push(copy(c))\n    p.cmds.push(c)'),
    ('    for c in q.cmds\n        p.cmds.push(copy(c))', '    for c in q.cmds\n        p.cmds = [copy(c)]'),
    ('            return x - y', '            return y - x'),
    ('    return (a.len() as i64) - (b.len() as i64)', '    return 0'),
    ('        while j > 0 and byte_cmp(&keys[idx[j - 1]], &keys[i]) > 0 bound', '        while j > 0 and byte_cmp(&keys[idx[j - 1]], &keys[i]) < 0 bound'),
    ('    return "[" + ",".join(parts) + "]"', '    return "[" + ", ".join(parts) + "]"'),
    ('parts.push(enc.jcs_string(&keys[i]) + ":" + canon(&vals[i]))', 'parts.push(enc.jcs_string(&keys[i]) + ": " + canon(&vals[i]))'),
    ('        enc.JInt(v: v) -> return v.tostr()\n', ''),
    ('pub fn vhash(j: &enc.JsonDoc) -> text = canon(j).sha256()', 'pub fn vhash(j: &enc.JsonDoc) -> text = enc.jcs(j).sha256()'),
    ('        o.vals[i as usize] = copy(result)', '        o.keys.push(copy(h))\n        o.vals.push(copy(result))'),
    ('        o.misses += 1', '        o.misses += 0'),
    ('    o.hits += 1', '    o.hits += 0'),
    ('" not recorded"', '" not found"'),
    ('    while i < 12 and i < cs.len() bound 12', '    while i < 11 and i < cs.len() bound 12'),
    ('    while i < n and d < b.cap bound 64', '    while i < n bound 64'),
    ('    return if d > b.cap then b.cap else d', '    return d'),
    ('        d = d * 2', '        d = d * 3'),
    ('    if status == "timed_out" or status == "unreachable"', '    if status == "timed_out"'),
    ('    return status == "failed" and b.retry_failed', '    return status == "failed"'),
    ('    if r.verdict == "done" or r.verdict == "give_up"\n        return', '    if r.verdict == "done"\n        return'),
    ('    if not transient(b, status) or r.attempt + 1 >= b.attempts', '    if not transient(b, status) or r.attempt >= b.attempts'),
    ('    r.wait_until = now + delay(b, r.attempt)', '    r.wait_until = now + delay(b, r.attempt + 1)'),
    ('pub fn due(r: &Retry, now: u64) -> bool = r.verdict == "retry" and now >= r.wait_until', 'pub fn due(r: &Retry, now: u64) -> bool = r.verdict == "retry" and now > r.wait_until'),
    ('= now >= issued and now - issued >= timeout', '= now - issued >= timeout'),
    ('= now >= issued and now - issued >= timeout', '= now >= issued and now - issued > timeout'),
]
killed = 0
with tempfile.TemporaryDirectory() as d:
    for i, (a, b) in enumerate(M):
        assert src.count(a) == 1, a
        pk = os.path.join(d, 'pk%d' % i); os.mkdir(pk)
        shutil.copy(os.path.join(root, 'lib/enc/enc.rce'), pk)
        open(os.path.join(pk, 'effect.rce'), 'w').write(src.replace(a, b) + '\n' + chk)
        dead = subprocess.run([RCE, 'test', pk], capture_output=True).returncode != 0
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
