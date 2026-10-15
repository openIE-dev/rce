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
M['proc'] = [
    ('pub fn wifexited(s: i64) -> bool = s & 127 == 0', 'pub fn wifexited(s: i64) -> bool = s & 255 == 0'),
    ('pub fn wexitstatus(s: i64) -> i64 = (s >> 8) & 255', 'pub fn wexitstatus(s: i64) -> i64 = (s >> 8) & 127'),
    ('    let sc = if v >= 128 then v - 256 else v', '    let sc = v'),
    ('pub fn wcoredump(s: i64) -> bool = s & 128 != 0', 'pub fn wcoredump(s: i64) -> bool = s & 64 != 0'),
    ('pub fn wifstopped(s: i64) -> bool = s & 255 == 127', 'pub fn wifstopped(s: i64) -> bool = s & 127 == 127'),
    ('pub fn wifcontinued(s: i64) -> bool = s == 65535', 'pub fn wifcontinued(s: i64) -> bool = s & 65535 == 65535'),
    ('    return if v >= 2147483648 then v - 4294967296 else v', '    return v'),
    ('pub fn core_dumped(s: i64) -> bool = wifsignaled(raw32(s)) and wcoredump(raw32(s))', 'pub fn core_dumped(s: i64) -> bool = wcoredump(raw32(s))'),
    ('pub fn success(s: i64) -> bool = raw32(s) == 0', 'pub fn success(s: i64) -> bool = raw32(s) & 65280 == 0'),
    ('    if n < 1 or n > 31\n', '    if n < 1 or n > 30\n'),
    ('XFSZ VTALRM PROF', 'XFSZ PROF VTALRM'),
    ('(if wcoredump(r) then " (core dumped)" else "")', '""'),
    ('    let u = if r < 0 then r + 4294967296 else r', '    let u = if r < 0 then 0 - r else r'),
    ('        return 128 + wtermsig(r)', '        return wtermsig(r)'),
    ('        elif v < 32 or (v >= 127 and v <= 160) or v == 173', '        elif v < 32 or v == 127'),
    ('fn os2c(s: &text) -> text = if s.contains("\\u{0}") then WITH_NUL else copy(s)', 'fn os2c(s: &text) -> text = copy(s)'),
    ('        if not placed and bytes_lt(&v.key, &x.key)', '        if not placed and bytes_lt(&x.key, &v.key)'),
    ('        if x.key == v.key\n            continue', '        if false\n            continue'),
    ('    if c.clear\n        var o: List[EnvVar] = []', '    if false\n        var o: List[EnvVar] = []'),
    ('    c.clear = true\n    c.vars = []', '    c.clear = true'),
    ('                if not any\n                    o = o + "env "', '                if true\n                    o = o + "env "'),
    ('    if c.program != c.args[0]\n', '    if false\n'),
    ('        if c.vars.len() == 0\n            o = o + "},\\n"', '        if false\n            o = o + "},\\n"'),
    ('    if c.stdout != ""\n', '    if false\n'),
    ('    if c.pgroup >= 0\n', '    if c.pgroup > 0\n'),
    ('            elif c == "#"\n', '            elif false\n'),
    ('    if st == "a"\n        out.push(tok)\n    return out', '    if st == "a" and tok != ""\n        out.push(tok)\n    return out'),
    ('                fail BadProc(why: "No closing quotation")\n            if c == "\'"', '                fail BadProc(why: "No closing")\n            if c == "\'"'),
    ('    return (v >= 48 and v <= 57) or (v >= 65', '    return (v >= 65'),
    ('    if s == ""\n        return "\'\'"', '    if false\n        return "\'\'"'),
    ('        if a == digest\n            return true', '        if a != digest\n            return true'),
    ('        ev.push("[" + jstr(&v.key) + "," + (if v.set then jstr(&v.val) else "null") + "]")', '        ev.push("[" + jstr(&v.key) + "," + jstr(&v.val) + "]")'),

    ('        elif v == 34\n            o = o + "\\\\\\""', '        elif false\n            o = o + "\\\\\\""'),
    ('            if esc_from == "\\"" and c != "\\"" and c != "\\\\"', '            if esc_from == "\\"" and c != "\\""'),
    ('        o = o + (if c == "\'" then', '        o = o + (if false then'),
    ('(if v < 16 then "0" else "") + hex(v)', 'hex(v)'),]
M['signal'] = [
    ('SIGBUS 7 10 10 10 10 Core', 'SIGBUS 7 10 10 10 10 Term'),
    ('SIGUSR1 10 30 16 16 30', 'SIGUSR1 10 30 16 16 31'),
    ('SIGPWR 30 29/- 19 19', 'SIGPWR 30 29 19 19'),
    ('SIGWINCH 28 28 20 23 28 Ign P2024', 'SIGWINCH 28 28 20 23 28 Ign P2001'),
    ('        return num(&r.mips)', '        return num(&r.parisc)'),
    ('        let k: usize = if target == "alpha" then 0 else 1', '        let k: usize = 0'),
    ('    if s == "" or s == "-"\n        return -1', '    if s == ""\n        return -1'),
    ('SIGUNUSED SIGINFO"', 'SIGUNUSED"'),
    ('and not (target == "darwin" and r.name == "SIGINFO")', ''),
    ('        if n - RTMIN <= 15', '        if n - RTMIN < 15'),
    ('    if n == 0\n        return "EXIT"', '    if false\n        return "EXIT"'),
    ('    var i: usize = 3\n', '    var i: usize = 0\n'),
    ('Stopped (signal)|Stopped|', 'Stopped|Stopped (signal)|'),
    ('    if n >= RTMIN and n <= RTMAX\n        return "Real-time signal "', '    if n >= 32 and n <= RTMAX\n        return "Real-time signal "'),
    ('    if code == 1 or code == 2 or code == 5 or code == 6', '    if code == 1 or code == 2'),
    ('fn is_rt(n: i64) -> bool = n >= 32', 'fn is_rt(n: i64) -> bool = n >= 40'),
    ('fn kernel_ignore(n: i64) -> bool = n == 18 or n == 17', 'fn kernel_ignore(n: i64) -> bool = n == 17'),
    ('    let ign = how == "ign" or (how == "dfl" and kernel_ignore(n))', '    let ign = how == "ign"'),
    ('    if ign\n        flush(p, n)', '    if false\n        flush(p, n)'),
    ('    if is_stop(n)\n        flush(p, 18)', '    if false\n        flush(p, 18)'),
    ('        flush(p, 19)\n        flush(p, 20)\n        flush(p, 21)', '        flush(p, 19)\n        flush(p, 20)'),
    ('    elif not p.std[n as usize]\n', '    else\n'),
    ('fn sync_sig(n: i64) -> bool = n == 11 or n == 7', 'fn sync_sig(n: i64) -> bool = n == 11'),
    ('        if best < 0 or (s and not bsync) or (s == bsync and n < best)', '        if best < 0 or (s == bsync and n < best)'),
    ('        if best < 0 or (s and not bsync) or (s == bsync and n < best)', '        if best < 0 or (s and not bsync) or n < best'),
    ('        if got.sig < 0 and q.sig == best', '        if q.sig == best'),
    ('    if code == 0\n        return 2', '    if code == 0\n        return 21'),
    ('fn is_stop(n: i64) -> bool = n == 19 or n == 20 or n == 21 or n == 22', 'fn is_stop(n: i64) -> bool = n == 19 or n == 20'),
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
