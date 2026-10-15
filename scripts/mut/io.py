# Mutation test for lib/io/io.rce: each mutant must fail the comptime
# checks or, for mutants marked host, the file-adapter host checks
# (tests/io/io_host.rce under rce host run --fs). Usage: RCE=path/to/rce python3 scripts/mut/io.py
import os, re, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
L = lambda p: open(os.path.join(root, p)).read()
src = L('lib/io/io.rce'); chk = L('tests/io/io_checks.rce'); host = L('tests/io/io_host.rce')
events = re.findall(r'^# event: (.*)$', host, re.M)
M = [
    ('return s == DELIVERED or s == FAILED or s == TIMED_OUT or s == UNREACHABLE', 'return s == DELIVERED or s == FAILED or s == TIMED_OUT', 0),
    ('pub const TIMED_OUT: text = "timed_out"', 'pub const TIMED_OUT: text = "timedout"', 0),
    ('        return has(data) and err == ""', '        return has(data)', 0),
    ('    return not has(data)', '    return true', 0),
    ('        some(x) -> return true', '        some(x) -> return false', 0),
    ('    p.next += 1', '    p.next += 2', 0),
    ('    if p.kinds[k] != what\n', '    if false\n', 0),
    ('    if i < 0\n        p.stray += 1', '    if i < -1\n        p.stray += 1', 0),
    ('    drop_at(p, k)\n', '', 0),
    ('        if i != k\n', '        if true\n', 0),
    ('    p.settled += 1', '    p.settled += 0', 0),
    ('    if n == 0 or cs[n - 1] != "\\r"', '    if n == 0', 0),
    ('    while i + 1 < n bound', '    while i < n bound', 0),
    ('    if r.eof\n        return out\n    let all', '    if false\n        return out\n    let all', 0),
    ('    r.lines += out.len() as u64', '    r.lines += 0', 0),
    ('    if r.rest != ""\n', '    if true\n', 0),
    ('            if o != "" and o.len() + l > w.max', '            if o != "" and o.len() + l >= w.max', 0),
    ('            if o != "" and o.len() + l > w.max', '            if o.len() + l > w.max', 0),
    ('        if pos >= w.off\n', '        if true\n', 0),
    ('    if w.flight > 0 or w.off >= w.data.len()', '    if w.off >= w.data.len()', 0),
    ('    if n > (w.flight as u64)', '    if n > (w.max as u64)', 0),
    ('    if not char_boundary(&w.data, k)\n', '    if false\n', 0),
    ('    if w.flight == 0\n        fail', '    if false\n        fail', 0),
    ('    return b < 128 or b >= 192', '    return b < 128', 0),
    ('    w.sent += n', '    w.sent += 1', 0),
    ('        return list\n', '        return true\n', 0),
    ('    if cs[0] == "/"\n', '    if false\n', 0),
    ('            if seg == ".."\n', '            if seg == "."\n', 0),
    ('    while i <= cs.len() bound', '    while i < cs.len() bound', 0),
]
killed = 0
with tempfile.TemporaryDirectory() as d:
    for i, (a, b, need_host) in enumerate(M):
        assert src.count(a) == 1, a
        ms = src.replace(a, b)
        f = os.path.join(d, 'io%d.rce' % i); open(f, 'w').write(ms + '\n' + chk)
        dead = subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0
        if not dead:
            mf = os.path.join(d, 'm%d.rce' % i); open(mf, 'w').write(ms)
            flat = subprocess.run([sys.executable, os.path.join(root, 'scripts/flatten.py'), mf, os.path.join(root, 'tests/io/io_checks.rce'), os.path.join(root, 'tests/io/io_host.rce')], capture_output=True, text=True).stdout
            ff = os.path.join(d, 'f%d.rce' % i); open(ff, 'w').write(flat)
            fs = os.path.join(d, 'fs%d' % i); os.mkdir(fs)
            r = subprocess.run([RCE, 'host', 'run', ff, '--log', os.path.join(d, 'log%d' % i), '--fs', fs], input='\n'.join(events) + '\n', capture_output=True, text=True)
            dead = ('good = %d, bad = 0 }' % len(events)) not in r.stdout
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
