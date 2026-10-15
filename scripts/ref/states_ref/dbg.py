# find the actual output of a failing lv() check by probing with checks
import re, subprocess, sys
RCE = '/tmp/rce-ide-0.1.3-x86_64-unknown-linux-gnu/rce'
lib = sys.argv[1]; tests = sys.argv[2]; idx = int(sys.argv[3])
s = open(tests).read(); H = s[:s.index('check l')]
cs = re.findall(r'^check (l[vx]\(.*\)) == (".*")$', s, re.M)
call, exp = cs[idx]
base = open(lib).read() + H + 'fn gotx() -> text = ' + call + '\n'
def ok(c):
    open('/tmp/probe.rce', 'w').write(base + c + '\n')
    r = subprocess.run([RCE, 'test', '/tmp/probe.rce'], capture_output=True, text=True)
    return 'check is false' not in r.stdout + r.stderr and r.returncode == 0
def q(x): return '"' + x.replace('\\', '\\\\').replace('"', '\\"') + '"'
alpha = sorted(set(' +-|,0123456789abcdefghijklmnopqrstuvwxyz_.'))
e = eval(exp)
lo, hi = 0, len(e)
while lo < hi:
    m = (lo + hi + 1) // 2
    if ok('check gotx().startswith(%s)' % q(e[:m])): lo = m
    else: hi = m - 1
got = e[:lo]
for _ in range(60):
    if ok('check gotx() == %s' % q(got)): break
    found = False
    for ch in alpha:
        if ok('check gotx().startswith(%s)' % q(got + ch)): got += ch; found = True; break
    if not found: got += '?'; break
print('exp', exp); print('got', got)
