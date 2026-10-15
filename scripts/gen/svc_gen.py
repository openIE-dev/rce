# Generates tests/os/svc_checks*.rce for lib/os/svc.rce. Every expected
# value is computed by scripts/ref/svc_ref.c: systemd's own exit-status,
# number, signal and Restart= code copied verbatim (svc_ref_build.py) and
# built with the host's gcc and glibc (2.38+, whose strtoul reads C23
# prefixes as systemd's builds do). The official cases of systemd's
# src/test/test-exit-status.c and src/test/test-signal-util.c are
# asserted against the reference first, then kept as checks.
# Usage: python3 scripts/gen/svc_gen.py /path/to/systemd
import itertools, os, random, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '../..')
S = sys.argv[1]
ref_c = subprocess.run(['python3', os.path.join(root, 'scripts/ref/svc_ref_build.py'), S], capture_output=True, text=True, check=True).stdout
open(os.path.join(root, 'scripts/ref/svc_ref.c'), 'w').write(ref_c)
subprocess.run(['gcc', '-O1', '-w', '-o', '/tmp/svc_ref', os.path.join(root, 'scripts/ref/svc_ref.c')], check=True)
def ref(lines):
    for l in lines: assert '\n' not in l
    out = subprocess.run(['/tmp/svc_ref'], input='\n'.join(lines) + '\n', capture_output=True, text=True, check=True).stdout
    r = out.split('\n')[:-1]; assert len(r) == len(lines); return r
def q(s):
    o = ''
    for c in s:
        if c == '\\': o += '\\\\'
        elif c == '"': o += '\\"'
        elif c == '\n': o += '\\n'
        elif c == '\t': o += '\\t'
        elif c == '\r': o += '\\r'
        elif ord(c) < 32: o += '\\u{%x}' % ord(c)
        else: o += c
    return '"' + o + '"'
rng = random.Random(44)
EINVAL, ERANGE = -22, -34
rows = []
# official: test-exit-status.c
off_es = [('11', 11), ('-1', -ERANGE * -1 * -1), ('256', ERANGE), ('foo', EINVAL), ('SUCCESS', 0), ('FAILURE', 1)]
off_es[1] = ('-1', ERANGE)
got = ref(['es\t' + s for s, _ in off_es]); assert got == [str(v) for _, v in off_es], got
names = [v.split(' ')[0] for v in ref(['name\t%d' % i for i in range(256)]) if v != '- -']
# every status 0..255 by number and its name (test-exit-status.c loops over them)
es_in = [s for s, _ in off_es] + [str(i) for i in range(256)]
sig_off = [('RTMIN', 34), ('RTMAX', 64), ('RTMIN+30', 64), ('RTMIN+31', ERANGE), ('RTMAX-30', 34), ('RTMAX-31', ERANGE), ('', EINVAL), ('hup', EINVAL), ('HOGEHOGE', EINVAL),
    ('RTMIN-5', EINVAL), ('RTMIN-    5', EINVAL), ('RTMIN    -5', EINVAL), ('RTMIN+    5', EINVAL), ('RTMIN    +5', EINVAL), ('RTMIN+100', ERANGE), ('RTMIN+-3', EINVAL), ('RTMIN++3', EINVAL), ('RTMIN+HUP', EINVAL), ('RTMIN3', EINVAL),
    ('RTMAX+5', EINVAL), ('RTMAX+    5', EINVAL), ('RTMAX    +5', EINVAL), ('RTMAX-    5', EINVAL), ('RTMAX    -5', EINVAL), ('RTMAX-100', ERANGE), ('RTMAX-+3', EINVAL), ('RTMAX--3', EINVAL), ('RTMAX-HUP', EINVAL)]
sig_in = []
for s, v in sig_off:  # test_signal_from_string_one: s and SIG+s
    sig_in += [s, 'SIG' + s]
    got = ref(['sig\t' + s, 'sig\tSIG' + s]); assert got == [str(v)] * 2, (s, got)
for s, v in [('3', 3), ('+5', 5), ('  +5', 5), ('10000', ERANGE), ('-2', ERANGE)]:  # _number: SIG+s is EINVAL
    got = ref(['sig\t' + s, 'sig\tSIG' + s]); assert got == [str(v), str(EINVAL)], (s, got)
    sig_in += [s, 'SIG' + s]
SIGN = ['HUP', 'INT', 'QUIT', 'ILL', 'TRAP', 'ABRT', 'BUS', 'FPE', 'KILL', 'USR1', 'SEGV', 'USR2', 'PIPE', 'ALRM', 'TERM', 'STKFLT', 'CHLD', 'CONT', 'STOP', 'TSTP', 'TTIN', 'TTOU', 'URG', 'XCPU', 'XFSZ', 'VTALRM', 'PROF', 'WINCH', 'IO', 'PWR', 'SYS']
for i in range(0, 66):  # test_signal_to_string_one over every signal
    sig_in += [str(i)]
    if 1 <= i <= 31: sig_in += [SIGN[i - 1], 'SIG' + SIGN[i - 1]]
    if 34 <= i <= 64: sig_in += ['RTMIN+%d' % (i - 34), 'SIGRTMIN+%d' % (i - 34), 'RTMAX-%d' % (64 - i)]
# random words in systemd's number syntax and around the names
def numword():
    n = rng.choice([0, 1, 2, 7, 8, 9, 15, 31, 64, 65, 127, 255, 256, 1000, 2**31 - 1, 2**31, 2**32, 2**64, 2**70, rng.randrange(300)])
    f = rng.choice(['%d', '0x%x', '0X%X', '0%o', '0o%o', '0O%o', '0b{b}', '0B{b}', '%d'])
    t = f.format(b=bin(n)[2:]) if '{b}' in f else f % n
    if rng.random() < .3: t = rng.choice(['+', '-', ' ', '  ', '\t', ' -', ' +', '\u000b', '-+', '0x', '0b', '0o']) + t
    if rng.random() < .15: t = t + rng.choice(['x', ' ', '8', '9', 'g', '.', '0b1'])
    if rng.random() < .1: t = t[:rng.randrange(len(t) + 1)]
    return t
def word():
    k = rng.random()
    if k < .5: return numword()
    if k < .7: return rng.choice(SIGN + ['RTMIN', 'RTMAX', 'hup', 'Term', 'SIG', 'SIGSIGHUP', 'SIGHUP ', 'TERM9']) if rng.random() < .5 else 'SIG' + rng.choice(SIGN)
    if k < .85: return rng.choice(['RTMIN', 'RTMAX']) + rng.choice(['+', '-', '']) + numword()
    return rng.choice(names or ['SUCCESS']) + rng.choice(['', '', '', ' ', 'X'])
for _ in range(1500): es_in.append(word()); sig_in.append(word())
# mangle_base before strtoul's own sign and prefix handling
for p in ['0b', '0B', '0o', '0O', '0x', '0X']:
    for t in ['-1', '+1', ' 1', '-0', '0b1', '0x1', '2', '8', '']:
        es_in.append(p + t); sig_in.append(p + t); sig_in.append('RTMIN+' + p + t); sig_in.append('RTMAX-' + p + t)
es_in = [w for w in es_in if '\n' not in w and '\t' not in w]; sig_in = [w for w in sig_in if '\n' not in w and '\t' not in w]
for w, v in zip(es_in, ref(['es\t' + w for w in es_in])): rows.append('check esf(%s) == %s' % (q(w), v if not v.startswith('-') else '0 - ' + v[1:]))
for w, v in zip(sig_in, ref(['sig\t' + w for w in sig_in])): rows.append('check sgf(%s) == %s' % (q(w), v if not v.startswith('-') else '0 - ' + v[1:]))
# names and classes of every status, names of signals
import re
ec = open(S + '/src/shared/exit-status.c').read()
for i in range(256):
    pass
shards = [rows]
rows = []
# sets
def setword():
    return word().replace('\t', ' ')
sets = ['', ' ', 'SIGTERM', '1 2 SIGKILL', 'TEMPFAIL 250 SIGUSR1', '0 SIGHUP SIGINT', '75 SIGPIPE 6', 'foo bar 3'] + [' '.join(setword() for _ in range(rng.randrange(1, 5))) for _ in range(400)]
sets = [s for s in sets if not any(c in s for c in '"\'\\\n\t')]
for s, v in zip(sets, ref(['set\t' + s for s in sets])):
    st, sg = v.split(';')
    rows.append('check sset(%s) == %s' % (q(s), q(v)))
shards.append(rows); rows = []
codes = [('exited', [0, 1, 2, 3, 6, 15, 75, 143, 250, 255]), ('killed', [1, 2, 6, 9, 11, 13, 15, 34, 64]), ('dumped', [6, 11, 15])]
small_sets = ['', '1 SIGKILL', '75 SIGPIPE 250', 'SIGTERM SIGUSR1 6', '0 15 SIGABRT', '2']
L = []
for c, sts in codes:
    for st in sts:
        for mode in ['daemon', 'command']:
            for ss in small_sets:
                L.append(('clean', c, st, mode, ss))
                L.append(('clean', c, st, mode, '-', ss))
        for ty in ['simple', 'oneshot', 'notify']:
            for ss in small_sets:
                L.append(('result', ty, c, st, ss))
got = ref(['\t'.join(map(str, x[:5])) for x in L])
for x, v in zip(L, got):
    if x[0] == 'clean':
        # without SuccessExitStatus= (systemd's helpers) the set is ignored
        rows.append('check clean(%s, %d, %s, %s, %s) == %s' % (q(x[1]), x[2], 'true' if x[3] == 'daemon' else 'false', q(x[-1]), 'false' if x[4] == '-' else 'true', 'true' if v == '1' else 'false'))
    else:
        rows.append('check xres(%s, %s, %d, %s) == %s' % (q(x[1]), q(x[2]), x[3], q(x[4]), q(v)))
shards.append(rows); rows = []
R = ['no', 'on-success', 'on-failure', 'on-abnormal', 'on-watchdog', 'on-abort', 'always']
RES = ['success', 'resources', 'protocol', 'timeout', 'exit-code', 'signal', 'core-dump', 'watchdog', 'start-limit-hit', 'oom-kill', 'condition']
L = []
for r in R:
    for ty in ['simple', 'oneshot']:
        for res in RES:
            for c, st in [('exited', 0), ('exited', 3), ('killed', 15), ('killed', 9), ('dumped', 11)]:
                for pv, fc in [('', ''), ('3', ''), ('', '3 SIGKILL'), ('SIGTERM', '0'), ('', '0 SIGTERM'), ('3', '3'), ('SIGSEGV', ''), ('', '11 SIGSEGV')]:
                    for fb in ['0', '1']:
                        if fb == '1' and rng.random() < .8: continue
                        L.append(('restart', r, ty, res, c, st, pv, fc, fb))
got = ref(['\t'.join(map(str, x)) for x in L])
for x, v in zip(L, got):
    y, reason = v.split(' ', 1)
    rows.append('check rs(%s, %s, %s, %s, %d, %s, %s, %s) == %s' % (q(x[1]), q(x[2]), q(x[3]), q(x[4]), x[5], q(x[6]), q(x[7]), 'true' if x[8] == '1' else 'false', q(y + ' ' + reason)))
# names and classes (exit_status_to_string / exit_status_class over 0..255)
for i, v in zip(range(257), ref(['name\t%d' % i for i in range(257)])):
    n, c = [x if x != '-' else '' for x in v.split(' ')]
    rows.append('check exit_status_name(%d) == %s and exit_status_class(%d) == %s' % (i, q(n), i, q(c)))
for i in range(0, 66):
    want = SIGN[i - 1] if 1 <= i <= 31 else ('RTMIN+%d' % (i - 34) if 34 <= i <= 64 else str(i))
    rows.append('check signal_name(%d) == %s' % (i, q(want)))
shards.append(rows)
helpers = open(os.path.join(here, 'svc_helpers.rce')).read()
for f in os.listdir(os.path.join(root, 'tests/os')):
    if f.startswith('svc_checks'): os.remove(os.path.join(root, 'tests/os', f))
n = 0; k = 0
out = []
for rs in shards:
    for i in range(0, len(rs), 1500): out.append(rs[i:i + 1500])
for k, rs in enumerate(out):
    name = 'svc_checks.rce' if k == 0 else 'svc_checks_%d.rce' % k
    open(os.path.join(root, 'tests/os', name), 'w').write('# Generated by scripts/gen/svc_gen.py; do not edit.\n' + helpers + '\n'.join(rs) + '\n')
    n += len(rs)
print(n, 'checks in', len(out), 'files')
