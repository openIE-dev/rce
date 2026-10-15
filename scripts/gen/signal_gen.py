# Generates tests/os/signal_checks*.rce for lib/os/signal.rce.
#  - official tables: signal(7) (vectors/os/signal.7: default actions and
#    standards; numbers on x86/ARM, Alpha/SPARC, MIPS, PA-RISC), the Linux
#    x86 uapi header (vectors/os/linux_x86_signal.h) and xnu's
#    bsd/sys/signal.h (vectors/os/xnu_signal.h), parsed here;
#  - live on this machine: glibc strsignal (ctypes), bash kill -l, Python
#    signal.Signals names;
#  - pending-signal scripts run by the reference program
#    scripts/ref/signal_ref.c (sigqueue, sigaction, sigtimedwait with every
#    signal blocked) on the real kernel.
# Usage: python3 scripts/gen/signal_gen.py   (needs gcc and bash)
import ctypes, os, random, re, signal, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '../..')
V = lambda n: open(os.path.join(root, 'vectors/os', n)).read()
def lit(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
rows = []
# signal(7) action table
s7 = V('signal.7')
act = {}
for m in re.finditer(r'^(SIG[A-Z0-9]+)\t(P1990|P2001|P2024|\\-)\t(Term|Core|Ign|Stop|Cont|)\t', s7, re.M):
    act[m.group(1)] = (m.group(3) or '-', '-' if m.group(2) == '\\-' else m.group(2))
assert len(act) >= 38, len(act)
for k, (a, st) in sorted(act.items()):
    rows.append('check act(%s) == %s' % (lit(k), lit(a)))
    rows.append('check std(%s) == %s' % (lit(k), lit(st)))
# signal(7) numbers table
tab = s7[s7.index('Signal\tx86/ARM'):]
tab = tab[:tab.index('.TE')]
nums = {}
for line in tab.splitlines():
    f = line.split('\t')
    if len(f) >= 5 and f[0].startswith('SIG'):
        cl = lambda x: x.replace('\\0', '').replace('\\-', '-').strip()
        nums[f[0]] = [cl(x) for x in f[1:5]]
assert len(nums) >= 38
def num(x):
    return -1 if x in ('-', '') else int(x)
for k, (x86, asp, mips, pa) in sorted(nums.items()):
    if x86 == '':  # SIGPOLL: "Same as SIGIO"
        x86, asp, mips, pa = nums['SIGIO']
    a, sp = (asp.split('/') + [asp])[:2] if '/' in asp else (asp, asp)
    for t, v in [('linux', x86), ('alpha', a), ('sparc', sp), ('mips', mips), ('parisc', pa)]:
        rows.append('check nm(%s, "%s") == %d' % (lit(k), t, num(v)))
# Linux uapi header agrees with the x86 column
for m in re.finditer(r'^#define\s+(SIG[A-Z0-9]+)\s+(\d+)\s*$', re.sub(r'/\*.*?\*/', '', V('linux_x86_signal.h'), flags=re.S), re.M):
    if m.group(1) in ('SIGRTMIN', 'SIGSTKSZ'):
        continue
    rows.append('check nm(%s, "linux") == %s' % (lit(m.group(1)), m.group(2)))
# xnu (default build: not _POSIX_C_SOURCE, so SIGEMT rather than SIGPOLL at 7)
xnu = V('xnu_signal.h')
xnu = xnu[:xnu.index('SIGSTKSZ')]
dar = {}
for m in re.finditer(r'^#define\s+(SIG[A-Z0-9]+)\s+(\d+)\b', xnu, re.M):
    if m.group(1) == 'SIGPOLL':
        continue
    dar[m.group(1)] = int(m.group(2))
for m in re.finditer(r'^#define\s+(SIG[A-Z0-9]+)\s+(SIG[A-Z0-9]+)\b', xnu, re.M):
    if m.group(2) in dar:
        dar[m.group(1)] = dar[m.group(2)]
assert len(dar) >= 31
for k, v in sorted(dar.items()):
    rows.append('check nm(%s, "darwin") == %d' % (lit(k), v))
for k in sorted(set(nums) - set(dar)):
    rows.append('check nm(%s, "darwin") == -1' % lit(k))
for n in range(1, 32):
    pref = sorted(k for k, v in dar.items() if v == n and k not in ('SIGIOT',))
    if len(pref) == 1:
        rows.append('check nameof(%d, "darwin") == %s' % (n, lit(pref[0])))
# live: Python's names, glibc strsignal, bash kill -l
for n in range(1, 32):
    rows.append('check nameof(%d, "linux") == %s' % (n, lit(signal.Signals(n).name)))
libc = ctypes.CDLL(None); libc.strsignal.restype = ctypes.c_char_p
for n in range(-2, 68):
    rows.append('check strsignal(%d) == %s' % (n, lit(libc.strsignal(n).decode())))
for n in range(0, 66):
    r = subprocess.run(['bash', '-c', 'kill -l %d' % n], capture_output=True, text=True)
    rows.append('check kill_l(%d) == %s' % (n, lit(r.stdout.strip() if r.returncode == 0 else '')))
# pending-signal scripts
tmp = tempfile.mkdtemp(); ref = os.path.join(tmp, 'signal_ref')
subprocess.run(['gcc', '-O2', '-o', ref, os.path.join(root, 'scripts/ref/signal_ref.c')], check=True)
rng = random.Random(40)
SIGS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 40, 50, 63, 64]
def script():
    pool = rng.sample(SIGS, rng.choice([2, 3, 5, 8]))
    ops = []
    for _ in range(rng.choice([4, 8, 12, 20])):
        k = rng.random()
        if k < 0.5:
            ops.append('send %d %d' % (rng.choice(pool), rng.randint(0, 99)))
        elif k < 0.62:
            ops.append('%s %d' % (rng.choice(['ign', 'dfl', 'catch']), rng.choice(pool)))
        else:
            ops.append('wait ' + ','.join(str(x) for x in rng.sample(pool, rng.randint(1, len(pool)))))
    return ';'.join(ops)
scripts = [script() for _ in range(1200)]
outs = subprocess.run([ref], input='\n'.join(scripts) + '\n', capture_output=True, text=True, check=True).stdout.splitlines()
assert len(outs) == len(scripts)
srows = ['check run(%s) == %s' % (lit(a), lit(b)) for a, b in zip(scripts, outs)]
hdr = '# Generated by scripts/gen/signal_gen.py (kernel %s, %s); do not edit.\n' % (os.uname().release, subprocess.run(['ldd', '--version'], capture_output=True, text=True).stdout.splitlines()[0])
helpers = open(os.path.join(here, 'signal_helpers.rce')).read()
shards = {'signal_checks.rce': rows + open(os.path.join(here, 'signal_derived.rce')).read().splitlines()}
for i in range(0, len(srows), 400):
    shards['signal_checks_%d.rce' % (i // 400 + 1)] = srows[i:i + 400]
for name, rs in shards.items():
    with open(os.path.join(root, 'tests/os', name), 'w') as f:
        f.write(hdr + helpers + '\n'.join(rs) + '\n')
print(len(rows), len(srows))
