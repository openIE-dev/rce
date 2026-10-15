# Generates tests/os/proc_checks*.rce for lib/os/proc.rce.
#  - official: Rust std's exitstatus_display_tests and debug_print
#    (vectors/os/rust_process_unix_tests.rs, rust_process_tests.rs,
#    restated in scripts/gen/proc_official.rce) and CPython's test_shlex
#    (vectors/os/test_shlex.py: posix_data, testQuote, testJoin,
#    testJoinRoundtrip), parsed from the vendored file;
#  - live: wait statuses of real processes on this kernel (exit codes,
#    deaths by signal with and without core dumps, stops and continues
#    seen through waitpid WUNTRACED | WCONTINUED), described by the
#    reference program scripts/ref/proc_ref.rs (Rust std, rustc 1.99.0);
#  - random: raw statuses and Command builds described by proc_ref, and
#    random words through Python's shlex.split / quote / join.
# Usage: python3 scripts/gen/proc_gen.py  (needs rustc; writes tests/os)
import os, random, resource, shlex, signal, subprocess, sys, tempfile, time
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '../..')
def lit(s):
    o = ''
    for c in s:
        o += {'"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t'}.get(c) or (c if ord(c) >= 32 and ord(c) != 127 and not 0x80 <= ord(c) < 0xa0 else '\\u{%x}' % ord(c))
    return '"' + o + '"'
US = '\x1f'
tmp = tempfile.mkdtemp()
ref = os.path.join(tmp, 'proc_ref')
subprocess.run(['rustc', '-O', '-o', ref, os.path.join(root, 'scripts/ref/proc_ref.rs')], check=True)
rv = subprocess.run(['rustc', '--version'], capture_output=True, text=True).stdout.strip()
def ask(lines):
    out = subprocess.run([ref], input='\n'.join(lines) + '\n', capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(lines)
    return out
def status_rows(raws):
    rows = []
    for r, line in zip(raws, ask(['S %d' % r for r in raws])):
        f = line.split(' ')
        rows.append('check st_row(%d) == %s' % (r, lit('|'.join([bytes.fromhex(f[0]).decode()] + f[1:]))))
    return rows
# ---- live statuses
def live():
    raws = []
    def run(argv, pre=None):
        p = subprocess.Popen(argv, preexec_fn=pre, cwd=tmp)
        return os.waitpid(p.pid, 0)[1]
    for n in list(range(0, 256, 1)):
        raws.append(run(['sh', '-c', 'exit %d' % n]))
    def core():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    def coreon():
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (1 << 30, 1 << 30))
        except ValueError:
            pass
    for sig in range(1, 32):
        if sig in (signal.SIGSTOP, signal.SIGTSTP, signal.SIGTTIN, signal.SIGTTOU, signal.SIGCHLD, signal.SIGCONT, signal.SIGURG, signal.SIGWINCH):
            continue
        raws.append(run(['sh', '-c', 'kill -%d $$; sleep 5' % sig], core))
        raws.append(run(['sh', '-c', 'kill -%d $$; sleep 5' % sig], coreon))
    for sig in (signal.SIGSTOP, signal.SIGSTOP):  # TSTP, TTIN, TTOU are discarded here (orphaned process group)
        p = subprocess.Popen(['sleep', '5'])
        os.kill(p.pid, sig)
        raws.append(os.waitpid(p.pid, os.WUNTRACED)[1])
        os.kill(p.pid, signal.SIGCONT)
        raws.append(os.waitpid(p.pid, os.WCONTINUED)[1])
        os.kill(p.pid, signal.SIGKILL)
        raws.append(os.waitpid(p.pid, 0)[1])
    return raws
# ---- shlex (official)
def shlex_official():
    src = open(os.path.join(root, 'vectors/os/test_shlex.py')).read()
    ns = {}
    exec(compile('import shlex\n' + src[src.index('data = r"""'):src.index('class ShlexTest')], 'test_shlex', 'exec'), ns)
    data = [x.split('|')[:-1] for x in ns['data'].splitlines()]
    posix = [x.split('|')[:-1] for x in ns['posix_data'].splitlines()]
    for it in data + posix:
        it[0] = it[0].replace(r'\n', '\n')
    rows = []
    for it in posix:  # testSplitPosix
        rows.append('check sp(%s) == %s' % (lit(it[0]), lit(US.join(it[1:]))))
    safe = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' + '@%_-+=:,./'
    unsafe = '"`$\\!' + '\xe9\xe0\xdf'
    q = [('', "''"), (safe, safe), ('test file name', "'test file name'")]
    q += [('test%sname' % u, "'test%sname'" % u) for u in unsafe]
    q += [("test%s'name'" % u, "'test%s'\"'\"'name'\"'\"''" % u) for u in unsafe]
    for a, b in q:  # testQuote
        rows.append('check qt(%s) == %s' % (lit(a), lit(b)))
    for ws, cmd in [(['a ', 'b'], "'a ' b"), (['a', ' b'], "a ' b'"), (['a', ' ', 'b'], "a ' ' b"), (['"a', 'b"'], '\'"a\' \'b"\'')]:
        rows.append('check jn(%s) == %s' % (lit(US.join(ws)), lit(cmd)))  # testJoin
    for it in data + posix:  # testJoinRoundtrip
        rows.append('check rt(%s, %d)' % (lit(US.join(it[1:])), len(it) - 1))
    return rows
# ---- random
def rstr(rng, alpha):
    return ''.join(rng.choice(alpha) for _ in range(rng.randint(0, 6)))
ALPHA = 'abcXYZ019 -_./=:"\'\\$\n\t\r\x00\x01\x1b\x7f\x85\xa0\xad\xe9\xfc\u2713'
def rcmd(rng):
    ops = [('prog', rng.choice(['ls', 'sh', '/bin/echo', 'a b', 'x"y', '\xe9t\xe9']))]
    for _ in range(rng.randint(0, 8)):
        k = rng.choice(['arg', 'arg', 'arg', 'arg0', 'env', 'env', 'rm', 'rm', 'clear', 'cwd', 'stdin', 'stdout', 'stderr', 'pgroup', 'uid', 'gid'])
        if k in ('arg', 'arg0', 'cwd'):
            v = rstr(rng, ALPHA)
        elif k in ('env', 'rm'):
            key = rng.choice(['A', 'B', 'PATH', 'Z_1', 'a', '\xe9']) 
            v = key + '=' + rstr(rng, ALPHA) if k == 'env' else key
        elif k == 'clear':
            v = ''
        elif k.startswith('std'):
            v = rng.choice(['null', 'piped', 'inherit'])
        else:
            v = str(rng.choice([0, 1, rng.randint(0, 70000)]))
        ops.append((k, v))
    return ops
def cmd_rows(cmds):
    lines = ['C ' + ' '.join('%s:%s' % (k, v.encode().hex()) for k, v in c) for c in cmds]
    rows = []
    for c, line in zip(cmds, ask(lines)):
        a, b = line.split(' ')
        spec = lit(US.join('%s:%s' % (k, v) for k, v in c))
        rows.append('check dbg(%s) == %s' % (spec, lit(bytes.fromhex(a).decode())))
        rows.append('check dbgp(%s) == %s' % (spec, lit(bytes.fromhex(b).decode())))
    return rows
SH = 'ab \t\n#\'"\\$x\xe9;|&'
def shlex_random(rng, n):
    rows = []
    for _ in range(n):
        s = ''.join(rng.choice(SH) for _ in range(rng.randint(0, 12)))
        try:
            r = US.join(shlex.split(s, comments=True))
            if r == '' and shlex.split(s, comments=True) == []:
                r = '!empty'
        except ValueError as e:
            r = '!' + str(e)
        rows.append('check sp(%s) == %s' % (lit(s), lit(r)))
        rows.append('check qt(%s) == %s' % (lit(s), lit(shlex.quote(s))))
    return rows
def main():
    rng = random.Random(39)
    hdr = '# Generated by scripts/gen/proc_gen.py (%s, Python %s, kernel %s); do not edit.\n' % (rv, sys.version.split()[0], os.uname().release)
    helpers = open(os.path.join(here, 'proc_helpers.rce')).read()
    off = open(os.path.join(here, 'proc_official.rce')).read()
    lv = live()
    shards = {
        'proc_checks.rce': [off] + shlex_official() + status_rows(sorted(set(lv))) + shlex_random(rng, 400),
    }
    raws = sorted(set(list(range(0, 1024)) + [rng.randint(0, 65535) for _ in range(1500)] + [0xffff, 0x10000, 0x7fffffff, -1, -2, -256, -2147483648, 0x137f, 0xff, 0x17f]))
    sr = status_rows(raws)
    cmds = cmd_rows([rcmd(rng) for _ in range(600)])
    shards['proc_checks_1.rce'] = sr[:len(sr) // 2]
    shards['proc_checks_2.rce'] = sr[len(sr) // 2:]
    shards['proc_checks_3.rce'] = cmds[:len(cmds) // 2]
    shards['proc_checks_4.rce'] = cmds[len(cmds) // 2:]
    for name, rows in shards.items():
        with open(os.path.join(root, 'tests/os', name), 'w') as f:
            f.write(hdr + helpers + '\n'.join(rows) + '\n')
    print('live statuses: %d distinct' % len(set(lv)))
main()
