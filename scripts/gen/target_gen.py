# Generates tests/os/target_checks*.rce for lib/os/target.rce. The
# reference is the WASI test suite's own runner (wasi-testsuite,
# test-runner/wasi_test_runner), imported as it is:
#  - every test configuration JSON in the suite's tests/ tree is read with
#    Config.from_file and must give the same operations;
#  - random configurations (legacy and operations style, well typed) give
#    the same operations or the same ValueError message;
#  - TestCaseRunner.run() is driven with the observed output of a fake
#    process (its pipes are StringIO, _wait returns the observed exit
#    code or raises TimeoutExpired), and the first failing operation must
#    be the one verdict() names; Outcome.evaluate gives the outcomes.
# Usage: python3 scripts/gen/target_gen.py /path/to/wasi-testsuite
import glob, io, json, os, random, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '../..')
W = sys.argv[1]
sys.path.insert(0, os.path.join(W, 'test-runner'))
import logging; logging.disable(logging.CRITICAL)
from wasi_test_runner import test_case as tc
from wasi_test_runner.test_suite_runner import TestCaseRunner
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
def summary(cfg, path):
    out = []
    for op in cfg.operations:
        if isinstance(op, tc.Run):
            r = '' if not op.root else os.path.relpath(str(op.root), os.path.dirname(path))
            out.append('run:' + ','.join(op.args) + '|' + ','.join('%s=%s' % kv for kv in op.env.items()) + '|' + r)
        elif isinstance(op, tc.Wait): out.append('wait:%d' % op.exit_code)
        elif isinstance(op, tc.Read): out.append('read:%s:%s' % (op.id, op.payload))
        elif isinstance(op, tc.Write): out.append('write:%s:%s' % (op.id, op.payload))
        elif isinstance(op, tc.Kill): out.append('kill:' + op.signal.name if op.signal.name in (k for k in __import__('signal').Signals.__members__) else 'kill:?')
        else: out.append(type(op).__name__.lower())
    return ';'.join(out) + '#' + cfg.world.value + '#' + ','.join(p.value for p in cfg.proposals)
def ref_parse(text):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 't.json'); open(p, 'w').write(text)
        try: return summary(tc.Config.from_file(p), p)
        except ValueError as e: return '!' + str(e)
def ref_verdict(cfg, stdout, stderr, code, timeout):
    r = TestCaseRunner.__new__(TestCaseRunner)
    tc.TestCaseRunnerBase.__init__(r, cfg)
    r._pipes = {}; r._proc = None; r._last_argv = []; r._windows_terminated_by_runner = False
    st = {'i': -1, 'fail': None}
    def do_run(run):
        r._pipes = {'stdin': io.StringIO(), 'stdout': io.StringIO(stdout), 'stderr': io.StringIO(stderr)}; r._proc = object()
    def _wait(timeout_s):
        assert r._proc is not None  # as TestCaseRunner._wait
        if timeout: raise subprocess.TimeoutExpired('x', 1)
        r._proc = None
        out = (code, r._pipes['stdout'].read(), r._pipes['stderr'].read())
        for v in r._pipes.values(): v.close()  # communicate() closes the pipes
        return out
    r.do_run = do_run; r._wait = _wait; r.do_cleanup = lambda ok: None
    r._runtime = type('RT', (), {'get_timeout_seconds': lambda self: 1})()
    orig = {k: getattr(r, k) for k in ['do_run', 'do_read', 'do_write', 'do_wait']}
    def wrap(k):
        def f(op):
            st['i'] += 1
            n = len(r._failures); orig[k](op)
            if len(r._failures) > n and st['fail'] is None: st['fail'] = '%s %d' % (k[3:], st['i'])
        return f
    for k in orig: setattr(r, k, wrap(k))
    try:
        res = r.run()
    except (ValueError, AssertionError):  # pipes closed / process collected by an earlier wait
        return 'error %d' % st['i']
    assert res.failed == (st['fail'] is not None)
    return st['fail'] or ''
rng = random.Random(45)
rows = []
# 1. every configuration in the suite
files = sorted(glob.glob(os.path.join(W, 'tests/**/*.json'), recursive=True))
real = []
for f in files:
    t = open(f).read()
    try: j = json.loads(t)
    except Exception: continue
    if not isinstance(j, dict) or not (set(j) & (tc.LEGACY_CONFIG_KEYS | tc.CONFIG_KEYS)): continue
    s = ref_parse(t)
    if j.get('servers') or any(o.get('type') not in ('run', 'wait', 'read', 'write', 'kill') for o in j.get('operations') or []):
        continue  # http / socket operations: their payloads are not modelled
    real.append((os.path.relpath(f, W), t, s))
for name, t, s in real:
    rows.append('# %s\ncheck pp(%s) == %s' % (name, q(t), q(s)))
shards = [rows]; rows = []
# 2. random configurations
WORDS = ['', 'a', 'hello', 'x y', 'ü', '\n', '0', '--flag', 'fs-tests.dir', 'é\n']
SIG = list(__import__('signal').Signals.__members__) + ['SIGFOO', 'sigterm', 'TERM']
def legacy():
    d = {}
    if rng.random() < .6: d['args'] = [rng.choice(WORDS) for _ in range(rng.randrange(4))]
    if rng.random() < .4: d['env'] = {rng.choice(['A', 'B', 'PATH', 'X_1']): rng.choice(WORDS) for _ in range(rng.randrange(3))}
    if rng.random() < .2: d['root'] = rng.choice(['fs-tests.dir', '', 'dir'])
    if rng.random() < .5: d['exit_code'] = rng.choice([0, 1, 2, 42, 255, -1])
    if rng.random() < .5: d['stdout'] = rng.choice(WORDS + ['hello\n'])
    if rng.random() < .3: d['stderr'] = rng.choice(WORDS)
    if rng.random() < .05: d['operations'] = []
    if rng.random() < .05: d['bogus'] = 1
    return d
def opcfg():
    ty = rng.choice(['run', 'wait', 'read', 'write', 'kill', 'nope', None, 'Run'])
    o = {} if ty is None else {'type': ty}
    if ty == 'run':
        if rng.random() < .5: o['args'] = [rng.choice(WORDS) for _ in range(rng.randrange(3))]
        if rng.random() < .3: o['env'] = {'A': rng.choice(WORDS)}
    if ty == 'wait' and rng.random() < .7: o['exit_code'] = rng.choice([0, 1, 3, 255])
    if ty in ('read', 'write'):
        if rng.random() < .7: o['id'] = rng.choice(['stdout', 'stderr', 'stdin']) if ty == 'read' else 'stdin'
        if rng.random() < .8: o['payload'] = rng.choice(WORDS + ['hello\n', 'abc'])
    if ty == 'kill' and rng.random() < .7: o['signal'] = rng.choice(SIG)
    return o
def v1():
    d = {}
    if rng.random() < .9: d['operations'] = [opcfg() for _ in range(rng.randrange(5))]
    if rng.random() < .3: d['proposals'] = rng.sample(['http', 'sockets', 'threads'], rng.randrange(3))
    if rng.random() < .3: d['world'] = rng.choice(['wasi:cli/command', 'wasi:http/service', 'wasi:cli/run'])
    if rng.random() < .05: d['stdout'] = 'x'
    if rng.random() < .05: d['debug'] = True
    return d
for d in [{'servers': []}, {'servers': [], 'args': []}, {'servers': [], 'operations': [{'type': 'run'}]}, {'operations': [{'type': 'kill'}]}, {'operations': [{'type': 'run'}, {'type': 'kill'}, {'type': 'wait'}]}, {'operations': [{'type': 'kill', 'signal': 'SIGCLD'}, {'type': 'kill', 'signal': 'SIGPOLL'}]}]:
    t = json.dumps(d); rows.append('check pp(%s) == %s' % (q(t), q(ref_parse(t))))
for _ in range(600):
    t = json.dumps(legacy() if rng.random() < .5 else v1(), ensure_ascii=rng.random() < .5)
    rows.append('check pp(%s) == %s' % (q(t), q(ref_parse(t))))
shards.append(rows); rows = []
# 3. verdicts: real and random plans against observed output
def cfg_of(t):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 't.json'); open(p, 'w').write(t); return tc.Config.from_file(p)
cases = []
for name, t, s in real:
    c = cfg_of(t)
    reads = [o for o in c.operations if isinstance(o, tc.Read)]
    waits = [o for o in c.operations if isinstance(o, tc.Wait)]
    if any(isinstance(o, tc.Kill) for o in c.operations) or any(isinstance(o, tc.Read) and o.id not in ('stdout', 'stderr') for o in c.operations): continue
    good_out = ''.join(o.payload for o in reads if o.id == 'stdout'); good_err = ''.join(o.payload for o in reads if o.id == 'stderr')
    code = waits[-1].exit_code if waits else 0
    for so, se, ec, to in [(good_out, good_err, code, False), (good_out + 'tail', good_err, code, False), (good_out[:-1], good_err, code, False), (good_out, good_err, code + 1, False), (good_out, good_err, code, True), ('', '', code, False)]:
        cases.append((t, so, se, ec, to))
for _ in range(500):
    ops = [{'type': 'run'}]
    for _ in range(rng.randrange(5)):
        k = rng.choice(['read', 'read', 'wait', 'write'])
        if k == 'read': ops.append({'type': 'read', 'id': rng.choice(['stdout', 'stderr']), 'payload': rng.choice(['', 'a', 'ab', 'hello\n', 'ü'])})
        elif k == 'wait': ops.append({'type': 'wait', 'exit_code': rng.choice([0, 1])})
        else: ops.append({'type': 'write', 'id': 'stdin', 'payload': 'x'})
    t = json.dumps({'operations': ops})
    cases.append((t, rng.choice(['', 'a', 'ab', 'abhello\n', 'hello\n', 'ü', 'aab']), rng.choice(['', 'a', 'ab', 'ü']), rng.choice([0, 1]), rng.random() < .1))
# a second run after a wait starts a new process with fresh pipes
for so in ['', 'a', 'aa']:
    for ops in [[{'type': 'run'}, {'type': 'wait'}, {'type': 'run'}, {'type': 'read', 'payload': 'a'}, {'type': 'wait'}], [{'type': 'run'}, {'type': 'read', 'payload': 'a'}, {'type': 'wait'}, {'type': 'run'}, {'type': 'read', 'payload': 'a'}]]:
        cases.append((json.dumps({'operations': ops}), so, '', 0, False))
for t, so, se, ec, to in cases:
    v = ref_verdict(cfg_of(t), so, se, ec, to)
    rows.append('check vd(%s, %s, %s, %d, %s) == %s' % (q(t), q(so), q(se), ec, 'true' if to else 'false', q(v)))
for ex in [False, True]:
    for ef in [False, True]:
        for fl in [False, True]:
            res = tc.Result(is_executed=ex, failures=[tc.Failure.expectation('x')] if fl else [])
            o = tc.Outcome.evaluate(ef, res)
            b = lambda x: 'true' if x else 'false'
            rows.append('check oc(%s, %s, %s) == %s and of(%s, %s, %s) == %s' % (b(ex), b(ef), b(fl), q(o.value), b(ex), b(ef), b(fl), b(o.is_failure)))
# the binding matrix: shape and lookups (its entries come from the WASI
# preview 1 witx, the preview 2 WIT packages and the platform manuals,
# not from an executable reference)
rows.append('check matrix_ok()')
rows.append('check bk("fs", "wasi-p2") == "wasi:filesystem/types" and bk("tty", "bare") == "UART" and bk("exec", "wasi-p1") == "" and bk("nope", "posix") == "" and bk("fs", "nope") == ""')
rows.append('check bc("bare") == "clock,device,tty" and bc("wasi-p1") == "fs,clock,net,tty"')
shards.append(rows)
helpers = open(os.path.join(here, 'target_helpers.rce')).read()
for f in os.listdir(os.path.join(root, 'tests/os')):
    if f.startswith('target_checks'): os.remove(os.path.join(root, 'tests/os', f))
out = []
for rs in shards:
    for i in range(0, len(rs), 300): out.append(rs[i:i + 300])
n = 0
for k, rs in enumerate(out):
    name = 'target_checks.rce' if k == 0 else 'target_checks_%d.rce' % k
    open(os.path.join(root, 'tests/os', name), 'w').write('# Generated by scripts/gen/target_gen.py; do not edit.\n' + helpers + '\n'.join(rs) + '\n')
    n += sum(1 for r in rs if 'check' in r)
print(n, 'checks in', len(out), 'files;', len(real), 'suite configurations')
