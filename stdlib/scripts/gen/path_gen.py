# Checks for lib/os/path.rce. Inputs are taken from CPython v3.13.5
# Lib/test/test_posixpath.py and Lib/test/test_ntpath.py (in vectors/os/):
# every argument tuple of a call to a covered function (assertEqual calls,
# ntpath tester() strings, NORMPATH_CASES, splitextTest), and every string
# literal of those test files used as a single path. Expected values come
# from CPython 3.13 posixpath and ntpath (the implementations those tests
# check; ntpath runs its pure-Python fallbacks off Windows), with
# os.getcwd patched to the tests' own "/home/user/bar" and, for ntpath,
# "C:\\Users\\u". Where a test states an expected value, the generator
# asserts that CPython gives it (for ntpath up to "/" versus "\", as
# test_ntpath's tester() compares). Plus seeded random pairs for join,
# relpath and commonpath, a few derived join inputs, and derived WASI
# cases.
# Usage: python3 scripts/gen/path_gen.py > tests/os/path_checks.rce
import ast, os, random, sys, posixpath, ntpath
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
R = random.Random(34)
FN = {'split', 'splitext', 'splitdrive', 'splitroot', 'basename', 'dirname', 'isabs', 'join', 'normpath', 'commonpath', 'relpath', 'isreserved'}
def q(s):
    o = ''
    for c in s:
        o += {'"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t'}.get(c) or (c if 32 <= ord(c) != 127 else '\\u{%x}' % ord(c))
    return '"' + o + '"'
calls = set(); corpus = set(); stated = []
def lit(n):
    try: return ast.literal_eval(n)
    except Exception: return None
def take(mod, call, want=None):
    if not (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)): return
    if call.func.value.id != mod or call.func.attr not in FN or call.keywords: return
    args = []
    for a in call.args:
        v = lit(a)
        if v is None or isinstance(a, ast.Starred): return
        args.append(v)
    flat = [y for x in args for y in (x if isinstance(x, (list, tuple)) else [x])]
    if not flat or not all(isinstance(x, str) for x in flat): return
    calls.add((mod, call.func.attr, repr(args)))
    if want is not None: stated.append((mod, call.func.attr, args, want))
for fname, mod in (('test_ntpath.py', 'ntpath'), ('test_posixpath.py', 'posixpath')):
    src = open(os.path.join(ROOT, 'vectors/os', fname)).read()
    t = ast.parse(src)
    for n in ast.walk(t):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 80 and '\n' not in n.value:
            corpus.add(n.value)
        if isinstance(n, ast.Call):
            want = lit(n.args[1]) if len(n.args) > 1 else None
            if isinstance(n.func, ast.Attribute) and n.func.attr in ('assertEqual', 'assertIs') and n.args and isinstance(n.args[0], ast.Call):
                take(mod, n.args[0], want)
            else:
                take(mod, n)
            if isinstance(n.func, ast.Name) and n.func.id == 'tester' and isinstance(n.args[0], ast.Constant):
                e = ast.parse(n.args[0].value.replace('\\', '\\\\'), mode='eval').body
                if isinstance(e, ast.Call): take('ntpath', e, want)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'NORMPATH_CASES':
            for a, b in lit(n.value):
                calls.add(('posixpath', 'normpath', repr([a]))); stated.append(('posixpath', 'normpath', [a], b))
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'splitextTest':
            a = lit(n.args[0])
            for p in (a, '/' + a, 'abc/' + a, 'abc.def/' + a, '/abc.def/' + a, a + '/'):
                calls.add(('posixpath', 'splitext', repr([p])))
os.getcwd = lambda: '/home/user/bar'
NTCWD = 'C:\\Users\\u'
def run(mod, fn, args):
    m = posixpath if mod == 'posixpath' else ntpath
    old = os.getcwd
    if mod == 'ntpath': os.getcwd = lambda: NTCWD
    try: return ('ok', getattr(m, fn)(*args))
    except ValueError as e: return ('err', None)
    finally: os.getcwd = old
bad = 0
for mod, fn, args, want in stated:
    st, got = run(mod, fn, args)
    norm = lambda x: x.replace('/', '\\') if isinstance(x, str) else tuple(norm(y) for y in x) if isinstance(x, (tuple, list)) else x
    if st == 'ok' and got != want and not (isinstance(want, tuple) and tuple(got) == want) and not (mod == 'ntpath' and norm(got) == norm(want)): bad += 1; print('stated mismatch', mod, fn, args, want, got, file=sys.stderr)
assert bad == 0
corpus = sorted(c for c in corpus if not any(0xD800 <= ord(x) <= 0xDFFF for x in c))
cl = [c for c in corpus if len(c) < 30]
for _ in range(150):
    for mod in ('posixpath', 'ntpath'):
        calls.add((mod, 'join', repr([R.choice(cl), R.choice(cl)] + ([R.choice(cl)] if R.random() < .3 else []))))
        calls.add((mod, 'relpath', repr([R.choice(cl) or 'a', R.choice(cl) or 'b'])))
        calls.add((mod, 'commonpath', repr([[R.choice(cl), R.choice(cl)]])))
# derived inputs (outputs still from CPython): empty and drive-only parts
for mod in ('posixpath', 'ntpath'):
    for t in (['', 'a'], ['', ''], ['', '/a'], ['', 'a', 'b'], ['a', '', 'b'], ['c:', 'a'], ['c:', '/a'], ['', 'c:a']):
        calls.add((mod, 'join', repr(t)))
P = {'posixpath': 'posix', 'ntpath': 'win'}
def pair(t): return 'Pair(head: %s, tail: %s)' % (q(t[0]), q(t[1]))
def root(t): return 'Root(drive: %s, root: %s, tail: %s)' % tuple(q(x) for x in t)
def lst(xs): return '[' + ', '.join(q(x) for x in xs) + ']'
lines = []
def emit(mod, fn, args):
    st, got = run(mod, fn, args)
    p = P[mod]
    if fn == 'relpath':
        args = list(args) + (['.'] if len(args) == 1 else [])
        a = '%s_relpath(&%s, &%s, &%s)' % (p, 'T(%s)' % q(args[0]), 'T(%s)' % q(args[1]), 'T(%s)' % q(os.getcwd() if mod == 'posixpath' else NTCWD))
        lines.append('check rel_%s(%s, %s) == %s' % (p, q(args[0]), q(args[1]), q(got) if st == 'ok' else '"ERR"'))
        return
    if fn == 'commonpath':
        lines.append('check common_%s(%s) == %s' % (p, lst(args[0]), q(got) if st == 'ok' else '"ERR"'))
        return
    if fn == 'join':
        lines.append('check jn_%s(%s) == %s' % (p, lst(args), q(got))); return
    if st != 'ok': return
    a = q(args[0]); f = '%s_%s' % (p, fn)
    if fn in ('split', 'splitext', 'splitdrive'): lines.append('check %s(&T(%s)) == %s' % (f, a, pair(got)) if False else 'check %s_s(%s) == %s' % (f, a, pair(got)))
    elif fn == 'splitroot': lines.append('check %s_s(%s) == %s' % (f, a, root(got)))
    elif fn in ('isabs', 'isreserved'): lines.append('check %s%s_s(%s)' % ('' if got else 'not ', f, a))
    else: lines.append('check %s_s(%s) == %s' % (f, a, q(got)))
for mod, fn, args in sorted(calls): emit(mod, fn, ast.literal_eval(args))
ncall = len(lines)
for c in corpus:
    for mod in ('posixpath', 'ntpath'):
        for fn in ('split', 'splitext', 'splitroot', 'basename', 'dirname', 'isabs', 'normpath') + (('splitdrive', 'isreserved') if mod == 'ntpath' else ()):
            if fn == 'isreserved' and c == '': continue
            emit(mod, fn, [c])
seen = set(); out = []
for l in lines:
    if l not in seen: seen.add(l); out.append(l)
# derived WASI cases
W = [('a/b/c', ['a', 'b', 'c']), ('./a//b/', ['a', 'b']), ('a/../b', ['b']), ('', []), ('.', []), ('a/..', []),
     ('..', None), ('a/../..', None), ('/etc/passwd', None), ('a/./../b/../../c', None), ('x/y/../../z', ['z'])]
for p, want in W:
    out.append('check wasi_s(%s) == %s' % (q(p), lst(want) if want is not None else '["ERR"]'))
PO = [(['/', '/tmp', '/tmp/x'], '/tmp/x/y', (2, 'y')), (['/', '/tmp'], '/tmp', (1, '.')), (['/data'], '/etc', None),
      (['/a/b', '/a'], '/a/b/../c', (1, 'c')), (['/a', '/a'], '/a/z', (1, 'z')), (['/'], '//x/./y', (0, 'x/y')), (['rel'], '/rel/x', None), (['/'], 'x', None)]
for pre, p, want in PO:
    out.append('check pre_s(%s, %s) == %s' % (lst(pre), q(p), 'Preopen(index: %d, rel: %s)' % (want[0], q(want[1])) if want else 'Preopen(index: -1, rel: "ERR")'))
H = '''# Generated by scripts/gen/path_gen.py (CPython 3.13.5 test inputs, CPython
# posixpath and ntpath outputs); do not edit.
fn T(t: text) -> text = t
'''
for p in ('posix', 'win'):
    for f in ('split', 'splitext', 'basename', 'dirname', 'isabs', 'normpath', 'splitroot') + (('splitdrive', 'isreserved') if p == 'win' else ()):
        rt = {'split': 'Pair', 'splitext': 'Pair', 'splitdrive': 'Pair', 'splitroot': 'Root', 'isabs': 'bool', 'isreserved': 'bool'}.get(f, 'text')
        H += 'fn %s_%s_s(t: text) -> %s = %s_%s(&t)\n' % (p, f, rt, p, f)
    H += 'fn jn_%s(ps: List[text]) -> text = %s_join(&ps)\n' % (p, p)
    H += 'fn common_%s(ps: List[text]) -> text = %s_commonpath(&ps) catch "ERR"\n' % (p, p)
    H += 'fn rel_%s(a: text, b: text) -> text\n    let c = %s\n    return %s_relpath(&a, &b, &c) catch "ERR"\n' % (p, q('/home/user/bar' if p == 'posix' else NTCWD), p)
H += '''fn wasi_s(t: text) -> List[text] = wasi_resolve(&t) catch ["ERR"]
fn pre_s(ps: List[text], t: text) -> Preopen = wasi_preopen(&ps, &t) catch Preopen(index: -1, rel: "ERR")
'''
sys.stdout.write(H + '\n'.join(out) + '\n')
print('%d stated expectations agree; %d call checks, %d total checks (%d corpus strings)' % (len(stated), ncall, len(out), len(corpus)), file=sys.stderr)
