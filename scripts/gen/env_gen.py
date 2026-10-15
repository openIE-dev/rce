# Checks for lib/os/env.rce.
#   official: every chk() of rust-lang/rust 8a9dd7abdf01
#     library/std/src/sys/args/windows/tests.rs (win_parse with exe
#     "TEST.EXE"), every assert_eq! of test_make_command_line in
#     library/std/src/sys/process/windows/tests.rs (win_command_line), and
#     golang/go 0b67e71bfae5 src/os/env_test.go expandTests with its
#     testGetenv table (expand). Copies are in vectors/os/.
#   derived: seeded round trips win_parse(win_command_line(p, args)) ==
#     [p] + args over awkward arguments, and hand-written entry, lookup,
#     update and snapshot cases.
# Usage: python3 scripts/gen/env_gen.py > tests/os/env_checks.rce
import os, random, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
V = lambda f: open(os.path.join(ROOT, 'vectors/os', f)).read()
def q(s):
    o = ''
    for c in s:
        o += {'"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t'}.get(c) or (c if 32 <= ord(c) != 127 else '\\u{%x}' % ord(c))
    return '"' + o + '"'
TOK = re.compile(r'r(#*)"(.*?)"\1|"((?:[^"\\]|\\.)*)"', re.S)
def rs_str(m):
    if m.group(2) is not None and m.group(0).startswith('r'): return m.group(2)
    s = m.group(3)
    return re.sub(r'\\u\{([0-9a-fA-F]+)\}|\\(.)', lambda x: chr(int(x.group(1), 16)) if x.group(1) else {'n': '\n', 't': '\t', '0': '\0', '\\': '\\', '"': '"'}[x.group(2)], s)
def strs(t): return [rs_str(m) for m in TOK.finditer(t)]
L = lambda xs: '[' + ', '.join(q(x) for x in xs) + ']'
out = []
src = V('rust_args_windows_tests.rs'); src = src[src.index('fn empty'):]
n = 0
for m in re.finditer(r'chk\(\s*(.*?),\s*&\[(.*?)\]\s*,?\s*\);', src, re.S):
    a = strs(m.group(1)); b = strs(m.group(2))
    assert len(a) == 1, m.group(0)
    out.append('check wp(%s) == %s' % (q(a[0]), L(b))); n += 1
src = V('rust_process_windows_tests.rs')
body = src[src.index('fn test_make_command_line'):src.index('fn windows_env_unicode_case')]
k = 0
for m in re.finditer(r'assert_eq!\(\s*test_wrapper\((.*?)\),\s*(.*?)\s*\);', body, re.S):
    inner = m.group(1)
    prog = strs(inner.split(',', 1)[0])[0]
    args = strs(inner[inner.index('&['):inner.index(']') + 1])
    force = inner.strip().endswith('true')
    want = strs(m.group(2))[0]
    out.append('check wcl(%s, %s, %s) == %s' % (q(prog), L(args), 'true' if force else 'false', q(want))); k += 1
go = V('go_os_env_test.go')
tbl = dict(re.findall(r'case "(.*?)":\s*return "(.*?)"', go))
e = 0
blk = go[go.index('var expandTests'):go.index('func TestExpand')]
for a, b in re.findall(r'\{"(.*?)", "(.*?)"\}', blk):
    out.append('check ex_go(%s) == %s' % (q(a), q(b))); e += 1
R = random.Random(35)
pieces = ['a', 'b c', '\\', '\\\\', '"', '\\"', ' ', '\t', '', 'x\\', 'é', '😅', '""', 'C:\\Program Files\\', "'"]
rt = 0
for _ in range(120):
    args = [''.join(R.choice(pieces) for _ in range(R.randint(0, 4))) for _ in range(R.randint(0, 4))]
    prog = R.choice(['prog', 'C:\\Program Files\\x.exe', 'a b', 'π'])
    out.append('check roundtrip(%s, %s, %s)' % (q(prog), L(args), 'true' if R.random() < .3 else 'false')); rt += 1
D = r'''check pe(("A=b=c"), false) == Var(key: "A", value: "b=c")
check pe(("=C:=C:\\x"), true) == Var(key: "=C:", value: "C:\\x")
check pe(("=C:=C:\\x"), false) == Var(key: "", value: "C:=C:\\x")
check pe(("NOEQ"), false) == Var(key: "NOEQ", value: "")
check pe((""), true) == Var(key: "", value: "")
check ge(("PATH"), false) == "/bin"
check ge(("path"), false) == "/x"
check ge(("path"), true) == "/bin"
check lk(("NOPE"), false) == none
check lk(("E"), false) == some("")
check vk(("A"), false) and not vk((""), false) and not vk(("A=B"), false) and vk(("=C:"), true) and not vk(("=C:"), false) and not vk(("A\u{0}"), true)
check (se(("PATH"), ("/y"), false) catch []) == [Var(key: "PATH", value: "/y"), Var(key: "path", value: "/x"), Var(key: "E", value: "")]
check (se(("Path"), ("/y"), true) catch []) == [Var(key: "PATH", value: "/y"), Var(key: "E", value: "")]
check (se(("N"), ("1"), false) catch []) == [Var(key: "PATH", value: "/bin"), Var(key: "path", value: "/x"), Var(key: "PATH", value: "/sbin"), Var(key: "E", value: ""), Var(key: "N", value: "1")]
check (se(("A=B"), ("1"), false) catch []) == []
check (se(("A"), ("x\u{0}"), false) catch []) == []
check un(("PATH"), false) == [Var(key: "path", value: "/x"), Var(key: "E", value: "")]
check un(("PATH"), true) == [Var(key: "E", value: "")]
check sn(RAW, false, false) == Snapshot(argv: ["prog", "-v"], vars: [Var(key: "HOME", value: "/h"), Var(key: "LANG", value: "C")])
check sn(RAW, true, false) == Snapshot(argv: ["prog", "-v"], vars: [Var(key: "HOME", value: "/h"), Var(key: "LANG", value: "C"), Var(key: "SECRET", value: "s"), Var(key: "Z", value: "1"), Var(key: "lang", value: "c")])
check sn(RAW, false, true) == Snapshot(argv: ["prog", "-v"], vars: [Var(key: "HOME", value: "/h"), Var(key: "LANG", value: "C")])
check sn(RAW2, false, false) == sn(RAW, false, false)
check ex(("$PATH:${E}x$NOPE."), false) == "/bin:x."
check ex(("${path}"), true) == "/bin"
check (wq(("a\u{0}"), false) catch "ERR") == "ERR"
check (wc0(("a\"b"), NOARGS, false) catch "ERR") == "ERR"
check wp("\"a b\"\tc\t\t\"d") == ["a b", "c", "d"]
'''
H = '''# Generated by scripts/gen/env_gen.py; do not edit.
fn T(t: text) -> text = t
fn wp(t: text) -> List[text]
    let e = "TEST.EXE"
    return win_parse(&t, &e)
fn wcl(p: text, a: List[text], f: bool) -> text = win_command_line(&p, &a, f) catch "ERR"
fn roundtrip(p: text, a: List[text], f: bool) -> bool
    let l = win_command_line(&p, &a, f) catch "ERR"
    var want: List[text] = [copy(p)]
    for x in a
        want.push(copy(x))
    let e = "X"
    return win_parse(&l, &e) == want
fn pe(t: text, w: bool) -> Var = parse_entry(&t, w)
fn ge(k: text, w: bool) -> text = getenv(&ENV, &k, w)
fn lk(k: text, w: bool) -> text? = lookup(&ENV, &k, w)
fn vk(k: text, w: bool) -> bool = valid_key(&k, w)
fn se(k: text, v: text, w: bool) -> List[Var] ! BadEnv = setenv(&ENV, &k, &v, w)
fn un(k: text, w: bool) -> List[Var] = unsetenv(&ENV, &k, w)
fn ex(t: text, w: bool) -> text = expand(&t, &ENV, w)
fn ex_go(t: text) -> text = expand(&t, &GO, false)
fn wq(a: text, f: bool) -> text ! BadEnv = win_quote(&a, f)
fn wc0(p: text, a: List[text], f: bool) -> text ! BadEnv = win_command_line(&p, &a, f)
fn sn(raw: List[text], all: bool, w: bool) -> Snapshot = snapshot(&ARGS, &raw, &GR, all, w)
const GO: List[Var] = [%s]
const ENV: List[Var] = [Var(key: "PATH", value: "/bin"), Var(key: "path", value: "/x"), Var(key: "PATH", value: "/sbin"), Var(key: "E", value: "")]
const ARGS: List[text] = ["prog", "-v"]
const RAW: List[text] = ["Z=1", "HOME=/h", "LANG=C", "SECRET=s", "HOME=/other", "lang=c"]
const RAW2: List[text] = ["LANG=C", "HOME=/h", "SECRET=x"]
const GR: List[text] = ["HOME", "LANG"]
const NOARGS: List[text] = []
''' % ', '.join('Var(key: %s, value: %s)' % (q(a), q(b)) for a, b in tbl.items())
sys.stdout.write(H + '\n'.join(out) + '\n' + D)
print('rust parse %d, rust command line %d, go expand %d, round trips %d, derived %d' % (n, k, e, rt, D.count('\ncheck') + 1), file=sys.stderr)
