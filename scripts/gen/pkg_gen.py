# Generates tests/os/pkg_checks*.rce for lib/os/pkg.rce.
#  - semver: every assertion of the semver crate's own tests (1.0.28,
#    vendored in vectors/os/semver/), each also confirmed against the
#    crate itself, and random versions, requirements, comparisons and
#    matches answered by the crate (scripts/ref/semver_ref.rs);
#  - test_identifier's 5.6M-character prerelease has a shard of its own.
# Usage: python3 scripts/gen/pkg_gen.py  (needs cargo and network)
import ast, os, random, re, subprocess
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '../..')
rng = random.Random(43)
rows = []
def q(s):
    o = ''
    for c in s:
        if c == '\\': o += '\\\\'
        elif c == '"': o += '\\"'
        elif c == '\n': o += '\\n'
        elif c == '\t': o += '\\t'
        elif c == '\r': o += '\\r'
        elif ord(c) < 32 or ord(c) == 127: o += '\\u{%x}' % ord(c)
        else: o += c
    return '"' + o + '"'
def b(x): return 'true' if x else 'false'

# ---- the crate as reference
SEMREF = '/tmp/semref'
os.makedirs(SEMREF + '/src', exist_ok=True)
open(SEMREF + '/Cargo.toml', 'w').write('[package]\nname = "semref"\nversion = "0.1.0"\nedition = "2021"\n[dependencies]\nsemver = "=1.0.28"\n')
open(SEMREF + '/src/main.rs', 'w').write(open(os.path.join(root, 'scripts/ref/semver_ref.rs')).read())
subprocess.check_call(['cargo', 'build', '--release', '-q'], cwd=SEMREF)
def esc(s): return s.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\0', '\\0')
def ask(reqs):
    inp = ''.join('\t'.join(esc(x) for x in r) + '\n' for r in reqs)
    out = subprocess.run([SEMREF + '/target/release/semref'], input=inp.encode(), capture_output=True, check=True).stdout.decode()
    lines = out.split('\n')[:-1]
    assert len(lines) == len(reqs)
    return [re.sub(r'\\(.)', lambda m: '\n' if m.group(1) == 'n' else m.group(1), l) for l in lines]
FN = {'V': 'vp', 'R': 'rp', 'C': 'cp', 'P': 'pp', 'B': 'bp'}
def lit(s):
    # a Rust string literal (escapes used by the tests: \\ \" \0 \n \t)
    o, i, s = '', 0, s[1:-1]
    while i < len(s):
        if s[i] == '\\':
            o += {'0': '\0', 'n': '\n', 't': '\t'}.get(s[i + 1], s[i + 1]); i += 2
        else:
            o += s[i]; i += 1
    return o

# ---- official assertions
T = {n: open(os.path.join(root, 'vectors/os/semver', n)).read() for n in ['test_version.rs', 'test_version_req.rs', 'test_identifier.rs']}
off = []   # (kind, args, expected answer as semref prints it, rce check)
S = r'("(?:[^"\\]|\\.)*")'
K = {'version': 'V', 'req': 'R', 'comparator': 'C', 'prerelease': 'P'}
for src in T.values():
    for k, a, m in re.findall(r'let err = (\w+)_err\(' + S + r'\);\s*assert_to_string\(\s*err,\s*' + S + r',?\s*\);', src):
        off.append(((K[k], lit(a)), 'err ' + lit(m)))
# requirement blocks
tv = T['test_version_req.rs']
for blk in re.split(r'\n\s*let ref r = ', tv)[1:]:
    blk = re.split(r'\n\s*let |\n}', blk)[0]
    m = re.match(r'req\(' + S + r'\);', blk)
    if not m: continue
    r = lit(m.group(1))
    off.append((('R', r), None))
    for s in re.findall(r'assert_to_string\(r, ' + S + r'\)', blk): off.append((('R', r), 'ok ' + lit(s)))
    for kind, lst in re.findall(r'assert_match_(all|none)\(\s*r,\s*&\[(.*?)\],?\s*\)', blk, re.S):
        for v in re.findall(S, lst): off.append((('M', r, lit(v)), 'true' if kind == 'all' else 'false'))
    for lst in re.findall(r'for s in &\[(.*?)\] \{\s*assert_eq!\(\*r, req\(s\)\);', blk, re.S):
        for s in re.findall(S, lst): off.append((('EQ', r, lit(s)), None))
for a, s in re.findall(r'let parsed = comparator\(' + S + r'\);\s*assert_to_string\(parsed, ' + S + r'\);', tv):
    off.append((('C', lit(a)), 'ok ' + lit(s)))
for fmt in re.findall(r'req\(&format!\(' + S + r', op\)\);', tv):
    for op in ['=', '>', '>=', '<', '<=', '~', '^']:
        off.append((('R', lit(fmt).replace('{}', op)), None))
assert 'assert!(req("^1") == req("^1"));' in tv and 'assert!(req("^1") != req("^2"));' in tv
off += [(('EQ', '^1', '^1'), None), (('NE', '^1', '^2'), None)]
for v in ['0.0.1', '0.1.0', '1.0.0']: off.append((('M', '*', v), 'true'))   # test_any: VersionReq::STAR
tver = T['test_version.rs']
for a, op, c in re.findall(r'assert!\(version\(' + S + r'\) (<=|>=|<|>) version\(' + S + r'\)\);', tver):
    off.append((('O', lit(a), lit(c)), op))
for a, op, c in re.findall(r'assert!\(!\(version\(' + S + r'\) (<=|>=|<|>) version\(' + S + r'\)\)\);', tver):
    off.append((('O', lit(a), lit(c)), '!' + op))
for k, a, c in re.findall(r'assert_(eq|ne)!\(version\(' + S + r'\), version\(' + S + r'\)\);', tver):
    off.append((('O', lit(a), lit(c)), '==' if k == 'eq' else '!='))
for a, s in re.findall(r'assert_to_string\(version\(' + S + r'\), ' + S + r'\);', tver):
    off.append((('V', lit(a)), 'ok ' + lit(s)))
for a, body in re.findall(r'let parsed = version\(' + S + r'\);\s*let expected = Version \{(.*?)\};', tver, re.S):
    f = dict(re.findall(r'(\w+): ([^,]+),', body))
    def idv(x):
        return '' if x.endswith('EMPTY') else lit(re.search(S, x).group(1))
    off.append((('F', lit(a)), '|'.join([f['major'], f['minor'], f['patch'], idv(f['pre']), idv(f['build'])])))
assert 'let expected = Version::new(1, 2, 3);' in tver; off.append((('F', '1.2.3'), '1|2|3||'))
vs = re.findall(S, re.search(r'fn test_spec_order\(\) \{\s*let vs = \[(.*?)\];', tver, re.S).group(1))
for x, y in zip(vs, vs[1:]): off.append((('O', lit(x), lit(y)), '<'))
for e, fill, al, w in re.findall(r'assert_eq!\(' + S + r', format!\("\{:(?:(.?)([<^>]))?(\d+)\}", version\)\);', tver):
    off.append((('PAD', '1.2.3-rc1', int(w), fill or ' ', al or '<'), lit(e)))
ti = T['test_identifier.rs']
assert 'let limit = if cfg!(miri) { 40 } else { 280 };' in ti and 'string.repeat(20000)' in ti
for n in range(280): off.append((('P1', n), None))
off.append((('PBIG',), None))
for k, f, a, c in re.findall(r'assert_(eq|ne)!\((prerelease|build_metadata)\(' + S + r'\), \w+\(' + S + r'\)\);', ti):
    off.append((('IDEQ' if k == 'eq' else 'IDNE', f, lit(a), lit(c)), None))
print(len(off), 'official assertions')

# confirm each against the crate, then emit
for (args, exp) in off:
    k = args[0]
    if k in FN:
        got = ask([args])[0]
        if exp is not None: assert got == exp, (args, exp, got)
        rows.append('check %s(%s) == %s' % (FN[k], q(args[1]), q(got)))
    elif k == 'M':
        got = ask([args])[0]; assert got == exp, (args, exp, got)
        rows.append('check rm(%s, %s) == %s' % (q(args[1]), q(args[2]), got))
    elif k in ('EQ', 'NE'):
        x, y = ask([('R', args[1]), ('R', args[2])])
        assert (x == y) == (k == 'EQ')
        rows.append('check rp(%s) %s rp(%s)' % (q(args[1]), '==' if k == 'EQ' else '!=', q(args[2])))
    elif k == 'O':
        got = int(ask([args])[0])
        ok = {'<': got < 0, '<=': got <= 0, '>': got > 0, '>=': got >= 0, '!<': not got < 0, '!>': not got > 0, '!>=': not got >= 0, '!<=': not got <= 0, '==': got == 0, '!=': got != 0}[exp]
        assert ok, (args, exp, got)
        rows.append('check vc(%s, %s) == %d' % (q(args[1]), q(args[2]), got))
    elif k == 'F':
        rows.append('check vf(%s) == %s' % (q(args[1]), q(exp)))
    elif k == 'PAD':
        rows.append('check vpad(%s, %d, %s, %s) == %s' % (q(args[1]), args[2], q(args[3]), q(args[4]), q(exp)))
    elif k == 'P1':
        got = ask([('P', '1' * args[1])])[0]; assert got == 'ok ' + '1' * args[1]
        rows.append('check pp(ones(%d)) == "ok " + ones(%d)' % (args[1], args[1]))
    elif k == 'PBIG':
        big_row = 'check pp(rep(ones(280), 20000)) == "ok " + rep(ones(280), 20000)'
    elif k in ('IDEQ', 'IDNE'):
        f = 'pp' if args[1] == 'prerelease' else 'bp'
        rows.append('check %s(%s) %s %s(%s)' % (f, q(args[2]), '==' if k == 'IDEQ' else '!=', f, q(args[3])))
n_official = len(rows)

# ---- random, answered by the crate
def num():
    return rng.choice(['0', '1', '2', '9', '10', '01', '00', str(rng.randrange(1000)), '18446744073709551615', '18446744073709551616', '9223372036854775808', '99999999999999999999'])
def ident():
    k = rng.random()
    if k < .3: return rng.choice(['alpha', 'beta', 'rc', 'pre', 'a', 'b', 'x', 'X', '-', 'a-b', '1a', '0a'])
    if k < .6: return rng.choice(['0', '1', '2', '10', '11', '01', '007', '99'])
    if k < .7: return ''
    return ''.join(rng.choice('abcAZ019-') for _ in range(rng.randrange(1, 5)))
def dotted(): return '.'.join(ident() for _ in range(rng.choice([1, 1, 2, 3])))
def junk(): return rng.choice([' ', '', '', '', '.', '-', '+', ',', '\0', '\t', 'ÿ', '中', '|', '*', 'x', '_', "'", '\\', '"', '\x7f', '\x01', '\n'])
def version():
    s = '.'.join(num() for _ in range(rng.choice([3, 3, 3, 3, 2, 1, 4])))
    if rng.random() < .4: s += '-' + dotted()
    if rng.random() < .3: s += '+' + dotted()
    if rng.random() < .15:
        i = rng.randrange(len(s) + 1); s = s[:i] + junk() + s[i:]
    return s
def good_version():
    s = '%d.%d.%d' % (rng.choice([0, 1, 2]), rng.choice([0, 1, 2, 3, 10]), rng.choice([0, 1, 2, 3, 9]))
    if rng.random() < .4: s += '-' + rng.choice(['alpha', 'alpha.1', 'alpha.beta', 'beta', 'beta.2', 'beta.11', 'rc.1', '1', '2', 'a', 'pre', '0'])
    if rng.random() < .2: s += '+' + rng.choice(['1', '01', 'build.5', '0', 'a'])
    return s
def comparator():
    op = rng.choice(['', '', '=', '>', '>=', '<', '<=', '~', '^', '> ', '>= ', '~ ', '=='])
    parts = [rng.choice(['0', '1', '2', '3', '10'])]
    for _ in range(rng.choice([0, 1, 2, 2, 2])):
        parts.append(rng.choice(['0', '1', '2', '3', '*', 'x', 'X', '01']))
    s = op + '.'.join(parts)
    if len(parts) == 3 and rng.random() < .35: s += '-' + rng.choice(['alpha', 'alpha.1', 'beta', 'rc.1', '1', '0', 'pre'])
    if len(parts) == 3 and rng.random() < .15: s += '+' + rng.choice(['meta', '1', '01'])
    if rng.random() < .1:
        i = rng.randrange(len(s) + 1); s = s[:i] + junk() + s[i:]
    return s
def req():
    if rng.random() < .05: return rng.choice(['*', 'x', 'X', ' *', '* ', '*, 1', 'x.1', '', ' '])
    return rng.choice([',', ', ', ' ,', ',  ']).join(comparator() for _ in range(rng.choice([1, 1, 2, 2, 3, 33])))
reqs = []
for _ in range(1500): reqs.append(('V', version()))
for _ in range(400): reqs.append(('P', dotted() + rng.choice(['', '', junk()])))
for _ in range(400): reqs.append(('B', dotted() + rng.choice(['', '', junk()])))
for _ in range(1200): reqs.append(('C', rng.choice(['', ' ']) + comparator() + rng.choice(['', '', ' ', junk()])))
for _ in range(1500): reqs.append(('R', req()))
for r, a in zip(reqs, ask(reqs)):
    rows.append('check %s(%s) == %s' % (FN[r[0]], q(r[1]), q(a)))
ords = [('O', good_version(), good_version()) for _ in range(1200)]
for r, a in zip(ords, ask(ords)): rows.append('check vc(%s, %s) == %s' % (q(r[1]), q(r[2]), a))
mreqs = []
while len(mreqs) < 3000:
    r = req()
    if ask([('R', r)])[0].startswith('ok'):
        for _ in range(3): mreqs.append(('M', r, good_version()))
for _ in range(300):   # ^0.0.z, ~ and = near zero: every version of the 0.0/0.1 neighbourhood
    z = '0.%d.%d' % (rng.choice([0, 1]), rng.randrange(4))
    r = rng.choice(['^', '~', '', '=', '>=', '<=']) + z + rng.choice(['', '', '-alpha', '-1'])
    mreqs.append(('M', r, '0.%d.%d%s' % (rng.choice([0, 1]), rng.randrange(4), rng.choice(['', '', '-alpha', '-beta', '-0']))))
for r, a in zip(mreqs, ask(mreqs)): rows.append('check rm(%s, %s) == %s' % (q(r[1]), q(r[2]), a))
kreqs = []
while len(kreqs) < 1000:
    c = comparator()
    if ask([('C', c)])[0].startswith('ok'):
        kreqs.append(('K', c, good_version()))
for r, a in zip(kreqs, ask(kreqs)): rows.append('check km(%s, %s) == %s' % (q(r[1]), q(r[2]), a))
print(n_official, 'official checks,', len(rows) - n_official, 'random semver checks')
semver_rows = rows

# ---- write
helpers = open(os.path.join(here, 'pkg_helpers.rce')).read()
hdr = '# Generated by scripts/gen/pkg_gen.py (semver 1.0.28, cargo 1.99); do not edit.\n'
for f in os.listdir(os.path.join(root, 'tests/os')):
    if f.startswith('pkg_checks'): os.remove(os.path.join(root, 'tests/os', f))
shards = []
N = 1200
for k in range(0, len(rows), N): shards.append(rows[k:k + N])
shards.append([big_row])
for k, rs in enumerate(shards):
    name = 'pkg_checks.rce' if k == 0 else 'pkg_checks_%d.rce' % k
    with open(os.path.join(root, 'tests/os', name), 'w') as f:
        f.write(hdr + helpers + '\n'.join(rs) + '\n')
print(len(rows) + 1, 'checks in', len(shards), 'files')
