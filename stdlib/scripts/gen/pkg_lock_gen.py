# Generates tests/os/pkg_checks_lock*.rce for the lockfile and digest
# parts of lib/os/pkg.rce.
#  - Cargo.lock: lockfiles cargo 1.99 wrote (`cargo generate-lockfile`:
#    registry, git, path and duplicate-version dependencies) and the
#    committed lockfiles of ripgrep, fd and cargo, in vectors/os/lock/;
#    each is parsed with Python's tomllib and compared record by record,
#    and must be written back byte for byte; dependency references are
#    resolved as cargo's format documents;
#  - package-lock.json: lockfiles npm 10 wrote (--package-lock-only),
#    compared with Python's json; for a sample of entries the tarball is
#    downloaded and hashed, and sri_match must accept the real digest and
#    not match a wrong one;
#  - crates.io: the `dl` of https://index.crates.io/config.json, the URL
#    of sample crates fetched for real (HTTP 200) and their SHA-256 equal
#    to the lock checksum;
#  - SRI: the examples of the W3C SRI specification (digests recomputed
#    with hashlib) and random metadata strings against a direct
#    transcription of its parse / strongest / match algorithms.
# Usage: python3 scripts/gen/pkg_lock_gen.py  (needs network)
import base64, hashlib, json, os, random, re, tomllib, urllib.request
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '../..')
rng = random.Random(44)
V = os.path.join(root, 'vectors/os/lock')
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
def b(x): return 'true' if x else 'false'
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'rce-stdlib-gen'}), timeout=60) as r:
        return r.read()
shards = []
# ---- Cargo.lock
for f in sorted(os.listdir(V)):
    if not f.endswith('.Cargo.lock'): continue
    t = open(os.path.join(V, f)).read()
    d = tomllib.loads(t)
    summ = [str(d.get('version', 0))]
    for p in d['package']:
        summ.append('|'.join([p['name'], p['version'], p.get('source', ''), p.get('checksum', ''), ','.join(p.get('dependencies', []))]))
    rows = ['check lk_rt(%s)' % q(t), 'check lk_sum(%s) == %s' % (q(t), q(';'.join(summ)))]
    # dependency references resolve to the package cargo means
    rd, rw = [], []
    for p in d['package']:
        for dep in p.get('dependencies', []):
            parts = dep.split(' ')
            want = [i for i, x in enumerate(d['package']) if x['name'] == parts[0] and (len(parts) < 2 or x['version'] == parts[1]) and (len(parts) < 3 or '(' + x.get('source', '') + ')' == ' '.join(parts[2:]))]
            if len(parts) < 3 and len(want) > 1:
                want = [i for i in want if 'source' not in d['package'][i]]
            assert len(want) == 1, (f, dep)
            rd.append(dep); rw.append(str(want[0]))
    rows.append('check lk_res(%s, %s) == %s' % (q(t), q(';'.join(rd)), q(','.join(rw))))
    shards.append(rows)
# crates.io fetch plan and real downloads
cfg = json.loads(get('https://index.crates.io/config.json'))
dl = cfg['dl']
t = open(os.path.join(V, 'gen1.Cargo.lock')).read(); d = tomllib.loads(t)
src = 'registry+https://github.com/rust-lang/crates.io-index'
plan = []
rows = []
for p in d['package']:
    if p.get('source') != src: continue
    url = '%s/%s/%s/download' % (dl, p['name'], p['version'])
    h = hashlib.sha256(get(url)).hexdigest()
    assert h == p['checksum'], p
    plan.append(url + ' ' + p['checksum'])
    rows.append('check cm(%s, %s)' % (q(p['checksum']), q(h.upper())))
    bad = ('0' if h[0] != '0' else '1') + h[1:]
    rows.append('check not cm(%s, %s)' % (q(p['checksum']), q(bad)))
rows.append('check lk_plan(%s, %s, %s) == %s' % (q(t), q(src), q(dl), q(';'.join(plan))))
# dl templates (Cargo book, registry index config): every marker
def prefix(n):
    return {1: '1', 2: '2', 3: '3/' + n[:1]}.get(len(n), n[:2] + '/' + n[2:4])
tpls = [dl, 'https://example.com/api/v1/crates', 'https://x.test/{crate}/{version}/dl', 'https://x.test/{prefix}/{crate}-{version}.crate', 'https://x.test/{lowerprefix}/{crate}/{sha256-checksum}', 'https://x.test/{crate}{crate}?v={version}']
names = ['a', 'Ab', 'Abc', 'serde', 'X-Y_z', 'tokio', 'IO', 'r2d2']
for tp in tpls:
    for n in names:
        v = rng.choice(['1.0.0', '0.1.2-alpha.1', '2.3.4+meta'])
        c = hashlib.sha256((n + v).encode()).hexdigest()
        if not re.search(r'\{(crate|version|prefix|lowerprefix|sha256-checksum)\}', tp):
            want = tp + '/' + n + '/' + v + '/download'
        else:
            want = tp.replace('{crate}', n).replace('{version}', v).replace('{prefix}', prefix(n)).replace('{lowerprefix}', prefix(n).lower()).replace('{sha256-checksum}', c)
        rows.append('check dlu(%s, %s, %s, %s) == %s' % (q(tp), q(n), q(v), q(c), q(want)))
for c in ['', 'g' * 64, 'A' * 64, '0' * 63, '0' * 65]:
    rows.append('check not cm(%s, %s)' % (q(c), q(c)))
# malformed locks
for t2, why in [('version = x\n', 'Cargo.lock: bad version'), ('[[package]]\nname = "a"\n', 'Cargo.lock: package without name or version'), ('[[package]]\nname = a\nversion = "1"\n', 'Cargo.lock: expected a string'), ('[[package]]\nname = "a"\nversion = "1"\nfoo = "x"\n', 'Cargo.lock: unknown key foo'), ('[[package]]\nname = "a"\nversion = "1"\ndependencies = [\n "b",\n', 'Cargo.lock: unterminated dependencies'), ('[[package]]\nname = "a"\nversion = "1"\ndependencies = ["b"]\n', 'Cargo.lock: dependencies are written one per line'), ('[[package]]\nname\n', 'Cargo.lock: expected key = value'), ('[[package]]\nname = "a\\\\b"\nversion = "1"\n', 'Cargo.lock: escapes are not written by cargo'), ('[[package]]\nname = "a"\nversion = "1"\ndependencies = [\n "b\\\\c",\n]\n', 'Cargo.lock: escapes are not written by cargo')]:
    try: tomllib.loads(t2); ok = True
    except Exception: ok = False
    rows.append('check lk_err(%s) == %s' % (q(t2), q(why)))
shards.append(rows)
# ---- npm
rows = []
tars = []
for f in sorted(os.listdir(V)):
    if not f.endswith('.package-lock.json'): continue
    t = open(os.path.join(V, f)).read(); d = json.loads(t)
    assert d['lockfileVersion'] in (2, 3)
    out = []
    for path, e in d['packages'].items():
        name = e.get('name') or path.split('node_modules/')[-1]
        fl = ('d' if e.get('dev') is True else '') + ('o' if e.get('optional') is True else '') + ('l' if e.get('link') is True else '')
        out.append('|'.join([path, name, e.get('version', ''), e.get('resolved', ''), e.get('integrity', ''), fl]))
        if e.get('integrity') and e.get('resolved'): tars.append((e['resolved'], e['integrity']))
    rows.append('check np(%s) == %s' % (q(t), q(';'.join(out))))
rows.append('check np("{\\"lockfileVersion\\": 1, \\"dependencies\\": {}}") == "!|||||"')
rows.append('check np("{\\"lockfileVersion\\": 3}") == "!|||||"')
rows.append('check np("not json") == "!|||||"')
shards.append(rows)
rows = []
rng.shuffle(tars)
for url, integ in tars[:40]:
    blob = get(url)
    ds = ' '.join('%s:%s' % (a, hashlib.new(a, blob).hexdigest()) for a in ['sha256', 'sha384', 'sha512'])
    assert any(base64.b64encode(hashlib.new(a, blob).digest()).decode() == integ.split('-', 1)[1] for a in ['sha512', 'sha1'] if integ.startswith(a))
    exp = integ.startswith('sha512')  # sha1 integrity is not an SRI algorithm: no known metadata -> true
    rows.append('check sm(%s, %s) == true' % (q(integ), q(ds)))
    wrong = ' '.join('%s:%s' % (a, hashlib.new(a, blob + b'x').hexdigest()) for a in ['sha256', 'sha384', 'sha512'])
    rows.append('check sm(%s, %s) == %s' % (q(integ), q(wrong), b(not exp)))
# ---- SRI spec examples and random metadata
def ref_parse(m):
    out = []
    for item in [x for x in re.split(r'[\t\n\f\r ]+', m) if x]:
        ex = item.split('?')[0]; av = ex.split('-')
        alg = av[0].lower()
        if alg not in ('sha256', 'sha384', 'sha512'): continue
        out.append((alg, av[1] if len(av) > 1 else ''))
    return out
RANK = {'sha256': 0, 'sha384': 1, 'sha512': 2}
def ref_strongest(s):
    if not s: return []
    best = max(RANK[a] for a, _ in s)
    return [x for x in s if RANK[x[0]] == best]
def ref_match(m, blob):
    p = ref_parse(m)
    if not p: return True
    return any(base64.b64encode(hashlib.new(a, blob).digest()).decode() == v for a, v in ref_strongest(p))
hello = b"alert('Hello, world.');"
assert base64.b64encode(hashlib.sha384(hello).digest()).decode() == 'H8BRh8j48O9oYatfu5AZzq6A9RINhZO5H16dQZngK7T62em8MUt1FLm52t+eX6xO'
assert base64.b64encode(hashlib.sha512(hello).digest()).decode() == 'Q2bFTOhEALkN8hOms2FKTDLy7eugP2zFZ1T8LCvX42Fp3WoNr3bjZSAHeOsHrbV1Fu9/A0EzCinRE7Af1ofPrw=='
def dstr(blob): return ' '.join('%s:%s' % (a, hashlib.new(a, blob).hexdigest()) for a in ['sha256', 'sha384', 'sha512'])
def token(blob):
    k = rng.random()
    a = rng.choice(['sha256', 'sha384', 'sha512', 'SHA512', 'Sha256', 'sha1', 'md5', 'sha', ''])
    try: v = base64.b64encode(hashlib.new(a.lower(), blob if rng.random() < .6 else blob + b'!').digest()).decode()
    except Exception: v = 'AAAA'
    t = a + rng.choice(['-', '-', '-', '', '--']) + v
    if rng.random() < .2: t += '?' + rng.choice(['', 'opt', 'a=b', 'x?y'])
    return t
for blob in [hello, b'', b'abc', bytes(range(256))] * 1:
    for _ in range(60):
        m = rng.choice([' ', '  ', '\t', '\n ', '\r\n']).join(token(blob) for _ in range(rng.randrange(0, 5)))
        if rng.random() < .3: m = rng.choice([' ', '\t']) + m + ' '
        rows.append('check sp(%s) == %s' % (q(m), q(' '.join(a + '-' + v for a, v in ref_parse(m)))))
        rows.append('check ss(%s) == %s' % (q(m), q(' '.join(a + '-' + v for a, v in ref_strongest(ref_parse(m))))))
        rows.append('check sm(%s, %s) == %s' % (q(m), q(dstr(blob)), b(ref_match(m, blob))))
ex = 'sha384-H8BRh8j48O9oYatfu5AZzq6A9RINhZO5H16dQZngK7T62em8MUt1FLm52t+eX6xO sha512-Q2bFTOhEALkN8hOms2FKTDLy7eugP2zFZ1T8LCvX42Fp3WoNr3bjZSAHeOsHrbV1Fu9/A0EzCinRE7Af1ofPrw=='
# a digest is only compared with metadata of its own algorithm
for blob in [hello, b'abc']:
    for a, o in [('sha512', 'sha256'), ('sha384', 'sha512'), ('sha256', 'sha384')]:
        m = a + '-' + base64.b64encode(hashlib.new(o, blob).digest()).decode()
        rows.append('check not sm(%s, %s)' % (q(m), q(o + ':' + hashlib.new(o, blob).hexdigest())))
rows += ['check sm(%s, %s)' % (q(ex), q(dstr(hello))), 'check ss(%s) == %s' % (q(ex), q(ex.split(' ')[1])), 'check sm("", "")', 'check sm("md5-xx sha1-yy", "")']
shards.append(rows)
helpers = open(os.path.join(here, 'pkg_lock_helpers.rce')).read() + open(os.path.join(here, 'pkg_helpers.rce')).read()
hdr = '# Generated by scripts/gen/pkg_lock_gen.py; do not edit.\n'
for f in os.listdir(os.path.join(root, 'tests/os')):
    if f.startswith('pkg_checks_lock'): os.remove(os.path.join(root, 'tests/os', f))
n = 0
for k, rs in enumerate(shards):
    with open(os.path.join(root, 'tests/os', 'pkg_checks_lock_%d.rce' % k), 'w') as f:
        f.write(hdr + helpers + '\n'.join(rs) + '\n')
    n += len(rs)
print(n, 'checks in', len(shards), 'files')
