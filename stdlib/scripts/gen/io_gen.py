# Derived vectors for lib/io/io.rce: a Python model of each part, run on
# fixed and seeded-random inputs, printed as rce checks.
# Usage: python3 scripts/gen/io_gen.py > generated section of tests/io/io_checks.rce
import random
R = random.Random(18)

def q(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"'
def ql(xs): return '[' + ', '.join(q(x) for x in xs) + ']'

# ---- line reader model
def strip_cr(s): return s[:-1] if s.endswith('\r') else s
def lines(chunks, fin):
    rest, out = '', []
    for c in chunks:
        parts = (rest + c).split('\n')
        out += [strip_cr(p) for p in parts[:-1]]; rest = parts[-1]
    if fin and rest != '': out.append(strip_cr(rest))
    return out, (rest if not fin else '')

# ---- writer model; acks: n >= 0 delivered n bytes, -1 failed
def writer(data, mx, acks):
    b = data.encode(); off = 0; chunks = []; sent = 0; ai = 0
    mx = max(mx, 1)
    while off < len(b) and ai < len(acks):
        o = off; e = off
        cs = data.encode()
        # take whole characters while within mx (first char always)
        i = off
        while i < len(b):
            l = 1
            while i + l < len(b) and (b[i + l] & 0xC0) == 0x80: l += 1
            if e > o and e + l - o > mx: break
            e = i + l; i += l
        ch = b[o:e].decode(); chunks.append(ch)
        a = acks[ai]; ai += 1
        if a >= 0: off += a; sent += a
    return chunks, off, sent, off >= len(b)

def bounds(s):
    b = s.encode(); return [i for i in range(len(b) + 1) if i == len(b) or (b[i] & 0xC0) != 0x80]

def rand_text(n):
    al = ['a', 'b', ' ', '\n', '\r', 'é', '€', '😀', 'z', '\n']
    return ''.join(R.choice(al) for _ in range(n))

def path_ok(p, lst):
    if p == '': return lst
    if p.startswith('/'): return False
    return all(s != '..' for s in p.split('/'))

print('# status and result shape (HC-3)')
for s in ['delivered', 'failed', 'timed_out', 'unreachable', 'Delivered', 'ok', '', 'timedout']:
    print('check sk(%s) == %s' % (q(s), 'true' if s in ('delivered', 'failed', 'timed_out', 'unreachable') else 'false'))
cases = [('delivered', 'x', ''), ('delivered', None, ''), ('delivered', 'x', 'e'), ('delivered', '', ''), ('failed', None, 'no such file'), ('failed', 'x', 'e'), ('failed', None, ''), ('timed_out', None, ''), ('unreachable', None, 'no adapter'), ('unreachable', 'x', ''), ('bogus', None, ''), ('bogus', 'x', '')]
for s, d, e in cases:
    ok = s in ('delivered', 'failed', 'timed_out', 'unreachable') and ((d is not None and e == '') if s == 'delivered' else d is None)
    dd = 'none' if d is None else 'some(%s)' % q(d)
    print('check rok(%s, %s, %s) == %s' % (q(s), dd, q(e), 'true' if ok else 'false'))
    print('check rget(%s, %s, %s) == %s' % (q(s), dd, q(e), dd if ok and s == 'delivered' else 'none'))

print('# request table')
def pend(ops):
    nxt = 0; table = []; ids = []; tags = []; settled = stray = 0
    for op in ops:
        if op[0] == 'i':
            nxt += 1; table.append((nxt, op[1], op[2])); ids.append(nxt)
        else:
            m = [t for t in table if t[0] == op[2]]
            if not m or m[0][1] != op[1]: stray += 1; tags.append('-')
            else: table.remove(m[0]); settled += 1; tags.append(m[0][2])
    return ids, tags, len(table), settled, stray
scen = [
    [('i', 'FileRead', 'a'), ('s', 'FileRead', 1)],
    [('i', 'FileRead', 'a'), ('s', 'FileRead', 1), ('s', 'FileRead', 1)],
    [('i', 'FileRead', 'a'), ('s', 'FileWrite', 1), ('s', 'FileRead', 1)],
    [('s', 'NetSend', 0), ('s', 'NetSend', 7)],
    [('i', 'A', 'x'), ('i', 'B', 'y'), ('i', 'A', 'z'), ('s', 'A', 3), ('s', 'B', 2), ('s', 'A', 1), ('s', 'A', 3)],
]
kinds = ['FileRead', 'FileWrite', 'NetSend', 'PtyRead']
for _ in range(25):
    ops = []; n = 0
    for _ in range(R.randint(1, 12)):
        if R.random() < 0.5: n += 1; ops.append(('i', R.choice(kinds), 't%d' % n))
        else: ops.append(('s', R.choice(kinds), R.randint(0, n + 1)))
    scen.append(ops)
for ops in scen:
    ids, tags, out, st, sy = pend(ops)
    ol = ', '.join('Op(add: %s, what: %s, tag: %s, req: %d)' % ('true' if o[0] == 'i' else 'false', q(o[1]), q(o[2] if o[0] == 'i' else ''), 0 if o[0] == 'i' else o[2]) for o in ops)
    print('check pend([%s]) == PendOut(ids: [%s], tags: %s, out: %d, settled: %d, stray: %d)' % (ol, ', '.join(map(str, ids)), ql(tags), out, st, sy))

print('# line reader')
fixed = [([''], True), (['a'], False), (['a'], True), (['a\n'], True), (['a\r\n', 'b'], True), (['a\r', '\nb\n'], True), (['\n\n'], True), (['x\ry\n'], True), (['é€', '\n😀'], True), (['a\r'], True), (['a\r\r\n'], True)]
for _ in range(40):
    t = rand_text(R.randint(0, 30)); cuts = sorted(R.sample(range(len(t) + 1), min(len(t) + 1, R.randint(0, 4))))
    ch = []; p = 0
    for c in cuts + [len(t)]: ch.append(t[p:c]); p = c
    fixed.append((ch, R.random() < 0.8))
for ch, fin in fixed:
    out, rest = lines(ch, fin)
    print('check read(%s, %s) == RdOut(lines: %s, rest: %s)' % (ql(ch), 'true' if fin else 'false', ql(out), q(rest)))
print('check after_eof()')

print('# writer (partial writes resent; chunks cut between characters)')
wf = [('hello', 2, [2, 2, 1]), ('hello', 10, [5]), ('héllo', 2, [1, 2, 2]), ('€€', 1, [3, 3]), ('😀a', 2, [4, 1]), ('abc', 3, [-1, 3]), ('abcdef', 4, [2, 4]), ('', 4, []), ('ab', 0, [1, 1]), ('abcd', 4, [0, 4])]
for _ in range(30):
    t = rand_text(R.randint(1, 20)); mx = R.randint(1, 8); acks = []
    _, off, _, done = writer(t, mx, acks)
    while not done and len(acks) < 40:
        ch, off, _, done = writer(t, mx, acks)
        if done: break
        last = ch[-1] if len(ch) > len(acks) else None
        if last is None:
            ch, _, _, _ = writer(t, mx, acks + [0]); last = ch[-1]
        if R.random() < 0.15: acks.append(-1); continue
        b = off
        opts = [x - off for x in bounds(t) if off < x <= off + len(last.encode())]
        acks.append(R.choice(opts) if R.random() < 0.6 else opts[-1])
        _, off, _, done = writer(t, mx, acks)
    wf.append((t, mx, acks))
for t, mx, acks in wf:
    ch, off, sent, done = writer(t, mx, acks)
    print('check wr(%s, %d, [%s]) == WrOut(chunks: %s, sent: %d, done: %s)' % (q(t), mx, ', '.join(map(str, acks)), ql(ch), sent, 'true' if done else 'false'))
print('check bad_acks()')

print('# file paths (the fs adapter rule)')
ps = ['a.txt', 'd/a.txt', './a', '.', '..', '../a', 'a/..', 'a/../b', '/etc/passwd', '/', '', 'a//b', 'a/', '...', '..a', 'a..', 'é/€', 'a/./b', '.hidden', ' ']
for p in ps:
    for lst in (False, True):
        print('check pok(%s, %s) == %s' % (q(p), 'true' if lst else 'false', 'true' if path_ok(p, lst) else 'false'))
