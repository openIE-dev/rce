#!/usr/bin/env python3
"""Generate tests/ui/view_checks*.rce for lib/ui/view.rce.

  python3 scripts/gen/view_gen.py --refresh   # re-run Vue 3.5.44 (needs
        node and `npm install` in scripts/ref/view_ref), rewrite the TSVs
  python3 scripts/gen/view_gen.py             # tests from the TSVs

Sources (vectors/SOURCE.md, "ui.view"):
  vectors/ui/vue_official.tsv  the cases of Vue's rendererChildren.spec.ts
      with Vue's operation log for each and ours
  vectors/ui/vue_live.tsv      seeded random child lists, same columns
  breadth-first search         least single-node moves (computed here)
  cbor2 5.6.5 + blake3 (Python) encodings and digests of random trees
"""
import hashlib, os, random, subprocess, sys
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rn import rn
from itertools import permutations
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(ROOT, 'scripts/ref/view_ref')
sys.path.insert(0, REF)
from model import diff, fmt, apply, key, tag, parse  # noqa: E402
V = os.path.join(ROOT, 'vectors/ui')
T = os.path.join(ROOT, 'tests/ui')

def rng(n): return [str(i) for i in n]
S = lambda *xs: [str(x) for x in xs]
# (name, old, new): rendererChildren.spec.ts in order. Tokens as in
# vue_ref.js; h('div', 'hello') (text children) is the text node ~hello.
OFFICIAL = [
    ('should patch previously empty children', [], ['~hello']),
    ('should patch previously null children', [], ['~hello']),
    ('array children -> text children', ['div:x'], ['~hello']),
    ('plain object child', [], ['~[object Object]']),
    ('keyed append', S(1), S(1, 2, 3)),
    ('keyed prepend', S(4, 5), S(1, 2, 3, 4, 5)),
    ('keyed insert in middle', S(1, 2, 4, 5), S(1, 2, 3, 4, 5)),
    ('keyed insert at beginning and end', S(2, 3, 4), S(1, 2, 3, 4, 5)),
    ('keyed insert to empty parent', [], S(1, 2, 3, 4, 5)),
    ('keyed remove all children from parent', S(1, 2, 3, 4, 5), []),
    ('keyed remove from beginning', S(1, 2, 3, 4, 5), S(3, 4, 5)),
    ('keyed remove from end', S(1, 2, 3, 4, 5), S(1, 2, 3)),
    ('keyed remove from middle', S(1, 2, 3, 4, 5), S(1, 2, 4, 5)),
    ('keyed moving single child forward', S(1, 2, 3, 4), S(2, 3, 1, 4)),
    ('keyed moving single child backwards', S(1, 2, 3, 4), S(1, 4, 2, 3)),
    ('keyed moving single child to end', S(1, 2, 3), S(2, 3, 1)),
    ('keyed swap first and last', S(1, 2, 3, 4), S(4, 2, 3, 1)),
    ('keyed move to left & replace', S(1, 2, 3, 4, 5), S(4, 1, 2, 3, 6)),
    ('keyed move to left and leaves hold', S(1, 4, 5), S(4, 6)),
    ('keyed moved and set to undefined element ending at the end', S(2, 4, 5), S(4, 5, 3)),
    ('keyed reverse element', S(*range(1, 9)), S(*range(8, 0, -1))),
    ('keyed something', S(0, 1, 2, 3, 4, 5), S(4, 3, 2, 1, 5, 0)),
    ('children with the same key but with different tag',
     ['div#1', 'div#2', 'div#3', 'div#4'], ['div#4', 'span#3', 'span#2', 'div#1']),
    ('children with the same tag, same key, but one with data and one without data', ['div:one'], ['div:four']),
    ('move a key in non-keyed nodes with a size up', ['1', 'span:a', 'span:b', 'span:c'],
     ['span:d', 'span:a', 'span:b', 'span:c', '1', 'span:e']),
    ('append elements with updating children without keys', ['span:hello'], ['span:hello', 'span:world']),
    ('unmoved text nodes with updating children without keys', ['~text', 'span:hello'], ['~text', 'span:hello']),
    ('changing text children with updating children without keys', ['~text', 'span:hello'], ['~text2', 'span:hello']),
    ('prepend element with updating children without keys', ['span:world'], ['span:hello', 'span:world']),
    ('prepend element of different tag type with updating children without keys', ['span:world'], ['div:hello', 'span:world']),
    ('remove elements with updating children without keys', ['span:one', 'span:two', 'span:three'], ['span:one', 'span:three']),
    ('remove a single text node with updating children without keys', ['~one'], []),
    ('remove a single text node when children are updated', ['~one'], ['div:two', 'span:three']),
    ('remove a text node among other elements', ['~one', 'span:two'], ['div:three']),
    ('reorder elements', ['span:one', 'div:two', 'b:three'], ['b:three', 'div:two', 'span:one']),
    ('should not de-opt when both head and tail change', ['!', 'div:x', '!'], ['p:a', 'div:x', 'p:b']),
]
# 'random shuffle' (14 keys, 5 samples of Math.random shuffles): seeded.
_r = random.Random(14)
for s in range(5):
    p = list(range(14)); _r.shuffle(p)
    OFFICIAL.append(('keyed random shuffle %d' % s, S(*range(14)), S(*p)))
DUP = (S(1, 2, 3, 4, 5), S(1, 6, 6, 3, 5))  # 'should warn with duplicate keys'

def vue(cases):
    inp = ''.join(','.join(a) + '/' + ','.join(b) + '\n' for a, b in cases)
    p = subprocess.run(['node', 'vue_ref.js'], cwd=REF, input=inp, capture_output=True, text=True, check=True)
    return p.stdout.split('\n')[:len(cases)]

def tok(R, i):
    r = R.random()
    if r < 0.5: return str(i)
    if r < 0.6: return R.choice('ab') + '#%d' % i
    if r < 0.8: return R.choice('sp') + ':x%d' % i
    if r < 0.95: return '~t%d' % i
    return '!'

def live_cases(seed, n):
    R = random.Random(seed); out = []
    for _ in range(n):
        k = R.randint(0, 10); u = [tok(R, i) for i in range(1, 15)]; R.shuffle(u)
        a = u[:k]; b = [x for x in a if R.random() < 0.8] + u[k:k + R.randint(0, 3)]
        if R.random() < 0.7: R.shuffle(b)
        b = [('s:y%d' % R.randint(0, 99) if x.startswith('s:') and R.random() < 0.3 else x) for x in b]
        if R.random() < 0.1:  # same key, other tag
            b = [('b#' + key(x) if key(x) and tag(x) != 'b' and R.random() < 0.3 else x) for x in b]
        out.append((a, b))
    return out

def write_tsv(path, rows, head):
    with open(path, 'w') as f:
        f.write('\t'.join(head) + '\n')
        for r in rows: f.write('\t'.join(r) + '\n')

def read_tsv(path):
    with open(path) as f:
        ls = f.read().split('\n')
    return [l.split('\t') for l in ls[1:] if l]

HEAD = ['name', 'old', 'new', 'vue', 'ours']
J = lambda xs: ','.join(xs)

def refresh():
    rows = []
    out = vue([(a, b) for _, a, b in OFFICIAL])
    for (n, a, b), v in zip(OFFICIAL, out):
        ops = diff(a, b); assert apply(a, b, ops)
        rows.append([n, J(a), J(b), v, fmt(a, b, ops)])
    write_tsv(os.path.join(V, 'vue_official.tsv'), rows, HEAD)
    cs = live_cases(49, 6000)
    rows = []
    for i, ((a, b), v) in enumerate(zip(cs, vue(cs))):
        ops = diff(a, b); assert apply(a, b, ops)
        rows.append([str(i), J(a), J(b), v, fmt(a, b, ops)])
    write_tsv(os.path.join(V, 'vue_live.tsv'), rows, HEAD)

def retagged(a, b):
    return any(key(x) and any(key(y) == key(x) and tag(y) != tag(x) for y in b) for x in a)

def q(s): return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

def bfs_moves(a, b):
    """least single-node moves turning order a into order b (same keys)"""
    a, b = tuple(a), tuple(b)
    if a == b: return 0
    seen = {a}; d = deque([(a, 0)])
    while d:
        x, k = d.popleft()
        for i in range(len(x)):
            y = x[:i] + x[i + 1:]
            for j in range(len(x)):
                z = y[:j] + (x[i],) + y[j:]
                if z == b: return k + 1
                if z not in seen: seen.add(z); d.append((z, k + 1))
    raise AssertionError

def bfs_ops(a, b):
    """least removals + creations + moves turning child list a into one
    whose (tag, key) sequence is b's; created nodes take any needed sig"""
    sig = lambda x: (tag(x), key(x))
    goal = tuple(sig(x) for x in b)
    start = tuple(('o', sig(x), i) for i, x in enumerate(a))
    look = lambda st: tuple(x[1] for x in st)
    if look(start) == goal: return 0
    need = sorted(set(goal)); seen = {start}; d = deque([(start, 0)])
    while d:
        st, k = d.popleft(); nxt = []
        for i in range(len(st)):
            y = st[:i] + st[i + 1:]
            nxt.append(y)
            for j in range(len(st)):
                if j != i: nxt.append(y[:j] + (st[i],) + y[j:])
        if len(st) < len(goal) + 1:
            for g in need:
                for j in range(len(st) + 1): nxt.append(st[:j] + (('n', g, 0),) + st[j:])
        for z in nxt:
            if look(z) == goal: return k + 1
            if z not in seen: seen.add(z); d.append((z, k + 1))
    raise AssertionError

# ---- trees for the digest and replay sweeps (prefix form, see helpers) ----
import cbor2, blake3  # noqa: E402  (pip: cbor2==5.6.5 blake3)
NAMES = ['id', 'class', 'style', 'l', 'aria-label', 'x']
def rand_tree(R, depth=0, keyed=None):
    t = {'tag': R.choice(['div', 'span', 'p', 'li']), 'key': keyed or '', 'attrs': {}, 'kids': []}
    for nm in R.sample(NAMES, R.randint(0, 3)): t['attrs'][nm] = R.choice(['1', '2', 'a', 'opacity:0.5', 'é'])
    if depth < 3:
        ks = R.random() < 0.5
        for i in range(R.randint(0, 4)):
            if R.random() < 0.25 and not ks: t['kids'].append({'tag': '', 'key': '', 'attrs': {}, 'kids': [], 'text': R.choice(['x', 'y', 'hello', 'é'])})
            else: t['kids'].append(rand_tree(R, depth + 1, 'k%d' % R.randint(0, 9) if ks else None))
        if ks:  # distinct sibling keys
            seen = set(); t['kids'] = [c for c in t['kids'] if not (c['key'] in seen or seen.add(c['key']))]
    return t
def mutate(R, t):
    t = {**t, 'attrs': dict(t['attrs']), 'kids': [mutate(R, c) for c in t['kids']]}
    if t['tag'] == '':
        if R.random() < 0.3: t['text'] = R.choice(['x', 'z', 'hello'])
        return t
    if R.random() < 0.3 and t['attrs']: del t['attrs'][R.choice(sorted(t['attrs']))]
    if R.random() < 0.3: t['attrs'][R.choice(NAMES)] = R.choice(['1', '3', 'b'])
    if R.random() < 0.4: R.shuffle(t['kids'])
    if R.random() < 0.2 and t['kids']: t['kids'].pop(R.randrange(len(t['kids'])))
    if R.random() < 0.2:
        keyed = any(c['key'] for c in t['kids'])
        n = rand_tree(R, 3, 'n%d' % R.randint(0, 99) if keyed else None)
        if not keyed or n['key'] not in {c['key'] for c in t['kids']}: t['kids'].insert(R.randint(0, len(t['kids'])), n)
    if R.random() < 0.05: t['tag'] = 'b'
    return t
def prefix(t):
    if t['tag'] == '': return '~' + t['text']
    s = ['<' + t['tag']] + (['#' + t['key']] if t['key'] else [])
    s += ['@%s=%s' % kv for kv in t['attrs'].items()]
    s += [prefix(c) for c in t['kids']] + ['>']
    return ' '.join(s)
def py_cbor(t):
    """independent encoding: cbor2 canonical of nested Python lists"""
    def conv(t):
        return [t['tag'], t['key'], dict(t['attrs']), t.get('text', ''), [conv(c) for c in t['kids']]]
    return cbor2.dumps(conv(t), canonical=True)
def py_frame(seq, t):
    return blake3.blake3(cbor2.dumps(['rce.ui.frame/1', seq, blake3.blake3(py_cbor(t)).digest()], canonical=True)).hexdigest()

def shard(name, lines, per):
    for f in os.listdir(T):
        if f.startswith('view_checks_' + name): os.remove(os.path.join(T, f))
    for i in range(0, len(lines), per):
        with open(os.path.join(T, 'view_checks_%s%d.rce' % (name, i // per)), 'w') as f:
            f.write(rn(HDR + ''.join(lines[i:i + per])))

HDR = '# generated by scripts/gen/view_gen.py; do not edit\n' + open(os.path.join(ROOT, 'scripts/gen/view_helpers.rce')).read()

def main():
    if '--refresh' in sys.argv: refresh()
    off = read_tsv(os.path.join(V, 'vue_official.tsv'))
    live = read_tsv(os.path.join(V, 'vue_live.tsv'))
    o = [HDR, '# ---- Vue rendererChildren.spec.ts: final children (checked by vue(),\n# which replays and compares) and the container operations; where our\n# log differs from Vue\'s the Vue log is in vectors/ui/vue_official.tsv\n']
    for n, a, b, v, ours in off:
        aa, bb = a.split(',') if a else [], b.split(',') if b else []
        assert fmt(aa, bb, diff(aa, bb)) == ours
        if v != ours:
            assert retagged(aa, bb), n
            assert ours.count('m ') <= v.count('m ')
        o.append('# %s%s\ncheck vue(%s, %s) == %s\n' % (n, '' if v == ours else ' (Vue: ' + v + ')', q(a), q(b), q(ours)))
    a, b = DUP
    o.append('# should warn with duplicate keys: such a list is not a valid view\nfn ok_list(s: text) -> bool\n    let v = lv(s)\n    return view_ok(&v)\ncheck ok_list(%s) == true\ncheck ok_list(%s) == false\n' % (q(J(a)), q(J(b))))
    # random shuffle with per-position style attributes (tree form)
    _r = random.Random(15)
    for s in range(5):
        p = list(range(14)); _r.shuffle(p)
        op = ['%.5f' % _r.random() for _ in range(14)]
        ta = '<span ' + ' '.join('<span #%d @style=opacity:1 ~%d >' % (i, i) for i in range(14)) + ' >'
        tb = '<span ' + ' '.join('<span #%d @style=opacity:%s ~%d >' % (p[i], op[i], p[i]) for i in range(14)) + ' >'
        o.append('check rt(%s, %s) != "BAD"\n' % (q(ta), q(tb)))
    # derived: exact operations on small trees
    o.append(open(os.path.join(ROOT, 'scripts/gen/view_derived.rce')).read())
    with open(os.path.join(T, 'view_checks.rce'), 'w') as f: f.write(rn(''.join(o)))
    # live Vue logs
    ls = []
    for n, a, b, v, ours in live:
        aa, bb = a.split(',') if a else [], b.split(',') if b else []
        assert fmt(aa, bb, diff(aa, bb)) == ours
        if v != ours: assert retagged(aa, bb) and ours.count('m ') <= v.count('m '), n
        ls.append('check vue(%s, %s) == %s\n' % (q(a), q(b), q(ours)))
    shard('live', ls, 750)
    # minimality: every permutation up to 6 keys, plus random lists with
    # insertions and removals (moves count only kept keys)
    ms = []
    for n in range(0, 7):
        base = S(*range(1, n + 1))
        for p in permutations(base):
            ms.append('check moves(%s, %s) == %d\n' % (q(J(base)), q(J(p)), bfs_moves(base, p)))
    R = random.Random(7)
    for _ in range(600):
        u = S(*range(1, 13)); R.shuffle(u)
        a = u[:R.randint(0, 8)]; kept = [x for x in a if R.random() < 0.8]
        b = kept + u[8:8 + R.randint(0, 3)]; R.shuffle(b)
        ka = [x for x in a if x in kept]; kb = [x for x in b if x in kept]
        ms.append('check moves(%s, %s) == %d\n' % (q(J(a)), q(J(b)), bfs_moves(ka, kb)))
    shard('min', ms, 400)
    # optimal unkeyed alignment: least container operations (removals,
    # creations, moves) by breadth-first search, lists of up to 5 children
    # mixing keyed spans, unkeyed <s>/<p> and text nodes
    R = random.Random(53); os_ = []; seen = set()
    while len(os_) < 1500:
        def lst(pfx):
            n = R.randint(0, 5); out = []; ks = set()
            for i in range(n):
                c = R.choice(['s', 'p', '~', 'k'])
                if c == 'k':
                    k = R.choice('123')
                    if k in ks: continue
                    ks.add(k); out.append(k)
                elif c == '~': out.append('~%s%d' % (pfx, i))
                else: out.append('%s:%s%d' % (c, pfx, i))
            return out
        a, b = lst('a'), lst('b')
        if (J(a), J(b)) in seen: continue
        seen.add((J(a), J(b)))
        os_.append('check cost(%s, %s) == %d\n' % (q(J(a)), q(J(b)), bfs_ops(a, b)))
    shard('opt', os_, 300)
    # trees: Python cbor2/blake3 encodings, replay of random edits
    R = random.Random(49); ts = []
    for i in range(400):
        a = rand_tree(R); b = mutate(R, a)
        pa, pb = prefix(a), prefix(b)
        ts.append('check cbor_hex(%s) == "%s"\n' % (q(pa), py_cbor(a).hex()))
        ts.append('check rt(%s, %s) == "%s"\n' % (q(pa), q(pb), blake3.blake3(py_cbor(b)).hexdigest()))
        ts.append('check fd(%d, %s) == "%s"\n' % (i * 977, q(pb), py_frame(i * 977, b)))
    shard('tree', ts, 300)

if __name__ == '__main__':
    main()
