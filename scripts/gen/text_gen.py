#!/usr/bin/env python3
"""Generate tests/ui/text_checks*.rce from the Unicode 16.0.0 test files
pinned in scripts/gen/text_data.py: GraphemeBreakTest, WordBreakTest,
SentenceBreakTest, LineBreakTest (every line), BidiTest (every line and
every paragraph direction in its bitset) and BidiCharacterTest (every
line), plus scripts/gen/text_derived.rce."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_data import get, ROOT, BC
OUT = os.path.join(ROOT, 'tests/ui')
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    if f.startswith('text_checks'): os.remove(os.path.join(OUT, f))
H = {
 'seg': '''
fn gk(cs: List[i64]) -> List[i64] = grapheme_breaks(&cs)
fn wk(cs: List[i64]) -> List[i64] = word_breaks(&cs)
fn sk(cs: List[i64]) -> List[i64] = sentence_breaks(&cs)
fn lk(cs: List[i64]) -> List[i64] = line_breaks(&cs)
''',
}
def seg_lines(name):
    out = []
    for l in get(name).split('\n'):
        l = l.split('#')[0].strip()
        if not l: continue
        toks = l.split()
        assert toks[0] in ('÷', '×') and toks[-1] == '÷', l
        cps, br = [], []
        for t in toks[1:]:
            if t in ('÷', '×'):
                if t == '÷': br.append(len(cps))
            else: cps.append(int(t, 16))
        out.append((cps, br))
    return out
shards = {}
def emit(shard, line):
    shards.setdefault(shard, []).append(line)
counts = {}
for name, fn, sh in [('GraphemeBreakTest.txt', 'gk', 'seg'), ('WordBreakTest.txt', 'wk', 'seg'), ('SentenceBreakTest.txt', 'sk', 'seg')]:
    cases = seg_lines(name)
    counts[name] = len(cases)
    for cps, br in cases:
        emit(sh, 'check %s([%s]) == [%s]' % (fn, ', '.join(map(str, cps)), ', '.join(map(str, br))))
if '--seg' not in sys.argv:
    cases = seg_lines('LineBreakTest.txt')
    counts['LineBreakTest.txt'] = len(cases)
    for k, (cps, br) in enumerate(cases):
        emit('line%d' % (k // 2000), 'check lk([%s]) == [%s]' % (', '.join(map(str, cps)), ', '.join(map(str, br))))
BH = '''
fn ints(t: text) -> List[i64]
    var o: List[i64] = []
    var v: i64 = 0
    var have = false
    var neg = false
    for c in t.chars()
        let b = c[0] as i64
        if b == 120
            neg = true
            have = true
        elif b >= 48 and b <= 57
            v = v * 10 + b - 48
            have = true
        else
            if have
                o.push(if neg then -1 else v)
            v = 0
            have = false
            neg = false
    if have
        o.push(if neg then -1 else v)
    return o
# BidiTest group: classes as letters (a = L .. w = PDI), cases joined by
# ";", each ending in its hex bitset (1 auto, 2 LTR, 4 RTL). Returns -1
# when all pass, else 4 * case + bit of the first failure.
fn btg(lev: text, ord: text, cases: text) -> i64
    let want = ints(lev)
    let wo = ints(ord)
    var ty: List[i64] = []
    var k: i64 = 0
    let nobr: List[i64] = []
    for c in cases.chars()
        let b = c[0] as i64
        if b == 59
            k += 1
        elif b >= 97
            ty.push(b - 97)
        else
            let bits = if b <= 57 then b - 48 else b - 87
            var bit: i64 = 1
            while bit <= 4
                if bits & bit != 0
                    let dir = if bit == 1 then 2 else (if bit == 2 then 0 else 1)
                    let r = bidi_levels(&ty, &nobr, dir)
                    if r.levels != want or bidi_order(&r.levels) != wo
                        return 4 * k + bit
                bit = bit * 2
            ty = []
    return -1
fn bct(cs: List[i64], dir: i64, para: i64, lev: List[i64], ord: List[i64]) -> bool
    let ty = bidi_types(&cs)
    let r = bidi_levels(&ty, &cs, dir)
    return r.para == para and r.levels == lev and bidi_order(&r.levels) == ord
'''
if '--seg' not in sys.argv:
    BI = {n: i for i, n in enumerate(BC)}
    lev = ord_ = None
    groups = []
    for l in get('BidiTest.txt').split('\n'):
        l = l.split('#')[0].strip()
        if not l: continue
        if l.startswith('@Levels:'):
            lev = l[8:].strip(); continue
        if l.startswith('@Reorder:'):
            ord_ = l[9:].strip(); continue
        if l.startswith('@'): continue
        a, b = l.split(';')
        enc = ''.join(chr(97 + BI[x]) for x in a.split()) + b.strip()
        if not groups or groups[-1][0] != (lev, ord_):
            groups.append(((lev, ord_), []))
        groups[-1][1].append(enc)
    nb = sum(len(g[1]) for g in groups)
    counts['BidiTest.txt'] = nb
    PER = 6000
    shard, size = 0, 0
    for (lv, od), cases in groups:
        for j in range(0, len(cases), 500):
            part = cases[j:j + 500]
            if size + len(part) > PER:
                shard += 1; size = 0
            size += len(part)
            emit('bidi%d' % shard, 'check btg("%s", "%s", "%s") == -1' % (lv, od, ';'.join(part)))
    nc = 0
    for l in get('BidiCharacterTest.txt').split('\n'):
        l = l.split('#')[0].strip()
        if not l: continue
        f = l.split(';')
        cps = [int(x, 16) for x in f[0].split()]
        lv = [-1 if x == 'x' else int(x) for x in f[3].split()]
        od = [int(x) for x in f[4].split()]
        emit('bidichar%d' % (nc // 4000), 'check bct([%s], %s, %s, [%s], [%s])' % (', '.join(map(str, cps)), f[1], f[2], ', '.join(map(str, lv)), ', '.join(map(str, od))))
        nc += 1
    counts['BidiCharacterTest.txt'] = nc

# ICU 76 live reference (scripts/gen/text_icu.py -> vectors/ui/icu76_live.tsv)
IH = """fn bil(cs: List[i64], dir: i64, para: i64, lev: List[i64]) -> bool
    let ty = bidi_types(&cs)
    let r = bidi_levels(&ty, &cs, dir)
    if r.para != para
        return false
    var i: usize = 0
    while i < lev.len()
        if r.levels[i] != -1 and r.levels[i] != lev[i]
            return false
        i += 1
    return true
"""
if os.path.exists(os.path.join(ROOT, 'vectors/ui/icu76_live.tsv')):
    ni = 0
    FN = {'g': 'gk', 'w': 'wk', 's': 'sk', 'l': 'lk'}
    for l in open(os.path.join(ROOT, 'vectors/ui/icu76_live.tsv')).read().split('\n'):
        if not l: continue
        f = l.split('\t')
        cps = ', '.join(str(int(x, 16)) for x in f[1].split())
        if f[0] == 'b':
            line = 'check bil([%s], %s, %s, [%s])' % (cps, f[2], f[3], ', '.join(f[4].split()))
        else:
            line = 'check %s([%s]) == [%s]' % (FN[f[0]], cps, ', '.join(f[2].split()))
        emit('icu%d' % (ni // 4000), line); ni += 1
    counts['icu76_live.tsv'] = ni
for sh, ls in shards.items():
    name = 'text_checks.rce' if sh == 'seg' else 'text_checks_%s.rce' % sh
    hd = H['seg'] + (BH if sh.startswith('bidi') else '') + (BH + IH if sh.startswith('icu') else '')
    tail = ''
    if sh == 'seg':
        tail = '\n# ---- derived (scripts/gen/text_derived.rce)\n' + open(os.path.join(ROOT, 'scripts/gen/text_derived.rce')).read()
    open(os.path.join(OUT, name), 'w').write('# generated by scripts/gen/text_gen.py; do not edit\n' + hd + '\n'.join(ls) + '\n' + tail)
sys.stderr.write(' '.join('%s %d' % kv for kv in counts.items()) + '\n')
