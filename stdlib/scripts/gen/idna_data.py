# Data section of lib/net/idna.rce from the Unicode 16.0.0 files, each
# pinned by sha256 (vectors/idna/ holds the smaller ones; the large UCD
# files are fetched from unicode.org into a cache and checked).
# Usage: python3 scripts/gen/idna_data.py > /tmp/data && splice between
# the "# ---- data" markers of lib/net/idna.rce.
import hashlib, os, sys, urllib.request
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
U = 'https://www.unicode.org/Public/16.0.0/'
PIN = {
 'IdnaMappingTable.txt': ('idna/16.0.0/', '6db2ef4ed35f3b3de74ebc2e00404a9607f76d499f576b8d4043cf14f1ed175c'),
 'IdnaTestV2.txt': ('idna/16.0.0/', 'ffcf59f6e97d765caa5ac8315e9511ba3b42955a554cbc431e4c016f20d25ca9'),
 'UnicodeData.txt': ('16.0.0/ucd/', 'ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f'),
 'CompositionExclusions.txt': ('16.0.0/ucd/', '89e83cf9cc8bef6c1f8bf77e42cf6f0341dfa42e66261f4dbe9b492e7a23c8ee'),
 'DerivedNormalizationProps.txt': ('16.0.0/ucd/', '4d4c03892dea9146d674b686e495df2d55a28d071ac474041d73518f887abddc'),
 'DerivedBidiClass.txt': ('16.0.0/ucd/extracted/', '71ed943a49c58568d8d92e80ecc2ba2f06e62aee9c8ebb0e6e8bd2c3ed8b180e'),
 'DerivedJoiningType.txt': ('16.0.0/ucd/extracted/', '6bd08b97da66b70ccfdab105a352de2984e02625239ec5695422c99b33d854f0'),
 'NormalizationTest.txt': ('16.0.0/ucd/', 'd811971453e7075e1ad56fb1b301eece5aa80757b81f6156e74a1bfb3ae5ceb1'),
}
def get(name):
    for d in (os.path.join(ROOT, 'vectors/idna'), os.path.expanduser('~/.cache/rce-ucd')):
        p = os.path.join(d, name)
        if os.path.exists(p): break
    else:
        os.makedirs(d, exist_ok=True)
        urllib.request.urlretrieve(U.replace('16.0.0/', '') + PIN[name][0] + name, p)
    b = open(p, 'rb').read()
    assert hashlib.sha256(b).hexdigest() == PIN[name][1], name
    return b.decode('utf-8')
def rows(t):
    for l in t.split('\n'):
        l = l.split('#')[0].strip()
        if l: yield [f.strip() for f in l.split(';')]
def rng(s):
    a, _, b = s.partition('..')
    return int(a, 16), int(b or a, 16)
def ranges(f, default):
    """code point -> value table as sorted (start, value) runs over 0..10FFFF"""
    v = [default] * 0x110000
    f(v)
    out = []
    for c in range(0x110000):
        if not out or out[-1][1] != v[c]: out.append((c, v[c]))
    return out

def main():
    KIND = {'valid': 0, 'mapped': 1, 'deviation': 2, 'disallowed': 3, 'ignored': 4}
    m = [None] * 0x110000
    for r in rows(get('IdnaMappingTable.txt')):
        a, b = rng(r[0]); k = KIND[r[1]]
        mp = tuple(int(x, 16) for x in r[2].split()) if len(r) > 2 and r[2] else ()
        for c in range(a, b + 1): m[c] = (k, mp if k in (1, 2) else ())
    assert all(x is not None for x in m)
    runs = []
    for c in range(0x110000):
        if not runs or runs[-1][1] != m[c] or m[c][0] in (1, 2): runs.append((c, m[c]))
    # merge consecutive mapped code points with the same target
    ms, mk, mo, mcp = [], [], [], []
    seen = {}
    for c, (k, mp) in runs:
        if ms and mk[-1] == k and k in (1, 2) and mo[-1] == seen.get(mp, -2): continue
        ms.append(c); mk.append(k)
        if k in (1, 2):
            if mp not in seen: seen[mp] = len(mcp); mcp.extend([len(mp)] + list(mp))
            mo.append(seen[mp])
        else: mo.append(-1)
    ud = list(rows(get('UnicodeData.txt')))
    ccc = [0] * 0x110000; mark = [0] * 0x110000; dec = {}
    i = 0
    while i < len(ud):
        r = ud[i]; c = int(r[0], 16)
        if r[1].endswith('First>'):
            e = int(ud[i + 1][0], 16); cs = range(c, e + 1); i += 2
        else: cs = [c]; i += 1
        for x in cs:
            ccc[x] = int(r[3]); mark[x] = 1 if r[2].startswith('M') else 0
            if r[5] and not r[5].startswith('<'): dec[x] = [int(y, 16) for y in r[5].split()]
    def full(x):
        if x not in dec: return [x]
        o = []
        for y in dec[x]: o += full(y)
        return o
    fce = set()
    for r in rows(get('DerivedNormalizationProps.txt')):
        if r[1] == 'Full_Composition_Exclusion':
            a, b = rng(r[0]); fce.update(range(a, b + 1))
    get('CompositionExclusions.txt')
    comp = sorted((d[0] * 0x110000 + d[1], x) for x, d in dec.items() if len(d) == 2 and x not in fce)
    dk = sorted(dec); dv = []; doff = []
    for x in dk:
        f = full(x); doff.append(len(dv)); dv.extend([len(f)] + f)
    BIDI = ['L', 'R', 'AL', 'AN', 'EN', 'ES', 'CS', 'ET', 'ON', 'BN', 'NSM', 'B', 'S', 'WS', 'LRE', 'LRO', 'RLE', 'RLO', 'PDF', 'LRI', 'RLI', 'FSI', 'PDI']
    LONG = {'Left_To_Right': 'L', 'Right_To_Left': 'R', 'Arabic_Letter': 'AL', 'European_Terminator': 'ET'}
    bt = get('DerivedBidiClass.txt')
    def fb(v):
        for l in bt.split('\n'):
            if l.startswith('# @missing:'):
                a, b = rng(l.split(':')[1].split(';')[0].strip()); n = BIDI.index(LONG[l.split(';')[1].strip()])
                for c in range(a, b + 1): v[c] = n
        for r in rows(bt):
            a, b = rng(r[0]); n = BIDI.index(r[1])
            for c in range(a, b + 1): v[c] = n
    bidi = ranges(fb, 0)
    JT = ['U', 'C', 'D', 'R', 'L', 'T']
    def fj(v):
        for r in rows(get('DerivedJoiningType.txt')):
            a, b = rng(r[0])
            for c in range(a, b + 1): v[c] = JT.index(r[1])
    jt = ranges(fj, 0)
    cccr = ranges(lambda v: v.__setitem__(slice(None), ccc), 0)
    markr = ranges(lambda v: v.__setitem__(slice(None), mark), 0)
    def tree(runs, leaf, ind='    '):
        """binary decision tree over sorted (start, value) runs covering all code points"""
        def t(lo, hi, ind):
            if hi - lo == 1: return ind + 'return ' + leaf(runs[lo][1]) + '\n'
            m = (lo + hi) // 2
            return ind + 'if c < %d\n' % runs[m][0] + t(lo, m, ind + '    ') + t(m, hi, ind)
        return t(0, len(runs), ind)
    def merge(rs):
        o = []
        for a, v in rs:
            if not o or o[-1][1] != v: o.append((a, v))
        return o
    def exact(keys, leaf, miss, var='c'):
        """keys sorted (key, value): binary search, equality at the leaves"""
        def t(lo, hi, ind):
            if hi - lo == 1:
                k, v = keys[lo]
                return ind + 'if %s == %d\n%s    return %s\n%sreturn %s\n' % (var, k, ind, leaf(v), ind, miss)
            m = (lo + hi) // 2
            return ind + 'if %s < %d\n' % (var, keys[m][0]) + t(lo, m, ind + '    ') + t(m, hi, ind)
        return t(0, len(keys), '    ')
    Lst = lambda xs: '[' + ', '.join(str(x) for x in xs) + ']'
    o = '# ---- data (generated by scripts/gen/idna_data.py from Unicode 16.0.0; do not edit)\n'
    o += '# Each table is a binary decision tree over code point ranges (rce 0.1.3\n# copies a const list on every access, so lists are kept for leaves).\n\n'
    srun = merge([(c, k) for c, k in zip(ms, mk)])
    o += '# IDNA status: 0 valid, 1 mapped, 2 deviation, 3 disallowed, 4 ignored (IdnaMappingTable.txt).\npub fn status(c: i64) -> i64\n' + tree(srun, str) + '\n'
    mrun = merge([(c, tuple(mcp[off + 1: off + 1 + mcp[off]]) if off >= 0 else None) for c, off in zip(ms, mo)])
    o += '# The mapping of a mapped or deviation code point ([] otherwise).\nfn mapping(c: i64) -> List[i64]\n' + tree(mrun, lambda v: Lst(v or ())) + '\n'
    o += '# Canonical_Combining_Class (UnicodeData.txt).\npub fn ccc(c: i64) -> i64\n' + tree(cccr, str) + '\n'
    o += '# General_Category=Mark (UnicodeData.txt).\npub fn is_mark(c: i64) -> bool\n' + tree(markr, lambda v: 'true' if v else 'false') + '\n'
    o += '# Bidi class index: 0 L, 1 R, 2 AL, 3 AN, 4 EN, 5 ES, 6 CS, 7 ET, 8 ON,\n# 9 BN, 10 NSM, 11 and above the other classes (DerivedBidiClass.txt).\npub fn bidi_class(c: i64) -> i64\n' + tree(bidi, str) + '\n'
    o += '# Joining type: 0 U (non joining), 1 C, 2 D, 3 R, 4 L, 5 T (DerivedJoiningType.txt).\npub fn joining_type(c: i64) -> i64\n' + tree(jt, str) + '\n'
    o += '# Full canonical decomposition, without Hangul (algorithmic); [] when none.\nfn decomposition(c: i64) -> List[i64]\n' + exact([(x, dv[f + 1: f + 1 + dv[f]]) for x, f in zip(dk, doff)], Lst, '[]') + '\n'
    o += '# Primary composite of first * 1114112 + second, or -1.\nfn composite(k: i64) -> i64\n' + exact(comp, str, '-1', 'k') + '\n'
    o += '# ---- end data\n'
    sys.stdout.write(o)
    print('runs map %d cps %d ccc %d mark %d bidi %d jt %d dec %d/%d comp %d' % (len(ms), len(mcp), len(cccr), len(markr), len(bidi), len(jt), len(dk), len(dv), len(comp)), file=sys.stderr)
if __name__ == '__main__': main()
