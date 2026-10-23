#!/usr/bin/env python3
"""Data section of lib/ui/text.rce from the Unicode 16.0.0 files, each pinned
by sha256. vectors/ui/ holds the property and small test files; the large
test files (LineBreakTest, BidiTest, BidiCharacterTest) and
DerivedCoreProperties are fetched from unicode.org into ~/.cache/rce-ucd/16
and checked. Usage: python3 scripts/gen/text_data.py (rewrites the data
section after the "# ---- data" marker of lib/ui/text.rce)."""
import hashlib, os, sys, urllib.request
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
U = 'https://www.unicode.org/Public/16.0.0/ucd/'
PIN = {
 'GraphemeBreakProperty.txt': ('auxiliary/', 'c29360bd6f7132811d701d29069541e827eb44bfc4c8fbde8c370d6982689dc1', 1),
 'GraphemeBreakTest.txt': ('auxiliary/', 'ee2b9354d270ac061b29f09662cafea06341d77e704b8cc6bd72aaeeda363cb5', 1),
 'WordBreakProperty.txt': ('auxiliary/', '476464e71a4b7b779b8ba7c5671f4338fea77da8e6b6b05fb82b3fdd14603779', 1),
 'WordBreakTest.txt': ('auxiliary/', 'ad985d5721f3fa6b45495663dfe44180f2f68976100dee0ea7451ef1a8f838e8', 1),
 'SentenceBreakProperty.txt': ('auxiliary/', '20aab5eca3842c7a27cc6756d74488a4a5f744c8dca2948ec1128f26a60d1f79', 1),
 'SentenceBreakTest.txt': ('auxiliary/', '0aef84034ee1789eb71021454fac384e83080b05922272d63cf297f4bf08150e', 1),
 'LineBreakTest.txt': ('auxiliary/', '910759a611a479f37df4f535d2e64d7589be3c5ac5f7491cf1fb4fa4cdb211e9', 0),
 'LineBreak.txt': ('', 'e97e4259d0d20fab150b9c7b4b28abfae5cd78ca97e7f4ac6ed20d685d5f4a7c', 1),
 'EastAsianWidth.txt': ('', '43adc76c0686a42cb370764eb8cfe2b2a45b10b855e5572a2db4a0eecce15d5b', 1),
 'emoji-data.txt': ('emoji/', 'f1365a5173eee18e1f98b240cdc492e84a25f1ce7e0c9d1094eb29c41a22696a', 1),
 'DerivedCoreProperties.txt': ('', '39d35161f2954497f69e08bdb9e701493f476a3d30222de20028feda36c1dabd', 0),
 'DerivedGeneralCategory.txt': ('extracted/', '7676ab755a41ef82108460238569e60ad65c191ddafe61b36c6765ec1353f293', 1),
 'DerivedBidiClass.txt': ('extracted/', '71ed943a49c58568d8d92e80ecc2ba2f06e62aee9c8ebb0e6e8bd2c3ed8b180e', 2),
 'BidiBrackets.txt': ('', 'b8f32554c6f658821fb0ee742d21c5b1f2086b9bf13071fed04894b022f93d67', 1),
 'BidiTest.txt': ('', '93e5eb9d88ca89dcf895f5576486a3363762ad2aa8f2db2fa56fe60cb82b9520', 0),
 'BidiCharacterTest.txt': ('', 'd04a51a90052dcd71c4e91ee5b3a9d973ee35c12406b5a99875ac8163c8f2804', 0),
}
def get(name):
    sub, sha, where = PIN[name]
    dirs = {1: os.path.join(ROOT, 'vectors/ui'), 2: os.path.join(ROOT, 'vectors/idna'), 0: os.path.expanduser('~/.cache/rce-ucd/16')}
    p = os.path.join(dirs[where], name)
    if not os.path.exists(p):
        os.makedirs(dirs[where], exist_ok=True)
        urllib.request.urlretrieve(U + sub + name, p)
    b = open(p, 'rb').read()
    assert hashlib.sha256(b).hexdigest() == sha, name
    return b.decode('utf-8')
def rows(t):
    for l in t.split('\n'):
        l = l.split('#')[0].strip()
        if l: yield [f.strip() for f in l.split(';')]
def missing(t):
    for l in t.split('\n'):
        if l.startswith('# @missing:'):
            yield [f.strip() for f in l[len('# @missing:'):].split(';')]
def rng(s):
    a, _, b = s.partition('..')
    return int(a, 16), int(b or a, 16)
def prop(name, names, default, field=1, filt=None):
    t = get(name)
    v = [default] * 0x110000
    for r in missing(t):
        if r[field] in names:
            a, b = rng(r[0])
            v[a:b + 1] = [names[r[field]]] * (b - a + 1)
    for r in rows(t):
        if filt and not filt(r): continue
        a, b = rng(r[0])
        v[a:b + 1] = [names[r[field]]] * (b - a + 1)
    return v
def runs(v):
    out = []
    for c in range(0x110000):
        if not out or out[-1][1] != v[c]: out.append((c, v[c]))
    return out
def tree(rs, leaf=str, ind='    '):
    def t(lo, hi, ind):
        if hi - lo == 1: return ind + 'return ' + leaf(rs[lo][1]) + '\n'
        m = (lo + hi) // 2
        return ind + 'if c < %d\n' % rs[m][0] + t(lo, m, ind + '    ') + t(m, hi, ind)
    return t(0, len(rs), ind)
def exact(keys, leaf, miss, var='c'):
    def t(lo, hi, ind):
        if hi - lo == 1:
            k, v = keys[lo]
            return ind + 'if %s == %d\n%s    return %s\n%sreturn %s\n' % (var, k, ind, leaf(v), ind, miss)
        m = (lo + hi) // 2
        return ind + 'if %s < %d\n' % (var, keys[m][0]) + t(lo, m, ind + '    ') + t(m, hi, ind)
    return t(0, len(keys), '    ')
def idx(names): return {n: i for i, n in enumerate(names)}

GCB = 'Other CR LF Control Extend ZWJ Regional_Indicator Prepend SpacingMark L V T LV LVT'.split()
WB = 'Other CR LF Newline Extend ZWJ Regional_Indicator Format Katakana Hebrew_Letter ALetter Single_Quote Double_Quote MidNumLet MidLetter MidNum Numeric ExtendNumLet WSegSpace'.split()
SB = 'Other CR LF Extend Sep Format Sp Lower Upper OLetter Numeric ATerm SContinue STerm Close'.split()
LB = 'XX BK CR LF CM NL SG WJ ZW GL SP ZWJ B2 BA BB HY CB CL CP EX IN NS OP QU IS NU PO PR SY AI AL CJ EB EM H2 H3 HL ID JL JV JT RI SA AK AP AS VF VI'.split()
EA = 'N A F H Na W'.split()
GC = 'Cn Lu Ll Lt Lm Lo Mn Mc Me Nd Nl No Pc Pd Ps Pe Pi Pf Po Sm Sc Sk So Zs Zl Zp Cc Cf Cs Co'.split()
BC = 'L R AL EN ES ET AN CS NSM BN B S WS ON LRE LRO RLE RLO PDF LRI RLI FSI PDI'.split()
BCLONG = dict(zip('Left_To_Right Right_To_Left Arabic_Letter European_Number European_Separator European_Terminator Arabic_Number Common_Separator Nonspacing_Mark Boundary_Neutral Paragraph_Separator Segment_Separator White_Space Other_Neutral Left_To_Right_Embedding Left_To_Right_Override Right_To_Left_Embedding Right_To_Left_Override Pop_Directional_Format Left_To_Right_Isolate Right_To_Left_Isolate First_Strong_Isolate Pop_Directional_Isolate'.split(), range(23)))

def tables():
    gcb = prop('GraphemeBreakProperty.txt', idx(GCB), 0)
    wb = prop('WordBreakProperty.txt', idx(WB), 0)
    sb = prop('SentenceBreakProperty.txt', idx(SB), 0)
    lb = prop('LineBreak.txt', idx(LB), 0)
    ea = prop('EastAsianWidth.txt', idx(EA), 0)
    gc = prop('DerivedGeneralCategory.txt', idx(GC), 0)
    bc = prop('DerivedBidiClass.txt', {**BCLONG, **idx(BC)}, 0)
    ep = prop('emoji-data.txt', {'Extended_Pictographic': 1}, 0, filt=lambda r: r[1] == 'Extended_Pictographic')
    incb = prop('DerivedCoreProperties.txt', {'Linker': 1, 'Consonant': 2, 'Extend': 3}, 0, field=2,
                filt=lambda r: r[1] == 'InCB' and r[2] != 'None')
    br = []
    for r in rows(get('BidiBrackets.txt')):
        br.append((int(r[0], 16), (int(r[1], 16), 1 if r[2] == 'o' else 2)))
    return dict(gcb=gcb, wb=wb, sb=sb, lb=lb, ea=ea, gc=gc, bc=bc, ep=ep, incb=incb, br=sorted(br))

def main():
    T = tables()
    o = '# ---- data (generated by scripts/gen/text_data.py from Unicode 16.0.0; do not edit)\n'
    o += '# Each table is a binary decision tree over code point ranges.\n\n'
    def doc(names): return ', '.join('%d %s' % (i, n) for i, n in enumerate(names))
    import textwrap
    def wrap(s): return '\n'.join('# ' + l for l in textwrap.wrap(s, 74)) + '\n'
    o += wrap('Grapheme_Cluster_Break: ' + doc(GCB) + ' (GraphemeBreakProperty.txt).') + 'pub fn gcb(c: i64) -> i64\n' + tree(runs(T['gcb'])) + '\n'
    o += wrap('Word_Break: ' + doc(WB) + ' (WordBreakProperty.txt).') + 'pub fn wb(c: i64) -> i64\n' + tree(runs(T['wb'])) + '\n'
    o += wrap('Sentence_Break: ' + doc(SB) + ' (SentenceBreakProperty.txt).') + 'pub fn sb(c: i64) -> i64\n' + tree(runs(T['sb'])) + '\n'
    o += wrap('Line_Break: ' + doc(LB) + ' (LineBreak.txt).') + 'pub fn lb(c: i64) -> i64\n' + tree(runs(T['lb'])) + '\n'
    o += wrap('East_Asian_Width: ' + doc(EA) + ' (EastAsianWidth.txt).') + 'pub fn ea(c: i64) -> i64\n' + tree(runs(T['ea'])) + '\n'
    o += wrap('General_Category: ' + doc(GC) + ' (DerivedGeneralCategory.txt).') + 'pub fn gc(c: i64) -> i64\n' + tree(runs(T['gc'])) + '\n'
    o += wrap('Bidi_Class: ' + doc(BC) + ' (DerivedBidiClass.txt).') + 'pub fn bidi_class(c: i64) -> i64\n' + tree(runs(T['bc'])) + '\n'
    o += '# Extended_Pictographic (emoji-data.txt).\npub fn ext_pict(c: i64) -> bool\n' + tree(runs(T['ep']), lambda v: 'true' if v else 'false') + '\n'
    o += '# Indic_Conjunct_Break: 0 None, 1 Linker, 2 Consonant, 3 Extend (DerivedCoreProperties.txt).\npub fn incb(c: i64) -> i64\n' + tree(runs(T['incb'])) + '\n'
    o += '# Bidi_Paired_Bracket of c, or -1 (BidiBrackets.txt).\npub fn bracket_pair(c: i64) -> i64\n' + exact([(k, v[0]) for k, v in T['br']], str, '-1') + '\n'
    o += '# Bidi_Paired_Bracket_Type: 0 None, 1 Open, 2 Close (BidiBrackets.txt).\npub fn bracket_type(c: i64) -> i64\n' + exact([(k, v[1]) for k, v in T['br']], str, '0') + '\n'
    p = os.path.join(ROOT, 'lib/ui/text.rce')
    s = open(p).read()
    k = s.index('# ---- data')
    open(p, 'w').write(s[:k] + o)
    sys.stderr.write('data: %d lines\n' % o.count('\n'))
if __name__ == '__main__':
    main()
