#!/usr/bin/env python3
"""Data section of lib/text/str.rce from the Unicode 15.1.0 files, each
pinned by sha256. Unicode 15.1.0 is the version of CPython 3.13's
unicodedata, the reference the module is checked against.

The files are read from ~/.cache/rce-ucd/15.1. A missing file is fetched
from https://www.unicode.org/Public/15.1.0/ucd/ or, when that host is not
reachable, from the copy of the same release in the unicode-org/unicodetools
repository (unicodetools/data/ucd/15.1.0/); either way the sha256 must match.

Every property is derived from the UCD files alone, by the rules of CPython's
Tools/unicode/makeunicodedata.py. When this script runs on CPython 3.13 (whose
unicodedata is 15.1.0) it then compares the derived data with CPython for
every code point: str.upper, lower, title, casefold of the one-character
string, the twelve str.is* predicates, and NFKD followed by the ASCII
"ignore" encoding (the first step of Django's slugify). Any difference stops
the run. Node is used, when present, to confirm the letter class of the
change-case word splitter (/[^\\p{L}\\d]/iu) on the code points whose Unicode
16.0 data (Node 22) equals 15.1.

Usage: python3 scripts/gen/str_data.py (rewrites the data section after the
"# ---- data" marker of lib/text/str.rce)."""
import hashlib, os, subprocess, sys, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
CACHE = os.path.expanduser('~/.cache/rce-ucd/15.1')
URLS = ['https://www.unicode.org/Public/15.1.0/ucd/',
        'https://raw.githubusercontent.com/unicode-org/unicodetools/main/unicodetools/data/ucd/15.1.0/']
PIN = {
    'UnicodeData.txt': ('', '2fc713e6a31a87c4850a37fe2caffa4218180fadb5de86b43a143ddb4581fb86'),
    'SpecialCasing.txt': ('', '55a477efd933a52cd27e6a9bf70265bb2d8814af31aab07767abc8eb421f27ef'),
    'CaseFolding.txt': ('', '4e55acfdc32825a22e87670e9056a3bf94ad7c5400065778e9e10f8314372bcf'),
    'DerivedCoreProperties.txt': ('', 'f55d0db69123431a7317868725b1fcbf1eab6b265d756d1bd7f0f6d9f9ee108b'),
    'DerivedNumericType.txt': ('extracted/', '0f22ca7770c836211e2b5390c6c9c3b5faae8e60cd8a59f244a43f9f5421cddd'),
}
N = 0x110000


def get(name):
    sub, sha = PIN[name]
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        os.makedirs(CACHE, exist_ok=True)
        err = None
        for u in URLS:
            try:
                urllib.request.urlretrieve(u + sub + name, p)
                break
            except Exception as e:  # try the next copy of the release
                err = e
        else:
            raise SystemExit('cannot fetch %s: %s' % (name, err))
    b = open(p, 'rb').read()
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit('%s: sha256 differs from the pin' % name)
    return b.decode('utf-8')


def rows(t):
    for l in t.split('\n'):
        l = l.split('#')[0].strip()
        if l:
            yield [f.strip() for f in l.split(';')]


def rng(s):
    a, _, b = s.partition('..')
    return int(a, 16), int(b or a, 16)


# Flag bits of a record (the same order is documented in str.rce).
ALPHA, DECIMAL, DIGIT, NUMERIC, SPACE, PRINTABLE, UPPER, LOWER, TITLE, CASED, \
    CASE_IGNORABLE, XID_START, XID_CONTINUE, GC_LU, GC_LL, WORD = [1 << i for i in range(16)]
FLAG_NAMES = ['ALPHA', 'DECIMAL', 'DIGIT', 'NUMERIC', 'SPACE', 'PRINTABLE', 'UPPER', 'LOWER',
              'TITLE', 'CASED', 'CASE_IGNORABLE', 'XID_START', 'XID_CONTINUE', 'GC_LU', 'GC_LL', 'WORD']


def derive():
    """Per code point: general category, bidi class, simple case mappings,
    decomposition, from UnicodeData.txt; the rest from the other files."""
    gc = ['Cn'] * N
    bidi = [''] * N
    su, sl, st = [None] * N, [None] * N, [None] * N
    decomp = {}
    first = None
    for l in get('UnicodeData.txt').split('\n'):
        if not l:
            continue
        f = l.split(';')
        c = int(f[0], 16)
        if f[1].endswith(', First>'):
            first = c
            continue
        lo = first if f[1].endswith(', Last>') else c
        first = None
        for x in range(lo, c + 1):
            gc[x] = f[2]
            bidi[x] = f[4]
        if f[5]:
            decomp[c] = f[5]
        if f[12]:
            su[c] = int(f[12], 16)
        if f[13]:
            sl[c] = int(f[13], 16)
        if f[14]:
            st[c] = int(f[14], 16)
    props = [set() for _ in range(N)]
    for r in rows(get('DerivedCoreProperties.txt')):
        if r[1] in ('Uppercase', 'Lowercase', 'Cased', 'Case_Ignorable', 'XID_Start', 'XID_Continue'):
            a, b = rng(r[0])
            for x in range(a, b + 1):
                props[x].add(r[1])
    ntype = [None] * N
    for r in rows(get('DerivedNumericType.txt')):
        a, b = rng(r[0])
        for x in range(a, b + 1):
            ntype[x] = r[1]
    special = {}
    for r in rows(get('SpecialCasing.txt')):
        # only unconditional mappings (no condition list), as CPython
        if len(r) > 4 and r[4]:
            continue
        c = int(r[0], 16)
        special[c] = tuple([int(x, 16) for x in r[i].split()] for i in (1, 2, 3))  # lower, title, upper
    fold, sfold = {}, {}
    for r in rows(get('CaseFolding.txt')):
        c = int(r[0], 16)
        if r[1] in ('C', 'F'):
            fold[c] = [int(x, 16) for x in r[2].split()]
        if r[1] in ('C', 'S'):
            sfold[c] = int(r[2], 16)
    return gc, bidi, su, sl, st, decomp, props, ntype, special, fold, sfold


def nfkd_ascii_table(decomp):
    """ASCII part of the NFKD form of each code point: the decomposition
    applied recursively (canonical and compatibility), Hangul syllables by
    the algorithm of Unicode 3.12 (all jamo, so no ASCII), then only the
    ASCII code points kept, as Python's encode('ascii', 'ignore') does."""
    memo = {}

    def full(c):
        if c in memo:
            return memo[c]
        if 0xAC00 <= c <= 0xD7A3:
            r = []  # Hangul syllables decompose to jamo only
        elif c in decomp:
            parts = decomp[c].split()
            if parts[0].startswith('<'):
                parts = parts[1:]
            r = []
            for p in parts:
                r.extend(full(int(p, 16)))
        else:
            r = [c]
        memo[c] = r
        return r
    out = {}
    for c in range(128, N):
        if 0xD800 <= c <= 0xDFFF:
            continue
        a = ''.join(chr(x) for x in full(c) if x < 128)
        if a:
            out[c] = a
    return out


def build():
    gc, bidi, su, sl, st, decomp, props, ntype, special, fold, sfold = derive()
    L_GC = ('Lu', 'Ll', 'Lt', 'Lm', 'Lo')
    flags = [0] * N
    up, lo, ti, cf = [None] * N, [None] * N, [None] * N, [None] * N
    for c in range(N):
        f = 0
        g = gc[c]
        if g in L_GC:
            f |= ALPHA
        if ntype[c] == 'Decimal':
            f |= DECIMAL
        if ntype[c] in ('Decimal', 'Digit'):
            f |= DIGIT
        if ntype[c] is not None:
            f |= NUMERIC
        if g == 'Zs' or bidi[c] in ('WS', 'B', 'S'):
            f |= SPACE
        if c == 32 or (g[0] not in ('C', 'Z')):
            f |= PRINTABLE
        p = props[c]
        if 'Uppercase' in p:
            f |= UPPER
        if 'Lowercase' in p:
            f |= LOWER
        if g == 'Lt':
            f |= TITLE
        if 'Cased' in p:
            f |= CASED
        if 'Case_Ignorable' in p:
            f |= CASE_IGNORABLE
        if 'XID_Start' in p:
            f |= XID_START
        if 'XID_Continue' in p:
            f |= XID_CONTINUE
        if g == 'Lu':
            f |= GC_LU
        if g == 'Ll':
            f |= GC_LL
        flags[c] = f
        u1 = su[c] if su[c] is not None else c
        l1 = sl[c] if sl[c] is not None else c
        t1 = st[c] if st[c] is not None else u1
        if c in special:
            sp = special[c]
            lo[c], ti[c], up[c] = list(sp[0]), list(sp[1]), list(sp[2])
        else:
            lo[c], ti[c], up[c] = [l1], [t1], [u1]
        cf[c] = fold.get(c, [c])
    # WORD: the letter class of change-case's /[^\p{L}\d]+/giu, that is the
    # code points c such that some letter or ASCII digit x has the simple
    # case folding of c (ECMA-262 Canonicalize with the u and i flags).
    sf = lambda x: sfold.get(x, x)
    keep = set()
    for x in range(N):
        if flags[x] & ALPHA or 48 <= x <= 57:
            keep.add(sf(x))
    for c in range(N):
        if not (48 <= c <= 57) and sf(c) in keep:
            flags[c] |= WORD
    asc = nfkd_ascii_table(decomp)
    return flags, up, lo, ti, cf, asc, gc


def verify(flags, up, lo, ti, cf, asc):
    import unicodedata
    if sys.version_info[:2] != (3, 13) or unicodedata.unidata_version != '15.1.0':
        sys.stderr.write('verify: skipped (needs CPython 3.13 with Unicode 15.1.0)\n')
        return
    preds = [(str.isalpha, ALPHA), (str.isdecimal, DECIMAL), (str.isdigit, DIGIT),
             (str.isnumeric, NUMERIC), (str.isspace, SPACE), (str.isprintable, PRINTABLE)]
    bad = 0
    for c in range(N):
        if 0xD800 <= c <= 0xDFFF:
            continue
        s = chr(c)
        f = flags[c]
        exp = ''.join(map(chr, up[c]))
        if s.upper() != exp: bad += 1; print('upper', hex(c))
        if s.lower() != ''.join(map(chr, lo[c])): bad += 1; print('lower', hex(c))
        if s.title() != ''.join(map(chr, ti[c])): bad += 1; print('title', hex(c))
        if s.casefold() != ''.join(map(chr, cf[c])): bad += 1; print('casefold', hex(c))
        for fn, m in preds:
            if fn(s) != bool(f & m): bad += 1; print(fn.__name__, hex(c))
        if s.isalnum() != bool(f & (ALPHA | DECIMAL | DIGIT | NUMERIC)): bad += 1; print('isalnum', hex(c))
        # one-character isupper/islower/istitle follow from the flags
        isup = bool(f & UPPER) and not (f & (LOWER | TITLE))
        islo = bool(f & LOWER) and not (f & (UPPER | TITLE))
        if s.isupper() != isup: bad += 1; print('isupper', hex(c))
        if s.islower() != islo: bad += 1; print('islower', hex(c))
        if s.istitle() != bool(f & (UPPER | TITLE)): bad += 1; print('istitle', hex(c))
        if s.isidentifier() != (bool(f & XID_START) or c == 95): bad += 1; print('isidentifier', hex(c))
        if ('a' + s).isidentifier() != bool(f & XID_CONTINUE): bad += 1; print('xid_continue', hex(c))
        if (('A' + s).isupper()) != (not (f & (LOWER | TITLE))): bad += 1; print('cased-upper', hex(c))
        if c >= 128:
            a = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
            if a != asc.get(c, ''): bad += 1; print('nfkd', hex(c))
        # Cased and Case_Ignorable through the final sigma rule of str.lower
        sig = ('A' + s + 'Σ').lower()[-1]
        want = 'ς' if (f & (CASED | CASE_IGNORABLE)) else 'σ'
        if c != 0x3a3 and sig != want: bad += 1; print('sigma-before', hex(c))
        sig2 = ('AΣ' + s).lower()[1]
        # after the sigma: case-ignorable characters are skipped, then a
        # cased character makes it non-final
        want2 = 'σ' if (f & CASED) and not (f & CASE_IGNORABLE) else 'ς'
        if c != 0x3a3 and sig2 != want2: bad += 1; print('sigma-after', hex(c))
        if bad > 20:
            raise SystemExit('verify: too many differences')
    if bad:
        raise SystemExit('verify: %d differences from CPython' % bad)
    sys.stderr.write('verify: every code point agrees with CPython %s (Unicode %s)\n'
                     % (sys.version.split()[0], unicodedata.unidata_version))


def verify_node(flags, gc):
    """The WORD class against Node's own regex on code points whose general
    category is the same in Node (Unicode 16.0) and here."""
    js = r"""
const re = /[^\p{L}\d]/iu, out = [];
for (let c = 0; c < 0x110000; c++) {
  if (c >= 0xD800 && c <= 0xDFFF) continue;
  const s = String.fromCodePoint(c);
  const isL = /\p{L}/u.test(s) ? 1 : 0;
  out.push(c.toString(36) + ':' + (re.test(s) ? 0 : 1) + isL);
}
process.stdout.write(out.join(','));
"""
    try:
        r = subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True)
    except Exception:
        sys.stderr.write('verify_node: skipped (no node)\n')
        return
    bad = checked = 0
    for item in r.stdout.split(','):
        k, v = item.split(':')
        c = int(k, 36)
        node_l = v[1] == '1'
        if node_l != (gc[c] in ('Lu', 'Ll', 'Lt', 'Lm', 'Lo')):
            continue  # the letter set changed between 15.1 and 16.0
        checked += 1
        word = v[0] == '1' and not (48 <= c <= 57)
        if word != bool(flags[c] & WORD):
            bad += 1
            print('word', hex(c))
    if bad:
        raise SystemExit('verify_node: %d differences' % bad)
    sys.stderr.write('verify_node: WORD agrees with node on %d code points\n' % checked)


def records(flags, up, lo, ti, cf, asc):
    """Unique records (flags, upper, lower, title, fold deltas or a special
    index, ASCII fold index) and the two-level index."""
    specials = []      # flat: per entry, lengths and code points of lower, upper, title, fold
    spec_ix = {}
    afold = ['']       # index 0: no ASCII part
    afold_ix = {'': 0}
    recs = []
    rec_ix = {}
    per = [0] * N
    for c in range(N):
        seqs = (lo[c], up[c], ti[c], cf[c])
        if all(len(s) == 1 for s in seqs):
            d = tuple(s[0] - c for s in seqs)
            sx = 0
        else:
            key = tuple(tuple(s) for s in seqs)
            if key not in spec_ix:
                spec_ix[key] = len(specials) + 1
                for s in seqs:
                    specials.append(len(s))
                    specials.extend(s)
            d = (0, 0, 0, 0)
            sx = spec_ix[key]
        a = asc.get(c, '') if c >= 128 else ''
        if a not in afold_ix:
            afold_ix[a] = len(afold)
            afold.append(a)
        rec = (flags[c],) + d + (sx, afold_ix[a])
        if rec not in rec_ix:
            rec_ix[rec] = len(recs)
            recs.append(rec)
        per[c] = rec_ix[rec]
    return recs, per, specials, afold


def two_level(per, shift):
    last = max(c for c in range(N) if per[c] != per[N - 1]) if per[N - 1] == 0 else N - 1
    top = (last >> shift) + 1
    blocks, bix, ix1 = [], {}, []
    size = 1 << shift
    for h in range(top):
        b = tuple(per[h * size:(h + 1) * size])
        if b not in bix:
            bix[b] = len(blocks)
            blocks.append(b)
        ix1.append(bix[b])
    ix2 = [x for b in blocks for x in b]
    return ix1, ix2


def rce_list(name, xs, ty='i64', per_line=24, pub=False):
    out = ('pub ' if pub else '') + 'const %s: List[%s] = [' % (name, ty)
    items = [str(x) for x in xs]
    lines = []
    for i in range(0, len(items), per_line):
        lines.append(', '.join(items[i:i + per_line]))
    if not lines:
        return out + ']\n'
    return out + '\n    ' + ',\n    '.join(lines) + ']\n'


def rce_text(t):
    o = '"'
    for ch in t:
        c = ord(ch)
        if ch == '"' or ch == '\\':
            o += '\\' + ch
        elif 32 <= c < 127:
            o += ch
        else:
            o += '\\u{%x}' % c
    return o + '"'


def main():
    flags, up, lo, ti, cf, asc, gc = build()
    if '--no-verify' not in sys.argv:
        verify(flags, up, lo, ti, cf, asc)
        verify_node(flags, gc)
    recs, per, specials, afold = records(flags, up, lo, ti, cf, asc)
    best = None
    for sh in range(4, 10):
        ix1, ix2 = two_level(per, sh)
        tot = len(ix1) + len(ix2)
        if best is None or tot < best[0]:
            best = (tot, sh, ix1, ix2)
    tot, sh, ix1, ix2 = best
    sys.stderr.write('records %d, specials %d ints, ascii folds %d, shift %d, index %d + %d\n'
                     % (len(recs), len(specials), len(afold), sh, len(ix1), len(ix2)))
    o = '# ---- data (generated by scripts/gen/str_data.py from Unicode 15.1.0; do not edit)\n'
    o += '# A code point c has the record STR_IX2[(STR_IX1[c >> STR_SHIFT] << STR_SHIFT) + (c & STR_MASK)]\n'
    o += '# (record 0 above the end of STR_IX1). Record fields: STR_RF flags (bits, in\n'
    o += '# order: ' + ', '.join(FLAG_NAMES[:8]) + ',\n# ' + ', '.join(FLAG_NAMES[8:]) + '),\n'
    o += '# STR_RL, STR_RU, STR_RT, STR_RC the deltas of the one-code-point lower, upper,\n'
    o += '# title and fold mappings, STR_RX the 1-based start in STR_SX of a mapping\n'
    o += '# longer than one code point (lengths and code points of lower, upper,\n'
    o += '# title, fold), STR_RA the index in STR_AF of the ASCII part of the NFKD form.\n\n'
    o += 'const STR_SHIFT: i64 = %d\n' % sh
    o += 'const STR_MASK: i64 = %d\n' % ((1 << sh) - 1)
    o += 'const STR_IX1N: i64 = %d\n\n' % len(ix1)
    o += '# Records and flags of the ASCII code points.\n'
    o += rce_list('STR_R7', per[:128], per_line=32) + '\n'
    o += rce_list('STR_A7', [recs[per[c]][0] for c in range(128)], per_line=16) + '\n'
    o += rce_list('STR_RF', [r[0] for r in recs]) + '\n'
    o += rce_list('STR_RL', [r[1] for r in recs]) + '\n'
    o += rce_list('STR_RU', [r[2] for r in recs]) + '\n'
    o += rce_list('STR_RT', [r[3] for r in recs]) + '\n'
    o += rce_list('STR_RC', [r[4] for r in recs]) + '\n'
    o += rce_list('STR_RX', [r[5] for r in recs]) + '\n'
    o += rce_list('STR_RA', [r[6] for r in recs]) + '\n'
    o += rce_list('STR_SX', specials) + '\n'
    o += rce_list('STR_AF', [rce_text(a) for a in afold], ty='text', per_line=12) + '\n'
    o += rce_list('STR_IX1', ix1, per_line=32) + '\n'
    o += rce_list('STR_IX2', ix2, per_line=32)
    p = os.path.join(ROOT, 'lib/text/str.rce')
    s = open(p, encoding='utf-8').read()
    k = s.index('# ---- data')
    open(p, 'w', encoding='utf-8').write(s[:k] + o)
    sys.stderr.write('data: %d lines\n' % o.count('\n'))


if __name__ == '__main__':
    main()
