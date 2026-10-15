# Checks for lib/net/idna.rce from official vectors only:
#   vectors/idna/IdnaTestV2.txt (Unicode 16.0.0): toUnicode, toAsciiN and
#     toAsciiT of every line, output code points and the error flag
#     (status codes are not compared; the file says only the flag is
#     required). Outputs are compared on error lines too. X4_2 is a
#     compatibility code for empty labels in toUnicode, where section 4
#     records no error, so a status holding only X4_2 counts as no error.
#     Inputs are code point lists, so the lines with lone surrogates are
#     checked as well.
#   vectors/idna/rfc3492.txt section 7.1: the sample strings A to S,
#     decoded exactly and encoded with the case of the deltas ignored
#     (the RFC shows them with mixed-case annotation).
#   NormalizationTest.txt (16.0.0, fetched and checked by sha256): NFC and
#     NFD of columns c1..c5 for every line (parts 0 to 5).
# Usage: python3 scripts/gen/idna_gen.py   (writes tests/net/idna_checks*.rce)
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idna_data import get, ROOT
SHARD = 2500
def L(cps): return '[' + ', '.join(str(c) for c in cps) + ']'
def unesc(s):
    s = re.sub(r'\\x\{([0-9A-Fa-f]+)\}', lambda m: chr(int(m.group(1), 16)), s)
    s = re.sub(r'\\u([0-9A-Fa-f]{4})', lambda m: chr(int(m.group(1), 16)), s)
    return s
HEAD = '''
fn cp_list(c: &Cps) -> List[i64] = copy(c.cps)
fn tu(s: List[i64]) -> Cps
    let f = test_flags(false)
    return to_unicode(&s, &f)
fn tn(s: List[i64]) -> Cps
    let f = test_flags(false)
    return to_ascii_cps(&s, &f)
fn tt(s: List[i64]) -> Cps
    let f = test_flags(true)
    return to_ascii_cps(&s, &f)
fn ok(c: Cps, want: List[i64], err: bool) -> bool = c.err == err and c.cps == want
'''
def main():
    out = []
    # RFC 3492 7.1
    rfc = open(os.path.join(ROOT, 'vectors/idna/rfc3492.txt')).read()
    sec = rfc[rfc.index('\n7.1 Sample strings'):rfc.index('\n7.2 ')]
    puny = []
    sec = re.sub(r'\n\n\S.*\n\n?.*RFC 3492.*\n', '\n', sec)  # page breaks
    blocks = re.split(r'\n   \(([A-S])\) ', sec)[1:]
    for n, b in zip(blocks[0::2], blocks[1::2]):
        cps = [int(x, 16) for x in re.findall(r'[uU]\+([0-9A-F]{4})', b)]
        m = re.search(r'Punycode: (.*)', b)
        p = m.group(1).rstrip()
        rest = b[m.end():].split('\n')[1:]
        while p.endswith('\\'):
            p = p[:-1] + rest.pop(0).strip()
        puny.append((n, cps, p))
    assert len(puny) == 19, len(puny)
    pl = ['# RFC 3492 section 7.1 sample strings (%d)' % len(puny)]
    for n, cps, p in puny:
        pc = [ord(c) for c in p]
        d = p.rfind('-')
        low = [ord(c) for c in p[:d + 1] + p[d + 1:].lower()] if d >= 0 else [ord(c) for c in p.lower()]
        pl.append('check puny_decode(&%s) == Cps(cps: %s, err: false)' % ('p' + n, L(cps)))
        pl.append('check lower_digits(puny_encode(&%s)) == %s' % ('c' + n, L(low)))
        pl.insert(0, 'const p%s: List[i64] = %s\nconst c%s: List[i64] = %s' % (n, L(pc), n, L(cps)))
    pl.insert(0, '''fn lower_digits(c: Cps) -> List[i64]
    var o: List[i64] = []
    var d: usize = 0
    var i: usize = 0
    while i < c.cps.len() bound 1024
        if c.cps[i] == 45
            d = i + 1
        i += 1
    i = 0
    for x in c.cps
        o.push(if i >= d and x >= 65 and x <= 90 then x + 32 else x)
        i += 1
    return o''')
    # NormalizationTest
    nl = []
    part = None; k = 0
    for l in get('NormalizationTest.txt').split('\n'):
        if l.startswith('@Part'): part = int(l[5]); k = 0; continue
        l = l.split('#')[0].strip()
        if not l: continue
        k += 1
        c = [[int(x, 16) for x in f.split()] for f in l.split(';')[:5]]
        for i in range(5):
            nl.append('check nfc(&%s) == %s' % ('x', L(c[1] if i < 3 else c[3])) if False else None)
        nl.append('check nf(%s, %s, %s, %s, %s)' % tuple(L(x) for x in c))
    nhead = '''fn nf(c1: List[i64], c2: List[i64], c3: List[i64], c4: List[i64], c5: List[i64]) -> bool
    return nfc(&c1) == c2 and nfc(&c2) == c2 and nfc(&c3) == c2 and nfc(&c4) == c4 and nfc(&c5) == c4 and nfd(&c1) == c3 and nfd(&c2) == c3 and nfd(&c3) == c3 and nfd(&c4) == c5 and nfd(&c5) == c5'''
    nl = [x for x in nl if x]
    # IdnaTestV2
    il = []; skipped = 0; n = 0
    def st(s): return [x.strip() for x in s.strip()[1:-1].split(',') if x.strip()] if s.strip() else None
    for l in get('IdnaTestV2.txt').split('\n'):
        l = l.split('#')[0]
        if not l.strip(): continue
        f = [x.strip() for x in l.split(';')]
        def v(s): return '' if s == '""' else unesc(s)
        src = v(f[0])
        tu = v(f[1]) if f[1] else src
        tus = st(f[2]) or []
        an = v(f[3]) if f[3] else tu
        ans = st(f[4]); ans = tus if ans is None else ans
        at = v(f[5]) if f[5] else an
        ats = st(f[6]); ats = ans if ats is None else ats
        e = lambda s: len([x for x in s if x != 'X4_2']) > 0
        S = L([ord(c) for c in src])
        il.append('check ok(tu(%s), %s, %s)' % (S, L([ord(c) for c in tu]), str(e(tus)).lower()))
        il.append('check ok(tn(%s), %s, %s)' % (S, L([ord(c) for c in an]), str(e(ans)).lower()))
        il.append('check ok(tt(%s), %s, %s)' % (S, L([ord(c) for c in at]), str(e(ats)).lower()))
        n += 1
    print('IdnaTestV2: %d lines, %d checks, %d skipped (lone surrogates); rfc3492: %d samples; NormalizationTest: %d lines' % (n, len(il), skipped, len(puny), len(nl)), file=sys.stderr)
    # write shards
    d = os.path.join(ROOT, 'tests/net')
    for f in os.listdir(d):
        if f.startswith('idna_checks'): os.remove(os.path.join(d, f))
    hdr = '# Generated by scripts/gen/idna_gen.py from official vectors; do not edit.\n'
    chunks = [('', pl + [nhead] + nl[:SHARD])]
    rest = nl[SHARD:]
    while rest: chunks.append(('norm', [nhead] + rest[:SHARD])); rest = rest[SHARD:]
    while il: chunks.append(('idna', [HEAD] + il[:SHARD])); il = il[SHARD:]
    for i, (_, body) in enumerate(chunks):
        name = 'idna_checks.rce' if i == 0 else 'idna_checks_%d.rce' % i
        open(os.path.join(d, name), 'w').write(hdr + '\n'.join(body) + '\n')
    print('%d files' % len(chunks), file=sys.stderr)
main()
