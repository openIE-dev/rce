#!/usr/bin/env python3
"""Data section of lib/ui/rules.rce and tests/ui/rules_checks_contrast.rce.

  python3 scripts/gen/rules_gen.py refresh WCAG AXE   # re-read the sources,
        rewrite the four files of vectors/ui named below
  python3 scripts/gen/rules_gen.py                    # data section and the
        contrast checks from those files

WCAG is the guidelines/ directory of a checkout of w3c/wcag, AXE the axe.js
of the axe-core npm package (run with node). Sources (vectors/SOURCE.md,
"ui.rules"):
  vectors/ui/wcag22-sc.tsv       every WCAG 2.2 success criterion of level A
      or AA, in document order (id, anchor, level, title), from
      guidelines/index.html and guidelines/sc/**
  vectors/ui/input-purposes.txt  the 53 input purposes of WCAG 2.2 section 7
      (guidelines/input-purposes.html)
  vectors/ui/aria-roles.tsv      the WAI-ARIA roles of axe-core's standards
      table (non-abstract, not deprecated, no doc-* or graphics-* module
      roles): role, type, required attributes, required context, name
      required, name from content
  vectors/ui/contrast-axe.tsv    colour pairs with axe-core's contrast ratio
      (axe.commons.color.getContrast) and the WCAG formula evaluated with
      60-digit decimal arithmetic
The linearisation table LIN is computed here from the WCAG 2.2 definition of
relative luminance with 60-digit decimal arithmetic: LIN[v] is the sRGB
channel value v/255 linearised, times 10^12, rounded to the nearest integer,
so each entry is within 1/2 of the exact value.
"""
import json, os, random, re, subprocess, sys
from decimal import Decimal, getcontext
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../scripts'))
from positional import read  # noqa: E402  (positional-cli.md PC-1)
getcontext().prec = 60
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.join(ROOT, 'vectors/ui')
LIB = os.path.join(ROOT, 'lib/ui/rules.rce')
T = os.path.join(ROOT, 'tests/ui/rules_checks_contrast.rce')
SCALE = 10 ** 12


def lin_exact(v):
    c = Decimal(v) / Decimal(255)
    if c <= Decimal('0.04045'):
        return c / Decimal('12.92')
    return ((c + Decimal('0.055')) / Decimal('1.055')) ** Decimal('2.4')


def lum_exact(rgb):
    r, g, b = (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255
    return Decimal('0.2126') * lin_exact(r) + Decimal('0.7152') * lin_exact(g) + Decimal('0.0722') * lin_exact(b)


def ratio_exact(a, b):
    la, lb = lum_exact(a), lum_exact(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + Decimal('0.05')) / (lo + Decimal('0.05'))


def refresh(wcag, axe):
    idx = open(os.path.join(wcag, 'index.html')).read()
    p = g = s = 0
    rows = []
    for m in re.finditer(r'<section[^>]*(class="(principle|guideline)"|data-include="(sc/[^"]+)")[^>]*>', idx):
        if m.group(2) == 'principle':
            p, g = p + 1, 0
        elif m.group(2) == 'guideline':
            g, s = g + 1, 0
        else:
            s += 1
            f = open(os.path.join(wcag, m.group(3))).read()
            anchor = re.search(r'<section id="([^"]+)"', f).group(1)
            title = re.sub(r'<[^>]+>', '', re.search(r'<h4>(.*?)</h4>', f, re.S).group(1)).strip()
            lv = re.search(r'<p class="conformance-level">([^<]*)</p>', f)
            lv = lv.group(1).strip() if lv else ''
            if lv in ('A', 'AA'):
                rows.append('%d.%d.%d\t%s\t%s\t%s' % (p, g, s, anchor, lv, title))
            elif not lv:
                assert 'Obsolete' in title, title  # 4.1.1 Parsing
    open(os.path.join(V, 'wcag22-sc.tsv'), 'w').write('\n'.join(rows) + '\n')
    ip = open(os.path.join(wcag, 'input-purposes.html')).read()
    toks = re.findall(r'<li>\s*<code[^>]*>([a-z0-9-]+)</code>', ip)
    open(os.path.join(V, 'input-purposes.txt'), 'w').write('\n'.join(toks) + '\n')
    js = r'''
const axe = require(process.argv[1]);
const R = axe.utils.getStandards().ariaRoles, out = [];
for (const [r, d] of Object.entries(R).sort()) {
  if (d.type === 'abstract' || d.deprecated || /^(doc|graphics)-/.test(r)) continue;
  out.push([r, d.type, (d.requiredAttrs || []).join(','), (d.requiredContext || []).join(','),
            d.accessibleNameRequired ? 1 : 0, d.nameFromContent ? 1 : 0].join('\t'));
}
const C = axe.commons.color, pairs = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const hex = n => [(n >> 16) & 255, (n >> 8) & 255, n & 255];
const con = pairs.map(([f, b]) => { const [r1, g1, b1] = hex(f), [r2, g2, b2] = hex(b);
  return C.getContrast(new C.Color(r2, g2, b2, 1), new C.Color(r1, g1, b1, 1)); });
process.stdout.write(JSON.stringify({ version: axe.version, roles: out, con }));
'''
    pairs = contrast_pairs()
    res = json.loads(subprocess.run(['node', '-e', js, axe], input=json.dumps(pairs), capture_output=True, text=True, check=True).stdout)
    open(os.path.join(V, 'aria-roles.tsv'), 'w').write('\n'.join(res['roles']) + '\n')
    o = ['# fg\tbg\taxe-core %s getContrast (shortest round-trip double)\tWCAG 2.2 formula, 60-digit decimal (30 shown)' % res['version']]
    for (f, b), c in zip(pairs, res['con']):
        o.append('%06x\t%06x\t%r\t%s' % (f, b, c, format(ratio_exact(f, b), '.30f')))
    open(os.path.join(V, 'contrast-axe.tsv'), 'w').write('\n'.join(o) + '\n')


def contrast_pairs():
    """Named pairs, the gray pairs nearest 4.5 and 3 on white and black, and
    300 pairs from seed 20261031."""
    ps = [(0x000000, 0xffffff), (0xffffff, 0x000000), (0x777777, 0x777777), (0x777777, 0xffffff), (0x767676, 0xffffff),
          (0x949494, 0xffffff), (0x959595, 0xffffff), (0x548cf2, 0x262930), (0xffffff, 0x548cf2), (0xcccccc, 0xffffff),
          (0x0a0a0a, 0x0b0b0b), (0x0b0b0b, 0x0a0a0a), (0xff0000, 0xffffff), (0x00ff00, 0x000000), (0x0000ff, 0xffffff)]
    for bg in (0xffffff, 0x000000):
        for k in (Decimal('4.5'), Decimal(3)):
            best = sorted(range(256), key=lambda v: abs(ratio_exact(v * 0x010101, bg) - k))[:2]
            ps += [(v * 0x010101, bg) for v in best]
    r = random.Random(20261031)
    for _ in range(300):
        ps.append((r.randrange(1 << 24), r.randrange(1 << 24)))
    out = []
    for x in ps:
        if x not in out:
            out.append(x)
    return out


def lit(xs):
    return '[' + ', '.join(xs) + ']'


def q(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def main():
    w, ops = read('scripts/gen/rules_gen.py', {'refresh': 0}, 'WCAG AXE', 0, 2)
    if 'refresh' in w:
        refresh(ops[0], ops[1])
    lin = [int((lin_exact(v) * SCALE).to_integral_value()) for v in range(256)]
    for v in range(256):
        assert abs(Decimal(lin[v]) - lin_exact(v) * SCALE) <= Decimal('0.5')
    roles = [l.split('\t') for l in open(os.path.join(V, 'aria-roles.tsv')).read().split('\n') if l]
    purposes = [l for l in open(os.path.join(V, 'input-purposes.txt')).read().split('\n') if l]
    o = '# ---- data (generated by scripts/gen/rules_gen.py; do not edit below)\n'
    o += ('# LIN[v]: the sRGB channel value v/255 linearised as in the WCAG 2.2\n'
          '# definition of relative luminance, times 10^12, rounded to the nearest\n'
          '# integer (60-digit decimal arithmetic); each entry is within 1/2 of the\n'
          '# exact value.\n')
    o += 'pub const LIN: [i64; 256] = %s\n\n' % lit(str(x) for x in lin)
    o += ('# WAI-ARIA roles (vectors/ui/aria-roles.tsv, from axe-core\'s standards\n'
          '# table): name, type, required states and properties, required context\n'
          '# roles, whether a name is required, whether the name comes from content.\n')
    n = len(roles)
    o += 'pub const ROLE_COUNT: i64 = %d\n' % n
    o += 'pub const ROLE_NAME: [text; %d] = %s\n' % (n, lit(q(r[0]) for r in roles))
    o += 'pub const ROLE_TYPE: [text; %d] = %s\n' % (n, lit(q(r[1]) for r in roles))
    o += 'pub const ROLE_REQ: [text; %d] = %s\n' % (n, lit(q(r[2]) for r in roles))
    o += 'pub const ROLE_CTX: [text; %d] = %s\n' % (n, lit(q(r[3]) for r in roles))
    o += 'pub const ROLE_NAME_REQ: [bool; %d] = %s\n' % (n, lit('true' if r[4] == '1' else 'false' for r in roles))
    o += 'pub const ROLE_NAME_CONTENT: [bool; %d] = %s\n\n' % (n, lit('true' if r[5] == '1' else 'false' for r in roles))
    o += '# The input purposes of WCAG 2.2 section 7 (vectors/ui/input-purposes.txt).\n'
    o += 'pub const PURPOSES: [text; %d] = %s\n' % (len(purposes), lit(q(p) for p in purposes))
    s = open(LIB).read()
    k = s.index('# ---- data')
    open(LIB, 'w').write(s[:k] + o)
    # contrast checks against axe-core
    rows = [l.split('\t') for l in open(os.path.join(V, 'contrast-axe.tsv')).read().split('\n') if l and not l.startswith('#')]
    t = ['# generated by scripts/gen/rules_gen.py; do not edit',
         '# Contrast against axe-core (vectors/ui/contrast-axe.tsv): ax(fg, bg, m)',
         '# holds when the ratio axe-core computes, m millionths truncated, lies in',
         '# the interval ui.rules can prove; at45/at3 are the 4.5 : 1 and 3 : 1',
         '# decisions, expected from axe-core\'s ratio (axe-core decides with >,',
         '# WCAG with >=; no pair below is within 10^-9 of a threshold).',
         'fn hx(s: &text) -> i64 = parse_color("#" + s)',
         'fn ax(f: text, b: text, m: i64) -> bool',
         '    let lo = ratio_micro_lo(hx(&f), hx(&b))',
         '    let hi = ratio_micro_hi(hx(&f), hx(&b))',
         '    return lo <= m + 1 and m <= hi',
         'fn at45(f: text, b: text) -> text = word(contrast_at_least(hx(&f), hx(&b), 9, 2))',
         'fn at3(f: text, b: text) -> text = word(contrast_at_least(hx(&f), hx(&b), 3, 1))',
         'fn word(d: i64) -> text',
         '    if d == 1',
         '        return "meets"',
         '    if d == 0',
         '        return "fails"',
         '    return "undecided"']
    for f, b, axr, dec in rows:
        a, d = float(axr), Decimal(dec)
        m = int(Decimal(axr) * 10 ** 6)
        for k in (Decimal('4.5'), Decimal(3)):
            assert abs(d - k) > Decimal('1e-9'), (f, b)
            assert (a > float(k)) == (d >= k), (f, b)  # axe-core and the decimal formula agree
        assert abs(Decimal(axr) - d) < Decimal('1e-12'), (f, b)
        t.append('check ax(%s, %s, %d) == true' % (q(f), q(b), m))
        t.append('check at45(%s, %s) == %s' % (q(f), q(b), q('meets' if a > 4.5 else 'fails')))
        t.append('check at3(%s, %s) == %s' % (q(f), q(b), q('meets' if a > 3 else 'fails')))
    open(T, 'w').write('\n'.join(t) + '\n')
    sys.stderr.write('rules_gen: %d roles, %d purposes, %d contrast pairs\n' % (n, len(purposes), len(rows)))


if __name__ == '__main__':
    main()
