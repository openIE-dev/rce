# Usage: python3 -I scripts/gen/fmt_ext_gen.py
# Writes tests/fmt/fmt_checks_5.rce and the shards after it: checks for
# the format mini-language, float text, float parsing and regex captures
# of lib/fmt/fmt.rce. CPython 3.13 is the oracle (format(), repr(),
# decimal.Decimal, float() and re with re.ASCII, "$" written as "\Z"),
# seed 20261027; float parsing is also checked against the sample of the
# parse-number-fxx-test-data corpus in vectors/fmt/ (its f64 bits, which
# CPython's float() must give too). Derived vectors, not official ones,
# except the corpus sample.
import decimal, math, os, random, re, struct, sys
assert sys.version_info[:2] == (3, 13), sys.version
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
r = random.Random(20261027)
decimal.getcontext().prec = 2000

def q(s):
    o = ''
    for ch in s:
        if ch == '\\': o += '\\\\'
        elif ch == '"': o += '\\"'
        elif ch == '\n': o += '\\n'
        elif ch == '\t': o += '\\t'
        elif not ch.isprintable() or (ord(ch) & 0xFFFE) == 0xFFFE or 0xFDD0 <= ord(ch) <= 0xFDEF: o += '\\u{%x}' % ord(ch)
        else: o += ch
    return '"' + o + '"'

def fl(x):
    return repr(x)

def il(v):
    return '%d' % v if v != -2**63 else '0 - 9223372036854775807 - 1'

def lst(xs):
    return '[' + ', '.join('%d' % x for x in xs) + ']'

def tlst(xs):
    return '[' + ', '.join(q(x) for x in xs) + ']'

def bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]

def from_bits(b):
    return struct.unpack('<d', struct.pack('<Q', b))[0]

def tryfmt(v, s):
    try:
        return format(v, s)
    except (ValueError, OverflowError):
        return None

# ---------------- integers and text ----------------
def int_checks():
    out = []
    vals = [0, 1, -1, 7, -42, 255, 1234, -1234567, 65, 233, 0x1F600, 2**63 - 1, -2**63, 10**18, -999]
    for _ in range(6):
        vals.append(r.randint(-2**63, 2**63 - 1))
    specs = ['', 'd', '5', '<5', '>5', '^7', '=+8', '*^9', '+', ' ', '-', '05', '010,', '08_', ',', '_',
             'b', '#b', '#o', 'o', 'x', '#x', 'X', '#X', '_x', '#_b', '010_x', '#012_X', 'n', '+d', ' 5d',
             '0=10', 'x<10', '>>8', '<<8', '==9', '0<6', '<06', '^+012,', '#010b', '=^12_d', '.^9', 'é>6',
             '025', '+030,', ' ,']
    for v in vals:
        for s in specs:
            t = tryfmt(v, s)
            out.append('check ck_fi(%s, %s, %s)' % (il(v), q(s), q(t)) if t is not None else 'check ck_fi_bad(%s, %s)' % (il(v), q(s)))
    for v in [65, 233, 0x1F600, 0, 10, 0x10FFFF]:
        for s in ['c', '>3c', '05c', '<4c', '^5c']:
            out.append('check ck_fi(%s, %s, %s)' % (il(v), q(s), q(format(v, s))))
    bad = [(-1, 'c'), (0x110000, 'c'), (0xD800, 'c'), (65, '+c'), (65, '#c'), (65, ',c'), (5, ',b'), (5, ',x'), (5, ',n'), (5, '_n'),
           (5, '.2'), (5, '.2d'), (5, 'z'), (5, 'zd'), (5, 's'), (5, 'q'), (5, '.'), (5, '5.'), (5, ',_'), (5, '_,'), (5, 'xx'),
           (5, '<<<'), (5, 'é'), (5, '1000001'), (5, '.1000001f')]
    for v, s in bad:
        if v != 0xD800:
            assert tryfmt(v, s) is None or s in ('1000001', '.1000001f'), (v, s)
        out.append('check ck_fi_bad(%s, %s)' % (il(v), q(s)))
    # float types applied to integers (converted to binary64)
    for v in [0, 7, -42, 1234567, 2**53 + 1, 2**63 - 1, -2**63, 10**17 + 3]:
        for s in ['e', '.2f', '%', ',.3f', 'g', '.3e', '+.1%', '020,.2f', 'E', 'G', '.0f']:
            out.append('check ck_fi(%s, %s, %s)' % (il(v), q(s), q(format(v, s))))
    return out

def text_checks():
    out = []
    strs = ['', 'a', 'abc', 'héllo', '日本語', 'xxxxxxxxxx', '\U0001F600!']
    specs = ['', 's', '5', '<5', '>5', '^5', '^6', '*^7', 'é>4', '.2', '5.2', '>8.3s', '05', '010', '0>5', '.0', '1',
             '.10', '\U0001F600^9', '<0', '0^6.1']
    for v in strs:
        for s in specs:
            out.append('check ck_ft(%s, %s, %s)' % (q(v), q(s), q(format(v, s))))
    for s in ['=5', '+', ' ', '-', '#', 'z', ',', '_', 'd', '5d', '.', 'ss', '>5x']:
        assert tryfmt('ab', s) is None, s
        out.append('check ck_ft_bad("ab", %s)' % q(s))
    return out

# ---------------- float text ----------------
def rand_double():
    k = r.randrange(10)
    if k == 0:
        return from_bits(r.getrandbits(52))                     # subnormal
    if k == 1:
        return float(r.randint(-10**7, 10**7)) / (10 ** r.randint(0, 6))
    if k == 2:
        return float(r.randint(-2**60, 2**60))
    e = r.randint(1, 2046)
    x = from_bits((e << 52) | r.getrandbits(52))
    return -x if r.random() < 0.3 else x

FSPECIAL = [0.0, -0.0, 1.0, -1.0, 0.1, 0.5, 1.5, 2.5, -2.5, 0.125, 0.375, 2.675, 1e16, 1e15, 9999999999999998.0,
            123456789.0, 1e-5, 1e-4, 0.0001234, 9.995, 9.9999, 99.995, 0.045, 1e22, 1e23, 5e-324, 1e-323,
            2.2250738585072014e-308, 2.225073858507201e-308, 1.7976931348623157e308, 1 / 3, 2 / 3, 100.0,
            1234567.891, 0.999999999, 1e-7, 123.456, 0.3, 1e100, 5e-310, 8.5, 9.5, 0.05, 1e300 * 1.5,
            2**53 + 0.0, 2**63 + 0.0, 4.35, 1e21, 1e-300, 0.6, 7.0e-10]

FSPECS = ['', 'e', '.0e', '.3e', '.17e', 'E', 'f', '.0f', '.1f', '.2f', '.10f', '.20f', 'F', 'g', '.0g', '.1g', '.3g',
          '.10g', '.17g', '#g', '#.3g', 'G', '.3', '.12', '#', '%', '.1%', 'n', '.3n', ',', '_', ',.2f', '15.3f',
          '<15.3e', '^15', '*>15.2f', '+.2e', ' g', '015.2f', '015,.2f', 'z.1f', 'z', '#.0f', '#.0e', '=+12.3f',
          '0=12,', '-^20.5g', ',.0f', '_.3f', '.25e', '.40f', '#.10', 'z.0e', '+015_.4e']

def float_spec_checks():
    out = []
    vals = list(FSPECIAL)
    for _ in range(25):
        vals.append(rand_double())
    for v in vals:
        specs = FSPECS if v in FSPECIAL[:20] else r.sample(FSPECS, 14)
        for s in specs:
            t = tryfmt(v, s)
            assert t is not None, (v, s)
            if len(t) > 400:
                continue
            out.append('check ck_ff(%s, %s, %s)' % (fl(v), q(s), q(t)))
    for v in [1.797693134862316e306, -1.7976931348623157e308, 1.7976931348623155e306]:
        for s in ['%', '010%', '+012,.2%', '.0%', '<9%']:
            out.append('check ck_ff(%s, %s, %s)' % (fl(v), q(s), q(format(v, s))))
    for s in ['d', 'x', 'c', 's', 'b', 'o', ',n', ',_', '.f', 'X', '.1000001f', '1000001']:
        out.append('check ck_ff_bad(1.5, %s)' % q(s))
    return out

def float_api_checks():
    out = []
    vals = list(FSPECIAL)
    for _ in range(80):
        vals.append(rand_double())
    for v in vals:
        p = r.randint(0, 25)
        out.append('check ck_frepr(%s, %s)' % (fl(v), q(repr(v))))
        out.append('check ck_ffix(%s, %d, %s)' % (fl(v), p, q(format(v, '.%df' % p))) if abs(v) < 1e40 else 'check ck_fexp(%s, %d, %s)' % (fl(v), p, q(format(v, '.%de' % p))))
        out.append('check ck_fexp(%s, %d, %s)' % (fl(v), p, q(format(v, '.%de' % p))))
    for v in vals[:60]:
        ex = format(decimal.Decimal(v), 'f')
        if len(ex) <= 400:
            out.append('check ck_fexact(%s, %s)' % (fl(v), q(ex)))
    out.append('check ck_fexact(5e-324, %s)' % q(format(decimal.Decimal(5e-324), 'f')))
    out.append('check ck_fexact(1.7976931348623157e308, %s)' % q(format(decimal.Decimal(1.7976931348623157e308), 'f')))
    out.append('check ck_fixbad(1.5, -1) and ck_fixbad(1.5, 1000001) and ck_expbad(1.5, -2) and ck_fixbad(1e300, 999990)')
    return out

# ---------------- float parsing ----------------
def parse_checks():
    out = []
    for line in open(os.path.join(root, 'vectors/fmt/parse-number-fxx-sample.txt')):
        name, h16, h32, h64, s = line.split()
        b = int(h64, 16)
        x = from_bits(b)
        assert bits(float(s)) == b, (s, h64)
        if math.isinf(x):
            out.append('check ck_fpo(%s)' % q(s))
        else:
            out.append('check ck_fp(%s, %s)' % (q(s), fl(x)))
    return out

def parse_extra_checks():
    out = []
    D = decimal.Decimal
    # halfway points between neighbouring binary64 values, and just
    # above and below them
    for _ in range(40):
        x = abs(rand_double())
        if x == 0 or math.isinf(math.nextafter(x, math.inf)):
            continue
        y = math.nextafter(x, math.inf)
        mid = (D(x) + D(y)) / 2
        m = format(mid, 'f')
        if len(m) > 900:
            m = format(mid, 'e')
        for t in [m, m + '0000001' if '.' in m and 'e' not in m else None]:
            if t is None:
                continue
            out.append('check ck_fp(%s, %s)' % (q(t), fl(float(t))))
        lo = format(mid - D(10) ** (mid.adjusted() - 790), 'e')
        out.append('check ck_fp(%s, %s)' % (q(lo), fl(float(lo))))
    # round trips of repr and of the exact decimal value
    for _ in range(150):
        x = rand_double()
        out.append('check ck_fp(%s, %s)' % (q(repr(x)), fl(x)))
    for _ in range(12):
        x = rand_double()
        e = format(D(x), 'f')
        out.append('check ck_fp(%s, %s)' % (q(e), fl(x)))
    forms = ['0', '-0', '+0', '.5', '5.', '-.5e1', '1E5', '1e+5', '1e-5', '00012.5000', '0.000', '-0e10', '1e-400',
             '-1e-400', '1e309', '-1e309', '1.7976931348623157e308', '1.7976931348623158e308', '1.7976931348623159e308',
             '2.4703282292062327e-324', '2.4703282292062328e-324', '4.9e-324', '2.2250738585072011e-308',
             '9007199254740993', '9007199254740993.0000000000001', '123456789012345678901234567890', '1' * 400,
             '0.' + '0' * 300 + '1', '1' + '0' * 308, '1' + '0' * 309, '0.1e-322', '1e22', '1e23', '8.5e-325',
             '1e' + '9' * 30, '1e-' + '9' * 30, '0e' + '9' * 30, '7.2057594037927933e16', '3.0e-0']
    for t in forms:
        v = float(t)
        out.append('check ck_fpo(%s)' % q(t) if math.isinf(v) else 'check ck_fp(%s, %s)' % (q(t), fl(v)))
    for t, at in [('', 0), ('-', 0), ('.', 0), ('e5', 0), ('1e', 1), ('1e+', 1), ('--1', 0), ('1..2', 2), (' 1', 0),
                  ('1 ', 1), ('inf', 0), ('nan', 0), ('1_000', 1), ('0x10', 1), ('1.5.', 3), ('+.e1', 0), ('1e5x', 3),
                  ('é', 0), ('1é', 1)]:
        out.append('check ck_fpbad(%s, %d)' % (q(t), at))
    for s, at, v, end in [('1.5abc', 0, 1.5, 3), ('x1.5', 1, 1.5, 4), ('1e', 0, 1.0, 1), ('1e+x', 0, 1.0, 1),
                          ('-', 0, 0.0, 0), ('12.5e3kg', 0, 12500.0, 6), ('a,-0.25,b', 2, -0.25, 7), ('5', 9, 0.0, 1),
                          ('é2.5', 2, 2.5, 5), ('7', -3, 7.0, 1)]:
        out.append('check ck_fscan(%s, %d, %s, %d)' % (q(s), at, fl(v), end))
    out.append('check ck_fscan_ovf("x-1e999!", 1, -1.7976931348623157e308, 7)')
    return out

# ---------------- regex ----------------
def py_pat(p):
    return re.compile(p.replace('$', '\\Z'), re.ASCII)

def bspan(s, a, b):
    if a < 0:
        return [-1, -1]
    return [len(s[:a].encode()), len(s[:b].encode())]

def slots(m, s):
    if m is None:
        return []
    o = []
    for k in range(m.re.groups + 1):
        o += bspan(s, *m.span(k))
    return o

RP = ['(a)(b)?', '(a|ab)(c|bcd)(d*)', '(?P<user>\\w+)@(?P<host>\\w+)\\.com', '(a*)*', '(a*)+', '(a|b)*', '(a|)*',
      '(a?)+?', 'a*?b', '(a+?)(a*)', '<(.+)>', '<(.+?)>', '(\\d+)-(\\d+)', '((a)|b)+', '(?:(a)|b)*c', 'x*',
      '\\b\\w+\\b', '\\B\\w\\B', '^(\\w+)\\s', '(\\w+)$', '(é+)(.)', '([^,]*),?', '(\\s*)(\\S+)',
      '(?P<y>\\d{4})-(?P<m>\\d\\d)', 'a{2,3}?', '(ab){1,2}?(ab)?', '[\\D]+', '[\\W\\d]+', '\\x61+', '\\u00e9',
      '(a)|(b)|(c)', '((((a))))', '(a)(?:b)(c)', '(|a)+', '()', '(a*)(a*)', '(a*?)(a*)', 'é|.', '(.)\\1' if False else '(.)(.)',
      '\\Aa|b\\Z', '(a|b)*?b', '[a-c]{,2}', '(?:a|(b))+', '(a)?(b)?(c)?', '\\101\\0', '[à-ÿ]+', '.{2,}?x',
      '(x?)*y', '(a+|b+)*c']
RS = ['', 'a', 'ab', 'abcd', 'aaa', 'xaby', 'me@host.com x@y.com', '<a><b>', '12-345 6-7', 'bab c', 'éée',
      'a,b,,c', '  foo bar', '2026-10 1999-12', 'abab', 'b', 'aab', 'cbac', 'abc d', 'xyx', 'àéÿ!',
      'A\u0000', 'xxyy', 'aaab', 'bbac']

def regex_checks():
    out = []
    for p in RP:
        py = py_pat(p)
        for s in RS:
            out.append('check ck_rx(%s, %s, %s, %s, %s)' % (q(p), q(s), lst(slots(py.search(s), s)),
                                                            lst(slots(py.match(s), s)), lst(slots(py.fullmatch(s), s))))
    return out

def regex_all_checks():
    out = []
    for p in RP:
        py = py_pat(p)
        for s in RS:
            if r.random() < 0.6:
                continue
            flat = []
            for m in py.finditer(s):
                flat += slots(m, s)
            out.append('check ck_rall(%s, %s, %s)' % (q(p), q(s), lst(flat)))
    for p, s, at in [('a', 'banana', 2), ('\\ba', 'a ab ba', 1), ('^a', 'aa', 1), ('é.', 'xéyéz', 2),
                     ('(\\w)', 'ab', 5), ('', 'ab', 1)]:
        cp = len(s.encode()[:at].decode('utf-8', 'ignore'))
        b = s.encode()
        while at < len(b) and 128 <= b[at] < 192:
            at += 1
        cp = len(b[:min(at, len(b))].decode())
        out.append('check ck_rat(%s, %s, %d, %s)' % (q(p), q(s), at, lst(slots(py_pat(p).search(s, cp), s))))
    # named groups
    for p, names in [('(?P<user>\\w+)@(?P<host>\\w+)', ['user', 'host', 'none']), ('(a)(?P<b>b)', ['b', 'a'])]:
        py = py_pat(p)
        idx = [py.groupindex.get(n, -1) for n in names]
        out.append('check ck_rnames(%s, %s, %s, %d)' % (q(p), tlst(names), lst(idx), py.groups))
    # resumable search fed in chunks
    for p in RP[::3]:
        py = py_pat(p)
        for s in RS[::2]:
            k = r.randint(1, 3)
            out.append('check ck_rscan(%s, %s, %d, %s)' % (q(p), q(s), k, lst(slots(py.search(s), s))))
    return out

TPL = [('(a)(b)?', 'abxa', '[\\1|\\2]'), ('(\\w+)@(\\w+)', 'me@host and you@there', '\\2 at \\1'),
       ('(?P<w>\\w+)', 'ab cd', '<\\g<w>>'), ('x*', 'abxd', '-'), ('', 'abc', '.'), ('a|(b)', 'abab', '{\\g<1>}'),
       ('(a)', 'aa', '\\g<0>\\g<1>\\n\\t\\\\'), ('(\\d+)', 'a1b22c333', '\\101\\0\\08'), ('b', 'abc', '\\&\\-\\%'),
       ('(é)', 'xéy', '\\1\\1'), ('\\s+', ' a  b ', '_'), ('(a)|(b)', 'ab', '[\\1\\2]'),
       ('(?P<x>a)(?P<y>b)?', 'aab', '\\g<y>\\g<x>'), ('a', 'aaa', '\\g<0>\\g<0>'), ('(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)', 'abcdefghijklm', '\\11\\10\\1')]

def replace_checks():
    out = []
    for p, s, t in TPL:
        py = py_pat(p)
        for c in [0, 1, 2, -1]:
            out.append('check ck_rsub(%s, %s, %s, %d, %s)' % (q(p), q(s), q(t), c, q(py.sub(t, s, count=c) if c >= 0 else s)))
    for p, t in [('(a)', '\\g<x'), ('(a)', '\\g<>'), ('(a)', '\\g<1a>'), ('(a)(b)', '\\3'), ('(a)', '\\g<9>'), ('(a)', '\\q'),
                 ('(a)', 'x\\'), ('(a)', '\\g<unknown>'), ('(a)', '\\g'), ('(a)', '\\400')]:
        try:
            py_pat(p).sub(t, 'a'); raise SystemExit('expected a refusal: ' + t)
        except (re.error, IndexError):
            pass
        out.append('check ck_rsub_bad(%s, %s)' % (q(p), q(t)))
    SP = [('x*', 'axbc'), (',', 'a,b,,c'), ('\\s+', '  a b  c '), ('', 'abc'), ('\\b', 'ab cd'), ('(?:x)', 'axbx'),
          ('[,;]\\s*', 'a, b;c'), ('é', 'aébé'), ('z', 'abc'), ('a|b', 'cabd')]
    for p, s in SP:
        py = py_pat(p)
        for m in [0, 1, 2, -1]:
            out.append('check ck_rsplit(%s, %s, %d, %s)' % (q(p), q(s), m, tlst(py.split(s, maxsplit=m) if m >= 0 else [s])))
    return out

def regex_bad_checks():
    out = []
    for p in ['a**', 'a*??', '^*', '\\b+', 'x{2}{3}', '[\\d-z]', '[a-\\d]', '\\1', '\\8', '(?P<1a>x)', 'a|*',
              '(?P<n>a)(?P<n>b)', '(?P<>a)', '\\x4', '\\u12', '\\U00110000', '[\\A]', '(?<n>a)', '\\z', '(?P=n)', 'a{3,2}']:
        try:
            py_pat(p); raise SystemExit('expected a refusal: ' + p)
        except re.error:
            pass
        out.append('check ck_rbad(%s)' % q(p))
    # accepted by CPython, refused here (not regular, or a literal that
    # must be escaped)
    for p in ['(?P<n>a)(?P=n)', '(?=a)', '(?!a)', '(?<=a)b', '(?i)a', '(?#c)a', 'a{}', 'a{x}', ']', '}', 'a*+', '(?>a)']:
        py_pat(p)
        out.append('check ck_rbad(%s)' % q(p))
    return out

HEAD = {
'5': '''# Checks for lib/fmt/fmt.rce, part 5: the format mini-language for
# integers and text against CPython 3.13 format(), generated by
# scripts/gen/fmt_ext_gen.py.
''',
'float': '''# Checks for lib/fmt/fmt.rce, part %s: float text (format(), repr,
# exact decimal values) against CPython 3.13, generated by
# scripts/gen/fmt_ext_gen.py.
''',
'parse': '''# Checks for lib/fmt/fmt.rce, part %s: float parsing against the
# parse-number-fxx-test-data sample in vectors/fmt/ and CPython 3.13
# float(), generated by scripts/gen/fmt_ext_gen.py.
''',
'regex': '''# Checks for lib/fmt/fmt.rce, part %s: regex captures, find_all,
# replace, split and the resumable search against CPython 3.13 re
# (re.ASCII, "$" as "\\\\Z"), generated by scripts/gen/fmt_ext_gen.py.
'''}

HELPERS = open(os.path.join(root, 'scripts/gen/fmt_ext_helpers.rce')).read()

def write(part, kind, lines):
    path = os.path.join(root, 'tests/fmt/fmt_checks_%d.rce' % part)
    h = HEAD[kind] if kind == '5' else HEAD[kind] % part
    open(path, 'w').write(h + '\n' + HELPERS + '\n# ---- generated by scripts/gen/fmt_ext_gen.py\n' + '\n'.join(lines) + '\n')

def shards(lines, n):
    return [lines[i:i + n] for i in range(0, len(lines), n)]

part = 5
write(part, '5', int_checks() + text_checks()); part += 1
for chunk in shards(float_spec_checks() + float_api_checks(), int(os.environ.get('FLOAT_SHARD', '700'))):
    write(part, 'float', chunk); part += 1
for chunk in shards(parse_checks() + parse_extra_checks(), int(os.environ.get('PARSE_SHARD', '700'))):
    write(part, 'parse', chunk); part += 1
for chunk in shards(regex_checks() + regex_all_checks() + replace_checks() + regex_bad_checks(), int(os.environ.get('REGEX_SHARD', '700'))):
    write(part, 'regex', chunk); part += 1
print('wrote parts 5..%d' % (part - 1))
