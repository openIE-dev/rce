#!/usr/bin/env python3
"""Vectors and checks for lib/text/str.rce.

  python3 scripts/gen/str_gen.py --refresh   recompute every expected value
  python3 scripts/gen/str_gen.py             write tests/text/str_checks*.rce

--refresh needs the three references and writes vectors/text/str_vectors.json:
  - CPython 3.13 with Unicode 15.1.0 (this interpreter): str and bytes methods,
    re for the overlapping matches of StrFinder;
  - Django 5.2.18 (pip install django==5.2.18), django.utils.text.slugify;
  - Node 22 with change-case 5.4.4 (npm install change-case@5.4.4, found
    through NODE_PATH or the current directory): split, camelCase, pascalCase,
    snakeCase, kebabCase, constantCase, capitalCase.
Every case is generated from fixed pools and a seeded generator, so a refresh
with the same references writes the same file. A case whose expected value is
not a reference result (an argument CPython refuses, the byte-offset rules of
an empty needle, the size limit) is marked "derived" and computed here from the
rule stated in str.rce.

The default mode reads the JSON only, so the checks can be rewritten without
the references."""
import hashlib, json, os, random, re, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
VEC = os.path.join(ROOT, 'vectors/text/str_vectors.json')
OUT = os.path.join(ROOT, 'tests/text')
LIMIT = 1000000

# ---------------------------------------------------------------- pools

ASCII = [
    '', 'a', 'A', 'abc', 'Hello, World!', "hello world's", "they're bill's friends",
    '  spaced  out  ', '\tTab\tSeparated\t', 'line1\nline2\r\nline3\rline4', 'a\n\nb\n',
    'UPPER lower Title', 'camelCaseWord', 'snake_case_word', 'kebab-case-word',
    'HTTPServerError', 'XMLHttpRequest', 'getHTTPResponseCode', 'version 1.2.3', 'v2Beta3',
    'a1b2c3', '123abc', 'ABC123def', '+42', '-17', '0x1F', '...', 'a.b.c', 'aaa', 'aaaa',
    'abab', 'abababababab', 'mississippi', 'the quick brown fox jumps over the lazy dog',
    '\x1c\x1d\x1e\x1f', '\x0b\x0c', '\x00\x01\x7f', 'x\x00y', '_private', '__dunder__',
    'Mixed_Case-and spaces', 'ALLCAPS', 'iPhone', 'eBay', 'McDonald', "O'Neil", 'a-b_c d.e/f',
    '--dashes--', '__under__', '  ', '\n', '\r\n', 'tab\there', 'col1\tcol2\tcol3\n\tx\ty',
]
WIDE = [
    'é', 'café', 'naïve', 'Straße', 'ß', 'İstanbul', 'ıi',
    'ǅungla', 'ǆ', 'Ǆ', 'ŉ', 'ǰ', 'ΐ', 'ᾳ', 'ᾼ',
    'ΣΑΣ', 'σας', 'ΌΣΟΣ', 'Α\'Σ',
    'Σ', 'AΣ', 'AΣb', 'AΣ\'', 'A­Σ', 'ΣΣΣ', 'a Σ b',
    'ὈΔΥΣΣΕΎΣ', 'Привет, мир',
    'ПРИВЕТ', '日本語テキスト', '한국어',
    'עברית', 'العربية', 'हिन्दी',
    '\U0001f600 emoji \U0001f44d\U0001f3fd', '\U00010400\U00010428 Deseret', '\U0001e900\U0001e922 Adlam',
    '\U0001d400\U0001d401 math', 'ＡＢＣ ａｂｃ', 'K Ω Å',
    'ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ', 'Ⅻ ⅻ ① ² ½ ٣ ۴ 五 〇',
    'á combining', 'ͅ', 'x­y', 'it’s', ' nbsp em　',
    '\u0085next line para', '᠎​﻿', '‍‎', ' private',
    '͸ unassigned', '￿', '\U0010ffff', 'Ꭰꭰᏸ Cherokee', 'აბგ Georgian',
    'Ꙋ ꙋ', 'ǈ ǉ', 'Ὠ ᾨ ᾠ', 'ﬓ', 'և',
    'ÀÉÎÕÜŸ', 'ẞ', 'ⱥȺ', 'Hello 世界', 'Zürich-Straße 12',
    '¿Qué tal?', 'İIıi', 'straße\tSTRASSE', '　　x　',
]
# characters for random texts: ASCII, the characters above, case and class edges
POOL = sorted(set(''.join(ASCII + WIDE)) | set(
    'aAzZ09 _-.,;:!?\'"()[]{}\\/@#$%^&*+=<>|~`\t\n\r\x0b\x0c\x1c\x1f\x7f'
    'àéîõüÿÀÉÎÕÜŸßŉǰĳıİſ'
    'ǄǅǆǇǈǉΐΰᾳῼᾈΣσςΑα'
    'АаЀѐևﬓაᲐᎠꭰ́̇ͅ­’ʼ'
    '٣²①½Ⅻ五〇  　\u0085  ᠎​﻿‍'
    '漢あカ한\U0001f600\U0001f3fd\U00010400\U00010428\U0001e900\U0001e922\U0001d400Ａａ'
    'KΩÅﬀﬃ͸￿\U0010ffffⅠⅰⒶⓐẞⱥȺ'))


def rnd_text(r, n, pool=POOL):
    return ''.join(r.choice(pool) for _ in range(n))


def b(s):
    return s.encode('utf-8')


def boff(s, i):
    """Byte offset of character index i of s (i may be -1 for not found)."""
    return -1 if i < 0 else len(b(s[:i]))


def up(bs, p):
    n = len(bs)
    if p <= 0: return 0
    if p >= n: return n
    while p < n and (bs[p] & 0xC0) == 0x80: p += 1
    return p


def down(bs, p):
    n = len(bs)
    if p <= 0: return 0
    if p >= n: return n
    while p > 0 and (bs[p] & 0xC0) == 0x80: p -= 1
    return p


def adj(start, end, n):
    if end > n: end = n
    elif end < 0: end = max(end + n, 0)
    if start < 0: start = max(start + n, 0)
    return start, end


def err(why, n):
    return {'err': why, 'n': n}


def digest(t):
    return '%d:%s' % (len(t), hashlib.sha256(b(t)).hexdigest())


def ldigest(ts):
    return '%d:%s' % (len(ts), hashlib.sha256(b('\u0001'.join(ts))).hexdigest())


# ---------------------------------------------------------------- CPython cases

PRED_ORDER = ['isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit', 'isidentifier',
              'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper']


def preds(s):
    return sum((1 << k) for k, name in enumerate(PRED_ORDER) if getattr(s, name)())


def too_large(t):
    return err('too large', len(t)) if len(t) > LIMIT else t


def case6(s):
    return [too_large(s.upper()), too_large(s.lower()), too_large(s.casefold()),
            too_large(s.title()), too_large(s.capitalize()), too_large(s.swapcase())]


def cp_cases(r):
    """Cases whose reference is CPython's str (or bytes) methods."""
    C = []

    def add(op, args, want, src='cpython'):
        C.append({'op': op, 'args': args, 'want': want, 'src': src})
    texts = ASCII + WIDE + [rnd_text(r, r.randint(0, 14)) for _ in range(80)]
    probes = ['a', ' ', '\u03a3', '\u00e9', 'ss', 'aa', 'ab', '\u00df', 'x', '\r\n']
    # search
    for k, s in enumerate(texts):
        subs = ['', probes[k % len(probes)], rnd_text(r, r.randint(1, 2))]
        if s:
            i = r.randint(0, len(s)); j = r.randint(i, min(len(s), i + 4))
            subs.append(s[i:j])
        for sub in subs:
            add('find', [s, sub], boff(s, s.find(sub)))
            add('rfind', [s, sub], boff(s, s.rfind(sub)))
            add('count', [s, sub], s.count(sub))
        bs = b(s)
        n = len(bs)
        for sub in (subs[-1], ''):
            st = r.randint(-n - 3, n + 3); en = r.randint(-n - 3, n + 3)
            if sub:
                add('find_in', [s, sub, st, en], bs.find(b(sub), st, en), 'cpython-bytes')
                add('rfind_in', [s, sub, st, en], bs.rfind(b(sub), st, en), 'cpython-bytes')
                add('count_in', [s, sub, st, en], bs.count(b(sub), st, en), 'cpython-bytes')
            else:
                a, e = adj(st, en, n)
                f = -1 if e - a < 0 else (up(bs, a) if up(bs, a) <= e else -1)
                rf = -1 if e - a < 0 else (down(bs, e) if down(bs, e) >= a else -1)
                if e - a < 0:
                    cnt = 0
                else:
                    ua, de = up(bs, a), down(bs, e)
                    cnt = 0 if ua > de else len(bs[ua:de].decode()) + 1
                add('find_in', [s, sub, st, en], f, 'derived')
                add('rfind_in', [s, sub, st, en], rf, 'derived')
                add('count_in', [s, sub, st, en], cnt, 'derived')
        # a character-aligned range through str (start and end as character indices)
        ci = r.randint(0, len(s)); cj = r.randint(0, len(s))
        for sub in subs[:2]:
            add('find_in', [s, sub, boff(s, ci), boff(s, cj)], boff(s, s.find(sub, ci, cj)))
            add('rfind_in', [s, sub, boff(s, ci), boff(s, cj)], boff(s, s.rfind(sub, ci, cj)))
            add('count_in', [s, sub, boff(s, ci), boff(s, cj)], s.count(sub, ci, cj))
    # replace
    news = ['', '-', '\u00df\u00df', 'XYZ']
    for k, s in enumerate(texts[::2]):
        olds = ['', probes[k % len(probes)]] + ([s[: max(1, len(s) // 3)]] if s else [])
        for j, old in enumerate(olds):
            new = news[(k + j) % len(news)]
            add('replace', [s, old, new], s.replace(old, new))
            for cnt in (1, 2):
                add('replace_n', [s, old, new, cnt], s.replace(old, new, cnt))
        add('replace_n', [s, 'a', 'b', 0], s.replace('a', 'b', 0))
    # partition and the splits
    seps = [' ', 'a', ',', '\u00e9', 'aa', '\r\n', 'ab']
    for k, s in enumerate(texts):
        mine = [seps[k % len(seps)]] + ([s[len(s) // 2: len(s) // 2 + 2]] if len(s) > 2 else [])
        for sep in mine:
            add('partition', [s, sep], list(s.partition(sep)))
            add('rpartition', [s, sep], list(s.rpartition(sep)))
            h, sp, t = s.partition(sep)
            add('split_once', [s, sep], [h, t] if sp else [])
            h, sp, t = s.rpartition(sep)
            add('rsplit_once', [s, sep], [h, t] if sp else [])
            for mx in (-1, 0, 1, 2):
                add('splitn', [s, sep, mx], s.split(sep, mx))
                add('rsplitn', [s, sep, mx], s.rsplit(sep, mx))
        if k % 3 == 0:
            for mx in (-1, 0, 1, 2):
                # an empty separator: re.split(""), which matches at every boundary
                want = [s] if mx == 0 else re.split('', s, maxsplit=0 if mx < 0 else mx)
                add('splitn', [s, '', mx], want, 'cpython-re' if mx != 0 else 'derived')
                rev = [s] if mx == 0 else [p[::-1] for p in re.split('', s[::-1], maxsplit=0 if mx < 0 else mx)][::-1]
                add('rsplitn', [s, '', mx], rev, 'derived')
            add('partition', [s, ''], ['', '', s], 'derived')
            add('rpartition', [s, ''], [s, '', ''], 'derived')
            add('split_once', [s, ''], ['', s], 'derived')
            add('rsplit_once', [s, ''], [s, ''], 'derived')
        if k % 2 == 0:
            add('splitlines', [s, False], s.splitlines())
            add('splitlines', [s, True], s.splitlines(True))
    for s in ['a\x0bb\x0cc\x1cd\x1de\x1ef\x85g\u2028h\u2029i', 'a\rb\r\nc\n\rd', '\r', '\r\n\r\n', 'x\r\n', 'x\r',
              '\n\n', 'no breaks', 'a\r\nb\nc\r\n', 'a\r\nb\nc', '\r\r\n', 'e\u0301\r\n\u00e9\n']:
        add('splitlines', [s, False], s.splitlines())
        add('splitlines', [s, True], s.splitlines(True))
    # repeat, slicing, padding
    fills = [' ', '*', '\u00e9', '\U0001f600']
    for k, s in enumerate(texts[::3]):
        for mul in (-1, 0, 3):
            add('repeat', [s, mul], s * mul)
        n = len(s)
        for _ in range(3):
            a = r.randint(-n - 2, n + 2); c = r.randint(-n - 2, n + 2)
            add('slice', [s, a, c], s[a:c])
            stp = r.choice([1, 2, 3, -1, -2, -3])
            sa = r.choice([None, a]); sc = r.choice([None, c])
            add('slice_step', [s, sa, sc, stp], s[sa:sc:stp])
            i = r.randint(-n - 1, n)
            add('char_at', [s, i], s[i] if -n <= i < n else 'NONE')
        add('slice_step', [s, None, None, -1], s[::-1])
        for w in (n - 1, n + 1, n + 4):
            f = fills[(k + w) % len(fills)]
            add('center', [s, w, f], s.center(w, f))
            add('ljust', [s, w, f], s.ljust(w, f))
            add('rjust', [s, w, f], s.rjust(w, f))
            add('zfill', [s, w], s.zfill(w))
    for s in ['', 'a', 'ab', 'abc', '\u00e9']:
        for w in (-1, 0, 1, 2, 3, 4, 5, 6):
            add('center', [s, w, '*'], s.center(w, '*'))
    add('slice_step', ['abc', None, None, 0], err('step', 0), 'derived')
    for s in ['abc\u00e9', '', '\u00e9']:
        for stp in (1 << 62, (1 << 63) - 1, -(1 << 62), -(1 << 63) + 1, 5, -5):
            for sa, sc in ((None, None), (1, None), (None, 1), (-(1 << 62), 1 << 62), (1 << 62, -(1 << 62))):
                add('slice_step', [s, sa, sc, stp], s[sa:sc:stp])
    for t in (1 << 62, LIMIT, LIMIT - 1):
        s = 'a\tb'
        n = 1 + (t - 1) + 1
        add('expandtabs', [s, t], s.expandtabs(t) if n <= LIMIT else err('too large', n), 'cpython' if n <= LIMIT else 'derived')
    add('center', ['ab', 5, 'ab'], err('fill', 2), 'derived')
    add('ljust', ['ab', 2, ''], err('fill', 0), 'derived')
    add('rjust', ['ab', 9, '\u00e9\u00e9\u00e9'], err('fill', 3), 'derived')
    for s in ['+', '-', '+1', '-1', '1-', '--1', '+-1', '\u00e9', '-\u00e9', '', '42']:
        for w in (0, 1, 3, 5):
            add('zfill', [s, w], s.zfill(w))
    for k, s in enumerate(texts[::6] + ['a\tb', '\tx', 'ab\tcd\tef', 'a\nb\tc', 'a\rb\tc', '\u00e9\t\u00e9\t', '\t\t\t',
                                        '12345678\tx', '\U0001f600\tx', 'col1\tcol2\tcol3\n\tx\ty']):
        for t in (-1, 0, 1, 4, 8):
            add('expandtabs', [s, t], s.expandtabs(t))
    # prefix and suffix, strip
    for k, s in enumerate(texts[::2]):
        for p in ['', s[:2], s + 'x'][k % 3:] + [s]:
            add('removeprefix', [s, p], s.removeprefix(p))
        for p in ['', s[-2:], 'x' + s][k % 3:] + [s]:
            add('removesuffix', [s, p], s.removesuffix(p))
        add('strips', [s], [s.lstrip(), s.rstrip(), s.strip()])
        chars = ['', ' ', 'a', 'ab', ' \t\n', '\u00e9\u03a3', s[:2]][k % 7]
        add('strips_chars', [s, chars], [s.lstrip(chars), s.rstrip(chars), s.strip(chars)])
    for s in ['\x1c\x1dx\x1e\x1f', '\u0085\u00a0x\u3000\u2028', '\u180ex\u200b', ' \t\n\x0b\x0c\r ', '', 'xx\u00e9yxx']:
        add('strips', [s], [s.lstrip(), s.rstrip(), s.strip()])
        add('strips_chars', [s, 'x\u00e9\x1c'], [s.lstrip('x\u00e9\x1c'), s.rstrip('x\u00e9\x1c'), s.strip('x\u00e9\x1c')])
    # predicates and case
    ident = ['_', '_x', 'x1', '1x', 'a b', 'a-b', '\u2118x', 'x\u00b7y', '\u00b7x', '\u212e', '\u2170x', '\u0663x',
             'x\u0663', '\u0e01\u0e33', 'a\u200d', 'a\u200c', '\u309b', '\u1885', 'ab\u0300', '\u0300ab']
    for s in texts + ident:
        add('preds', [s], preds(s))
        add('case', [s], case6(s))
    for s in ['\u03a3', 'A\u03a3', '\u03a3A', 'A\u03a3 b', 'A\u03a3\'s', 'A\'\u03a3\'', 'A\u03a3\u0301', 'A\u0301\u03a3',
              '\u03a3\u03a3', 'A\u03a3\u03a3', 'A\u03a3\u03a3 ', 'A\u03a3.\u03a3', 'a\u03a3\u00ad\u00adb', 'a\u03a3\u00ad\u00ad',
              '\u00ad\u03a3', '1\u03a3', 'A1\u03a3', '\u1fbc\u03a3', '\u0345\u03a3', 'A\u03a3\u0345', '\U00010400\u03a3',
              '\u03a3\U00010428', '\u0130\u03a3', '\u01c5\u03a3x', 'AB\u03a3\u03a3\u03a3 \u03a3\u03a3 A\u03a3', '\u03a3a', '\u03a3\u00ad']:
        add('case', [s], case6(s))
        add('preds', [s], preds(s))
    # chr, ord, offsets
    for c in (0, 65, 233, 931, 0x10ffff, 0xd7ff, 0xe000, 0x1f600):
        add('chr', [c], chr(c))
    for c in (-1, 0xd800, 0xdfff, 0x110000, 1 << 40):
        add('chr', [c], err('code point', c), 'derived')
    for s in texts[::4]:
        add('ord', [s], ord(s[0]) if s else -1, 'cpython' if s else 'derived')
        n = len(s)
        for k in (-1, 0, n // 2, n, n + 1):
            add('char_offset', [s, k], boff(s, k) if 0 <= k <= n else -1, 'derived')
        bs = b(s)
        for p in (-2, 1, len(bs) // 2, len(bs), len(bs) + 2):
            add('char_index', [s, p], len(bs[:down(bs, p)].decode()), 'derived')
    # StrFinder
    needles = ['a', 'aa', 'aba', 'ab', '\u00e9\u00e9', 'xyzx', 'aab']
    for k, s in enumerate(texts[::5] + ['aaaa', 'abababab', 'aabaabaaab', '\u00e9\u00e9\u00e9', 'xyzxyzxyz', 'aaabaaab']):
        for nd in [needles[k % len(needles)], needles[(k + 3) % len(needles)]] + ([s[1:3]] if len(s) > 3 else []):
            ov = [boff(s, m.start()) for m in re.finditer('(?=' + re.escape(nd) + ')', s)]
            no = [boff(s, m.start()) for m in re.finditer(re.escape(nd), s)]
            add('finder_all', [nd, s, True], ov, 'cpython-re')
            add('finder_all', [nd, s, False], no, 'cpython-re')
            bs = b(s)
            for st in (-3, len(bs) // 2):
                add('finder_find', [nd, s, st], bs.find(b(nd), st), 'cpython-bytes')
            if len(s) >= 2:
                cut = r.randint(0, len(s)); cut2 = r.randint(cut, len(s))
                chunks = [s[:cut], s[cut:cut2], s[cut2:]]
                add('scan', [nd, chunks, True], ov, 'cpython-re')
                add('scan', [nd, chunks, False], no, 'cpython-re')
        add('finder_all', ['', s, False], [boff(s, i) for i in range(len(s) + 1)], 'derived')
    for s in ('', 'ab', '\u00e9x'):
        add('finder_find', ['', s, 1], up(b(s), 1) if 1 <= len(b(s)) else -1, 'derived')
        add('scan', ['', [s, s], False], [boff(s + s, i) for i in range(len(s) * 2)], 'derived')
    return C


def record_reps():
    """One code point of every record of the generated table, every code
    point that starts or ends a run of one record, and every code point
    that a case mapping changes."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import str_data
    flags, up_, lo_, ti_, cf_, asc, _gc = str_data.build()
    recs, per, _sx, _af = str_data.records(flags, up_, lo_, ti_, cf_, asc)
    seen, reps, runs = set(), [], []
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF:
            continue
        if per[c] not in seen:
            seen.add(per[c]); reps.append(c)
        if c > 0 and per[c] != per[c - 1]:
            runs.append(c)
            if not (0xD800 <= c - 1 <= 0xDFFF):
                runs.append(c - 1)
    mapped = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF and
              (up_[c] != [c] or lo_[c] != [c] or ti_[c] != [c] or cf_[c] != [c])]
    return reps, sorted(set(runs)), mapped, sorted(asc)


def table_cases():
    C = []
    reps, runs, mapped, folded = record_reps()
    pool = sorted(set(reps) | set(runs))
    for i in range(0, len(pool), 48):
        s = ''.join(chr(c) for c in pool[i:i + 48])
        C.append({'op': 'flags_each', 'args': [s], 'want': [preds(ch) for ch in s], 'src': 'cpython'})
    for i in range(0, len(mapped), 40):
        s = ' '.join(chr(c) for c in mapped[i:i + 40])
        C.append({'op': 'case', 'args': [s], 'want': case6(s), 'src': 'cpython'})
    for i in range(0, len(reps), 40):
        s = ' '.join(chr(c) for c in reps[i:i + 40])
        C.append({'op': 'case', 'args': [s], 'want': case6(s), 'src': 'cpython'})
    return C, folded


# ---------------------------------------------------------------- big inputs

def build(spec):
    return ''.join(p * k for p, k in spec)


def big_cases():
    """Texts above one window (STR_WIN bytes of match starts) and at the size
    limit, built in the checks by str_repeat; results compared by length and
    sha256."""
    from django.utils.text import slugify
    C = []

    def add(op, args, want, src='cpython'):
        C.append({'op': op, 'args': args, 'want': want, 'src': src})
    W = 999000
    specs = [
        ([['x', W - 2], ['NEEDLE', 1], ['x', 100]], ['NEEDLE', 'xN', 'zz']),
        ([['ab', 499999]], ['ab', 'ba', 'a']),
        ([['\u00e9a', 400000]], ['\u00e9a', 'a\u00e9', 'a']),
        ([['a', LIMIT]], ['a', 'aa', 'b']),
        ([['\u4e2d', 666000], ['ab', 3]], ['ab', '\u4e2d\u4e2d', 'b']),
        ([['\u4e2d', 300000], ['ba', 10], ['\u4e2d', 300000]], ['ba', 'aba', '\u4e2db']),
        ([['line\r\n', 150000]], ['\r\n', 'line', 'e\r']),
    ]
    for spec, subs in specs:
        s = build(spec)
        bs = b(s)
        for sub in subs:
            add('big_find', [{'build': spec}, sub], boff(s, s.find(sub)) if sub in s else -1)
            add('big_rfind', [{'build': spec}, sub], boff(s, s.rfind(sub)) if sub in s else -1)
            add('big_count', [{'build': spec}, sub], s.count(sub))
            add('big_count_in', [{'build': spec}, sub, 5, len(bs) - 5], bs.count(b(sub), 5, len(bs) - 5), 'cpython-bytes')
            t = s.replace(sub, 'Q')
            add('big_replace', [{'build': spec}, sub, 'Q'], err('too large', len(t)) if len(t) > LIMIT else digest(t))
            t = s.replace(sub, '\u00df\u00df\u00df')
            add('big_replace', [{'build': spec}, sub, '\u00df\u00df\u00df'], err('too large', len(t)) if len(t) > LIMIT else digest(t))
            t = s.replace(sub, '', 3)
            add('big_replace_n', [{'build': spec}, sub, '', 3], digest(t))
            for k in (-1, 2):
                ps = s.split(sub, k)
                src = 'cpython'
                if len(ps) > LIMIT:
                    ps = s.split(sub, LIMIT - 1)
                    src = 'derived'
                add('big_splitn', [{'build': spec}, sub, k], ldigest(ps), src)
        add('big_splitlines', [{'build': spec}], ldigest(s.splitlines()))
        add('big_case3', [{'build': spec}], [x if isinstance(x, dict) else digest(x) for x in
                                             (too_large(s.upper()), too_large(s.lower()), too_large(s.casefold()), too_large(s.capitalize()))])
        add('big_slug', [{'build': spec}], digest(slugify(s)), 'django')
    # per-character functions on moderate texts
    for spec in ([['\u03a3\u03b1 \u00c9t\u00e9 ', 12000]], [['Hello World ', 9000]], [['\u03a3', 40000], ['a', 1]]):
        s = build(spec)
        add('big_title', [{'build': spec}], [digest(s.title()), digest(s.swapcase()), str(preds(s))], 'cpython')
    for spec in ([['\u03a3\u03b1 ', 12000]], [['A\u00ad\u03a3 ', 10000]]):
        s = build(spec)
        add('big_case3', [{'build': spec}], [x if isinstance(x, dict) else digest(x) for x in
                                             (too_large(s.upper()), too_large(s.lower()), too_large(s.casefold()), too_large(s.capitalize()))])
    # results above the limit
    add('repeat', ['ab', 600000], err('too large', 1200000), 'derived')
    add('repeat', ['\u00e9', LIMIT + 1], err('too large', LIMIT + 1), 'derived')
    add('repeat', ['ab', 1 << 62], err('too large', 9223372036854775807), 'derived')
    add('center', ['ab', LIMIT + 5, '*'], err('too large', LIMIT + 5), 'derived')
    add('zfill', ['1', LIMIT + 1], err('too large', LIMIT + 1), 'derived')
    s = build([['\u00df', 600000]])
    add('big_case3', [{'build': [['\u00df', 600000]]}], [x if isinstance(x, dict) else digest(x) for x in
                                                         (too_large(s.upper()), too_large(s.lower()), too_large(s.casefold()), too_large(s.capitalize()))])
    add('big_expandtabs', [{'build': [['\t', 200000]]}, 1], digest(('\t' * 200000).expandtabs(1)))
    add('big_expandtabs', [{'build': [['ab\tc', 120000]]}, 4], digest(('ab\tc' * 120000).expandtabs(4)))
    add('big_expandtabs', [{'build': [['\t', 400000]]}, 4], err('too large', 1600000), 'derived')
    return C


# ---------------------------------------------------------------- Django and change-case

def slug_cases(folded, r):
    from django.utils.text import slugify
    import django
    C = []
    texts = ASCII + WIDE + [rnd_text(r, r.randint(0, 16)) for _ in range(150)] + [
        'Hello, World!', ' Leading and trailing ', '__init__', '-x-', '_-_x_-_', 'a - b', 'a_-_b',
        'Côte d\'Ivoire', '½ ¼ ¾', 'Ⅷ ㎒ ﬁ', 'mañana', 'Ångström',
        'Grüße aus Köln', 'İstanbul ı', '中文 only', 'emoji \U0001f600 here',
        'tabs\tand\nnew\x0blines\x0c\r', '\x1c\x1d\x1e\x1fx', ' nb sp', 'x́y', '①②③']
    for s in texts:
        C.append({'op': 'slug', 'args': [s], 'want': slugify(s), 'src': 'django'})
    for i in range(0, len(folded), 60):
        s = ' '.join(chr(c) for c in folded[i:i + 60])
        C.append({'op': 'slug', 'args': [s], 'want': slugify(s), 'src': 'django'})
    s = build([['Café Été ', 50000]])
    C.append({'op': 'big_slug', 'args': [{'build': [['Café Été ', 50000]]}], 'want': digest(slugify(s)), 'src': 'django'})
    return C, django.__version__


NODE_JS = r"""
const fs = require('fs');
const inp = JSON.parse(fs.readFileSync(0, 'utf8'));
const chars = JSON.parse(process.argv[2]);
import(require('url').pathToFileURL(process.argv[1]).href).then(cc => {
  const o = {locale: false};
  const out = inp.map(s => [cc.split(s), cc.camelCase(s, o), cc.pascalCase(s, o), cc.snakeCase(s, o),
    cc.kebabCase(s, o), cc.constantCase(s, o), cc.capitalCase(s, o)]);
  const props = {};
  for (const ch of chars) {
    props[ch] = [/\p{Lu}/u.test(ch), /\p{Ll}/u.test(ch), /\p{L}/u.test(ch), /[^\p{L}\d]/iu.test(ch),
      ch.toUpperCase(), ch.toLowerCase()];
  }
  process.stdout.write(JSON.stringify({out, props}));
});
"""


def change_case_pkg():
    """Directory and package.json of change-case (NODE_PATH or ./node_modules)."""
    dirs = [d for d in os.environ.get('NODE_PATH', '').split(os.pathsep) if d] + [os.path.join(os.getcwd(), 'node_modules')]
    for d in dirs:
        p = os.path.join(d, 'change-case', 'package.json')
        if os.path.exists(p):
            return os.path.dirname(p), json.load(open(p))
    raise SystemExit('change-case not found (set NODE_PATH)')


def change_case_version():
    return change_case_pkg()[1]['version']


def node_run(inputs, chars):
    d, pkg = change_case_pkg()
    entry = os.path.join(d, pkg.get('main', 'dist/index.js'))
    p = subprocess.run(['node', '-e', NODE_JS, '--', entry, json.dumps(chars)], input=json.dumps(inputs),
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def case_cases(r):
    import unicodedata
    ids = ['fooBar', 'FooBar', 'foo_bar', 'FOO_BAR', 'foo-bar', 'foo bar', 'Foo Bar', 'HTTPServerError',
           'getHTTPResponseCode', 'XMLHttpRequest', 'version 1.2.3', 'v2Beta3', 'iPhone', 'eBay', 'a1B2c3',
           'ABCdef', 'ABCDef', 'ABcDEf', 'aBCd', 'snake_case_here', 'kebab-case-here', '  leading and trailing  ',
           'Ünïcödé Wörds', 'ΣΑΣ σας', 'İstanbul', 'straße',
           'ǅemal', 'Привет мир', '日本語テキスト',
           'emoji\U0001f600word', 'Ⅻ', 'version 1 2', 'test2Test', 'Test2test', 'TEST2Test', '1st place',
           "don't stop", 'it’s', 'aͅb', 'x́y', '٣٤ digits', 'mixed nbsp', '__dunder__',
           'ALLCAPS', 'a', '', '_', '1', '12abc', 'abc12', 'ab12CD', 'CamelCASEWord', 'snake__double', 'IOStream',
           'ὈΔΥΣΣΕΎΣ', 'KK', 'ﬁnance', 'ŉame', 'ǆz']
    texts = ids + ASCII + [rnd_text(r, r.randint(0, 14)) for _ in range(300)]
    chars = sorted(set(''.join(texts)))
    res = node_run(texts, chars)
    props = res['props']
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import str_data
    flags, *_ = str_data.build()
    # characters whose Unicode 16.0 data (Node) differs from 15.1 (CPython)
    bad = set()
    for ch in chars:
        lu, ll, isl, sep, u, l = props[ch]
        cat = unicodedata.category(ch)
        word = not sep
        if (lu != (cat == 'Lu') or ll != (cat == 'Ll') or isl != cat.startswith('L') or u != ch.upper()
                or l != ch.lower() or word != (bool(flags[ord(ch)] & str_data.WORD) or '0' <= ch <= '9')):
            bad.add(ch)
    C, dropped = [], {'unicode16': 0, 'astral': 0}
    for s, (ws, camel, pascal, snake, kebab, const, capital) in zip(texts, res['out']):
        if any(ch in bad for ch in s):
            dropped['unicode16'] += 1
            continue
        # change-case upper-cases the first UTF-16 unit of a word: a word that
        # starts with a letter above U+FFFF that has an uppercase form differs
        if any(w and ord(w[0]) > 0xFFFF and w[0].upper() != w[0] for w in ws):
            dropped['astral'] += 1
            continue
        C.append({'op': 'words', 'args': [s], 'want': ws, 'src': 'change-case'})
        C.append({'op': 'wcase', 'args': [s], 'want': [camel, pascal, snake, kebab, const, capital], 'src': 'change-case'})
    return C, change_case_version(), dropped, sorted(bad)


# ---------------------------------------------------------------- refresh

def refresh():
    import unicodedata
    if sys.version_info[:2] != (3, 13) or unicodedata.unidata_version != '15.1.0':
        raise SystemExit('--refresh needs CPython 3.13 with Unicode 15.1.0')
    r = random.Random(20261025)
    cases = cp_cases(r)
    tc, folded = table_cases()
    cases += tc
    cases += big_cases()
    cases.append({'op': 'chunks', 'args': [{'build': [['a', LIMIT]]}, 'a'], 'want': [LIMIT - 2, 3], 'src': 'derived'})
    cases.append({'op': 'chunks', 'args': [{'build': [['ab', 1000]]}, 'b'], 'want': [1001], 'src': 'derived'})
    sc, dj = slug_cases(folded, r)
    cases += sc
    cc, ccv, dropped, bad = case_cases(r)
    cases += cc
    meta = {'cpython': sys.version.split()[0], 'unicode': unicodedata.unidata_version, 'django': dj,
            'change_case': ccv, 'dropped': dropped, 'unicode16_differs': [ord(c) for c in bad],
            'seed': 20261025}
    os.makedirs(os.path.dirname(VEC), exist_ok=True)
    with open(VEC, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'cases': cases}, f, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
        f.write('\n')
    sys.stderr.write('%d cases, meta %s\n' % (len(cases), json.dumps(meta)[:400]))


# ---------------------------------------------------------------- checks

HEAD = '''
# Checks for lib/text/str.rce, generated by scripts/gen/str_gen.py from
# vectors/text/str_vectors.json ({src}). Do not edit.

fn t_err(e: BadStr) -> text = "ERR:" + e.why + ":" + e.n.tostr()

fn t_build(ps: List[text], ks: List[i64]) -> text
    var parts: List[text] = []
    for i in 0..ps.len()
        parts.push(str_repeat(&ps[i], ks[i]) catch |e| t_err(e))
    return "".join(parts)

fn t_digest(t: text) -> text = t.len_chars().tostr() + ":" + t.sha256()

fn t_dlist(ts: List[text]) -> text = ts.len().tostr() + ":" + "\\u{1}".join(ts).sha256()

fn t_find(s: text, x: text) -> i64 = str_find(&s, &x)

fn t_rfind(s: text, x: text) -> i64 = str_rfind(&s, &x)

fn t_count(s: text, x: text) -> i64 = str_count(&s, &x)

fn t_find_in(s: text, x: text, i: i64, j: i64) -> i64 = str_find_in(&s, &x, i, j)

fn t_rfind_in(s: text, x: text, i: i64, j: i64) -> i64 = str_rfind_in(&s, &x, i, j)

fn t_count_in(s: text, x: text, i: i64, j: i64) -> i64 = str_count_in(&s, &x, i, j)

fn t_replace(s: text, a: text, z: text) -> text = str_replace(&s, &a, &z) catch |e| t_err(e)

fn t_replace_n(s: text, a: text, z: text, k: i64) -> text = str_replace_n(&s, &a, &z, k) catch |e| t_err(e)

fn t_partition(s: text, p: text) -> List[text]
    let r = str_partition(&s, &p)
    return [copy(r.0), copy(r.1), copy(r.2)]

fn t_rpartition(s: text, p: text) -> List[text]
    let r = str_rpartition(&s, &p)
    return [copy(r.0), copy(r.1), copy(r.2)]

fn t_split_once(s: text, p: text) -> List[text]
    let r = str_split_once(&s, &p)
    var o: List[text] = []
    if r.issome()
        let q = r.or(("", ""))
        o.push(copy(q.0))
        o.push(copy(q.1))
    return o

fn t_rsplit_once(s: text, p: text) -> List[text]
    let r = str_rsplit_once(&s, &p)
    var o: List[text] = []
    if r.issome()
        let q = r.or(("", ""))
        o.push(copy(q.0))
        o.push(copy(q.1))
    return o

fn t_splitn(s: text, p: text, k: i64) -> List[text] = str_splitn(&s, &p, k)

fn t_rsplitn(s: text, p: text, k: i64) -> List[text] = str_rsplitn(&s, &p, k)

fn t_splitlines(s: text, k: bool) -> List[text] = str_splitlines(&s, k)

fn t_repeat(s: text, k: i64) -> text = str_repeat(&s, k) catch |e| t_err(e)

fn t_slice(s: text, a: i64, z: i64) -> text = str_slice(&s, a, z)

fn t_slice_step(s: text, a: i64?, z: i64?, k: i64) -> text = str_slice_step(&s, a, z, k) catch |e| t_err(e)

fn t_char_at(s: text, i: i64) -> text = str_char_at(&s, i).or("NONE")

fn t_center(s: text, w: i64, f: text) -> text = str_center(&s, w, &f) catch |e| t_err(e)

fn t_ljust(s: text, w: i64, f: text) -> text = str_ljust(&s, w, &f) catch |e| t_err(e)

fn t_rjust(s: text, w: i64, f: text) -> text = str_rjust(&s, w, &f) catch |e| t_err(e)

fn t_zfill(s: text, w: i64) -> text = str_zfill(&s, w) catch |e| t_err(e)

fn t_expandtabs(s: text, k: i64) -> text = str_expandtabs(&s, k) catch |e| t_err(e)

fn t_removeprefix(s: text, p: text) -> text = str_removeprefix(&s, &p)

fn t_removesuffix(s: text, p: text) -> text = str_removesuffix(&s, &p)

fn t_strips(s: text) -> List[text] = [str_lstrip(&s), str_rstrip(&s), str_strip(&s)]

fn t_strips_chars(s: text, c: text) -> List[text] = [str_lstrip_chars(&s, &c), str_rstrip_chars(&s, &c), str_strip_chars(&s, &c)]

fn t_bit(b: bool, k: i64) -> i64 = if b then k else 0

# isalnum isalpha isascii isdecimal isdigit isidentifier islower isnumeric
# isprintable isspace istitle isupper, bits 1, 2, 4, ... 2048
fn t_preds(s: text) -> i64 = t_bit(str_isalnum(&s), 1) + t_bit(str_isalpha(&s), 2) + t_bit(str_isascii(&s), 4) + t_bit(str_isdecimal(&s), 8) + t_bit(str_isdigit(&s), 16) + t_bit(str_isidentifier(&s), 32) + t_bit(str_islower(&s), 64) + t_bit(str_isnumeric(&s), 128) + t_bit(str_isprintable(&s), 256) + t_bit(str_isspace(&s), 512) + t_bit(str_istitle(&s), 1024) + t_bit(str_isupper(&s), 2048)

fn t_flags_each(s: text) -> List[i64]
    var o: List[i64] = []
    for c in s.chars()
        o.push(t_preds(c))
    return o

fn t_case(s: text) -> List[text] = [str_upper(&s) catch |e| t_err(e), str_lower(&s) catch |e| t_err(e), str_casefold(&s) catch |e| t_err(e), str_title(&s) catch |e| t_err(e), str_capitalize(&s) catch |e| t_err(e), str_swapcase(&s) catch |e| t_err(e)]

fn t_dg(t: text) -> text = if t.startswith("ERR:") then t else t_digest(t)

fn t_dcase3(s: text) -> List[text] = [t_dg(str_upper(&s) catch |e| t_err(e)), t_dg(str_lower(&s) catch |e| t_err(e)), t_dg(str_casefold(&s) catch |e| t_err(e)), t_dg(str_capitalize(&s) catch |e| t_err(e))]

fn t_dtitle(s: text) -> List[text] = [t_dg(str_title(&s) catch |e| t_err(e)), t_dg(str_swapcase(&s) catch |e| t_err(e)), t_preds(copy(s)).tostr()]

# lengths of the chunk lists of sx_split_chunks (a private function)
fn t_chunks(s: text, p: text) -> List[i64]
    var o: List[i64] = []
    for c in sx_split_chunks(&s, &p)
        o.push(c.len() as i64)
    return o

fn t_chr(c: i64) -> text = str_chr(c) catch |e| t_err(e)

fn t_ord(s: text) -> i64 = str_ord(&s)

fn t_char_offset(s: text, k: i64) -> i64 = str_char_offset(&s, k)

fn t_char_index(s: text, p: i64) -> i64 = str_char_index(&s, p)

fn t_words(s: text) -> List[text] = str_words(&s)

fn t_wcase(s: text) -> List[text] = [str_camel(&s) catch |e| t_err(e), str_pascal(&s) catch |e| t_err(e), str_snake(&s) catch |e| t_err(e), str_kebab(&s) catch |e| t_err(e), str_constant(&s) catch |e| t_err(e), str_capital(&s) catch |e| t_err(e)]

fn t_slug(s: text) -> text = str_slugify(&s) catch |e| t_err(e)

fn t_finder_all(n: text, h: text, o: bool) -> List[i64]
    let f = str_finder(&n)
    return str_finder_all(&f, &h, o)

fn t_finder_find(n: text, h: text, st: i64) -> i64
    let f = str_finder(&n)
    return str_finder_find(&f, &h, st)

fn t_scan(n: text, cs: List[text], o: bool) -> List[i64]
    let f = str_finder(&n)
    var st = str_scan(&f, o)
    var out: List[i64] = []
    for c in cs
        for x in str_scan_feed(&mut st, &c)
            out.push(x)
    return out
'''



PROPS = """
# Seeded property sweeps for lib/text/str.rce (derived): the search,
# split and replace functions against StrFinder and against each other,
# StrScan over random chunkings, slicing and offset round trips, case
# mapping chunk by chunk, and StrBuf. Written by scripts/gen/str_gen.py.

fn t_err(e: BadStr) -> text = "ERR:" + e.why + ":" + e.n.tostr()

fn t_rnd(s: i64) -> i64 = (s * 1103515245 + 12345) & 2147483647

const T_POOL: text = "aab\\u{e9}\\u{3a3}\\u{3c3}\\u{df} \\t\\n-_A\\u{1f600}\\u{130}xyz\\u{e9}a\\u{1c5}"

fn t_gen(seed: i64, n: i64) -> text
    let cs = T_POOL.chars()
    var ps: List[text] = []
    var s = seed
    for _i in 0..n
        s = t_rnd(s)
        ps.push(copy(cs[(s % (cs.len() as i64)) as usize]))
    return "".join(ps)

fn t_last(xs: &[i64]) -> i64 = if xs.len() == 0 then -1 else xs[xs.len() - 1]

fn t_first(xs: &[i64]) -> i64 = if xs.len() == 0 then -1 else xs[0]

# leftmost non-overlapping matches by repeated str_find_in
fn t_walk(s: &text, sub: &text) -> List[i64]
    var out: List[i64] = []
    let n = s.len_bytes() as i64
    let m = sub.len_bytes() as i64
    var p: i64 = 0
    while p <= n bound 1000
        let q = str_find_in(s, sub, p, n)
        if q < 0
            break
        out.push(q)
        p = q + m
    return out

fn t_prop_search(seed: i64) -> bool
    let s = t_gen(seed, seed % 29 + 1)
    let sub = t_gen(seed * 7 + 3, seed % 3 + 1)
    let f = str_finder(&sub)
    let ov = str_finder_all(&f, &s, true)
    let no = str_finder_all(&f, &s, false)
    var ok = t_walk(&s, &sub) == no and str_count(&s, &sub) == (no.len() as i64)
    ok = ok and str_find(&s, &sub) == t_first(&ov) and str_rfind(&s, &sub) == t_last(&ov)
    ok = ok and str_finder_find(&f, &s, 0) == t_first(&ov)
    let ps = str_splitn(&s, &sub, -1)
    ok = ok and ps.len() == no.len() + 1 and sub.join(ps) == s
    let rps = str_rsplitn(&s, &sub, -1)
    ok = ok and sub.join(rps) == s
    let mark = "<>"
    let rep = str_replace(&s, &sub, &mark) catch |e| t_err(e)
    ok = ok and rep == mark.join(str_splitn(&s, &sub, -1))
    let pt = str_partition(&s, &sub)
    let rpt = str_rpartition(&s, &sub)
    ok = ok and pt.0 + pt.1 + pt.2 == s and rpt.0 + rpt.1 + rpt.2 == s
    # a scan over three random chunks finds the overlapping matches
    let nc = s.len_chars() as i64
    let c1 = t_rnd(seed) % (nc + 1)
    let c2 = c1 + t_rnd(seed + 1) % (nc - c1 + 1)
    var st = str_scan(&f, true)
    var got: List[i64] = []
    let a = str_slice(&s, 0, c1)
    let b = str_slice(&s, c1, c2)
    let c = str_slice(&s, c2, nc)
    for x in str_scan_feed(&mut st, &a)
        got.push(x)
    for x in str_scan_feed(&mut st, &b)
        got.push(x)
    for x in str_scan_feed(&mut st, &c)
        got.push(x)
    ok = ok and got == ov and a + b + c == s and str_scan_pos(&st) == (s.len_bytes() as i64)
    return ok

fn t_prop_slice(seed: i64) -> bool
    let s = t_gen(seed, seed % 17)
    let nc = s.len_chars() as i64
    var ok = true
    for k in 0..(nc + 1)
        let off = str_char_offset(&s, k)
        ok = ok and str_char_index(&s, off) == k and str_slice(&s, 0, k) + str_slice(&s, k, nc) == s
        ok = ok and s[0:off as usize] == str_slice(&s, 0, k)
    return ok

# upper and casefold give the same text chunk by chunk
fn t_prop_case(seed: i64) -> bool
    let s = t_gen(seed, seed % 31 + 2)
    let nc = s.len_chars() as i64
    let k = t_rnd(seed) % (nc + 1)
    let a = str_slice(&s, 0, k)
    let b = str_slice(&s, k, nc)
    let u = str_upper(&s) catch |e| t_err(e)
    let ua = str_upper(&a) catch |e| t_err(e)
    let ub = str_upper(&b) catch |e| t_err(e)
    let f = str_casefold(&s) catch |e| t_err(e)
    let fa = str_casefold(&a) catch |e| t_err(e)
    let fb = str_casefold(&b) catch |e| t_err(e)
    return u == ua + ub and f == fa + fb

fn t_prop_buf(seed: i64) -> bool
    var b = str_buf()
    var ps: List[text] = []
    var s = seed
    for _i in 0..(seed % 300)
        s = t_rnd(s)
        let p = t_gen(s, s % 9)
        str_buf_add(&mut b, &p)
        ps.push(p)
    let whole = "".join(ps)
    let t = str_buf_text(&b) catch |e| t_err(e)
    return t == whole and "".join(str_buf_chunks(&b)) == whole and str_buf_chars(&b) == (whole.len_chars() as i64)

fn t_prop_lines(seed: i64) -> bool
    let s = t_gen(seed, seed % 23)
    let keep = str_splitlines(&s, true)
    let ls = str_splitlines(&s, false)
    let r = str_rstrip(&s)
    return keep.len() == ls.len() and "".join(keep) == s and str_strip(&s) == str_lstrip(&r)

fn t_props(lo: i64, hi: i64) -> bool
    var ok = true
    for seed in lo..hi
        ok = ok and t_prop_search(seed) and t_prop_slice(seed) and t_prop_case(seed) and t_prop_lines(seed)
    return ok

fn t_bufs(lo: i64, hi: i64) -> bool
    var ok = true
    for seed in lo..hi
        ok = ok and t_prop_buf(seed)
    return ok

# appends across the run (64 pieces) and block (64 runs) boundaries
fn t_buf_many(n: i64) -> bool
    var b = str_buf()
    let ab = "ab"
    for _i in 0..n
        str_buf_add(&mut b, &ab)
    let want = str_repeat(&ab, n) catch |e| t_err(e)
    let t = str_buf_text(&b) catch |e| t_err(e)
    return t == want and str_buf_chars(&b) == 2 * n

# pieces of the given sizes (characters), each built by str_repeat
fn t_buf_big(ks: List[i64]) -> List[text]
    var b = str_buf()
    let x = "x"
    for k in ks
        let p = str_repeat(&x, k) catch |e| t_err(e)
        str_buf_add(&mut b, &p)
    var out: List[text] = []
    for c in str_buf_chunks(&b)
        out.push(c.len_chars().tostr())
    out.push(str_buf_text(&b) catch |e| t_err(e))
    return out

fn t_buf_cp(c: i64) -> text
    var b = str_buf()
    let _r = str_buf_add_cp(&mut b, c) catch ()
    return str_buf_text(&b) catch |e| t_err(e)

check t_props(1, 150)
check t_props(150, 300)
check t_props(300, 450)
check t_props(450, 600)
check t_bufs(1, 60)
check t_bufs(60, 120)
check t_buf_many(5000)
check t_buf_big([600000, 600000, 600000]) == ["600000", "600000", "600000", "ERR:too large:1800000"]
check t_buf_big([400000, 400000, 300000]) == ["800000", "300000", "ERR:too large:1100000"]
check t_buf_big([]) == [""]
check t_buf_cp(233) == "\\u{e9}" and t_buf_cp(55296) == ""
"""


def lit_text(t):
    o = '"'
    for ch in t:
        c = ord(ch)
        if ch == '"': o += '\\"'
        elif ch == '\\': o += '\\\\'
        elif ch == '\n': o += '\\n'
        elif ch == '\r': o += '\\r'
        elif ch == '\t': o += '\\t'
        elif 32 <= c < 127: o += ch
        else: o += '\\u{%x}' % c
    return o + '"'


def lit(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return lit_text(v)
    if isinstance(v, list):
        return '[' + ', '.join(lit(x) for x in v) + ']'
    if isinstance(v, dict) and 'err' in v:
        return lit_text('ERR:%s:%d' % (v['err'], v['n']))
    if isinstance(v, dict) and 'build' in v:
        return 't_build(%s, %s)' % (lit([p for p, _ in v['build']]), lit([k for _, k in v['build']]))
    raise ValueError(v)


def opt(v):
    return 'none' if v is None else 'some(%d)' % v


def expr(c):
    op, a, w = c['op'], c['args'], c['want']
    if op in ('find', 'rfind', 'count', 'replace', 'partition', 'rpartition', 'split_once', 'rsplit_once',
              'removeprefix', 'removesuffix', 'strips_chars'):
        call = 't_%s(%s, %s)' % (op, lit(a[0]), lit(a[1]))
        if op == 'replace':
            call = 't_replace(%s, %s, %s)' % (lit(a[0]), lit(a[1]), lit(a[2]))
        return call + ' == ' + lit(w)
    if op in ('find_in', 'rfind_in', 'count_in'):
        return 't_%s(%s, %s, %d, %d) == %d' % (op, lit(a[0]), lit(a[1]), a[2], a[3], w)
    if op == 'replace_n':
        return 't_replace_n(%s, %s, %s, %d) == %s' % (lit(a[0]), lit(a[1]), lit(a[2]), a[3], lit(w))
    if op in ('splitn', 'rsplitn'):
        return 't_%s(%s, %s, %d) == %s' % (op, lit(a[0]), lit(a[1]), a[2], lit(w))
    if op == 'splitlines':
        return 't_splitlines(%s, %s) == %s' % (lit(a[0]), lit(a[1]), lit(w))
    if op in ('repeat', 'zfill', 'expandtabs', 'char_at', 'char_offset', 'char_index'):
        return 't_%s(%s, %d) == %s' % (op, lit(a[0]), a[1], lit(w))
    if op == 'slice':
        return 't_slice(%s, %d, %d) == %s' % (lit(a[0]), a[1], a[2], lit(w))
    if op == 'slice_step':
        return 't_slice_step(%s, %s, %s, %d) == %s' % (lit(a[0]), opt(a[1]), opt(a[2]), a[3], lit(w))
    if op in ('center', 'ljust', 'rjust'):
        return 't_%s(%s, %d, %s) == %s' % (op, lit(a[0]), a[1], lit(a[2]), lit(w))
    if op in ('strips', 'preds', 'case', 'ord', 'words', 'wcase', 'slug', 'flags_each'):
        return 't_%s(%s) == %s' % (op, lit(a[0]), lit(w))
    if op == 'chr':
        return 't_chr(%d) == %s' % (a[0], lit(w))
    if op == 'finder_all':
        return 't_finder_all(%s, %s, %s) == %s' % (lit(a[0]), lit(a[1]), lit(a[2]), lit(w))
    if op == 'finder_find':
        return 't_finder_find(%s, %s, %d) == %d' % (lit(a[0]), lit(a[1]), a[2], w)
    if op == 'scan':
        return 't_scan(%s, %s, %s) == %s' % (lit(a[0]), lit(a[1]), lit(a[2]), lit(w))
    if op in ('big_find', 'big_rfind', 'big_count'):
        return 't_%s(%s, %s) == %d' % (op[4:], lit(a[0]), lit(a[1]), w)
    if op == 'big_count_in':
        return 't_count_in(%s, %s, %d, %d) == %d' % (lit(a[0]), lit(a[1]), a[2], a[3], w)
    if op == 'chunks':
        return 't_chunks(%s, %s) == %s' % (lit(a[0]), lit(a[1]), lit(w))
    if op == 'big_replace':
        return 't_digest(t_replace(%s, %s, %s)) == %s' % (lit(a[0]), lit(a[1]), lit(a[2]), lit(w)) if isinstance(w, str) else \
            't_replace(%s, %s, %s) == %s' % (lit(a[0]), lit(a[1]), lit(a[2]), lit(w))
    if op == 'big_replace_n':
        return 't_digest(t_replace_n(%s, %s, %s, %d)) == %s' % (lit(a[0]), lit(a[1]), lit(a[2]), a[3], lit(w)) if isinstance(w, str) else \
            't_replace_n(%s, %s, %s, %d) == %s' % (lit(a[0]), lit(a[1]), lit(a[2]), a[3], lit(w))
    if op == 'big_splitn':
        return 't_dlist(t_splitn(%s, %s, %d)) == %s' % (lit(a[0]), lit(a[1]), a[2], lit(w))
    if op == 'big_splitlines':
        return 't_dlist(t_splitlines(%s, false)) == %s' % (lit(a[0]), lit(w))
    if op == 'big_case3':
        return 't_dcase3(%s) == %s' % (lit(a[0]), lit(w))
    if op == 'big_title':
        return 't_dtitle(%s) == %s' % (lit(a[0]), lit(w))
    if op == 'big_expandtabs':
        return ('t_digest(t_expandtabs(%s, %d)) == %s' if isinstance(w, str) else 't_expandtabs(%s, %d) == %s') % (lit(a[0]), a[1], lit(w))
    if op == 'big_slug':
        return 't_digest(t_slug(%s)) == %s' % (lit(a[0]), lit(w))
    raise ValueError(op)


# fuel weight of one case, in units of about 100,000 ticks (a file has 500)
def weight(c):
    op = c['op']
    full = c['args'] and isinstance(c['args'][0], dict) and c['args'][0].get('build') == [['a', LIMIT]]
    if op == 'big_title':
        return 250
    if op in ('big_case3', 'chunks') or (op == 'big_splitn' and full):
        return 150
    if op == 'big_expandtabs':
        return 300 if c['args'][0]['build'][0][1] >= 300000 else 150
    if op.startswith('big_'):
        return 10
    if op == 'flags_each':
        return 8
    if op in ('case', 'wcase', 'slug', 'words'):
        return 1 + len(c['args'][0]) // 40
    return 1


def write_checks(cases, meta):
    groups = {}
    order = []
    for c in cases:
        g = 'big' if c['op'].startswith('big_') else c['op']
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(c)
    shards, cur, cw = [], [], 0
    for g in order:
        for c in groups[g]:
            w = weight(c)
            if cur and (cw + w > 300 or len(cur) >= 500):
                shards.append(cur); cur, cw = [], 0
            cur.append(c); cw += w
    if cur:
        shards.append(cur)
    old = [f for f in os.listdir(OUT) if f.startswith('str_checks')]
    for f in old:
        os.remove(os.path.join(OUT, f))
    src = 'CPython %s, Unicode %s, Django %s, change-case %s' % (meta['cpython'], meta['unicode'], meta['django'], meta['change_case'])
    for k, sh in enumerate(shards):
        name = 'str_checks.rce' if k == 0 else 'str_checks_%d.rce' % (k + 1)
        lines = [HEAD.replace('{src}', src)]
        for c in sh:
            lines.append('# %s %s' % (c['op'], c['src']))
            lines.append('check ' + expr(c))
        with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUT, 'str_checks_props.rce'), 'w', encoding='utf-8') as f:
        f.write(PROPS.lstrip('\n'))
    sys.stderr.write('%d checks in %d files, and str_checks_props.rce\n' % (len(cases), len(shards)))


def main():
    if '--refresh' in sys.argv:
        refresh()
    data = json.load(open(VEC, encoding='utf-8'))
    write_checks(data['cases'], data['meta'])


if __name__ == '__main__':
    main()
