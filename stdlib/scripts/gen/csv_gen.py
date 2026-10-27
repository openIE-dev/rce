# Generates the Unicode tables at the end of lib/data/csv.rce and the
# vectors of tests/data/csv_checks*.rce.
# Reference: CPython 3.13 (python3 -I scripts/gen/csv_gen.py; the run
# checks the version) and its csv module: csv.reader and csv.writer under
# each dialect, csv.DictReader, csv.Sniffer, float() and complex(); the
# csv-spectrum 2.0.0 files in vectors/data/csv-spectrum/ with their JSON.
# Seeded (random.Random), so the output is the same on every run.
# Usage: python3 -I scripts/gen/csv_gen.py   (rewrites the files in place)
import csv, io, json, os, random, re, sys, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# rce text literal: printable characters as they are, everything else
# (controls, format characters such as U+FEFF, separators) as \u{...}.
def lit(s):
    o = '"'
    for c in s:
        if c == '"': o += '\\"'
        elif c == '\\': o += '\\\\'
        elif c == '\n': o += '\\n'
        elif c == '\t': o += '\\t'
        elif c == '\r': o += '\\r'
        elif c == ' ' or (c.isprintable() and not 0xfff0 <= ord(c) <= 0xffff): o += c
        else: o += '\\u{%x}' % ord(c)
    return o + '"'

def lits(xs):
    return '[' + ', '.join(lit(x) for x in xs) + ']'

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
LIB = os.path.join(ROOT, 'lib/data/csv.rce')
TESTS = os.path.join(ROOT, 'tests/data')
SPECTRUM = os.path.join(ROOT, 'vectors/data/csv-spectrum')
assert sys.version_info[:2] == (3, 13), sys.version
assert unicodedata.unidata_version == '15.1.0'
MARK = '# ---------------- generated tables (scripts/gen/csv_gen.py) ----------------\n'

# ---------------- tables ----------------

def word_ranges():
    w = re.compile(r'\w')
    r, s = [], None
    for cp in range(0x110000):
        isw = bool(w.match(chr(cp)))
        assert isw == (chr(cp).isalnum() or chr(cp) == '_')
        if isw and s is None:
            s = cp
        if not isw and s is not None:
            r.append((s, cp - 1))
            s = None
    if s is not None:
        r.append((s, 0x10ffff))
    return [x for x in r if x[1] >= 128]

def digit_starts():
    d = [cp for cp in range(0x110000) if unicodedata.decimal(chr(cp), None) == 0]
    for s in d:
        assert all(unicodedata.decimal(chr(s + i), None) == i for i in range(10))
    assert len(d) * 10 == sum(1 for cp in range(0x110000) if unicodedata.decimal(chr(cp), None) is not None)
    return [s for s in d if s >= 128]

def ilist(name, xs):
    out = 'const %s: List[i64] = [\n' % name
    for i in range(0, len(xs), 12):
        out += '    ' + ', '.join(str(x) for x in xs[i:i + 12]) + ',\n'
    return out + ']\n'

def tables():
    wr = word_ranges()
    t = '# Python\'s \\w outside ASCII: ranges [WORD_LO[i], WORD_HI[i]] of\n'
    t += '# str.isalnum() (Unicode %s, CPython %d.%d.%d).\n' % ((unicodedata.unidata_version,) + sys.version_info[:3])
    t += ilist('WORD_LO', [a for a, b in wr]) + ilist('WORD_HI', [b for a, b in wr])
    t += '# First code point (digit zero) of each run of ten Unicode decimal digits\n'
    t += '# outside ASCII (unicodedata.decimal).\n'
    t += ilist('DIGIT0', digit_starts())
    sp = [cp for cp in range(0x110000) if chr(cp).isspace()]
    assert sp == [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 5760] + list(range(8192, 8203)) + [8232, 8233, 8239, 8287, 12288], sp
    t += '# The 127 ASCII characters csv.Sniffer counts (chr(0) to chr(126)).\n'
    t += 'const ASCII127: List[text] = [' + ', '.join(lit(chr(i)) for i in range(127)) + ']\n'
    return t

def write_tables():
    src = open(LIB).read()
    k = src.index(MARK)
    open(LIB, 'w').write(src[:k + len(MARK)] + tables())

# ---------------- a model of lib/data/csv.rce ----------------
# The model computes what CPython does not report (the column of an
# error); every run checks it against CPython on everything CPython does
# report: the rows, csv.reader.line_num after each, and each error.

class D:
    def __init__(self, delim=',', quote='"', escape='', dquote=True, skipsp=False, strict=False, quoting=0, eol='\r\n'):
        self.delim, self.quote, self.escape, self.dquote = delim, quote, escape, dquote
        self.skipsp, self.strict, self.quoting, self.eol = skipsp, strict, quoting, eol
    def kw(self):
        return dict(delimiter=self.delim, quotechar=self.quote or None, escapechar=self.escape or None,
                    doublequote=self.dquote, skipinitialspace=self.skipsp, strict=self.strict,
                    quoting=self.quoting, lineterminator=self.eol)
    def rce(self):
        return 'xdl(%s, %s, %s, %s, %s, %s, %d, %s)' % (lit(self.delim), lit(self.quote), lit(self.escape),
            str(self.dquote).lower(), str(self.skipsp).lower(), str(self.strict).lower(), self.quoting, lit(self.eol))

S_REC, S_FLD, S_ESC, S_IN, S_QIN, S_QESC, S_QQ, S_CRNL, S_AESC = range(9)
EOL = None
LIMIT_MSG = 'field larger than field limit (%d)'
BAD_UTF8 = 'invalid UTF-8 (replaced by U+FFFD)'

def py_float_ok(t):
    try:
        float(t)
        return True
    except ValueError:
        return False

def py_complex_ok(t):
    try:
        complex(t)
        return True
    except ValueError:
        return False

class Model:
    def __init__(self, d, univ=True, limit=1000000):
        self.d, self.univ, self.limit = d, univ, limit
        self.q = '' if d.quoting == 3 else d.quote
        self.st = S_REC; self.fields = []; self.quoted = []; self.buf = []; self.unq = False
        self.cr = False; self.skip = False; self.bol = True
        self.line = 0; self.col = 0; self.row = 0
        self.rows = []; self.errs = []
    def reset(self):
        self.st = S_REC; self.fields = []; self.quoted = []; self.buf = []; self.unq = False
    def add(self, c):
        if len(self.buf) >= self.limit:
            return LIMIT_MSG % self.limit
        self.buf.append(c)
        return ''
    def save(self):
        f = ''.join(self.buf); self.buf = []
        if self.unq and f != '' and self.d.quoting in (2, 4) and not py_float_ok(f):
            return 'could not convert string to float'
        self.fields.append(f); self.quoted.append(not self.unq)
        return ''
    def kind(self, c):
        d = self.d
        if c is EOL: return 'eol'
        if c == d.delim: return 'del'
        if self.q and c == self.q: return 'quo'
        if d.escape and c == d.escape: return 'esc'
        if c == '\n': return 'lf'
        if c == '\r': return 'cr'
        return 'oth'
    def step(self, c):
        k = self.kind(c); d = self.d
        nl = k in ('lf', 'cr', 'eol')
        if self.st == S_REC:
            if k == 'eol': return ''
            if k in ('lf', 'cr'):
                self.st = S_CRNL; return ''
            self.st = S_FLD
        st = self.st
        if st == S_FLD:
            self.unq = True
            if nl:
                self.st = S_REC if k == 'eol' else S_CRNL
                return self.save()
            if k == 'quo':
                self.unq = False; self.st = S_QIN; return ''
            if k == 'esc':
                self.st = S_ESC; return ''
            if d.skipsp and c == ' ': return ''
            if k == 'del': return self.save()
            self.st = S_IN
            return self.add(c)
        if st == S_ESC:
            if k in ('lf', 'cr'):
                self.st = S_AESC; return self.add(c)
            self.st = S_IN
            return self.add('\n' if k == 'eol' else c)
        if st in (S_IN, S_AESC):
            if k == 'eol' and st == S_AESC: return ''
            if nl:
                self.st = S_REC if k == 'eol' else S_CRNL
                return self.save()
            if k == 'esc':
                self.st = S_ESC; return ''
            if k == 'del':
                self.st = S_FLD; return self.save()
            return self.add(c)
        if st == S_QIN:
            if k == 'eol': return ''
            if k == 'esc':
                self.st = S_QESC; return ''
            if k == 'quo':
                self.st = S_QQ if d.dquote else S_IN; return ''
            return self.add(c)
        if st == S_QESC:
            self.st = S_QIN
            return self.add('\n' if k == 'eol' else c)
        if st == S_QQ:
            if k == 'quo':
                self.st = S_QIN; return self.add(c)
            if k == 'del':
                self.st = S_FLD; return self.save()
            if nl:
                self.st = S_REC if k == 'eol' else S_CRNL
                return self.save()
            if not d.strict:
                self.st = S_IN; return self.add(c)
            return "'%s' expected after '%s'" % (d.delim, d.quote)
        if k in ('lf', 'cr'): return ''
        if k == 'eol':
            self.st = S_REC; return ''
        return "new-line character seen in unquoted field - do you need to open the file with newline=''?"
    def fail(self, why, col):
        self.row += 1
        self.errs.append((self.row, self.line, col, why))
        self.reset(); self.skip = True
    def emit(self):
        self.row += 1
        self.rows.append((self.row, self.line, self.fields, self.quoted))
        self.fields = []; self.quoted = []
    def eol(self):
        e = self.step(EOL)
        if e:
            self.fail(e, self.col + 1); self.skip = False
        elif self.st == S_REC:
            self.emit()
    def feed(self, text, start=0):
        for i, c in enumerate(text):
            if (start + i) in self.want:
                ln = self.line + 1 if (self.bol or self.cr) else self.line
                cl = 1 if (self.bol or self.cr) else self.col + 1
                self.errs.append((self.row + 1, ln, cl, BAD_UTF8))
            if self.cr:
                self.cr = False
                if c == '\n':
                    if self.skip:
                        self.skip = False
                    else:
                        self.col += 1
                        e = self.step(c)
                        if e:
                            self.fail(e, self.col); self.skip = False
                        else:
                            self.eol()
                    self.bol = True
                    continue
                if self.skip: self.skip = False
                else: self.eol()
                self.bol = True
            if self.bol:
                self.bol = False; self.line += 1; self.col = 0
            if self.skip:
                self.col += 1
                if c == '\n':
                    self.skip = False; self.bol = True
                elif self.univ and c == '\r':
                    self.cr = True
                continue
            k = self.kind(c)
            self.col += 1
            e = self.step(c)
            if e:
                self.fail(e, self.col)
                if k == 'lf':
                    self.skip = False; self.bol = True
                elif k == 'cr' and self.univ:
                    self.cr = True
            elif k == 'lf':
                self.eol(); self.bol = True
            elif k == 'cr' and self.univ:
                self.cr = True
    want = ()  # positions of characters that replaced ill-formed bytes
    def finish(self):
        if self.cr:
            self.cr = False
            if self.skip: self.skip = False
            else: self.eol()
            self.bol = True
        elif not self.bol:
            if self.skip: self.skip = False
            else: self.eol()
            self.bol = True
        if self.buf or self.st == S_QIN:
            if self.d.strict:
                self.row += 1
                self.errs.append((self.row, self.line, self.col + 1, 'unexpected end of data'))
                self.reset()
            else:
                e = self.save()
                if e:
                    self.row += 1
                    self.errs.append((self.row, self.line, self.col + 1, e))
                    self.reset()
                else:
                    self.emit(); self.reset()

def model(data, d, univ=True, limit=1000000):
    m = Model(d, univ, limit)
    m.feed(data)
    m.finish()
    return m

def cpython(data, d, univ=True, limit=None):
    """Outcomes of csv.reader in order: ('row', line_num, values) or ('err', line_num, message)."""
    old = csv.field_size_limit()
    if limit is not None:
        csv.field_size_limit(limit)
    try:
        r = csv.reader(io.StringIO(data, newline='' if univ else '\n'), **d.kw())
        out = []
        while True:
            try:
                v = next(r)
                out.append(('row', r.line_num, v))
            except StopIteration:
                break
            except (csv.Error, ValueError) as e:
                out.append(('err', r.line_num, str(e)))
        return out
    finally:
        csv.field_size_limit(old)

def cell_val(f, q, quoting):
    if not q and f == '' and quoting in (4, 5): return None
    if not q and f != '' and quoting in (2, 4): return float(f)
    return f

def same_val(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (a != a and b != b) or a == b or (a == b == 0)
    return type(a) == type(b) and a == b

def agree(data, d, univ=True, limit=None):
    """The model's outcome, after checking it against CPython."""
    m = model(data, d, univ, 1000000 if limit is None else limit)
    ref = cpython(data, d, univ, limit)
    mine = sorted([('row', r[0], r) for r in m.rows] + [('err', e[0], e) for e in m.errs], key=lambda x: x[1])
    assert len(mine) == len(ref), (data, d.kw(), mine, ref)
    for (k, _, x), (k2, ln, v) in zip(mine, ref):
        assert k == k2, (data, d.kw(), mine, ref)
        if k == 'row':
            assert x[1] == ln, (data, d.kw(), mine, ref)
            vals = [cell_val(f, q, d.quoting) for f, q in zip(x[2], x[3])]
            assert len(vals) == len(v) and all(same_val(a, b) for a, b in zip(vals, v)), (data, d.kw(), vals, v)
        else:
            assert x[1] == ln, (data, d.kw(), mine, ref)
            if x[3].startswith('could not convert'):
                assert v.startswith(x[3]), (data, v)
            else:
                assert x[3] == v, (data, d.kw(), x, v)
    return m

# ---------------- check files ----------------

HEAD = """# Generated by scripts/gen/csv_gen.py; do not edit.
# Expected values come from CPython %s's csv module (see the generator).
fn xdl(delim: text, quote: text, escape: text, dq: bool, sk: bool, st: bool, qm: i64, eol: text) -> CsvDialect = CsvDialect(delim: delim, quote: quote, escape: escape, dquote: dq, skipsp: sk, strict: st, quoting: qm, eol: eol)
fn xq(m: text) -> List[bool]
    var q: List[bool] = []
    for c in m.chars()
        q.push(c == "1")
    return q
fn xrw(row: u64, line: u64, fs: List[text], m: text) -> CsvRow = CsvRow(row: row, line: line, fields: fs, quoted: xq(m))
fn xer(row: u64, line: u64, col: u64, why: text) -> CsvErr = CsvErr(row: row, line: line, col: col, why: why)
fn xout(rs: List[CsvRow], es: List[CsvErr]) -> CsvOut = CsvOut(rows: rs, errs: es)
fn xadd(o: &mut CsvOut, p: CsvOut)
    for x in p.rows
        o.rows.push(copy(x))
    for x in p.errs
        o.errs.push(copy(x))
""" % '.'.join(map(str, sys.version_info[:3]))

READ_HEAD = """# Chunks of t cut at the character positions in cuts (ascending), through
# csv_feed and csv_finish.
fn xfeed(t: &text, d: &CsvDialect, univ: bool, lim: u64, cuts: &[usize]) -> CsvOut
    var r = mk_reader(d, univ)
    csv_set_limit(&mut r, lim)
    var o = xout([], [])
    let cs = t.chars()
    var a: usize = 0
    for c in cuts
        let piece = "".join(&cs[a:c])
        xadd(&mut o, csv_feed(&mut r, &piece))
        a = c
    let rest = "".join(&cs[a:cs.len()])
    xadd(&mut o, csv_feed(&mut r, &rest))
    xadd(&mut o, csv_finish(&mut r))
    return o
# The state machine alone, without the whole-line fast path.
fn xslow(t: &text, d: &CsvDialect, univ: bool, lim: u64) -> CsvOut
    var r = mk_reader(d, univ)
    csv_set_limit(&mut r, lim)
    let cs = t.chars()
    return rd_chars(&mut r, &cs, true)
# The same records and errors whole, through the state machine alone, and
# in chunks.
fn xrd(t: text, d: CsvDialect, univ: bool, lim: u64, cuts: List[usize], want: CsvOut) -> bool
    let one: List[usize] = []
    let whole = xfeed(&t, &d, univ, lim, &one)
    let sl = xslow(&t, &d, univ, lim)
    let ch = xfeed(&t, &d, univ, lim, &cuts)
    return whole == want and sl == want and ch == want
"""

def rce_out(rows, errs):
    rs = ', '.join('xrw(%d, %d, %s, %s)' % (r[0], r[1], lits(r[2]), lit(''.join('1' if q else '0' for q in r[3]))) for r in rows)
    es = ', '.join('xer(%d, %d, %d, %s)' % (e[0], e[1], e[2], lit(e[3])) for e in errs)
    return 'xout([%s], [%s])' % (rs, es)

def cuts_of(R, n, k):
    if n == 0:
        return []
    return sorted(set(R.randint(0, n) for _ in range(k)))

def read_check(R, data, d, univ=True, limit=None, k=3):
    m = agree(data, d, univ, limit)
    cuts = cuts_of(R, len(data), k)
    return 'check xrd(%s, %s, %s, %d, [%s], %s)' % (lit(data), d.rce(), str(univ).lower(),
        1000000 if limit is None else limit, ', '.join(map(str, cuts)), rce_out(m.rows, m.errs))

EXCEL = D()
DIALECTS = [
    ('excel', D()), ('excel strict', D(strict=True)), ('excel_tab', D(delim='\t')),
    ('unix', D(quoting=1, eol='\n')), ('skipinitialspace', D(skipsp=True)),
    ('escapechar', D(escape='\\')), ('escapechar, no doublequote', D(escape='\\', dquote=False)),
    ('QUOTE_NONE with escapechar', D(quoting=3, escape='\\')), ('QUOTE_NONE, no quotechar', D(quoting=3, quote='')),
    ('QUOTE_NONNUMERIC', D(quoting=2)), ('QUOTE_STRINGS', D(quoting=4)), ('QUOTE_NOTNULL', D(quoting=5)),
    ('space delimiter, skipinitialspace', D(delim=' ', skipsp=True)), ('semicolon and apostrophe', D(delim=';', quote="'")),
    ('strict, escapechar, no doublequote', D(strict=True, escape='\\', dquote=False)),
    ('non-ASCII delimiter and quote', D(delim='¦', quote='«')), ('pipe delimiter', D(delim='|', skipsp=True, strict=True)),
]
ALPHA = ['a', 'b', 'x', '1', '2', '.', 'e', '-', ',', ';', '|', '¦', '«', '"', "'", '\\', ' ', ' ', '\t',
         '\n', '\n', '\r', '\r\n', '\r\n', 'é', '€', '😀', '\x00', '\ufeff']

HAND = [
    # RFC 4180 section 2
    'aaa,bbb,ccc\r\nzzz,yyy,xxx\r\n', 'aaa,bbb,ccc\r\nzzz,yyy,xxx', '"aaa","bbb","ccc"\r\nzzz,yyy,xxx\r\n',
    '"aaa","b\r\nbb","ccc"\r\nzzz,yyy,xxx\r\n', '"aaa","b""bb","ccc"\r\n', 'field_name,field_name,field_name\r\naaa,bbb,ccc\r\n',
    # line ends: LF, CRLF, lone CR, mixed, blank lines, none at the end
    'a,b\nc,d\n', 'a,b\rc,d\r', 'a,b\r\rc', 'a\r\n\r\nb', '\n\n', '\r', '', ',', ',,\n', 'a,', '"",""\n', '""',
    # quotes inside and after fields, unterminated
    'a"b,c\n', '"a"b,c\n', '"a" ,b\n', '"a\n', '"a""\n', 'a,"b\r\nc', '"""",x\n', '"a"""b",c\n',
    # NUL and a byte-order mark are ordinary characters
    'a\x00b,c\n', '\ufeffa,b\n',
    # embedded line ends in quoted fields, CR alone inside
    '"a\rb",c\r\n', '"a\r\nb\nc",d', '"\n",\n',
]
ESC_HAND = ['a\\,b,c\n', 'a\\\nb\nc\n', 'a\\\r\nb\n', 'a\\\rb\n', 'a\\', '"a\\"b",c\n', '"a\\\nb",c\n', '\\\n\n', 'a\\\\,b\n', '"a"\\b,c\n']

def gen_read(R):
    """Shards of reader checks: (title, lines)."""
    out = []
    lines = ['# hand cases: RFC 4180 section 2, line ends, quotes, NUL, BOM (excel and strict)']
    for t in HAND:
        lines.append(read_check(R, t, D()))
        lines.append(read_check(R, t, D(strict=True)))
    lines.append('# escape characters (escapechar, QUOTE_NONE)')
    for t in ESC_HAND:
        for d in (D(escape='\\'), D(escape='\\', dquote=False), D(quoting=3, escape='\\')):
            lines.append(read_check(R, t, d))
    lines.append('# quotes after a closing quote without doublequote (the quote is then an ordinary character)')
    for t in ['"a"""\n', '"a""b"\n', '"a""b,c"\n', '"a"" ,b"\n', '"a"b"c,d\n', '""""\n', '"a",""""\n', '"a""\n"b\n']:
        for d in (D(dquote=False), D(dquote=False, strict=True), D(dquote=False, skipsp=True)):
            lines.append(read_check(R, t, d))
    lines.append('# lines that end only at LF (io.StringIO(data) without newline="", as has_header reads)')
    for t in ['a,b\rc\n', 'a\r\nb\n', '"a\rb"\n', 'a\r\r\nb', 'x\ry\nz\n']:
        lines.append(read_check(R, t, D(), univ=False))
    lines.append('# field size limit (csv.field_size_limit(4))')
    for t in ['abcd,e\n', 'abcde,f\ng\n', '"abcdef"\nxy\n', 'ab\\\ncd\n', 'a,"bcd\nefgh"\n,z']:
        lines.append(read_check(R, t, D(escape='\\'), limit=4))
    out.append(lines)
    for name, d in DIALECTS:
        lines = ['# seeded random input under %s' % name]
        for _ in range(26):
            n = R.randint(0, 22)
            t = ''.join(R.choice(ALPHA) for _ in range(n))
            lines.append(read_check(R, t, d, univ=R.random() < 0.9, limit=R.choice([None, None, None, 3])))
        out.append(lines)
    # lines with quote characters and no escape character, the lines the
    # reader takes whole by pieces, against the state machine and CPython
    for name, d in [('excel', D()), ('excel strict', D(strict=True)), ('no doublequote', D(dquote=False)),
                    ('skipinitialspace', D(skipsp=True)), ('QUOTE_NONNUMERIC', D(quoting=2)), ('QUOTE_STRINGS', D(quoting=4)),
                    ('QUOTE_NOTNULL', D(quoting=5)), ('non-ASCII quote, strict', D(delim=';', quote='«', strict=True))]:
        lines = ['# seeded lines full of quote characters under %s' % name]
        al = [d.quote, d.quote, d.quote, d.delim, d.delim, ' ', 'a', '1', '.', '\n', '\r\n', 'é']
        for _ in range(25):
            t = ''.join(R.choice(al) for _ in range(R.randint(0, 24)))
            lines.append(read_check(R, t, d, limit=R.choice([None, None, None, None, 3])))
        out.append(lines)
    # tables written by csv.writer, read back
    lines = ['# tables written by csv.writer under each dialect, read back']
    for name, d in DIALECTS:
        if d.quoting in (2, 4):
            continue
        for _ in range(3):
            rows = [[''.join(R.choice(ALPHA) for _ in range(R.randint(0, 5))) for _ in range(R.randint(1, 4))] for _ in range(R.randint(1, 4))]
            buf = io.StringIO()
            try:
                csv.writer(buf, **d.kw()).writerows(rows)
            except csv.Error:
                continue
            lines.append(read_check(R, buf.getvalue(), d))
    out.append(lines)
    return out

def write_file(name, body):
    open(os.path.join(TESTS, name), 'w').write(body)

def shard(blocks, base, head, per):
    """Put the blocks (lists of lines, the first a comment) in files of about per checks."""
    files, cur, n = [], [], 0
    for b in blocks:
        for line in b:
            if line.startswith('check') and n >= per:
                files.append(cur); cur, n = [], 0
            if line.startswith('check'):
                n += 1
            cur.append(line)
    if cur:
        files.append(cur)
    return files

# ---------------- writer ----------------

def w_model(cells, d):
    """join_append of _csv.c: the record text, or (error, 1-based field)."""
    parts, size, nf, null = [], 0, 0, False
    def field(f, null, quoted):
        if f == '' and d.delim == ' ' and d.skipsp:
            if d.quoting == 3 or (null and d.quoting in (4, 5)):
                return None, 'empty field must be quoted if delimiter is a space and skipinitialspace is true'
            quoted = True
        o = ''
        for c in f:
            esc = False
            if c == d.delim or (d.escape and c == d.escape) or (d.quote and c == d.quote) or c in '\n\r' or c in d.eol:
                if d.quoting == 3:
                    esc = True
                else:
                    if d.quote and c == d.quote:
                        if d.dquote:
                            o += d.quote
                        else:
                            esc = True
                    elif d.escape and c == d.escape:
                        esc = True
                    if not esc:
                        quoted = True
                if esc:
                    if not d.escape:
                        return None, 'need to escape, but no escapechar set'
                    o += d.escape
            o += c
        return (d.quote + o + d.quote if quoted else o), None
    for c in cells:
        if c is None:
            q, f, null = d.quoting in (2, 1), '', True
        elif isinstance(c, str):
            q, f, null = d.quoting in (2, 1, 4, 5), c, False
        else:
            q, f, null = d.quoting in (1, 5), str(c), False
        t, e = field(f, null, q)
        if e:
            return None, (e, nf + 1)
        if nf > 0:
            parts.append(d.delim)
        parts.append(t)
        nf += 1
    if nf == 1 and parts == ['']:
        if d.quoting == 3 or (null and d.quoting in (4, 5)):
            return None, ('single empty field record must be quoted', 1)
        t, e = field('', True, True)
        if e:
            return None, (e, 1)
        parts = [t]
    return ''.join(parts) + d.eol, None

def w_cpython(cells, d):
    buf = io.StringIO()
    try:
        csv.writer(buf, **d.kw()).writerow(cells)
        return buf.getvalue(), None
    except csv.Error as e:
        return None, str(e)

def rce_cell(c):
    if c is None:
        return 'CsvNull'
    if isinstance(c, str):
        return 'CsvStr(t: %s)' % lit(c)
    return 'CsvNum(t: %s)' % lit(str(c))

W_HEAD = """fn xr(fs: List[text]) -> List[text] = fs
fn xw(cells: List[CsvCell], d: CsvDialect) -> text = csv_write_cells(&cells, &d) catch |e| "ERR " + e.col.tostr() + " " + copy(e.why)
fn xws(rows: List[List[text]], d: CsvDialect) -> text = csv_write(&rows, &d) catch |e| "ERR " + e.row.tostr() + " " + e.col.tostr() + " " + copy(e.why)
fn xde(d: CsvDialect) -> text = csv_dialect_error(&d)
"""

W_ALPHA = ['a', 'b', ' ', ',', ';', '"', "'", '\\', '\n', '\r', '\t', 'é', '😀', '|', 'x', '']
W_DIALECTS = DIALECTS + [
    ('lineterminator "\\n"', D(eol='\n')), ('lineterminator "!"', D(eol='!')), ('lineterminator "ab"', D(eol='ab')),
    ('QUOTE_NONE with escapechar and quotechar', D(quoting=3, escape='\\')),
    ('QUOTE_NONE, no escapechar', D(quoting=3)), ('space delimiter, QUOTE_NONE, escapechar', D(delim=' ', skipsp=True, quoting=3, escape='\\')),
    ('space delimiter, QUOTE_NOTNULL', D(delim=' ', skipsp=True, quoting=5)), ('space delimiter, QUOTE_STRINGS', D(delim=' ', skipsp=True, quoting=4)),
]

def write_check(cells, d):
    t, e = w_model(cells, d)
    rt, re_ = w_cpython(cells, d)
    if e is None:
        assert rt == t and re_ is None, (cells, d.kw(), t, rt, re_)
        want = t
    else:
        assert rt is None and re_ == e[0], (cells, d.kw(), e, rt, re_)
        want = 'ERR %d %s' % (e[1], e[0])
    return 'check xw([%s], %s) == %s' % (', '.join(rce_cell(c) for c in cells), d.rce(), lit(want))

def rand_cell(R, typed):
    x = R.random()
    if typed and x < 0.15:
        return None
    if typed and x < 0.3:
        return R.choice([0, 1, -7, 1.5, -0.25, 1e20, 3.14159, float('inf'), 10 ** 12])
    return ''.join(R.choice(W_ALPHA) for _ in range(R.randint(0, 4)))

def gen_write(R):
    blocks = []
    lines = ['# csv.writer.writerow, hand cases']
    for cells, d in [
            (['a', 'b'], D()), (['a,b', 'c"d', 'e\nf', 'g\rh'], D()), ([''], D()), ([], D()), (['', ''], D()),
            ([None], D()), ([None], D(quoting=5)), ([None], D(quoting=4)), ([''], D(quoting=3)), (['a b'], D(delim=' ')),
            (['', 'a'], D(delim=' ', skipsp=True)), ([None, 'a'], D(delim=' ', skipsp=True, quoting=5)),
            (['x"y'], D(dquote=False, escape='\\')), (['x"y'], D(dquote=False)), (['a\\b'], D(escape='\\')),
            (['a,b'], D(quoting=3, escape='\\')), (['a,b'], D(quoting=3)), (['a"b'], D(quoting=3, escape='\\')),
            ([1, 2.5, 'x', None], D(quoting=2)), ([1, 'x', None, ''], D(quoting=4)), ([1, 'x', None, ''], D(quoting=5)),
            (['abc'], D(eol='b')), (['a!b'], D(eol='!')), (['é,😀'], D(delim='¦', quote='«'))]:
        lines.append(write_check(cells, d))
    blocks.append(lines)
    for name, d in W_DIALECTS:
        lines = ['# seeded random records under %s' % name]
        typed = d.quoting in (2, 4, 5) or R.random() < 0.3
        for _ in range(14):
            cells = [rand_cell(R, typed) for _ in range(R.randint(0, 4))]
            lines.append(write_check(cells, d))
        blocks.append(lines)
    lines = ['# csv.writer.writerows (several records in one text)']
    for name, d in W_DIALECTS[:10]:
        rows = [[rand_cell(R, False) for _ in range(R.randint(0, 3))] for _ in range(R.randint(1, 4))]
        buf = io.StringIO()
        try:
            csv.writer(buf, **d.kw()).writerows(rows)
            want = buf.getvalue()
        except csv.Error as e:
            k = next(i for i, r in enumerate(rows) if w_model(r, d)[1])
            want = 'ERR %d %d %s' % (k + 1, w_model(rows[k], d)[1][1], str(e))
        lines.append('check xws([%s], %s) == %s' % (', '.join('xr(%s)' % lits(r) for r in rows), d.rce(), lit(want)))
    blocks.append(lines)
    # dialect validation
    lines = ['# dialect validation: csv.reader(io.StringIO(""), **dialect) and its error']
    combos = []
    for delim in [',', '', 'ab', '\r', '\n', ' ', '"', '\\', ';', '!']:
        combos.append(D(delim=delim))
    for quote in ['', "'", 'ab', ' ', '\n', ',', '!']:
        combos.append(D(quote=quote))
        combos.append(D(quote=quote, quoting=3))
        combos.append(D(quote=quote, skipsp=True))
    for esc in ['\\', 'ab', ' ', '\r', '"', ',', '!']:
        combos.append(D(escape=esc))
        combos.append(D(escape=esc, skipsp=True))
    for qm in [-1, 0, 1, 2, 3, 4, 5, 6]:
        combos.append(D(quoting=qm))
    for eol in ['\n', '', '!', ',', '"\n', '\\']:
        combos.append(D(eol=eol))
        combos.append(D(eol=eol, escape='\\'))
    for _ in range(30):
        ch = ['', ',', ';', '"', "'", ' ', '\\', '\n', '!', 'ab']
        combos.append(D(delim=R.choice(ch[1:]), quote=R.choice(ch), escape=R.choice(ch), skipsp=R.random() < 0.5,
                        quoting=R.randint(0, 5), eol=R.choice(['\r\n', '\n', '!', ';'])))
    for d in combos:
        try:
            csv.reader(io.StringIO(''), **d.kw())
            want = ''
        except (TypeError, ValueError) as e:
            want = str(e)
        lines.append('check xde(%s) == %s' % (d.rce(), lit(want)))
    blocks.append(lines)
    return blocks

# ---------------- sniffing ----------------

SN_HEAD = """fn xsn0(s: &text, ds: text?) -> text ! CsvError
    let d = try csv_sniff(s, ds)
    return d.delim + "|" + d.quote + "|" + d.dquote.tostr() + "|" + d.skipsp.tostr()
fn xsn(s: text, ds: text?) -> text = xsn0(&s, ds) catch |e| "ERR " + copy(e.why)
fn xhh0(s: &text) -> text ! CsvError
    let h = try csv_has_header(s)
    return h.tostr()
fn xhh(s: text) -> text = xhh0(&s) catch |e| "ERR " + copy(e.why)
"""

def sn_ref(sample, delims):
    try:
        d = csv.Sniffer().sniff(sample, delims)
        return '%s|%s|%s|%s' % (d.delimiter, d.quotechar, str(bool(d.doublequote)).lower(), str(bool(d.skipinitialspace)).lower())
    except csv.Error as e:
        return 'ERR ' + str(e)

def hh_ref(sample):
    try:
        return str(csv.Sniffer().has_header(sample)).lower()
    except StopIteration:
        raise AssertionError('has_header raised StopIteration on %r' % sample)
    except Exception as e:
        return 'ERR ' + str(e)

WORDS = ['name', 'id', 'city', 'price', 'qty', 'date', 'note', 'Zoë', 'naïve', '東京', 'a b', 'x,y', 'say "hi"', "it's"]

def rand_value(R, kind):
    if kind == 'int':
        return str(R.randint(-50, 99999))
    if kind == 'float':
        return R.choice(['%.2f' % (R.random() * 1000), '1e%d' % R.randint(-5, 5), '-0.5', '3j', '1+2j', 'inf', 'nan'])
    if kind == 'date':
        return '2026-%02d-%02d' % (R.randint(1, 12), R.randint(1, 28))
    if kind == 'code':
        return ''.join(R.choice('ABCDEF') for _ in range(4))
    return R.choice(WORDS + ['', ' ', 'hello world', 'O\'Brien', 'say ""no""', 'a\nb', 'é€😀', 'x;y', 'p|q', 'tab\there'])

def rand_table_sample(R):
    delim = R.choice([',', ',', ';', '\t', '|', ':', ' ', '#', '/'])
    quote = R.choice(['"', '"', "'"])
    quoting = R.choice([csv.QUOTE_MINIMAL, csv.QUOTE_MINIMAL, csv.QUOTE_ALL, csv.QUOTE_NONNUMERIC])
    eol = R.choice(['\r\n', '\n', '\r'])
    ncol = R.randint(1, 6)
    kinds = [R.choice(['int', 'float', 'date', 'code', 'str']) for _ in range(ncol)]
    rows = []
    if R.random() < 0.7:
        rows.append([R.choice(WORDS) for _ in range(ncol)])
    for _ in range(R.randint(1, 30)):
        row = [rand_value(R, k) for k in kinds]
        if quoting == csv.QUOTE_NONNUMERIC:
            row = [float(v) if k in ('int',) else v for v, k in zip(row, kinds)]
        if R.random() < 0.1:
            row = row[:R.randint(0, ncol)]
        rows.append(row)
    buf = io.StringIO()
    try:
        csv.writer(buf, delimiter=delim, quotechar=quote, quoting=quoting, lineterminator=eol,
                   escapechar=None if quoting != csv.QUOTE_NONE else '\\').writerows(rows)
    except csv.Error:
        return None
    t = buf.getvalue()
    if R.random() < 0.3:
        t = t.replace(delim, delim + ' ')
    if R.random() < 0.3:
        t = t[:R.randint(0, len(t))]
    return t

def gen_sniff(R):
    blocks = []
    lines = ['# csv.Sniffer.sniff and has_header, hand cases']
    hand = ['a,b,c\n1,2,3\n4,5,6\n', 'a;b;c\n1;2;3\n', 'name\tage\nann\t3\nbob\t40\n', '"a","b"\n"c","d"\n',
            "'a', 'b'\n'c', 'd'\n", '"x""y",z\n', 'abc\n', '', '\n\n', '1 2 3\n4 5 6\n', 'a:b\n1:2\n', '"a"\n"b"\n',
            'a,b\r\n1,2\r\n', 'a,b\r1,2\r', 'h1,h2\n"a\nb",c\n', 'x|y\n1|2\n3|4\n', '"a"b,c\n', 'a, "b", c\n',
            ',"",,\n', '"\'",\'"\'\n']
    for t in hand:
        lines.append('check xsn(%s, none) == %s' % (lit(t), lit(sn_ref(t, None))))
        lines.append('check xhh(%s) == %s' % (lit(t), lit(hh_ref(t))))
    # the vote reads rows 2 to 22 only: a change in row 15 counts, one in row 23 does not
    nums = ''.join('%d,%d\n' % (i, i * 7) for i in range(2, 15))
    for t in ['a,b\n' + nums + 'x,y\n', 'a,b\n' + nums + ''.join('%d,%d\n' % (i, i) for i in range(15, 23)) + 'x,y\n',
              'abc,de\n' + ''.join('xy,z%d\n' % (i % 10) for i in range(2, 16)) + 'longer,zz\n']:
        lines.append('check xhh(%s) == %s' % (lit(t), lit(hh_ref(t))))
    for t, ds in [('a,b;c\n1,2;3\n', ';'), ('a,b;c\n1,2;3\n', ','), ('a,b\n1,2\n', ''), ('a b|c\n1 2|3\n', '| '), ('"a";"b"\n', ',')]:
        lines.append('check xsn(%s, some(%s)) == %s' % (lit(t), lit(ds), lit(sn_ref(t, ds))))
    blocks.append(lines)
    lines = ['# seeded random tables written by csv.writer (delimiters, quote characters,',
             '# quoting modes, line ends, spaces after delimiters, truncated samples)']
    n = 0
    while n < 150:
        t = rand_table_sample(R)
        if t is None:
            continue
        lines.append('check xsn(%s, none) == %s' % (lit(t), lit(sn_ref(t, None))))
        lines.append('check xhh(%s) == %s' % (lit(t), lit(hh_ref(t))))
        if R.random() < 0.2:
            ds = ''.join(R.sample([',', ';', '\t', '|', ':', ' '], 2))
            lines.append('check xsn(%s, some(%s)) == %s' % (lit(t), lit(ds), lit(sn_ref(t, ds))))
        n += 1
    blocks.append(lines)
    lines = ['# seeded random text']
    al = ['a', 'b', '1', ',', ';', ':', ' ', '\t', '"', "'", '\n', '\n', '\r\n', '|', 'é', '_', '-', '.']
    for _ in range(60):
        t = ''.join(R.choice(al) for _ in range(R.randint(0, 60)))
        lines.append('check xsn(%s, none) == %s' % (lit(t), lit(sn_ref(t, None))))
        lines.append('check xhh(%s) == %s' % (lit(t), lit(hh_ref(t))))
    blocks.append(lines)
    return blocks

# ---------------- number syntax ----------------

NUM_HEAD = """# For each text, two digits: float(t) and complex(t) succeed (1) or not (0).
fn xnum(ts: List[text]) -> text
    var o = ""
    for t in ts
        o = o + (if csv_float_ok(&t) then "1" else "0") + (if csv_complex_ok(&t) then "1" else "0")
    return o
"""

def gen_num(R):
    lines = ['# float() and complex() of str, CPython: hand cases, then seeded random text']
    hand = ['1', '-1', '+1.5', '1.', '.5', '.', '1e5', '1E-5', '1e', '1e+', 'e5', '1_000', '1__0', '_1', '1_', '1_e5', '1e1_0',
            'inf', '-INF', 'Infinity', 'infin', 'nan', '+nan', '-nAn', 'nan1', '  1  ', '\t1\n', '1 2', '', ' ', '\x0c1\x0b',
            '\x1c1', '1\x1c', '\u20001\u3000', '\xa01', '١٢٣', '١.٥', '𝟏', '1j', '1J', 'j', '+j', '-j', '1+j', '1-2j', '1+2', '(1)',
            '( 1+2j )', '(1+2j', '1+2j)', '()', 'infj', 'nanj', '1e5j', '1+infj', '+1-1e5J', '0x10', '1,5', '١_٢', '1\x00',
            '½', '²', '-', '+', '+-1', '1e5.5', '00', '0_0', '1.5_5', '1._5', '(j)', ' ( j ) ', '(\t1)', 'jj', '1jj', '1+-2j',
            '1ej', '1e+j', '1E-J', '1+1ej', '2e_1j', 'infej', '.e1', '1.e1', '-.5e-5j']
    def ref(t):
        return '%d%d' % (py_float_ok(t), py_complex_ok(t))
    for i in range(0, len(hand), 12):
        b = hand[i:i + 12]
        lines.append('check xnum(%s) == %s' % (lits(b), lit(''.join(ref(t) for t in b))))
    al = ['0', '1', '9', '.', 'e', 'E', '+', '-', '_', 'j', 'J', '(', ')', ' ', '\t', 'i', 'n', 'f', 'a', 't', 'y',
          '٣', '\u2000', '\x1c', 'x']
    for _ in range(40):
        b = [''.join(R.choice(al) for _ in range(R.randint(0, 7))) for _ in range(12)]
        lines.append('check xnum(%s) == %s' % (lits(b), lit(''.join(ref(t) for t in b))))
    return [lines]

# ---------------- header rows, csv-spectrum, bytes ----------------

TB_HEAD = """fn xr(fs: List[text]) -> List[text] = fs
fn xm(ks: List[text], vs: List[text]) -> Map[text, text]
    var m: Map[text, text] = {}
    var i: usize = 0
    while i < ks.len() bound 1000
        m.insert(copy(ks[i]), copy(vs[i]))
        i += 1
    return m
fn xtab(h: List[text], recs: List[Map[text, text]], es: List[CsvErr]) -> CsvTable = CsvTable(header: h, recs: recs, errs: es)
fn xtb(t: text, d: CsvDialect) -> CsvTable = csv_table(&t, &d) catch |e| CsvTable(header: [copy(e.why)], recs: [], errs: [])
# The same table through csv_feed_bytes in chunks and csv_records.
fn xtbb(b: List[i64], cuts: List[usize], d: CsvDialect) -> CsvTable
    let o = xby(b, cuts, d)
    var k: usize = 0
    while k < o.rows.len() and o.rows[k].fields.len() == 0 bound 1000
        k += 1
    if k >= o.rows.len()
        return CsvTable(header: [], recs: [], errs: [])
    var rest: List[CsvRow] = []
    var j = k + 1
    while j < o.rows.len() bound 1000
        rest.push(copy(o.rows[j]))
        j += 1
    return csv_records(&o.rows[k].fields, &rest)
fn xby(b: List[i64], cuts: List[usize], d: CsvDialect) -> CsvOut
    var r = mk_reader(&d, true)
    var o = xout([], [])
    var a: usize = 0
    var ends = copy(cuts)
    ends.push(b.len())
    for c in ends
        var piece: List[i64] = []
        var i = a
        while i < c bound 100000
            piece.push(b[i])
            i += 1
        xadd(&mut o, csv_feed_bytes(&mut r, &piece))
        a = c
    xadd(&mut o, csv_finish(&mut r))
    return o
"""

def table_of(m):
    """csv_table's result from the model's rows and errors."""
    rows = list(m.rows)
    k = 0
    while k < len(rows) and rows[k][2] == []:
        k += 1
    if k == len(rows):
        return [], [], sorted(m.errs)
    h = rows[k][2]
    recs, errs = [], []
    for w in rows[k + 1:]:
        n = len(w[2])
        if n == 0:
            continue
        if n != len(h):
            errs.append((w[0], w[1], min(n, len(h)) + 1, 'expected %d fields, found %d' % (len(h), n)))
        else:
            recs.append((list(h), list(w[2])))
    return h, recs, sorted(m.errs + errs)

def rce_map(ks, vs):
    return 'xm(%s, %s)' % (lits(ks), lits(vs))

def rce_table(h, recs, errs):
    return 'xtab(xr(%s), [%s], [%s])' % (lits(h), ', '.join(rce_map(k, v) for k, v in recs),
        ', '.join('xer(%d, %d, %d, %s)' % (e[0], e[1], e[2], lit(e[3])) for e in errs))

def dictreader_agrees(t, d, h, recs):
    """csv.DictReader gives the same dicts for the rows of the header's width."""
    dr = csv.DictReader(io.StringIO(t, newline=''), **d.kw())
    got = [x for x in dr if None not in x and None not in x.values()]
    want = [dict(zip(k, v)) for k, v in recs]
    assert got == want, (t, got, want)

def table_check(t, d):
    m = agree(t, d)
    h, recs, errs = table_of(m)
    if not m.errs and m.rows and m.rows[0][2] != []:
        dictreader_agrees(t, d, h, recs)
    return 'check xtb(%s, %s) == %s' % (lit(t), d.rce(), rce_table(h, recs, errs))

def seq_at(b, i):
    x = b[i]
    if x < 128: return 1
    if 194 <= x <= 223: lo, hi, n = 128, 191, 2
    elif x == 224: lo, hi, n = 160, 191, 3
    elif x == 237: lo, hi, n = 128, 159, 3
    elif 225 <= x <= 239: lo, hi, n = 128, 191, 3
    elif x == 240: lo, hi, n = 144, 191, 4
    elif 241 <= x <= 243: lo, hi, n = 128, 191, 4
    elif x == 244: lo, hi, n = 128, 143, 4
    else: return -1
    for k in range(1, n):
        j = i + k
        if j >= len(b): return -k
        l, h = (lo, hi) if k == 1 else (128, 191)
        if not l <= b[j] <= h: return -k
    return n

def decode_marks(b0):
    b = b0[3:] if b0[:3] == b'\xef\xbb\xbf' else b0
    out, marks, i = '', set(), 0
    while i < len(b):
        n = seq_at(b, i)
        if n > 0:
            out += b[i:i + n].decode(); i += n
        else:
            marks.add(len(out)); out += '\ufffd'; i += -n
    assert out == b0.decode('utf-8-sig', 'replace'), (b0, out)
    return out, marks

def bytes_model(b, d):
    t, marks = decode_marks(b)
    agree(t, d)
    m = Model(d)
    m.want = marks
    m.feed(t)
    m.finish()
    return m

def gen_tables(R):
    blocks = []
    lines = ['# csv-spectrum 2.0.0 (vectors/data/csv-spectrum): each file read as text by',
             '# csv_table and as bytes in chunks by csv_feed_bytes and csv_records, against its JSON']
    for name in sorted(os.listdir(os.path.join(SPECTRUM, 'csvs'))):
        b = open(os.path.join(SPECTRUM, 'csvs', name), 'rb').read()
        js = json.load(open(os.path.join(SPECTRUM, 'json', name[:-4] + '.json')))
        if isinstance(js, dict):
            js = [js]
        t = b.decode('utf-8')
        m = agree(t, D())
        h, recs, errs = table_of(m)
        got = [dict(zip(k, v)) for k, v in recs]
        if name == 'location_coordinates.csv':
            # the JSON of this file gives another phone number than the file holds
            assert len(got) == 1 and got[0]['Contact Phone Number'] == '2095257564' and js[0]['Contact Phone Number'] == '1234567890'
            assert {k: v for k, v in got[0].items() if k != 'Contact Phone Number'} == {k: v for k, v in js[0].items() if k != 'Contact Phone Number'}
        else:
            assert got == js, (name, got, js)
        assert not errs
        lines.append('# %s' % name)
        lines.append('check xtb(%s, csv_excel()) == %s' % (lit(t), rce_table(h, recs, errs)))
        cuts = cuts_of(R, len(b), 4)
        lines.append('check xtbb([%s], [%s], csv_excel()) == %s' % (', '.join(map(str, b)), ', '.join(map(str, cuts)), rce_table(h, recs, errs)))
    blocks.append(lines)
    lines = ['# header rows: csv_table against csv.DictReader; ragged rows are errors']
    hand = ['a,b\n1,2\n3\n4,5,6\n', '\n\na,b\n\n1,2\n', 'a,a\n1,2\n', 'a\n', '', 'a,b\r\n"x,y",z\r\n,\r\n',
            'k\n"unterminated', 'a,b\n1,"2"x\n3,4\n']
    for t in hand:
        lines.append(table_check(t, D()))
    lines.append(table_check('a,b\n1,"2"x\n3,4\n', D(strict=True)))
    for _ in range(40):
        d = R.choice([D(), D(delim=';'), D(delim='\t'), D(skipsp=True), D(strict=True)])
        ncol = R.randint(1, 4)
        rows = [[R.choice(WORDS) for _ in range(ncol)]]
        for _ in range(R.randint(0, 6)):
            k = ncol if R.random() < 0.75 else R.randint(0, ncol + 2)
            rows.append([''.join(R.choice(W_ALPHA[:-1]) for _ in range(R.randint(0, 3))) for _ in range(k)])
        buf = io.StringIO()
        csv.writer(buf, **d.kw()).writerows(rows)
        t = buf.getvalue()
        if R.random() < 0.2:
            t = t[:R.randint(0, len(t))]
        lines.append(table_check(t, d))
    blocks.append(lines)
    lines = ['# UTF-8 bytes in chunks: sequences cut by a chunk end, a byte-order mark,',
             '# ill-formed bytes replaced by U+FFFD and reported where they are']
    pieces = [b'a', b',', b'"', b'\n', b'\r\n', b' ', 'é'.encode(), '€'.encode(), '😀'.encode(),
              b'\x80', b'\xff', b'\xc0\xaf', b'\xe2\x82', b'\xf0\x9f\x98', b'\xed\xa0\x80', b'\xef\xbb\xbf', 'b1'.encode()]
    hand_b = [b'\xef\xbb\xbfa,b\n', b'\xef\xbb', b'\xef\xbbX', b'a,\xff\n\xe2\x82', '😀,€\n'.encode(), b'"a\x80b",c\n',
              b'\xef\xbb\xbf\xef\xbb\xbfx\n', b'']
    for b in hand_b:
        m = bytes_model(b, D())
        cuts = sorted(set(range(1, len(b))))
        lines.append('check xby([%s], [%s], csv_excel()) == %s' % (', '.join(map(str, b)), ', '.join(map(str, cuts)), rce_out(m.rows, m.errs)))
    for _ in range(40):
        d = R.choice([D(), D(strict=True), D(delim=';')])
        b = b''.join(R.choice(pieces) for _ in range(R.randint(0, 14)))
        m = bytes_model(b, d)
        cuts = cuts_of(R, len(b), 5)
        lines.append('check xby([%s], [%s], %s) == %s' % (', '.join(map(str, b)), ', '.join(map(str, cuts)), d.rce(), rce_out(m.rows, m.errs)))
    blocks.append(lines)
    return blocks

API = r"""# ---- the public functions, hand cases
fn xread(t: text, d: CsvDialect) -> List[List[text]] = csv_read(&t, &d) catch |e| [xr(["ERR", e.row.tostr(), e.line.tostr(), e.col.tostr(), copy(e.why)])]
fn xrdr0(d: &CsvDialect) -> u64 ! CsvError
    let r = try csv_reader(d)
    return csv_line_num(&r)
fn xrdr(d: CsvDialect) -> text
    let n = xrdr0(&d) catch |_e| 999
    return if n == 999 then "refused" else n.tostr()
fn xstream() -> bool
    let d = csv_excel()
    var r = csv_reader(&d) catch |_e| mk_reader(&d, true)
    let t1 = "a,b\r"
    let t2 = "\nc,\"d"
    let t3 = "\ne\"\n"
    let t4 = "x\n"
    let a = csv_feed(&mut r, &t1)
    let b = csv_feed(&mut r, &t2)
    let c = csv_feed(&mut r, &t3)
    let e = csv_finish(&mut r)
    let f = csv_feed(&mut r, &t4)
    return a.rows.len() == 0 and b.rows == [xrw(1, 1, ["a", "b"], "00")] and c.rows == [xrw(2, 3, ["c", "d\ne"], "01")] and e.rows.len() == 0 and f.rows.len() == 0 and csv_line_num(&r) == 3
fn xcells(t: text, d: CsvDialect) -> List[CsvCell]
    let o = csv_parse(&t, &d) catch |_e| xout([], [])
    if o.rows.len() == 0
        return []
    return csv_cells(&o.rows[0], &d)
fn xrecs() -> CsvTable
    let h = ["a", "b"]
    let rows = [xrw(2, 2, ["1", "2"], "00"), xrw(3, 3, [], ""), xrw(4, 5, ["x"], "1"), xrw(5, 6, ["p", "q", "r"], "000")]
    return csv_records(&h, &rows)
fn xmany(n: usize, s: text) -> text
    var ps: List[text] = []
    var i: usize = 0
    while i < n bound 1000000
        ps.push(copy(s))
        i += 1
    return "".join(&ps)
# A field longer than one text, fed in chunks, is an error, and the next line reads.
fn xhuge() -> bool
    let d = csv_excel()
    var r = mk_reader(&d, true)
    let t1 = "\""
    let t2 = "\"\nok\n"
    let a = csv_feed(&mut r, &t1)
    let big = xmany(400000, "a")
    let b = csv_feed(&mut r, &big)
    let c = csv_feed(&mut r, &big)
    let e = csv_feed(&mut r, &big)
    let f = csv_feed(&mut r, &t2)
    let g = csv_finish(&mut r)
    return a.rows.len() + b.rows.len() + c.rows.len() == 0 and e.errs == [xer(1, 1, 1000002, "field larger than field limit (1000000)")] and f.rows == [xrw(2, 2, ["ok"], "0")] and g.rows.len() == 0
# Plain and quoted records over three chunks of 300,000 characters (more than one text).
fn xbig() -> bool
    let d = csv_excel()
    var r = mk_reader(&d, true)
    let p1 = xmany(20000, "abc,def,123,x\n")
    let p2 = xmany(12000, "\"abc\",\"de,f\",\"12\"\"3\",x\n")
    let a = csv_feed(&mut r, &p1)
    let b = csv_feed(&mut r, &p2)
    let c = csv_feed(&mut r, &p1)
    let e = csv_finish(&mut r)
    return a.rows.len() == 20000 and b.rows.len() == 12000 and c.rows.len() == 20000 and e.rows.len() == 0 and b.rows[11999].fields == ["abc", "de,f", "12\"3", "x"] and b.rows[11999].row == 32000 and csv_line_num(&r) == 52000
fn xwbig() -> text
    let f = xmany(600000, "a")
    let rows = [xr([copy(f)]), xr([copy(f)])]
    let d = csv_excel()
    let ls = csv_write_lines(&rows, &d) catch |_e| []
    let one = csv_write(&rows, &d) catch |e| "ERR " + e.row.tostr() + " " + copy(e.why)
    let q = xmany(600000, "\"")
    let qs = [q]
    let two = csv_write_row(&qs, &d) catch |e| "ERR " + e.col.tostr() + " " + copy(e.why)
    return ls.len().tostr() + " " + one + " / " + two
check xread("a,b\r\n1,2\r\n", csv_excel()) == [xr(["a", "b"]), xr(["1", "2"])]
check xread("a,\"b\"x\n1\n", csv_rfc4180()) == [xr(["ERR", "1", "1", "6", "',' expected after '\"'"])]
check xread("a\t\"b c\"\n", csv_excel_tab()) == [xr(["a", "b c"])]
check xread("a\t\"b c\"\n", csv_tsv()) == [xr(["a", "\"b c\""])]
check xread("\"a\",\"b\"\n", csv_unix()) == [xr(["a", "b"])]
check xread("x\n", xdl("", "\"", "", true, false, false, 0, "\r\n")) == [xr(["ERR", "0", "0", "0", "\"delimiter\" must be a 1-character string"])]
check xrdr(csv_excel()) == "0"
check xrdr(xdl(",", ",", "", true, false, false, 0, "\r\n")) == "refused"
check xstream()
check xcells("1,,\"\",\"x\"\n", xdl(",", "\"", "", true, false, false, 4, "\r\n")) == [CsvNum(t: "1"), CsvNull, CsvStr(t: ""), CsvStr(t: "x")]
check xcells("1,,\"\",x\n", xdl(",", "\"", "", true, false, false, 5, "\r\n")) == [CsvStr(t: "1"), CsvNull, CsvStr(t: ""), CsvStr(t: "x")]
check xcells("1.5,,\"a\"\n", xdl(",", "\"", "", true, false, false, 2, "\r\n")) == [CsvNum(t: "1.5"), CsvStr(t: ""), CsvStr(t: "a")]
check xcells("1,x\n", xdl(",", "\"", "", true, false, false, 2, "\r\n")) == []
check xrecs() == CsvTable(header: ["a", "b"], recs: [xm(["a", "b"], ["1", "2"])], errs: [xer(4, 5, 2, "expected 2 fields, found 1"), xer(5, 6, 3, "expected 2 fields, found 3")])
fn xok() -> bool
    let d = csv_excel()
    return csv_dialect_error(&d) == ""
check xok()
# An ill-formed byte is replaced in the chunk it ends, a cut sequence waits for the next chunk.
fn xbytes_now() -> bool
    let d = csv_excel()
    var r = mk_reader(&d, true)
    let b1 = [97, 255]
    let b2 = [98, 226]
    let b3 = [130, 172, 10]
    let a = csv_feed_bytes(&mut r, &b1)
    let b = csv_feed_bytes(&mut r, &b2)
    let c = csv_feed_bytes(&mut r, &b3)
    let e = csv_finish(&mut r)
    return a.errs == [xer(1, 1, 2, "invalid UTF-8 (replaced by U+FFFD)")] and b.errs.len() == 0 and c.rows == [xrw(1, 1, ["a\u{fffd}b\u{20ac}"], "0")] and e.rows.len() == 0
check xbytes_now()
"""
# checks heavy enough to have a file each (one comptime fuel budget per file)
API_BIG = ['check xhuge()', 'check xbig()',
           'check xwbig() == "2 ERR 2 output larger than a text (1000000 characters) / ERR 1 record larger than a text (1000000 characters)"']

def main():
    write_tables()
    R = random.Random(20261025)
    files = []
    for lines in shard(gen_read(R), 'csv_checks', HEAD + READ_HEAD, 60):
        files.append(HEAD + READ_HEAD + '\n'.join(lines) + '\n')
    for lines in shard(gen_write(R), 'csv_checks', HEAD + W_HEAD, 200):
        files.append(HEAD + W_HEAD + '\n'.join(lines) + '\n')
    for lines in shard(gen_sniff(R), 'csv_checks', HEAD + SN_HEAD, 120):
        files.append(HEAD + SN_HEAD + '\n'.join(lines) + '\n')
    for lines in shard(gen_num(R), 'csv_checks', HEAD + NUM_HEAD, 200):
        files.append(HEAD + NUM_HEAD + '\n'.join(lines) + '\n')
    for lines in shard(gen_tables(R), 'csv_checks', HEAD + TB_HEAD, 120):
        files.append(HEAD + TB_HEAD + '\n'.join(lines) + '\n')
    k = API.index('check ')
    files.append(HEAD + TB_HEAD + API)
    for c in API_BIG:
        files.append(HEAD + TB_HEAD + API[:k] + c + '\n')
    for i, body in enumerate(files):
        write_file('csv_checks%s.rce' % ('' if i == 0 else '_%d' % (i + 1)), body)
    print('%d check files' % len(files))

if __name__ == '__main__':
    main()
