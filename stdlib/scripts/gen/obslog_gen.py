# Usage (from stdlib/): python3 scripts/gen/obslog_gen.py
# Writes vectors/obs/log_ref.json (inputs and reference outputs) and the
# generated checks tests/obs/log_checks_2.rce (logfmt), _3 (sampling and RFC
# 3339 times) and _4 (JSON and records) of
# lib/obs/log.rce. Derived vectors, seed 20261027:
# - logfmt: github.com/go-logfmt/logfmt v0.6.0 run by scripts/gen/obslog_go
#   (Go 1.24, `go run .`): MarshalKeyvals of random pairs (with refusals),
#   EncodeKeyval pair by pair (a refused key writes nothing), the Decoder on
#   single lines (round trips of the encoded lines, random and mutated
#   lines, every syntax error) and on multi-line texts (LF, CRLF, empty
#   lines, a final CR; Go stops at the first error, so only the lines up to
#   it are compared);
# - sampling: zapcore.NewSamplerWithOptions of go.uber.org/zap v1.27.0 with
#   a SamplerHook, levels Debug..Error as severities 5, 9, 13, 17;
# - JSON: CPython json.dumps(dict(pairs), ensure_ascii=..., separators=(",",
#   ":")), a number lexeme spliced in where json.scanner.NUMBER_RE accepts
#   it; records through the pair rule of log_kvs restated here (time text
#   from datetime, the OpenTelemetry short names), their JSON from json.dumps
#   and their logfmt from go-logfmt;
# - RFC 3339: the text of random instants from datetime (fraction digits
#   truncated, fixed offsets), and parsing restated as a regular expression
#   with the date checked by datetime and the instant in integers.
import json, os, random, re, subprocess, sys
from datetime import datetime, timedelta, timezone
from json.scanner import NUMBER_RE

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rcelit import lit
from _rn import esc

ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
R = random.Random(20261027)
I64MIN, I64MAX = -2**63, 2**63 - 1

def ilit(v):
    return '(0 - 9223372036854775807 - 1)' if v == I64MIN else str(v)

def L(s):
    return esc(lit(s))

ALPHA = list("abcxyz019 =\"\\/'u-") + ['\n', '\r', '\t', '\x01', '\x1f', '\x7f', '\x08', '\x0c', '\x00', 'é', '€', '😀', '\ufffd', '\u2028', '日']
LFA = list('abk= "\\ \tu0d83DCnx') + ['é', '\ufffd', '😀', '\x01', '\x7f', "'", '/']

def rstr(alpha, lo, hi):
    return ''.join(R.choice(alpha) for _ in range(R.randint(lo, hi)))

def rval():
    t = R.random()
    if t < 0.7:
        return ['s', R.choice(['null', '', 'plain', 'x=y', '"q"']) if R.random() < 0.15 else rstr(ALPHA, 0, 8)]
    if t < 0.8:
        return ['i', str(R.choice([0, -1, 42, I64MIN, I64MAX, R.randint(-10**12, 10**12)]))]
    if t < 0.9:
        return ['b', R.choice(['true', 'false'])]
    return ['n', '']

def rkey():
    if R.random() < 0.1:
        return R.choice(['', ' ', '=', '"', '\ufffd', ' =\n'])
    return rstr(ALPHA, 1, 6)

# ---------------- inputs ----------------
enc_in = []
enc_in.append([['a', 's', 'null'], ['b', 's', ''], ['c', 'n', ''], ['d', 'i', str(I64MIN)], ['e', 'i', str(I64MAX)], ['f', 'b', 'false']])
enc_in.append([['k\ufffdey', 's', 'v\ufffd'], ['x y', 's', 'a b']])
enc_in.append([['ok', 's', '1'], [' ', 's', 'refused']])
enc_in.append([['k', 's', 'tab\there'], ['q', 's', 'say "hi" \\ back'], ['c', 's', '\x00\x01\x1f\x7f\x08\x0c\r\n']])
enc_in.append([['u', 's', 'é€😀日\u2028'], ['eq', 's', 'a=b']])
for _ in range(220):
    enc_in.append([[rkey()] + rval() for _ in range(R.randint(1, 5))])

dec_in = [
    '', '   ', 'a', 'a b c', 'a=', 'a= b', 'a=""', 'a="" b', '=a', 'a==b', 'a=b=c', 'a="x', 'a="x\\"',
    'a="\\q"', 'a="\x01\\n"', 'a="\x01"', '\ufffdk=1', 'k\ufffd', 'k\ufffd=1', 'é=1', 'a"b=1', 'a=b"c', 'a="x"b=2',
    'a="x""', 'a="\\ud800"', 'a="\\ud800\\u0041"', 'a="\\udc00\\ud800"', 'a="\\ud83d\\ude00"', 'a="\\u12"', 'a="\\u12g4"',
    'a="\\/\\\'\\b\\f\\t\\r\\n"', '  a  ', 'a=1\tb=2\x01c=3', 'a="\\u00E9\\u00e9"', 'a="\\ud83d"', 'a="\\ud83dx"',
    'a="\\\\"', 'a="\\', 'a="\\"', 'k=v\x7f', '\x7f=1', 'a=\ufffd', 'a="\\ufffd"', 'a="é\\n"', 'a=\x00b',
]
for e in enc_in[:60]:
    pass  # filled after the first Go run (round trips)
for _ in range(170):
    s = rstr(LFA, 0, 14)
    dec_in.append(s.rstrip('\r'))

def go(payload):
    p = subprocess.run(['go', 'run', '.'], cwd=os.path.join(HERE, 'obslog_go'), input=json.dumps(payload).encode(), capture_output=True, check=True)
    return json.loads(p.stdout)

first = go({'encode': enc_in, 'encode_skip': [], 'decode': [], 'streams': [], 'sample': []})
lines = [s for s in first['encode'] if s is not None]
for s in lines:
    dec_in.append(s)
for s in lines[:80]:
    t = list(s)
    for _ in range(R.randint(1, 2)):
        if R.random() < 0.5 and t:
            del t[R.randrange(len(t))]
        else:
            t.insert(R.randint(0, len(t)), R.choice(['"', '=', ' ', '\\', 'u']))
    dec_in.append(''.join(t).rstrip('\r'))
dec_in = [d for d in dec_in if '\n' not in d]

streams = ['', '\n', 'a=1', 'a=1\n', 'a=1\r\n', 'a=1\r', 'a=1\r\r\nb=2', '\n\na=1\n\n', 'a=1\nb="x\nc=3', 'x=1\r\n\r\ny="\\q"\nz=1']
pool = [d for d in dec_in if d != ''] + lines
for _ in range(40):
    parts = [R.choice(pool) if R.random() < 0.85 else '' for _ in range(R.randint(1, 6))]
    s = ''
    for i, p in enumerate(parts):
        s += p
        if i + 1 < len(parts) or R.random() < 0.5:
            s += R.choice(['\n', '\r\n'])
    if R.random() < 0.1:
        s += '\r'
    streams.append(s)

def fnv(s):
    h = 2166136261
    for b in s.encode():
        h = ((h ^ b) * 16777619) & 0xffffffff
    return h

# two messages in one zap counter bucket
msgs = ['request served', 'cache miss', 'retrying', 'é']
seen = {}
coll = None
for i in range(200000):
    m = 'm%d' % i
    b = fnv(m) % 4096
    if b in seen:
        coll = (seen[b], m)
        break
    seen[b] = m
samples = []
for k in range(36):
    tick = R.choice([0, 1, 5, 100, 1000])
    first_n = R.choice([0, 1, 2, 3, 10])
    there = R.choice([0, 1, 2, 5])
    t = R.choice([0, 7, -50, 10**18])
    ents = []
    ms = msgs + (list(coll) if k % 3 == 0 else [])
    for _ in range(R.randint(20, 90)):
        t += R.choice([0, 0, 1, 2, 3, 20, 200])
        ents.append([t, R.choice([-1, 0, 1, 2]), R.choice(ms)])
    samples.append({'tick': tick, 'first': first_n, 'thereafter': there, 'entries': ents})

# ---------------- records (pair rule restated) ----------------
NAMES = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL']
def sev_name(s):
    if s < 1 or s > 24:
        return str(s)
    k = (s - 1) % 4
    return NAMES[(s - 1) // 4] + ('' if k == 0 else str(k + 1))

def time_text(t, frac, off):
    if t > 9 * 10**18 or t < -9 * 10**18:
        return str(t)
    off = max(-1439, min(1439, off))
    frac = max(0, min(9, frac))
    secs, ns = divmod(t, 10**9)
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=secs + off * 60)
    out = dt.strftime('%Y-%m-%dT%H:%M:%S')
    if dt.year < 1000:
        out = '%04d' % dt.year + out[out.index('-'):]
    if frac > 0:
        out += '.' + ('%09d' % ns)[:frac]
    if off == 0:
        return out + 'Z'
    a = abs(off)
    return out + ('-' if off < 0 else '+') + '%02d:%02d' % (a // 60, a % 60)

def cut(s, mx):
    b = s.encode()
    if len(b) <= mx:
        return s
    k = max(mx, 0)
    while k > 0 and (b[k] & 0xC0) == 0x80:
        k -= 1
    return b[:k].decode()

def dedupe(pairs):
    out = []
    at = {}
    for k, ty, v in pairs:
        if k in at:
            out[at[k]] = [k, ty, v]
        else:
            at[k] = len(out)
            out.append([k, ty, v])
    return out

def rec_pairs(rec, f):
    mx = max(1, min(10**6, f['max']))
    ps = []
    if rec['time'] is not None:
        ps.append(['time', 's', time_text(rec['time'], f['frac'], f['off'])])
    ps.append(['level', 's', sev_name(rec['sev'])])
    ps.append(['msg', 's', cut(rec['msg'], mx)])
    for k, ty, v in rec['kvs']:
        if k in ('time', 'level', 'msg'):
            k = 'fields.' + k
        ps.append([cut(k, mx), ty, cut(v, mx) if ty in ('s', 'num') else v])
    return dedupe(ps)

NUMS = ['0', '-0', '1.5', '1e10', '1E+2', '-0.0e-0', '12345678901234567890', '01', '1.', '.5', '+1', '1e', 'NaN', 'Infinity', '--1', '1.5e3x', '', '-', '0x10', ' 1']
def json_line(pairs, ascii):
    d = {}
    raws = {}
    for i, (k, ty, v) in enumerate(pairs):
        if ty == 's':
            d[k] = v
        elif ty == 'i':
            d[k] = int(v)
        elif ty == 'b':
            d[k] = v == 'true'
        elif ty == 'n':
            d[k] = None
        elif NUMBER_RE.fullmatch(v):
            ph = '@@num%d@@' % i
            raws[ph] = v
            d[k] = ph
        else:
            d[k] = v
    out = json.dumps(d, ensure_ascii=ascii, separators=(',', ':'))
    for ph, v in raws.items():
        # a later value of the same key may have replaced the placeholder
        assert out.count('"%s"' % ph) == (1 if ph in d.values() else 0)
        out = out.replace('"%s"' % ph, v)
    return out

json_in = []
JKEYS = ['a', 'b', 'k', 'é', 'x y', '"q"', '\n', '😀', '', '\x7f']
for _ in range(200):
    ps = []
    for _ in range(R.randint(1, 6)):
        k = R.choice(JKEYS) if R.random() < 0.6 else rstr(ALPHA, 0, 5)
        if R.random() < 0.15:
            ps.append([k, 'num', R.choice(NUMS)])
        else:
            ty, v = rval()
            ps.append([k, ty, v])
    json_in.append({'pairs': ps, 'ascii': R.random() < 0.5})

recs = []
for _ in range(110):
    t = R.choice([None, None, 0, -1, 1700000000123456789, R.randint(-9 * 10**18, 9 * 10**18), 9 * 10**18 + 1, I64MIN, I64MAX, -86400 * 10**9 * 365 * 200])
    kvs = []
    for _ in range(R.randint(0, 5)):
        k = R.choice(['time', 'level', 'msg', 'fields.time', 'user', 'user', 'n', 'x y', '', '=']) if R.random() < 0.5 else rstr(ALPHA, 1, 5)
        if R.random() < 0.1:
            kvs.append([k, 'num', R.choice(NUMS)])
        else:
            ty, v = rval()
            kvs.append([k, ty, v])
    rec = {'time': t, 'sev': R.choice([1, 5, 9, 10, 13, 17, 21, 24, 0, -3, 25, 99]), 'msg': rstr(ALPHA, 0, 10), 'kvs': kvs}
    f = {'frac': R.choice([9, 9, 0, 3, 6, -1, 12]), 'off': R.choice([0, 0, 60, -330, 2000, -2000]), 'ascii': R.random() < 0.5, 'max': R.choice([65536, 65536, 5, 1, 0, 3])}
    recs.append({'rec': rec, 'fmt': f, 'pairs': rec_pairs(rec, f)})

def go_pair(p):
    k, ty, v = p
    if ty == 'num':
        return [k, 's', v]
    return [k, ty, v]

ref = go({'encode': enc_in, 'encode_skip': enc_in + [[go_pair(p) for p in r['pairs']] for r in recs], 'decode': dec_in, 'streams': streams, 'sample': samples})
skip_rec = ref['encode_skip'][len(enc_in):]
for r, s in zip(recs, skip_rec):
    r['logfmt'] = s
    r['json'] = json_line(r['pairs'], r['fmt']['ascii'])
for j in json_in:
    j['json'] = json_line(j['pairs'], j['ascii'])

# RFC 3339 times: text of random instants (datetime), and parsing, with
# the date checked by datetime.date and the instant by integer arithmetic.
RFC = re.compile(r'(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(?:([Zz])|([+-])(\d{2}):(\d{2}))')
def parse_ref(s):
    m = RFC.fullmatch(s)
    if not m:
        return None
    y, mo, d, h, mi, se = (int(m.group(k)) for k in range(1, 7))
    try:
        day = datetime(y, mo, d).toordinal() - datetime(1970, 1, 1).toordinal()
    except ValueError:
        return None
    if h > 23 or mi > 59 or se > 59 or abs(day) > 106748:
        return None
    off = 0
    if m.group(9):
        oh, om = int(m.group(10)), int(m.group(11))
        if oh > 23 or om > 59:
            return None
        off = (oh * 60 + om) * (-1 if m.group(9) == '-' else 1)
    ns = int((m.group(7) or '').ljust(9, '0'))
    return day * 86400 * 10**9 + ((h * 60 + mi) * 60 + se) * 10**9 + ns - off * 60 * 10**9

time_fmt = []
for _ in range(60):
    t = R.choice([R.randint(-9 * 10**18, 9 * 10**18), R.randint(-10**12, 10**12), 0, -1, 9 * 10**18, -9 * 10**18])
    frac, off = R.choice([0, 1, 3, 6, 9, -2, 15]), R.choice([0, 1, -1, 330, -570, 1439, -1439, 5000])
    time_fmt.append([t, frac, off, time_text(t, frac, off)])
time_parse = ['1970-01-01T00:00:00Z', '1970-01-01t00:00:00z', '2024-02-29T12:00:00.5+01:00', '2023-02-29T00:00:00Z', '2024-13-01T00:00:00Z',
    '2024-01-01T23:59:60Z', '2024-01-01T24:00:00Z', '2024-01-01T00:00:00.1234567891Z', '2024-01-01T00:00:00.Z', '2024-01-01T00:00:00',
    '2024-01-01T00:00:00+24:00', '2024-01-01T00:00:00+05:60', '2024-01-01 00:00:00Z', '0001-01-01T00:00:00Z', '9999-12-31T23:59:59Z',
    '1677-09-21T00:12:43.145224192Z', '2262-04-11T23:47:16.854775807Z', '1677-09-24T00:00:00Z', '2262-04-08T00:00:00Z', '2024-01-01T00:00:00Zx',
    '+2024-01-01T00:00:00Z', '2024-1-01T00:00:00Z', '2024-01-01T00:00:00-00:00', '1969-12-31T23:59:59.999999999Z', '2000-02-29T00:00:00-23:59']
for _ in range(40):
    t = R.randint(-9 * 10**18, 9 * 10**18)
    txt = time_text(t, R.choice([0, 3, 6, 9, 1, 2]), R.choice([0, 60, -330, 1439]))
    if R.random() < 0.2:
        txt = txt.replace('T', 't').replace('Z', 'z')
    time_parse.append(txt)
time_parse = [[x, parse_ref(x)] for x in time_parse]

os.makedirs(os.path.join(ROOT, 'vectors', 'obs'), exist_ok=True)
with open(os.path.join(ROOT, 'vectors', 'obs', 'log_ref.json'), 'w') as fh:
    json.dump({
        'source': 'scripts/gen/obslog_gen.py (seed 20261027): go-logfmt v0.6.0 and zap v1.27.0 through scripts/gen/obslog_go, CPython json',
        'encode': [{'pairs': e, 'marshal': m, 'keyvals': k} for e, m, k in zip(enc_in, ref['encode'], ref['encode_skip'])],
        'decode': [{'line': l, 'pairs': d['pairs'], 'err': d['err']} for l, d in zip(dec_in, ref['decode'])],
        'streams': [{'text': t, 'records': s['records'], 'err': s['err']} for t, s in zip(streams, ref['streams'])],
        'sample': [dict(s, keep=k) for s, k in zip(samples, ref['sample'])],
        'json': json_in,
        'records': recs,
        'time_text': time_fmt,
        'time_parse': time_parse,
    }, fh, ensure_ascii=False, indent=0)

# ---------------- checks ----------------
HEAD = '''# Generated by scripts/gen/obslog_gen.py; do not edit.
fn gk(k: text, ty: text, v: text) -> LogKv
    if ty == "i"
        return LogKv(key: k, val: LvInt(v: olg_dec(&v).v))
    if ty == "b"
        return LogKv(key: k, val: LvBool(v: v == "true"))
    if ty == "n"
        return LogKv(key: k, val: LvNull)
    if ty == "num"
        return LogKv(key: k, val: LvNum(raw: v))
    return LogKv(key: k, val: LvStr(t: v))
fn gkv(xs: List[text]) -> List[LogKv]
    var o: List[LogKv] = []
    var i: usize = 0
    while i + 2 < xs.len() bound 4096
        o.push(gk(copy(xs[i]), copy(xs[i + 1]), copy(xs[i + 2])))
        i += 3
    return o
fn gti(k: text, v: i64) -> LogKv = LogKv(key: k, val: LvInt(v: v))
fn genc(kvs: List[LogKv]) -> text = lf_encode(&kvs) catch "ERR"
fn gskip(kvs: List[LogKv]) -> text = lf_encode_skip(&kvs)
fn gflat(l: &LfLine) -> List[text]
    var o: List[text] = []
    for kv in l.kvs
        o.push(copy(kv.key))
        o.push(lv_text(&kv.val))
    if not l.ok
        o.push("!" + l.at.tostr() + ":" + l.why)
    return o
fn gdec(line: text) -> List[text]
    let l = lf_scan(&line)
    return gflat(&l)
# The first n lines of lf_lines, flattened with "|" after each line.
fn glines(s: text, n: i64) -> List[text]
    let ls = lf_lines(&s)
    var o: List[text] = []
    var i: i64 = 0
    while i < n and i < (ls.len() as i64) bound 4096
        for x in gflat(&ls[i as usize])
            o.push(x)
        o.push("|")
        i += 1
    if (ls.len() as i64) < n
        o.push("short")
    return o
fn gnlines(s: text) -> i64 = lf_lines(&s).len() as i64
fn gjson(kvs: List[LogKv], a: bool) -> text = log_json_obj(&kvs, a)
fn grec(t: i64, has_t: bool, sev: i64, msg: text, kvs: List[LogKv]) -> LogRec = LogRec(time: if has_t then some(t) else none, sev: sev, msg: msg, kvs: kvs)
fn gtag(x: &LogVal) -> text
    match x
        LvStr(t: _t) -> return "s"
        LvInt(v: _v) -> return "i"
        LvBool(v: _b) -> return "b"
        LvNull -> return "n"
        LvNum(raw: _r) -> return "num"
fn gpairs(r: LogRec, f: LogFmt) -> List[text]
    var o: List[text] = []
    for kv in log_kvs(&r, &f)
        o.push(copy(kv.key))
        o.push(gtag(&kv.val))
        o.push(if gtag(&kv.val) == "n" then "" else lv_text(&kv.val))
    return o
fn glf(r: LogRec, f: LogFmt) -> text = log_logfmt(&r, &f)
fn gjs(r: LogRec, f: LogFmt) -> text = log_json(&r, &f)
fn gtime(t: i64, frac: i64, off: i64) -> text
    let f = LogFmt(frac: frac, offset_min: off, ascii: false, max_value: 100)
    return log_time_text(t, &f)
fn gparse(s: text) -> text
    let kvs = [LogKv(key: "time", val: LvStr(t: s))]
    let r = log_from_kvs(&kvs)
    return if r.time.issome() then r.time.or(0).tostr() else "none"
fn gsample(tick: i64, first: i64, there: i64, ts: List[i64], sevs: List[i64], ms: List[text]) -> text
    var s = log_sampler(tick, first, there)
    var o: List[text] = []
    var i: usize = 0
    while i < ts.len() bound 4096
        let r = LogRec(time: some(ts[i]), sev: sevs[i], msg: copy(ms[i]), kvs: [])
        o.push(if log_sample(&mut s, &r) then "1" else "0")
        i += 1
    return "".join(o)
'''

def kvlit(pairs):
    out = []
    for k, ty, v in pairs:
        if ty == 'i':
            out.append('gti(%s, %s)' % (L(k), ilit(int(v))))
        else:
            out.append('gk(%s, %s, %s)' % (L(k), L(ty), L(v)))
    return '[' + ', '.join(out) + ']'

def tlist(xs):
    return '[' + ', '.join(L(x) for x in xs) + ']'

def flat_go(pairs, err, extra):
    o = []
    for k, v in pairs:
        o += [k, v]
    if err is not None:
        o.append('!%d:%s' % (err['pos'] - 1, err['msg']))
    return o + extra

checks = []
groups = []
for e, m, k in zip(enc_in, ref['encode'], ref['encode_skip']):
    checks.append('check genc(%s) == %s' % (kvlit(e), L(m if m is not None else 'ERR')))
    checks.append('check gskip(%s) == %s' % (kvlit(e), L(k)))
for l, d in zip(dec_in, ref['decode']):
    checks.append('check gdec(%s) == %s' % (L(l), tlist(flat_go(d['pairs'], d['err'], []))))
for t, s in zip(streams, ref['streams']):
    recs_go = s['records']
    exp = []
    for i, rp in enumerate(recs_go):
        last = i == len(recs_go) - 1
        exp += flat_go(rp, s['err'] if last else None, ['|'])
    checks.append('check glines(%s, %d) == %s' % (L(t), len(recs_go), tlist(exp)))
    if s['err'] is None:
        checks.append('check gnlines(%s) == %d' % (L(t), len(recs_go)))
groups.append(checks)
checks = []
for s, keep in zip(samples, ref['sample']):
    ts = '[' + ', '.join(str(e[0]) for e in s['entries']) + ']'
    sv = '[' + ', '.join(str(9 + 4 * e[1]) for e in s['entries']) + ']'
    ms = tlist([e[2] for e in s['entries']])
    checks.append('check gsample(%d, %d, %d, %s, %s, %s) == %s' % (s['tick'], s['first'], s['thereafter'], ts, sv, ms, L(''.join('1' if x else '0' for x in keep))))
for t, frac, off, txt in time_fmt:
    checks.append('check gtime(%s, %s, %s) == %s' % (ilit(t), ilit(frac), ilit(off), L(txt)))
for txt, v in time_parse:
    checks.append('check gparse(%s) == %s' % (L(txt), L('none' if v is None else str(v))))
groups.append(checks)
checks = []
for j in json_in:
    checks.append('check gjson(%s, %s) == %s' % (kvlit(j['pairs']), 'true' if j['ascii'] else 'false', L(j['json'])))
for r in recs:
    rec, f = r['rec'], r['fmt']
    rl = 'grec(%s, %s, %s, %s, %s)' % (ilit(rec['time'] if rec['time'] is not None else 0), 'true' if rec['time'] is not None else 'false', ilit(rec['sev']), L(rec['msg']), kvlit(rec['kvs']))
    fl = 'LogFmt(frac: %s, offset_min: %s, ascii: %s, max_value: %s)' % (ilit(f['frac']), ilit(f['off']), 'true' if f['ascii'] else 'false', ilit(f['max']))
    flat = []
    for k, ty, v in r['pairs']:
        flat += [k, ty, v]
    checks.append('check gpairs(%s, %s) == %s' % (rl, fl, tlist(flat)))
    checks.append('check glf(%s, %s) == %s' % (rl, fl, L(r['logfmt'])))
    checks.append('check gjs(%s, %s) == %s' % (rl, fl, L(r['json'])))

groups.append(checks)
# one file per part: logfmt, sampling, JSON and records
os.makedirs(os.path.join(ROOT, 'tests', 'obs'), exist_ok=True)
for f in os.listdir(os.path.join(ROOT, 'tests', 'obs')):
    if re.fullmatch(r'log_checks_\d+\.rce', f):
        os.remove(os.path.join(ROOT, 'tests', 'obs', f))
for n, g in enumerate(groups):
    with open(os.path.join(ROOT, 'tests', 'obs', 'log_checks_%d.rce' % (n + 2)), 'w') as fh:
        fh.write(HEAD + '\n' + '\n'.join(g) + '\n')
checks = [c for g in groups for c in g]
n = len(groups)
print('%d checks in %d files: encode %d, decode %d, streams %d, sample %d, time %d, json %d, records %d' % (
    len(checks), n, 2 * len(enc_in), len(dec_in), len(streams), len(samples), len(time_fmt) + len(time_parse), len(json_in), 3 * len(recs)))
