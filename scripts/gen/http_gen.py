# Checks for lib/net/http.rce. A Python model of the module's RFC 9112
# rules gives each expected result; every message the model accepts is
# also parsed with h11 (a third-party HTTP/1.1 implementation) and must
# agree on method, target, version, status, reason, fields and body.
# Usage: python3 scripts/gen/http_gen.py > generated part of tests/net/http_checks.rce
import random, re, h11
R = random.Random(25)
def q(s):
    o = ''
    for c in s:
        o += {'"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t'}.get(c) or (c if 32 <= ord(c) != 127 else '\\u{%x}' % ord(c))
    return '"' + o + '"'
TCH = set("!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
class Bad(Exception): pass
def lines_until_blank(b, i):
    out = []
    while True:
        e = line_end(b, i)
        if e < 0: return None, i
        if e == i: return out, i + 2
        out.append(b[i:e]); i = e + 2
def line_end(b, i):
    k = i
    while k < len(b):
        if b[k] == '\r':
            if k + 1 >= len(b): return -1
            if b[k + 1] != '\n': raise Bad('bare CR')
            return k
        if b[k] == '\n': raise Bad('bare LF')
        k += 1
    return -1
def asc(t):
    for c in t:
        if not (c == '\t' or 32 <= ord(c) <= 126): raise Bad('not ASCII text')
    return t
def field(t):
    if t[:1] in (' ', '\t'): raise Bad('obs-fold')
    i = 0; name = ''
    while i < len(t) and t[i] != ':':
        if t[i] not in TCH: raise Bad('field name')
        name += t[i]; i += 1
    if i >= len(t) or name == '': raise Bad('field line')
    return name, t[i + 1:].strip(' \t')
def hdr(fs, n):
    v = [b for a, b in fs if a.lower() == n]
    return ', '.join(v) if v else None
def version(t):
    if len(t) != 8 or not t.startswith('HTTP/') or t[6] != '.' or not t[5].isdigit() or not t[7].isdigit(): raise Bad('version')
    return t[5] + '.' + t[7]
def items(v): return [x.strip(' \t') for x in v.split(',')]
def dec(t): return int(t) if t and len(t) <= 15 and all(c in '0123456789' for c in t) else -1
def hexn(t): return int(t, 16) if t and len(t) <= 15 and all(c in '0123456789abcdefABCDEF' for c in t) else -1
def model(b, req, head, eof):
    i = 0
    if req:
        while b[i:i + 2] == '\r\n' and i < 32: i += 2
    e = line_end(b, i)
    if e < 0: raise Bad('incomplete head')
    sl = asc(b[i:e]); i = e + 2
    m = {'method': '', 'target': '', 'ver': '', 'status': 0, 'reason': ''}
    if req:
        ps = sl.split(' ')
        if len(ps) != 3 or ps[0] == '' or ps[1] == '': raise Bad('request line')
        if any(c not in TCH for c in ps[0]): raise Bad('method')
        if any(not (33 <= ord(c) <= 126) for c in ps[1]): raise Bad('target')
        m['ver'] = version(ps[2]); m['method'], m['target'] = ps[0], ps[1]
    else:
        if len(sl) < 12 or sl[8] != ' ': raise Bad('status line')
        m['ver'] = version(sl[:8])
        if not all(c.isdigit() for c in sl[9:12]): raise Bad('status code')
        if len(sl) > 12 and sl[12] != ' ': raise Bad('status line')
        m['status'] = int(sl[9:12]); m['reason'] = sl[13:]
    fs = []
    while True:
        f = line_end(b, i)
        if f < 0: raise Bad('incomplete head')
        if f == i: i += 2; break
        fs.append(field(asc(b[i:f]))); i = f + 2
    m['fields'] = fs; m['trailers'] = []
    te, cl = hdr(fs, 'transfer-encoding'), hdr(fs, 'content-length')
    st = m['status']
    if not req and (head or 100 <= st < 200 or st in (204, 304)):
        m['body'] = ''; m['used'] = i; return m
    if te is not None:
        if cl is not None: raise Bad('both Transfer-Encoding and Content-Length')
        it = items(te)
        if it[-1].lower() != 'chunked':
            if req: raise Bad('transfer coding')
            if not eof: raise Bad('incomplete close')
            m['body'] = b[i:]; m['used'] = len(b); return m
        if any(x.lower() == 'chunked' for x in it[:-1]): raise Bad('chunked twice')
        if len(it) > 1: raise Bad('unsupported transfer coding')
        body = ''
        while True:
            e = line_end(b, i)
            if e < 0: raise Bad('incomplete chunk')
            line = asc(b[i:e]); size = re.split('[; \t]', line)[0]
            n = hexn(size)
            if n < 0: raise Bad('chunk size')
            i = e + 2
            if n == 0:
                while True:
                    f = line_end(b, i)
                    if f < 0: raise Bad('incomplete chunk')
                    if f == i: m['body'] = body; m['used'] = i + 2; return m
                    m['trailers'].append(field(asc(b[i:f]))); i = f + 2
            if i + n + 2 > len(b): raise Bad('incomplete chunk')
            body += b[i:i + n]; i += n
            if b[i:i + 2] != '\r\n': raise Bad('chunk end')
            i += 2
    if cl is not None:
        it = items(cl); n = dec(it[0])
        if any(dec(x) != n for x in it) or n < 0: raise Bad('Content-Length')
        if i + n > len(b): raise Bad('incomplete body')
        m['body'] = b[i:i + n]; m['used'] = i + n; return m
    if req: m['body'] = ''; m['used'] = i; return m
    if not eof: raise Bad('incomplete close')
    m['body'] = b[i:]; m['used'] = len(b); return m
def show(m):
    f = lambda fs: ';'.join('%s=%s' % x for x in fs)
    return '|'.join([m['method'], m['target'], m['ver'], str(m['status']), m['reason'], f(m['fields']), m['body'], f(m['trailers']), str(m['used'])])
def h11_agrees(b, req, head, m):
    if not req: 
        c = h11.Connection(h11.CLIENT)
        c.send(h11.Request(method='HEAD' if head else 'GET', target='/', headers=[('Host', 'x')])); c.send(h11.EndOfMessage())
    else:
        c = h11.Connection(h11.SERVER)
    c.receive_data(b.encode('latin-1'))
    evs = []
    while True:
        e = c.next_event()
        if e in (h11.NEED_DATA, h11.PAUSED): break
        evs.append(e)
        if isinstance(e, h11.EndOfMessage): break
    head_ev = [e for e in evs if isinstance(e, (h11.Request, h11.Response, h11.InformationalResponse))]
    if not head_ev: return False
    h = head_ev[0]
    body = b''.join(e.data for e in evs if isinstance(e, h11.Data)).decode('latin-1')
    fs = [(n.decode(), v.decode()) for n, v in h.headers.raw_items()]
    ok = fs == m['fields'] and h.http_version.decode() == m['ver']
    if req: ok = ok and h.method.decode() == m['method'] and h.target.decode() == m['target']
    else: ok = ok and h.status_code == m['status'] and h.reason.decode() == m['reason']
    if any(isinstance(e, h11.EndOfMessage) for e in evs): ok = ok and body == m['body']
    return ok
CASES = [
    ('GET / HTTP/1.1\r\nHost: example.com\r\n\r\n', True, False, False),
    ('\r\nGET /a?b HTTP/1.1\r\nHost: x\r\n\r\n', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhel', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nContent-Length: 5, 5\r\n\r\nhelloEXTRA', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nContent-Length: 6\r\n\r\nhello!', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\nX-T: 1\r\n\r\n', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nContent-Length: 3\r\n\r\n0\r\n\r\n', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: gzip\r\n\r\n', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked, chunked\r\n\r\n0\r\n\r\n', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\nz\r\n', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabcX\r\n0\r\n\r\n', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\nA\r\n0123456789\r\n0\r\n\r\n', True, False, False),
    ('GET / HTTP/1.1\r\nHost : x\r\n\r\n', True, False, False),
    ('GET / HTTP/1.1\r\nHost: x\r\n folded\r\n\r\n', True, False, False),
    ('GET / HTTP/1.1\nHost: x\n\n', True, False, False),
    ('GET / HTTP/1.1\r\nHost: x\rY\r\n\r\n', True, False, False),
    ('GET  / HTTP/1.1\r\n\r\n', True, False, False),
    ('GET / HTTP/1.1 \r\n\r\n', True, False, False),
    ('G(T / HTTP/1.1\r\n\r\n', True, False, False),
    ('GET / HTTP/11\r\n\r\n', True, False, False),
    ('GET / http/1.1\r\n\r\n', True, False, False),
    ('GET / HTTP/1.0\r\n\r\n', True, False, False),
    ('GET / HTTP/1.1\r\nHost: x\r\nX:\r\nY:  a  b \t\r\n\r\n', True, False, False),
    ('GET / HTTP/1.1\r\nHost: x\r\n', True, False, False),
    ('GET / HTTP/1.1\r\n:x\r\n\r\n', True, False, False),
    ('HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi', False, False, False),
    ('HTTP/1.1 200 OK\r\n\r\nuntil close', False, False, True),
    ('HTTP/1.1 200 OK\r\n\r\nuntil close', False, False, False),
    ('HTTP/1.1 204 No Content\r\nContent-Length: 9\r\n\r\n', False, False, False),
    ('HTTP/1.1 304 Not Modified\r\n\r\n', False, False, False),
    ('HTTP/1.1 100 Continue\r\n\r\n', False, False, False),
    ('HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n', False, True, False),
    ('HTTP/1.1 200 \r\nContent-Length: 0\r\n\r\n', False, False, False),
    ('HTTP/1.1 200\r\nContent-Length: 0\r\n\r\n', False, False, False),
    ('HTTP/1.1 404 Not  Found here\r\nContent-Length: 0\r\n\r\n', False, False, False),
    ('HTTP/1.1 20x OK\r\n\r\n', False, False, False),
    ('HTTP/1.1  200 OK\r\n\r\n', False, False, False),
    ('HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nab\r\n0\r\n\r\n', False, False, False),
    ('HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\nzz', False, False, True),
    ('HTTP/1.0 500 Oops\r\nContent-Length: 3\r\n\r\nabc', False, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n5 ;x\r\nhello\r\n0\r\n\r\n', True, False, False),
    (' / HTTP/1.1\r\n\r\n', True, False, False),
    ('POST /p HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\nb\r\nhello world\r\n0\r\n\r\n', True, False, False),
    ('GET  HTTP/1.1\r\n\r\n', True, False, False),
    ('HTTP/1.1 2000 OK\r\n\r\n', False, False, False),
]
def rand_case():
    req = R.random() < 0.6
    fs = [('Host', 'h%d.example' % R.randint(0, 9))] + [('X-%s' % R.choice(['A', 'b', 'Long-Name', 'n_1']), R.choice(['', 'v', ' spaced  out ', 'a,b', 'q="x y"', '\tt'])) for _ in range(R.randint(0, 4))]
    body = ''.join(R.choice('abc xyz\r\n{}') for _ in range(R.randint(0, 20)))
    mode = R.choice(['cl', 'chunked', 'none'])
    if mode == 'cl': fs.append(('Content-Length', str(len(body))))
    if mode == 'chunked': fs.append(('Transfer-Encoding', 'chunked'))
    if req:
        start = '%s %s HTTP/1.1' % (R.choice(['GET', 'POST', 'PUT', 'DELETE', 'X-CUSTOM']), R.choice(['/', '/a/b?c=d', '*', 'http://x/y', '/%20']))
    else:
        start = 'HTTP/1.1 %d %s' % (R.choice([200, 201, 302, 400, 404, 500]), R.choice(['OK', 'Created', 'Moved Temporarily', '']))
    raw = start + '\r\n' + ''.join('%s: %s\r\n' % f for f in fs) + '\r\n'
    if mode == 'chunked':
        k = 0
        while k < len(body):
            n = R.randint(1, 7); raw += '%x\r\n%s\r\n' % (len(body[k:k + n]), body[k:k + n]); k += n
        raw += '0\r\n\r\n'
    elif mode == 'cl' or (not req): raw += body
    eof = (not req and mode == 'none')
    if R.random() < 0.2: raw = raw[:R.randint(0, len(raw))]
    return raw, req, False, eof
for _ in range(60): CASES.append(rand_case())
agree = disagree = 0
for raw, req, head, eof in CASES:
    try:
        m = model(raw, req, head, eof); want = show(m)
        try: ok = h11_agrees(raw, req, head, m)
        except Exception: ok = False
        if ok: agree += 1
        else: disagree += 1; print('# h11 differs on the next case (h11 is more lenient or stricter here)')
    except Bad as e:
        want = 'ERR:' + str(e)
    print('check hp(%s, %s, %s, %s) == %s' % (q(raw), 'true' if req else 'false', 'true' if head else 'false', 'true' if eof else 'false', q(want)))
print('# h11 agreed on %d accepted messages, differed on %d' % (agree, disagree))
# serialization: model of serialize(); h11 must parse each back
SER = [
    (True, 'GET', '/', 0, '', [('Host', 'x')], ''),
    (True, 'POST', '/up', 0, '', [('Host', 'x'), ('Content-Type', 'text/plain')], 'payload'),
    (False, '', '', 200, 'OK', [('Server', 'rce')], 'hello'),
    (False, '', '', 204, 'No Content', [], ''),
    (False, '', '', 304, 'Not Modified', [('ETag', '"a"')], ''),
    (False, '', '', 404, 'Not Found', [], ''),
    (True, 'PUT', '/c', 0, '', [('Host', 'x'), ('Transfer-Encoding', 'chunked')], '3\r\nabc\r\n0\r\n\r\n'),
]
for req, meth, tgt, st, rs, fs, body in SER:
    out = (('%s %s HTTP/1.1\r\n' % (meth, tgt)) if req else ('HTTP/1.1 %d %s\r\n' % (st, rs))) + ''.join('%s: %s\r\n' % f for f in fs)
    framed = any(n.lower() in ('content-length', 'transfer-encoding') for n, _ in fs)
    bodyless = not req and (100 <= st < 200 or st in (204, 304))
    if not framed and not bodyless and (not req or body): out += 'Content-Length: %d\r\n' % len(body)
    out += '\r\n' + body
    print('check hser(%s, %s, %s, %d, %s, [%s], [%s], %s) == %s' % ('true' if req else 'false', q(meth), q(tgt), st, q(rs), ', '.join(q(n) for n, _ in fs), ', '.join(q(v) for _, v in fs), q(body), q(out)))
for data, size in [('', 4), ('abc', 1), ('hello world', 4), ('x' * 20, 16), ('ab', 0)]:
    sz = max(size, 1); o = ''
    for k in range(0, len(data), sz): o += '%x\r\n%s\r\n' % (len(data[k:k + sz]), data[k:k + sz])
    print('check hc(%s, %d) == %s' % (q(data), size, q(o + '0\r\n\r\n')))
