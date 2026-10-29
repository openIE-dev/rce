# Checks for lib/net/dns.rce. Messages are built and written by dnspython
# 2.9.0 (a third-party implementation, with its name compression); the
# module must decode each to dnspython's own reading (names in
# presentation form, record data uncompressed as dnspython writes it with
# compression off; dnspython compresses case-insensitively, so its reading
# of the wire, not the message it started from, is the reference). Encoding is checked byte for byte against an
# uncompressed writer here, whose output dnspython must read back as the
# same message. Malformed wires are an error. Usage:
#   python3 scripts/gen/dns_gen.py > generated part of tests/net/dns_checks.rce
import random, struct
import dns.message, dns.name, dns.rrset, dns.rdatatype, dns.rdataclass, dns.flags, dns.rdata
R = random.Random(53)
def q(s):
    o = ''
    for c in s:
        o += {'"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t'}.get(c) or (c if 32 <= ord(c) != 127 else '\\u{%x}' % ord(c))
    return '"' + o + '"'
def hexs(b): return q(b.hex())
def rname():
    labs = []
    for _ in range(R.randint(0, 4)):
        labs.append(''.join(R.choice(['a', 'b', 'Z', '0', '-', '_', 'x', 'mail', 'é', '.', '\\', ' ', '(', '"', ';', '@', '$']) for _ in range(R.randint(1, 6))))
    return dns.name.Name([l.encode('utf-8') for l in labs] + [b''])
def pres(n): return n.to_text()  # dnspython escapes like RFC 1035 5.1
POOL = [dns.name.from_text(t) for t in ['example.com.', 'www.example.com.', 'mail.example.com.', 'Example.COM.', 'a.b.c.example.com.']]
def nm(): return R.choice(POOL) if R.random() < 0.6 else rname()
def rd(t):
    if t == 'A': return '%d.%d.%d.%d' % tuple(R.randint(0, 255) for _ in range(4))
    if t == 'AAAA': return '2001:db8::%x' % R.randint(0, 65535)
    if t in ('NS', 'CNAME', 'PTR'): return nm().to_text()
    if t == 'MX': return '%d %s' % (R.randint(0, 65535), nm().to_text())
    if t == 'SOA': return '%s %s %d %d %d %d %d' % (nm().to_text(), nm().to_text(), R.randint(0, 2**32 - 1), 3600, 600, 86400, 60)
    if t == 'TXT': return ' '.join('"%s"' % ''.join(R.choice('ab c=') for _ in range(R.randint(0, 9))) for _ in range(R.randint(1, 3)))
    if t == 'SRV': return '%d %d %d %s' % (R.randint(0, 9), R.randint(0, 9), R.randint(1, 65535), nm().to_text())
TYPES = ['A', 'AAAA', 'NS', 'CNAME', 'PTR', 'MX', 'SOA', 'TXT', 'SRV']
def build():
    m = dns.message.Message(id=R.randint(0, 65535))
    m.flags = R.randint(0, 65535) & ~0x7800 & ~0x0070  # opcode QUERY, z bits clear
    for sect in (m.question,):
        for _ in range(R.randint(0, 1)):
            m.question.append(dns.rrset.RRset(nm(), dns.rdataclass.IN, dns.rdatatype.from_text(R.choice(TYPES))))
    for sect in (m.answer, m.authority, m.additional):
        seen = set()
        for _ in range(R.randint(0, 3)):
            t = R.choice(TYPES); n = nm()
            if (n, t) in seen: continue
            seen.add((n, t))
            sect.append(dns.rrset.from_text(n, R.randint(0, 2**31 - 1), 'IN', t, rd(t)))
    return m
def records(sect):
    out = []
    for rrset in sect:
        for r in rrset:
            out.append((pres(rrset.name), r.rdtype, rrset.rdclass, rrset.ttl, r.to_wire(None, None)))
    return out
def plain(m):
    o = struct.pack('>HHHHHH', m.id, m.flags, len(m.question), sum(len(x) for x in m.answer), sum(len(x) for x in m.authority), sum(len(x) for x in m.additional))
    for qq in m.question: o += qq.name.to_wire() + struct.pack('>HH', qq.rdtype, qq.rdclass)
    for sect in (m.answer, m.authority, m.additional):
        for rrset in sect:
            for r in rrset:
                d = r.to_wire(None, None)
                o += rrset.name.to_wire() + struct.pack('>HHIH', r.rdtype, rrset.rdclass, rrset.ttl, len(d)) + d
    return o
def rlist(rs): return '[' + ', '.join('Record(name: %s, rtype: %d, rclass: %d, ttl: %d, rdata: hexb(%s))' % (q(n), t, c, ttl, hexs(d)) for n, t, c, ttl, d in rs) + ']'
def qlist(m): return '[' + ', '.join('Question(name: %s, qtype: %d, qclass: %d)' % (q(pres(x.name)), x.rdtype, x.rdclass) for x in m.question) + ']'
print('# dnspython messages (compressed): decode, then encode uncompressed')
for _ in range(40):
    wire = build().to_wire()
    # dnspython compresses case-insensitively, so the wire may change the
    # case of a name; the expected message is dnspython's reading of it
    m = dns.message.from_wire(wire)
    p = plain(m)
    back = dns.message.from_wire(p)
    assert back == m and back.id == m.id and back.flags == m.flags
    lit = 'Dns(id: %d, flags: %d, questions: %s, answers: %s, authority: %s, additional: %s)' % (m.id, m.flags, qlist(m), rlist(records(m.answer)), rlist(records(m.authority)), rlist(records(m.additional)))
    print('check dec(%s) == %s' % (hexs(wire), lit))
    print('check enc(%s) == %s' % (lit, hexs(p)))
print('# names: presentation form both ways')
for t in ['.', 'example.com.', 'Example.COM.', 'a\\.b.c.', 'a\\\\b.', 'x\\032y.', '\\000.', 'caf\\195\\169.', '_srv._tcp.x.', 'a' * 63 + '.']:
    n = dns.name.from_text(t)
    print('check nw(%s) == %s' % (q(t), hexs(n.to_wire())))
    print('check rn(%s) == %s' % (hexs(n.to_wire()), q(n.to_text())))
for t in ['a' * 64 + '.', 'a..b.', '..', 'x\\256.', 'x\\25', '.'.join(['a' * 63] * 4) + '.', 'é.']:
    print('check not nw_ok(%s)' % q(t))
print('# malformed wires')
hdr1 = struct.pack('>HHHHHH', 1, 0, 1, 0, 0, 0)
bad = [b'', b'\x00' * 11, hdr1, hdr1 + b'\x03abc', hdr1 + b'\xc0\x0c\x00\x01\x00\x01', hdr1 + b'\x01a\xc0\x0c\x00\x01\x00\x01',
       hdr1 + b'\xc0\x20\x00\x01\x00\x01', hdr1 + b'\x40a\x00\x00\x01\x00\x01', hdr1 + b'\x80a\x00\x00\x01\x00\x01',
       hdr1 + b'\x00\x00\x01\x00\x01\xff', struct.pack('>HHHHHH', 1, 0, 0, 1, 0, 0) + b'\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x03abc',
       struct.pack('>HHHHHH', 1, 0, 0, 1, 0, 0) + b'\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x05abcd',
       hdr1 + (b'\x3f' + b'a' * 63) * 4 + b'\x00\x00\x01\x00\x01',
       struct.pack('>HHHHHH', 1, 0, 0, 1, 0, 0) + b'\x00\x00\x06\x00\x01\x00\x00\x00\x00\x00\x17\x00\x00' + b'\x00' * 21]
for b in bad:
    try: dns.message.from_wire(b); ok = True
    except Exception: ok = False
    assert not ok, b
    print('check not dec_ok(%s)' % hexs(b))
print('# names at an offset: forward pointer, long pointer chains, label types, length 256')
def rnat(b, off, want): print('check rn_at(%s, %d) == %s' % (hexs(b), off, q(want)))
rnat(b'\xc0\x02\x01a\x00', 0, 'ERR')
chain = b'\x01a\x00' + b''.join(struct.pack('>H', 0xC000 | (0 if i == 0 else 3 + 2 * (i - 1))) for i in range(130))
rnat(chain, 3 + 2 * 127, 'a.')
rnat(chain, 3 + 2 * 129, 'ERR')
rnat(b'\x40' + b'a' * 64 + b'\x00', 0, 'ERR')
rnat(b'\x80' + b'a' * 64 + b'\x00', 0, 'ERR')
rnat((b'\x3f' + b'a' * 63) * 3 + b'\x3e' + b'a' * 62 + b'\x00', 0, 'ERR')
rnat((b'\x3f' + b'a' * 63) * 3 + b'\x3d' + b'a' * 61 + b'\x00', 0, ('a' * 63 + '.') * 3 + 'a' * 61 + '.')
# a legal backward pointer chain
w = struct.pack('>HHHHHH', 7, 0x8180, 1, 1, 0, 0) + b'\x03www\x07example\x03com\x00\x00\x01\x00\x01' + b'\xc0\x0c\x00\x01\x00\x01\x00\x00\x0e\x10\x00\x04\x5d\xb8\xd8\x22'
m = dns.message.from_wire(w)
print('check dec(%s) == Dns(id: 7, flags: 33152, questions: [Question(name: "www.example.com.", qtype: 1, qclass: 1)], answers: [Record(name: "www.example.com.", rtype: 1, rclass: 1, ttl: 3600, rdata: [93, 184, 216, 34])], authority: [], additional: [])' % hexs(w))
print('check atx([93, 184, 216, 34]) == "93.184.216.34"')
print('check rdata_ok()')
