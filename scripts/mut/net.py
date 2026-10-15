# Mutation test for lib/net/{url,http,dns}.rce: each mutant must fail its
# file's comptime checks. Usage: RCE=path/to/rce python3 scripts/mut/net.py
import os, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
L = lambda p: open(os.path.join(root, p)).read()
M = {
 'url': [
    ('        return c == 32 or c == 34 or c == 60 or c == 62 or c == 96', '        return c == 32 or c == 34 or c == 60 or c == 62'),
    ('        return q or c == 39', '        return q'),
    ('    let p = q or c == 63 or c == 94 or c == 96 or c == 123 or c == 125', '    let p = q or c == 63 or c == 96 or c == 123 or c == 125'),
    ('or c == 91 or c == 92 or c == 93 or c == 124\n\nfn pct_byte', 'or c == 91 or c == 92 or c == 93\n\nfn pct_byte'),
    ('    elif cs.len() >= 2 and cs[0] == "0"\n        r = 8', '    elif cs.len() >= 3 and cs[0] == "0"\n        r = 8'),
    ('        if ns[i] > 255\n', '        if ns[i] > 256\n'),
    ('    if parts.len() > 4\n', '    if parts.len() > 5\n'),
    ('        if j - i > blen\n', '        if j - i >= blen\n'),
    ('                    elif piece == 0\n                        fail', '                    elif piece == -2\n                        fail'),
    ('fn forbidden_domain(c: i64) -> bool = forbidden_host(c) or c <= 31 or c == 37 or c == 127', 'fn forbidden_domain(c: i64) -> bool = forbidden_host(c) or c <= 31 or c == 127'),
    ('    return ipv4_number(&last) >= 0', '    return false'),
    ('    if s == "https" or s == "wss"\n        return 443', '    if s == "https"\n        return 443'),
    ('    return l == ".." or l == ".%2e" or l == "%2e." or l == "%2e%2e"', '    return l == ".." or l == ".%2e" or l == "%2e."'),
    ('    if u.scheme == "file" and u.path.len() == 1 and is_wdl(&u.path[0], true)\n        return', '    if false\n        return'),
    ('                if rem1 == 63 or rem1 == 35', '                if rem1 == 63'),
    ('                    if h == "localhost"', '                    if h == "localhost.com"'),
    ('                u.query = u.query + pct_encode(&cc, if sp then 3 else 2)', '                u.query = u.query + pct_encode(&cc, 2)'),
    ('    elif not u.opaque and u.path.len() > 1 and u.path[0] == ""\n        o = o + "/."', '    elif false\n        o = o + "/."'),
    ('pub fn search(u: &Url) -> text = if u.has_query and u.query != "" then', 'pub fn search(u: &Url) -> text = if u.has_query then'),
 ],
 'http': [
    ('            if b[k + 1] != 10\n', '            if false\n'),
    ('        if b[k] == 10\n            fail', '        if false\n            fail'),
    ('    if cs.len() > 0 and (cs[0] == " " or cs[0] == "\\t")\n        fail BadHttp(why: "obs-fold")', '    if false\n        fail BadHttp(why: "obs-fold")'),
    ('        if cl != none\n            fail BadHttp(why: "both', '        if false\n            fail BadHttp(why: "both'),
    ('            if dec(&it) != n\n', '            if false\n'),
    ('    let no_body = not req and (head_req or (m.status >= 100 and m.status < 200) or m.status == 204 or m.status == 304)', '    let no_body = not req and (head_req or m.status == 204 or m.status == 304)'),
    ('        while i + 1 < b.len() and b[i] == 13 and b[i + 1] == 10 bound 16', '        while false bound 16'),
    ('            if c == ";" or c == " " or c == "\\t"', '            if c == ";"'),
    ('            o = if found then o + ", " + values[i] else copy(values[i])', '            o = copy(values[i])'),
    ('    while lo < hi and (cs[lo] == " " or cs[lo] == "\\t") bound', '    while lo < hi and cs[lo] == " " bound'),
    ('        if ps.len() != 3 or ps[0] == "" or ps[1] == ""', '        if ps.len() != 3'),
    ('    if cs.len() > 12 and cs[12] != " "\n', '    if false\n'),
    ('    if not framed and not bodyless and (not m.request or m.body.len() > 0)', '    if not framed and (not m.request or m.body.len() > 0)'),
    ('            if c.len() != 1 or (x < 32 and x != 9) or x == 127', '            if c.len() != 1 or x == 127'),
    ('        let k = if n - i < sz then n - i else sz', '        let k = sz'),
    ('        elif x >= 97 and x <= 102\n            d = x - 87', '        elif x >= 97 and x <= 102\n            d = x - 88'),
 ],
 'dns': [
    ('            if t >= limit\n', '            if t > limit + 1000\n'),
    ('            if hops > 128\n', '            if hops > 100000\n'),
    ('        if l & 192 != 0\n', '        if false\n'),
    ('        if wire > 255\n', '        if wire > 256\n'),
    ('        if l.len() > 63\n', '        if l.len() > 64\n'),
    ('        if x == 34 or x == 36 or x == 40 or x == 41 or x == 46 or x == 59 or x == 64 or x == 92', '        if x == 46 or x == 92'),
    ('        elif x > 32 and x < 127', '        elif x >= 32 and x < 127'),
    ('    if p != b.len() as i64\n        fail BadDns(why: "trailing bytes")', '    if false\n        fail BadDns(why: "trailing bytes")'),
    ('        let fixed_before = if t == MX then 2 else (if t == SRV then 6 else 0)', '        let fixed_before = if t == MX then 2 else 0'),
    ('    if t == NS or t == CNAME or t == PTR or t == MX or t == SOA or t == SRV', '    if t == NS or t == CNAME or t == MX or t == SOA or t == SRV'),
    ('            if q + 20 != end\n', '            if false\n'),
    ('        out.push(Record(name: copy(r.name), rtype: t, rclass: c, ttl: hi * 65536 + lo, rdata: d))', '        out.push(Record(name: copy(r.name), rtype: t, rclass: c, ttl: lo, rdata: d))'),
    ('                if v > 255\n', '                if v > 256\n'),
    ('    if t == A and n != 4\n', '    if false\n'),
    ('    f = f | 1024', '    f = f | 2048'),
 ],
}
killed = total = 0
with tempfile.TemporaryDirectory() as d:
    for name, ms in M.items():
        src = L('lib/net/%s.rce' % name); chk = L('tests/net/%s_checks.rce' % name)
        for i, (a, b) in enumerate(ms):
            assert src.count(a) == 1, (name, a)
            f = os.path.join(d, '%s%d.rce' % (name, i)); open(f, 'w').write(src.replace(a, b) + '\n' + chk)
            dead = subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0
            killed += dead; total += 1
            print('%s %s %2d %s' % ('killed' if dead else 'SURVIVED', name, i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, total))
sys.exit(0 if killed == total else 1)
