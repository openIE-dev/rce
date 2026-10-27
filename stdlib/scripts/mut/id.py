# Mutation test for lib/id/id.rce: apply each mutant to a copy of the
# module, append each tests/id/id_checks*.rce, and require at least one
# `rce test` to fail. Usage: RCE=path/to/rce python3 scripts/mut/id.py
# Equivalent mutants left out: lowercasing only part of A..Z in the
# urn:uuid: prefix test (only U, R, N, I and D matter), reading '-' as a
# sign in uuid_parse_lax (every '-' is removed before), stripping trailing
# braces down to one character (a text of braces never leaves 32
# characters), and '+' for '|' between disjoint Snowflake fields.
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
src = open(os.path.join(root, 'lib/id/id.rce')).read()
chks = [open(f).read() for f in sorted(glob.glob(os.path.join(root, 'tests/id/id_checks*.rce')))]
M = [
    # bytes and text
    ('        if x < 0 or x > 255\n            fail BadId(at: i, why: "a byte must be 0..255")', '        if x < 0 or x > 256\n            fail BadId(at: i, why: "a byte must be 0..255")'),
    ('    if c >= 65 and c <= 70\n        return c - 55', '    if c >= 65 and c <= 71\n        return c - 55'),
    ('        if dashes and (i == 4 or i == 6 or i == 8 or i == 10)', '        if dashes and (i == 4 or i == 6 or i == 8 or i == 11)'),
    ('        if c >= 65 and c <= 90\n            c += 32', '        if c >= 65 and c <= 90\n            c += 31'),
    ('    let hyph = i + 8 < n and s[i + 8] == 45', '    let hyph = i + 9 < n and s[i + 8] == 45'),
    ('        if e >= n or s[e] != 125', '        if e >= n or s[e] == 125'),
    ('    if e != n\n        fail BadId(at: e as i64, why: "unexpected text after the UUID")', '    if e > n\n        fail BadId(at: e as i64, why: "unexpected text after the UUID")'),
    # lax parse
    ('        if hit\n            i += m', '        if hit\n            i += 1'),
    ('fn id_pyspace(c: i64) -> bool = c == 32 or (c >= 9 and c <= 13)', 'fn id_pyspace(c: i64) -> bool = c == 32 or (c >= 9 and c <= 12)'),
    ('    if p < 32 and h[p] == 43\n', '    if false\n'),
    ('        if p < 32 and h[p] == 95\n            p += 1', '        if false\n            p += 1'),
    ('            if prev == 95\n', '            if false\n'),
    ('    if prev == 95\n        fail', '    if false\n        fail'),
    ('    while lo < hi and (b2[lo] == 123 or b2[lo] == 125)', '    while lo < hi and b2[lo] == 123'),
    # fields and versions
    ('    if v < 192\n        return UUID_VAR_RFC', '    if v < 191\n        return UUID_VAR_RFC'),
    ('    if b[8] < 128 or b[8] >= 192\n        return -1', '    if b[8] < 128\n        return -1'),
    ('        return (id_int(&b, 0, 4) << 28) | (id_int(&b, 4, 6) << 12) | (id_int(&b, 6, 8) & 4095)\n    return', '        return (id_int(&b, 0, 4) << 28) | (id_int(&b, 4, 6) << 12) | (id_int(&b, 6, 8) & 2047)\n    return'),
    ('    return ((b[8] & 63) << 8) | b[9]', '    return ((b[8] & 127) << 8) | b[9]'),
    ('fn id_fdiv(a: i64, b: i64) -> i64 = if a < 0 and a % b != 0 then a / b - 1 else a / b', 'fn id_fdiv(a: i64, b: i64) -> i64 = a / b'),
    ('    o[8] = 128 | (o[8] & 63)', '    o[8] = 128 | (o[8] & 127)'),
    ('    if version != 3 and version != 5 and version != 8', '    if version != 3 and version != 5'),
    ('    o = id_cat(o, [16 | ((ticks >> 56) & 15), (ticks >> 48) & 255])', '    o = id_cat(o, [16 | ((ticks >> 56) & 7), (ticks >> 48) & 255])'),
    ('    o = id_cat(o, [96 | ((ticks >> 8) & 15), ticks & 255])', '    o = id_cat(o, [96 | ((ticks >> 8) & 15), (ticks >> 1) & 255])'),
    ('    o = id_cat(o, [96 | ((t >> 8) & 15), t & 255])', '    o = id_cat(o, [96 | ((t >> 9) & 15), t & 255])'),
    ('    o = id_cat(o, [16 | ((t >> 56) & 15), (t >> 48) & 255])', '    o = id_cat(o, [16 | ((t >> 52) & 15), (t >> 48) & 255])'),
    ('fn id_tail(clock_seq: i64, node_id: i64) -> List[i64] = id_cat([128 | (clock_seq >> 8), clock_seq & 255], id_be(node_id, 6))', 'fn id_tail(clock_seq: i64, node_id: i64) -> List[i64] = id_cat([128 | (clock_seq >> 8), clock_seq & 255], id_be(node_id, 5))'),
    # version 7 and its state
    ('        seq = ((r[6] & 127) << 24) | (r[7] << 16) | (r[8] << 8) | r[9]', '        seq = ((r[6] & 255) << 24) | (r[7] << 16) | (r[8] << 8) | r[9]'),
    ('    if unix_ms > g.ms\n        seq', '    if unix_ms >= g.ms\n        seq'),
    ('        if seq == 0\n            ms += 1', '        if seq == 0\n            ms += 0'),
    ('    o.push(((seq << 2) & 255) | (r[10] & 3))', '    o.push(((seq << 2) & 255) | (r[10] & 1))'),
    # ULID
    ('    if c == 77 or c == 78\n        return c - 57', '    if c == 77 or c == 78 or c == 79\n        return c - 57'),
    ('    let c = if c0 >= 97 and c0 <= 122 then c0 - 32 else c0', '    let c = c0'),
    ('            if v > 7\n                fail BadId(at: 0, why: "the first symbol must be 0..7")', '            if v > 8\n                fail BadId(at: 0, why: "the first symbol must be 0..7")'),
    ('    var nb: i64 = 2\n', '    var nb: i64 = 1\n'),
    ('        if c == 105 or c == 73 or c == 108 or c == 76', '        if c == 105 or c == 73 or c == 108'),
    ('    if g.ms >= 0 and unix_ms <= g.ms', '    if g.ms >= 0 and unix_ms < g.ms'),
    ('        while i >= 6 bound 10', '        while i >= 7 bound 10'),
    # TypeID
    ('    if p.len() > 63\n', '    if p.len() > 64\n'),
    ('        if not letter and (i == 0 or i + 1 == p.len())', '        if not letter and i == 0'),
    ('    if cut == 0\n', '    if false\n'),
    ('        if id_b32val(c) < 0 or (c >= 65 and c <= 90)', '        if id_b32val(c) < 0'),
    # KSUID
    ('    if c >= 97 and c <= 122\n        return c - 61', '    if c >= 97 and c <= 122\n        return c - 62'),
    ('    var pad = 27 - (digits.len() as i64)', '    var pad = 26 - (digits.len() as i64)'),
    ('    if acc[0] != 0\n', '    if false\n'),
    ('    if unix_s < KSUID_EPOCH or unix_s - KSUID_EPOCH > 4294967295', '    if unix_s < KSUID_EPOCH'),
    ('        if k[i as usize] == 0\n            k[i as usize] = 255', '        if k[i as usize] == 0\n            k[i as usize] = 254'),
    ('    k[18] = n >> 8', '    k[17] = n >> 8'),
    # Nano ID
    ('        o.push(a[(x & 63) as usize])', '        o.push(a[(x & 31) as usize])'),
    ('    let cut = 256 - 256 % n', '    let cut = 256'),
    ('        block = (1.6 * 256.0 * (size as f64) / (cut as f64)).ceil() as i64', '        block = (1.6 * 256.0 * (size as f64) / (cut as f64)).floor() as i64'),
    ('            var k = g.block - 1', '            var k = g.block - 2'),
    ('                if y < g.cutoff\n', '                if y <= g.cutoff\n'),
    # Snowflake
    ('pub fn snow_twitter() -> SnowLayout = SnowLayout(epoch_ms: 1288834974657, worker_bits: 10, seq_bits: 12)', 'pub fn snow_twitter() -> SnowLayout = SnowLayout(epoch_ms: 1288834974657, worker_bits: 10, seq_bits: 11)'),
    ('    if now_ms < g.last_ms\n', '    if now_ms < g.last_ms - 1\n'),
    ('        if g.seq >= (1 << g.lay.seq_bits) - 1', '        if g.seq >= (1 << g.lay.seq_bits)'),
    ('    return (t << (l.worker_bits + l.seq_bits)) | (worker << l.seq_bits) | seq', '    return (t << (l.worker_bits + l.seq_bits)) | (worker << l.seq_bits) | (seq & 2047)'),
    ('worker: (sid >> s) & ((1 << w) - 1)', 'worker: (sid >> s) & ((1 << w) - 2)'),
]

def run(item):
    i, (a, b) = item
    assert src.count(a) == 1, a
    with tempfile.TemporaryDirectory() as d:
        for j, c in enumerate(chks):
            f = os.path.join(d, 'm%d_%d.rce' % (i, j))
            open(f, 'w').write(src.replace(a, b) + '\n' + c)
            if subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0:
                return True
    return False

with ThreadPoolExecutor(int(os.environ.get('JOBS', os.cpu_count() or 1))) as ex:
    res = list(ex.map(run, enumerate(M)))
for i, ((a, _b), dead) in enumerate(zip(M, res)):
    print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (sum(res), len(M)))
sys.exit(0 if all(res) else 1)
