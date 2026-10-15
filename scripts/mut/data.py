# Mutation test for lib/data/*.rce: each mutant must fail the comptime
# checks (any shard of tests/data/<name>_checks*.rce). Modules with a
# "# uses:" line are tested as a package with those modules.
# Equivalent mutant removed: the E0 hi bound in utf8_text (a surrogate
# that gets past it is refused by cp_text). For feed: searching the
# source history for the target's session from i instead of i + 1 (the
# entry at i was already checked against the target history).
# Usage: RCE=path/to/rce python3 scripts/mut/data.py [module...]
import glob, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
MODS = ('events', 'feed')
SRC = {n: open(os.path.join(root, 'lib/data/%s.rce' % n)).read() for n in MODS}
USES = {n: (re.search(r'^# uses: (.*)$', SRC[n], re.M).group(1).split() if re.search(r'^# uses: ', SRC[n], re.M) else []) for n in MODS}
CHK = {n: [open(f).read() for f in sorted(glob.glob(os.path.join(root, 'tests/data/%s_checks*.rce' % n)))] for n in MODS}
M = [('events',) + m for m in [
    ('    if x >= 97 and x <= 102\n        return x - 87', '    if x >= 97 and x <= 102\n        return x - 86'),
    ('            elif e == "n"\n                o = o + "\\n"', '            elif e == "n"\n                o = o + "n"'),
    ('                    u = 65536 + ((u - 55296) << 10) + (l - 56320)', '                    u = 65536 + ((u - 55296) << 10) + l'),
    ('        elif c.len() == 1 and c[0] < 32\n', '        elif false\n'),
    ('    if cs[k] == "0"\n        k += 1\n    else', '    if false\n        k += 1\n    else'),
    ('        if k >= cs.len() or not ev_digit(&cs[k])\n            fail BadEvent(why: "JSON: bad fraction")', '        if k >= cs.len()\n            fail BadEvent(why: "JSON: bad fraction")'),
    ('    if ev_skip(&cs, at) != cs.len()', '    if false'),
    ('        while sorted and j > 0 and', '        while j > 0 and'),
    ('            return a[i] < b[i]\n        i += 1\n    return a.len() < b.len()', '            return a[i] > b[i]\n        i += 1\n    return a.len() < b.len()'),
    ('        if not ((x >= 97 and x <= 122) or (x >= 48 and x <= 57))', '        if not ((x >= 65 and x <= 122) or (x >= 48 and x <= 57))'),
    ('        if c.len() == 1 and (x < 32 or x == 127)', '        if c.len() == 1 and (x < 9 or x == 127)'),
    ('    let leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0', '    let leap = y % 4 == 0'),
    ('    if ev_n(&cs, 11, 2) > 23 or ev_n(&cs, 14, 2) > 59 or ev_n(&cs, 17, 2) > 60', '    if ev_n(&cs, 11, 2) > 24 or ev_n(&cs, 14, 2) > 59 or ev_n(&cs, 17, 2) > 60'),
    ('        if k == s\n            return false', '        if false\n            return false'),
    ('    if attr_text(e, "specversion") != "1.0"', '    if false'),
    ('        if prev != "" and not ev_key_lt(&prev, &a.name)', '        if false'),
    ('        if opt and attr_text(e, copy(a.name)) == ""', '        if false'),
    ('    return p[1] == "json" or p[1].endswith("+json")', '    return p[1] == "json"'),
    ('    if v < -2147483648 or v > 2147483647', '    if false'),
    ('            if keys[j] == keys[i]\n', '            if false\n'),
    ('    if dj.issome() and b64.issome()', '    if false'),
    ('    if ev_json_ct(e)\n        return DJson(v: v)', '    if true\n        return DJson(v: v)'),
    ('        if c.len() == 1 and x > 32 and x < 127 and x != 34 and x != 37', '        if c.len() == 1 and x > 32 and x < 127 and x != 34'),
    ('            if x == 224\n                lo = 160', '            if x == 224\n                lo = 128'),
    ('        elif x >= 194 and x <= 223', '        elif x >= 192 and x <= 223'),
    ('            o = o + cs[i + 1]\n            i += 2', '            o = o + cs[i]\n            i += 2'),
    ('    if m.startswith("application/cloudevents-batch")', '    if false'),
    ('        if (if fold then h.name.lower() == name.lower() else h.name == name)', '        if h.name == name'),
    ('    if ev_implied_json(e)\n        hs.push(Hdr(name: "content-type", value: "application/json"))\n    return Msg', '    return Msg'),
    ('    if key_mapper == "partitionkey" and r.has_key and not has_attr(&e, "partitionkey")', '    if false'),
    ('    let pk = key_mapper == "partitionkey" and has_attr(e, "partitionkey")\n    return Record(has_key: pk, key: if pk then attr_text(e, "partitionkey") else "", headers: hs', '    let pk = false\n    return Record(has_key: pk, key: if pk then attr_text(e, "partitionkey") else "", headers: hs'),
    ('        BBytes(b: x) -> return if x.len() == 0 then DNone else DBin(b: copy(x))', '        BBytes(b: x) -> return DBin(b: copy(x))'),
    ('    if not has_ct or is_json_type(ct)\n        return DJson', '    if is_json_type(ct)\n        return DJson'),
]] + [('feed',) + m for m in [
    ('    let prev = if log.len() == 0 then copy(GENESIS) else copy(log[log.len() - 1].hash)', '    let prev = copy(GENESIS)'),
    ('vals: [copy(ev), events.JsStr(t: copy(prev_hash)), events.JsNum(raw: seq.tostr())', 'vals: [copy(ev), events.JsStr(t: copy(prev_hash)), events.JsNum(raw: (seq + 1).tostr())'),
    ('        if e.seq != i as i64 or e.prev_hash != prev or', '        if e.seq != i as i64 or'),
    ('    if d != t.digest\n', '    if false\n'),
    ('    if t.height < 0 or t.height > log.len() as i64\n', '    if t.height < 0 or t.height > log.len() as i64 + 1\n'),
    ('    if p.len() != 2 or p[1].len() != 64 or not p[0].isdigit()', '    if p.len() != 2 or not p[0].isdigit()'),
    ('    if e.cmd != "getMore"\n', '    if false\n'),
    ('    if not e.server or e.code == 43\n', '    if not e.server\n'),
    ('    if e.wire >= 9\n', '    if e.wire >= 8\n'),
    ('        return e.labels.contains("ResumableChangeStreamError")', '        return true'),
    ('    let tok = if o.start_after != "" then copy(o.start_after) else copy(o.resume_after)', '    let tok = copy(o.resume_after)'),
    ('    if o.start_after == "" and o.resume_after == "" and o.start_at_op_time == "" and batch_len == 0 and pbrt == ""', '    if false'),
    ('    if batch_len == 0 and pbrt != ""\n', '    if pbrt != ""\n'),
    ('    cs.token = if last and pbrt != "" then copy(pbrt) else copy(id)', '    cs.token = copy(id)'),
    ('        if cs.opts.start_after != "" and not cs.returned\n', '        if cs.opts.start_after != ""\n'),
    ('    if cs.op_time != ""\n', '    if false\n'),
    ('        if frags[i] != want or (want > 1 and ofs[i] != of) or ofs[i] < 1 or frags[i] > ofs[i]', '        if frags[i] != want or ofs[i] < 1'),
    ('    if want != 1\n        fail', '    if false\n        fail'),
    ('    if src.session_id == tgt.session_id\n', '    if false\n'),
    ('        if fd_has_session(&s.session_id, &tgt.history, j)', '        if fd_has_session(&s.session_id, &tgt.history, j + 1)'),
    ('    if c.last_seq == seq\n', '    if false\n'),
    ('        if x.session_id != session and (h.len() as i64) < HISTORY_SIZE', '        if (h.len() as i64) < HISTORY_SIZE'),
    ('        if ib.key_height[i] + ib.window > h\n', '        if true\n'),
    ('    if ib.used >= ib.budget\n', '    if ib.used > ib.budget\n'),
    ('        if ib.pending_height[i] < durable\n', '        if ib.pending_height[i] <= durable\n'),
    ('    ib.used = 0\n', '    ib.used = ib.used\n'),
]]
def one(ab):
    i, (name, a, b) = ab
    src = SRC[name]
    assert src.count(a) == 1, a
    with tempfile.TemporaryDirectory() as d:
        for u in USES[name]:
            shutil.copy(os.path.join(root, 'lib', u + '.rce'), d)
        for chk in CHK[name]:
            f = os.path.join(d, name + '.rce'); open(f, 'w').write(src.replace(a, b) + '\n' + chk)
            if subprocess.run([RCE, 'test', d if USES[name] else f], capture_output=True).returncode != 0:
                return i, a, True
        return i, a, False
sel = sys.argv[1:] or MODS
killed = 0; n = 0
with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
    for i, a, dead in ex.map(one, [x for x in enumerate(M) if x[1][0] in sel]):
        killed += dead; n += 1
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, n))
sys.exit(0 if killed == n else 1)
