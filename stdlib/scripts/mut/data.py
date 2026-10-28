# Mutation test for lib/data/*.rce: each mutant must fail the comptime
# checks (any shard of tests/data/<name>_checks*.rce). A module that
# imports others is tested as a package with the modules it needs
# (scripts/uses.py reads them from its `use` items).
# Equivalent mutant removed: the E0 hi bound in utf8_text (a surrogate
# that gets past it is refused by cp_text). For feed: searching the
# source history for the target's session from i instead of i + 1 (the
# entry at i was already checked against the target history). For cdc:
# the empty transaction id of a non-transactional message (pgoutput and
# wal2json format 1) - PostgreSQL decodes such a message outside any
# transaction, where the current id is already "". For sync: the
# RFC 6902 4.4 check that a move's from is not a proper prefix of its
# path (the remove of from takes the path's parent with it, so the add
# fails anyway; the check stays as the RFC's stated rule).
# Usage: RCE=path/to/rce python3 scripts/mut/data.py [module...]
import glob, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
MODS = ('events', 'feed', 'cdc', 'sync')
SRC = {n: open(os.path.join(root, 'lib/data/%s.rce' % n)).read() for n in MODS}
sys.path.insert(0, os.path.join(root, 'scripts'))
import uses
USES = {n: uses.deps(os.path.join(root, 'lib/data/%s.rce' % n)) for n in MODS}
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

]] + [('cdc',) + m for m in [
    ('    if c >= 97 and c <= 102\n        return c - 87', '    if c >= 97 and c <= 102\n        return c - 86'),
    ('        o = HEXU[(v & 15) as usize] + o', '        o = o + HEXU[(v & 15) as usize]'),
    ('    var v: i64 = if top >= 128 then top - 256 else top', '    var v: i64 = top'),
    ('        if cp >= 65536 and (cp < 126976 or cp >= 131072)\n            return false', '        if false\n            return false'),
    ('            lo = if x == 224 then 160 else 128', '            lo = 128'),
    ('        if cd_ci_prefix(t, "NaN") or cd_ci_prefix(t, "Infinity") or cd_ci_prefix(t, "-Infinity")', '        if cd_ci_prefix(t, "NaN") or cd_ci_prefix(t, "Infinity")'),
    ('        return events.JsBool(v: t == "t")', '        return events.JsBool(v: t != "f" and t != "t")'),
    ('    if oid == 17\n', '    if false\n'),
    ('        elif kd == 117\n            have = false', '        elif kd == 117\n            have = true'),
    ('        if have and (rel.key[i] or not key_only)', '        if have'),
    ('            key.push((f & 1) == 1)', '            key.push(true)'),
    ('        if not have_key\n            key = pg_keyof(&post, &rel)', '        if false\n            key = pg_keyof(&post, &rel)'),
    ('        let cts = try rd_i64(&mut r)\n', '\n'),
    ('            let ns = cd_field(&j, "schema") + "." + cd_field(&j, "table")', '            let ns = cd_field(&j, "table")'),
    ('            let key = if a == "U" or a == "D" then w2_cols(&ident) else events.JsNull', '            let key = if a == "D" then w2_cols(&ident) else events.JsNull'),
    ('    if coll != ""\n        ns = db + "." + coll', '    if coll != ""\n        ns = coll'),
    ('    elif tbl != "" and sch != ""\n        ns = sch + "." + tbl', '    elif tbl != "" and sch != ""\n        ns = db + "." + tbl'),
    ('    if pos == ""\n        pos = cd_field(&src, "ord")', '    if pos == ""\n        pos = ""'),
    ('    let post = if op == "m" then msg else post0', '    let post = post0'),
    ('    if t == "update" or t == "replace"\n        return "u"', '    if t == "update"\n        return "u"'),
    ('    let ns = if coll == "" then cd_field(&nsj, "db") else cd_field(&nsj, "db") + "." + coll', '    let ns = cd_field(&nsj, "db") + "." + coll'),
    ('    if b.len() < 9 or b[0] != 130', '    if b.len() < 9'),
    ('        doc = mg_trunc(&doc, &p, mg_int(&ns) as usize)', '        doc = copy(doc)'),
    ('        t = cd_field(j, "$numberInt")', '        t = ""'),
    ('        events.JsArr(items: xs) -> return mg_unset_arr(xs, &path[0], &rest)\n', ''),
    ('        doc = mg_unset(&doc, &p)', '        doc = copy(doc)'),
    ('        nv.push(mg_set(&nul, rest, v))\n', '\n'),
    ('    let op = if del then "d" else (if rev.startswith("1-") then "c" else "u")', '    let op = if del then "d" else "u"'),
    ('    let post = if del then events.JsNull else doc', '    let post = doc'),
    ('        if i == 8\n            return (v << 8) | b', '        if i == 8\n            return (v << 7) | b'),
    ('                p.push((try rd_u8(&mut r)) != 0)', '                p.push((try rd_u8(&mut r)) == 0)'),
    ('            if not sq_undef(&v)\n', '            if true\n'),
    ('                let key = if patch then sq_pkof(&nw, &pk) else sq_pkof(&old, &pk)', '                let key = sq_pkof(&nw, &pk)'),
    ('        if t == 61\n            var o', '        if false\n            var o'),
    ('    if not cd_is_null(&m)\n        let xs = cd_items(&m)', '    if false\n        let xs = cd_items(&m)'),
    ('    return es.len() > 0 and cd_is_arr(&es[0])', '    return es.len() > 0'),
    ('    let o: usize = if t.len() == 9 then 2 else 4', '    let o: usize = 2'),
    ('    let dom = if t.len() == 9 then "" else (if t[2] == "_" then "" else copy(t[2]))', '    let dom = if t.len() == 9 then "" else copy(t[2])'),
    ('            f.headers.push(events.JsArr(items: [events.JsStr(t: copy(c[0])), events.JsStr(t: js_trim(&vj))]))', '            f.headers.push(events.JsArr(items: [events.JsStr(t: copy(c[0])), events.JsStr(t: copy(vj))]))'),
    ('    if nf == want + 1\n        f.reply = copy(ps[3])', '    if false\n        f.reply = copy(ps[3])'),
    ('    return cd_field(&j, "stream") + ":" + cd_field(&j, "seq") + (if dup then ":dup" else "")', '    return cd_field(&j, "stream") + ":" + cd_field(&j, "seq")'),
]] + [('sync',) + m for m in [
    ('    if not p.startswith("/")\n        fail BadSync(why: "pointer: must start with /")', '    if false\n        fail BadSync(why: "pointer: must start with /")'),
    ('            if i + 1 < cs.len() and cs[i + 1] == "0"\n                o = o + "~"', '            if i + 1 < cs.len() and cs[i + 1] == "0"\n                o = o + "/"'),
    ('            else\n                fail BadSync(why: "pointer: bad ~ escape")', '            else\n                o = o + "~"'),
    ('        o = o + (if c == "~" then "~0" else (if c == "/" then "~1" else c))', '        o = o + (if c == "/" then "~1" else c)'),
    ('    if t.len() > 1 and t.startswith("0")\n        return -1', '    if false\n        return -1'),
    ('    if n < 0 or n as usize >= xs.len()\n        fail BadSync(why: "pointer: bad array index")', '    if n < 0 or n as usize > xs.len()\n        fail BadSync(why: "pointer: bad array index")'),
    ('            e -= 1\n            i += 1', '            i += 1'),
    ('        e += if eneg then -x else x', '        e += x'),
    ('    while b > a and dc[b - 1] == "0" bound 16777216\n        b -= 1\n        e += 1', '    while false bound 16777216\n        b -= 1\n        e += 1'),
    ('        events.JsNull -> return sy_is_null(b)', '        events.JsNull -> return true'),
    ('    if ks.len() != k2.len()\n        return false', '    if false\n        return false'),
    ('    if xs.len() != ys.len()\n        return false', '    if xs.len() > ys.len()\n        return false'),
    ('        if i + 1 != toks.len()\n            fail BadSync(why: "patch: path not found")', '        if false\n            fail BadSync(why: "patch: path not found")'),
    ('    if n < 0 or n as usize > xs.len() or (not last and n as usize == xs.len())', '    if n < 0 or n as usize > xs.len()'),
    ('            if last\n                o.push(copy(v))\n                o.push(copy(xs[j]))', '            if last\n                o.push(copy(xs[j]))\n                o.push(copy(v))'),
    ('        fail BadSync(why: "patch: cannot remove the root")', '        return events.JsNull'),
    ('    if not found\n        fail BadSync(why: "patch: path not found")\n    return events.JsObj(keys: nk, vals: nv)', '    return events.JsObj(keys: nk, vals: nv)'),
    ('    if sy_dup_members(op)\n', '    if false\n'),
    ('        if name == "copy"\n            return try jp_add(doc, &path, 0, &v)', '        if name == "copy"\n            return try jp_add(doc, &src, 0, &v)'),
    ('        if not json_eq(&have, &v)\n', '        if false\n'),
    ('    fail BadSync(why: "patch: unknown op")', '    return copy(doc)'),
    ('        if sy_is_null(&pv[i])\n            if at >= 0', '        if false\n            if at >= 0'),
    ('            let nv = merge_patch(&vs[at as usize], &pv[i])', '            let nv = copy(pv[i])'),
    ('            vs.push(merge_patch(&nul, &pv[i]))', '            vs.push(copy(pv[i]))'),
    ('    while k > n bound 16777216\n        k -= 1', '    while k > n + 1 bound 16777216\n        k -= 1'),
    ('        let p = at + "/" + ptr_escape(&ka[i])', '        let p = at + "/" + ka[i]'),
    ('    return a.who < b.who', '    return a.who > b.who'),
    ('                    if mo_same(&q, &o.id)\n                        dead = true', '                    if false\n                        dead = true'),
    ('            vs.push(copy(vis[vis.len() - 1].value))', '            vs.push(copy(vis[0].value))'),
    ('        if o.key == key and not o.del', '        if o.key == key'),
    ('            if t.replica == s.replica and t.n > n', '            if t.replica == s.replica'),
    ('    return PN(p: copy(c.p), n: gc_inc(&c.n, replica, -by))', '    return PN(p: copy(c.p), n: gc_inc(&c.n, replica, by))'),
    ('    if b.ts > a.ts or (b.ts == a.ts and b.replica > a.replica)', '    if b.ts > a.ts'),
    ('        if t.elem == elem and not g.contains(&t.tag)', '        if not g.contains(&t.tag)'),
    ('    for t in b.gone\n        if not g.contains(&t)\n            g.push(copy(t))', '    for t in b.gone\n        if false\n            g.push(copy(t))'),
    ('        if not s.gone.contains(&t.tag) and not out.contains(&t.elem)', '        if not out.contains(&t.elem)'),
    ('        if m.id > last + 1\n            fail BadSync(why: "push: mutation id gap")', '        if false\n            fail BadSync(why: "push: mutation id gap")'),
    ('        if m.id == last + 1\n            let st', '        if m.id >= last\n            let st'),
    ('    if not ok or by.isnone()\n        return copy(st)', '    if by.isnone()\n        return copy(st)'),
    ('        if m.id > p.lmid\n            keep.push(copy(m))', '        if m.id >= p.lmid\n            keep.push(copy(m))'),
    ('        if kv_get(db, &k).isnone()\n            ops.push(pl_op("del", &k, none))', '        if false\n            ops.push(pl_op("del", &k, none))'),
    ('    if op == "!="\n        return c != 0', '    if op == "!="\n        return c == 0'),
    ('    if op == "<="\n        return c <= 0', '    if op == "<="\n        return c < 0'),
    ('    if op == ">="\n        return c >= 0', '    if op == ">="\n        return c > 0'),
    ('        none -> return false\n\nfn cond_value', '        none -> return true\n\nfn cond_value'),
    ('        if not rows_has(&b, &r)\n            dels.push(copy(r))', '        if true\n            dels.push(copy(r))'),
    ('        elif c.action == "U"\n            nx.push(copy(c.new))', '        elif c.action == "U"\n            nx.push(copy(c.old))'),
    ('        let key = if c.action == "I" then row_id(&c.new) else row_id(&c.old)', '        let key = row_id(&c.new)'),
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
