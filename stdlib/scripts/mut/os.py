# Mutation test for lib/os/{path,env,perm,clock}.rce: each mutant must fail the comptime
# checks (three equivalent mutants removed: a dot test when there is
# neither dot nor separator, a UNC share without a second separator, and
# keeping an empty drive; for env two equivalent ones in Go's "${"
# scanning, whose fast paths give what the general scan gives). Usage: RCE=path/to/rce python3 scripts/mut/os.py
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
MODS = ('path', 'env', 'perm', 'clock')
SRC = {n: open(os.path.join(root, 'lib/os/%s.rce' % n)).read() for n in MODS}
# every shard (tests/os/<name>_checks*.rce); a mutant is killed when any shard fails
CHK = {n: [open(f).read() for f in sorted(glob.glob(os.path.join(root, 'tests/os/%s_checks*.rce' % n)))] for n in MODS}
M = [('path',) + m for m in [
    ('            if cs[k] != "."\n', '            if true\n'),
    ('    if cs.len() < 2 or cs[1] != "/" or (cs.len() > 2 and cs[2] == "/")', '    if cs.len() < 2 or cs[1] != "/"'),
    ('        elif b.startswith("/") or path == ""', '        elif b.startswith("/")'),
    ('        elif path.endswith("/")\n            path = path + b', '        elif false\n            path = path + b'),
    ('    if head != "" and not all\n        head = rstrip_seps(&head, false)', '    if head != ""\n        head = rstrip_seps(&head, false)'),
    ('        if c != ".." or (init == "" and out.len() == 0) or (out.len() > 0 and out[out.len() - 1] == "..")', '        if c != ".." or (out.len() > 0 and out[out.len() - 1] == "..")'),
    ('    return if o == "" then "." else o', '    return o'),
    ('    while i < sl.len() and i < tl.len() and sl[i] == tl[i] bound', '    while false bound'),
    ('            let start: usize = if up == "\\\\\\\\?\\\\UNC\\\\" then 8 else 2', '            let start: usize = 2'),
    ('        if n.len() > 2 and n[2] == "\\\\"\n', '        if n.len() > 2 and n[2] == "/"\n'),
    ('    return (s.len() >= 3 and s[1] == ":" and s[2] == "\\\\") or', '    return (s.len() >= 2 and s[1] == ":") or'),
    ('            if q.drive.lower() != rd.lower()\n', '            if q.drive != rd\n'),
    ('    if rp != "" and rr == "" and dc.len() > 0 and not (dc[dc.len() - 1] == ":" or is_sep(&dc[dc.len() - 1], true))', '    if false'),
    ('            elif i == 0 and r.root != ""\n', '            elif false\n'),
    ('    if prefix == "" and comps.len() == 0\n', '    if comps.len() == 0\n'),
    ('    if win_normcase(&sa.drive) != win_normcase(&pa.drive)\n', '    if false\n'),
    ('    while i < sl.len() and i < tl.len() and win_normcase(&sl[i]) == win_normcase(&tl[i]) bound', '    while i < sl.len() and i < tl.len() and sl[i] == tl[i] bound'),
    ('        if r.root != r0.root\n', '        if false\n'),
    ('        if p.startswith("/") != abs\n', '        if false\n'),
    ('            return text_lt(&a[i], &b[i])', '            return false'),
    ('        return n != "." and n != ".."', '        return true'),
    ('        if c.len() == 1 and (c[0] < 32 or c == "\\"" or', '        if c.len() == 1 and (c == "\\"" or'),
    ('    let s = rstrip_spaces(&stem).upper()', '    let s = stem.upper()'),
    ('COM9 COM\\u{b9}', 'COM9'),
    ('            if out.len() == 0\n                fail BadPath(why: "escapes the directory")', '            if false\n                fail BadPath(why: "escapes the directory")'),
    ('        if ok and qc.len() as i64 >= blen\n', '        if ok and qc.len() as i64 > blen\n'),
    ('    var ok = preopens[i].startswith("/") and qc.len() <= pc.len()', '    var ok = qc.len() <= pc.len()'),
]] + [('env',) + m for m in [
    ('    var i: usize = if win and cs.len() > 0 then 1 else 0\n    while i < cs.len() bound 16777216\n        if cs[i] == "="', '    var i: usize = 0\n    while i < cs.len() bound 16777216\n        if cs[i] == "="'),
    ('fn same_key(a: &text, b: &text, win: bool) -> bool = if win then upper_ascii(a) == upper_ascii(b) else a == b', 'fn same_key(a: &text, b: &text, win: bool) -> bool = a == b'),
    ('    var i: usize = if win then 1 else 0\n    while i < cs.len() bound 16777216\n        if cs[i] == "=" or', '    var i: usize = 1\n    while i < cs.len() bound 16777216\n        if cs[i] == "=" or'),
    ('            if not done\n                o.push', '            if true\n                o.push'),
    ('    if not done\n        o.push(Var(key: copy(key)', '    if false\n        o.push(Var(key: copy(key)'),
    ('        if not dup\n            vs.push(v)', '        if true\n            vs.push(v)'),
    ('        while j > 0 and key_lt(&vs[j].key, &vs[j - 1].key) bound', '        while false bound'),
    ('            return a[i] < b[i]', '            return a[i] > b[i]'),
    ('        return Name(name: "", width: 1)', '        return Name(name: "", width: 0)'),
    ('            elif n.name == ""\n                o = o + "$"', '            elif n.name == ""\n                o = o'),
    ('fn special(c: &text) -> bool = c == "*" or c == "#" or', 'fn special(c: &text) -> bool = c == "*" or'),
    ('    return x == 95 or (x >= 48', '    return (x >= 48'),
    ('        if c == "\\""\n            q = not q', '        if false\n            q = not q'),
    ('            if i < n and cs[i] == "\\""\n                var h', '            if false\n                var h'),
    ('                if k % 2 == 1\n', '                if k % 2 == 0\n'),
    ('            if i < n and cs[i] == "\\""\n                cur = cur + "\\""\n                i += 1', '            if false\n                cur = cur + "\\""\n                i += 1'),
    ('    if cur != "" or q\n        out.push(cur)', '    if cur != ""\n        out.push(cur)'),
    ('    var quote = force or arg == ""', '    var quote = force'),
    ('                while h <= bs bound', '                while h < bs bound'),
    ('        while h < bs bound 16777216\n            o = o + "\\\\"\n            h += 1\n        o = o + "\\""', '        while false bound 16777216\n            o = o + "\\\\"\n            h += 1\n        o = o + "\\""'),
    ('    while i < n and ws(&cs[i]) bound 16777216\n        i += 1\n    out.push(cur)', '    out.push(cur)'),
]] + [('perm',) + m for m in [
    ('        return "p"\n', '        return "f"\n'),
    ('    let e = if special then (if x then lo else up) else (if x then "x" else "-")', '    let e = if special then lo else (if x then "x" else "-")'),
    ('    return type_char(m) + trio(m, 6, m & S_ISUID != 0, "s", "S") + trio(m, 3, m & S_ISGID != 0, "s", "S")', '    return type_char(m) + trio(m, 6, m & S_ISUID != 0, "s", "S") + trio(m, 3, m & S_ISUID != 0, "s", "S")'),
    ('        let men = if k < 5 then (v & (S_ISUID | S_ISGID)) | S_ISVTX | 511 else 4095', '        let men = 4095'),
    ('        if v < 0 or k != cs.len()\n', '        if v < 0\n'),
    ('            elif c == "o"\n                affected = affected | 519', '            elif c == "o"\n                affected = affected | 7'),
    ('                if v < 0 or affected != 0 or (k < cs.len() and cs[k] != ",")', '                if v < 0 or (k < cs.len() and cs[k] != ",")'),
    ('            elif c == "g"\n                value = 56\n                i += 1', '            elif c == "g"\n                value = 448\n                i += 1'),
    ('                    elif d == "s"\n                        value = value | 3072', '                    elif d == "s"\n                        value = value | 2048'),
    # equivalent, removed: men = value when mentioned == 0 (only '+'/'-' reach it, and omit is used only by '=')
    # equivalent, removed: early 'left == 0' return in access_check
    ('        let omit = (if dir then S_ISUID | S_ISGID else 0) & ~c.mentioned', '        let omit: i64 = 0'),
    ('            if value & 146 != 0\n                spread = spread | 146', '            if false\n                spread = spread | 146'),
    ('            if nm & 73 != 0 or dir\n', '            if nm & 73 != 0\n'),
    ('        value = value & (if c.affected != 0 then c.affected else ~umask) & ~omit', '        value = value & (if c.affected != 0 then c.affected else 4095) & ~omit'),
    ('            let preserved = (if c.affected != 0 then ~c.affected else 0) | omit', '            let preserved = (if c.affected != 0 then ~c.affected else 0)'),
    ('    if c.uid == owner\n', '    if c.uid == owner and false\n'),
    ('            if g == group\n                in_group = true', '            if false\n                in_group = true'),
    ('    return cls & want == want', '    return cls & want != 0 or want == 0'),
    ('        if first and j - i > 2 and cs[i] == "0" and (cs[i + 1] == "x" or cs[i + 1] == "X")', '        if false'),
    ('        if v < 0 or (first and v >= 281474976710656) or (not first and v >= 4294967296)', '        if v < 0 or (first and v >= 281474976710656)'),
    ('    if parts.len() > 15\n', '    if parts.len() > 16\n'),
    ('    var o = "S-1-" + (if s.authority < 4294967296 then', '    var o = "S-1-" + (if s.authority < 281474976710656 then'),
    ('        o.push((x >> 16) & 255)', '        o.push((x >> 8) & 255)'),
    ('    if b.len() < 8 or b[0] != 1 or b[1] > 15 or b.len() != 8 + 4 * b[1] as usize', '    if b.len() < 8 or b[1] > 15 or b.len() != 8 + 4 * b[1] as usize'),
    ('    if not has_dacl\n        return true', '    if not has_dacl\n        return false'),
    ('        if not a.allow and a.mask & left != 0\n', '        if not a.allow and a.mask & want != 0\n'),
    ('    if abi >= 3\n        r = r | LL_TRUNCATE', '    if abi >= 4\n        r = r | LL_TRUNCATE'),
    ('        let need = want & l.handled', '        let need = want'),
    ('    if rule.len() > p.len()\n        return false', '    if false\n        return false'),
    ('            r = r | LL_WRITE_FILE | LL_TRUNCATE |', '            r = r | LL_WRITE_FILE |'),
    ('    if rights & LL_EXECUTE != 0\n', '    if rights & LL_READ_FILE != 0\n'),
]] + [('clock',) + m for m in [
    # equivalent, removed: padding when width == cells (zero spaces either way); the
    # seq tie-break in before() (timers are inserted in seq order, so equal deadlines
    # already queue FIFO)
    ('    if extra > MAX_U - secs\n', '    if extra > MAX_U - secs + 1\n'),
    ('    if nanos < 0\n        fail BadClock(why: "negative nanoseconds")', '    if false\n        fail BadClock(why: "negative nanoseconds")'),
    ('nanos: (ms % 1000) as i64 * 1000000)', 'nanos: (ms % 1000) as i64 * 100000)'),
    ('    let cut: usize = if cs.len() > 9 then cs.len() - 9 else 0', '    let cut: usize = if cs.len() > 9 then cs.len() - 8 else 0'),
    ('            if secs > (MAX_U - v as u64) / 10\n', '            if secs > MAX_U / 10\n'),
    ('    if n > MAX_U / unit\n', '    if n > MAX_U / unit + 1\n'),
    ('pub fn subsec_micros(d: &Dur) -> i64 = d.nanos / 1000', 'pub fn subsec_micros(d: &Dur) -> i64 = d.nanos / 1000000'),
    ('    if d.secs == 0\n        return part.tostr()', '    if false\n        return part.tostr()'),
    ('pub fn as_micros(d: &Dur) -> text = in_units(d, 6, 1000)', 'pub fn as_micros(d: &Dur) -> text = in_units(d, 3, 1000)'),
    ('    if n >= NANOS_PER_SEC\n        n -= NANOS_PER_SEC\n        if s == MAX_U', '    if n > NANOS_PER_SEC\n        n -= NANOS_PER_SEC\n        if s == MAX_U'),
    ('        if s == 0\n            return none\n        s -= 1', '        s -= 1'),
    ('pub fn saturating_sub(a: &Dur, b: &Dur) -> Dur = checked_sub(a, b).or(dur_zero())', 'pub fn saturating_sub(a: &Dur, b: &Dur) -> Dur = checked_sub(a, b).or(dur_max())'),
    ('    if dur_cmp(a, b) >= 0\n        return checked_sub(a, b)', '    if true\n        return checked_sub(a, b)'),
    ('    if k < 0 or k > 4294967295\n        return none\n    let ku', '    if k < 0\n        return none\n    let ku'),
    ('    if ku != 0 and a.secs > (MAX_U - extra) / ku', '    if ku != 0 and a.secs > MAX_U / ku'),
    ('    let extra = (carry * NPS + a.nanos as u64 % ku) / ku', '    let extra = carry * NPS / ku'),
    ('    if k <= 0 or k > 4294967295\n        return none\n    let ku = k as u64\n    let s', '    if k < 0 or k > 4294967295\n        return none\n    let ku = k as u64\n    let s'),
    ('    let lim = if prec < 0 then 9 else prec', '    let lim = 9'),
    ('    if fp > 0 and fp >= dv * 5\n', '    if fp > 0 and fp > dv * 5\n'),
    ('        if not (tie and not odd)\n', '        if true\n'),
    ('        let odd = if pos > 0 then copy(buf[(pos - 1) as usize]) % 2 == 1 else ip % 2 == 1', '        let odd = ip % 2 == 1'),
    ('                if buf[r as usize] < 9\n', '                if buf[r as usize] < 10\n'),
    ('                if ip == MAX_U\n                    big = true', '                if false\n                    big = true'),
    ('    let end = if prec >= 0 then (if prec < 9 then prec else 9) else pos', '    let end = if prec >= 0 then prec else pos'),
    ('    let pre = if align == ">" then padn else (if align == "^" then padn / 2 else 0)', '    let pre = if align == ">" then padn else (if align == "^" then (padn + 1) / 2 else 0)'),
    ('    if d.nanos >= 1000000\n', '    if d.nanos > 1000000\n'),
    ('    let p = if plus then "+" else ""', '    let p = ""'),
    ('    var cells = prefix.chars().len() + postfix.chars().len()', '    var cells = prefix.len() + postfix.len()'),
    ('        if d > (MAX_SECS - t) as u64\n', '        if d > (MAX_SECS - t) as u64 + 1\n'),
    ('    let room = MAX_SECS as u64 + (0 - (t + 1)) as u64 + 1\n    if d > room', '    let room = MAX_SECS as u64 + (0 - (t + 1)) as u64\n    if d > room'),
    ('        if d > (t - MIN_SECS) as u64\n', '        if d > (t - MIN_SECS) as u64 + 1\n'),
    ('    let room = t as u64 + MAX_SECS as u64 + 1\n    if d > room', '    let room = t as u64 + MAX_SECS as u64\n    if d > room'),
    ('        if s == MIN_SECS\n            return none\n        s -= 1', '        s -= 1'),
    ('        if s == MAX_SECS\n            return none\n        s += 1\n        n -= NANOS_PER_SEC', '        s += 1\n        n -= NANOS_PER_SEC'),
    ('    return a as u64 + (0 - (b + 1)) as u64 + 1', '    return a as u64 + (0 - (b + 1)) as u64'),
    ('    if a.nanos >= b.nanos\n        return Dur(secs: diff_u', '    if a.nanos > b.nanos\n        return Dur(secs: diff_u'),
    ('        return Diff(ok: true, d: mag(a, b))', '        return Diff(ok: true, d: mag(b, a))'),
    ('    if ts_cmp(a, b) >= 0\n        return Diff', '    if ts_cmp(a, b) > 0\n        return Diff'),
    ('    if t.secs < 0 or t.secs as u64 > (MAX_U - t.nanos as u64) / NPS\n', '    if t.secs < 0 or t.secs as u64 > MAX_U / NPS\n'),
    ('    if seconds > MAX_SECS as u64\n', '    if false\n'),
    ('    if nanos < 0 or nanos >= NANOS_PER_SEC\n', '    if nanos < 0 or nanos > NANOS_PER_SEC\n'),
    ('    if m.seen and ts_cmp(r, &m.last) < 0\n', '    if ts_cmp(r, &m.last) < 0\n'),
    ('        m.raised += 1\n', ''),
    ('        if not placed and before(&t, &x)\n', '        if not placed and before(&x, &t)\n'),
    ('    let at = ts_saturating_add(now, d)', '    let at = copy(now)'),
    ('        if x.id == id\n            hit = true', '        if x.id == id and not hit\n            hit = true'),
    ('        if ts_cmp(&x.at, now) <= 0\n', '        if ts_cmp(&x.at, now) < 0\n'),
    ('    return some(copy(q.list[0].at))', '    return some(copy(q.list[q.list.len() - 1].at))'),
]]
def one(ab):
    i, (name, a, b) = ab
    src = SRC[name]
    assert src.count(a) == 1, a
    with tempfile.TemporaryDirectory() as d:
        for chk in CHK[name]:
            f = os.path.join(d, 'm.rce'); open(f, 'w').write(src.replace(a, b) + '\n' + chk)
            if subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0:
                return i, a, True
        return i, a, False
killed = 0
with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
    for i, a, dead in ex.map(one, enumerate(M)):
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)

