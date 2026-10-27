# Mutation test for lib/obs/log.rce: apply each mutant to a copy of the
# module, append each tests/obs/log_checks*.rce, and require at least one
# `rce test` to fail. Usage: RCE=path/to/rce python3 scripts/mut/obs.py
import glob, os, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
src = open(os.path.join(root, 'lib/obs/log.rce')).read()
chks = [open(f).read() for f in sorted(glob.glob(os.path.join(root, 'tests/obs/log_checks*.rce')))]
M = [
    # severity
    ('    return if k == 0 then base else base + (k + 1).tostr()', '    return if k == 0 then base else base + k.tostr()'),
    ('        b.push(if x >= 97 and x <= 122 then x - 32 else x)', '        b.push(if x >= 97 and x <= 122 then x - 31 else x)'),
    ('        if d < 0 or d > 9 or acc < (lim + d) / 10', '        if d < 0 or d > 9'),
    # logfmt writing
    ('        if not ((c.len_bytes() == 1 and (x <= 32 or x == 61 or x == 34)) or c == "\\u{fffd}")', '        if not ((c.len_bytes() == 1 and (x < 32 or x == 61 or x == 34)) or c == "\\u{fffd}")'),
    ('    if s == "null"\n        return ["\\"null\\""]', '    if s == "nil"\n        return ["\\"null\\""]'),
    ('            e = if b < 32 then copy(OLG_LF_ESC[b as usize]) else (if b == 92 then "\\\\\\\\" else "\\\\\\"")', '            e = if b < 32 then copy(OLG_LF_ESC[b as usize]) else (if b == 92 then "\\\\" else "\\\\\\"")'),
    ('        elif b == 239 and olg_is_fffd(s, i)\n            e = "\\\\ufffd"', '        elif b == 239 and olg_is_fffd(s, i) and false\n            e = "\\\\ufffd"'),
    ('fn olg_lf_needs_quote(s: &text) -> bool = olg_has_ctl(s) or s.contains(" ") or', 'fn olg_lf_needs_quote(s: &text) -> bool = olg_has_ctl(s) or'),
    ('            fail BadLog(at: i as i64, why: "invalid key")', '            i += 1\n            continue'),
    # logfmt reading
    ('            if e == 34 or e == 92 or e == 47 or e == 39', '            if e == 34 or e == 92 or e == 47'),
    ('                    if rr < 56320 and r1 >= 56320 and r1 < 57344', '                    if r1 >= 56320 and r1 < 57344'),
    ('                        rr = 65533', '                        rr = 65533\n                        r += 6'),
    ('        elif c == 34 or c < 32\n            return bad', '        elif c == 34\n            return bad'),
    ('        if multi and olg_has_fffd(&key)', '        if olg_has_fffd(&key) and false'),
    ('                    if c == 34\n                        return olg_lf_fault(kvs, pos, "unexpected', '                    if c == 34 and false\n                        return olg_lf_fault(kvs, pos, "unexpected'),
    ('                        return olg_lf_fault(kvs, pos, "invalid quoted value")', '                        return olg_lf_fault(kvs, q, "invalid quoted value")'),
    ('        return s[0:n - 1]\n    return copy(s)', '        return copy(s)\n    return copy(s)'),
    ('            elif (rd.rest.len_bytes() + (i - a)) as i64 > rd.max', '            elif (rd.rest.len_bytes() + (i - a)) as i64 > rd.max + 1'),
    ('            rd.skip = true\n', ''),
    # JSON
    ('            if b >= 32 and b != 34 and b != 92 and not (ascii and b == 127)', '            if b >= 32 and b != 34 and b != 92'),
    ('                e = "\\\\u" + olg_hex4(55296 + (u >> 10)) + "\\\\u" + olg_hex4(56320 + (u & 1023))', '                e = "\\\\u" + olg_hex4(55296 + (u >> 10)) + "\\\\u" + olg_hex4(56320 + (u & 511))'),
    ('        let k = olg_digits(s, i + 1)\n        if k == i + 1\n            return false', '        let k = olg_digits(s, i + 1)'),
    ('            out[j as usize].val = copy(kv.val)', '            out.push(copy(kv))'),
    # records and times
    ('        let k = if kv.key == "time" or kv.key == "level" or kv.key == "msg" then "fields." + kv.key else copy(kv.key)', '        let k = if kv.key == "time" or kv.key == "level" then "fields." + kv.key else copy(kv.key)'),
    ('    while k > 0 and ((t[k as usize] as i64) & 192) == 128 bound 4', '    while k > 0 and ((t[k as usize] as i64) & 192) == 64 bound 4'),
    ('        while k > frac bound 9\n            fr = fr / 10', '        while k > frac bound 9\n            fr = (fr + 5) / 10'),
    ('    return o + (if off < 0 then "-" else "+") + olg_pad(a / 60, 2)', '    return o + (if off <= 0 then "+" else "-") + olg_pad(a / 60, 2)'),
    ('    let leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0', '    let leap = y % 4 == 0'),
    ('    if days > 106748 or days < -106748\n        return bad\n', ''),
    ('    if t > 9000000000000000000 or t < -9000000000000000000', '    if t > 9200000000000000000 or t < -9000000000000000000'),
    # filters, ring, sampler
    ('        LcNe(key: k, t: t) -> return olg_val_text(r, k) != some(copy(t))', '        LcNe(key: k, t: t) -> return olg_find(&r.kvs, k) >= 0 and olg_val_text(r, k) != some(copy(t))'),
    ('    if r.sev < f.min_sev or r.sev > f.max_sev', '    if r.sev < f.min_sev or r.sev >= f.max_sev'),
    ('    g.head = (g.head + 1) % g.cap', '    g.head = 0'),
    ('    let drop = n > s.first and (s.thereafter == 0 or (n - s.first) % s.thereafter != 0)', '    let drop = n > s.first and (s.thereafter == 0 or n % s.thereafter != 0)'),
    ('    if ra > tn\n', '    if ra >= tn\n'),
    ('    let key = (r.sev, olg_fnv32a(&r.msg) % 4096)', '    let key = (r.sev, olg_fnv32a(&r.msg) % 4093)'),
    # line limit
    ('            if len + pl > room\n                cut = true', '            if len + pl > room + 600000\n                cut = true'),
]
killed = 0
with tempfile.TemporaryDirectory() as d:
    for i, (a, b) in enumerate(M):
        assert src.count(a) == 1, a
        dead = False
        for j, c in enumerate(chks):
            f = os.path.join(d, 'm%d_%d.rce' % (i, j))
            open(f, 'w').write(src.replace(a, b) + '\n' + c)
            if subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0:
                dead = True; break
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
