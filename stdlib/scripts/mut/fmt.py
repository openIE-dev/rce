# Mutation test for lib/fmt/fmt.rce: apply each mutant to a copy of the
# module, append each tests/fmt/fmt_checks*.rce, and require at least one
# `rce test` to fail. Usage: RCE=path/to/rce python3 scripts/mut/fmt.py
# (JOBS mutants at a time, default the number of CPUs).
import concurrent.futures, glob, os, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
src = open(os.path.join(root, 'lib/fmt/fmt.rce')).read()
chks = [open(f).read() for f in sorted(glob.glob(os.path.join(root, 'tests/fmt/fmt_checks*.rce')))]
M = [
    ('        let a = if d < 0 then 0 - d else d', '        let a = d'),
    ('    if c >= 65 and c <= 90\n        return c - 55', '    if c >= 65 and c <= 90\n        return c - 56'),
    ('        if d >= r\n', '        if d > r\n'),
    ('        if acc == lim\n', '        if acc == lim + 1\n'),
    ('    if i >= n\n        fail BadInt(at: i as i64)', '    if i > n\n        fail BadInt(at: i as i64)'),
    ('        if k > 0 and (nd - k) % 3 == 0', '        if k > 0 and (nd - k) % 3 == 1'),
    ('            if i + 1 < n and cs[i + 1] == "{"', '            if i + 1 < n and cs[i + 1] == "}"'),
    ('                if idx < 0\n                    next += 1', '                if idx < 0\n                    next += 2'),
    ('    return rx_cat(rx_cat(rx_cat(ins(OP_SPLIT, 1, na + 2), a), ins(OP_JMP, nb + 1, 0)), b)', '    return rx_cat(rx_cat(rx_cat(ins(OP_SPLIT, 1, na + 1), a), ins(OP_JMP, nb + 1, 0)), b)'),
    ('    return hit != (cl[at] == 1)', '    return hit'),
    ('        return [48, 57, 65, 90, 95, 95, 97, 122]', '        return [48, 57, 65, 90, 97, 122]'),
    ('        return [9, 13, 32, 32]', '        return [9, 10, 32, 32]'),
    ('    if c >= 224\n        return 3', '    if c >= 224\n        return 2'),
    ('        if c == 93 and not first', '        if c == 93'),
    ('            if hi < lo\n', '            if hi < lo - 1\n'),
    ('        o.push(t & NB_MASK)\n        c = t >> 24\n    if c > 0\n        o.push(c)\n    return nb_trim(o)', '        o.push(t & NB_MASK)\n        c = t >> 23\n    if c > 0\n        o.push(c)\n    return nb_trim(o)'),
    ('            # the estimate was one too large: add the divisor back\n            qh -= 1', '            # the estimate was one too large: add the divisor back\n            qh -= 0'),
    ('        while qh >= NB_BASE or qh * vs > ((rh << 24) | u[j + n - 2]) bound 3', '        while qh >= NB_BASE bound 3'),
    ('    if d > 5 or (d == 5 and (nd > keep + 1 or dx[k] % 2 == 1))', '    if d >= 5'),
    ('        if dx[1] > 5 or (dx[1] == 5 and nd > 1)', '        if dx[1] >= 5'),
    ('        if decpt <= -4 or decpt > 16', '        if decpt <= -4 or decpt > 15'),
    ('        let lim = if add_dot0 then prec - 1 else prec', '        let lim = prec'),
    ('        if ea < 10\n            o.push(48)', '        if ea < 1\n            o.push(48)'),
    ('    while m % 2 == 0 bound 64\n        m = m / 2\n        e += 1', '    while m % 4 == 0 bound 64\n        m = m / 2\n        e += 1'),
    ('    if ty == 37\n        body.push(37)', '    if ty == 37\n        body.push(36)'),
    ('        x = x * 100.0', '        x = x * 10.0'),
    ('    if f.z and dx.len() == 2 and dx[1] == 0', '    if f.z'),
    ('            min_width -= 1\n', ''),
    ('        if f.align == 94\n            lp = npad / 2', '        if f.align == 94\n            lp = npad - npad / 2'),
    ('    if not f.fill_set and i < n and cs[i] == "0"', '    if i < n and cs[i] == "0"'),
    ('        if g.align == 0\n            g.align = 61', '        if g.align == 0\n            g.align = 62'),
    ('    if g == 95 and (t == 98 or t == 111 or t == 120 or t == 88)\n        return 4', '    if g == 95 and (t == 98 or t == 111 or t == 120 or t == 88)\n        return 3'),
    ('    let lp = if f.align == 62 then total - n else (if f.align == 94 then (total - n) / 2 else 0)', '    let lp = if f.align == 62 then total - n else (if f.align == 94 then (total - n + 1) / 2 else 0)'),
    ('        if v < 0 or v > 1114111 or (v >= 55296 and v <= 57343)', '        if v < 0 or v > 1114111'),
    ('        if nb_bit(n, s - 1) == 1 and (low or m % 2 == 1)', '        if nb_bit(n, s - 1) == 1 and low'),
    ('        if e10 < 0 and e10 >= -22', '        if e10 < 0 and e10 >= -23'),
    ('    if sticky\n        ds.push(1)', '    if false\n        ds.push(1)'),
    ('    if dexp > 309', '    if dexp > 308'),
    ('    let t = (nb_bitlen(&p) - nb_bitlen(&dig) + 56).max(0)', '    let t = (nb_bitlen(&p) - nb_bitlen(&dig) + 30).max(0)'),
    ('    if m == 9007199254740992\n        m = 4503599627370496\n        u += 1', '    if m == 9007199254740992\n        m = 4503599627370496'),
    ('    if r.end == 0 or r.end < (s.len_bytes() as i64)', '    if r.end == 0'),
    ('        let head = if lazy then ins(OP_SPLIT, n + 4, 1) else ins(OP_SPLIT, 1, n + 4)', '        let head = if lazy then ins(OP_SPLIT, n + 4, 1) else ins(OP_SPLIT, n + 4, 1)'),
    ('then pc + re.code[3 * pc as usize + 2] else pc + 1', 'then pc + 1 else pc + 1'),
    ('            lk += 1\n', '            lk += 0\n'),
    ('(st.nonempty and st.pos == st.origin)', '(st.nonempty and false)'),
    ('    let nul = nullable(&f) and n - m >= 2', '    let nul = false'),
    ('        return rx_cat(copy(f), star(f, lazy))', '        return star(f, lazy)'),
    ('    return rx_cat(f, if lazy then ins(OP_SPLIT, 1, 0 - n) else ins(OP_SPLIT, 0 - n, 1))', '    return rx_cat(f, ins(OP_SPLIT, 0 - n, 1))'),
    ('            if peek(p) == 63\n                lazy = true', '            if peek(p) == 63\n                lazy = false'),
    ('            return (k as i64) + 1', '            return k as i64'),
    ('                    if d == 110\n                        lit = 10', '                    if d == 110\n                        lit = 110'),
    ('                let r2 = if e == 68 or e == 87 or e == 83 then cp_complement(&er) else er', '                let r2 = er'),
    ('        let nd: i64 = if e == 120 then 2 else (if e == 117 then 4 else 8)', '        let nd: i64 = if e == 120 then 2 else (if e == 117 then 4 else 4)'),
    (' and not (a == 0 and pos == 0 and at_end))', ')'),
    ('            if not ((st.mode == 2 and not at_end) or', '            if not ((st.mode == 3 and not at_end) or'),
    ('            if (a as usize) < ncap\n', '            if (a as usize) < 2\n'),
    ('    out.push(s[last:n])\n    return out', '    return out'),
    ('    if not st.found and (st.mode == 0 or st.pos == st.origin)', '    if not st.found'),
    ('    while j > 0 and (s[j] as i64) >= 128 and (s[j] as i64) < 192 bound 4\n        j -= 1\n    return rx_cp_at(s, j)', '    return -1'),
]
def run(i, d):
    a, b = M[i]
    for j, c in enumerate(chks):
        f = os.path.join(d, 'm%d_%d.rce' % (i, j))
        open(f, 'w').write(src.replace(a, b) + '\n' + c)
        dead = subprocess.run([RCE, 'test', f], capture_output=True).returncode != 0
        os.remove(f)
        if dead:
            return True
    return False
for a, b in M:
    assert src.count(a) == 1, a
with tempfile.TemporaryDirectory() as d:
    with concurrent.futures.ThreadPoolExecutor(int(os.environ.get('JOBS', os.cpu_count() or 1))) as ex:
        res = list(ex.map(lambda i: run(i, d), range(len(M))))
for i, dead in enumerate(res):
    print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, M[i][0].strip().split('\n')[0]))
print('%d of %d mutants killed' % (sum(res), len(M)))
sys.exit(0 if all(res) else 1)
