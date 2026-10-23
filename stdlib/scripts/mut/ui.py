# Mutation test for lib/ui/text.rce: each mutant must fail at least one
# shard of tests/ui/text_checks*.rce. The shards named by the mutant's hint
# run first; the rest run only if those all pass. Usage:
# RCE=path/to/rce python3 scripts/mut/ui.py
# Removed as equivalent: LB10's gcat = 1 for a lone CM (gcat is read only
# for QU and pictographic items, and a lone CM is neither once it is AL);
# W1's ON for NSM after an isolate initiator or PDI (no later rule in the
# sequence tells ON from PDI); N0's NSM after an opening bracket (N1 then
# sees that bracket on the left and a strong of the same direction before
# the closing bracket, so it resolves the same).
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
SRC = open(os.path.join(root, 'lib/ui/text.rce')).read()
UTF8 = open(os.path.join(root, 'lib/text/utf8.rce')).read()
SH = sorted(glob.glob(os.path.join(root, 'tests/ui/text_checks*.rce')))
def shards(h):
    return [f for f in SH if os.path.basename(f).startswith('text_checks' + h)]
M = [
 # UAX 29 graphemes
 ('', '        if prev == G_CR and cur == G_LF\n            brk = false', '        if false\n            brk = false'),
 ('', '        elif prev == G_L and (cur == G_L or cur == G_V or cur == G_LV or cur == G_LVT)', '        elif prev == G_L and (cur == G_L or cur == G_V or cur == G_LV)'),
 ('', '        elif (prev == G_LVT or prev == G_T) and cur == G_T', '        elif prev == G_T and cur == G_T'),
 ('', '        elif cur == G_EXTEND or cur == G_ZWJ or cur == G_SPACINGMARK or prev == G_PREPEND', '        elif cur == G_EXTEND or cur == G_ZWJ or prev == G_PREPEND'),
 ('', '        elif conj == 2 and ic == 2\n', '        elif conj >= 1 and ic == 2\n'),
 ('', '        elif emo == 2 and ext_pict(c)\n', '        elif emo >= 1 and ext_pict(c)\n'),
 ('', '        elif prev == G_RI and cur == G_RI and ri % 2 == 1', '        elif prev == G_RI and cur == G_RI'),
 ('', '        elif emo == 1 and cur == G_EXTEND\n            emo = 1', '        elif false\n            emo = 1'),
 # words
 ('', '        elif rp == W_ZWJ and ext_pict(copy(cs[i]))', '        elif false'),
 ('', '        elif rp == W_WSEG and cur == W_WSEG', '        elif false'),
 ('', '            elif w_ah(a) and w_midlq(cur) and w_ah(b2)', '            elif w_ah(a) and w_midlq(cur)'),
 ('', '            elif a == W_HL and cur == W_SQ\n', '            elif false\n'),
 ('', '            elif a2 == W_NUMERIC and w_midnq(a) and cur == W_NUMERIC', '            elif w_midnq(a) and cur == W_NUMERIC'),
 ('', '            elif a == W_ENL and (w_ah(cur) or cur == W_NUMERIC or cur == W_KATAKANA)', '            elif a == W_ENL and w_ah(cur)'),
 ('', '                brk = cnt % 2 == 0', '                brk = false'),
 ('', 'fn w_midnq(k: i64) -> bool = k == W_MIDNUM or k == W_MIDNUMLET or k == W_SQ', 'fn w_midnq(k: i64) -> bool = k == W_MIDNUM or k == W_MIDNUMLET'),
 # sentences
 ('', '            if a == S_ATERM and cur == S_NUMERIC', '            if false'),
 ('', '            elif a == S_ATERM and (a2 == S_UPPER or a2 == S_LOWER) and cur == S_UPPER', '            elif a == S_ATERM and cur == S_UPPER'),
 ('', '            elif pat and not sp and (cur == S_CLOSE or cur == S_SP or s_para(cur))', '            elif pat and (cur == S_CLOSE or cur == S_SP or s_para(cur))'),
 ('', '        if k == S_OLETTER or k == S_UPPER or s_para(k) or s_aterm(k)\n            return false', '        if k == S_OLETTER or k == S_UPPER\n            return false'),
 # UAX 14
 ('_line', '    if k == L_SA\n        let g = gc(c)', '    if false\n        let g = gc(c)'),
 ('_line', '    if k == L_CJ\n        return L_NS', '    if false\n        return L_NS'),
 ('_line', '    if z >= 0 and copy(it[z as usize].raw) == L_ZW\n        return 1', '    if false\n        return 1'),
 ('_line', '    if rp == L_ZWJ\n        return 0', '    if false\n        return 0'),
 ('_line', '    if b == L_GL and a != L_SP and a != L_BA and a != L_HY', '    if b == L_GL and a != L_SP'),
 ('_line', '    if qa == L_OP\n        return 0', '    if a == L_OP\n        return 0'),
 ('_line', '    if qa == L_QU and it[q as usize].gcat == GC_PI', '    if qa == L_QU'),
 ('_line', '    if a == L_SP and b == L_IS and b2 == L_NU\n        return 1', '    if false\n        return 1'),
 ('_line', '    if qa == L_B2 and b == L_B2', '    if a == L_B2 and b == L_B2'),
 ('_line', '    if b == L_QU and not it[pj as usize].east\n        return 0', '    if false\n        return 0'),
 ('_line', '    if (a == L_HY or it[pj as usize].cp == 8208) and b == L_AL', '    if a == L_HY and b == L_AL'),
 ('_line', '    if a2 == L_HL and (a == L_HY or (a == L_BA and not it[pj as usize].east)) and b != L_HL', '    if a2 == L_HL and a == L_HY and b != L_HL'),
 ('_line', '        if b != L_NU and (a == L_CL or a == L_CP)\n            r = pj2', '        if false\n            r = pj2'),
 ('_line', '    if (a == L_PO or a == L_PR) and b == L_OP and (b2 == L_NU or (b2 == L_IS and b3 == L_NU))', '    if (a == L_PO or a == L_PR) and b == L_OP and b2 == L_NU'),
 ('_line', '    if (a == L_JT or a == L_H3) and b == L_JT', '    if a == L_JT and b == L_JT'),
 ('_line', '    if ak_set(it, pj) and ak_set(it, bi) and b2 == L_VF', '    if ak_set(it, pj) and ak_set(it, bi)'),
 ('_line', '    if (a == L_AL or a == L_HL or a == L_NU) and b == L_OP and not it[i as usize].east', '    if (a == L_AL or a == L_HL or a == L_NU) and b == L_OP'),
 ('_line', '        if cnt % 2 == 1\n            return 0', '        if cnt > 0\n            return 0'),
 ('_line', '    if b == L_EM and (a == L_EB or (it[pj as usize].pict and it[pj as usize].gcat == GC_CN))', '    if b == L_EM and a == L_EB'),
 ('_line', '            if base >= 0\n                x.gone = true', '            if true\n                x.gone = true'),
 # UAX 9
 ('_bidi', 'const MAX_DEPTH: i64 = 125', 'const MAX_DEPTH: i64 = 124'),
 ('_bidi', '            if nl <= MAX_DEPTH and over_iso == 0 and over_emb == 0\n                let ov', '            if nl <= MAX_DEPTH and over_emb == 0\n                let ov'),
 ('_bidi', '            elif over_iso == 0\n                over_emb += 1', '            else\n                over_emb += 1'),
 ('_bidi', '                rtl = first_strong(&t, i + 1, match_pdi(&t, i)) == 1', '                rtl = first_strong(&t, i + 1, n) == 1'),
 ('_bidi', '            elif valid_iso > 0\n                over_emb = 0', '            elif valid_iso > 0\n                over_emb = over_emb'),
 ('_bidi', '            elif not top.iso and st.len() >= 2\n                st.pop()', '            elif st.len() >= 2\n                st.pop()'),
 ('_bidi', '        elif k == B_B\n            lv.push(para)', '        elif k == B_B\n            lv.push(top.level)'),
 ('_bidi', '    if not is_iso_init(copy(orig[last]))\n        var q = last + 1', '    if true\n        var q = last + 1'),
 ('_bidi', '        elif x == B_EN and last_strong == B_AL\n            ty[k] = B_AN', '        elif false\n            ty[k] = B_AN'),
 ('_bidi', '        elif x == B_CS and a == B_AN and b == B_AN\n            ty[k] = B_AN', '        elif false\n            ty[k] = B_AN'),
 ('_bidi', '            let adj = (k > 0 and ty[k - 1] == B_EN) or (e < m and ty[e] == B_EN)', '            let adj = k > 0 and ty[k - 1] == B_EN'),
 ('_bidi', '        elif x == B_EN and last_strong == B_L\n            ty[k] = B_L', '        elif false\n            ty[k] = B_L'),
 ('_bidi', '            let d = if lead == trail and lead >= 0 then (if lead == 0 then B_L else B_R) else e_dir', '            let d = e_dir'),
 ('_bidi', '            elif x == B_AN or x == B_EN\n                lv[at] = level + 2', '            elif x == B_AN\n                lv[at] = level + 2'),
 ('_bidi', '        elif trail and (k == B_WS or is_iso_init(k) or k == B_PDI)', '        elif trail and k == B_WS'),
 ('_bidi', '        elif is_removed(k)\n            trail = trail', '        elif false\n            trail = trail'),
 ('_bidi', '    while L >= lo_odd\n', '    while L > lo_odd\n'),
 ('_bidichar', '            set = if ctx != e then ctx else e', '            set = e'),
 ('_bidichar', 'fn canon_bracket(c: i64) -> i64 = if c == 9001 then 12296 else (if c == 9002 then 12297 else c)', 'fn canon_bracket(c: i64) -> i64 = c'),
 ('_bidichar', '                if so.len() >= 63\n', '                if so.len() >= 64\n'),
 ('_bidichar', '                        while so.len() as i64 > d\n', '                        while so.len() as i64 > d + 1\n'),
 ('_bidichar', '            if d == e\n                found_e = true', '            if false\n                found_e = true'),
 # width (derived checks in the first shard)
 ('', '    if e == 2 or e == 5\n        return 2', '    if e == 5\n        return 2'),
 ('', '        elif c == 65039 and pict\n            w = 2', '        elif false\n            w = 2'),
 ('', '    if c == 8203 or (c >= 4448 and c <= 4607) or (c >= 55216 and c <= 55295)', '    if c == 8203'),
]
def run(src, f):
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'utf8.rce'), 'w').write(UTF8)
        open(os.path.join(d, 'text.rce'), 'w').write(src + '\n' + open(f).read())
        return subprocess.run([RCE, 'test', d], capture_output=True).returncode != 0
def one(ab):
    i, (hint, a, b) = ab
    assert SRC.count(a) == 1, a
    src = SRC.replace(a, b)
    first = shards(hint) if hint else shards('.rce')
    rest = [f for f in SH if f not in first]
    for f in first + rest:
        if run(src, f):
            return i, a, True
    return i, a, False
killed = 0
with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
    for i, a, dead in ex.map(one, enumerate(M)):
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
