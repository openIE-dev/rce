# Mutation test for lib/ui/input.rce: each mutant must fail at least one
# shard of tests/ui/input_checks*.rce (the official and derived shard runs
# first). Usage: RCE=path/to/rce python3 scripts/mut/input.py
import glob, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
SRC = open(os.path.join(root, 'lib/ui/input.rce')).read()
UTF8 = open(os.path.join(root, 'lib/text/utf8.rce')).read()
SH = sorted(glob.glob(os.path.join(root, 'tests/ui/input_checks*.rce')))
M = [
 # parse_event
 ('            return if more then wait() else gk(K_ESC)', '            return gk(K_ESC)'),
 ('            if c2 >= 80 and c2 <= 83\n                return gkm(K_FN, c2 - 79, 0)', '            if c2 >= 80 and c2 <= 82\n                return gkm(K_FN, c2 - 79, 0)'),
 ('        if c1 == 27\n            return gk(K_ESC)', '        if false\n            return gk(K_ESC)'),
 ('            r.ev.key.mods = r.ev.key.mods | M_ALT', '            r.ev.key.mods = M_ALT'),
 ('    if c == 10 and not raw\n', '    if c == 10\n'),
 ('    if c >= 1 and c <= 26\n', '    if c >= 1 and c <= 25\n'),
 ('        return gkm(K_CHAR, c - 28 + 52, M_CONTROL)', '        return gkm(K_CHAR, c - 28 + 52, 0)'),
 ('        return gkm(K_CHAR, 32, M_CONTROL)', '        return gkm(K_CHAR, 0, M_CONTROL)'),
 ('    return gkm(K_CHAR, cp, if is_upper(cp) then M_SHIFT else 0)', '    return gkm(K_CHAR, cp, 0)'),
 # utf8
 ('    elif c >= 240 and c <= 247\n        need = 4', '    elif c >= 240 and c <= 247\n        need = 3'),
 ('            if b[i] & 192 != 128\n                return bad()', '            if false\n                return bad()'),
 ('    if b.len() < need\n        return wait()', '    if b.len() <= need\n        return wait()'),
 # csi
 ('        if d >= 65 and d <= 69\n', '        if d >= 65 and d <= 68\n'),
 ('        return gkm(K_BACKTAB, 0, M_SHIFT)', '        return gkm(K_BACKTAB, 0, 0)'),
 ('        if l == 99\n            return got(blank(E_DA1))', '        if false\n            return got(blank(E_DA1))'),
 ('        if l < 64 or l > 126\n', '        if l < 64\n'),
 ('        if l == 82\n            return parse_cursor_position(b)', '        if false\n            return parse_cursor_position(b)'),
 ('    if c == 83\n        return gkm(K_FN, 4, 0)', '    if c == 83\n        return gkm(K_FN, 3, 0)'),
 # numbers
 ('    if t.len() > 0 and t[0] == 43\n        i = 1', '    if false\n        i = 1'),
 ('        if v > max\n            return -1', '        if false\n            return -1'),
 ('fn dec16(v: i64) -> i64 = if v == 0 then 65535 else v - 1', 'fn dec16(v: i64) -> i64 = v - 1'),
 ('    return [m, if k < 0 then 1 else k]', '    return [m, 1]'),
 ('    if m & 128 != 0\n        s = s | S_NUM_LOCK', '    if m & 128 != 0\n        s = s | S_CAPS_LOCK'),
 ('pub fn act_of(k: i64) -> i64 = if k == 2 then REPEAT else (if k == 3 then RELEASE else PRESS)', 'pub fn act_of(k: i64) -> i64 = if k == 2 then REPEAT else (if k >= 3 then RELEASE else PRESS)'),
 ('    return (m & 1) | (if m & 2 != 0 then M_ALT else 0) | (if m & 4 != 0 then M_CONTROL else 0) | (m & 56)', '    return (m & 1) | (if m & 2 != 0 then M_ALT else 0) | (if m & 4 != 0 then M_CONTROL else 0) | (m & 24)'),
 # kb flags, cursor
 ('    if b.len() < 5\n        return wait()\n    var e = blank(E_KB_FLAGS)', '    if b.len() < 4\n        return wait()\n    var e = blank(E_KB_FLAGS)'),
 ('    e.x = b[3] & 15', '    e.x = b[3] & 31'),
 ('    e.x = dec16(x)\n    e.y = dec16(y)', '    e.x = dec16(y)\n    e.y = dec16(x)'),
 # modifier key code
 ('    elif n > 3\n        let d = copy(b[n - 2])', '    elif n > 4\n        let d = copy(b[n - 2])'),
 # csi u
 ('            return [K_ENTER, 0, S_KEYPAD]', '            return [K_ENTER, 0, 0]'),
 ('    if cp >= 57376 and cp <= 57398\n', '    if cp >= 57376 and cp <= 57397\n'),
 ('        elif cp == 10 and not raw\n            code = K_ENTER', '        elif cp == 10\n            code = K_ENTER'),
 ('            code = if mods & M_SHIFT != 0 then K_BACKTAB else K_TAB', '            code = K_TAB'),
 ('        if not utf8.is_scalar(cp)\n            return bad()', '        if false\n            return bad()'),
 ('        if arg < 12\n            mods = mods |', '        if arg < 6\n            mods = mods |'),
 ('            mods = mods & (63 - M_SHIFT)', '            mods = mods'),
 ('        if sh >= 0 and utf8.is_scalar(sh)', '        if sh >= 0'),
 ('    return got(key_ev(code, arg, mods, act, kst | st))', '    return got(key_ev(code, arg, mods, act, st))'),
 # special key code
 ('    if first == 1 or first == 7\n', '    if first == 1\n'),
 ('    elif first >= 23 and first <= 26\n        arg = first - 12', '    elif first >= 23 and first <= 26\n        arg = first - 11'),
 ('    elif first >= 31 and first <= 34\n', '    elif first >= 31 and first <= 33\n'),
 # mouse
 ('    let bn = (cb & 3) | ((cb & 192) >> 4)', '    let bn = cb & 3'),
 ('        kind = if drag then MS_MOVED else MS_UP'.replace('kind', 'act'), '        act = MS_UP'),
 ('    elif bn == 6 and not drag\n', '    elif bn == 6\n'),
 ('    if cb & 16 != 0\n        mods = mods | M_CONTROL', '    if cb & 16 != 0\n        mods = mods | M_ALT'),
 ('    if cb < 32\n        return bad()\n    let k = parse_cb(cb - 32)', '    if cb < 0\n        return bad()\n    let k = parse_cb(cb)'),
 ('    if b.len() < 6\n        return wait()', '    if b.len() < 5\n        return wait()'),
 ('    let x = if b[4] > 32 then b[4] - 32 else 0', '    let x = b[4] - 32'),
 ('    if l == 109 and k[0] == MS_DOWN\n', '    if k[0] == MS_DOWN\n'),
 ('    if not utf8_ok(b, 3)\n        return bad()', '    if false\n        return bad()'),
 # paste and stream
 ('    e.paste = utf8.utf8_decode_lossy(&s)', '    e.paste = s'),
 ('        let more = i + 1 < read.len() or more_after', '        let more = i + 1 < read.len()'),
 ('        elif r.st == BAD\n            buf = []', '        elif false\n            buf = []'),
 # DOM
 ('    if cs[i] == "0"\n        return false', '    if false\n        return false'),
 ('            loc = if k.arg < 6 then LOC_LEFT else LOC_RIGHT', '            loc = LOC_LEFT'),
 ('        if c < 32 or (c >= 127 and c < 160)', '        if c < 32'),
 ('meta: m & (M_SUPER | M_META) != 0', 'meta: m & M_META != 0'),
 ('caps: k.locks & S_CAPS_LOCK != 0', 'caps: k.locks & S_NUM_LOCK != 0'),
 ('        st.buttons = st.buttons & (63 - buttons_bit(btn))', '        st.buttons = 0'),
 ('    elif m.act == MS_DRAG\n        st.buttons', '    elif false\n        st.buttons'),
 ('    if button == PB_AUX\n        return 4', '    if button == PB_AUX\n        return 2'),
 ('        dy = -1', '        dy = 1'),
 ('        return if e == IME_START then 1 else -1', '        return 1'),
 ('    return copy(n[i as usize])\npub fn mod_name', '    return copy(n[(i + 1) % 13 as usize])\npub fn mod_name'),
]
def run(src, f):
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'utf8.rce'), 'w').write(UTF8)
        open(os.path.join(d, 'input.rce'), 'w').write(src + '\n' + open(f).read())
        return subprocess.run([RCE, 'test', d], capture_output=True).returncode != 0
def one(ab):
    i, (a, b) = ab
    assert SRC.count(a) == 1, a
    src = SRC.replace(a, b)
    for f in SH:
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
