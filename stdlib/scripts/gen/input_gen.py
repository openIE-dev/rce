#!/usr/bin/env python3
"""Checks for lib/ui/input.rce. Official: vectors/ui/crossterm_official.tsv,
every assertion of the 30 unit tests in crossterm 0.29.0's
src/event/sys/unix/parse.rs (vectors/ui/crossterm_parse.rs), recorded by
running those tests in scripts/ref/input_ref (each assert_eq! also checks
the value it records). Live: vectors/ui/crossterm_live.tsv from the same
harness's main (seeded single parse_event calls and Parser::advance
streams). Derived: scripts/gen/input_derived.rce.
Usage: python3 scripts/gen/input_gen.py"""
import ast, os, re, glob
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
OUT = os.path.join(ROOT, 'tests/ui')
FN = {'parse_event': 'pe', 'parse_csi': 'pc', 'parse_csi_cursor_position': 'parse_cursor_position',
      'parse_csi_modifier_key_code': 'parse_modifier_key_code', 'parse_csi_normal_mouse': 'parse_normal_mouse',
      'parse_csi_rxvt_mouse': 'parse_rxvt_mouse', 'parse_csi_sgr_mouse': 'parse_sgr_mouse',
      'parse_csi_special_key_code': 'parse_special_key_code', 'parse_csi_u_encoded_key_code': 'pu',
      'parse_utf8_char': 'uc'}
H = '''fn pe(b: List[i64], more: bool) -> text
    let p = parse_event(&b, more, false)
    return parsed_text(&p)
fn per(b: List[i64], more: bool, raw: bool) -> text
    let p = parse_event(&b, more, raw)
    return parsed_text(&p)
fn pc(b: List[i64]) -> text
    let p = parse_csi(&b, false)
    return parsed_text(&p)
fn pu(b: List[i64]) -> text
    let p = parse_csi_u(&b, false)
    return parsed_text(&p)
fn uc(b: List[i64]) -> text
    let p = parse_utf8_char(&b)
    if p.st == GOT
        return "char " + p.ev.x.tostr()
    return parsed_text(&p)
fn sub_text(f: text, b: List[i64]) -> text
    var p = parse_utf8_char(&b)
    if f == "parse_cursor_position"
        p = parse_cursor_position(&b)
    elif f == "parse_modifier_key_code"
        p = parse_modifier_key_code(&b)
    elif f == "parse_normal_mouse"
        p = parse_normal_mouse(&b)
    elif f == "parse_rxvt_mouse"
        p = parse_rxvt_mouse(&b)
    elif f == "parse_sgr_mouse"
        p = parse_sgr_mouse(&b)
    elif f == "parse_special_key_code"
        p = parse_special_key_code(&b)
    return parsed_text(&p)
fn stream(b: List[i64], raw: bool) -> text
    let d = decode(&b, raw)
    var o = ""
    var i: usize = 0
    while i < d.events.len()
        if i > 0
            o = o + ";"
        o = o + ev_text(&d.events[i])
        i += 1
    o = o + "|"
    for x in d.pending
        o = o + hx(x)
    return o
fn hx(x: i64) -> text
    let d = "0123456789abcdef"
    let cs = d.chars()
    return copy(cs[(x >> 4) as usize]) + copy(cs[(x & 15) as usize])
'''
def rbytes(lit):
    if lit.startswith('&['):
        return [int(x, 16) for x in lit[2:-1].split(', ')]
    if lit.endswith('.as_bytes()'):
        return list(ast.literal_eval(lit[:-len('.as_bytes()')]).encode())
    return list(ast.literal_eval(lit))
def q(t): return '"' + t.replace('\\', '\\\\').replace('"', '\\"') + '"'
def L(b): return '[' + ', '.join(map(str, b)) + ']'
def main():
    for f in glob.glob(os.path.join(OUT, 'input_checks*.rce')): os.remove(f)
    sh = {}
    def emit(k, l): sh.setdefault(k, []).append(l)
    n_off = 0
    for line in open(os.path.join(ROOT, 'vectors/ui/crossterm_official.tsv')).read().split('\n'):
        if not line: continue
        expr, want = line.split('\t')
        m = re.match(r'(\w+)\((.*?)(?:, (true|false))?\)(\.unwrap\(\))?$', expr)
        f, lit, more = m.group(1), m.group(2), m.group(3)
        b = rbytes(lit)
        g = FN[f]
        if g == 'pe':
            emit('', 'check pe(%s, %s) == %s' % (L(b), more, q(want)))
        elif g in ('pc', 'pu', 'uc'):
            emit('', 'check %s(%s) == %s' % (g, L(b), q(want)))
        else:
            emit('', 'check sub_text("%s", %s) == %s' % (g, L(b), q(want)))
        n_off += 1
    # UI Events and Pointer Events tables (official)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from input_data import spec_tables
    c, k = spec_tables()
    for _, v, req in c:
        emit('', 'check is_code("%s") and code_required("%s") == %s' % (v, v, 'true' if req == 'Yes' else 'false'))
    for _, v, req in k:
        emit('', 'check is_named_key("%s") and is_key_value("%s") and key_required("%s") == %s' % (v, v, v, 'true' if req == 'Yes' else 'false'))
    # implicit function keys named in the spec text
    emit('', 'check is_named_key("F24") and is_named_key("Soft8") and is_code("F13") and is_code("F14") and is_code("F15")')
    n_off += len(c) + len(k) + 1
    # Pointer Events 3 button and buttons tables
    for b, bits in [(-1, 0), (0, 1), (1, 4), (2, 2), (3, 8), (4, 16), (5, 32)]:
        emit('', 'check buttons_bit(%d) == %d' % (b, bits)); n_off += 1
    n_live = 0
    for line in open(os.path.join(ROOT, 'vectors/ui/crossterm_live.tsv')).read().split('\n'):
        if not line: continue
        f = line.split('\t')
        b = list(bytes.fromhex(f[1]))
        if f[0] == 'e':
            l = 'check per(%s, %s, %s) == %s' % (L(b), 'true' if f[2] == '1' else 'false', 'true' if f[3] == '1' else 'false', q(f[4]))
        else:
            l = 'check stream(%s, %s) == %s' % (L(b), 'true' if f[2] == '1' else 'false', q(f[3] + '|' + f[4]))
        emit('live%d' % (n_live // 2500), l); n_live += 1
    der = open(os.path.join(ROOT, 'scripts/gen/input_derived.rce')).read()
    for k, ls in sh.items():
        name = 'input_checks.rce' if k == '' else 'input_checks_%s.rce' % k
        tail = '\n# ---- derived (scripts/gen/input_derived.rce)\n' + der if k == '' else ''
        open(os.path.join(OUT, name), 'w').write('# generated by scripts/gen/input_gen.py; do not edit\n' + H + '\n'.join(ls) + '\n' + tail)
    print('official', n_off, 'live', n_live)
main()
