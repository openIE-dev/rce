# Usage: python3 scripts/gen/fmt_gen.py, then replace the lines after
# "# ---- generated" in tests/fmt/fmt_checks.rce.
# CPython is the oracle (seed 20261015): int formatting in radixes 2..36
# against a reference conversion, str.format for templates and the
# ',' and '0N' format specs, and the re module for regex (re.ASCII; the
# library's "$" is translated to "\Z", since Python's "$" also matches
# before a final newline). Derived vectors, not official ones.
import random, re
r = random.Random(20261015)
D = '0123456789abcdefghijklmnopqrstuvwxyz'
def to(v, b):
    if v == 0: return '0'
    s = ''; x = abs(v)
    while x: s = D[x % b] + s; x //= b
    return ('-' if v < 0 else '') + s
def q(s):
    o = ''
    for ch in s:
        if ch == '\\': o += '\\\\'
        elif ch == '"': o += '\\"'
        elif ch == '\n': o += '\\n'
        elif ch == '\t': o += '\\t'
        elif ord(ch) < 32: o += '\\u{%x}' % ord(ch)
        else: o += ch
    return '"' + o + '"'
MIN, MAX = -2**63, 2**63 - 1
vals = [0, 1, -1, MAX, MIN, MIN + 1, 35, -36, 255, 2**32, -2**31]
for _ in range(25): vals.append(r.randint(MIN, MAX))
for _ in range(10): vals.append(r.randint(-10**6, 10**6))
for i, v in enumerate(vals):
    b = [10, 16, 2, 36, 8, 7][i % 6] if i > 10 else 10
    s = to(v, b)
    assert int(s, b) == v
    print('check int_ok(%d, %d, %s)' % (v, b, q(s)) if v != MIN else 'check int_ok(0 - 9223372036854775807 - 1, %d, %s)' % (b, q(s)))
for v in [0, 7, -42, 1234567, -1234567, 999, 1000, MAX, -10**9, 12]:
    vs = '%d' % v if v != MIN else '0 - 9223372036854775807 - 1'
    print('check group_dec(%s, &c) == %s' % (vs, q(format(v, ','))))
for v, w in [(-42, 6), (42, 6), (0, 3), (123456, 3), (-5, 2), (-5, 1), (7, 0)]:
    print('check zero_pad(%d, %d) == %s' % (v, w, q(format(v, '0%dd' % w))))
bad = ['', '-', '+', ' 1', '1 ', '0x10', '1_000', '9223372036854775808', '-9223372036854775809', '99999999999999999999', '12a', '--1', '+-1', 'é']
for s in bad: print('check int_bad(%s, 10)' % q(s))
print('check int_bad("2", 2) and int_bad("z", 35) and int_bad("8000000000000000", 16) and not int_bad("7fffffffffffffff", 16) and not int_bad("-8000000000000000", 16)')
for s, b, v in [('+17', 10, 17), ('-0', 10, 0), ('007', 10, 7), ('Ff', 16, 255), ('zz', 36, 1295), ('-9223372036854775808', 10, MIN), ('9223372036854775807', 10, MAX)]:
    vs = '%d' % v if v != MIN else '0 - 9223372036854775807 - 1'
    print('check parse_is(%s, %d, %s)' % (q(s), b, vs))
T = [('{} + {} = {}', ['1', '2', '3']), ('{1}{0}{1}', ['a', 'b']), ('{{}} {}', ['x']), ('{{{0}}}', ['é']), ('', []), ('plain', ['unused']), ('{}{}', ['', 'z']), ('{10}', [str(i) for i in range(11)])]
for t, a in T: print('check format(&%s, &%s) catch "ERR") == %s' % (q(t), '[' + ', '.join(q(x) for x in a) + ']', q(t.format(*a))) if False else 'check fmt_ok(%s, [%s], %s)' % (q(t), ', '.join(q(x) for x in a), q(t.format(*a))))
for t, a in [('{', []), ('}', []), ('{}', []), ('{2}', ['a', 'b']), ('{x}', ['a']), ('a}b', [])]:
    try: t.format(*a); raise SystemExit('expected failure ' + t)
    except (IndexError, ValueError, KeyError): pass
    print('check fmt_bad(%s, [%s])' % (q(t), ', '.join(q(x) for x in a)))
P = ['abc', 'a.c', 'a*', 'a+b*', 'colou?r', '(ab|cd)+', '^ab', 'b$', '[a-c]+x', '[^a-c]', '\\d{3}-\\d{4}', '\\w+@\\w+\\.com', 'a{2,3}', 'a{2,}', 'x(?:yz)*', '(a|b)*abb', '\\s+', '[\\-.]', '\\.\\*', 'é.', '.', 'a|', '(|a)b', '[]a]', '[a-]', '\\D\\W\\S', 'h(e|a)llo|world', '(a*)*b', 'a{0}b', '[0-9a-fA-F]{2}', '\\w+', '\\s', '...', '[^x]{2}']
S = ['', 'abc', 'ac', 'aac', 'abbb', 'color', 'colour', 'abcdab', 'ab', 'xb', 'ccx', 'abx', 'd', '555-1234', 'me@host.com', 'aa', 'aaaa', 'xyzyz', 'babb', 'abb', ' \t', '-', '.*', 'éa', 'é', '\n', 'a', 'b', ']', 'q-', '!x ', 'hello', 'hallo world', 'aaab', 'Fe', 'a_b', '\r', '\x0b', '€', '日本', 'a€']
n = 0
for p in P:
    py = re.compile(p.replace('$', '\\Z'), re.ASCII)
    for s in S:
        f = py.fullmatch(s) is not None; g = py.search(s) is not None
        print('check re_ok(%s, %s, %s, %s)' % (q(p), q(s), 'true' if f else 'false', 'true' if g else 'false')); n += 1
for p in ['(', ')', 'a)', '*a', 'a**' if False else '+', '[a', '[z-a]', 'a{2,1}', 'a{', 'a{1001}', '\\q', '(?=a)', '[b-a]', '[\\d-z]' if False else '?', '[\\d-z]', '{1}']:
    print('check re_bad(%s)' % q(p))
