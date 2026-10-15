# rce text literal for a Python str (shared by the data.* generators).
def lit(s):
    o = '"'
    for c in s:
        x = ord(c)
        if c == '"': o += '\\"'
        elif c == '\\': o += '\\\\'
        elif c == '\n': o += '\\n'
        elif c == '\t': o += '\\t'
        elif c == '\r': o += '\\r'
        elif x < 32 or 0x7f <= x <= 0x9f or x in (0x2028, 0x2029) or 0xfff0 <= x <= 0xffff: o += '\\u{%x}' % x
        else: o += c
    return o + '"'
def lits(xs):
    return '[' + ', '.join(lit(x) for x in xs) + ']'
