# JSON with number lexemes kept, and the canonical text data.events
# json_canon writes (members sorted by UTF-8 bytes of the name, strings
# as enc.jcs_string writes them, numbers as their source lexeme).
import json


class Num(str):
    pass


def loads(s):
    return json.loads(s, parse_int=Num, parse_float=Num, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def jstr(s):
    o = '"'
    for ch in s:
        x = ord(ch)
        if ch == '"': o += '\\"'
        elif ch == '\\': o += '\\\\'
        elif ch == '\b': o += '\\b'
        elif ch == '\f': o += '\\f'
        elif ch == '\n': o += '\\n'
        elif ch == '\r': o += '\\r'
        elif ch == '\t': o += '\\t'
        elif x < 0x20: o += '\\u%04x' % x
        else: o += ch
    return o + '"'


def canon(v):
    if v is None: return 'null'
    if v is True: return 'true'
    if v is False: return 'false'
    if isinstance(v, Num): return str(v)
    if isinstance(v, int): return str(v)
    if isinstance(v, str): return jstr(v)
    if isinstance(v, list): return '[' + ','.join(canon(x) for x in v) + ']'
    if isinstance(v, dict):
        ks = sorted(v.keys(), key=lambda k: k.encode('utf-8'))
        return '{' + ','.join(jstr(k) + ':' + canon(v[k]) for k in ks) + '}'
    if isinstance(v, float): raise TypeError('float')
    raise TypeError(type(v))


def change(src, op, ns, key=None, pre=None, post=None, delta=None, pos='', tx=''):
    return canon({'after': post, 'before': pre, 'delta': delta, 'key': key, 'ns': ns, 'op': op, 'pos': pos, 'src': src, 'tx': tx})
