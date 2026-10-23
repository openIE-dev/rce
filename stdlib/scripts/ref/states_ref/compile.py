"""SCXML (ecmascript subset) -> lib/ui/states.rce chart text."""
import re, sys
import xml.etree.ElementTree as ET
SC = '{http://www.w3.org/2005/07/scxml}'
def tag(e): return e.tag.replace(SC, '')
STATES = ('state', 'parallel', 'final', 'history')
class Err(Exception): pass
def atom(x):
    x = x.strip()
    if re.fullmatch(r'-?\d+', x): return 'n:' + x
    if re.fullmatch(r'[A-Za-z_]\w*', x): return 'v:' + x
    raise Err('expr ' + x)
def expr(x):
    x = x.strip()
    m = re.fullmatch(r'(.+?)\s*\+\s*(.+)', x)
    if m: return 'add:' + atom(m.group(1)) + ':' + atom(m.group(2))
    return atom(x)
OPS = {'==': 'eq', '===': 'eq', '!=': 'ne', '!==': 'ne', '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge'}
def cond(x):
    if x is None: return '-'
    x = x.strip()
    if x == 'true': return 't'
    if x == 'false': return 'f'
    m = re.fullmatch(r"In\('([^']+)'\)", x)
    if m: return 'in:' + m.group(1)
    m = re.fullmatch(r"_event\.name\s*==\s*'([^']+)'", x)
    if m: return 'ev:' + m.group(1)
    m = re.fullmatch(r'(\w+)\s*(===|!==|==|!=|<=|>=|<|>)\s*(-?\w+)', x)
    if m: return 'cmp:%s:%s:%s' % (OPS[m.group(2)], atom(m.group(1)), atom(m.group(3)))
    m = re.fullmatch(r'[A-Za-z_]\w*', x)
    if m: return 'bound:' + x
    raise Err('cond ' + x)
def ms(d):
    d = d.strip().strip("'")
    m = re.fullmatch(r'(\d+(?:\.\d+)?)(ms|s)', d)
    if not m: raise Err('delay ' + d)
    return str(int(round(float(m.group(1)) * (1 if m.group(2) == 'ms' else 1000))))
class C:
    def __init__(s): s.out = []; s.machines = []
    def actions(s, e):
        for a in e:
            t = tag(a)
            if t == 'raise': s.out.append('raise ' + a.get('event'))
            elif t == 'log': s.out.append('log ' + (a.get('label') or '-').replace(' ', '_'))
            elif t == 'assign': s.out.append('assign %s %s' % (a.get('location').strip(), expr(a.get('expr'))))
            elif t == 'send':
                d = a.get('delay') or a.get('delayexpr')
                tg = a.get('target'); tg = 'self' if tg is None else ('parent' if tg == '#_parent' else 'bad')
                s.out.append('send %s %s %s' % (a.get('event'), ms(d) if d else '0', tg))
            elif t == 'if':
                s.out.append('if ' + cond(a.get('cond')))
                for b in a:
                    bt = tag(b)
                    if bt == 'elseif': s.out.append('elif ' + cond(b.get('cond')))
                    elif bt == 'else': s.out.append('else')
                    else: s.actions([b])
                s.out.append('fi')
            elif t in ('elseif', 'else'): pass
            else: raise Err('action ' + t)
    def machine(s, root):
        mi = len(s.machines); s.machines.append(root)
        lines = ['machine']
        rid = '_scxml'
        lines.append('state %s root - %s' % (rid, (root.get('initial') or '-').replace(' ', ',')))
        body = []; invokes = []
        for d in root.iter(SC + 'data'):
            pass
        dm = root.find(SC + 'datamodel')
        datas = []
        def walk(e, pid):
            for c in e:
                t = tag(c)
                if t == 'datamodel':
                    for d in c:
                        datas.append('data %s %s' % (d.get('id'), expr(d.get('expr')) if d.get('expr') else '-'))
                elif t in STATES:
                    sid = c.get('id') or ('_s%d_%d' % (mi, len(lines)))
                    kind = {'parallel': 'parallel', 'final': 'final'}.get(t, 'atomic')
                    if t == 'history': kind = 'deep' if c.get('type') == 'deep' else 'shallow'
                    ini = c.get('initial')
                    iel = c.find(SC + 'initial')
                    if iel is not None: ini = iel.find(SC + 'transition').get('target')
                    lines.append('state %s %s %s %s' % (sid, kind, pid, (ini or '-').replace(' ', ',')))
                    if iel is not None:
                        s.out = []; s.actions(iel.find(SC + 'transition'))
                        body.append(['initial ' + sid] + s.out + ['end'])
                    for x in c:
                        xt = tag(x)
                        if xt in ('onentry', 'onexit'):
                            s.out = []; s.actions(x); body.append([xt + ' ' + sid] + s.out + ['end'])
                        elif xt == 'transition':
                            s.out = []; s.actions(x)
                            ev = (x.get('event') or '').split()
                            tg = (x.get('target') or '').split()
                            body.append(['transition %s %s %s %s %s' % (sid, ','.join(ev) or '-', cond(x.get('cond')),
                                         'int' if x.get('type') == 'internal' else 'ext', ','.join(tg) or '-')] + s.out + ['end'])
                        elif xt == 'invoke':
                            sub = x.find(SC + 'content').find(SC + 'scxml')
                            invokes.append((sid, sub))
                        elif xt in ('donedata',): raise Err('donedata')
                    walk(c, sid)
                elif t in ('onentry', 'onexit', 'transition', 'initial', 'invoke', 'datamodel', 'script'):
                    if pid == rid and t in ('script',): raise Err('script')
        walk(root, rid)
        res = lines + datas
        for b in body: res += b
        subs = []
        for sid, sub in invokes:
            n, txt = s.machine_later(sub)
            res.append('invoke %s %d' % (sid, n))
        return res
    def machine_later(s, sub):
        s.pending.append(sub); return s.base + len(s.pending), None
def compile_doc(text):
    root = ET.fromstring(text)
    c = C(); out = []; queue = [root]; c.pending = []
    i = 0
    while i < len(queue):
        c.base = len(queue) - 1; c.pending = []
        out += c.machine(queue[i])
        queue += c.pending
        i += 1
    return '\n'.join(out)
if __name__ == '__main__':
    print(compile_doc(open(sys.argv[1]).read()))
