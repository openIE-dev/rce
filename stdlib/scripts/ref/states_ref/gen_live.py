"""Random statecharts as SCXML (ecmascript) for the SCION live reference."""
import random
def chart(R):
    ids = []; n = [0]
    def nid(p): n[0] += 1; i = '%s%d' % (p, n[0]); ids.append(i); return i
    finals_top = []
    def st(depth, parent_kind):
        r = R.random()
        if depth >= 3 or r < 0.35:
            k = 'final' if (parent_kind != 'parallel' and R.random() < 0.12) else 'state'
            return {'t': k, 'id': nid('f' if k == 'final' else 's'), 'kids': []}
        k = 'parallel' if r < 0.55 else 'state'
        node = {'t': k, 'id': nid('p' if k == 'parallel' else 'c'), 'kids': []}
        for _ in range(R.randint(2, 3)):
            node['kids'].append(st(depth + 1, k))
        if k == 'state' and R.random() < 0.3:
            node['kids'].append({'t': 'history', 'id': nid('h'), 'deep': R.random() < 0.5, 'kids': []})
        if k == 'state' and R.random() < 0.3:
            ch = [c for c in node['kids'] if c['t'] != 'history']
            node['initial'] = R.choice(ch)['id']
        return node
    top = [st(1, 'state') for _ in range(R.randint(1, 3))]
    top.append({'t': 'final', 'id': 'done', 'kids': []})
    allst = []
    def coll(x, anc):
        x['anc'] = anc; allst.append(x)
        for c in x['kids']: coll(c, anc + [x['id']])
    for x in top: coll(x, [])
    tg = [x['id'] for x in allst]
    def acts(raises=True):
        o = []
        for _ in range(R.randint(0, 2)):
            r = R.random()
            if r < 0.3:
                if raises: o.append('<raise event="%s"/>' % R.choice(['x', 'y']))
            elif r < 0.6: o.append('<assign location="Var1" expr="Var1 + 1"/>')
            elif r < 0.8: o.append('<if cond="Var1 &gt; 1">%s<else/><assign location="Var2" expr="Var2 + 1"/></if>' % ('<raise event="y"/>' if raises else '<log label="I" expr="0"/>'))
            else: o.append('<log label="L%d" expr="0"/>' % R.randint(0, 9))
        return ''.join(o)
    def xml(x):
        if x['t'] == 'history':
            d = [c for c in allst if x['id'] in []]
            sib = [c for c in allst if c['anc'] and c['anc'][-1] == x['anc'][-1] and c['t'] != 'history']
            return '<history id="%s" type="%s"><transition target="%s"/></history>' % (x['id'], 'deep' if x['deep'] else 'shallow', R.choice(sib)['id'])
        a = ' initial="%s"' % x['initial'] if 'initial' in x else ''
        s = '<%s id="%s"%s>' % (x['t'], x['id'], a)
        if x['t'] != 'final' or R.random() < 0.5:
            s += '<onentry>%s</onentry><onexit>%s</onexit>' % (acts(False), acts(False))
        if x['t'] != 'final':
            for _ in range(R.randint(0, 3)):
                r = R.random()
                if r < 0.15:
                    ev = ''; cnd = ' cond="Var1 &lt; 3"'; body = '<assign location="Var1" expr="Var1 + 1"/>'
                else:
                    ev = ' event="%s"' % R.choice(['a', 'b', 'c', 'x', 'y', 'done.state.*', 'a b', '*']); cnd = R.choice(['', '', ' cond="Var2 == 0"', " cond=\"In('%s')\"" % R.choice(tg)])
                    body = acts() if ev in (' event="a"', ' event="b"', ' event="c"', ' event="a b"') else '<assign location="Var2" expr="Var2 + 1"/>'
                t = '' if R.random() < 0.15 else ' target="%s"' % R.choice(tg)
                if ev == ' event="done.state.*"' or (ev == ' event="*"'): t = R.choice(['', ' target="done"'])
                ty = ' type="internal"' if R.random() < 0.2 else ''
                s += '<transition%s%s%s%s>%s</transition>' % (ev, cnd, t, ty, body)
        for c in x['kids']: s += xml(c)
        return s + '</%s>' % x['t']
    body = ''.join(xml(x) for x in top)
    doc = ('<scxml xmlns="http://www.w3.org/2005/07/scxml" version="1.0" datamodel="ecmascript">'
           '<datamodel><data id="Var1" expr="0"/><data id="Var2" expr="0"/></datamodel>%s</scxml>' % body)
    evs = [R.choice(['a', 'b', 'c', 'x']) for _ in range(R.randint(0, 6))]
    return doc, evs
