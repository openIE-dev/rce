# Law-derived reference for lib/ui/view.rce's child-list diff.
# Child tokens (as in vue_ref.js): "N" keyed <span>, "t#N" keyed <t>,
# "t:x" unkeyed <t> labelled x, "~x" text node, "!" comment.
import re
def parse(t):
    t = str(t)
    if t == '!': return ('!', None, '!')
    if t[0] == '~': return ('', None, t[1:])
    if ':' in t: g, l = t.split(':', 1); return (g, None, l)
    if '#' in t: g, k = t.split('#'); return (g, k, k)
    return ('span', t, t)
def tag(x): return parse(x)[0]
def key(x): return parse(x)[1]
def same(a, b): return tag(a) == tag(b) and key(a) == key(b)
def label(x): return parse(x)[2]
def lis(a):
    tails, prev = [], [-1]*len(a)
    for i, x in enumerate(a):
        lo, hi = 0, len(tails)
        while lo < hi:
            m = (lo+hi)//2
            if a[tails[m]] < x: lo = m+1
            else: hi = m
        if lo > 0: prev[i] = tails[lo-1]
        if lo == len(tails): tails.append(i)
        else: tails[lo] = i
    out, k = [], tails[-1] if tails else -1
    while k >= 0: out.append(k); k = prev[k]
    return out[::-1]
def match(old, new):
    """m[j] = old index matched to new j, or -1. Prefix, suffix, then
    keyed by key, unkeyed to the first free same-tag unkeyed new node."""
    m = [-1]*len(new)
    i = 0
    while i < len(old) and i < len(new) and same(old[i], new[i]): m[i] = i; i += 1
    e1, e2 = len(old)-1, len(new)-1
    while e1 >= i and e2 >= i and same(old[e1], new[e2]): m[e2] = e1; e1 -= 1; e2 -= 1
    kpos = {key(new[j]): j for j in range(i, e2+1) if key(new[j]) is not None}
    for o in range(i, e1+1):
        if key(old[o]) is not None:
            j = kpos.get(key(old[o]))
            if j is not None and tag(new[j]) == tag(old[o]): m[j] = o
        else:
            for j in range(i, e2+1):
                if m[j] < 0 and key(new[j]) is None and tag(new[j]) == tag(old[o]):
                    m[j] = o; break
    return m, i, e1, e2
def diff(old, new):
    """ops: ('r', o) | ('c', j, anchor_j) | ('m', j, anchor_j); anchor -1 = end"""
    m, i, e1, e2 = match(old, new)
    anc = e2+1 if e2+1 < len(new) else -1
    if i > e1: return [('c', j, anc) for j in range(i, e2+1)]
    used = set(x for x in m if x >= 0)
    ops = [('r', o) for o in range(len(old)) if o not in used]
    cj = [j for j in range(len(new)) if m[j] >= 0]
    keep = set(cj[t] for t in lis([m[j] for j in cj]))
    for j in range(len(new)-1, -1, -1):
        a = j+1 if j+1 < len(new) else -1
        if m[j] < 0: ops.append(('c', j, a))
        elif j not in keep: ops.append(('m', j, a))
    return ops
def fmt(old, new, ops):
    m = match(old, new)[0]
    def lab(j): return '-' if j < 0 else (label(old[m[j]]) if m[j] >= 0 else label(new[j]))
    out = []
    for o in ops:
        if o[0] == 'r': out.append('r ' + label(old[o[1]]))
        else: out.append(o[0] + ' ' + lab(o[1]) + ' ' + lab(o[2]))
    return ';'.join(out)
def apply(old, new, ops):
    """replay ops on the old list; returns the final list of (tag, key) and
    checks the result equals new."""
    m = match(old, new)[0]
    live = [('o', i) for i in range(len(old))]
    def ident(j): return ('o', m[j]) if m[j] >= 0 else ('n', j)
    for o in ops:
        if o[0] == 'r': live.remove(('o', o[1])); continue
        x = ident(o[1])
        if o[0] == 'm': live.remove(x)
        else: assert x not in live
        if o[2] < 0: live.append(x)
        else: live.insert(live.index(ident(o[2])), x)
    return live == [ident(j) for j in range(len(new))]
