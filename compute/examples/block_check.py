#!/usr/bin/env python3
# An independent evaluation of block.rce's step on Tokens(ids: [0, 1, 2]):
# binary64 arithmetic in the order of spec 6.13 GT-4, exp from Python's
# decimal (correctly rounded), to cross-check the reference bit for bit.
import math, re
from decimal import Decimal, localcontext
def E(x):
    with localcontext() as c:
        c.prec = 60
        return float(Decimal(x).exp())
src = open(__import__('os').path.join(__import__('os').path.dirname(__file__), 'block.rce')).read()
def mat(name):
    m = re.search(r'\n    %s: [^=]*= (\[.*\])\n' % name, src)
    return eval(m.group(1))
emb, wq, wk, wv, wo, g1, g2, w1, w2 = [mat(n) for n in ['emb','wq','wk','wv','wo','g1','g2','w1','w2']]
def matmul(a, b):
    out = []
    for r in a:
        row = []
        for j in range(len(b[0])):
            acc = 0.0
            for k in range(len(r)):
                acc = acc + r[k] * b[k][j]
            row.append(acc)
        out.append(row)
    return out
def rmsnorm(x, w, eps):
    out = []
    for r in x:
        ss = 0.0
        for v in r: ss = ss + v * v
        inv = 1.0 / math.sqrt(ss / float(len(r)) + eps)
        out.append([(v * inv) * w[i] for i, v in enumerate(r)])
    return out
def heads(m):  # [3,4] -> [2][3][2]
    return [[[m[s][h*2+t] for t in range(2)] for s in range(3)] for h in range(2)]
def unheads(a):  # [2][3][2] -> [3][4]
    return [[a[h][s][t] for h in range(2) for t in range(2)] for s in range(3)]
def attn(q, k, v):
    scale = 1.0 / math.sqrt(2.0)
    out = []
    for h in range(2):
        rows = []
        for i in range(3):
            a = []
            for j in range(i + 1):
                acc = 0.0
                for t in range(2): acc = acc + q[h][i][t] * k[h][j][t]
                a.append(acc)
            m = -math.inf
            for x in a:
                if x * scale > m: m = x * scale
            p = []; s = 0.0
            for x in a:
                e = E(x * scale - m); p.append(e); s = s + e
            inv = 1.0 / s
            p = [x * inv for x in p]
            row = []
            for t in range(2):
                acc = 0.0
                for j in range(i + 1): acc = acc + p[j] * v[h][j][t]
                row.append(acc)
            rows.append(row)
        out.append(rows)
    return out
def silu(x):
    if x >= 0: return x / (1.0 + E(-x))
    e = E(x); return (x * e) / (1.0 + e)
add = lambda a, b: [[x + y for x, y in zip(r, s)] for r, s in zip(a, b)]
x = [emb[i] for i in [0, 1, 2]]
n = rmsnorm(x, g1, 0.00001)
q, k, v = heads(matmul(n, wq)), heads(matmul(n, wk)), heads(matmul(n, wv))
a = unheads(attn(q, k, v))
h = add(x, matmul(a, wo))
f = [[silu(z) for z in r] for r in matmul(rmsnorm(h, g2, 0.00001), w1)]
y = add(h, matmul(f, w2))
print(repr(y))
