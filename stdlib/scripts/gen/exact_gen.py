# Regenerates tests/core/exact_checks.rce vector sections: unzip
# https://speleotrove.com/decimal/dectest.zip (version 2.62) into /tmp/dectest,
# run `python3 exact_gen.py 50 80 6`, and splice /tmp/gen/*.rce after the
# hand-written head of the checks file.
import re, random, sys, json, fractions, decimal, math
from decimal import Decimal
D='/tmp/dectest/'
MODES={'half_even':'0','half_up':'1','half_down':'2','down':'3','up':'4','floor':'5','ceiling':'6'}
def toks(line):
    return re.findall(r"'[^']*'|\"[^\"]*\"|\S+", line)
def num(t):
    t=t.strip("'\"")
    if re.search(r'(?i)nan|inf|#',t): return None
    try: d=Decimal(t)
    except Exception: return None
    s,dg,e=d.as_tuple()
    c=int(''.join(map(str,dg)) or '0')
    if s and c==0: return None  # negative zero: not represented
    if len(str(c))>60 or abs(e)>300: return None
    return (-c if s else c, e)
def dv(v): return 'Dec(coef: bp("%d"), exp: %d)'%v
picked=[];checks=[]
def scan(f, op, fn, maxn, allowed):
    rounding='half_up'; n=0; ok=[]
    for line in open(D+f):
        line=line.split('--')[0].strip()
        if not line: continue
        m=re.match(r'(\w+):\s*(\S+)',line)
        if m and m.group(1).lower()=='rounding': rounding=m.group(2).lower(); continue
        if m: continue
        t=toks(line)
        if len(t)<5 or t[1].lower()!=op or '->' not in t: continue
        a=t.index('->'); args=t[2:a]; res=t[a+1]; cond=set(x.lower() for x in t[a+2:])
        if not cond<=allowed or rounding not in MODES: continue
        if res.startswith('-0') or res.startswith("'-0"): continue
        vs=[num(x) for x in args]; r=num(res)
        if None in vs or r is None or len(vs)!=2: continue
        ok.append((line,rounding,vs,r))
    step=max(1,len(ok)//maxn)
    for line,rounding,vs,r in ok[::step][:maxn]:
        picked.append(f+': '+line)
        if op=='quantize':
            checks.append('check dec_quantize(%s, %d, %s) == %s'%(dv(vs[0]),vs[1][1],MODES[rounding],dv(r)))
        else:
            checks.append('check %s(%s, %s) == %s'%(fn,dv(vs[0]),dv(vs[1]),dv(r)))
    return len(ok)
na=scan('add.decTest','add','dec_add',int(sys.argv[1]),set())
nm=scan('multiply.decTest','multiply','dec_mul',int(sys.argv[1]),set())
nq=scan('quantize.decTest','quantize',None,int(sys.argv[2]),{'inexact','rounded'})
nb=scan('subtract.decTest','subtract','dec_sub',int(sys.argv[1]),set())
print(na,nm,nq,nb,len(checks),file=sys.stderr)
open('/tmp/gen/dectest_selected.txt','w').write('\n'.join(picked)+'\n')
open('/tmp/gen/dectest_checks.rce','w').write('\n'.join(checks)+'\n')
# CPython cross-check vectors for Big and Rat (seeded)
rnd=random.Random(20261015); out=[]
def B(x): return 'bp("%d")'%x
for i in range(int(sys.argv[3])):
    a=rnd.randrange(-10**rnd.randint(1,60),10**rnd.randint(1,60)); b=rnd.randrange(-10**rnd.randint(1,40),10**rnd.randint(1,40)) or 7
    q=abs(a)//abs(b)*(1 if (a<0)==(b<0) else -1)
    out.append('check big_mul(%s, %s) == %s'%(B(a),B(b),B(a*b)))
    out.append('check big_div(%s, %s) == %s and big_mod(%s, %s) == %s'%(B(a),B(b),B(a//b),B(a),B(b),B(a%b)))
    out.append('check big_quot(%s, %s) == %s'%(B(a),B(b),B(q)))
    out.append('check big_gcd(%s, %s) == %s'%(B(a),B(b),B(math.gcd(a,b))))
    out.append('check big_isqrt(%s) == %s'%(B(abs(a)),B(math.isqrt(abs(a)))))
    x=fractions.Fraction(a,b); y=fractions.Fraction(rnd.randint(-10**9,10**9) or 1, rnd.randint(1,10**9))
    R=lambda f:'rat_make(%s, %s)'%(B(f.numerator),B(f.denominator))
    out.append('check rat_tostr(rat_add(%s, %s)) == "%s"'%(R(x),R(y),str(x+y)))
    out.append('check rat_tostr(rat_div(%s, %s)) == "%s"'%(R(x),R(y),str(x/y)))
open('/tmp/gen/cpython_checks.rce','w').write('\n'.join(out)+'\n')
