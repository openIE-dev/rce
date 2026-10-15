import random, subprocess, sys
sys.path.insert(0, '.')
from gen_live import chart
from compile import compile_doc
from mk_tests import q
N = int(sys.argv[1]); seed = int(sys.argv[2])
R = random.Random(seed); cases = []
for _ in range(N):
    d, e = chart(R); cases.append((d, e))
import json
REF = sys.argv[4] if len(sys.argv) > 4 else 'xstate'
if REF == 'xstate':
    p = subprocess.run(['node', 'xstate_live.js'], input=''.join(json.dumps({'src': compile_doc(d), 'evs': e}) + '\n' for d, e in cases), capture_output=True, text=True, timeout=600)
else:
    p = subprocess.run(['node', 'scion_live.js'], input=''.join(d + '\t' + ','.join(e) + '\n' for d, e in cases), capture_output=True, text=True, timeout=600)
outs = p.stdout.split('\n')
H = '''fn lv(src: text, evs: text) -> text = lvm(src, evs, false)
fn lx(src: text, evs: text) -> text = lvm(src, evs, true)
fn lvm(src: text, evs: text, xs: bool) -> text
    var es: List[text] = []
    if evs != ""
        es = evs.split(",")
    let r = run_mode(&src, &es, xs)
    if r.err != ""
        return "error " + r.err
    var vs: List[text] = []
    for i in 0..r.vals.len()
        vs.push(if r.isset[i] == 1 then r.vals[i].tostr() else "undefined")
    let cf = copy(r.leaves)
    let tr = copy(r.trace)
    let head = if r.outcome != "" then "final " else "live "
    return head + ",".join(cf) + " | " + " ".join(tr) + " | " + ",".join(vs)
'''
lines = [H]; skipped = 0
for (d, e), o in zip(cases, outs):
    if o.startswith('ERR'): skipped += 1; continue
    src = compile_doc(d)
    order = [l.split()[1] for l in src.split('\n') if l.startswith('state ')]
    head, tr, dm = o.split(' | ')
    fin, conf = head.split(' ', 1) if ' ' in head else (head, '')
    conf = [x for x in order if x in conf.split(',')] if conf else []
    tops = [l.split()[1] for l in src.split('\n') if l.startswith('state ') and l.split()[2] == 'final' and l.split()[3] == '_scxml']
    fin = 'final' if any(x in tops for x in conf) else 'live'
    lines.append('check ' + ('lx' if REF == 'xstate' else 'lv') + '(%s, "%s") == %s\n' % (q(src), ','.join(e), q('%s %s | %s | %s' % (fin, ','.join(conf), tr, dm))))
open(sys.argv[3], 'w').write(''.join(lines))
print('skipped', skipped, file=sys.stderr)
