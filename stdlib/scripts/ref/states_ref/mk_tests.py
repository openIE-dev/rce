import sys, os, glob
sys.path.insert(0, '.')
from compile import compile_doc
def q(s): return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'
out = ['fn oc(src: text) -> text\n    let none_evs: List[text] = []\n    let r = run(&src, &none_evs)\n    if r.err != ""\n        return "error " + r.err\n    return r.outcome\n']
for f in sorted(glob.glob('scxml/*.scxml')):
    out.append('# %s\ncheck oc(%s) == "pass"\n' % (os.path.basename(f), q(compile_doc(open(f).read()))))
open('rce/off.rce', 'w').write(''.join(out))
