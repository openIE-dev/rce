"""Patch SCION 4.3.28 (@scion-scxml/core-base) to SCXML 1.0 Appendix D
termination: the interpreter stops when a <final> child of <scxml> is
entered (not when every atomic state is final), and later events are
ignored. Usage: python3 patch_scion.py <node_modules dir>"""
import sys, os
p = os.path.join(sys.argv[1], '@scion-scxml/core-base/lib/scion-core-base.js')
s = open(p).read()
a = "this._isInFinalState = this._configuration.iter().every(function(s){ return s.typeEnum === FINAL; });"
b = "this._isInFinalState = this._configuration.iter().some(function(s){ return s.typeEnum === FINAL && s.parent && s.parent.id.indexOf('$generated-scxml') === 0; }); /* rce: SCXML App. D */"
c = "    if(this._isStepping) throw new Error('Cannot call gen during a big-step');\n"
d = c + "    if(this._isInFinalState) return; /* rce: SCXML App. D */\n"
if b not in s:
    assert s.count(a) == 1 and s.count(c) == 1
    s = s.replace(a, b).replace(c, d)
    open(p, 'w').write(s)
print('patched')
