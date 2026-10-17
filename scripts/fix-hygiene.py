#!/usr/bin/env python3
"""Mechanical fixes for the one-language hygiene findings of a .rce file, driven by the checker:
W-UNUSED-1 `x`  -> the binding is named `_x`; W-DROP-1 -> `let _ = EXPR`; W-LEN-1 -> `.len()` becomes `.len_bytes()`
(the unqualified text length was the byte count). Lines with more than one candidate are reported, not edited.
usage: fix-hygiene.py RCE_BIN FILE"""
import re, subprocess, sys
rce, path = sys.argv[1:3]
lines = open(path).read().split('\n')
skipped = []
for _ in range(6):
    o = subprocess.run([rce, 'check', path], capture_output=True, text=True)
    out = o.stdout + o.stderr
    if out.startswith('ok'):
        break
    errs = re.findall(r'line (\d+): (W-[A-Z]+-\d+) ([^\n;]*)', out)
    changed = False
    for ln, rule, msg in errs:
        i = int(ln) - 1
        L = lines[i]
        if rule == 'W-UNUSED-1':
            m = re.search(r'`([^`]+)`', msg)
            if not m: continue
            n = m.group(1)
            if n.startswith('use '): continue
            pat = re.compile(r'(?<![\w.])' + re.escape(n) + r'(?!\w)')
            # prefer the right side of `field: name` (a pattern binding)
            mm = re.search(r'(\w+:\s*)' + re.escape(n) + r'(?!\w)', L)
            if mm:
                lines[i] = L[:mm.end() - len(n)] + '_' + n + L[mm.end():]; changed = True
            elif len(pat.findall(L)) >= 1:
                lines[i] = pat.sub('_' + n, L, count=1); changed = True
        elif rule == 'W-DROP-1':
            ind = re.match(r'\s*', L).group(0)
            if not L.strip().startswith('let _ ='):
                lines[i] = ind + 'let _ = ' + L.strip(); changed = True
        elif rule == 'W-LEN-1':
            if L.count('.len()') == 1:
                lines[i] = L.replace('.len()', '.len_bytes()'); changed = True
            else:
                skipped.append((ln, L.strip()))
    open(path, 'w').write('\n'.join(lines))
    if not changed: break
for s in skipped: print('NOT EDITED', path, s)
o = subprocess.run([rce, 'check', path], capture_output=True, text=True)
print((o.stdout + o.stderr).split('\n')[0][:300])
