#!/usr/bin/env python3
"""One language: a loop variable that its body never reads is named `_i` (W-UNUSED-1). Works on rce text
embedded in Rust test files, including programs written as one string with `\\n` escapes.
usage: fix-unused-loops.py FILE.rs..."""
import re, sys
HDR = re.compile(r'^(\s*)for (\w+) in ([^{]*?)(?: par)?\s*$')
for path in sys.argv[1:]:
    src = open(path).read()
    # pseudo-lines: split on real newlines and literal \n; remember separators for exact rebuild
    toks = re.split(r'(\\n|\n)', src)
    items = [(toks[i], toks[i + 1] if i + 1 < len(toks) else '') for i in range(0, len(toks), 2)]
    texts = [t for t, _ in items]
    def indent(t):
        # inside a string literal the program text may follow an opening quote
        m = re.match(r'^([^"]*")?(\s*)', t)
        return len(m.group(2)), (m.group(1) or '')
    n = 0
    for k, t in enumerate(texts):
        m = re.match(r'^(.*?"|)(\s*)for (\w+) in (\S.*?)(?: par)?(\s*(?:\\n)?"?[,);]*)$', t)
        if not m or m.group(3).startswith('_') or t.rstrip().endswith('{'): continue
        if '..' not in m.group(4) and 'range' not in m.group(4): continue
        ind = len(m.group(2)); name = m.group(3); body = []
        j = k + 1
        while j < len(texts):
            u = texts[j]
            if u.strip() == '' or u.strip() in ('"', '";', '",', '");'):
                if u.strip() == '': j += 1; continue
                break
            mi = re.match(r'^(?:[^"]*")?(\s*)', u)
            if len(mi.group(1)) <= ind and not (items[j-1][1] == '\\n' and False): break
            body.append(u); j += 1
        if not body: continue
        if not any(re.search(r'(?<![\w.])' + re.escape(name) + r'(?!\w)', b) for b in body):
            texts[k] = t.replace('for ' + name + ' in', 'for _' + name + ' in', 1); n += 1
    out = ''.join(texts[i] + items[i][1] for i in range(len(items)))
    if out != src: open(path, 'w').write(out)
    print(path, n)
