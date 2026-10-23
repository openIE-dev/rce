#!/usr/bin/env python3
"""Data section of lib/ui/input.rce. is_upper is Rust's char::is_uppercase
(rustc 1.99.0, char::UNICODE_VERSION 17.0.0): the Uppercase property of
DerivedCoreProperties-17.0.0.txt, fetched into ~/.cache/rce-ucd/17 and
checked against its sha256. The UI Events tables are read from
vectors/ui/uievents-code.html (w3c/uievents-code
b201684d1de0af90bc403814bbdee6aa96647130) and vectors/ui/uievents-key.html
(w3c/uievents-key 140cae88d6039b3fb1a51787978678115111c433). Usage: python3 scripts/gen/input_data.py"""
import hashlib, os, sys, urllib.request
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_data import tree
DCP = ('https://www.unicode.org/Public/17.0.0/ucd/DerivedCoreProperties.txt',
       '24c7fed1195c482faaefd5c1e7eb821c5ee1fb6de07ecdbaa64b56a99da22c08')
def get(url, sha):
    p = os.path.expanduser('~/.cache/rce-ucd/17/' + url.rsplit('/', 1)[1])
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        urllib.request.urlretrieve(url, p)
    b = open(p, 'rb').read()
    assert hashlib.sha256(b).hexdigest() == sha, p
    return b.decode()
def upper():
    v = [0] * 0x110000
    for l in get(*DCP).split('\n'):
        l = l.split('#')[0].strip()
        if not l: continue
        f = [x.strip() for x in l.split(';')]
        r, p = f[0], f[1]
        if p != 'Uppercase': continue
        a, _, b = r.partition('..')
        for c in range(int(a, 16), int(b or a, 16) + 1): v[c] = 1
    return v
def runs(v):
    out = []
    for c in range(0x110000):
        if not out or out[-1][1] != v[c]: out.append((c, v[c]))
    return out
import re
def spec_tables():
    v = os.path.join(ROOT, 'vectors/ui/')
    c = re.findall(r'<td class="code-table-code"><code class="code" id="code-([^"]*)">"([^"]*)"</code>\s*<td class="code-table-required">(\w*)', open(v + 'uievents-code.html').read())
    k = re.findall(r'<td class="key-table-key"><code class="key" id="key-([^"]*)">"([^"]*)"</code>\s*<td class="key-table-required">(\w*)', open(v + 'uievents-key.html').read())
    assert len(c) == 172 and len(k) == 284, (len(c), len(k))
    return c, k
def tlist(name, xs):
    return 'fn %s() -> List[text]\n    return [%s]\n' % (name, ', '.join('"%s"' % x for x in xs))
def main():
    p = os.path.join(ROOT, 'lib/ui/input.rce')
    s = open(p).read()
    mark = '# ---- data (scripts/gen/input_data.py; do not edit below)\n'
    head = s[:s.index(mark) + len(mark)]
    body = 'pub fn is_upper(c: i64) -> bool\n' + tree(runs(upper()), leaf=lambda x: 'true' if x else 'false')
    c, k = spec_tables()
    body += tlist('code_values', [x[1] for x in c]) + tlist('code_required_values', [x[1] for x in c if x[2] == 'Yes'])
    body += tlist('key_values', [x[1] for x in k]) + tlist('key_required_values', [x[1] for x in k if x[2] == 'Yes'])
    open(p, 'w').write(head + body)
if __name__ == '__main__': main()
