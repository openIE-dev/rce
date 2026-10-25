# Flatten a foundation module and the modules it names on its
# "# uses:" line into one file, for `rce run`, which takes a single file
# (needed where each event must get its own fuel, e.g. Ed25519 per event).
# Dependency bodies lose their `module` lines; the main module loses its
# `use` lines and the `alias.` qualifiers of those modules, and so does a
# dependency for the other dependencies it uses, and so do the extra files.
# Usage: python3 scripts/flatten.py lib/energy/energy.rce [extra.rce...] > out.rce
import re, sys, os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
main = open(sys.argv[1]).read()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uses
try:
    deps = uses.deps(sys.argv[1])
except uses.UsesError as e:
    sys.exit('flatten.py: %s' % e)
out = []
lines = main.split('\n')
head, body = lines[0], lines[1:]
out.append(head)
out += [l for l in body if l.startswith('#!')]
aliases = []
srcs = []
for d in deps:
    src = open(os.path.join(root, 'lib', d + '.rce')).read().split('\n')
    name = src[0].split()[1]
    aliases.append(name.split('.')[-1])
    srcs.append((name, src))
inlined = {name for name, _src in srcs}
def strip(l):
    for a in aliases:
        l = re.sub(r'(?<![\w.])%s\.(?=[A-Za-z_])' % re.escape(a), '', l)
    return l
# A dependency that uses another inlined dependency (data.sync uses
# data.events) loses that `use` line and its qualifiers too.
for name, src in srcs:
    for l in src[1:]:
        if l.startswith('#!'):
            continue
        m = re.match(r'^use (\S+)$', l)
        if m and m.group(1) in inlined:
            continue
        out.append(strip(l) if re.match(r'^use ', l) is None else l)
for l in body:
    if l.startswith('#!') or re.match(r'^use ', l):
        continue
    out.append(strip(l))
for extra in sys.argv[2:]:
    out.append('')
    out += [strip(l) for l in open(extra).read().split('\n')]
print('\n'.join(out))
