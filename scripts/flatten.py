# Flatten a foundation module and the modules it names on its
# "# uses:" line into one file, for `rce run`, which takes a single file
# (needed where each event must get its own fuel, e.g. Ed25519 per event).
# Dependency bodies lose their `module` lines; the main module loses its
# `use` lines and the `alias.` qualifiers of those modules.
# Usage: python3 scripts/flatten.py lib/energy/energy.rce [extra.rce...] > out.rce
import re, sys, os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
main = open(sys.argv[1]).read()
m = re.search(r'^# uses: (.*)$', main, re.M)
deps = m.group(1).split() if m else []
out = []
lines = main.split('\n')
head, body = lines[0], lines[1:]
out.append(head)
out += [l for l in body if l.startswith('#!')]
aliases = []
for d in deps:
    src = open(os.path.join(root, 'lib', d + '.rce')).read().split('\n')
    name = src[0].split()[1]
    aliases.append(name.split('.')[-1])
    out += [l for l in src[1:] if not l.startswith('#!')]
for l in body:
    if l.startswith('#!') or re.match(r'^use ', l):
        continue
    for a in aliases:
        l = re.sub(r'(?<![\w.])%s\.(?=[A-Za-z_])' % re.escape(a), '', l)
    out.append(l)
for extra in sys.argv[2:]:
    out.append('')
    out.append(open(extra).read())
print('\n'.join(out))
