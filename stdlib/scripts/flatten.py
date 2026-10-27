# Flatten a foundation module and the modules it imports (its `use` items
# and theirs, scripts/uses.py) into one file, for `rce run`, which takes a single file
# (needed where each event must get its own fuel, e.g. Ed25519 per event).
# The flat file means what the module means with its imports linked
# (specification 4.1): each imported module loses its `module` line and its
# `check`, `example` and `intent` items, and every module loses its `use`
# lines and the `alias.` qualifiers they introduce, outside strings and comments.
# Usage: python3 scripts/flatten.py lib/energy/energy.rce [extra.rce...] > out.rce
#        python3 scripts/flatten.py items crypto pq   the items of lib/crypto.rce
#        and lib/pq.rce, flattened the same way, for a test that reads their
#        private items (scripts/ct_check.sh)
import re, sys, os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uses

def aliases(lines):
    """The names a module's `use` lines bind (`use a.b` binds b, `use a.b as c` binds c)."""
    out = []
    for l in lines:
        m = uses.EXACT.match(l)
        if m:
            out.append(m.group(2) or m.group(1).split('.')[-1])
    return out

def unqualify(line, names):
    """`line` with `name.` removed before an identifier, in code only."""
    if not names:
        return line
    pat = re.compile(r'(?<![\w.])(?:%s)\.(?=[A-Za-z_])' % '|'.join(map(re.escape, names)))
    out, i, code = [], 0, ''
    while i < len(line):
        c = line[i]
        if c == '#':
            break
        if c == '"':
            j = i + 1
            while j < len(line) and line[j] != '"':
                j += 2 if line[j] == '\\' else 1
            out.append(pat.sub('', code)); code = ''
            out.append(line[i:j + 1]); i = j + 1
            continue
        code += c; i += 1
    out.append(pat.sub('', code))
    return ''.join(out) + line[i:]

def body(lines, drop_tests):
    """The lines of a module without its `use` lines (and, for an imported
    module, without its `check`, `example` and `intent` items)."""
    names = aliases(lines)
    out, skipping = [], False
    for l in lines:
        if skipping and (l[:1].isspace() or l == ''):
            if l == '':
                skipping = False
                out.append(l)
            continue
        skipping = False
        if uses.LOOKS.match(l):
            continue
        if drop_tests and re.match(r'^(check|example|intent)\b', l):
            skipping = True
            continue
        out.append(unqualify(l, names))
    return out

def lib_items(d):
    src = open(os.path.join(root, 'lib', d + '.rce')).read().split('\n')
    return body(src[1:], True)

if sys.argv[1] == 'items':
    out = []
    for n in sys.argv[2:]:
        out += lib_items(n) + ['']
    print('\n'.join(out))
    sys.exit(0)
try:
    deps = uses.deps(sys.argv[1])
except uses.UsesError as e:
    sys.exit('flatten.py: %s' % e)
lines = open(sys.argv[1]).read().split('\n')
out = [lines[0]]
for d in deps:
    out += lib_items(d)
out += body(lines[1:], False)
for extra in sys.argv[2:]:
    out.append('')
    out.append(open(extra).read())
print('\n'.join(out))
