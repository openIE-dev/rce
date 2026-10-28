#!/usr/bin/env python3
"""The library modules a module of lib/ needs, read from its `use` items.

A module names each module it imports in a `use` item (specification 4.1),
one per line at the start of the line:

    use text.utf8
    use enc as e

The modules a module needs are those it imports and, in turn, the modules
they import. A module path `a.b` is the file lib/a/b.rce; a one-segment
path `a` is lib/a/a.rce or lib/a.rce. The file must declare that module.
`std.text`, `std.collections` and `std.math` are part of the checker and
have no file. Any line that starts with "use " must be exactly that form,
and must name a module that exists: a malformed line, an unknown module, a
module named twice or a module importing itself is an error that names the
file and line, never a silent skip.

    python3 scripts/uses.py lib/a/b.rce     prints the a/b paths it needs, one per line
    python3 scripts/uses.py --all           checks every module of lib/
    python3 scripts/uses.py --selftest      checks the parser itself
Exit status 2 on any error."""
import glob, os, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LOOKS = re.compile(r'^use\b')
SEG = r'[a-z][a-z0-9_]*'
EXACT = re.compile(r'^use (%s(?:\.%s)*)(?: as (%s))?$' % (SEG, SEG, SEG))
BUILT_IN = {'std.text', 'std.collections', 'std.math'}
class UsesError(Exception): pass
def parse(src, name='<src>', self_mod=None):
    """Module paths ('a.b') that src imports, in order of their `use` lines."""
    found, at = [], {}
    for n, line in enumerate(src.split('\n'), 1):
        if not LOOKS.match(line):
            continue
        where = '%s:%d' % (name, n)
        m = EXACT.match(line)
        if not m:
            raise UsesError('%s: malformed use item %r; expected "use a.b" or "use a.b as c"' % (where, line))
        mod = m.group(1)
        if mod in at:
            raise UsesError('%s: %s is imported twice (first at line %d)' % (where, mod, at[mod]))
        if mod == self_mod:
            raise UsesError('%s: %s imports itself' % (where, mod))
        at[mod] = n
        found.append(mod)
    return found
def module_file(x):
    """'a/b' for the file of module x under lib/, or None when x has no file."""
    a = x.split('.')
    cands = ['/'.join(a)] + (['%s/%s' % (a[0], a[0])] if len(a) == 1 else [])
    for c in cands:
        if os.path.isfile(os.path.join(ROOT, 'lib', c + '.rce')):
            return c
    return None
def module_name(src):
    m = re.search(r'^module (\S+)\s*$', src, re.M)
    return m.group(1) if m else None
def direct(path):
    """'a/b' paths of the modules path imports, checked to exist."""
    src = open(path).read()
    rel = os.path.relpath(path, ROOT)
    out = []
    for x in parse(src, rel, module_name(src)):
        if x in BUILT_IN:
            continue
        f = module_file(x)
        if f is None:
            raise UsesError('%s: unknown module %s (no lib/%s.rce)' % (rel, x, x.replace('.', '/')))
        decl = module_name(open(os.path.join(ROOT, 'lib', f + '.rce')).read())
        if decl != x:
            raise UsesError('%s: lib/%s.rce declares module %r, not %s' % (rel, f, decl, x))
        out.append(f)
    return out
def deps(path):
    """'a/b' paths (relative to lib/, no extension) that path needs: the
    modules it imports and theirs, each once, in the order first reached."""
    seen, order, todo = set(), [], direct(path)
    me = os.path.relpath(os.path.abspath(path), os.path.join(ROOT, 'lib'))[:-len('.rce')]
    while todo:
        f = todo.pop(0)
        if f in seen or f == me:
            continue
        seen.add(f)
        order.append(f)
        todo += direct(os.path.join(ROOT, 'lib', f + '.rce'))
    return order
def selftest():
    ok = {'use text.utf8': ['text.utf8'], 'module x\nuse enc\nuse data.events': ['enc', 'data.events'],
          'use enc as e': ['enc'], 'module x\n# no deps here': [], '# use enc': [], 'fn user() -> i64 = 1': [],
          'used_by = 1': []}
    bad = ['use text/utf8', 'use text.utf8 enc', 'use text.utf8,enc', 'use Version', 'use  enc', 'use enc ',
           'use enc as', 'use enc.', 'use enc\nuse enc', 'use ']
    for s, want in ok.items():
        assert parse(s) == want, (s, parse(s))
    for s in bad:
        try:
            parse(s)
        except UsesError:
            continue
        raise AssertionError('accepted %r' % s)
    try:
        parse('module a.b\nuse a.b', self_mod='a.b'); raise AssertionError('self')
    except UsesError:
        pass
    print('uses.py selftest: %d good, %d malformed rejected' % (len(ok), len(bad) + 1))
def main(a):
    try:
        if a == ['--selftest']:
            selftest()
        elif a == ['--all']:
            fs = sorted(glob.glob(os.path.join(ROOT, 'lib/*.rce')) + glob.glob(os.path.join(ROOT, 'lib/*/*.rce')))
            n = sum(1 for f in fs if deps(f))
            print('uses.py: %d modules, %d with dependencies, every use item well formed' % (len(fs), n))
        elif len(a) == 1:
            for d in deps(a[0]):
                print(d)
        else:
            sys.exit(__doc__)
    except UsesError as e:
        sys.stderr.write('uses.py: error: %s\n' % e)
        sys.exit(2)
if __name__ == '__main__':
    main(sys.argv[1:])
