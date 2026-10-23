#!/usr/bin/env python3
"""The dependency marker of a foundation module, parsed strictly.

A module that needs other foundation modules names them on one line:

    # uses: text.utf8, enc.enc

Exactly "# uses: " and then one or more module paths separated by ", ".
A module path is lowercase dotted segments ([a-z][a-z0-9_]*, at least
two); a.b is lib/a/b.rce, which must declare "module a.b" (lib/a/a.rce
may declare just "module a", as enc.enc does). Any line
that looks like the marker ("#", optional spaces, "uses", optional spaces,
":") must be exactly that form: a malformed list, an unknown or
duplicated module, a module naming itself, or a second marker line is an
error that names the file and line, never a silent skip.

    python3 scripts/uses.py lib/a/b.rce     prints a/b paths, one per line
    python3 scripts/uses.py --all           checks every lib/*/*.rce
    python3 scripts/uses.py --selftest      checks the parser itself
Exit status 2 on any error."""
import glob, os, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LOOKS = re.compile(r'^#\s*uses\s*:', re.I)
MOD = r'[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+'
EXACT = re.compile(r'^# uses: (%s(?:, %s)*)$' % (MOD, MOD))
class UsesError(Exception): pass
def parse(src, name='<src>', exists=None, self_mod=None):
    """Module paths ('a.b') named by src's marker, [] if it has none."""
    found = None
    for n, line in enumerate(src.split('\n'), 1):
        if not LOOKS.match(line):
            continue
        where = '%s:%d' % (name, n)
        if found is not None:
            raise UsesError('%s: second "# uses:" line (first at line %d)' % (where, found[0]))
        m = EXACT.match(line)
        if not m:
            raise UsesError('%s: malformed dependency list %r; expected "# uses: a.b, c.d" (lowercase dotted module paths, ", " between)' % (where, line))
        mods = m.group(1).split(', ')
        if len(set(mods)) != len(mods):
            raise UsesError('%s: a module is listed twice in %r' % (where, line))
        if self_mod in mods:
            raise UsesError('%s: %s lists itself' % (where, self_mod))
        if exists:
            for x in mods:
                why = exists(x)
                if why:
                    raise UsesError('%s: %s' % (where, why))
        found = (n, mods)
    return found[1] if found else []
def lib_exists(x):
    p = os.path.join(ROOT, 'lib', *x.split('.')) + '.rce'
    if not os.path.isfile(p):
        return 'unknown module %s (no lib/%s.rce)' % (x, x.replace('.', '/'))
    a = x.split('.')
    names = {x} | ({a[0]} if len(a) == 2 and a[0] == a[1] else set())
    decl = module_name(open(p).read())
    if decl not in names:
        return 'lib/%s.rce declares module %r, not %s' % (x.replace('.', '/'), decl, ' or '.join(sorted(names)))
    return None
def module_name(src):
    m = re.search(r'^module (\S+)\s*$', src, re.M)
    return m.group(1) if m else None
def deps(path):
    """'a/b' paths (relative to lib/, no extension) that path uses."""
    src = open(path).read()
    self_mod = module_name(src)
    return [x.replace('.', '/') for x in parse(src, os.path.relpath(path, ROOT), lib_exists, self_mod)]
def selftest():
    ok = {'# uses: text.utf8': ['text.utf8'], '# uses: enc.enc, data.events': ['enc.enc', 'data.events'],
          'module x\n# no deps here': [], '# Cargo uses Version parsing': [], '#: uses: x': []}
    bad = ['# uses: text/utf8', '# uses: text.utf8 enc.enc', '# uses: text.utf8,enc.enc', '# uses: Version, Comparator',
           '# uses:text.utf8', '#uses: text.utf8', '# Uses: text.utf8', '# uses: text', '# uses: text.utf8,',
           '# uses: ', '# uses: text.utf8 ', '# uses: a.b, a.b', '# uses: a.b\n# uses: c.d']
    for s, want in ok.items():
        assert parse(s) == want, (s, parse(s))
    for s in bad:
        try:
            parse(s)
        except UsesError:
            continue
        raise AssertionError('accepted %r' % s)
    try:
        parse('module a.b\n# uses: a.b', self_mod='a.b'); raise AssertionError('self')
    except UsesError:
        pass
    print('uses.py selftest: %d good, %d malformed rejected' % (len(ok), len(bad) + 1))
def main(a):
    try:
        if a == ['--selftest']:
            selftest()
        elif a == ['--all']:
            fs = sorted(glob.glob(os.path.join(ROOT, 'lib/*/*.rce')))
            n = sum(1 for f in fs if deps(f))
            print('uses.py: %d modules, %d with dependencies, all markers well formed' % (len(fs), n))
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
