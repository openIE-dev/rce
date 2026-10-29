#!/usr/bin/env python3
"""Deterministic rename of reserved words used as names in .rce sources.

usage: rename-reserved.py MAP FILE_OR_DIR...   (rewrites in place; prints one line per changed file)
MAP lines: OLD NEW  ('#' comments). Word tokens outside string literals and
comments are renamed, everywhere (declarations, uses, after a dot), so a field
and its accesses stay consistent. A file that already contains NEW as a word
token outside strings and comments is reported (COLLISION) and left unchanged.
"""
import re, sys, os

def load(p):
    m = {}
    for l in open(p):
        l = l.split('#', 1)[0].split()
        if len(l) == 2:
            m[l[0]] = l[1]
    return m

TOK = re.compile(r'"(?:\\.|[^"\\\n])*"|#[^\n]*|[A-Za-z_][A-Za-z0-9_]*')

def words(src):
    for t in TOK.finditer(src):
        s = t.group(0)
        if s[0] not in '"#':
            yield s

def rename(src, m):
    return TOK.sub(lambda t: t.group(0) if t.group(0)[0] in '"#' else m.get(t.group(0), t.group(0)), src)

def main(a):
    m = load(a[0])
    files = []
    for p in a[1:]:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                if '/.git' in r or '/target' in r:
                    continue
                files += [os.path.join(r, f) for f in sorted(fs) if f.endswith('.rce')]
        else:
            files.append(p)
    bad = 0
    for f in sorted(files):
        s = open(f, encoding='utf-8').read()
        ws = set(words(s))
        if not (ws & set(m)):
            continue
        clash = sorted(set(m[o] for o in ws & set(m)) & ws)
        if clash:
            print('COLLISION', f, clash); bad += 1; continue
        n = rename(s, m)
        open(f, 'w', encoding='utf-8').write(n)
        print('renamed', f)
    return 1 if bad else 0

sys.exit(main(sys.argv[1:]))
