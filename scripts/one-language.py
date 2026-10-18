#!/usr/bin/env python3
"""Bring the library and its check shards to the one-language rules, mechanically and idempotently.
For every lib/<area>/<name>.rce (with its dependency modules from `# uses:`) and each of its check shards the
assembled module is checked with `rce check` and the findings are fixed in the source file that owns the line:
  W-LEX-4   invisible, bidi, control characters -> \\u{HEX}           (done on the text, before checking)
  W-UNUSED-1 `x` -> `_x`; W-DROP-1 -> `let _ = EXPR`; W-UNUSED-2 `while true bound N` -> `loop bound N`
  W-LEN-1   text `.len()` -> `.len_bytes()` (the old byte count); on a line with several `.len()` each one is
            tried and kept only if the checker then accepts its type (a list `.len()` stays)
Anything else is printed as TODO. usage: one-language.py RCE_BIN [FILE ...]   (default: every lib module)"""
import glob, os, re, subprocess, sys, tempfile
RCE = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LEX = re.compile('[\u202a-\u202e\u2066-\u2069\u061c\u200e\u200f\u200b-\u200d\u2060-\u2064\ufeff\u00ad\u034f\u180e\u115f\u1160\u17b4\u17b5\u3164\uffa0\ufff9-\ufffb\U000e0000-\U000e007f\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufdd0-\ufdef\U00040000-\U000dffff\ufffe\uffff]')
LENP = re.compile(r'\.len(_bytes)?\(\)')
def esc(t): return LEX.sub(lambda m: '\\u{%X}' % ord(m.group(0)), t)
_uses = {}
def uses(f):
    if f not in _uses: _uses[f] = _uses_run(f)
    return _uses[f]
def _uses_run(f):
    return subprocess.run([sys.executable, os.path.join(HERE, 'uses.py'), f], capture_output=True, text=True, cwd=ROOT).stdout.split()
def shards(f):
    t = 'tests/' + f[4:-4] + '_checks'
    return [c for c in [t + '.rce'] + sorted(glob.glob(t + '_*.rce')) if os.path.exists(os.path.join(ROOT, c))]
def run_unit(f, shard, todo):
    parts = {f: open(os.path.join(ROOT, f)).read()}
    if shard: parts[shard] = open(os.path.join(ROOT, shard)).read()
    lines = {k: v.split('\n') for k, v in parts.items()}
    tent = {}      # (file,line) -> ordinal tried as text
    status = {}    # (file,line) -> list of 'u','t','l'
    for _ in range(40):
        d = tempfile.mkdtemp()
        for u in uses(f): open(os.path.join(d, os.path.basename(u) + '.rce'), 'w').write(open(os.path.join(ROOT, 'lib', u + '.rce')).read())
        names = list(lines); off = {}
        asm = []
        for k in names: off[k] = len(asm); asm += lines[k]
        main = os.path.join(d, os.path.basename(f)); open(main, 'w').write('\n'.join(asm))
        o = subprocess.run([RCE, 'check', d if uses(f) else main], capture_output=True, text=True); out = o.stdout + o.stderr
        errs = []
        for ln in out.split('\n'):
            m = re.match(r'(?:REFUSED|FAIL) (\S+): (.*)', ln)
            if m and os.path.basename(m.group(1)) == os.path.basename(f):
                errs += re.findall(r'line (\d+): (W-[A-Z]+-\d+) ([^;]*)', m.group(2))
        if not errs:
            for k in names: open(os.path.join(ROOT, k), 'w').write('\n'.join(lines[k]))
            return True
        changed = False
        byline = {}
        for ln, rule, msg in errs: byline.setdefault(int(ln) - 1, []).append((rule, msg))
        for gi, es in byline.items():
            k = [n for n in names if off[n] <= gi][-1]; i = gi - off[k]; L = lines[k][i]; key = (k, i)
            rules = [r for r, _ in es]
            for rule, msg in es:
                if rule == 'W-UNUSED-1':
                    m = re.search(r'`([^`]+)`', msg)
                    if not m or m.group(1).startswith('use '): continue
                    n = m.group(1); pat = re.compile(r'(?<![\w.])' + re.escape(n) + r'(?!\w)')
                    mm = re.search(r'(\w+:\s*)' + re.escape(n) + r'(?!\w)', L)
                    if mm: L = L[:mm.end() - len(n)] + '_' + n + L[mm.end():]; changed = True
                    elif pat.search(L): L = pat.sub('_' + n, L, count=1); changed = True
                elif rule == 'W-DROP-1' and not L.strip().startswith('let _ ='):
                    L = re.match(r'\s*', L).group(0) + 'let _ = ' + L.strip(); changed = True
                elif rule == 'W-UNUSED-2':
                    mm = re.match(r'(\s*)while true( bound .*)$', L)
                    if mm: L = mm.group(1) + 'loop' + mm.group(2); changed = True
                elif rule == 'W-LEN-1':
                    occ = list(LENP.finditer(L))
                    st = status.setdefault(key, ['u'] * len(occ))
                    if len(st) != len(occ): st = status[key] = ['u'] * len(occ)
                    # an earlier tentative text guess that the checker now rejects is a list
                    if key in tent and any(r not in ('W-LEN-1',) for r in rules):
                        j = tent.pop(key); st[j] = 'l'
                        L = rebuild(L, occ, st); occ = list(LENP.finditer(L))
                    if 'u' in st:
                        j = st.index('u'); st[j] = 't'; tent[key] = j
                        L = rebuild(L, occ, st); changed = True
                    else: todo.append((k, i + 1, 'W-LEN-1 unresolved', L.strip()))
                elif rule not in ('W-NAME-1', 'W-LEX-4'):
                    pass
            if L != lines[k][i]: lines[k][i] = L
            else:
                # a tentative guess followed by another error on the same line: revert
                if key in tent and any(r != 'W-LEN-1' for r in rules):
                    st = status[key]; j = tent.pop(key); st[j] = 'l'
                    occ = list(LENP.finditer(L)); lines[k][i] = rebuild(L, occ, st); changed = True
        if not changed: break
    for gi, es in byline.items():
        k = [n for n in names if off[n] <= gi][-1]
        for rule, msg in es:
            if rule not in ('W-NAME-1',): todo.append((k, gi - off[k] + 1, rule, msg[:100]))
    for k in names: parts[k] = '\n'.join(lines[k])
    for k in names: open(os.path.join(ROOT, k), 'w').write(parts[k])
    return False
def rebuild(L, occ, st):
    out = []; p = 0
    for m, s in zip(occ, st):
        out.append(L[p:m.start()]); out.append('.len_bytes()' if s == 't' else '.len()'); p = m.end()
    return ''.join(out) + L[p:]
def one(a):
    t = []; ok = run_unit(a[0], a[1], t); return a, ok, t
def main():
    files = sys.argv[2:] or sorted(glob.glob(os.path.join(ROOT, 'lib', '*', '*.rce')))
    files = [os.path.relpath(f, ROOT) if os.path.isabs(f) else f for f in files]
    for f in files:   # the text pass first, over the module and its shards
        for p in [f] + shards(f):
            a = open(os.path.join(ROOT, p)).read(); b = esc(a)
            if a != b: open(os.path.join(ROOT, p), 'w').write(b); print('escaped', p)
    todo = []
    clean = []
    for f in files:   # phase 1: each module alone, in order (its fixes are the only writes to it)
        if run_unit(f, None, todo): clean.append(f)
        else: print('not clean:', f)
    # phase 2: the shards of a clean module are independent files, so they run in parallel
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(f, sh) for f in clean for sh in shards(f)]
    with ProcessPoolExecutor(int(os.environ.get('JOBS', '6'))) as ex:
        for a, ok, t in ex.map(one, jobs, chunksize=4):
            todo += t
            if not ok: print('not clean:', *a)
    seen = set()
    for t in todo:
        if t not in seen: seen.add(t); print('TODO', *t)
if __name__ == '__main__': main()
