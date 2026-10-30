# Checker conformance for the context table of RCE spec 5.2 (and the call
# rules of 5.2 and 5.11): one probe program per (context, effect) cell and
# per (caller, callee) pair, each admitted or reported as requiring change by `rce check` as
# the table says, with the rule the spec names (W-EFF-1, W-SPAWN-1,
# W-ND-1 ...). Then the official corpus copied from openIE-dev/rce
# (vectors/effect/rce-examples/): each accept program is admitted, each
# ill_typed program requires change with the code in MANIFEST.
# Usage: RCE=path/to/rce python3 tests/effect/effect_probes.py   (exit 1 on any mismatch)
import os, re, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

# The table, transcribed from 5.2 (comptime fail is a compile error, so
# its cell is not probed).
SPEC = {
    'fn':     {'emit': 0, 'fail': 1, 'spawn': 0, 'send': 0, 'recv': 0, 'select': 0, 'foreign': 0},
    'task':   {'emit': 1, 'fail': 1, 'spawn': 1, 'send': 1, 'recv': 1, 'select': 1, 'foreign': 1},
    'main':   {'emit': 1, 'fail': 1, 'spawn': 1, 'send': 1, 'recv': 1, 'select': 1, 'foreign': 1},
    'step':   {'emit': 1, 'fail': 1, 'spawn': 0, 'send': 1, 'recv': 0, 'select': 0, 'foreign': 0, 'state': 1},
    'actor':  {'emit': 1, 'fail': 1, 'spawn': 0, 'send': 1, 'recv': 0, 'select': 0, 'foreign': 0, 'state': 1},
    'comptime': {'emit': 0, 'spawn': 0, 'send': 0, 'recv': 0, 'select': 0, 'foreign': 0},
}
CALLS = {('fn', 'task'): 0, ('step', 'task'): 0, ('actor', 'task'): 0, ('task', 'task'): 1, ('main', 'task'): 1,
         ('fn', 'fn'): 1, ('step', 'fn'): 1, ('actor', 'fn'): 1, ('task', 'fn'): 1, ('main', 'fn'): 1}

COMMON = '''error Bad(n: i64)
extern "C" fn c_tick(w: &World) -> u32
fn worker(w: chan i64)
    w <- 1
fn pure_helper(x: i64) -> i64 = x + 1
'''
BODY = {
    'emit': ['emit 1'],
    'fail': ['fail Bad(n: 1)'],
    'spawn': ['scope', '    spawn worker(c)'],
    'send': ['c <- 1'],
    'recv': ['let _v = <- c'],
    'select': ['select', '    recv some(x) from c -> emit x', '    recv none from c -> emit 0'],
    'foreign': ['let _t = c_tick(w)'],
    'state': ['n += 1'],
    'call_task': ['worker(c)'],
    'call_fn': ['let _y = pure_helper(1)'],
}
def prog(ctx, body):
    # one language: an unread name requires change (W-UNUSED-1), so a probe names `c` and the
    # world `_c`, `_w` unless its body reads them; the world is held by main and tasks only
    txt = ' '.join(body)
    c = 'c' if re.search(r'\bc\b', txt) else '_c'
    w = 'w' if 'c_tick(w)' in txt else '_w'
    b = '\n'.join('    ' + l for l in body)
    nl = '\n'
    if ctx == 'fn':
        return COMMON + 'fn probe() -> i64 ! Bad\n    let %s = chan[i64](1)\n%s\n' % (c, b) + ('' if body == BODY['fail'] else '    return 0\n')
    if ctx == 'task':
        return COMMON + 'fn probe(%s: &World, %s: chan i64) -> i64 ! Bad\n%s\n' % (w, c, b) + ('' if body == BODY['fail'] else '    return 0\n')
    if ctx == 'main':
        return COMMON + 'fn main(%s: &World)\n    let %s = chan[i64](1)\n%s\n' % (w, c, b)
    if ctx == 'step':
        # a state step cannot take a channel in its event (W-TY-2) and
        # cannot start an actor (W-EFF-1), so it makes its own channel
        return COMMON + 'event Go\nstate S\n    n: i64 = 0\nstep S on Go\n    let %s = chan[i64](1)\n%s\n' % (c, b)
    if ctx == 'actor':
        return COMMON + 'event Go(c: chan i64)\nactor A\n    n: i64 = 0\n    on Go(c: %s)\n%s\n' % (c, nl.join('        ' + l for l in body))
    if ctx == 'comptime':
        return COMMON + 'const N: usize = 16\nfn probe() -> usize\n    comptime\n        let %s = chan[i64](1)\n%s\n    return N\n' % (c, nl.join('        ' + l for l in body))

# Cells the checker cannot be shown to allow would be listed here and
# reported, not failed. None: a step reaches a channel only by making one
# (a state step's event may not carry a channel, W-TY-2, and actors start
# only in main or a task, W-EFF-1), and sending on it is admitted.
GAPS = set()
RULES = ('W-EFF-1', 'W-SPAWN-1')

def check(src, d, name):
    f = os.path.join(d, name + '.rce'); open(f, 'w').write(src)
    r = subprocess.run([RCE, 'check', f], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()

bad = 0; n = 0
with tempfile.TemporaryDirectory() as d:
    for ctx, row in SPEC.items():
        for eff, want in row.items():
            ok, out = check(prog(ctx, BODY[eff]), d, '%s_%s' % (ctx, eff)); n += 1
            if (ctx, eff) in GAPS and not ok:
                print('gap %s/%s: spec allows, not expressible: %s' % (ctx, eff, out.split(': ', 2)[-1][:120])); continue
            if not ok and want == 0 and not any(r in out for r in RULES):
                bad += 1; print('MISMATCH %s/%s: requires change under another rule: %s' % (ctx, eff, out[:300])); continue
            if ok != bool(want):
                bad += 1; print('MISMATCH %s/%s: spec %s, checker %s: %s' % (ctx, eff, 'allows' if want else 'forbids', 'admitted' if ok else 'requires change', out[:300]))
            else:
                print('ok %s/%s %s' % (ctx, eff, 'admitted' if ok else out.split(': ', 2)[-1][:90]))
    for (ctx, callee), want in CALLS.items():
        ok, out = check(prog(ctx, BODY['call_' + callee]), d, 'call_%s_%s' % (ctx, callee)); n += 1
        if ok != bool(want):
            bad += 1; print('MISMATCH call %s->%s: spec %s, checker %s: %s' % (ctx, callee, 'allows' if want else 'forbids', 'admitted' if ok else 'requires change', out[:300]))
        else:
            print('ok call %s->%s %s' % (ctx, callee, 'admitted' if ok else out.split(': ', 2)[-1][:90]))
    corpus = os.path.join(root, 'vectors/effect/rce-examples')
    if os.path.isdir(corpus):
        for line in open(os.path.join(corpus, 'MANIFEST')):
            if not line.strip() or line.startswith('#'): continue
            f, code = line.split()
            r = subprocess.run([RCE, 'check', os.path.join(corpus, f)], capture_output=True, text=True); n += 1
            out = (r.stdout + r.stderr).strip()
            good = (r.returncode == 0) if code == 'admit' else (r.returncode != 0 and (' ' + code + ' ') in out)
            if not good:
                bad += 1; print('MISMATCH corpus %s: want %s: %s' % (f, code, out[:300]))
            else:
                print('ok corpus %s %s' % (f, code))
print('%d of %d effect probes agree' % (n - bad, n))
sys.exit(1 if bad else 0)
