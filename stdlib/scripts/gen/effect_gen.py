# Derived vectors for lib/effect/effect.rce, printed as rce checks.
# Canonical JSON and oracle keys follow rce src/canon.rs; with ref PATH
# PATH (scripts/ref/canon built with cargo) every one is confirmed
# against that Rust code. Usage:
#   python3 scripts/gen/effect_gen.py [ref scripts/ref/canon/target/release/canon-ref]
import hashlib, json, random, subprocess, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../scripts'))
from positional import read  # noqa: E402  (positional-cli.md PC-1)
R = random.Random(19)
def q(s):
    # an rce string literal: \" \\ \n \r \t, \u{X} for other controls
    o = ''
    for c in s:
        o += {'"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t'}.get(c) or (c if ord(c) >= 32 and ord(c) != 127 else '\\u{%x}' % ord(c))
    return '"' + o + '"'

CTX = ['fn', 'task', 'main', 'step', 'actor', 'comptime']
EFF = ['emit', 'fail', 'spawn', 'send', 'recv', 'select', 'state', 'foreign']
def allows(c, e):
    if c not in CTX or e not in EFF: return False
    if e == 'fail': return True
    if c in ('fn', 'comptime'): return False
    if c in ('task', 'main'): return e != 'state'
    return e in ('emit', 'send', 'state')
def may_call(c, k):
    if c not in CTX: return False
    if k == 'fn': return True
    if k in ('task', 'foreign'): return c in ('main', 'task')
    return False

# ---- canonical JSON (canon.rs: bytewise key sort, serde escapes)
def canon(v): return json.dumps(v, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
def rj(v):
    if v is None: return 'enc.JNull'
    if v is True or v is False: return 'enc.JBool(v: %s)' % ('true' if v else 'false')
    if isinstance(v, int): return 'enc.JInt(v: %d)' % v
    if isinstance(v, str): return 'enc.JStr(t: %s)' % q(v)
    if isinstance(v, list): return 'enc.JArr(items: [%s])' % ', '.join(rj(x) for x in v)
    return 'enc.JObj(keys: [%s], vals: [%s])' % (', '.join(q(k) for k in v), ', '.join(rj(x) for x in v.values()))
def rand_text():
    al = ['a', 'Z', '"', '\\', '\n', '\t', '\x01', '\x1f', '\x7f', 'é', '\ue000', '😀', '\uffff', ' ', '/', '\u2028']
    return ''.join(R.choice(al) for _ in range(R.randint(0, 5)))
def rand_val(d=0):
    k = R.randint(0, 6 if d < 3 else 3)
    if k == 0: return None
    if k == 1: return R.random() < 0.5
    if k == 2: return R.choice([0, -1, 1, 2**53, -2**53, 9007199254740993, 2**63 - 1, -2**63, R.randint(-1000, 1000)])
    if k == 3: return rand_text()
    if k == 4: return [rand_val(d + 1) for _ in range(R.randint(0, 3))]
    keys = list(dict.fromkeys(rand_text() for _ in range(R.randint(0, 4))))
    R.shuffle(keys)
    return {kk: rand_val(d + 1) for kk in keys}

def main():
    ref = read('scripts/gen/effect_gen.py', {'ref': 1}, '', 0, 0)[0].get('ref')
    print('# context table (5.2), call and spawn rules')
    for c in CTX + ['nope']:
        for e in EFF + ['print']:
            print('check ok2(allows, %s, %s) == %s' % (q(c), q(e), 'true' if allows(c, e) else 'false'))
    for c in CTX + ['nope']:
        for k in ['fn', 'task', 'foreign', 'actor']:
            print('check ok2(may_call, %s, %s) == %s' % (q(c), q(k), 'true' if may_call(c, k) else 'false'))
            print('check ok2(may_spawn, %s, %s) == %s' % (q(c), q(k), 'true' if (k == 'task' and allows(c, 'spawn')) else 'false'))

    print('# canonical JSON and oracle keys (rce src/canon.rs)')
    vals = [None, True, False, 0, -0, 1, -1, 2**63 - 1, -2**63, '', 'a"b\\c', '\x00\x08\t\n\x0b\x0c\r\x1f\x7f', 'é😀\ue000\u2028', [], {}, [1, [2, [3]]],
            {'b': 1, 'a': 2}, {'\ue000': 1, '😀': 2, 'z': 3, '': 4, 'é': 5}, {'aa': 1, 'a': 2, 'ab': 3}, {'x': {'d': [], 'c': {}}, 'w': None}]
    vals += [rand_val() for _ in range(40)]
    if ref:
        out = subprocess.run([ref], input='\n'.join(json.dumps(v) for v in vals) + '\n', capture_output=True, text=True, check=True).stdout.split('\n')
        for v, line in zip(vals, out):
            h, c = line.split(' ', 1)
            assert c == canon(v) and h == hashlib.sha256(c.encode()).hexdigest(), (v, line)
        print('# (all %d confirmed against scripts/ref/canon)' % len(vals))
    for v in vals:
        c = canon(v)
        print('check cj(%s) == %s' % (rj(v), q(c)))
        print('check hj(%s) == %s' % (rj(v), q(hashlib.sha256(c.encode()).hexdigest())))

    print('# recorded handler: hits, misses, the escalation text')
    recs = [({'op': 'price', 'sku': 'A1'}, 7), ({'sku': 'A1', 'op': 'price'}, 8), ({'op': 'price', 'sku': 'B2'}, 9), ([1, 2], 'pair')]
    asks = [{'sku': 'A1', 'op': 'price'}, {'op': 'price', 'sku': 'B2'}, {'op': 'price', 'sku': 'C3'}, [2, 1], [1, 2], None]
    tab = {}
    for i, r in recs: tab[canon(i)] = r
    res = []; hits = misses = 0; last = ''
    for a in asks:
        c = canon(a)
        if c in tab: hits += 1; res.append(tab[c])
        else: misses += 1; last = 'oracle miss: input %s not recorded' % hashlib.sha256(c.encode()).hexdigest()[:12]; res.append('<miss>')
    print('check oracle_run([%s], [%s], [%s]) == OrOut(got: [%s], hits: %d, misses: %d, last: %s)' % (
        ', '.join(rj(i) for i, _ in recs), ', '.join(rj(r) for _, r in recs), ', '.join(rj(a) for a in asks),
        ', '.join(rj(x) for x in res), hits, misses, q(last)))

    print('# backoff, retry, deadlines')
    def delay(base, cap, n):
        d = base; i = 0
        while i < n and d < cap: d *= 2; i += 1
        return min(d, cap)
    for base, cap in [(1, 8), (3, 100), (5, 5), (0, 10), (2, 1)]:
        for n in [0, 1, 2, 3, 5, 10, 63]:
            print('check dl(Backoff(base: %d, cap: %d, attempts: 3, retry_failed: false), %d) == %d' % (base, cap, n, delay(base, cap, n)))
    ST = ['delivered', 'failed', 'timed_out', 'unreachable']
    for _ in range(30):
        base, cap, att, rf = R.randint(1, 4), R.randint(4, 40), R.randint(1, 5), R.random() < 0.5
        seq = [R.choice(ST[1:]) if R.random() < 0.75 else 'delivered' for _ in range(R.randint(1, 7))]
        nows = []; now = 0
        for _s in seq: now += R.randint(0, 3); nows.append(now)
        att0 = 0; wait = 0; verdict = 'pending'; log = []
        for s, now in zip(seq, nows):
            if verdict not in ('done', 'give_up'):
                if s == 'delivered': verdict = 'done'
                elif not (s in ('timed_out', 'unreachable') or (s == 'failed' and rf)) or att0 + 1 >= att: verdict = 'give_up'
                else: wait = now + delay(base, cap, att0); att0 += 1; verdict = 'retry'
            log.append('%s@%d' % (verdict, wait))
        print('check retry_run(Backoff(base: %d, cap: %d, attempts: %d, retry_failed: %s), [%s], [%s]) == %s' % (
            base, cap, att, 'true' if rf else 'false', ', '.join(q(s) for s in seq), ', '.join(str(x) for x in nows), q(' '.join(log))))
    for iss, to, now in [(0, 5, 4), (0, 5, 5), (10, 0, 10), (10, 3, 9), (10, 3, 13), (2**63, 1, 2**63), (5, 2, 100)]:
        print('check expired(%d, %d, %d) == %s' % (iss, to, now, 'true' if now >= iss and now - iss >= to else 'false'))

if __name__ == '__main__':
    main()
