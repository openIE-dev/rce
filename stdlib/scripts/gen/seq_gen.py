# Usage: python3 scripts/gen/seq_gen.py (from stdlib/). Writes the generated
# checks of col.seq (tests/col/seq_checks.rce and seq_checks_N.rce, with
# scripts/gen/seq_helpers.rce in front of each shard) and of the col set
# algebra and searches (tests/col/col_checks_2.rce).
# CPython 3.13 is the oracle: itertools (groupby, batched, pairwise,
# zip_longest, accumulate, product, permutations, combinations,
# combinations_with_replacement, cycle, islice, chain and the recipes
# sliding_window, unique_everseen, unique_justseen and roundrobin of the
# itertools documentation), bisect, heapq (merge, nsmallest, nlargest),
# collections.Counter, functools.cmp_to_key, sorted and set. Seed
# 20261025. Derived vectors, not official ones.
import bisect, collections, functools, heapq, itertools, operator, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcelit import lit, lits

assert sys.version_info[:2] >= (3, 13), 'CPython 3.13 or later is the oracle'
r = random.Random(20261025)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')

# ---- the itertools documentation recipes (Python 3.13)
def sliding_window(iterable, n):
    iterator = iter(iterable)
    window = collections.deque(itertools.islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)
def unique_everseen(iterable, key=None):
    seen = set()
    if key is None:
        for element in itertools.filterfalse(seen.__contains__, iterable):
            seen.add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen.add(k)
                yield element
def unique_justseen(iterable, key=None):
    if key is None:
        return map(operator.itemgetter(0), itertools.groupby(iterable))
    return map(next, map(operator.itemgetter(1), itertools.groupby(iterable, key)))
def roundrobin(*iterables):
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = itertools.cycle(itertools.islice(iterators, num_active))
        yield from map(next, iterators)

# ---- key functions and comparators (the t_ functions of seq_helpers.rce)
k0 = lambda x: abs(x) % 5
k1 = lambda x: abs(x) // 4
k2 = len
k3 = lambda x: (abs(x) % 3, -x)
kp = operator.itemgetter(0)
def sgn(v): return (v > 0) - (v < 0)
c0 = functools.cmp_to_key(lambda a, b: sgn(abs(a) - abs(b)))
c1 = functools.cmp_to_key(lambda a, b: sgn(b - a))
def _c2(a, b):
    if len(a) != len(b): return -1 if len(a) < len(b) else 1
    if a == b: return 0
    return -1 if b < a else 1
c2 = functools.cmp_to_key(_c2)
p0 = lambda x: abs(x) % 3 == 0
lin = lambda acc, x: 2 * acc + x

# ---- rce literals
def il(xs): return '[' + ', '.join('%d' % x for x in xs) + ']'
def tl(xs): return lits(xs)
def opt(v): return 'none' if v is None else 'some(%d)' % v
def b(v): return 'true' if v else 'false'
def nest(ls, f=il):
    ls = [list(l) for l in ls]
    flat = [x for l in ls for x in l]
    return '%s, %s' % (f(flat), il([len(l) for l in ls]))
def pl(ps): return '[' + ', '.join('(%d, %d)' % p for p in ps) + ']'

ALPHA = ['a', 'b', 'Z', 'é', '日', 'ab', '']
def rints(n, lo=-12, hi=12): return [r.randint(lo, hi) for _ in range(n)]
def rtexts(n): return [''.join(r.choice(ALPHA) for _ in range(r.randint(0, 3))) for _ in range(n)]
def rlen(): return r.choice([0, 1, 2, 3, 5, 8, 13, 20])
def cuts(n):
    o = []
    while n > 0:
        c = r.randint(1, max(1, n)) if r.random() < 0.5 else r.randint(1, 3)
        c = min(c, n); o.append(c); n -= c
    if r.random() < 0.3: o.insert(r.randint(0, len(o)), 0)
    return o

sections = []   # (name, [check lines])
def sec(name):
    rows = []; sections.append((name, rows)); return rows

# ---------------- sorting ----------------
rows = sec('sorted, sorted(reverse=True)')
for _ in range(40):
    xs = rints(rlen())
    rows.append('check t_sorted(%s, %s, %s)' % (il(xs), il(sorted(xs)), il(sorted(xs, reverse=True))))
for _ in range(25):
    xs = rtexts(rlen())
    rows.append('check t_sorted_t(%s, %s, %s)' % (tl(xs), tl(sorted(xs)), tl(sorted(xs, reverse=True))))
rows = sec('argsort views')
for _ in range(40):
    xs = rints(rlen()); n = len(xs)
    a = sorted(range(n), key=xs.__getitem__)
    ka = sorted(range(n), key=lambda i: k0(xs[i]))
    kd = sorted(range(n), key=lambda i: k0(xs[i]), reverse=True)
    ca = sorted(range(n), key=lambda i: c0(xs[i]))
    rows.append('check t_arg(%s, %s, %s, %s, %s)' % (il(xs), il(a), il(ka), il(kd), il(ca)))
rows = sec('sorted(key=...), sorted(key=..., reverse=True)')
for _ in range(40):
    xs = rints(r.choice([0, 1, 5, 12, 30, 60]))
    rows.append('check t_sbk(%s, %s, %s, %s, %s, %s)' % (il(xs), il(sorted(xs, key=k0)), il(sorted(xs, key=k0, reverse=True)), il(sorted(xs, key=k1)), il(sorted(xs, key=k3)), il(sorted(xs, key=k3, reverse=True))))
for _ in range(25):
    xs = rtexts(rlen())
    rows.append('check t_sbk_t(%s, %s, %s)' % (tl(xs), tl(sorted(xs, key=k2)), tl(sorted(xs, key=k2, reverse=True))))
rows = sec('sorted(key=functools.cmp_to_key(cmp))')
for _ in range(40):
    xs = rints(r.choice([0, 1, 2, 3, 7, 16, 33, 64]))
    rows.append('check t_sw(%s, %s, %s)' % (il(xs), il(sorted(xs, key=c0)), il(sorted(xs, key=c1))))
for _ in range(25):
    xs = rtexts(rlen())
    rows.append('check t_sw_t(%s, %s)' % (tl(xs), tl(sorted(xs, key=c2))))
rows = sec('gather, is_sorted')
for _ in range(20):
    xs = rints(rlen())
    idx = [r.randrange(len(xs)) for _ in range(r.randint(0, 6))] if xs else []
    rows.append('check t_gather(%s, %s, %s)' % (il(xs), il(idx), il([xs[i] for i in idx])))
for _ in range(10):
    xs = rints(r.randint(0, 6))
    idx = [r.randrange(len(xs)) for _ in range(r.randint(0, 3))] if xs else []
    bad = len(xs) + r.randint(0, 3)
    idx2 = idx + [bad] + [0]
    rows.append('check t_gather_bad(%s, %s, %d)' % (il(xs), il(idx2), bad))
for _ in range(25):
    xs = rints(rlen())
    if r.random() < 0.5: xs = sorted(xs, key=k0) if r.random() < 0.5 else sorted(xs)
    pl_ = all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
    kk = all(k0(xs[i]) <= k0(xs[i + 1]) for i in range(len(xs) - 1))
    rows.append('check t_issorted(%s, %s, %s)' % (il(xs), b(pl_), b(kk)))
rows = sec('selection (sorted(xs)[n]), heapq.nsmallest, heapq.nlargest')
for _ in range(30):
    xs = rints(rlen()); n = r.randint(0, len(xs) + 1)
    w = sorted(xs)[n] if n < len(xs) else None
    wk = sorted(xs, key=k0)[n] if n < len(xs) else None
    rows.append('check t_nth(%s, %d, %s, %s)' % (il(xs), n, opt(w), opt(wk)))
for _ in range(30):
    xs = rints(rlen()); k = r.randint(0, len(xs) + 2)
    rows.append('check t_top(%s, %d, %s, %s, %s, %s)' % (il(xs), k, il(heapq.nsmallest(k, xs)), il(heapq.nlargest(k, xs)), il(heapq.nsmallest(k, xs, key=k0)), il(heapq.nlargest(k, xs, key=k0))))
for _ in range(15):
    xs = rtexts(rlen()); k = r.randint(0, len(xs) + 2)
    rows.append('check t_top_t(%s, %d, %s, %s, %s, %s)' % (tl(xs), k, tl(heapq.nsmallest(k, xs)), tl(heapq.nlargest(k, xs)), tl(heapq.nsmallest(k, xs, key=k2)), tl(heapq.nlargest(k, xs, key=k2))))

# ---------------- searching ----------------
rows = sec('bisect_left, bisect_right (sorted and unsorted lists)')
for i in range(60):
    xs = rints(rlen(), -6, 6)
    if i % 4 != 3: xs.sort()
    x = r.randint(-7, 7)
    rows.append('check t_bis(%s, %d, %d, %d)' % (il(xs), x, bisect.bisect_left(xs, x), bisect.bisect_right(xs, x)))
for _ in range(15):
    xs = sorted(rtexts(rlen())); x = rtexts(1)[0]
    rows.append('check t_bis_t(%s, %s, %d, %d)' % (tl(xs), lit(x), bisect.bisect_left(xs, x), bisect.bisect_right(xs, x)))
for _ in range(30):
    xs = sorted(rints(rlen(), -6, 6)); x = r.randint(-7, 7)
    lo = r.randint(0, len(xs) + 1); hi = r.randint(0, len(xs))
    rows.append('check t_bis_in(%s, %d, %d, %d, %d, %d)' % (il(xs), x, lo, hi, bisect.bisect_left(xs, x, lo, hi), bisect.bisect_right(xs, x, lo, hi)))
for _ in range(30):
    xs = sorted(rints(rlen(), -20, 20), key=k1); k = r.randint(0, 6)
    rows.append('check t_bisk(%s, %d, %d, %d)' % (il(xs), k, bisect.bisect_left(xs, k, key=k1), bisect.bisect_right(xs, k, key=k1)))
for _ in range(30):
    xs = sorted(rints(rlen(), -6, 6)); x = r.randint(-7, 7)
    i = bisect.bisect_left(xs, x)
    rows.append('check t_bs(%s, %d, %s)' % (il(xs), x, opt(i if i < len(xs) and xs[i] == x else None)))
rows = sec('insort_left, insort_right (and key=)')
for _ in range(25):
    xs = sorted(rints(rlen(), -6, 6)); x = r.randint(-7, 7)
    a = list(xs); bisect.insort_left(a, x); c = list(xs); bisect.insort_right(c, x)
    assert a == c
    rows.append('check t_ins(%s, %d, %s, %d, %d)' % (il(xs), x, il(a), bisect.bisect_left(xs, x), bisect.bisect_right(xs, x)))
for _ in range(25):
    xs = sorted([(r.randint(0, 4), r.randint(0, 99)) for _ in range(rlen())], key=kp)
    x = (r.randint(0, 5), 100 + r.randint(0, 99))
    a = list(xs); bisect.insort_left(a, x, key=kp); c = list(xs); bisect.insort_right(c, x, key=kp)
    rows.append('check t_insk(%s, (%d, %d), %s, %s, %d, %d)' % (pl(xs), x[0], x[1], pl(a), pl(c), bisect.bisect_left(xs, x[0], key=kp), bisect.bisect_right(xs, x[0], key=kp)))
rows = sec('heapq.merge (two lists: sorted and unsorted inputs; many lists and key=: sorted inputs, stable sort otherwise)')
for i in range(30):
    a = rints(rlen(), -6, 6); c = rints(rlen(), -6, 6)
    if i % 3 != 2: a.sort(); c.sort()
    rows.append('check t_merge(%s, %s, %s)' % (il(a), il(c), il(list(heapq.merge(a, c)))))
for i in range(30):
    ls = [rints(r.choice([0, 1, 2, 4, 7]), -9, 9) for _ in range(r.randint(0, 6))]
    if i % 3 != 2:
        ls = [sorted(l, key=k0) for l in ls] if i % 3 == 0 else [sorted(l) for l in ls]
    # heapq.merge for sorted inputs; for other inputs the documented result
    # is the stable sort of the concatenation
    flat = list(itertools.chain.from_iterable(ls))
    want = list(heapq.merge(*ls)) if all(l == sorted(l) for l in ls) else sorted(flat)
    wk = list(heapq.merge(*ls, key=k0)) if all(l == sorted(l, key=k0) for l in ls) else sorted(flat, key=k0)
    assert want == sorted(flat) and wk == sorted(flat, key=k0)
    rows.append('check t_merge_all(%s, %s, %s)' % (nest(ls), il(want), il(wk)))

rows = sec('heapq.merge of two sorted streams fed in chunks (seq_merge_step)')
# A second seeded stream, so that the vectors above stay as they were.
r2 = random.Random(20261027)
def cuts2(n):
    o = []
    while n > 0:
        c = min(n, r2.randint(0, 4)); o.append(c); n -= c
    if r2.random() < 0.3: o.insert(r2.randint(0, len(o)), 0)
    return o
for _ in range(40):
    a = sorted(r2.randint(-6, 6) for _ in range(r2.choice([0, 1, 3, 8, 15])))
    c = sorted(r2.randint(-6, 6) for _ in range(r2.choice([0, 1, 3, 8, 15])))
    n = r2.randint(1, 5)
    rows.append('check t_merge_step(%s, %s, %s, %s, %d, %s)' % (il(a), il(cuts2(len(a))), il(c), il(cuts2(len(c))), n, il(list(heapq.merge(a, c)))))

# ---------------- grouping and transformation ----------------
rows = sec('itertools.groupby, group by key')
for _ in range(30):
    xs = rints(rlen(), -9, 9)
    g = [(k, list(v)) for k, v in itertools.groupby(xs, k0)]
    rows.append('check t_runs(%s, %s, %s)' % (il(xs), il([k for k, _ in g]), nest([v for _, v in g])))
    g = [(k, list(v)) for k, v in itertools.groupby(sorted(xs, key=k0), k0)]
    rows.append('check t_gby(%s, %s, %s)' % (il(xs), il([k for k, _ in g]), nest([v for _, v in g])))
for _ in range(15):
    xs = rtexts(rlen())
    g = [(k, list(v)) for k, v in itertools.groupby(sorted(xs, key=k2), k2)]
    rows.append('check t_gby_t(%s, %s, %s)' % (tl(xs), il([k for k, _ in g]), nest([v for _, v in g], tl)))
rows = sec('itertools.batched, sliding_window, pairwise')
for _ in range(30):
    xs = rints(rlen()); n = r.randint(1, 6)
    rows.append('check t_chunks(%s, %d, %s)' % (il(xs), n, nest(itertools.batched(xs, n))))
    rows.append('check t_windows(%s, %d, %s)' % (il(xs), n, nest(sliding_window(xs, n))))
for _ in range(5):
    xs = rints(rlen())
    try: list(itertools.batched(xs, 0)); raise SystemExit('batched(0) should fail')
    except ValueError: pass
    rows.append('check t_size_bad(%s, 0)' % il(xs))
rows.append('check t_size_bad([1, 2, 3], 1000001)')
for _ in range(20):
    xs = rints(rlen()); pw = list(itertools.pairwise(xs))
    rows.append('check t_pairwise(%s, %s, %s)' % (il(xs), il([p[0] for p in pw]), il([p[1] for p in pw])))
rows = sec('zip, zip_longest, zip(strict=True), unzip, enumerate')
for _ in range(20):
    a = rints(rlen()); c = rtexts(rlen())
    z = list(zip(a, c))
    rows.append('check t_zip(%s, %s, %s, %s)' % (il(a), tl(c), il([p[0] for p in z]), tl([p[1] for p in z])))
    S = object()
    z = [(-1 if x is S else x, '-' if y is S else y) for x, y in itertools.zip_longest(a, c, fillvalue=S)]
    rows.append('check t_zipl(%s, %s, %s, %s)' % (il(a), tl(c), il([p[0] for p in z]), tl([p[1] for p in z])))
    try: list(zip(a, c, strict=True)); ok = True
    except ValueError: ok = False
    rows.append('check t_zipx(%s, %s, %s)' % (il(a), tl(c), b(ok)))
for _ in range(15):
    n = rlen(); a = rints(n); c = rtexts(n)
    ps = list(zip(a, c)); ua, uc = (list(t) for t in zip(*ps)) if ps else ([], [])
    rows.append('check t_unzip([%s], %s, %s)' % (', '.join('(%d, %s)' % (x, lit(y)) for x, y in ps), il(ua), tl(uc)))
    st = 1000 + 7 * len(c)
    rows.append('check t_enum(%s, %s, %d, %s)' % (tl(c), il([i for i, _ in enumerate(c)]), st, il([i for i, _ in enumerate(c, st)])))
rows = sec('itertools.chain.from_iterable')
for _ in range(20):
    ls = [rints(r.choice([0, 0, 1, 3, 6])) for _ in range(r.randint(0, 6))]
    flat = list(itertools.chain.from_iterable(ls))
    rows.append('check t_flatten(%s, %s)' % (il(flat), il([len(l) for l in ls])))
rows = sec('dict.fromkeys, unique_everseen(key=), unique_justseen')
for _ in range(30):
    xs = rints(rlen(), -6, 6)
    rows.append('check t_dedupe(%s, %s, %s, %s)' % (il(xs), il(list(dict.fromkeys(xs))), il(list(unique_everseen(xs, k0))), il(list(unique_justseen(xs)))))
for _ in range(15):
    xs = rtexts(rlen())
    rows.append('check t_dedupe_t(%s, %s, %s, %s)' % (tl(xs), tl(list(dict.fromkeys(xs))), tl(list(unique_everseen(xs, k2))), tl(list(unique_justseen(xs)))))
rows = sec('collections.Counter items and most_common')
for _ in range(30):
    xs = rints(rlen(), -5, 5); c = collections.Counter(xs); n = r.randint(0, len(c) + 2)
    mc = c.most_common(n)
    rows.append('check t_counter(%s, %s, %s, %d, %s, %s)' % (il(xs), il(list(c)), il(list(c.values())), n, il([k for k, _ in mc]), il([v for _, v in mc])))
for _ in range(15):
    xs = rtexts(rlen()); c = collections.Counter(xs); n = r.randint(0, len(c) + 2)
    mc = c.most_common(n)
    rows.append('check t_counter_t(%s, %s, %s, %d, %s, %s)' % (tl(xs), tl(list(c)), il(list(c.values())), n, tl([k for k, _ in mc]), il([v for _, v in mc])))
rows = sec('partition, itertools.accumulate')
for _ in range(20):
    xs = rints(rlen())
    rows.append('check t_partition(%s, %s, %s)' % (il(xs), il([x for x in xs if p0(x)]), il([x for x in xs if not p0(x)])))
    rows.append('check t_acc(%s, %s, %s)' % (il(xs), il(list(itertools.accumulate(xs))), il(list(itertools.accumulate(xs, max)))))
    init = r.randint(-3, 3)
    rows.append('check t_scan(%s, %d, %s)' % (il(xs[:12]), init, il(list(itertools.accumulate(xs[:12], lin, initial=init))[1:])))
rows = sec('itertools.product, permutations, combinations, combinations_with_replacement')
for _ in range(15):
    a = rints(r.randint(0, 4)); c = rtexts(r.randint(0, 4)); z = list(itertools.product(a, c))
    rows.append('check t_product(%s, %s, %s, %s)' % (il(a), tl(c), il([p[0] for p in z]), tl([p[1] for p in z])))
for _ in range(20):
    ls = [rints(r.randint(0, 3)) for _ in range(r.randint(0, 4))]
    z = [list(t) for t in itertools.product(*ls)]
    lim = len(z) if r.random() < 0.3 else 1000000
    rows.append('check t_prod_all(%s, %d, %s)' % (nest(ls), lim, nest(z)))
    if len(z) > 0:
        rows.append('check t_prod_all(%s, %d, %s)' % (nest(ls), len(z) - 1, nest([[-999]])))
for _ in range(20):
    xs = rints(r.randint(0, 5)); k = r.randint(0, len(xs) + 1)
    for name, f in (('t_perm', itertools.permutations), ('t_comb', itertools.combinations), ('t_comb_rep', itertools.combinations_with_replacement)):
        z = [list(t) for t in f(xs, k)]
        rows.append('check %s(%s, %d, 1000000, %s)' % (name, il(xs), k, nest(z)))
        if len(z) > 0:
            rows.append('check %s(%s, %d, %d, %s)' % (name, il(xs), k, len(z) - 1, nest([[-999]])))
for mode, f in ((1, itertools.permutations), (2, itertools.combinations), (3, itertools.combinations_with_replacement)):
    for n, k in ((0, 0), (0, 1), (1, 0), (3, 2), (4, 4), (5, 2), (6, 3), (2, 5), (7, 1)):
        z = [list(t) for t in f(range(n), k)]
        rows.append('check t_cursor(%d, %d, %d, %s)' % (mode, n, k, nest(z)))
rows = sec('islice(cycle(xs), start, start + n), roundrobin')
for _ in range(20):
    xs = rints(r.randint(0, 5)); s = r.randint(0, 12); n = r.randint(0, 14)
    rows.append('check t_cycle(%s, %d, %d, %s)' % (il(xs), s, n, il(list(itertools.islice(itertools.cycle(xs), s, s + n)))))
rows.append('check t_cycle([1, 2], 0, 1000001, [0 - 999])')
for _ in range(20):
    ls = [rints(r.choice([0, 1, 2, 5])) for _ in range(r.randint(0, 5))]
    rows.append('check t_interleave(%s, %s)' % (nest(ls), il(list(roundrobin(*ls)))))

# ---------------- chunked state ----------------
rows = sec('chunked state: Counter.update, unique_everseen, nsmallest/nlargest, sliding_window, batched, groupby')
for _ in range(20):
    xs = rints(r.choice([0, 3, 10, 25, 40]), -6, 6); cu = cuts(len(xs)); c = collections.Counter(xs)
    n = r.randint(0, len(c) + 1); mc = c.most_common(n); probe = r.randint(-7, 7)
    rows.append('check t_counts_feed(%s, %s, %s, %s, %d, %s, %s, %d, %d)' % (il(xs), il(cu), il(list(c)), il(list(c.values())), n, il([k for k, _ in mc]), il([v for _, v in mc]), probe, c[probe]))
    rows.append('check t_seen_feed(%s, %s, %s)' % (il(xs), il(cu), il(list(unique_everseen(xs)))))
    k = r.randint(0, 6)
    rows.append('check t_top_feed(%s, %s, %d, %s, %s, %s, %s)' % (il(xs), il(cu), k, il(heapq.nsmallest(k, xs)), il(heapq.nlargest(k, xs)), il(heapq.nsmallest(k, xs, key=k0)), il(heapq.nlargest(k, xs, key=k0))))
    w = r.randint(1, 5)
    rows.append('check t_win_feed(%s, %s, %d, %s)' % (il(xs), il(cu), w, nest(sliding_window(xs, w))))
    rows.append('check t_batch_feed(%s, %s, %d, %s)' % (il(xs), il(cu), w, nest(itertools.batched(xs, w))))
    g = [(k, list(v)) for k, v in itertools.groupby(xs, k0)]
    rows.append('check t_runs_feed(%s, %s, %s, %s)' % (il(xs), il(cu), il([k for k, _ in g]), nest([v for _, v in g])))

# ---------------- col: set algebra and searches ----------------
col_rows = []
for _ in range(60):
    a = sorted(set(r.randint(-8, 8) for _ in range(r.randint(0, 9))))
    c = sorted(set(r.randint(-8, 8) for _ in range(r.randint(0, 9))))
    if r.random() < 0.2: c = sorted(set(a) | set(c))
    if r.random() < 0.1: c = list(a)
    A, B = set(a), set(c)
    col_rows.append('check t_setops(%s, %s, %s, %s, %s, %s, [%s])' % (il(a), il(c), il(sorted(A | B)), il(sorted(A & B)), il(sorted(A - B)), il(sorted(A ^ B)),
        ', '.join(b(v) for v in (A <= B, A < B, A >= B, A.isdisjoint(B), B <= A, B < A))))
for _ in range(25):
    xs = rints(rlen(), -5, 5)
    col_rows.append('check t_setfrom(%s, %s)' % (il(xs), il(sorted(set(xs)))))
for _ in range(50):
    s = sorted(set(r.randint(-10, 10) for _ in range(r.randint(0, 8)))); k = r.randint(-11, 11)
    i, j = bisect.bisect_left(s, k), bisect.bisect_right(s, k)
    fl = s[j - 1] if j > 0 else None; ce = s[i] if i < len(s) else None
    lo_ = s[i - 1] if i > 0 else None; hi_ = s[j] if j < len(s) else None
    k2_ = r.randint(-11, 11)
    rng = [x for x in s if k <= x < k2_]
    col_rows.append('check t_nav(%s, %d, %d, %d, %s, %s, %s, %s, %d, %s)' % (il(s), k, i, j, opt(fl), opt(ce), opt(lo_), opt(hi_), k2_, il(rng)))
col_hdr = '''# Helpers of the generated col checks: set algebra against Python's set,
# lower_bound/upper_bound against bisect, neighbour searches against
# bisect, on sets and on ordered map keys.
fn t_setops(a: List[i64], b: List[i64], u: List[i64], x: List[i64], d: List[i64], sd: List[i64], tests: List[bool]) -> bool
    let got = [set_is_subset(&a, &b), set_is_proper_subset(&a, &b), set_is_superset(&a, &b), set_is_disjoint(&a, &b), set_is_subset(&b, &a), set_is_proper_subset(&b, &a)]
    return set_union(&a, &b) == u and set_inter(&a, &b) == x and set_diff(&a, &b) == d and set_symdiff(&a, &b) == sd and got == tests and set_is_valid(&u) and set_is_disjoint(&d, &b)
fn t_setfrom(xs: List[i64], want: List[i64]) -> bool = set_from(&xs) == want and set_is_valid(&want) and (xs.len() < 2 or set_is_valid(&xs) == (xs == want))
fn t_nav(s: List[i64], k: i64, lb: usize, ub: usize, fl: i64?, ce: i64?, lo: i64?, hi: i64?, k2: i64, rng: List[i64]) -> bool
    var m: OrdMap[i64, i64] = OrdMap(keys: [], vals: [])
    for x in s
        om_insert(&mut m, x, 0 - x)
    let a = lower_bound(&s, &k) == lb and upper_bound(&s, &k) == ub and set_range(&s, &k, &k2) == rng
    let c = set_floor(&s, &k) == fl and set_ceiling(&s, &k) == ce and set_lower(&s, &k) == lo and set_higher(&s, &k) == hi
    let d = om_floor_key(&m, &k) == fl and om_ceiling_key(&m, &k) == ce and om_lower_key(&m, &k) == lo and om_higher_key(&m, &k) == hi
    return a and c and d
'''

# ---------------- write ----------------
gen_line = '# Generated by scripts/gen/seq_gen.py (CPython %d.%d.%d, seed 20261025); do not edit.\n' % sys.version_info[:3]
helpers = open(os.path.join(HERE, 'seq_helpers.rce')).read()
rows_all = []
for name, rs in sections:
    rows_all.append('# ---- ' + name)
    rows_all.extend(rs)
SHARD = 330
shards = []
cur = []
for row in rows_all:
    cur.append(row)
    if sum(1 for x in cur if x.startswith('check ')) >= SHARD:
        shards.append(cur); cur = []
if cur: shards.append(cur)
out = os.path.join(ROOT, 'tests', 'col')
for f in os.listdir(out):
    if f.startswith('seq_checks') and f[len('seq_checks'):].rstrip('.rce').lstrip('_').isdigit() or f == 'seq_checks.rce':
        os.remove(os.path.join(out, f))
total = 0
for i, rs in enumerate(shards):
    name = 'seq_checks.rce' if i == 0 else 'seq_checks_%d.rce' % (i + 1)
    with open(os.path.join(out, name), 'w') as f:
        f.write(gen_line + helpers + '\n' + '\n'.join(rs) + '\n')
    total += sum(1 for x in rs if x.startswith('check '))
with open(os.path.join(out, 'col_checks_2.rce'), 'w') as f:
    f.write(gen_line + col_hdr + '\n' + '\n'.join(col_rows) + '\n')
print('col.seq: %d checks in %d shards; col: %d checks' % (total, len(shards), len(col_rows)))
