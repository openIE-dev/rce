#!/usr/bin/env python3
"""E4's declared analysis: ordinary least squares, adjusted R², and a bootstrap over programs.

Standard library only, so that the analysis runs the same on every host.
PROTOCOL.md ("Declared analysis") states what is computed; report.py calls
`regime_b` on the rows of one results file.
"""
import math
import random

# two-sided 95% Student t quantiles (0.975) for 1..30 degrees of freedom
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


class Singular(Exception):
    pass


def mean_interval(xs):
    """Mean and its two-sided 95% t interval; the interval is None for fewer than two values."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t = T975[n - 2] if n - 1 <= len(T975) else 1.960
    h = t * sd / math.sqrt(n)
    return m, (m - h, m + h)


def _solve(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise Singular("singular design")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def ols(xs, y):
    """OLS of y on an intercept and the columns xs (lists). Returns r2, adj_r2 and the coefficients."""
    n, k = len(y), len(xs)
    if n <= k + 1:
        raise Singular("%d rows for %d predictors" % (n, k))
    # standardise each predictor so that bits and bytes and seconds are on one scale
    cols = []
    for x in xs:
        mu = sum(x) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in x) / n)
        if sd == 0:
            raise Singular("a predictor is constant")
        cols.append(([(v - mu) / sd for v in x], mu, sd))
    ym = sum(y) / n
    sst = sum((v - ym) ** 2 for v in y)
    if sst == 0:
        raise Singular("the response is constant")
    yc = [v - ym for v in y]
    xtx = [[sum(a * b for a, b in zip(ci[0], cj[0])) for cj in cols] for ci in cols]
    xty = [sum(a * b for a, b in zip(ci[0], yc)) for ci in cols]
    beta = _solve(xtx, xty)
    predicted = [sum(beta[j] * cols[j][0][i] for j in range(k)) for i in range(n)]
    sse = sum((yc[i] - predicted[i]) ** 2 for i in range(n))
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    coef = [beta[j] / cols[j][2] for j in range(k)]
    intercept = ym - sum(coef[j] * cols[j][1] for j in range(k))
    return {"r2": r2, "adj_r2": adj, "coef": coef, "intercept": intercept, "n": n}


def shares(rows):
    """Adjusted R² of joules on (ε, λ), on the baseline, and their difference, for one set of rows."""
    y = [r["joules"] for r in rows]
    e = ols([[r["eps"] for r in rows], [r["lam"] for r in rows]], y)
    b = ols([[r["base"] for r in rows]], y)
    return {"eps_lam": e["adj_r2"], "baseline": b["adj_r2"], "baseline_minus_eps_lam": b["adj_r2"] - e["adj_r2"],
            "eps_lam_r2": e["r2"], "baseline_r2": b["r2"]}


def regime_b(rows, resamples, seed, level=0.95):
    """The declared analysis for regime B over rows {program, joules, eps, lam, base}.

    Point estimates on all rows; percentile intervals from `resamples` draws
    of whole programs with replacement (all configurations of a drawn program
    come together). A draw whose design is singular is skipped and counted.
    """
    point = shares(rows)
    progs = sorted({r["program"] for r in rows})
    by = {p: [r for r in rows if r["program"] == p] for p in progs}
    rng = random.Random(seed)
    draws = {k: [] for k in ("eps_lam", "baseline", "baseline_minus_eps_lam")}
    skipped = 0
    for _ in range(resamples):
        pick = [rng.choice(progs) for _ in progs]
        sample = [r for p in pick for r in by[p]]
        try:
            s = shares(sample)
        except Singular:
            skipped += 1
            continue
        for k in draws:
            draws[k].append(s[k])
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = {"point": point, "programs": len(progs), "rows": len(rows), "resamples": resamples,
           "skipped": skipped, "seed": seed, "level": level, "interval": {}}
    for k, v in draws.items():
        if not v:
            out["interval"][k] = None
            continue
        v.sort()
        out["interval"][k] = (_quantile(v, lo_q), _quantile(v, hi_q))
    return out


def _quantile(sorted_v, q):
    """Linear interpolation between order statistics (type 7)."""
    h = (len(sorted_v) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (h - lo) * (sorted_v[hi] - sorted_v[lo])
