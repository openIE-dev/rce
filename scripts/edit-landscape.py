#!/usr/bin/env python3
"""The edit landscape of the corpus programs (critique 4: smoothness, RCE-JEPA H11 and RJ-F4).

For every admitted program of the corpus population (scripts/corpus.py) with
example lines, the script walks chains of typed edits away from the program
under a fixed seed and records, at each nominal edit distance d = 1..5, the
checker outcome and a candidate energy E, then runs a greedy descent on E
from chain points at distance k = 1, 3, 5 and records whether it returns to
an admitted program equal in behaviour to the original. Everything it
measures is registered in
openie-research-experiments/rce/bench/edit-landscape/PREREG.json.

Edits are symmetric typed single-token changes on code lines only (never on
intent, example or check lines, comments or string contents):

- rename: one occurrence of a non-keyword identifier becomes another
  identifier name of the program;
- operator: one operator becomes another of its class (+ - * /, < <= > >=,
  == !=, and or);
- literal: an integer n becomes n+1 or n-1 (n > 0); true and false swap;
- type: one of u64 i64 f64 becomes another of the three; text and bool swap.

Each edit has an inverse in the set, so the original is always reachable.

Outcome of a point: admitted_equal, admitted_differs (admitted, behaviour
differs on a probe), requires_change_examples (the only rule named is
W-INT-1), requires_change (other rules). Behaviour is judged on probes: the
original's example event lists and every ordered pair of them concatenated,
traced by `rce trace --json FILE --streams` on the text with its example and
check lines removed; a probe differs when its (outcome, hashed state)
sequence differs from the original's or cannot be traced.

Energy: E(x) = R(x) + B(x), R = the number of distinct rule ids `rce check`
names (0 when admitted), B = the share of probes that differ.

    scripts/edit-landscape.py --rce PATH/TO/rce --out DIR [--jobs N]

writes DIR/points.jsonl, DIR/descents.jsonl and DIR/summary.json. The run is
a pure function of the checkout and the seed (each family has its own
seeded generator, so the job count does not change the result).
"""
import argparse, importlib.util, json, multiprocessing, re, shutil, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


corpus = _load("corpus", "corpus.py")
tc = _load("table_control", "table-control.py")

SEED = 0x5EED_C0DE_0004
CHAINS, DEPTH = 6, 5
DESCENT_FROM = (1, 3, 5)
NEIGHBOURS = 64
IDENT = re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*(?![\w])")
OPER = re.compile(r"(?<![<>=!+\-*/&|-])(<=|>=|==|!=|<|>|\+|-(?![>=])|\*|/|\band\b|\bor\b)(?![=<>])")
INT = re.compile(r"(?<![\w.])\d+(?![\w.])")
BOOL = re.compile(r"\b(true|false)\b")
TYPE = re.compile(r"\b(u64|i64|f64|text|bool)\b")
TYPE_OTHERS = {"u64": ["i64", "f64"], "i64": ["u64", "f64"], "f64": ["u64", "i64"], "text": ["bool"], "bool": ["text"]}


def code_spans(src):
    """(line index, text, [(start, end) of code segments outside strings and comments]) of editable lines."""
    lines = src.split("\n")
    protected = set()
    for i, l in enumerate(lines):
        if l.startswith("intent "):
            protected.add(i)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("    ") or lines[j].strip() == "":
                    protected.add(j)
                else:
                    break
        if re.match(r"\s*(example|check)\b", l) or l.lstrip().startswith("#"):
            protected.add(i)
    out = []
    for i, l in enumerate(lines):
        if i in protected or not l.strip():
            continue
        segs, last = [], 0
        for m in re.finditer(corpus.STRING.pattern + "|#", l):
            if m.group(0) == "#":
                segs.append((last, m.start()))
                last = None
                break
            segs.append((last, m.start()))
            last = m.end()
        if last is not None:
            segs.append((last, len(l)))
        out.append((i, l, [s for s in segs if s[1] > s[0]]))
    return lines, out


def edits(src):
    """Every single typed edit of `src` as (class, line, start, end, replacement), in a fixed order."""
    lines, spans = code_spans(src)
    names = set()
    for _, l, segs in spans:
        for a, b in segs:
            names.update(m.group(0) for m in IDENT.finditer(l, a, b) if m.group(0) not in corpus.KEYWORDS)
    names = sorted(names)
    out = []
    for i, l, segs in spans:
        for a, b in segs:
            for m in IDENT.finditer(l, a, b):
                w = m.group(0)
                if w in corpus.KEYWORDS:
                    continue
                out += [("rename", i, m.start(), m.end(), n) for n in names if n != w]
            for m in OPER.finditer(l, a, b):
                cls = next(c for c in corpus.OP_CLASSES if m.group(0) in c)
                out += [("operator", i, m.start(), m.end(), o) for o in cls if o != m.group(0)]
            for m in INT.finditer(l, a, b):
                n = int(m.group(0))
                out.append(("literal", i, m.start(), m.end(), str(n + 1)))
                if n > 0:
                    out.append(("literal", i, m.start(), m.end(), str(n - 1)))
            for m in BOOL.finditer(l, a, b):
                out.append(("literal", i, m.start(), m.end(), "false" if m.group(0) == "true" else "true"))
            for m in TYPE.finditer(l, a, b):
                out += [("type", i, m.start(), m.end(), t) for t in TYPE_OTHERS[m.group(0)]]
    return lines, out


def apply(lines, e):
    _, i, a, b, rep = e
    new = list(lines)
    new[i] = lines[i][:a] + rep + lines[i][b:]
    return "\n".join(new)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda k: v[k])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(xs) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / (sxx * syy) ** 0.5


class Family:
    def __init__(self, rce_bin, index, path, src, tmp):
        self.rce = corpus.Rce(rce_bin)
        self.index, self.path, self.src = index, path, src
        self.tmp = Path(tmp)
        self.rng = corpus.Rng(SEED + index)
        exs = tc.examples(src)
        streams = [", ".join(seq) for _, _, seq, _ in exs]
        self.probes = streams + [a + ", " + b for a in streams for b in streams]
        self.cache = {}
        self.ref = self.trace(src)

    def trace(self, text):
        f = self.tmp / "s.rce"
        f.write_text(tc.strip_obligations(text), encoding="utf-8")
        return tc.trace(self.rce, f, self.probes, self.tmp)

    def eval(self, text):
        key = corpus.sha(text)
        if key in self.cache:
            return self.cache[key]
        f = self.tmp / "x.rce"
        f.write_text(text, encoding="utf-8")
        c = self.rce.run("check", str(f))
        msg = c.stdout + c.stderr
        rules = [] if c.returncode == 0 else sorted(set(re.findall(r"\bW-[A-Z0-9]+-\d+\b|\bW-PLACE\b", msg))) or ["(no rule id)"]
        tr = self.trace(text)
        if tr is None or len(tr) != len(self.ref):
            b = 1.0
        else:
            b = sum(1 for x, y in zip(tr, self.ref) if x != y) / len(self.ref)
        if not rules:
            outcome = "admitted_equal" if b == 0 else "admitted_differs"
        elif rules == ["W-INT-1"]:
            outcome = "requires_change_examples"
        else:
            outcome = "requires_change"
        r = {"rules": rules, "B": b, "E": len(rules) + b, "outcome": outcome}
        self.cache[key] = r
        return r

    def chain(self, c):
        text, seen, pts = self.src, {corpus.sha(self.src)}, []
        for d in range(1, DEPTH + 1):
            lines, es = edits(text)
            cands = []
            for e in es:
                t = apply(lines, e)
                if corpus.sha(t) not in seen:
                    cands.append((e, t))
            if not cands:
                break
            e, text = cands[self.rng.next() % len(cands)]
            seen.add(corpus.sha(text))
            r = self.eval(text)
            pts.append({"path": self.path, "chain": c, "d": d, "edit": e[0], "text_sha256": corpus.sha(text), **r, "_text": text})
        return pts

    def descend(self, start, k):
        text, cur = start["_text"], start
        steps = 0
        while cur["E"] > 0 and steps < k + 3:
            lines, es = edits(text)
            if len(es) > NEIGHBOURS:
                idx, pick = list(range(len(es))), []
                for _ in range(NEIGHBOURS):
                    j = self.rng.next() % len(idx)
                    pick.append(idx.pop(j))
                es = [es[j] for j in pick]
            best = None
            for e in es:
                t = apply(lines, e)
                r = self.eval(t)
                if best is None or r["E"] < best[1]["E"]:
                    best = (t, r)
            steps += 1
            if best is None or best[1]["E"] >= cur["E"]:
                break
            text, cur = best
        return {"path": self.path, "chain": start["chain"], "k": k, "start_E": start["E"], "start_outcome": start["outcome"],
                "steps": steps, "end_E": cur["E"], "end_outcome": cur["outcome"], "end_is_original": text == self.src,
                "success": cur["outcome"] == "admitted_equal", "reached_admitted": cur["outcome"] in ("admitted_equal", "admitted_differs")}


def run_family(args):
    rce_bin, index, path, src = args
    tmp = tempfile.mkdtemp(prefix="edit-landscape-")
    try:
        fam = Family(rce_bin, index, path, src, tmp)
        if fam.ref is None:
            return {"path": path, "excluded": "the original cannot be traced on its probes"}
        pts, des = [], []
        for c in range(CHAINS):
            ch = fam.chain(c)
            pts += ch
            for p in ch:
                if p["d"] in DESCENT_FROM:
                    des.append(fam.descend(p, p["d"]))
        for p in pts:
            del p["_text"]
        return {"path": path, "probes": len(fam.probes), "points": pts, "descents": des, "evaluations": len(fam.cache), "rce_cpu_s": round(fam.rce.cpu, 1)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def population(rce_bin, tmp):
    rce = corpus.Rce(rce_bin)
    cands = [(p, None) for p, _, _ in corpus.population()] + [(p, s) for p, s, _ in corpus.conformance(True)]
    seen, out = set(), []
    f = Path(tmp) / "orig.rce"
    for p, src in cands:
        src = src if src is not None else p.read_text(encoding="utf-8")
        f.write_text(src, encoding="utf-8")
        if rce.run("check", str(f)).returncode != 0:
            continue
        nl = rce.run("fmt", "--nameless", str(f)).stdout
        if not nl.strip() or corpus.sha(nl) in seen:
            continue
        seen.add(corpus.sha(nl))
        if tc.examples(src):
            out.append((p.relative_to(ROOT).as_posix(), src))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rce", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip()
    tmp = tempfile.mkdtemp(prefix="edit-landscape-pop-")
    fams = population(a.rce, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    work = [(a.rce, i, p, s) for i, (p, s) in enumerate(fams)]
    with multiprocessing.Pool(a.jobs) as pool:
        res = pool.map(run_family, work, chunksize=1)
    excluded = [r for r in res if "excluded" in r]
    res = [r for r in res if "excluded" not in r]
    pts = [p for r in res for p in r["points"]]
    des = [d for r in res for d in r["descents"]]
    with open(out / "points.jsonl", "w") as fh:
        for p in pts:
            fh.write(json.dumps(p, sort_keys=True) + "\n")
    with open(out / "descents.jsonl", "w") as fh:
        for d in des:
            fh.write(json.dumps(d, sort_keys=True) + "\n")

    # flips: transitions d-1 -> d along each chain (d = 0 is the admitted original)
    flips = trans = a2r = r2a = 0
    for r in res:
        by = {}
        for p in r["points"]:
            by.setdefault(p["chain"], []).append(p)
        for ch in by.values():
            prev = True
            for p in sorted(ch, key=lambda x: x["d"]):
                now = p["outcome"].startswith("admitted")
                trans += 1
                if now != prev:
                    flips += 1
                    a2r += prev
                    r2a += now
                prev = now
    d1 = [p for p in pts if p["d"] == 1]
    mean_by_d = {0: 0.0}
    for d in range(1, DEPTH + 1):
        xs = [p["E"] for p in pts if p["d"] == d]
        mean_by_d[d] = round(sum(xs) / len(xs), 4) if xs else None
    vals = [mean_by_d[d] for d in range(DEPTH + 1)]
    mono = all(vals[i] <= vals[i + 1] for i in range(DEPTH))
    xs = [0] * len(res) + [p["d"] for p in pts]
    ys = [0.0] * len(res) + [p["E"] for p in pts]
    fam_rows = []
    for r in res:
        prof = [0.0]
        for d in range(1, DEPTH + 1):
            e = [p["E"] for p in r["points"] if p["d"] == d]
            prof.append(sum(e) / len(e) if e else None)
        ok = [v for v in prof if v is not None]
        nondec = all(ok[i] <= ok[i + 1] + 1e-12 for i in range(len(ok) - 1))
        rho = spearman([0] + [p["d"] for p in r["points"]], [0.0] + [p["E"] for p in r["points"]])
        fam_rows.append({"path": r["path"], "profile": [round(v, 4) if v is not None else None for v in prof], "nondecreasing": nondec, "spearman": None if rho is None else round(rho, 4)})
    n_f = len(fam_rows)
    h11p = sum(f["nondecreasing"] for f in fam_rows) / n_f if n_f else None
    h11s = sum(1 for f in fam_rows if f["spearman"] is not None and f["spearman"] > 0) / n_f if n_f else None
    desc = {}
    for k in DESCENT_FROM:
        ds = [d for d in des if d["k"] == k]
        desc[k] = {"n": len(ds), "success": sum(d["success"] for d in ds), "reached_admitted": sum(d["reached_admitted"] for d in ds),
                   "end_is_original": sum(d["end_is_original"] for d in ds), "already_admitted_equal_at_start": sum(d["start_outcome"] == "admitted_equal" for d in ds),
                   "success_rate": round(sum(d["success"] for d in ds) / len(ds), 4) if ds else None}
    outcomes_by_d = {d: {o: sum(1 for p in pts if p["d"] == d and p["outcome"] == o) for o in ("admitted_equal", "admitted_differs", "requires_change_examples", "requires_change")} for d in range(1, DEPTH + 1)}
    rules = {}
    for p in pts:
        for x in p["rules"]:
            rules[x] = rules.get(x, 0) + 1
    h11_ok = h11p is not None and h11p >= 0.80
    s3 = desc[3]["success_rate"]
    summary = {
        "commit": commit, "seed": hex(SEED), "families": n_f, "excluded": excluded, "points": len(pts), "descents": len(des),
        "edits_by_class": {c: sum(1 for p in pts if p["edit"] == c) for c in ("rename", "operator", "literal", "type")},
        "outcomes_by_d": outcomes_by_d,
        "flip_rate": round(flips / trans, 4) if trans else None, "transitions": trans, "flips_admitted_to_change": a2r, "flips_change_to_admitted": r2a,
        "d1_requires_change_share": round(sum(1 for p in d1 if not p["outcome"].startswith("admitted")) / len(d1), 4) if d1 else None,
        "mean_energy_by_d": mean_by_d, "mean_energy_nondecreasing": mono,
        "spearman_E_d": round(spearman(xs, ys), 4),
        "H11_primary_share_nondecreasing": round(h11p, 4) if h11p is not None else None,
        "H11_secondary_share_spearman_positive": round(h11s, 4) if h11s is not None else None,
        "H11": "supported" if h11_ok else "against",
        "descent": desc,
        "RJ-F4": "observed" if (not h11_ok and s3 is not None and s3 < 0.5) else "not observed",
        "rule_counts": dict(sorted(rules.items(), key=lambda kv: -kv[1])),
        "families_detail": fam_rows,
        "rce_cpu_s": round(sum(r["rce_cpu_s"] for r in res), 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "families_detail"}, indent=1))


if __name__ == "__main__":
    main()
