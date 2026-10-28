#!/usr/bin/env python3
"""The Form corpus (RCE-JEPA workstream W1) and the H1 measurement.

Walks the hand-written programs of this repository, keeps the ones `rce
check` accepts standalone, and writes one row per program with its intent
block, nameless canonical source, shape hash, verdict, the size of the Form
and of each emitted program, and rho = b(code) / b(P) as preregistered in
openie-research-experiments/rce/bench/form-space/PREREG.json. Negatives are
the conformance tests that must be reported (with their diagnostic code) and
checker-verified mutants of the accepted programs under a fixed seed: three
whitespace-token mutants and two of each typed operator (rename, operator,
literal, guard, type), each classified as reported (with its diagnostic),
caught by the program's own example lines (W-INT-1), accepted with the same
shape, accepted and equivalent (same trace heads), or accepted with a
behaviour change the examples did not catch.

    scripts/corpus.py --rce PATH/TO/rce --out DIR

writes DIR/forms.jsonl, DIR/negatives.jsonl and DIR/summary.json. The run is
a pure function of the checkout and the seed. Every row names the commit.
"""
import argparse, gzip, hashlib, json, math, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 0x5EED_C0DE_0001
TARGETS = ["python", "rust", "js", "scala"]
RUNTIME = {"python": "rt.py", "rust": "rt.rs", "js": "rt.js", "scala": "rt.scala"}
TOKEN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^\sA-Za-z0-9_]')
STRING = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def strip_comments(text, marker):
    """Remove comments that start with `marker` outside strings, line by line."""
    out = []
    for line in text.split("\n"):
        cut = None
        for m in re.finditer(STRING.pattern + "|" + re.escape(marker), line):
            if m.group(0) == marker:
                cut = m.start()
                break
        out.append(line[:cut] if cut is not None else line)
    return "\n".join(out)


def tokens(text, marker):
    return TOKEN.findall(strip_comments(text, marker))


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def gz_bits(text):
    return len(gzip.compress(text.encode(), 9)) * 8


class Rce:
    def __init__(self, bin_):
        self.bin = bin_
        self.cpu = 0.0

    def run(self, *args, stdin=None):
        t0 = time.time()
        r = subprocess.run([self.bin, *args], capture_output=True, text=True, input=stdin)
        self.cpu += time.time() - t0
        return r


def intent_block(src):
    """The intent line and its indented example lines, as written."""
    lines = src.split("\n")
    out = []
    for i, l in enumerate(lines):
        if l.startswith("intent "):
            out.append(l)
            for m in lines[i + 1:]:
                if m.startswith("    ") or m.strip() == "":
                    if m.strip():
                        out.append(m)
                else:
                    break
    return "\n".join(out)


def population():
    """(path, licence, origin) of every candidate program, by the rule of the preregistration."""
    rows = []
    for d, origin in [("spec/examples", "spec-examples"), ("spec/examples-0.9/ok", "spec-examples-0.9"), ("programs", "programs"),
                      ("apps", "apps"), ("book/listings", "book-listings"), ("stdlib/lib", "stdlib")]:
        for p in sorted((ROOT / d).rglob("*.rce")):
            if "/fit/" in p.as_posix() or "_FIXTURE" in p.name:
                continue
            rows.append((p, "MIT OR Apache-2.0", origin))
    return rows


def conformance(expect_accept):
    """Conformance tests: (path, source without the meta block, meta) for the expectation."""
    sys.path.insert(0, str(ROOT / "conformance" / "harness"))
    import rcetest  # noqa: E402
    out = []
    for p in sorted((ROOT / "conformance" / "tests").rglob("*.rce")):
        if "_FIXTURE" in p.name:
            continue
        raw = p.read_text(encoding="utf-8", errors="surrogateescape")
        try:
            src, meta = rcetest.split(raw)
        except ValueError:
            continue
        exp = str(meta.get("expect"))
        is_accept = exp == "accept"
        is_fit = exp == "fit" or exp.startswith("W-")
        if (expect_accept and is_accept) or (not expect_accept and is_fit):
            if meta.get("fixtures"):
                continue  # a package, not a standalone program
            out.append((p, src, meta))
    return out


def mutate(src, rng):
    """One whitespace-token mutation (delete, duplicate or swap) of a source text; returns (kind, text)."""
    toks = [m for m in re.finditer(r"\S+|\n", src)]
    words = [i for i, m in enumerate(toks) if m.group(0) != "\n"]
    if len(words) < 2:
        return None
    kind = ["delete", "duplicate", "swap"][rng.next() % 3]
    i = words[rng.next() % len(words)]
    parts = [m.group(0) for m in toks]
    if kind == "delete":
        parts[i] = ""
    elif kind == "duplicate":
        parts[i] = parts[i] + " " + parts[i]
    else:
        j = words[rng.next() % len(words)]
        parts[i], parts[j] = parts[j], parts[i]
    out = []
    for m, t in zip(toks, parts):
        out.append(t if t == "\n" or not out or out[-1].endswith("\n") else " " + t)
    return kind, "".join(out)


KEYWORDS = set("""intent example state event error step on fn let var if elif else return emit fail try catch for in
while loop bound match with arena device struct kind trait impl pub use module type ghost requires ensures decreases
label atomic par spawn chan select recv send after break continue check and or not true false none some self comptime
extern actor supervisor node const by as mut then""".split())
OP_CLASSES = [["+", "-", "*", "/"], ["<", "<=", ">", ">="], ["==", "!="], ["and", "or"]]
TYPE_SWAP = {"u64": "i64", "i64": "u64", "text": "bool", "bool": "text", "f64": "i64"}


def segments(src):
    """(is_string, text) segments so that operators never touch string contents."""
    out, last = [], 0
    for m in STRING.finditer(src):
        if m.start() > last:
            out.append((False, src[last:m.start()]))
        out.append((True, m.group(0)))
        last = m.end()
    if last < len(src):
        out.append((False, src[last:]))
    return out


def occurrences(src, pattern):
    """Matches of `pattern` outside strings, as (segment index, match)."""
    segs = segments(src)
    out = []
    for k, (is_str, text) in enumerate(segs):
        if is_str:
            continue
        for m in re.finditer(pattern, text):
            out.append((k, m))
    return segs, out


def rebuild(segs, k, m, replacement):
    text = segs[k][1]
    segs = list(segs)
    segs[k] = (False, text[:m.start()] + replacement + text[m.end():])
    return "".join(t for _, t in segs)


def mutate_typed(src, rng, op):
    """One token-class mutation; returns the mutated text or None when the operator does not apply."""
    if op == "rename":
        segs, occ = occurrences(src, r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*(?![\w])")
        occ = [(k, m) for k, m in occ if m.group(0) not in KEYWORDS]
        names = sorted({m.group(0) for _, m in occ})
        if len(names) < 2 or not occ:
            return None
        k, m = occ[rng.next() % len(occ)]
        others = [n for n in names if n != m.group(0)]
        return rebuild(segs, k, m, others[rng.next() % len(others)])
    if op == "operator":
        segs, occ = occurrences(src, r"(?<![<>=!+\-*/&|-])(<=|>=|==|!=|<|>|\+|-(?![>=])|\*|/|\band\b|\bor\b)(?![=<>])")
        if not occ:
            return None
        k, m = occ[rng.next() % len(occ)]
        cls = next(c for c in OP_CLASSES if m.group(0) in c)
        others = [o for o in cls if o != m.group(0)]
        return rebuild(segs, k, m, others[rng.next() % len(others)])
    if op == "literal":
        segs, ints = occurrences(src, r"(?<![\w.])\d+(?![\w.])")
        strs = [(k, seg) for k, seg in enumerate(segs) if seg[0]]
        choices = [("int", x) for x in ints] + [("str", x) for x in strs]
        if not choices:
            return None
        kind, x = choices[rng.next() % len(choices)]
        if kind == "int":
            k, m = x
            return rebuild(segs, k, m, str(int(m.group(0)) + 1))
        k, (_, text) = x
        segs = list(segs)
        segs[k] = (True, text[:-1] + "x" + text[-1])
        return "".join(t for _, t in segs)
    if op == "guard":
        lines = src.split("\n")
        cands = [i for i, l in enumerate(lines) if re.match(r"\s*(if|elif|requires|ensures|fail)\b", l)]
        if not cands:
            return None
        i = cands[rng.next() % len(cands)]
        l = lines[i]
        if re.match(r"\s*(if|elif)\b", l):
            ind = re.match(r"\s*", l).group(0)
            kw = l.strip().split()[0]
            lines[i] = f"{ind}{kw} true"
        else:
            del lines[i]
        return "\n".join(lines)
    if op == "type":
        segs, occ = occurrences(src, r"\b(u64|i64|text|bool|f64)\b")
        if not occ:
            return None
        k, m = occ[rng.next() % len(occ)]
        return rebuild(segs, k, m, TYPE_SWAP[m.group(0)])
    return None


class Rng:
    def __init__(self, s):
        self.s = s & ((1 << 64) - 1)

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)


def first_code(text):
    m = re.search(r"\bW-[A-Z0-9]+-\d+\b", text)
    return m.group(0) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rce", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutants", type=int, default=3, help="whitespace-token mutants per accepted program")
    ap.add_argument("--typed-depth", type=int, default=2, help="mutants per typed operator per accepted program")
    ap.add_argument("--typed-depth-with-examples", type=int, default=2, help="the same for programs with example lines (whose mutants can be scored)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rce = Rce(a.rce)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip()
    runtimes = {t: (ROOT / "spec" / "checker" / "src" / "emit" / RUNTIME[t]).read_text() for t in TARGETS}

    cands = [(p, lic, origin, None) for p, lic, origin in population()]
    cands += [(p, "MIT OR Apache-2.0", "conformance-accept", src) for p, src, _ in conformance(True)]
    rows, fit = [], []
    tmpdir = out / ".tmp"
    tmpdir.mkdir(exist_ok=True)
    for p, lic, origin, src in cands:
        rel = p.relative_to(ROOT).as_posix()
        if src is None:
            src = p.read_text(encoding="utf-8")
            file = str(p)
        else:
            file = str(tmpdir / p.name)
            Path(file).write_text(src, encoding="utf-8")
        t0 = time.time()
        r = rce.run("check", file)
        check_s = round(time.time() - t0, 4)
        if r.returncode != 0:
            fit.append({"path": rel, "origin": origin, "why": (r.stdout + r.stderr).strip()[:300]})
            continue
        nameless = rce.run("fmt", "--nameless", file).stdout
        if not nameless.strip():
            fit.append({"path": rel, "origin": origin, "why": "no nameless projection"})
            continue
        row = {
            "format": "rce-corpus/2", "commit": commit, "path": rel, "origin": origin, "licence": lic,
            "source_sha256": sha(src), "shape_hash": sha(nameless), "intent": intent_block(src),
            "source": src, "nameless": nameless, "verdict": "accept",
            "form": {"tokens": len(tokens(nameless, "#")), "bytes": len(nameless.encode()), "gz_bits": gz_bits(nameless)},
            "code": {},
        }
        for t in TARGETS:
            e = rce.run("emit", t, file)
            if e.returncode != 0:
                row["code"][t] = {"error": (e.stdout + e.stderr).strip()[:200]}
                continue
            text = e.stdout
            prefix_ok = text.startswith(runtimes[t])
            prog = text[len(runtimes[t]):] if prefix_ok else text
            marker = "#" if t == "python" else "//"
            row["code"][t] = {"tokens": len(tokens(prog, marker)), "bytes": len(prog.encode()), "gz_bits": gz_bits(prog), "runtime_prefix_verbatim": prefix_ok}
        tr = rce.run("trace", "--json", file)
        if tr.returncode == 0:
            try:
                j = json.loads(tr.stdout)
                heads = [t["steps"][-1]["hash"] for t in j["traces"] if t["steps"]]
                row["trace_heads"] = heads
            except (ValueError, KeyError):
                row["trace_heads"] = []
        else:
            row["trace_heads"] = []
        row["cpu_s"] = round(time.time() - t0, 3)
        row["check_s"] = check_s
        rows.append(row)

    # near-duplicates: one row per shape hash (the first in path order)
    seen, uniq = set(), []
    for r in rows:
        if r["shape_hash"] in seen:
            r["duplicate_of_shape"] = True
            continue
        seen.add(r["shape_hash"])
        uniq.append(r)
    # vocabularies over the deduplicated population, then bits and rho
    vocab_form = set()
    vocab_code = {t: set() for t in TARGETS}
    for r in uniq:
        vocab_form.update(tokens(r["nameless"], "#"))
    # the vocabularies need the emitted texts again; re-emit (cheap) rather than hold them all
    for r in uniq:
        p = ROOT / r["path"] if r["origin"] != "conformance-accept" else tmpdir / Path(r["path"]).name
        for t in TARGETS:
            if "tokens" not in r["code"].get(t, {}):
                continue
            text = rce.run("emit", t, str(p)).stdout
            prog = text[len(runtimes[t]):] if text.startswith(runtimes[t]) else text
            vocab_code[t].update(tokens(prog, "#" if t == "python" else "//"))
    lvf = math.log2(len(vocab_form))
    lvc = {t: math.log2(len(vocab_code[t])) if vocab_code[t] else 0.0 for t in TARGETS}
    for r in uniq:
        r["form"]["bits"] = round(r["form"]["tokens"] * lvf, 1)
        for t in TARGETS:
            c = r["code"].get(t, {})
            if "tokens" in c:
                c["bits"] = round(c["tokens"] * lvc[t], 1)
                c["rho"] = round(c["bits"] / r["form"]["bits"], 3) if r["form"]["bits"] else None
                c["rho_bytes"] = round(c["bytes"] / r["form"]["bytes"], 3) if r["form"]["bytes"] else None
                c["rho_gz"] = round(c["gz_bits"] / r["form"]["gz_bits"], 3) if r["form"]["gz_bits"] else None
        h = int(r["shape_hash"][:16], 16) % 10
        r["split"] = "train" if h < 8 else ("validation" if h == 8 else "test")

    # negatives: the conformance tests that expect `fit`, and token mutants of the accepted programs
    negs = []
    for p, src, meta in conformance(False):
        negs.append({"format": "rce-corpus/2", "commit": commit, "path": p.relative_to(ROOT).as_posix(), "origin": "conformance-fit",
                     "licence": "MIT OR Apache-2.0", "source": src, "source_sha256": sha(src), "verdict": "fit",
                     "diagnostic": meta.get("diagnostic"), "clauses": meta.get("clauses"), "kind": "conformance"})
    rng = Rng(SEED)
    mut_stats = {"fit": 0, "accept": 0, "accept_same_shape": 0, "accept_equivalent": 0, "accept_behaviour_change": 0, "caught_by_examples": 0}
    scores = []

    def classify(r, text, kind, op):
        f = tmpdir / f"mut-{r['shape_hash'][:12]}-{op}-{rng.next() % 1000000}.rce"
        f.write_text(text, encoding="utf-8")
        c = rce.run("check", str(f))
        row = {"format": "rce-corpus/2", "commit": commit, "path": r["path"], "origin": r["origin"], "licence": r["licence"],
               "source": text, "source_sha256": sha(text), "kind": "mutant", "operator": op, "mutation": kind, "parent_shape_hash": r["shape_hash"]}
        if c.returncode != 0:
            msg = (c.stdout + c.stderr).strip()
            code = first_code(msg)
            row.update(verdict="fit", diagnostic=code, message=msg[:300],
                       outcome="caught_by_examples" if code == "W-INT-1" else "kept")
            mut_stats["fit"] += 1
            if code == "W-INT-1":
                mut_stats["caught_by_examples"] += 1
        else:
            nl = rce.run("fmt", "--nameless", str(f)).stdout
            same = sha(nl) == r["shape_hash"]
            row.update(verdict="accept", shape_hash=sha(nl), same_shape_as_parent=same)
            if same:
                row["outcome"] = "accept_same_shape"
            else:
                tr = rce.run("trace", "--json", str(f))
                heads = []
                if tr.returncode == 0:
                    try:
                        j = json.loads(tr.stdout)
                        heads = [t["steps"][-1]["hash"] for t in j["traces"] if t["steps"]]
                    except (ValueError, KeyError):
                        heads = []
                row["trace_heads"] = heads
                if not r["trace_heads"]:
                    row["outcome"] = "accept_no_examples"
                elif heads == r["trace_heads"]:
                    row["outcome"] = "accept_equivalent"
                else:
                    row["outcome"] = "accept_behaviour_change"
            mut_stats["accept"] += 1
            for k in ("accept_same_shape", "accept_equivalent", "accept_behaviour_change"):
                if row["outcome"] == k:
                    mut_stats[k] += 1
        f.unlink()
        return row

    for r in uniq:
        made = []
        for k in range(a.mutants):
            m = mutate(r["source"], rng)
            if m is not None:
                made.append(classify(r, m[1], m[0], "token"))
        depth = a.typed_depth_with_examples if r["trace_heads"] else a.typed_depth
        seen_mut = set()
        for op in ("rename", "operator", "literal", "guard", "type"):
            for _ in range(depth):
                t = mutate_typed(r["source"], rng, op)
                if t is not None and t != r["source"] and sha(t) not in seen_mut:
                    seen_mut.add(sha(t))
                    made.append(classify(r, t, op, op))
        negs.extend(made)
        if r["trace_heads"]:
            caught = sum(1 for x in made if x["outcome"] == "caught_by_examples")
            uncaught = sum(1 for x in made if x["outcome"] == "accept_behaviour_change")
            if caught + uncaught:
                scores.append({"path": r["path"], "caught": caught, "uncaught": uncaught, "score": round(caught / (caught + uncaught), 3)})
    for f in tmpdir.iterdir():
        f.unlink()
    tmpdir.rmdir()

    with open(out / "forms.jsonl", "w") as fh:
        for r in uniq:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    with open(out / "negatives.jsonl", "w") as fh:
        for r in negs:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")

    def med_iqr(xs):
        xs = sorted(x for x in xs if x is not None)
        if not xs:
            return None
        n = len(xs)
        q = lambda f: xs[min(n - 1, max(0, int(round(f * (n - 1)))))]
        return {"n": n, "median": q(0.5), "q1": q(0.25), "q3": q(0.75), "min": xs[0], "max": xs[-1]}

    summary = {
        "commit": commit, "seed": hex(SEED), "candidates": len(cands), "accepted": len(rows), "fit": len(fit),
        "unique_shapes": len(uniq), "duplicates": len(rows) - len(uniq),
        "by_origin": {o: sum(1 for r in uniq if r["origin"] == o) for o in sorted({r["origin"] for r in uniq})},
        "with_intent": sum(1 for r in uniq if r["intent"]),
        "split": {s: sum(1 for r in uniq if r["split"] == s) for s in ("train", "validation", "test")},
        "vocab": {"form": len(vocab_form), **{t: len(vocab_code[t]) for t in TARGETS}},
        "rho": {t: med_iqr([r["code"].get(t, {}).get("rho") for r in uniq]) for t in TARGETS},
        "rho_bytes": {t: med_iqr([r["code"].get(t, {}).get("rho_bytes") for r in uniq]) for t in TARGETS},
        "rho_gz": {t: med_iqr([r["code"].get(t, {}).get("rho_gz") for r in uniq]) for t in TARGETS},
        "form_tokens": med_iqr([r["form"]["tokens"] for r in uniq]),
        "negatives": {"conformance": sum(1 for n in negs if n["kind"] == "conformance"), **{"mutants_" + k: v for k, v in mut_stats.items()},
                      "mutants_by_operator": {op: {o: sum(1 for n in negs if n.get("operator") == op and n.get("outcome") == o) for o in sorted({n.get("outcome") for n in negs if n.get("operator") == op})} for op in ("token", "rename", "operator", "literal", "guard", "type")},
                      "mutant_diagnostics": {c: sum(1 for n in negs if n["kind"] == "mutant" and n.get("diagnostic") == c) for c in sorted({n.get("diagnostic") for n in negs if n["kind"] == "mutant" and n.get("diagnostic")})}},
        "intent_mutation_score": {"programs": len(scores), "median": (sorted(x["score"] for x in scores)[len(scores) // 2] if scores else None),
                                  "pooled_caught": sum(x["caught"] for x in scores), "pooled_uncaught": sum(x["uncaught"] for x in scores), "rows": scores},
        "check_s": med_iqr([r["check_s"] for r in uniq]),
        "fit_rows": fit, "rce_cpu_s": round(rce.cpu, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "fit_rows"}, indent=1))


if __name__ == "__main__":
    main()
