#!/usr/bin/env python3
"""The Form corpus (RCE-JEPA workstream W1) and the H1 measurement.

Walks the hand-written programs of this repository, keeps the ones `rce
check` accepts standalone, and writes one row per program with its intent
block, nameless canonical source, shape hash, verdict, the size of the Form
and of each emitted program, and rho = b(code) / b(P) as preregistered in
openie-research-experiments/rce/bench/form-space/PREREG.json. Negatives are
the conformance tests that must be rejected (with their diagnostic code) and
checker-verified token mutants of the accepted programs under a fixed seed.

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
            if "/refused/" in p.as_posix() or "_FIXTURE" in p.name:
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
        is_reject = exp == "reject" or exp.startswith("W-")
        if (expect_accept and is_accept) or (not expect_accept and is_reject):
            if meta.get("fixtures"):
                continue  # a package, not a standalone program
            out.append((p, src, meta))
    return out


def mutate(src, rng):
    """One token-level mutation (delete, duplicate or swap) of a source text; returns (kind, text)."""
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
    ap.add_argument("--mutants", type=int, default=3, help="token mutants per accepted program")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rce = Rce(a.rce)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip()
    runtimes = {t: (ROOT / "spec" / "checker" / "src" / "emit" / RUNTIME[t]).read_text() for t in TARGETS}

    cands = [(p, lic, origin, None) for p, lic, origin in population()]
    cands += [(p, "MIT OR Apache-2.0", "conformance-accept", src) for p, src, _ in conformance(True)]
    rows, refused = [], []
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
        if r.returncode != 0:
            refused.append({"path": rel, "origin": origin, "why": (r.stdout + r.stderr).strip()[:300]})
            continue
        nameless = rce.run("fmt", "--nameless", file).stdout
        if not nameless.strip():
            refused.append({"path": rel, "origin": origin, "why": "no nameless projection"})
            continue
        row = {
            "format": "rce-corpus/1", "commit": commit, "path": rel, "origin": origin, "licence": lic,
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

    # negatives: conformance rejects, and token mutants of the accepted programs
    negs = []
    for p, src, meta in conformance(False):
        negs.append({"format": "rce-corpus/1", "commit": commit, "path": p.relative_to(ROOT).as_posix(), "origin": "conformance-reject",
                     "licence": "MIT OR Apache-2.0", "source": src, "source_sha256": sha(src), "verdict": "reject",
                     "diagnostic": meta.get("diagnostic"), "clauses": meta.get("clauses"), "kind": "conformance"})
    rng = Rng(SEED)
    mut_stats = {"reject": 0, "accept": 0, "accept_same_shape": 0}
    for r in uniq:
        for k in range(a.mutants):
            m = mutate(r["source"], rng)
            if m is None:
                continue
            kind, text = m
            f = tmpdir / f"mut-{r['shape_hash'][:12]}-{k}.rce"
            f.write_text(text, encoding="utf-8")
            c = rce.run("check", str(f))
            if c.returncode != 0:
                msg = (c.stdout + c.stderr).strip()
                negs.append({"format": "rce-corpus/1", "commit": commit, "path": r["path"], "origin": r["origin"], "licence": r["licence"],
                             "source": text, "source_sha256": sha(text), "verdict": "reject", "diagnostic": first_code(msg),
                             "message": msg[:300], "kind": "mutant", "mutation": kind, "parent_shape_hash": r["shape_hash"]})
                mut_stats["reject"] += 1
            else:
                nl = rce.run("fmt", "--nameless", str(f)).stdout
                same = sha(nl) == r["shape_hash"]
                mut_stats["accept_same_shape" if same else "accept"] += 1
                negs.append({"format": "rce-corpus/1", "commit": commit, "path": r["path"], "origin": r["origin"], "licence": r["licence"],
                             "source": text, "source_sha256": sha(text), "verdict": "accept", "kind": "mutant", "mutation": kind,
                             "parent_shape_hash": r["shape_hash"], "shape_hash": sha(nl), "same_shape_as_parent": same})
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
        "commit": commit, "seed": hex(SEED), "candidates": len(cands), "accepted": len(rows), "refused": len(refused),
        "unique_shapes": len(uniq), "duplicates": len(rows) - len(uniq),
        "by_origin": {o: sum(1 for r in uniq if r["origin"] == o) for o in sorted({r["origin"] for r in uniq})},
        "with_intent": sum(1 for r in uniq if r["intent"]),
        "split": {s: sum(1 for r in uniq if r["split"] == s) for s in ("train", "validation", "test")},
        "vocab": {"form": len(vocab_form), **{t: len(vocab_code[t]) for t in TARGETS}},
        "rho": {t: med_iqr([r["code"].get(t, {}).get("rho") for r in uniq]) for t in TARGETS},
        "rho_bytes": {t: med_iqr([r["code"].get(t, {}).get("rho_bytes") for r in uniq]) for t in TARGETS},
        "rho_gz": {t: med_iqr([r["code"].get(t, {}).get("rho_gz") for r in uniq]) for t in TARGETS},
        "form_tokens": med_iqr([r["form"]["tokens"] for r in uniq]),
        "negatives": {"conformance": sum(1 for n in negs if n["kind"] == "conformance"), **{"mutants_" + k: v for k, v in mut_stats.items()}},
        "refused_rows": refused, "rce_cpu_s": round(rce.cpu, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "refused_rows"}, indent=1))


if __name__ == "__main__":
    main()
