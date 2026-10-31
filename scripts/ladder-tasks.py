#!/usr/bin/env python3
"""The fixed task set of the exact-proposer ladder (`rce repair`, spec 10.13).

    scripts/ladder-tasks.py [seed N] [per-op K] RCE OUT.jsonl

Population: every .rce file under spec/examples, programs and stdlib/tests
that `rce check` admits standalone, has at least one example or check, and
defines at least one function, step, actor, constant or state field outside
its intent, example and check lines. For each program, and for each of the
operators `operator`, `literal`, `guard` and `rename` of scripts/corpus.py
(in that order), up to K (default 4) mutants are drawn with the seeded generator of
scripts/corpus.py (default seed 0x5EED_C0DE_0001). A mutant is a task when

  - the line it changes is not an intent, example or check line (the
    mutation changes the program, never what it is judged by);
  - `rce check` reports it as requiring change only under W-INT-1 (an
    example or check does not hold); and
  - `rce equiv` does not judge it behaviour-equal to the original (exit 0).

Mutants equal to one already drawn for the same program are drawn once.
Each line of OUT.jsonl is one task with the original and the mutant text and
their SHA-256; the last line printed is the SHA-256 of OUT.jsonl. The run is
a pure function of the checkout, the seed and K.
"""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

spec = importlib.util.spec_from_file_location("corpus", Path(__file__).resolve().parent / "corpus.py")
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)

ROOT = Path(__file__).resolve().parent.parent
OPS = ["operator", "literal", "guard", "rename"]
DEFS = ("fn ", "pub fn ", "step ", "actor ", "const ", "pub const ", "state ", "impl ")


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def spec_lines(src):
    """0-based indices of intent, example and check lines (and an intent's indented block)."""
    out = set()
    lines = src.split("\n")
    in_intent = False
    for i, l in enumerate(lines):
        s = l.strip()
        if l.startswith("intent "):
            in_intent = True
            out.add(i)
            continue
        if in_intent and (l.startswith(" ") or s == ""):
            out.add(i)
            continue
        in_intent = False
        if s.startswith(("check ", "example ")):
            out.add(i)
    return out


def changed_lines(a, b):
    x, y = a.split("\n"), b.split("\n")
    if len(x) != len(y):
        p = 0
        while p < min(len(x), len(y)) and x[p] == y[p]:
            p += 1
        return {p}
    return {i for i in range(len(x)) if x[i] != y[i]}


def population():
    out = []
    for d, origin in [("spec/examples", "spec-examples"), ("programs", "programs"), ("stdlib/tests", "stdlib-tests")]:
        for p in sorted((ROOT / d).rglob("*.rce")):
            if "/refused/" in p.as_posix() or "/change/" in p.as_posix() or "_FIXTURE" in p.name:
                continue
            out.append((p, origin))
    return out


def main(argv):
    seed, per_op = corpus.SEED, 4
    i = 0
    while i < len(argv) and argv[i] in ("seed", "per-op"):
        if argv[i] == "seed":
            seed = int(argv[i + 1], 0)
        else:
            per_op = int(argv[i + 1])
        i += 2
    if len(argv) - i != 2:
        sys.exit("usage: scripts/ladder-tasks.py [seed N] [per-op K] RCE OUT.jsonl")
    rce, out = argv[i], Path(argv[i + 1])
    rng = corpus.Rng(seed)
    tmp = Path(tempfile.mkdtemp(prefix="ladder-tasks-"))
    stats = {"programs_seen": 0, "programs_admitted": 0, "programs_used": 0, "drawn": 0, "spec_line": 0, "admitted": 0,
             "other_rule": 0, "equivalent": 0, "tasks": 0}
    tasks = []
    for p, origin in population():
        stats["programs_seen"] += 1
        src = p.read_text(encoding="utf-8")
        sl = spec_lines(src)
        if not sl or not any(l.lstrip().startswith(DEFS) for k, l in enumerate(src.split("\n")) if k not in sl):
            continue
        f = tmp / p.name
        f.write_text(src, encoding="utf-8")
        r = subprocess.run([rce, "check", str(f)], capture_output=True, text=True)
        if r.returncode != 0:
            continue
        stats["programs_admitted"] += 1
        used = False
        seen = set()
        rel = p.relative_to(ROOT).as_posix()
        for op in OPS:
            for k in range(per_op):
                t = corpus.mutate_typed(src, rng, op)
                if t is None or t == src or sha(t) in seen:
                    continue
                seen.add(sha(t))
                stats["drawn"] += 1
                if changed_lines(src, t) & sl:
                    stats["spec_line"] += 1
                    continue
                g = tmp / ("mut-" + p.name)
                g.write_text(t, encoding="utf-8")
                c = subprocess.run([rce, "check", str(g)], capture_output=True, text=True)
                msg = (c.stdout + c.stderr).strip()
                if c.returncode == 0:
                    stats["admitted"] += 1
                    continue
                body = msg.split(": ", 1)[1] if ": " in msg else msg
                codes = set(x for x in __import__("re").findall(r"\bW-[A-Z0-9]+-\d+\b", body))
                if codes != {"W-INT-1"}:
                    stats["other_rule"] += 1
                    continue
                e = subprocess.run([rce, "equiv", "--fuel", "2000", str(f), str(g)], capture_output=True, text=True)
                if e.returncode == 0:
                    stats["equivalent"] += 1
                    continue
                used = True
                line = min(changed_lines(src, t)) + 1
                tasks.append({"format": "rce-ladder-task/1", "id": f"{rel}#{op}{k}", "path": rel, "origin": origin, "operator": op,
                              "line": line, "original_sha256": sha(src), "mutant_sha256": sha(t), "check": msg[:400],
                              "equiv_status": e.returncode, "original": src, "mutant": t})
        if used:
            stats["programs_used"] += 1
    stats["tasks"] = len(tasks)
    with open(out, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n")
    for x in tmp.iterdir():
        x.unlink()
    tmp.rmdir()
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    summary = {"commit": commit, "seed": hex(seed), "per_op": per_op, **stats, "tasks_sha256": h}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
