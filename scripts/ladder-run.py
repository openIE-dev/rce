#!/usr/bin/env python3
"""Run the exact-proposer ladder (`rce repair`, spec 10.13) over a task set.

    scripts/ladder-run.py [bound N] RCE TASKS.jsonl OUTDIR

For each task of TASKS.jsonl (written by scripts/ladder-tasks.py) the
mutant is written to a file and `rce repair json [bound N] FILE` runs on it.
Each closed repair is then compared with the original program:

  - restored: `rce hash` gives the repair and the original the same program
    hash (the same definitions up to layout, so the same behaviour);
  - behaviour-equal: `rce equiv` exits 0 (the same behaviour on the
    explored alphabet, 10.11);
  - examples only: `rce equiv` exits 1 (a distinguishing event sequence
    exists): the repair passes every example and check, but it is not the
    original program;
  - undecided: `rce equiv` exits 3 (its fuel bound was reached);
  - not compared: `rce equiv` cannot compare the two (no Form to explore,
    for a program of functions only); passing the examples is all that is
    known.

OUTDIR/runs.jsonl holds one row per task (with every attempt the ladder
logged); OUTDIR/summary.json and OUTDIR/REPORT.md hold the totals: tasks
closed per rung, the share closed with zero model calls, candidates tried,
wall time, and the equivalence split (also per mutation operator: tasks,
closed, restored). No model is called by this script or
by `rce repair`.
"""
import hashlib, json, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# the rungs of 10.13, in order: a reuse, b lookup by type, c altered rules,
# d bounded enumeration, e renames
RUNGS = ("a", "b", "c", "d", "e")


def program_hash(rce, f):
    r = subprocess.run([rce, "hash", str(f)], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.endswith("  program")]
    return lines[-1].split()[0] if r.returncode == 0 and lines else None


def main(argv):
    bound = None
    i = 0
    if argv[:1] == ["bound"]:
        bound = int(argv[1])
        i = 2
    if len(argv) - i != 3:
        sys.exit("usage: scripts/ladder-run.py [bound N] RCE TASKS.jsonl OUTDIR")
    rce, tasks_path, outdir = argv[i], Path(argv[i + 1]), Path(argv[i + 2])
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in tasks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    tmp = Path(tempfile.mkdtemp(prefix="ladder-run-"))
    rows = []
    t_all = time.time()
    for t in tasks:
        name = Path(t["path"]).name
        orig = tmp / ("orig-" + name)
        mut = tmp / name
        orig.write_text(t["original"], encoding="utf-8")
        mut.write_text(t["mutant"], encoding="utf-8")
        cmd = [rce, "repair", "json"] + (["bound", str(bound)] if bound is not None else []) + [str(mut)]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        try:
            rep = json.loads(r.stdout)
        except ValueError:
            rep = {"closed": False, "rung": None, "tried": {}, "verdict": (r.stdout + r.stderr)[:300], "attempts": []}
        row = {"id": t["id"], "path": t["path"], "operator": t["operator"], "exit": r.returncode, "wall_s": round(wall, 3),
               "closed": rep.get("closed", False), "rung": rep.get("rung"), "tried": rep.get("tried", {}),
               "edit": rep.get("edit"), "diff": rep.get("diff"), "verdict": rep.get("verdict"),
               "model_calls": rep.get("model_calls", 0), "attempts": rep.get("attempts", [])}
        if row["closed"] and rep.get("repaired"):
            fixed = tmp / ("fixed-" + name)
            fixed.write_text(rep["repaired"], encoding="utf-8")
            ho, hf = program_hash(rce, orig), program_hash(rce, fixed)
            if ho and ho == hf:
                row["equivalence"] = "restored"
            else:
                e = subprocess.run([rce, "equiv", "fuel", "2000", str(orig), str(fixed)], capture_output=True, text=True)
                row["equiv_exit"] = e.returncode
                row["equivalence"] = {0: "behaviour-equal", 1: "examples only", 3: "undecided"}.get(e.returncode, "not compared")
                if row["equivalence"] == "not compared":
                    row["equiv_note"] = (e.stdout + e.stderr).strip()[:200]
            fixed.unlink()
        rows.append(row)
        orig.unlink()
        mut.unlink()
        print(f"{t['id']}: {'closed ' + str(row['rung']) if row['closed'] else 'not closed'}"
              f" {row.get('equivalence', '')} ({sum(row['tried'].values())} tried, {row['wall_s']} s)", flush=True)
    tmp.rmdir()
    wall_all = time.time() - t_all
    n = len(rows)
    closed = [r for r in rows if r["closed"]]
    per_rung = {k: sum(1 for r in closed if r["rung"] == k) for k in RUNGS}
    eq = {}
    for r in closed:
        eq[r["equivalence"]] = eq.get(r["equivalence"], 0) + 1
    by_op = {}
    for r in rows:
        o = by_op.setdefault(r["operator"], {"tasks": 0, "closed": 0, "restored": 0})
        o["tasks"] += 1
        o["closed"] += r["closed"]
        o["restored"] += r.get("equivalence") == "restored"
    tried = {k: sum(r["tried"].get(k, 0) for r in rows) for k in RUNGS}
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rce_sha = hashlib.sha256(Path(rce).read_bytes()).hexdigest()
    summary = {
        "format": "rce-ladder-summary/1", "commit": commit, "rce_sha256": rce_sha, "bound": bound if bound is not None else 2000,
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(), "tasks": n, "closed": len(closed),
        "not_closed": n - len(closed), "closed_per_rung": per_rung,
        "share_closed_zero_model_calls": round(len(closed) / n, 4) if n else None,
        "model_calls": sum(r["model_calls"] for r in rows), "equivalence": eq, "by_operator": by_op,
        "candidates_tried": tried, "candidates_tried_total": sum(tried.values()),
        "wall_s_total": round(wall_all, 1), "wall_s_repair_sum": round(sum(r["wall_s"] for r in rows), 1),
        "wall_s_median": sorted(r["wall_s"] for r in rows)[n // 2] if n else None,
    }
    with open(outdir / "runs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    rep = [f"# Ladder run: {n} tasks", "",
           f"Checker commit `{commit}`, `rce` SHA-256 `{rce_sha}`, bound {summary['bound']} candidates for each of rungs d and e, task set SHA-256 `{summary['tasks_sha256']}`.", "",
           "| Rung | Closed |", "|---|---:|"]
    names = {"a": "a reuse", "b": "b lookup by type", "c": "c altered rules", "d": "d bounded enumeration", "e": "e renames"}
    for k in RUNGS:
        rep.append(f"| {names[k]} | {per_rung[k]} |")
    rep += [f"| not closed | {n - len(closed)} |", "",
            f"Closed with zero model calls: {len(closed)} of {n} ({100 * len(closed) / n:.1f}%)." if n else "", "",
            "| Closed repair compared with the original | Tasks |", "|---|---:|"]
    for k in ("restored", "behaviour-equal", "examples only", "undecided", "not compared"):
        rep.append(f"| {k} | {eq.get(k, 0)} |")
    rep += ["", "| Mutation operator | Tasks | Closed | Restored |", "|---|---:|---:|---:|"]
    for k, v in sorted(by_op.items()):
        rep.append(f"| {k} | {v['tasks']} | {v['closed']} | {v['restored']} |")
    rep += ["", f"Candidates judged by the checker: {', '.join(f'{k} {v}' for k, v in tried.items())} (total {sum(tried.values())}).",
            f"Wall time: {summary['wall_s_total']} s in all, median {summary['wall_s_median']} s per task.", ""]
    (outdir / "REPORT.md").write_text("\n".join(rep))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
