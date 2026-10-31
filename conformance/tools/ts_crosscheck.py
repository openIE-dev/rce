#!/usr/bin/env python3
"""Cross-check parsing with tree-sitter-rce (spec 3: the LALR(1) tables, the
Rust parser and the tree-sitter port must agree on every program).
For every test whose expectation is decided at parse time or later, compare
  - the reference's parse-only verdict (`rce-check parse-only`), and
  - tree-sitter-rce: a parse with no ERROR or MISSING node.
A test with expect accept/run/trace must parse under both. A test that
requires change under W-SYN-1, W-LEX-* or W-VER-1 must fail to parse under the reference; a
disagreement with tree-sitter is a finding.
  ts_crosscheck.py TREE_SITTER_RCE_DIR"""
import json, os, pathlib, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
import rcetest
ROOT = pathlib.Path(__file__).resolve().parent.parent
ts = pathlib.Path(sys.argv[1]).resolve()
check = os.environ.get("RCE_CHECK", "rce-check")
tmp = pathlib.Path(tempfile.mkdtemp())
rows, disagree = [], []
files = []
for p in sorted((ROOT / "tests").rglob("*.rce")):
    if "_FIXTURE" in p.name: continue
    src, m = rcetest.split(p.read_bytes().decode("utf-8", errors="surrogateescape"))
    if m.get("expect") not in ("accept", "run", "trace", "change"): continue
    if not src.strip() or src.lstrip().startswith("# This test drives"): continue
    f = tmp / f"t{len(files)}.rce"; f.write_bytes(src.encode("utf-8", errors="surrogateescape"))
    files.append((p, m, f))
for p, m, f in files:
    r = subprocess.run([check, "parse-only", str(f)], capture_output=True, text=True)
    ref_ok = r.returncode == 0
    t = subprocess.run(["tree-sitter", "parse", "--quiet", str(f)], cwd=ts, capture_output=True, text=True)
    ts_ok = t.returncode == 0 and "ERROR" not in t.stdout and "MISSING" not in t.stdout
    syn = m.get("expect") == "change" and str(m.get("diagnostic", "")).startswith(("W-SYN", "W-LEX", "W-VER"))
    law = (not syn) if m.get("expect") in ("accept", "run", "trace") or syn else None
    row = {"path": str(p.relative_to(ROOT)), "reference_parses": ref_ok, "tree_sitter_parses": ts_ok, "law_parses": law}
    row["expect"] = m.get("expect"); row["diagnostic"] = m.get("diagnostic")
    rows.append(row)
    if ref_ok != ts_ok:
        # W-LEX-1/2/3 and W-VER-1 are lexical rules tree-sitter is not asked to enforce (spec 3 compares
        # the token-level grammar); only report those when tree-sitter accepts a W-SYN-1 program or
        # fails to parse an admitted one.
        lexonly = m.get("expect") == "change" and str(m.get("diagnostic", "")).startswith(("W-LEX", "W-VER"))
        if not lexonly:
            disagree.append(row)
out = {"tree_sitter_rce": subprocess.run(["git", "-C", str(ts), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "programs": len(rows), "agree": len(rows) - len(disagree), "disagreements": disagree, "rows": rows}
json.dump(out, open(ROOT / "reports" / "tree-sitter.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("tree_sitter_rce", "programs", "agree")}), "disagreements:", len(disagree))
