#!/usr/bin/env python3
"""`for ... par` reductions on the normative fixed tree (C.8, T18).

For m >= 2 contributions the left subtree takes the first h, h the largest
power of two below m; one contribution is itself; none is the identity.
Float addition is not associative, so the summands are chosen so that the
bits depend on the tree. The oracle is this definition, evaluated in
binary64 by Python; the reference is only asked whether it agrees.
  gen_tree.py OUT_DIR
"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
sys.path.insert(0, os.path.dirname(__file__))
import rcetest
from refclient import Ref
from gen_numbers import ftext

def tree(xs, ident, op):
    if not xs: return ident
    if len(xs) == 1: return xs[0]
    h = 1
    while h * 2 < len(xs): h *= 2
    return op(tree(xs[:h], ident, op), tree(xs[h:], ident, op))

CASES = {
    # body expression in i, its Python value, operator, identity
    "sum": ("1.0 / ((i as f64) + 1.0) + (if i % 3 == 0 then 1e16 else -1e16 + 3.0)",
            lambda i: 1.0 / (float(i) + 1.0) + (1e16 if i % 3 == 0 else -1e16 + 3.0), "+=", 0.0, lambda a, b: a + b),
    "prod": ("1.0 + 1.0 / ((i as f64) + 7.0)",
             lambda i: 1.0 + 1.0 / (float(i) + 7.0), "*=", 1.0, lambda a, b: a * b),
}

def main():
    out = sys.argv[1]; ref = Ref(); findings = []; n = 0
    d = os.path.join(out, "concurrency", "generated"); os.makedirs(d, exist_ok=True)
    for name, (body, f, op, ident, pyop) in CASES.items():
        for m in list(range(0, 41)) + [63, 64, 65, 100]:
            want = pyop(ident, tree([f(i) for i in range(m)], ident, pyop))
            init = "1.0" if op == "*=" else "0.0"
            src = f"# rce-version: 0.6\nfn main()\n    var s: f64 = {init}\n    for i in 0..{m} par\n        s {op} {body}\n    emit s\n"
            outs = ["emit " + ftext(want)]
            r = ref.run(src)
            if r.get("verdict") == "fit": sys.exit(f"generator wrote a program the checker reports: {r}\n{src}")
            got = (r.get("outputs"), r.get("outcome"))
            if got != (outs, "done"):
                findings.append({"case": src, "reference": got, "oracle": outs, "kind": "tree"})
            tid = f"tree-{name}-{m:03d}"
            rcetest.write(os.path.join(d, tid + ".rce"), src, {"id": f"concurrency/generated/{tid}", "area": "concurrency",
                "clauses": ["CH14-M-ca5a98d7", "T18", "HC-12", "W-PAR-3"], "expect": "run", "outcome": "done", "outputs": outs,
                "flags": ["generated"], "provenance": "gen_tree.py; oracle is the C.8 tree definition in binary64"})
            n += 1
    json.dump(findings, open(os.path.join(out, "..", "findings", "tree.json"), "w"), indent=1)
    print(json.dumps({"generated": n, "disagreements": len(findings)}))

if __name__ == "__main__":
    main()
