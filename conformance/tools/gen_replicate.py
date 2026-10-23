#!/usr/bin/env python3
"""Replicated-log tests (E.2b: HC-18, HC-18a, HC-19, HC-20; T31, T39, T41).
The expected committed log is derived from the script alone: exactly the
entries proposed through a leader that held a quorum, in proposal order;
nothing a leader proposed after losing its quorum is ever committed.
  gen_replicate.py TESTS_DIR SEED COUNT"""
import os, pathlib, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
import rcetest
out, seed, count = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rnd = random.Random(seed); d = out / "host-contract" / "generated"; d.mkdir(parents=True, exist_ok=True)
SRC = "# This test drives the replicated log of Appendix E.2b; it has no program.\n"
for k in range(count):
    n = rnd.choice([3, 5, 7]); a = [f"e{i}" for i in range(rnd.randint(1, 6))]
    b = [f"f{i}" for i in range(rnd.randint(0, 4))]
    stale = [f"stale{i}" for i in range(rnd.randint(1, 3))]
    minority = rnd.randint(1, (n - 1) // 2)
    script = [{"elect": 200}, {"propose": a}, {"run": 40}]
    deposed = rnd.random() < 0.7
    if deposed:
        script += [{"isolate_leader_with": minority}, {"propose_old": stale}, {"run": 120}, {"snapshot": True},
                   {"elect": 300}, {"propose": b}, {"run": 40}, {"heal": True}, {"run": 160}]
    else:
        f = rnd.randrange(n)
        script += [{"crash": f}, {"propose": b}, {"run": 60}, {"restart": f}, {"run": 160}]
    meta = {"id": f"host-contract/generated/replicate-{seed}-{k:03d}", "area": "host-contract",
            "clauses": ["HC-18", "HC-18a", "HC-19", "T41", "T31"] + (["HC-20", "T39"] if deposed else []),
            "expect": "replicate", "nodes": n, "script": script,
            "replicate_expect": {"committed_all": a + b, **({"deposed_frozen": True} if deposed else {})},
            "flags": ["generated"], "provenance": f"gen_replicate.py seed {seed}"}
    rcetest.write(d / f"replicate-{seed}-{k:03d}.rce", SRC, meta)
print(f"wrote {count}")
