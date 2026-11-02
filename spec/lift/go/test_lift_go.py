#!/usr/bin/env python3
"""Tests of the Go lift probe (spec/lift/go/lift_go.py).

    python3 spec/lift/go/test_lift_go.py    run the tests (needs tree-sitter and tree-sitter-go)

door.go lifts to the committed door.lifted.rce byte for byte with exit 0.
fixtures/counter.go holds Go constructs the probe does not carry; the probe
prints no Form for it, exits 1, and its standard output is the committed
fixtures/counter.requires-change byte for byte: one line per construct, in
the form `requires change FILE: line N: RULE: reason`. `report door.go` is
the committed door.report.json byte for byte (Appendix D.8, L5 and L6);
with the rce binary ($RCE, or spec/checker/target/release/rce, when it is
built) and the hand-written door, GO-EQUIV closes at `bisimilar`, and with a
counterpart that behaves otherwise it fails and the report exits 4.
tests/lift_go.rs and tests/lift_authority.rs check the committed files
without Python.
"""
import json
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "lift_go.py")
FIX = os.path.join(HERE, "fixtures")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RCE = os.environ.get("RCE", os.path.join(ROOT, "spec", "checker", "target", "release", "rce"))
HAND = os.path.join(ROOT, "spec", "examples", "61_dim_minimal.rce")
LINE = re.compile(r"^requires change (\S+): line (\d+): (L-GO-[A-Z]+): (.+)$")


def probe(args, cwd):
    return subprocess.run([sys.executable, PROBE] + args, capture_output=True, cwd=cwd)


class Probe(unittest.TestCase):
    def test_door_lifts_to_the_committed_form(self):
        p = probe(["lift", "door.go"], HERE)
        self.assertEqual(p.returncode, 0, p.stdout)
        with open(os.path.join(HERE, "door.lifted.rce"), "rb") as f:
            self.assertEqual(p.stdout, f.read())

    def test_constructs_not_carried_require_change(self):
        p = probe(["lift", "counter.go"], FIX)
        self.assertEqual(p.returncode, 1, "a program with a construct the probe does not carry requires change")
        with open(os.path.join(FIX, "counter.requires-change"), "rb") as f:
            self.assertEqual(p.stdout, f.read())
        rules = {l.split(":")[0] for l in probe(["rules"], HERE).stdout.decode().splitlines()}
        for line in p.stdout.decode().splitlines():
            m = LINE.match(line)
            self.assertIsNotNone(m, line)
            self.assertIn(m.group(3), rules)

    def test_door_report_is_the_committed_one(self):
        p = probe(["report", "door.go"], HERE)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(os.path.join(HERE, "door.report.json"), "rb") as f:
            self.assertEqual(p.stdout, f.read())
        rep = json.loads(p.stdout)
        self.assertEqual([(c["id"], c["requires"], c["verdict"]) for c in rep["claims"]],
                         [("GO-ROWS", "derived", "closed"), ("GO-AGREE", "replay", "open"), ("GO-EQUIV", "bisimilar", "open")])

    def test_a_program_that_requires_change_has_no_report(self):
        p = probe(["report", "counter.go"], FIX)
        self.assertEqual(p.returncode, 1)
        with open(os.path.join(FIX, "counter.requires-change"), "rb") as f:
            self.assertEqual(p.stdout, f.read())

    @unittest.skipUnless(os.path.exists(RCE), "the rce binary is not built")
    def test_equiv_evidence(self):
        p = probe(["report", "door.go", RCE, HAND], HERE)
        self.assertEqual(p.returncode, 0, p.stdout)
        c = {x["id"]: x for x in json.loads(p.stdout)["claims"]}["GO-EQUIV"]
        self.assertEqual((c["verdict"], c["reason"]), ("closed", "rce equiv: bisimilar"))
        self.assertEqual(c["evidence"][0]["bindings"], c["bindings"])
        other = os.path.join(ROOT, "spec", "examples", "17_traffic_light.rce")
        p = probe(["report", "door.go", RCE, other], HERE)
        self.assertEqual(p.returncode, 4, "a counterpart that behaves otherwise fails GO-EQUIV")
        c = {x["id"]: x for x in json.loads(p.stdout)["claims"]}["GO-EQUIV"]
        self.assertEqual(c["verdict"], "failed")

    def test_usage(self):
        self.assertEqual(probe([], HERE).returncode, 2)
        self.assertEqual(probe(["door.go"], HERE).returncode, 2)
        self.assertEqual(probe(["report", "door.go", "rce"], HERE).returncode, 2)


if __name__ == "__main__":
    unittest.main()
