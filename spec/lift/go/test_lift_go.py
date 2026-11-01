#!/usr/bin/env python3
"""Tests of the Go lift probe (spec/lift/go/lift_go.py).

    python3 spec/lift/go/test_lift_go.py    run the tests (needs tree-sitter and tree-sitter-go)

door.go lifts to the committed door.lifted.rce byte for byte with exit 0.
fixtures/counter.go holds Go constructs the probe does not carry; the probe
prints no Form for it, exits 1, and its standard output is the committed
fixtures/counter.requires-change byte for byte: one line per construct, in
the form `requires change FILE: line N: RULE: reason`. tests/lift_go.rs
checks the committed files without Python.
"""
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "lift_go.py")
FIX = os.path.join(HERE, "fixtures")
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

    def test_usage(self):
        self.assertEqual(probe([], HERE).returncode, 2)
        self.assertEqual(probe(["door.go"], HERE).returncode, 2)


if __name__ == "__main__":
    unittest.main()
