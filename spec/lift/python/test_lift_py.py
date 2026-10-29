#!/usr/bin/env python3
"""Tests of the Python lift (spec/lift/python/lift_py.py).

    python3.13 spec/lift/python/test_lift_py.py          run the tests
    python3.13 spec/lift/python/test_lift_py.py record   write fixtures/pos/*.lifted.rce from the lift
                                                         and fixtures/pos/*.stdout from CPython

The positive fixtures lift, the lifted text is the committed one byte for
byte, `rce check` accepts it and `rce run` prints exactly the standard
output recorded from CPython 3.13 under the oracle's pinned environment
(lang/python3.13/oracle/cpython-3.13.16.toml). Every negative fixture
requires change with the rules fixtures/neg/EXPECTED names, one line per
construct in the form `requires change FILE: line N: RULE: reason`. The
`rce` binary is $RCE, or spec/checker/target/release/rce. The test of the
recorded output against CPython itself runs when /usr/bin/python3.13 is
present. tests/lift_py.rs checks the committed lifted files without Python.
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lift_py  # noqa: E402
import verify_lift  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RCE = os.environ.get("RCE", os.path.join(ROOT, "spec", "checker", "target", "release", "rce"))
POS = os.path.join(HERE, "fixtures", "pos")
NEG = os.path.join(HERE, "fixtures", "neg")
ORACLE = "/usr/bin/python3.13"
ORACLE_ENV = {"LC_ALL": "C.UTF-8", "TZ": "UTC", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
LINE = re.compile(r"^requires change (\S+): line (\d+): (L-PY-[A-Z]+): (.+)$")


def pos_names():
    return sorted(f[:-3] for f in os.listdir(POS) if f.endswith(".py"))


def oracle_stdout(path):
    with tempfile.TemporaryDirectory() as d:
        env = dict(ORACLE_ENV, PYTHONPYCACHEPREFIX=os.path.join(d, "no-bytecode-cache"))
        src = os.path.join(d, "main.py")
        with open(path, "rb") as f, open(src, "wb") as g:
            g.write(f.read())
        p = subprocess.run([ORACLE, "-P", "-s", "-S", src], capture_output=True, env=env, cwd=d)
        assert p.returncode == 0 and not p.stderr, (path, p.returncode, p.stderr)
        return p.stdout


def lift_file(path):
    with open(path, "rb") as f:
        return lift_py.lift(os.path.basename(path), f.read())


class Positive(unittest.TestCase):
    def test_fixtures_lift_and_match_cpython(self):
        for n in pos_names():
            with self.subTest(fixture=n):
                text, diags, _ = lift_file(os.path.join(POS, n + ".py"))
                self.assertEqual(diags, [])
                with open(os.path.join(POS, n + ".lifted.rce"), encoding="utf-8") as f:
                    self.assertEqual(text, f.read(), "the lift is the committed text byte for byte")
                path = os.path.join(POS, n + ".lifted.rce")
                chk = subprocess.run([RCE, "check", path], capture_output=True)
                self.assertEqual(chk.returncode, 0, chk.stdout)
                status, out, cls, stopped = verify_lift.observe(RCE, path)
                self.assertIsNone(stopped)
                with open(os.path.join(POS, n + ".stdout"), "rb") as f:
                    self.assertEqual(out, f.read())

    @unittest.skipUnless(os.path.exists(ORACLE), "CPython 3.13 is not at /usr/bin/python3.13")
    def test_recorded_stdout_is_cpythons(self):
        for n in pos_names():
            with self.subTest(fixture=n):
                with open(os.path.join(POS, n + ".stdout"), "rb") as f:
                    self.assertEqual(oracle_stdout(os.path.join(POS, n + ".py")), f.read())

    def test_lift_is_deterministic(self):
        for n in pos_names():
            a = lift_file(os.path.join(POS, n + ".py"))[0]
            b = lift_file(os.path.join(POS, n + ".py"))[0]
            self.assertEqual(a, b)


class Negative(unittest.TestCase):
    def expected(self):
        out = {}
        with open(os.path.join(NEG, "EXPECTED"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    name, rules = line.split(":", 1)
                    out[name.strip()] = set(rules.split())
        return out

    def test_every_negative_fixture_requires_change_with_its_rules(self):
        exp = self.expected()
        names = sorted(f for f in os.listdir(NEG) if f.endswith(".py"))
        self.assertEqual(sorted(exp), names, "fixtures/neg/EXPECTED names every negative fixture")
        for n in names:
            with self.subTest(fixture=n):
                p = subprocess.run([sys.executable, os.path.join(HERE, "lift_py.py"), "lift", os.path.join(NEG, n)],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 1)
                lines = p.stdout.strip().split("\n")
                rules = set()
                for line in lines:
                    m = LINE.match(line)
                    self.assertIsNotNone(m, line)
                    rules.add(m.group(3))
                    self.assertIn(m.group(3), lift_py.RULES)
                self.assertEqual(rules, exp[n])

    def test_function_level_coverage(self):
        src = (b"def area(w, h):\n    return w * h\n\n\ndef first(xs):\n    return xs[0]\n\n\n"
               b"def mean(xs):\n    return sum(xs) / len(xs)\n\n\nprint(area(3, 4))\n")
        res = {n: d for n, _, d in lift_py.lift_functions("m.py", src)}
        self.assertEqual(res["area"], [])
        self.assertEqual({d[1] for d in res["first"]}, {"L-PY-POLY"})
        self.assertIn("L-PY-FLOAT", {d[1] for d in res["mean"]})


def record():
    for n in pos_names():
        text, diags, _ = lift_file(os.path.join(POS, n + ".py"))
        assert not diags, (n, diags)
        with open(os.path.join(POS, n + ".lifted.rce"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(POS, n + ".stdout"), "wb") as f:
            f.write(oracle_stdout(os.path.join(POS, n + ".py")))
        print("recorded", n)


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()
    else:
        unittest.main()
