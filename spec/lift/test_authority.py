#!/usr/bin/env python3
"""Tests of the evidence authority rule (spec/lift/authority.py; Appendix
D.8, L5 and L6).

    python3 spec/lift/test_authority.py    run the tests

Every vector of authority-vectors.json gets its verdict and reason from
authority.close; spec/checker/tests/lift_authority.rs checks the same
vectors without Python.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import authority  # noqa: E402


class Vectors(unittest.TestCase):
    def test_every_vector_gets_its_verdict(self):
        t = authority.load()
        with open(os.path.join(HERE, "authority-vectors.json"), encoding="utf-8") as f:
            vs = json.load(f)["vectors"]
        for x in vs:
            with self.subTest(vector=x["name"]):
                got = authority.close(x["claim"], x["bindings"], x["records"], t, x["deps"])
                self.assertEqual(got, (x["verdict"], x["reason"]))

    def test_inferred_closes_nothing(self):
        t = authority.load()
        inferred = next(l for l in t["levels"] if l["authority"] == "inferred")
        self.assertEqual(inferred["closes"], [])
        for l in t["levels"]:
            for req in l["closes"]:
                b = {"source": "a" * 64}
                c = {"id": "C", "revision": 1, "requires": req, "binds": ["source"]}
                rec = {"claim": "C", "revision": 1, "verifier": "v", "authority": l["authority"], "kind": "inferred",
                       "result": "holds", "bindings": b, "detail": ""}
                self.assertEqual(authority.close(c, b, [rec], t)[0], "undecided", (l["authority"], req))
                rec["kind"] = l["kind"]
                self.assertEqual(authority.close(c, b, [rec], t)[0], "closed", (l["authority"], req))
                rec["bindings"] = {"source": "0" * 64}
                self.assertEqual(authority.close(c, b, [rec], t)[0], "failed", (l["authority"], req))


if __name__ == "__main__":
    unittest.main()
