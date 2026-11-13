#!/usr/bin/env python3
"""Regression tests for the production acceptance gate.

The green fixture is test-only. It is not a claim about the repository.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import accept  # noqa: E402

REV = "a" * 40
OTHER = "b" * 40
COMPARED = ["outputs", "state", "refusal"]


def statements():
    doc = accept.load_json(accept.OBLIGATIONS_PATH)
    return [{"id": row["id"], "statement": row["statement"]} for row in doc["obligations"]]


def green_module(ident="case.rce"):
    return {
        "id": ident,
        "path": ident,
        "admitted": True,
        "status": "supported",
        "floor_tag": None,
        "program_functions": {"class": "syntactic", "roots": 1, "tag": None},
        "whole_module": {
            "status": "bytes-equal",
            "tag": None,
            "bytes_equal": True,
            "results_equal": True,
            "compared": list(COMPARED),
        },
        "output_semantics": {"hard": True, "status": "equal", "compared": list(COMPARED)},
    }


def equal_case(ident, exit_equal=True):
    return {
        "module": ident,
        "status": "equal",
        "exit_equal": exit_equal,
        "outputs_equal": True,
        "state_equal": True,
        "refusal_equal": True,
        "compared": list(COMPARED),
        "executed": True,
    }


def green_evidence(modules=None):
    modules = modules or [green_module()]
    ids = [m["id"] for m in modules]
    proofs = []
    for oid, _normative, needs in accept.REQUIRED:
        if not needs:
            continue
        proofs.append({
            "obligation": oid,
            "theorem": oid.replace("-", "_"),
            "path": "model.lean",
        })
    evidence = {
        "format": "rce-prod-evidence/1",
        "label": "test-only",
        "corpus_kind": "test-only",
        "revision": REV,
        "built_from": "",
        "producer": "test-only",
        "gate_finishes_compiler": False,
        "denominator_modules": len(modules),
        "admitted_programs": accept.recompute_floor(modules)["admitted_programs"],
        "obligations_sha256": accept.sha256_file(accept.OBLIGATIONS_PATH),
        "modules": modules,
        "floor": accept.recompute_floor(modules),
        "refusals": [],
        "backends": {
            name: {"cases": [equal_case(i) for i in ids]}
            for name in ("form", "scheduled", "core")
        },
        "equivalence": {"modules": [equal_case(i) for i in ids]},
        "constructs": {
            name: {"status": "supported", "modules": 0, "counted_as_support": False}
            for name in ("break", "catch", "pure-count")
        },
        "metering": {
            "semantic_budget": {
                "kind": "semantic",
                "fields": list(accept.SEMANTIC_FIELDS),
                "agrees": True,
            },
            "measured_joules": None,
        },
        "proofs": {
            "statements": statements(),
            "model_proofs": proofs,
            "tests": [{
                "id": "examples_are_covered_or_require_change",
                "path": "spec/checker/tests/form_bridge.rs",
                "discharges": [],
            }],
            "unproved_obligations": ["measured-joules"],
        },
    }
    return accept.seal(evidence)


def green_claim():
    return {
        "format": "rce-prod-claim/1",
        "revision": REV,
        "kind": "production",
        "subject": "test-only",
        "surface": {
            "includes_output_semantics": True,
            "includes_whole_module_form": True,
            "includes_effects": True,
            "backends": ["form", "scheduled", "core"],
        },
        "asserts": {
            "production_compiler": True,
            "whole_module_ready": True,
            "function_floor_implies_modules": False,
            "measured_joules": False,
            "backends_ready": ["form", "scheduled", "core"],
            "syntactic_program_examples": 1,
            "whole_modules_constructed": 1,
            "denominator_modules": 1,
            "proved": [oid for oid, _n, needs in accept.REQUIRED if needs],
        },
        "scope": {"text": "", "excludes": []},
    }


def drop_equal_actions(evidence, status):
    """An action row may say equal only when the module record says supported."""
    for name in evidence["backends"]:
        for row in evidence["backends"][name]["cases"]:
            row["status"] = status
            row["outputs_equal"] = None
            row["state_equal"] = None
            row["refusal_equal"] = None
            row["compared"] = []
    for row in evidence["equivalence"]["modules"]:
        row["status"] = status
        row["outputs_equal"] = None
        row["state_equal"] = None
        row["refusal_equal"] = None
        row["compared"] = []


def judge(claim, evidence, revision=REV, corpus=None):
    tmp = tempfile.mkdtemp()
    try:
        ev_path = os.path.join(tmp, "evidence.json")
        shutil.copy(os.path.join(HERE, "testdata", "model.lean"), os.path.join(tmp, "model.lean"))
        with open(ev_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f)
        return accept.decide(claim, evidence, revision, corpus, ev_path)
    finally:
        shutil.rmtree(tmp)


def codes(reasons):
    return [r.split(":", 1)[0] for r in reasons]


class GateTests(unittest.TestCase):
    def test_obligations_match_the_gate(self):
        _statements, errors = accept.load_obligations()
        self.assertEqual(errors, [])

    def test_manifest_is_the_examples_directory(self):
        root = os.path.abspath(os.path.join(HERE, "..", ".."))
        examples = os.path.join(root, "spec", "examples")
        self.assertEqual(accept.manifest_ids(), accept.corpus_files(examples))
        self.assertEqual(len(accept.manifest_ids()), 60)

    def test_positive_fixture_is_test_only_and_passes(self):
        reasons = judge(green_claim(), green_evidence())
        self.assertEqual(reasons, [], reasons)

    def test_test_only_evidence_does_not_support_a_repository_claim(self):
        claim = green_claim()
        claim["subject"] = "repository"
        reasons = judge(claim, green_evidence())
        self.assertIn("subject-mismatch", codes(reasons))
        self.assertNotEqual(reasons, [])

    def test_missing_module(self):
        evidence = green_evidence()
        evidence["denominator_modules"] = 2
        evidence["floor"]["denominator"] = 2
        evidence = accept.seal(evidence)
        claim = green_claim()
        claim["asserts"]["denominator_modules"] = 2
        reasons = judge(claim, evidence)
        self.assertIn("missing-module", codes(reasons))

    def test_missing_backend(self):
        claim = green_claim()
        claim["surface"]["backends"] = ["form", "scheduled", "core", "rust"]
        claim["asserts"]["backends_ready"] = ["form", "scheduled", "core", "rust"]
        reasons = judge(claim, green_evidence())
        self.assertIn("missing-backend", codes(reasons))

    def test_mismatched_outputs(self):
        evidence = green_evidence()
        evidence["backends"]["form"]["cases"][0]["status"] = "differ"
        evidence["backends"]["form"]["cases"][0]["outputs_equal"] = False
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("backend-output-mismatch", codes(reasons))

    def test_mismatched_refusals(self):
        evidence = green_evidence()
        row = evidence["backends"]["core"]["cases"][0]
        row["status"] = "differ"
        row["refusal_equal"] = False
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("backend-refusal-mismatch", codes(reasons))

    def test_exit_success_is_not_result_agreement(self):
        evidence = green_evidence()
        row = evidence["backends"]["scheduled"]["cases"][0]
        row["exit_equal"] = True
        row["outputs_equal"] = False
        row["state_equal"] = False
        row["refusal_equal"] = False
        row["compared"] = ["exit"]
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("exit-status-not-result", codes(reasons))

    def test_stale_revision(self):
        evidence = green_evidence()
        evidence["revision"] = OTHER
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence, revision=REV)
        self.assertIn("stale-revision", codes(reasons))
        claim = green_claim()
        claim["revision"] = OTHER
        reasons = judge(claim, green_evidence(), revision=REV)
        self.assertIn("stale-revision", codes(reasons))
        reasons = judge(green_claim(), green_evidence(), revision=OTHER)
        self.assertIn("stale-revision", codes(reasons))

    def test_stale_checker_revision(self):
        evidence = green_evidence()
        evidence["label"] = "repository"
        evidence["built_from"] = "b" * 12
        evidence = accept.seal(evidence)
        claim = green_claim()
        claim["subject"] = "repository"
        reasons = judge(claim, evidence)
        self.assertIn("stale-revision", codes(reasons))
        self.assertTrue(any("checker build" in r for r in reasons))
        evidence = green_evidence()
        evidence["label"] = "repository"
        evidence["built_from"] = ""
        evidence = accept.seal(evidence)
        reasons = judge(claim, evidence)
        self.assertIn("stale-revision", codes(reasons))

    def test_stale_action_mapping(self):
        evidence = green_evidence()
        digest = evidence["evidence_sha256"]
        evidence["backends"]["form"]["cases"][0]["outputs_equal"] = False
        self.assertEqual(evidence["evidence_sha256"], digest)
        reasons = judge(green_claim(), evidence)
        self.assertIn("malformed-evidence", codes(reasons))
        self.assertTrue(any("evidence_sha256 does not match" in r for r in reasons))
        evidence = green_evidence()
        evidence["modules"][0]["status"] = "refused"
        evidence["floor"] = accept.recompute_floor(evidence["modules"])
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("malformed-evidence", codes(reasons))
        self.assertTrue(any("marks case.rce equal" in r for r in reasons))
        evidence = green_evidence()
        evidence["backends"]["core"]["cases"].append(dict(evidence["backends"]["core"]["cases"][0]))
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("malformed-evidence", codes(reasons))
        self.assertTrue(any("maps case.rce twice" in r for r in reasons))

    def test_malformed_evidence(self):
        tmp = tempfile.mkdtemp()
        try:
            claim = os.path.join(tmp, "claim.json")
            evidence = os.path.join(tmp, "evidence.json")
            with open(claim, "w", encoding="utf-8") as f:
                f.write("{}\n")
            with open(evidence, "w", encoding="utf-8") as f:
                f.write("{not json\n")
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "accept.py"), "validate", claim, evidence, REV],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("malformed-evidence", proc.stdout)
            with open(evidence, "w", encoding="utf-8") as f:
                f.write("{}\n")
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "accept.py"), "validate", claim, evidence, REV],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("malformed-claim", proc.stdout)
            self.assertIn("malformed-evidence", proc.stdout)
        finally:
            shutil.rmtree(tmp)

    def test_missing_proof_obligation(self):
        evidence = green_evidence()
        evidence["proofs"]["model_proofs"] = []
        evidence["proofs"]["unproved_obligations"] = list(accept.REQUIRED_IDS)
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("unproved-obligation", codes(reasons))
        self.assertTrue(any("test or a proof statement" in r for r in reasons))

    def test_refusal_counted_as_support(self):
        evidence = green_evidence()
        evidence["modules"][0]["program_functions"]["tag"] = "catch"
        evidence["refusals"] = [{
            "module": "case.rce",
            "tag": "catch",
            "class": "unsupported",
            "counted_as_support": True,
        }]
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("refusal-counted-as-support", codes(reasons))

    def test_function_floor_does_not_imply_whole_modules(self):
        modules = []
        for n in range(60):
            ident = "m%02d.rce" % n
            module = green_module(ident)
            module["status"] = "refused"
            module["whole_module"] = {
                "status": "refused",
                "tag": "out" if n >= 5 else "catch",
                "bytes_equal": None,
                "results_equal": None,
                "compared": [],
            }
            module["output_semantics"] = {"hard": True, "status": "refused", "compared": []}
            if n < 5:
                module["program_functions"] = {"class": "syntactic", "roots": 1, "tag": None}
            else:
                module["program_functions"] = {"class": "refused", "roots": 1, "tag": "catch"}
                module["floor_tag"] = "catch"
            modules.append(module)
        evidence = green_evidence(modules)
        evidence["equivalence"] = {"modules": []}
        for name in evidence["backends"]:
            evidence["backends"][name] = {"cases": []}
        evidence["constructs"]["catch"] = {"status": "refused", "modules": 55, "counted_as_support": False}
        evidence = accept.seal(evidence)
        claim = green_claim()
        claim["asserts"]["syntactic_program_examples"] = 5
        claim["asserts"]["whole_modules_constructed"] = 0
        claim["asserts"]["denominator_modules"] = 60
        reasons = judge(claim, evidence)
        self.assertIn("function-floor-not-modules", codes(reasons))
        self.assertIn("whole-module-incomplete", codes(reasons))
        self.assertIn("output-semantics-incomplete", codes(reasons))
        self.assertTrue(any("5 of 60" in r for r in reasons))

    def test_semantic_option_b_is_not_added_to_the_syntactic_floor(self):
        module = green_module()
        module["program_functions"] = {"class": "semantic", "roots": 1, "tag": None}
        evidence = green_evidence([module])
        evidence["floor"]["syntactic_program_examples"] = 1
        evidence["floor"]["semantic_option_b_examples"] = 0
        evidence = accept.seal(evidence)
        claim = green_claim()
        reasons = judge(claim, evidence)
        self.assertIn("inflated-floor", codes(reasons))

    def test_experimental_scoped_claim_can_pass(self):
        module = green_module()
        module["status"] = "refused"
        module["whole_module"] = {
            "status": "refused", "tag": "global", "bytes_equal": None,
            "results_equal": None, "compared": [],
        }
        module["output_semantics"] = {"hard": True, "status": "refused", "compared": []}
        module["program_functions"] = {"class": "syntactic", "roots": 1, "tag": None}
        evidence = green_evidence([module])
        drop_equal_actions(evidence, "refused")
        evidence["metering"]["semantic_budget"]["agrees"] = None
        evidence["constructs"] = {
            "break": {"status": "unverified", "modules": 0, "counted_as_support": False},
            "catch": {"status": "unverified", "modules": 0, "counted_as_support": False},
            "pure-count": {"status": "unverified", "modules": 0, "counted_as_support": False},
        }
        evidence["equivalence"] = {"modules": [{
            "module": "case.rce", "status": "refused", "outputs_equal": None,
            "state_equal": None, "refusal_equal": None, "compared": [],
        }]}
        evidence["refusals"] = [{
            "module": "case.rce", "tag": "global", "class": "unsupported",
            "counted_as_support": False,
        }]
        evidence = accept.seal(evidence)
        claim = green_claim()
        claim["kind"] = "experimental"
        claim["asserts"]["production_compiler"] = False
        claim["asserts"]["whole_module_ready"] = False
        claim["asserts"]["backends_ready"] = []
        claim["asserts"]["proved"] = []
        claim["asserts"]["syntactic_program_examples"] = 1
        claim["asserts"]["whole_modules_constructed"] = 0
        claim["surface"]["includes_output_semantics"] = False
        claim["surface"]["includes_whole_module_form"] = False
        claim["surface"]["includes_effects"] = False
        claim["surface"]["backends"] = []
        claim["scope"] = {
            "text": "This is not a production claim. It records one syntactic program function and no whole module.",
            "excludes": [
                "whole-module-form-path",
                "source-form-scheduled-equivalence",
                "backend-parity",
                "output-semantics",
                "semantic-budget-omega",
                "break-admission",
                "catch-shape",
                "pure-count-c7-fg7",
            ],
        }
        reasons = judge(claim, evidence)
        self.assertEqual(reasons, [], reasons)

    def test_experimental_cannot_assert_production(self):
        claim = green_claim()
        claim["kind"] = "experimental"
        claim["scope"]["text"] = "not a production claim"
        reasons = judge(claim, green_evidence())
        self.assertIn("experimental-asserts-production", codes(reasons))

    def test_missing_joules_do_not_fail_a_semantic_budget(self):
        reasons = judge(green_claim(), green_evidence())
        self.assertEqual(reasons, [])
        self.assertIsNone(green_evidence()["metering"]["measured_joules"])

    def test_joules_do_not_discharge_the_semantic_budget(self):
        evidence = green_evidence()
        evidence["metering"]["semantic_budget"]["agrees"] = None
        evidence["metering"]["measured_joules"] = {
            "kind": "meter", "source": "rapl", "joules": "1.5", "revision": REV,
        }
        evidence = accept.seal(evidence)
        claim = green_claim()
        claim["asserts"]["measured_joules"] = True
        reasons = judge(claim, evidence)
        self.assertIn("semantic-budget-unverified", codes(reasons))
        self.assertNotIn("measurement-missing", codes(reasons))

    def test_modelled_joules_fail_a_measured_claim(self):
        evidence = green_evidence()
        evidence["metering"]["measured_joules"] = {
            "kind": "modelled", "source": "estimate", "joules": "1", "revision": REV,
        }
        evidence = accept.seal(evidence)
        claim = green_claim()
        claim["asserts"]["measured_joules"] = True
        reasons = judge(claim, evidence)
        self.assertIn("measurement-not-meter", codes(reasons))
        self.assertNotIn("semantic-budget-unverified", codes(reasons))

    def test_output_is_hard_for_whole_module_form(self):
        evidence = green_evidence()
        evidence["modules"][0]["output_semantics"]["hard"] = False
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("output-semantics-incomplete", codes(reasons))
        self.assertTrue(any("not optional because the target is Form" in r for r in reasons))

    def test_output_cannot_be_excluded_from_whole_module_form(self):
        claim = green_claim()
        claim["kind"] = "experimental"
        claim["asserts"]["production_compiler"] = False
        claim["asserts"]["whole_module_ready"] = False
        claim["asserts"]["proved"] = []
        claim["scope"] = {
            "text": "not a production claim",
            "excludes": ["output-semantics"],
        }
        reasons = judge(claim, green_evidence())
        self.assertIn("output-not-optional", codes(reasons))

    def test_a_test_does_not_discharge_a_proof(self):
        evidence = green_evidence()
        evidence["proofs"]["tests"][0]["discharges"] = ["whole-module-form-path"]
        evidence["proofs"]["model_proofs"] = [
            row for row in evidence["proofs"]["model_proofs"]
            if row["obligation"] != "whole-module-form-path"
        ]
        evidence["proofs"]["model_proofs"].append({
            "obligation": "whole-module-form-path",
            "theorem": "examples_are_covered_or_require_change",
            "path": "spec/checker/tests/form_bridge.rs",
        })
        evidence["proofs"]["unproved_obligations"] = ["measured-joules"]
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("test-not-proof", codes(reasons))
        self.assertIn("unproved-obligation", codes(reasons))

    def test_stale_obligations_hash(self):
        evidence = green_evidence()
        evidence["obligations_sha256"] = "0" * 64
        evidence = accept.seal(evidence)
        reasons = judge(green_claim(), evidence)
        self.assertIn("stale-obligations", codes(reasons))

    def test_process_error_is_not_support(self):
        module = green_module()
        module["status"] = "process_error"
        module["program_functions"] = {"class": "process_error", "roots": 0, "tag": "lowering: internal: broken"}
        module["floor_tag"] = "lowering: internal: broken"
        module["whole_module"] = {
            "status": "process_error", "tag": None, "bytes_equal": None,
            "results_equal": None, "compared": [],
        }
        module["output_semantics"] = {"hard": True, "status": "process_error", "compared": []}
        evidence = green_evidence([module])
        drop_equal_actions(evidence, "process_error")
        evidence["refusals"] = [{
            "module": "case.rce", "tag": "lowering: internal: broken",
            "class": "process_error", "counted_as_support": False,
        }]
        evidence = accept.seal(evidence)
        claim = green_claim()
        claim["asserts"]["syntactic_program_examples"] = 0
        claim["asserts"]["whole_modules_constructed"] = 0
        reasons = judge(claim, evidence)
        self.assertIn("process-error", codes(reasons))
        self.assertNotIn("supported", codes(reasons))

    def test_assemble_keeps_a_missing_module_unverified(self):
        tmp = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp, "modules"))
            manifest = {
                "format": "rce-prod-manifest/1",
                "label": "test-only",
                "corpus_kind": "test-only",
                "built_from": "",
                "revision": REV,
                "ids": ["present.rce", "absent.rce"],
            }
            with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with open(os.path.join(tmp, "modules", "present.rce.json"), "w", encoding="utf-8") as f:
                json.dump(green_module("present.rce"), f)
            out = os.path.join(tmp, "evidence.json")
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "accept.py"), "assemble", tmp, out, REV],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            evidence = accept.load_json(out)
            ids = [m["id"] for m in evidence["modules"]]
            self.assertEqual(ids, ["present.rce", "absent.rce"])
            self.assertEqual(evidence["denominator_modules"], 2)
            absent = evidence["modules"][1]
            self.assertEqual(absent["status"], "unverified")
            claim = green_claim()
            claim["asserts"]["denominator_modules"] = 2
            claim["asserts"]["syntactic_program_examples"] = 1
            claim["asserts"]["whole_modules_constructed"] = 1
            reasons = accept.decide(claim, evidence, REV, None, out)
            self.assertNotEqual(reasons, [])
            self.assertIn("whole-module-incomplete", codes(reasons))
        finally:
            shutil.rmtree(tmp)

    def test_cli_passes_the_positive_fixture(self):
        tmp = tempfile.mkdtemp()
        try:
            ev = os.path.join(tmp, "evidence.json")
            cl = os.path.join(tmp, "claim.json")
            shutil.copy(os.path.join(HERE, "testdata", "model.lean"), os.path.join(tmp, "model.lean"))
            with open(ev, "w", encoding="utf-8") as f:
                json.dump(green_evidence(), f)
            with open(cl, "w", encoding="utf-8") as f:
                json.dump(green_claim(), f)
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "accept.py"), "validate", cl, ev, REV],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("accept: PASS", proc.stdout)
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "accept.py"), "validate", cl, ev, REV, "no-such"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 1)
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "accept.py"), "validate"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 2)
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "accept.py"), "validate", cl, os.path.join(tmp, "missing.json"), REV],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("missing-evidence", proc.stdout)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
