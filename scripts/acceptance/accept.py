#!/usr/bin/env python3
"""Production acceptance gate for a claim against a bound evidence record.

The gate is a function of the claim, the evidence and the revision. It does
not decide by a model. A missing, malformed or stale record fails. A refusal
is not support. A test is not a proof. A count of program functions is not
whole-module support. The order value omega is semantic fuel, cost, output
and memo state, not a physical meter. emit and out stay required on any
surface that includes output, including whole-module Form evaluation.

This gate does not finish the compiler.

  python3 scripts/acceptance/accept.py validate CLAIM EVIDENCE REVISION
  python3 scripts/acceptance/accept.py validate CLAIM EVIDENCE REVISION CORPUS_DIR
  python3 scripts/acceptance/accept.py seal EVIDENCE OUT
  python3 scripts/acceptance/accept.py assemble CASES_DIR OUT REVISION

CORPUS_DIR is required when the evidence corpus is spec-examples. It is the
directory of single-file examples (spec/examples). assemble reads
CASES_DIR/manifest.json and CASES_DIR/modules/<id>.json. An id with no file
stays in the denominator as unverified.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OBLIGATIONS_PATH = os.path.join(HERE, "obligations.json")
MANIFEST_PATH = os.path.join(HERE, "spec-examples.txt")
REV_RE = re.compile(r"^[0-9a-f]{40}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
BUILT_RE = re.compile(r"^[0-9a-f]{12}$")
JOULE_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")
THEOREM_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
COMPARED = ("outputs", "state", "refusal")
SEMANTIC_FIELDS = ["fuel", "cost", "output", "memo"]
PRODUCTION_BACKENDS = ("form", "scheduled", "core")
MODULE_STATUS = ("supported", "refused", "unverified", "defect", "process_error")
FN_CLASS = (
    "syntactic",
    "semantic",
    "refused",
    "defect",
    "process_error",
    "unverified",
    "not-admitted",
)
CASE_STATUS = ("equal", "differ", "refused", "unverified", "defect", "process_error")
CONSTRUCTS = {
    "break-admission": "break",
    "catch-shape": "catch",
    "pure-count-c7-fg7": "pure-count",
}

# Enforcement lives here. obligations.json must match these flags.
REQUIRED = (
    ("whole-module-form-path", True, True),
    ("source-form-scheduled-equivalence", True, True),
    ("backend-parity", True, True),
    ("refusal-accounting", True, True),
    ("output-semantics", True, True),
    ("semantic-budget-omega", True, True),
    ("break-admission", True, True),
    ("catch-shape", True, True),
    ("pure-count-c7-fg7", True, True),
    ("measured-joules", False, False),
)
REQUIRED_IDS = tuple(row[0] for row in REQUIRED)
PROOF_IDS = tuple(row[0] for row in REQUIRED if row[2])


def pairs(seq):
    seen = set()
    obj = {}
    for key, value in seq:
        if key in seen:
            raise ValueError("duplicate key %s" % key)
        seen.add(key)
        obj[key] = value
    return obj


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=pairs)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(obj):
    body = dict(obj)
    body.pop("evidence_sha256", None)
    body["evidence_sha256"] = sha256_text(canonical(body))
    return body


def manifest_ids():
    ids = []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line)
    return ids


def corpus_files(directory):
    names = []
    for name in os.listdir(directory):
        if name.endswith(".rce") and os.path.isfile(os.path.join(directory, name)):
            names.append(name)
    names.sort()
    return names


def load_obligations():
    """Return (statements, errors). Flags that differ from REQUIRED fail."""
    errors = []
    try:
        doc = load_json(OBLIGATIONS_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return {}, ["malformed-obligations: %s" % e]
    rows = doc.get("obligations") if isinstance(doc, dict) else None
    if doc.get("format") != "rce-prod-obligations/1" or not isinstance(rows, list):
        return {}, ["malformed-obligations: obligations.json is not rce-prod-obligations/1"]
    got = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            errors.append("malformed-obligations: a row has no id")
            continue
        got[row["id"]] = row
    if tuple(got) != REQUIRED_IDS and list(got) != list(REQUIRED_IDS):
        # Order is part of the contract, so a reader meets the obligations once.
        if set(got) != set(REQUIRED_IDS) or [r["id"] for r in rows] != list(REQUIRED_IDS):
            errors.append("malformed-obligations: obligation ids differ from the gate")
    statements = {}
    for oid, normative, proof in REQUIRED:
        row = got.get(oid)
        if not isinstance(row, dict):
            errors.append("malformed-obligations: missing %s" % oid)
            continue
        if row.get("normative") is not normative or row.get("needs_model_proof") is not proof:
            errors.append("malformed-obligations: flags for %s differ from the gate" % oid)
        statement = row.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            errors.append("malformed-obligations: %s has no statement" % oid)
        else:
            statements[oid] = statement
    return statements, errors


def recompute_floor(modules):
    floor = {
        "denominator": len(modules),
        "syntactic_program_examples": 0,
        "semantic_option_b_examples": 0,
        "function_refused": 0,
        "function_defect": 0,
        "function_process_error": 0,
        "function_unverified": 0,
        "function_not_admitted": 0,
        "whole_modules_constructed": 0,
        "supported_modules": 0,
        "refused_modules": 0,
        "unverified_modules": 0,
        "defect_modules": 0,
        "process_error_modules": 0,
        "admitted_programs": 0,
    }
    fn_key = {
        "syntactic": "syntactic_program_examples",
        "semantic": "semantic_option_b_examples",
        "refused": "function_refused",
        "defect": "function_defect",
        "process_error": "function_process_error",
        "unverified": "function_unverified",
        "not-admitted": "function_not_admitted",
    }
    st_key = {
        "supported": "supported_modules",
        "refused": "refused_modules",
        "unverified": "unverified_modules",
        "defect": "defect_modules",
        "process_error": "process_error_modules",
    }
    for m in modules:
        fn = m.get("program_functions") or {}
        floor[fn_key.get(fn.get("class"), "function_unverified")] += 1
        floor[st_key.get(m.get("status"), "unverified_modules")] += 1
        whole = m.get("whole_module") or {}
        if whole.get("bytes_equal") is True:
            floor["whole_modules_constructed"] += 1
        if m.get("admitted") is True:
            floor["admitted_programs"] += 1
    return floor


def is_str_list(value):
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def has_compared(value):
    return isinstance(value, list) and all(name in value for name in COMPARED)


def add(reasons, code, message):
    reasons.append("%s: %s" % (code, message))


def require_dict(value, name, reasons):
    if not isinstance(value, dict):
        add(reasons, "malformed-evidence", "%s is missing" % name)
        return False
    return True


def check_schema(claim, evidence, reasons):
    """Return False when the record is too incomplete to judge further."""
    ok = True
    if not isinstance(claim, dict) or claim.get("format") != "rce-prod-claim/1":
        add(reasons, "malformed-claim", "format is not rce-prod-claim/1")
        ok = False
    if not isinstance(evidence, dict) or evidence.get("format") != "rce-prod-evidence/1":
        add(reasons, "malformed-evidence", "format is not rce-prod-evidence/1")
        ok = False
    if not ok:
        return False
    surface = claim.get("surface")
    asserts = claim.get("asserts")
    scope = claim.get("scope")
    if claim.get("kind") not in ("production", "experimental"):
        add(reasons, "malformed-claim", "kind is not production or experimental")
        ok = False
    if claim.get("subject") not in ("repository", "test-only"):
        add(reasons, "malformed-claim", "subject is not repository or test-only")
        ok = False
    if not isinstance(claim.get("revision"), str):
        add(reasons, "malformed-claim", "revision is missing")
        ok = False
    if not isinstance(surface, dict):
        add(reasons, "malformed-claim", "surface is missing")
        ok = False
    else:
        for key in ("includes_output_semantics", "includes_whole_module_form", "includes_effects"):
            if not isinstance(surface.get(key), bool):
                add(reasons, "malformed-claim", "surface.%s is not a boolean" % key)
                ok = False
        if not is_str_list(surface.get("backends")):
            add(reasons, "malformed-claim", "surface.backends is not a list of names")
            ok = False
    if not isinstance(asserts, dict):
        add(reasons, "malformed-claim", "asserts is missing")
        ok = False
    else:
        for key in ("production_compiler", "whole_module_ready", "function_floor_implies_modules", "measured_joules"):
            if not isinstance(asserts.get(key), bool):
                add(reasons, "malformed-claim", "asserts.%s is not a boolean" % key)
                ok = False
        for key in ("syntactic_program_examples", "whole_modules_constructed", "denominator_modules"):
            if not isinstance(asserts.get(key), int) or isinstance(asserts.get(key), bool):
                add(reasons, "malformed-claim", "asserts.%s is not an integer" % key)
                ok = False
        if not is_str_list(asserts.get("backends_ready")) or not is_str_list(asserts.get("proved")):
            add(reasons, "malformed-claim", "asserts.backends_ready and asserts.proved are lists of names")
            ok = False
    if not isinstance(scope, dict) or not isinstance(scope.get("text"), str) or not is_str_list(scope.get("excludes")):
        add(reasons, "malformed-claim", "scope.text and scope.excludes are required")
        ok = False
    if evidence.get("label") not in ("repository", "test-only"):
        add(reasons, "malformed-evidence", "label is not repository or test-only")
        ok = False
    if evidence.get("corpus_kind") not in ("spec-examples", "test-only"):
        add(reasons, "malformed-evidence", "corpus_kind is not spec-examples or test-only")
        ok = False
    if not isinstance(evidence.get("revision"), str):
        add(reasons, "malformed-evidence", "revision is missing")
        ok = False
    if not isinstance(evidence.get("built_from"), str):
        add(reasons, "malformed-evidence", "built_from is missing")
        ok = False
    if not isinstance(evidence.get("denominator_modules"), int) or isinstance(evidence.get("denominator_modules"), bool):
        add(reasons, "malformed-evidence", "denominator_modules is not an integer")
        ok = False
    if not isinstance(evidence.get("modules"), list):
        add(reasons, "malformed-evidence", "modules is missing")
        ok = False
    if evidence.get("gate_finishes_compiler") is not False:
        add(reasons, "gate-overclaim", "the acceptance gate does not finish the compiler")
    if not isinstance(evidence.get("producer"), str) or not evidence.get("producer"):
        add(reasons, "malformed-evidence", "producer is missing")
        ok = False
    if not isinstance(evidence.get("evidence_sha256"), str):
        add(reasons, "malformed-evidence", "evidence_sha256 is missing")
        ok = False
    if not isinstance(evidence.get("obligations_sha256"), str):
        add(reasons, "malformed-evidence", "obligations_sha256 is missing")
        ok = False
    for key in ("backends", "equivalence", "constructs", "floor", "proofs", "metering"):
        if not isinstance(evidence.get(key), dict):
            add(reasons, "malformed-evidence", "%s is missing" % key)
            ok = False
    if not isinstance(evidence.get("refusals"), list):
        add(reasons, "malformed-evidence", "refusals is missing")
        ok = False
    return ok


def exclusion_allowed(claim, oid):
    surface = claim["surface"]
    whole = surface["includes_whole_module_form"]
    output = surface["includes_output_semantics"]
    effects = surface["includes_effects"]
    if oid == "output-semantics":
        return not whole and not output
    if oid == "semantic-budget-omega":
        return not whole and not effects
    if oid == "refusal-accounting":
        return False
    if oid == "backend-parity":
        return claim["asserts"]["backends_ready"] == []
    if oid in ("whole-module-form-path", "source-form-scheduled-equivalence"):
        return not whole
    if oid in CONSTRUCTS:
        return True
    if oid == "measured-joules":
        return not claim["asserts"]["measured_joules"]
    return False


def applies(claim, oid):
    if oid == "measured-joules":
        return claim["asserts"]["measured_joules"] is True
    if claim["kind"] == "production":
        return True
    if oid in claim["scope"]["excludes"] and exclusion_allowed(claim, oid):
        return False
    return True


def module_ids(evidence):
    return [m.get("id") for m in evidence["modules"] if isinstance(m, dict)]


def proof_file_text(evidence_path, rel):
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or rel.startswith("..") or "/../" in ("/" + rel):
        return None
    parts = rel.split("/")
    if ".." in parts:
        return None
    root = os.path.dirname(os.path.abspath(evidence_path))
    path = os.path.normpath(os.path.join(root, rel))
    if os.path.commonpath([root, path]) != root:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def looks_like_test(path, theorem, tests):
    name = path.replace("\\", "/")
    if "/tests/" in ("/" + name) or name.endswith("_test.rs") or name.startswith("test_") or "/test_" in ("/" + name):
        return True
    for row in tests:
        if isinstance(row, dict) and (row.get("id") == theorem or row.get("path") == path):
            if row.get("discharges"):
                return True
    return False


def valid_proofs(evidence, evidence_path, reasons):
    """Model proofs whose files contain the named theorem. Tests are rejected."""
    proofs = evidence["proofs"]
    tests = proofs.get("tests")
    rows = proofs.get("model_proofs")
    if not isinstance(tests, list) or not isinstance(rows, list):
        add(reasons, "malformed-evidence", "proofs.tests and proofs.model_proofs are lists")
        return []
    good = []
    for row in tests:
        if not isinstance(row, dict):
            add(reasons, "malformed-evidence", "a test row is not an object")
            continue
        discharges = row.get("discharges")
        if discharges not in ([], None) and discharges != []:
            add(reasons, "test-not-proof", "test %s lists discharges; a test is not a proof" % row.get("id"))
    for row in rows:
        if not isinstance(row, dict):
            add(reasons, "malformed-evidence", "a model proof row is not an object")
            continue
        oid = row.get("obligation")
        theorem = row.get("theorem")
        path = row.get("path")
        if oid not in REQUIRED_IDS or not isinstance(theorem, str) or not THEOREM_NAME.match(theorem or ""):
            add(reasons, "malformed-evidence", "a model proof row is incomplete")
            continue
        if looks_like_test(path or "", theorem, tests):
            add(reasons, "test-not-proof", "%s cites %s; a test does not discharge %s" % (path, theorem, oid))
            continue
        text = proof_file_text(evidence_path, path)
        if text is None:
            add(reasons, "unproved-obligation", "%s cites %s and the file is not readable" % (oid, path))
            continue
        if not re.search(r"(?m)^theorem\s+%s\b" % re.escape(theorem), text):
            add(reasons, "unproved-obligation", "%s does not contain theorem %s" % (path, theorem))
            continue
        good.append(row)
    return good


def check_revisions(claim, evidence, revision, reasons):
    if not REV_RE.match(revision or ""):
        add(reasons, "malformed-claim", "the revision argument is not 40 hex digits")
        return
    if claim.get("revision") != revision or evidence.get("revision") != revision:
        add(
            reasons,
            "stale-revision",
            "claim %s, evidence %s, argument %s" % (claim.get("revision"), evidence.get("revision"), revision),
        )
    built = evidence.get("built_from") or ""
    if evidence.get("label") == "repository":
        if not BUILT_RE.match(built) or not revision.startswith(built):
            add(reasons, "stale-revision", "checker build %s does not match revision %s" % (built or "missing", revision))
    elif built and (not BUILT_RE.match(built) or not revision.startswith(built)):
        add(reasons, "stale-revision", "built_from %s does not match revision %s" % (built, revision))
    expect = sha256_file(OBLIGATIONS_PATH)
    if evidence.get("obligations_sha256") != expect:
        add(reasons, "stale-obligations", "evidence is not bound to scripts/acceptance/obligations.json")


def check_hash(evidence, reasons):
    digest = evidence.get("evidence_sha256")
    if not isinstance(digest, str) or not SHA_RE.match(digest):
        add(reasons, "malformed-evidence", "evidence_sha256 is not 64 hex digits")
        return
    body = dict(evidence)
    body.pop("evidence_sha256", None)
    if sha256_text(canonical(body)) != digest:
        add(reasons, "malformed-evidence", "evidence_sha256 does not match the record")


def check_modules(evidence, corpus_dir, reasons):
    modules = evidence["modules"]
    ids = []
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            add(reasons, "malformed-evidence", "modules[%d] is not an object" % index)
            continue
        ident = module.get("id")
        if not isinstance(ident, str) or not ident:
            add(reasons, "malformed-evidence", "a module has no id")
            continue
        ids.append(ident)
        if module.get("status") not in MODULE_STATUS:
            add(reasons, "malformed-evidence", "%s status is not a known class" % ident)
        fn = module.get("program_functions")
        whole = module.get("whole_module")
        output = module.get("output_semantics")
        if not isinstance(fn, dict) or fn.get("class") not in FN_CLASS:
            add(reasons, "malformed-evidence", "%s program_functions.class is missing" % ident)
        elif fn.get("class") in ("syntactic", "semantic") and fn.get("tag") is not None:
            add(reasons, "refusal-counted-as-support", "%s is counted syntactic or semantic with refusal tag %s" % (ident, fn.get("tag")))
        if not isinstance(whole, dict):
            add(reasons, "malformed-evidence", "%s whole_module is missing" % ident)
        if not isinstance(output, dict) or not isinstance(output.get("hard"), bool):
            add(reasons, "malformed-evidence", "%s output_semantics is missing" % ident)
        if module.get("status") == "supported":
            if not isinstance(whole, dict) or whole.get("bytes_equal") is not True or whole.get("results_equal") is not True or not has_compared(whole.get("compared")):
                add(reasons, "exit-status-not-result", "%s is supported without equal outputs, state and refusal" % ident)
            if not isinstance(output, dict) or output.get("status") != "equal" or output.get("hard") is not True or not has_compared(output.get("compared")):
                add(reasons, "output-semantics-incomplete", "%s is supported without hard output agreement" % ident)
    if len(ids) != len(set(ids)):
        add(reasons, "malformed-evidence", "module ids are not unique")
    if evidence.get("denominator_modules") != len(modules):
        add(
            reasons,
            "missing-module",
            "denominator %s but %d module records" % (evidence.get("denominator_modules"), len(modules)),
        )
    kind = evidence.get("corpus_kind")
    if kind == "spec-examples":
        pinned = manifest_ids()
        if corpus_dir is None:
            add(reasons, "missing-module", "spec-examples evidence needs the examples directory")
        else:
            if not os.path.isdir(corpus_dir):
                add(reasons, "missing-module", "corpus directory %s is not readable" % corpus_dir)
            else:
                on_disk = corpus_files(corpus_dir)
                have = set(ids)
                for name in pinned:
                    if name not in have:
                        add(reasons, "missing-module", "%s is in the corpus manifest and missing from the evidence" % name)
                for name in on_disk:
                    if name not in have:
                        add(reasons, "missing-module", "%s is on disk and missing from the evidence" % name)
                allowed = set(pinned) | set(on_disk)
                for name in have:
                    if name not in allowed:
                        add(reasons, "unknown-module", "%s is not in the manifest or the examples directory" % name)
    return ids


def check_floor(evidence, reasons):
    modules = [m for m in evidence["modules"] if isinstance(m, dict)]
    got = evidence.get("floor")
    if not isinstance(got, dict):
        return
    actual = recompute_floor(modules)
    for key, value in actual.items():
        if got.get(key) != value:
            code = "inflated-floor" if isinstance(got.get(key), int) and got.get(key) > value and key in (
                "syntactic_program_examples",
                "whole_modules_constructed",
                "supported_modules",
                "admitted_programs",
            ) else "malformed-evidence"
            add(reasons, code, "floor.%s is %s and the cases sum to %s" % (key, got.get(key), value))


def check_refusals(evidence, reasons):
    supported = set()
    for module in evidence["modules"]:
        if isinstance(module, dict) and module.get("status") == "supported":
            supported.add(module.get("id"))
    for row in evidence.get("refusals") or []:
        if not isinstance(row, dict):
            add(reasons, "malformed-evidence", "a refusal row is not an object")
            continue
        if row.get("counted_as_support") is not False:
            add(reasons, "refusal-counted-as-support", "%s counts a refusal as support" % row.get("module"))
        if row.get("module") in supported:
            add(reasons, "refusal-counted-as-support", "%s is supported and also recorded as a refusal" % row.get("module"))


def check_case_row(row, where, reasons):
    if not isinstance(row, dict):
        add(reasons, "malformed-evidence", "%s is not an object" % where)
        return
    status = row.get("status")
    if status not in CASE_STATUS:
        add(reasons, "malformed-evidence", "%s status is not a known class" % where)
        return
    if status == "equal":
        if row.get("outputs_equal") is not True or row.get("state_equal") is not True or row.get("refusal_equal") is not True or not has_compared(row.get("compared")):
            add(reasons, "exit-status-not-result", "%s is equal without outputs, state and refusal" % where)
        if row.get("exit_equal") is True and (row.get("outputs_equal") is not True or row.get("state_equal") is not True or row.get("refusal_equal") is not True):
            add(reasons, "exit-status-not-result", "%s has exit success without result agreement" % where)


def backend_problem(row):
    status = row.get("status") if isinstance(row, dict) else None
    if not isinstance(row, dict) or status in (None, "unverified"):
        return "missing-backend"
    if status == "refused":
        return "backend-refused"
    if status == "process_error":
        return "process-error"
    if status == "defect":
        return "semantic-defect"
    if status == "differ" or status == "equal":
        if row.get("outputs_equal") is False:
            return "backend-output-mismatch"
        if row.get("state_equal") is False:
            return "backend-state-mismatch"
        if row.get("refusal_equal") is False:
            return "backend-refusal-mismatch"
        if status != "equal":
            return "backend-output-mismatch"
    return None


def check_measurement(claim, evidence, reasons):
    metering = evidence.get("metering") if isinstance(evidence.get("metering"), dict) else {}
    joules = metering.get("measured_joules", None)
    if joules is not None:
        if not isinstance(joules, dict) or joules.get("kind") != "meter" or not isinstance(joules.get("source"), str) or not joules.get("source") or not isinstance(joules.get("joules"), str) or not JOULE_RE.match(joules.get("joules") or "") or joules.get("revision") != claim.get("revision"):
            add(reasons, "measurement-not-meter", "measured joules need a meter source, a numeric reading and the same revision")
    elif claim["asserts"]["measured_joules"] is True:
        add(reasons, "measurement-missing", "the claim asserts measured joules and the evidence has no meter reading")


def check_semantic_budget(evidence, reasons):
    metering = evidence.get("metering") if isinstance(evidence.get("metering"), dict) else {}
    budget = metering.get("semantic_budget")
    if not isinstance(budget, dict) or budget.get("kind") != "semantic" or budget.get("fields") != SEMANTIC_FIELDS:
        add(reasons, "semantic-budget-unverified", "omega is fuel, cost, output and memo; the record does not show that state")
        return
    if budget.get("agrees") is True:
        return
    if budget.get("agrees") is False:
        add(reasons, "semantic-budget-mismatch", "fuel, cost, output and memo do not agree; a joule figure would not discharge this")
    else:
        add(reasons, "semantic-budget-unverified", "fuel, cost, output and memo were not shown to agree; a missing meter reading is not this failure")


def check_output(evidence, reasons):
    bad = 0
    out_tag = 0
    for module in evidence["modules"]:
        if not isinstance(module, dict):
            continue
        output = module.get("output_semantics") if isinstance(module.get("output_semantics"), dict) else {}
        whole = module.get("whole_module") if isinstance(module.get("whole_module"), dict) else {}
        tag = whole.get("tag") or ""
        if isinstance(tag, str) and (tag == "out" or tag.startswith("out:") or tag.startswith("emit")):
            out_tag += 1
        if output.get("hard") is not True or output.get("status") != "equal" or not has_compared(output.get("compared")):
            bad += 1
    if bad:
        add(
            reasons,
            "output-semantics-incomplete",
            "%d module(s) lack equal emit and out results; this stays required for whole-module Form evaluation and is not optional because the target is Form" % bad,
        )
    elif out_tag:
        add(reasons, "output-semantics-incomplete", "a whole-module record still carries an out or emit refusal")


def check_construct(evidence, oid, reasons):
    name = CONSTRUCTS[oid]
    row = (evidence.get("constructs") or {}).get(name)
    if not isinstance(row, dict):
        add(reasons, "construct-unverified", "%s has no construct record" % name)
        return
    if row.get("counted_as_support") is not False:
        add(reasons, "refusal-counted-as-support", "%s counts a refusal as support" % name)
    status = row.get("status")
    if status == "supported" and row.get("modules") == 0:
        return
    if status == "refused":
        add(reasons, "construct-refused", "%s is refused on %s module(s) and is not support" % (name, row.get("modules")))
    elif status == "defect":
        add(reasons, "semantic-defect", "%s is a semantic defect, not support" % name)
    elif status == "process_error":
        add(reasons, "process-error", "%s ended in a process error" % name)
    else:
        add(reasons, "construct-unverified", "%s is not shown as supported" % name)


def check_equivalence(evidence, ids, reasons):
    rows = (evidence.get("equivalence") or {}).get("modules")
    if not isinstance(rows, list):
        add(reasons, "equivalence-incomplete", "equivalence.modules is missing")
        return
    by_id = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("module"), str):
            by_id[row["module"]] = row
    missing = 0
    for ident in ids:
        row = by_id.get(ident)
        if row is None or row.get("status") != "equal" or row.get("outputs_equal") is not True or row.get("state_equal") is not True or row.get("refusal_equal") is not True or not has_compared(row.get("compared")):
            missing += 1
            if isinstance(row, dict) and row.get("status") == "equal":
                add(reasons, "exit-status-not-result", "%s equivalence is marked equal without outputs, state and refusal" % ident)
    if missing:
        add(reasons, "equivalence-incomplete", "%d module(s) lack source, Form and scheduled agreement of outputs, state and refusal" % missing)


def check_backends(claim, evidence, ids, reasons):
    backends = evidence.get("backends") if isinstance(evidence.get("backends"), dict) else {}
    ready = claim["asserts"]["backends_ready"]
    surface = claim["surface"]["backends"]
    if len(ready) != len(set(ready)) or len(surface) != len(set(surface)):
        add(reasons, "malformed-claim", "backend names are repeated")
    for name in ready:
        if name not in surface:
            add(reasons, "malformed-claim", "%s is ready but not on the claimed surface" % name)
    if claim["kind"] != "production" and not ready:
        add(reasons, "missing-backend", "backend parity applies and no backend is ready")
        return
    required = list(ready)
    if claim["kind"] == "production":
        for name in PRODUCTION_BACKENDS:
            if name not in surface or name not in ready:
                add(reasons, "production-surface-incomplete", "a production claim gates %s on its own" % name)
        required = list(surface)
    for name in required:
        block = backends.get(name)
        cases = block.get("cases") if isinstance(block, dict) else None
        if not isinstance(cases, list):
            add(reasons, "missing-backend", "%s has no case list" % name)
            continue
        by_id = {}
        for row in cases:
            if isinstance(row, dict) and isinstance(row.get("module"), str):
                by_id[row["module"]] = row
        grouped = {}
        for ident in ids:
            row = by_id.get(ident)
            if row is None:
                grouped.setdefault("missing-backend", []).append(ident)
                continue
            check_case_row(row, "%s %s" % (name, ident), reasons)
            problem = backend_problem(row)
            if problem:
                grouped.setdefault(problem, []).append(ident)
        messages = {
            "missing-backend": "%s has no result comparison for %d module(s), including %s",
            "backend-refused": "%s refused %d module(s), including %s; a refusal is not backend support",
            "process-error": "%s hit a process error on %d module(s), including %s",
            "semantic-defect": "%s and the reference differ semantically on %d module(s), including %s",
            "backend-output-mismatch": "%s outputs differ on %d module(s), including %s",
            "backend-state-mismatch": "%s state differs on %d module(s), including %s",
            "backend-refusal-mismatch": "%s refusal behaviour differs on %d module(s), including %s",
        }
        for problem, idents in grouped.items():
            add(reasons, problem, messages[problem] % (name, len(idents), idents[0]))


def check_whole(claim, evidence, reasons):
    modules = [m for m in evidence["modules"] if isinstance(m, dict)]
    floor = recompute_floor(modules)
    d = floor["denominator"]
    supported = floor["supported_modules"]
    whole = floor["whole_modules_constructed"]
    syntactic = floor["syntactic_program_examples"]
    if claim["asserts"]["denominator_modules"] != d or evidence.get("denominator_modules") != d:
        add(reasons, "missing-module", "claimed denominator %s, evidence denominator %s, cases %s" % (
            claim["asserts"]["denominator_modules"], evidence.get("denominator_modules"), d))
    if claim["asserts"]["syntactic_program_examples"] != syntactic:
        add(reasons, "unsupported-claim", "claimed syntactic program examples %s but the cases sum to %s" % (
            claim["asserts"]["syntactic_program_examples"], syntactic))
    if claim["asserts"]["whole_modules_constructed"] != whole:
        add(reasons, "unsupported-claim", "claimed whole-module constructs %s but the cases sum to %s" % (
            claim["asserts"]["whole_modules_constructed"], whole))
    needs_modules = claim["kind"] == "production" or claim["asserts"]["whole_module_ready"] is True or applies(claim, "whole-module-form-path")
    if needs_modules and supported != d:
        defect = floor["defect_modules"]
        process = floor["process_error_modules"]
        add(
            reasons,
            "whole-module-incomplete",
            "supported %d of %d; byte-level constructs %d of %d; syntactic program examples %d of %d do not establish whole-module readiness" % (
                supported, d, whole, d, syntactic, d),
        )
        add(
            reasons,
            "function-floor-not-modules",
            "syntactic program examples %d of %d are not whole-module support (byte-level constructs %d of %d, supported %d of %d)" % (
                syntactic, d, whole, d, supported, d),
        )
        if defect:
            add(reasons, "semantic-defect", "%d module(s) have a semantic mismatch and are not support" % defect)
        if process:
            add(reasons, "process-error", "%d module(s) ended in a process error and are not support" % process)
    elif claim["asserts"]["function_floor_implies_modules"] is True:
        add(reasons, "function-floor-not-modules", "the claim treats the function floor as whole-module readiness")


def check_scope(claim, reasons):
    if claim["asserts"]["function_floor_implies_modules"] is True:
        add(reasons, "function-floor-not-modules", "function_floor_implies_modules is true")
    excludes = claim["scope"]["excludes"]
    if len(excludes) != len(set(excludes)):
        add(reasons, "malformed-claim", "scope.excludes repeats an obligation")
    for oid in excludes:
        if oid not in REQUIRED_IDS:
            add(reasons, "malformed-claim", "scope.excludes names unknown %s" % oid)
    if claim["kind"] == "production":
        if claim["asserts"]["production_compiler"] is not True or claim["asserts"]["whole_module_ready"] is not True:
            add(reasons, "malformed-claim", "a production claim asserts production_compiler and whole_module_ready")
        if excludes:
            add(reasons, "production-excludes-obligation", "a production claim cannot exclude %s" % ", ".join(excludes))
        surface = claim["surface"]
        if surface["includes_output_semantics"] is not True or surface["includes_whole_module_form"] is not True or surface["includes_effects"] is not True:
            add(reasons, "production-surface-incomplete", "production includes whole-module Form, output semantics and the semantic order value")
    else:
        if claim["asserts"]["production_compiler"] is not False or claim["asserts"]["whole_module_ready"] is not False:
            add(reasons, "experimental-asserts-production", "an experimental claim asserts production or whole-module readiness")
        if "not a production claim" not in claim["scope"]["text"]:
            add(reasons, "experimental-unscoped", "an experimental claim says, in scope.text, that it is not a production claim")
        for oid in excludes:
            if oid in REQUIRED_IDS and not exclusion_allowed(claim, oid):
                if oid == "output-semantics":
                    add(reasons, "output-not-optional", "emit and out cannot be excluded while the surface includes output or whole-module Form evaluation")
                elif oid == "semantic-budget-omega":
                    add(reasons, "semantic-budget-not-optional", "omega cannot be excluded while the surface includes effects or whole-module Form evaluation; this is not a meter exclusion")
                elif oid == "refusal-accounting":
                    add(reasons, "refusal-not-excludable", "refusals cannot be counted as support on any claim")
                else:
                    add(reasons, "experimental-unscoped", "%s cannot be excluded for this surface" % oid)


def check_subject(claim, evidence, reasons):
    if claim["subject"] != evidence["label"]:
        add(reasons, "subject-mismatch", "claim subject %s and evidence label %s differ" % (claim["subject"], evidence["label"]))
    if claim["subject"] == "repository" and evidence["label"] == "test-only":
        add(reasons, "subject-mismatch", "test-only evidence cannot support a repository claim")


def check_statements(evidence, statements, reasons):
    rows = (evidence.get("proofs") or {}).get("statements")
    if not isinstance(rows, list):
        add(reasons, "missing-proof-obligation", "proof statements are missing")
        return
    got = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            got[row["id"]] = row.get("statement")
    for oid in REQUIRED_IDS:
        if got.get(oid) != statements.get(oid):
            add(reasons, "missing-proof-obligation", "the proof statement for %s is missing or differs from the contract" % oid)


def check_unproved(evidence, good, reasons):
    proved = {row["obligation"] for row in good}
    want = [oid for oid in REQUIRED_IDS if oid not in proved]
    got = (evidence.get("proofs") or {}).get("unproved_obligations")
    if got != want:
        add(reasons, "missing-proof-obligation", "unproved obligations are %s and the record says %s" % (want, got))
    return proved


def decide(claim, evidence, revision, corpus_dir, evidence_path):
    reasons = []
    statements, obligation_errors = load_obligations()
    reasons.extend(obligation_errors)
    if not check_schema(claim, evidence, reasons):
        return reasons
    check_hash(evidence, reasons)
    check_revisions(claim, evidence, revision, reasons)
    check_subject(claim, evidence, reasons)
    check_scope(claim, reasons)
    if obligation_errors:
        return reasons
    check_statements(evidence, statements, reasons)
    ids = check_modules(evidence, corpus_dir, reasons)
    check_floor(evidence, reasons)
    check_refusals(evidence, reasons)
    good = valid_proofs(evidence, evidence_path, reasons)
    proved = check_unproved(evidence, good, reasons)
    check_measurement(claim, evidence, reasons)
    # Cross-check: a supported module's equivalence and form case agree.
    eq_rows = (evidence.get("equivalence") or {}).get("modules")
    eq_by = {}
    if isinstance(eq_rows, list):
        for row in eq_rows:
            if isinstance(row, dict):
                eq_by[row.get("module")] = row
    form_cases = ((evidence.get("backends") or {}).get("form") or {}).get("cases") if isinstance(evidence.get("backends"), dict) else None
    form_by = {}
    if isinstance(form_cases, list):
        for row in form_cases:
            if isinstance(row, dict):
                form_by[row.get("module")] = row
    for module in evidence["modules"]:
        if not isinstance(module, dict) or module.get("status") != "supported":
            continue
        ident = module.get("id")
        eq = eq_by.get(ident)
        form = form_by.get(ident)
        if not isinstance(eq, dict) or eq.get("status") != "equal":
            add(reasons, "malformed-evidence", "%s is supported without an equal equivalence row" % ident)
        if not isinstance(form, dict) or form.get("status") != "equal":
            add(reasons, "malformed-evidence", "%s is supported without an equal form backend case" % ident)
    if applies(claim, "whole-module-form-path") or claim["kind"] == "production" or claim["asserts"]["whole_module_ready"]:
        check_whole(claim, evidence, reasons)
    else:
        # Counts that the claim does state still have to match, including on a scoped claim.
        modules = [m for m in evidence["modules"] if isinstance(m, dict)]
        floor = recompute_floor(modules)
        if claim["asserts"]["denominator_modules"] != floor["denominator"]:
            add(reasons, "missing-module", "claimed denominator does not match the cases")
        if claim["asserts"]["syntactic_program_examples"] != floor["syntactic_program_examples"]:
            add(reasons, "unsupported-claim", "claimed syntactic count does not match the cases")
        if claim["asserts"]["whole_modules_constructed"] != floor["whole_modules_constructed"]:
            add(reasons, "unsupported-claim", "claimed whole-module construct count does not match the cases")
        if claim["asserts"]["function_floor_implies_modules"] is True:
            add(reasons, "function-floor-not-modules", "the claim treats the function floor as whole-module readiness")
    if applies(claim, "source-form-scheduled-equivalence"):
        check_equivalence(evidence, [i for i in ids if isinstance(i, str)], reasons)
    if applies(claim, "backend-parity") or claim["asserts"]["backends_ready"]:
        check_backends(claim, evidence, [i for i in ids if isinstance(i, str)], reasons)
    if applies(claim, "output-semantics"):
        check_output(evidence, reasons)
    if applies(claim, "semantic-budget-omega"):
        check_semantic_budget(evidence, reasons)
    for oid in CONSTRUCTS:
        if applies(claim, oid):
            check_construct(evidence, oid, reasons)
    if applies(claim, "refusal-accounting"):
        # Already enforced on every well-formed record. Repeat if a supported
        # count hides a refusal, which check_floor and check_refusals cover.
        pass
    for oid in PROOF_IDS:
        if applies(claim, oid) and (claim["kind"] == "production" or oid in claim["asserts"]["proved"]) and oid not in proved:
            add(reasons, "unproved-obligation", "%s has no model proof; a test or a proof statement does not discharge it" % oid)
    claimed_proved = claim["asserts"]["proved"]
    for oid in claimed_proved:
        if oid not in REQUIRED_IDS:
            add(reasons, "malformed-claim", "proved names unknown %s" % oid)
        elif oid not in proved:
            add(reasons, "unproved-obligation", "the claim says %s is proved and no model proof discharges it" % oid)
    return list(dict.fromkeys(sorted(reasons)))


def cmd_validate(args):
    usage = "python3 scripts/acceptance/accept.py validate CLAIM EVIDENCE REVISION [CORPUS_DIR]"
    if len(args) not in (3, 4):
        sys.stderr.write("usage: %s\n" % usage)
        return 2
    claim_path, evidence_path, revision = args[0], args[1], args[2]
    corpus = args[3] if len(args) == 4 else None
    reasons = []
    try:
        claim = load_json(claim_path)
    except FileNotFoundError:
        reasons.append("missing-claim: %s" % claim_path)
        claim = None
    except (OSError, ValueError, json.JSONDecodeError) as e:
        reasons.append("malformed-claim: %s" % e)
        claim = None
    try:
        evidence = load_json(evidence_path)
    except FileNotFoundError:
        reasons.append("missing-evidence: %s" % evidence_path)
        evidence = None
    except (OSError, ValueError, json.JSONDecodeError) as e:
        reasons.append("malformed-evidence: %s" % e)
        evidence = None
    if claim is not None and evidence is not None:
        if corpus is not None and isinstance(evidence, dict) and evidence.get("corpus_kind") == "test-only":
            reasons.append("malformed-evidence: a corpus directory was given for a test-only corpus")
        reasons.extend(decide(claim, evidence, revision, corpus, evidence_path))
    reasons = list(dict.fromkeys(sorted(reasons)))
    for reason in reasons:
        print(reason)
    if reasons:
        print("accept: FAIL %d" % len(reasons))
        return 1
    print("accept: PASS")
    return 0


def cmd_seal(args):
    usage = "python3 scripts/acceptance/accept.py seal EVIDENCE OUT"
    if len(args) != 2:
        sys.stderr.write("usage: %s\n" % usage)
        return 2
    try:
        evidence = load_json(args[0])
    except (OSError, ValueError, json.JSONDecodeError) as e:
        sys.stderr.write("malformed-evidence: %s\n" % e)
        return 1
    if not isinstance(evidence, dict):
        sys.stderr.write("malformed-evidence: the record is not an object\n")
        return 1
    sealed = seal(evidence)
    with open(args[1], "w", encoding="utf-8") as f:
        json.dump(sealed, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")
    print(sealed["evidence_sha256"])
    return 0


def unverified_module(ident):
    return {
        "id": ident,
        "path": ident,
        "admitted": False,
        "status": "unverified",
        "floor_tag": None,
        "program_functions": {"class": "unverified", "roots": 0, "tag": None},
        "whole_module": {
            "status": "unverified",
            "tag": None,
            "bytes_equal": None,
            "results_equal": None,
            "compared": [],
        },
        "output_semantics": {"hard": True, "status": "unverified", "compared": []},
    }


def cmd_assemble(args):
    usage = "python3 scripts/acceptance/accept.py assemble CASES_DIR OUT REVISION"
    if len(args) != 3:
        sys.stderr.write("usage: %s\n" % usage)
        return 2
    cases, out, revision = args
    if not REV_RE.match(revision):
        sys.stderr.write("usage: revision is 40 hex digits\n")
        return 2
    manifest_path = os.path.join(cases, "manifest.json")
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        sys.stderr.write("malformed-evidence: %s\n" % e)
        return 1
    if not isinstance(manifest, dict) or manifest.get("format") != "rce-prod-manifest/1":
        sys.stderr.write("malformed-evidence: manifest format is not rce-prod-manifest/1\n")
        return 1
    ids = manifest.get("ids")
    if not is_str_list(ids) or not ids or len(ids) != len(set(ids)):
        sys.stderr.write("malformed-evidence: manifest ids are missing or repeated\n")
        return 1
    if manifest.get("revision") not in (None, revision):
        sys.stderr.write("stale-revision: manifest %s argument %s\n" % (manifest.get("revision"), revision))
        return 1
    label = manifest.get("label")
    kind = manifest.get("corpus_kind")
    built = manifest.get("built_from")
    if label not in ("repository", "test-only") or kind not in ("spec-examples", "test-only") or not isinstance(built, str):
        sys.stderr.write("malformed-evidence: manifest label, corpus_kind or built_from\n")
        return 1
    statements, errors = load_obligations()
    if errors:
        for error in errors:
            sys.stderr.write(error + "\n")
        return 1
    modules = []
    refusals = []
    for ident in ids:
        path = os.path.join(cases, "modules", ident + ".json")
        if os.path.exists(path):
            try:
                module = load_json(path)
            except (OSError, ValueError, json.JSONDecodeError) as e:
                sys.stderr.write("malformed-evidence: %s: %s\n" % (ident, e))
                return 1
            if not isinstance(module, dict) or module.get("id") != ident:
                sys.stderr.write("malformed-evidence: %s id does not match the file\n" % ident)
                return 1
        else:
            module = unverified_module(ident)
        modules.append(module)
        if module.get("status") != "supported":
            status = module.get("status")
            klass = {
                "refused": "unsupported",
                "defect": "defect",
                "process_error": "process_error",
            }.get(status, "unverified")
            refusals.append({
                "module": ident,
                "tag": (module.get("floor_tag") or (module.get("whole_module") or {}).get("tag") or status),
                "class": klass,
                "counted_as_support": False,
            })
    floor = recompute_floor(modules)
    floor["denominator"] = len(modules)
    cases_by_backend = {}
    for name in ("form", "scheduled", "core", "rust", "python", "js", "scala"):
        rows = []
        for module in modules:
            rows.append({
                "module": module["id"],
                "status": "unverified",
                "exit_equal": None,
                "outputs_equal": None,
                "state_equal": None,
                "refusal_equal": None,
                "compared": [],
                "executed": False,
            })
        cases_by_backend[name] = {"cases": rows}
    equivalence = []
    for module in modules:
        equivalence.append({
            "module": module["id"],
            "status": "unverified",
            "outputs_equal": None,
            "state_equal": None,
            "refusal_equal": None,
            "compared": [],
        })
    evidence = {
        "format": "rce-prod-evidence/1",
        "label": label,
        "corpus_kind": kind,
        "revision": revision,
        "built_from": built,
        "producer": "scripts/acceptance/accept.py assemble",
        "gate_finishes_compiler": False,
        "denominator_modules": len(modules),
        "admitted_programs": floor["admitted_programs"],
        "obligations_sha256": sha256_file(OBLIGATIONS_PATH),
        "modules": modules,
        "floor": floor,
        "refusals": refusals,
        "backends": cases_by_backend,
        "equivalence": {"modules": equivalence},
        "constructs": {
            name: {"status": "unverified", "modules": 0, "counted_as_support": False}
            for name in ("break", "catch", "pure-count")
        },
        "metering": {
            "semantic_budget": {"kind": "semantic", "fields": SEMANTIC_FIELDS, "agrees": None},
            "measured_joules": None,
        },
        "proofs": {
            "statements": [{"id": oid, "statement": statements[oid]} for oid in REQUIRED_IDS],
            "model_proofs": [],
            "tests": [],
            "unproved_obligations": list(REQUIRED_IDS),
        },
        "whole_refusal_counts": {},
        "first_refusal_counts": {},
    }
    sealed = seal(evidence)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sealed, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")
    print("accept: assembled %d modules, unverified gaps included" % len(modules))
    return 0


def main():
    usage = (
        "python3 scripts/acceptance/accept.py validate CLAIM EVIDENCE REVISION [CORPUS_DIR]\n"
        "python3 scripts/acceptance/accept.py seal EVIDENCE OUT\n"
        "python3 scripts/acceptance/accept.py assemble CASES_DIR OUT REVISION"
    )
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.stderr.write("usage: %s\n" % usage)
        return 2
    command = sys.argv[1]
    if command not in ("validate", "seal", "assemble"):
        sys.stderr.write("usage: %s\n" % usage)
        return 2
    args = sys.argv[2:]
    for arg in args:
        if arg.startswith("-"):
            sys.stderr.write("usage: %s\n" % usage)
            return 2
    if command == "validate":
        return cmd_validate(args)
    if command == "seal":
        return cmd_seal(args)
    return cmd_assemble(args)


if __name__ == "__main__":
    sys.exit(main())
