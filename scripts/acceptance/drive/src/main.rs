//! Corpus evidence for the production acceptance gate.
//!
//! The driver calls the public Form bridge and the reference interpreter.
//! It does not modify the bridge. A module is not marked supported: byte
//! identity, where it holds, is recorded apart from source, Form and
//! scheduled result agreement. Every manifest name stays in the denominator.
use std::collections::BTreeMap;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command;

use rce_spec_check::form::bridge;
use rce_spec_check::form::ir as fir;
use rce_spec_check::one::eval;
use rce_spec_check::one::ir::Module as Core;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

struct Exam {
    admitted: bool,
    status: &'static str,
    floor_tag: Option<String>,
    fn_class: &'static str,
    fn_roots: usize,
    fn_tag: Option<String>,
    whole_status: &'static str,
    whole_tag: Option<String>,
    bytes_equal: Option<bool>,
    source_trace_sha256: Option<String>,
    core_status: &'static str,
    core_executed: bool,
    core_detail: String,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next();
    let extra = args.next();
    if out.is_none() || extra.is_some() || out.as_deref().unwrap_or("").starts_with('-') {
        eprintln!("usage: rce-acceptance-drive OUT");
        std::process::exit(2);
    }
    let out = out.unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    match drive(&root, Path::new(&out)) {
        Ok(line) => {
            eprintln!("{line}");
        }
        Err(e) => {
            eprintln!("acceptance-drive: {e}");
            std::process::exit(1);
        }
    }
}

fn drive(root: &Path, out: &Path) -> Result<String, String> {
    let revision = git_head(root)?;
    if revision.len() != 40 || !revision.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("git revision {revision} is not 40 hex digits"));
    }
    let revision = revision.to_ascii_lowercase();
    let manifest = read_manifest(&root.join("scripts/acceptance/spec-examples.txt"))?;
    let obligations = fs::read(&root.join("scripts/acceptance/obligations.json"))
        .map_err(|e| format!("obligations.json: {e}"))?;
    let obligations_sha256 = hex(&Sha256::digest(&obligations));
    let doc: Value =
        serde_json::from_slice(&obligations).map_err(|e| format!("obligations.json: {e}"))?;
    let rows = doc
        .get("obligations")
        .and_then(|v| v.as_array())
        .ok_or("obligations.json has no obligations")?;
    let mut statements = Vec::new();
    let mut unproved = Vec::new();
    for row in rows {
        let id = row
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("obligation without id")?;
        let statement = row
            .get("statement")
            .and_then(|v| v.as_str())
            .ok_or("obligation without statement")?;
        statements.push(json!({"id": id, "statement": statement}));
        unproved.push(Value::String(id.to_string()));
    }
    let mut ids = manifest.clone();
    let dir = root.join("spec/examples");
    let mut on_disk = fs::read_dir(&dir)
        .map_err(|e| format!("spec/examples: {e}"))?
        .map(|e| e.map_err(|err| err.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    on_disk.sort_by_key(|e| e.file_name());
    for entry in &on_disk {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".rce") && entry.path().is_file() && !ids.contains(&name) {
            ids.push(name);
        }
    }
    let mut exams: Vec<(String, Exam)> = Vec::new();
    for id in &ids {
        let path = dir.join(id);
        let exam = if path.is_file() {
            let src = fs::read_to_string(&path).map_err(|e| format!("{id}: {e}"))?;
            match catch_unwind(AssertUnwindSafe(|| examine(&src))) {
                Ok(exam) => exam,
                Err(payload) => process_exam(&panic_text(payload)),
            }
        } else {
            missing_exam()
        };
        exams.push((id.clone(), exam));
    }
    if exams.iter().any(|(_, e)| e.status == "supported") {
        return Err("refusing to mark a module supported without source, Form and scheduled result agreement".into());
    }
    let modules: Vec<Value> = exams.iter().map(|(id, e)| module_json(id, e)).collect();
    let floor = floor_json(&exams);
    let evidence = json!({
        "format": "rce-prod-evidence/1",
        "label": "repository",
        "corpus_kind": "spec-examples",
        "revision": revision,
        "built_from": rce_spec_check::GIT_COMMIT,
        "producer": "scripts/acceptance/drive",
        "gate_finishes_compiler": false,
        "denominator_modules": exams.len(),
        "admitted_programs": exams.iter().filter(|(_, e)| e.admitted).count(),
        "obligations_sha256": obligations_sha256,
        "modules": modules,
        "floor": floor,
        "refusals": refusals(&exams),
        "backends": backends(&exams),
        "equivalence": {"modules": equivalence(&exams)},
        "constructs": constructs(&exams),
        "metering": {
            "semantic_budget": {
                "kind": "semantic",
                "fields": ["fuel", "cost", "output", "memo"],
                "agrees": Value::Null,
                "globals_outside_omega": globals_outside(&exams),
            },
            "measured_joules": Value::Null,
        },
        "proofs": {
            "statements": statements,
            "model_proofs": [],
            "tests": [{
                "id": "examples_are_covered_or_require_change",
                "path": "spec/checker/tests/form_bridge.rs",
                "discharges": [],
            }],
            "unproved_obligations": unproved,
        },
        "whole_refusal_counts": tag_counts(exams.iter().filter_map(|(_, e)| e.whole_tag.as_deref())),
        "first_refusal_counts": tag_counts(exams.iter().filter_map(|(_, e)| e.floor_tag.as_deref())),
    });
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    fs::write(
        out,
        serde_json::to_string_pretty(&evidence).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    Ok(format!(
        "acceptance-drive: revision {revision} modules {} admitted {} syntactic {} semantic {} whole-bytes {} supported {} refused {} defect {} unverified {} process {}",
        exams.len(),
        exams.iter().filter(|(_, e)| e.admitted).count(),
        exams.iter().filter(|(_, e)| e.fn_class == "syntactic").count(),
        exams.iter().filter(|(_, e)| e.fn_class == "semantic").count(),
        exams.iter().filter(|(_, e)| e.bytes_equal == Some(true)).count(),
        exams.iter().filter(|(_, e)| e.status == "supported").count(),
        exams.iter().filter(|(_, e)| e.status == "refused").count(),
        exams.iter().filter(|(_, e)| e.status == "defect").count(),
        exams.iter().filter(|(_, e)| e.status == "unverified").count(),
        exams.iter().filter(|(_, e)| e.status == "process_error").count(),
    ))
}

fn examine(src: &str) -> Exam {
    let ast = match rce_spec_check::program(src) {
        Ok(ast) => ast,
        Err(e) => return admit_failed(&e),
    };
    let source_trace_sha256 = match catch_unwind(AssertUnwindSafe(|| {
        rce_spec_check::one::trace::reference(&ast)
    })) {
        Ok(text) => Some(hex(&Sha256::digest(text.as_bytes()))),
        Err(_) => None,
    };
    let core_m = match rce_spec_check::one::core_of(&ast) {
        Ok(core_m) => core_m,
        Err(e) => return lowering(&e, source_trace_sha256),
    };
    let (whole_status, whole_tag, bytes_equal, module_status) = whole_module(&core_m);
    let (fn_class, fn_roots, fn_tag, floor_tag) = functions(&core_m);
    let (core_status, core_executed, core_detail) = core_observation(&core_m);
    Exam {
        admitted: true,
        status: module_status,
        floor_tag,
        fn_class,
        fn_roots,
        fn_tag,
        whole_status,
        whole_tag,
        bytes_equal,
        source_trace_sha256,
        core_status,
        core_executed,
        core_detail,
    }
}

fn admit_failed(err: &str) -> Exam {
    let internal = err.contains("internal:");
    let mut exam = if internal {
        process_exam(err)
    } else {
        missing_exam()
    };
    exam.admitted = false;
    exam.status = if internal { "process_error" } else { "refused" };
    exam.fn_class = "not-admitted";
    exam.floor_tag = Some(one_line(&format!("program: {err}")));
    exam.fn_tag = exam.floor_tag.clone();
    exam.whole_status = exam.status;
    exam
}

fn lowering(err: &str, trace: Option<String>) -> Exam {
    let internal = err.contains("internal:");
    let tag = one_line(&format!("lowering: {err}"));
    Exam {
        admitted: true,
        status: if internal { "process_error" } else { "refused" },
        floor_tag: Some(tag.clone()),
        fn_class: if internal { "process_error" } else { "refused" },
        fn_roots: 0,
        fn_tag: Some(tag),
        whole_status: "unverified",
        whole_tag: None,
        bytes_equal: None,
        source_trace_sha256: trace,
        core_status: "unverified",
        core_executed: false,
        core_detail: "the module did not lower, so the core evaluator was not run".into(),
    }
}

fn missing_exam() -> Exam {
    Exam {
        admitted: false,
        status: "unverified",
        floor_tag: None,
        fn_class: "unverified",
        fn_roots: 0,
        fn_tag: None,
        whole_status: "unverified",
        whole_tag: None,
        bytes_equal: None,
        source_trace_sha256: None,
        core_status: "unverified",
        core_executed: false,
        core_detail: "the example file is missing; the case stays in the denominator".into(),
    }
}

fn process_exam(err: &str) -> Exam {
    let tag = one_line(&format!("process: {err}"));
    Exam {
        admitted: false,
        status: "process_error",
        floor_tag: Some(tag.clone()),
        fn_class: "process_error",
        fn_roots: 0,
        fn_tag: Some(tag),
        whole_status: "process_error",
        whole_tag: None,
        bytes_equal: None,
        source_trace_sha256: None,
        core_status: "process_error",
        core_executed: false,
        core_detail: one_line(err),
    }
}

fn whole_module(core_m: &Core) -> (&'static str, Option<String>, Option<bool>, &'static str) {
    match bridge::construct(core_m) {
        Err(e) => {
            let tag = one_line(e.tag());
            let (whole_status, module_status) = class_tag(&tag);
            (whole_status, Some(tag), None, module_status)
        }
        Ok(form_m) => match bridge::schedule(&form_m) {
            Err(e) => {
                let tag = one_line(e.tag());
                ("defect", Some(tag), Some(false), "defect")
            }
            Ok(scheduled) => match bridge::construct(&scheduled) {
                Err(e) => {
                    let tag = one_line(e.tag());
                    ("defect", Some(tag), Some(false), "defect")
                }
                Ok(back) => {
                    if fir::encode(&form_m) == fir::encode(&back) {
                        // Byte identity is not source, Form and scheduled result agreement.
                        ("bytes-equal", None, Some(true), "unverified")
                    } else {
                        ("defect", Some("round-trip".into()), Some(false), "defect")
                    }
                }
            },
        },
    }
}

fn class_tag(tag: &str) -> (&'static str, &'static str) {
    if tag == "defect" || tag == "round-trip" || tag.starts_with("defect:") {
        ("defect", "defect")
    } else {
        ("refused", "refused")
    }
}

fn functions(core_m: &Core) -> (&'static str, usize, Option<String>, Option<String>) {
    let roots = root_names(core_m);
    if roots.is_empty() {
        return (
            "refused",
            0,
            Some("no program root".into()),
            Some("no program root".into()),
        );
    }
    let mut bad: Option<String> = None;
    let mut rewrote = false;
    for name in &roots {
        match closure_round_trip(core_m, name) {
            Ok(semantic) => rewrote |= semantic,
            Err(tag) => {
                bad = Some(tag);
                break;
            }
        }
    }
    match bad {
        None if !rewrote => ("syntactic", roots.len(), None, None),
        None => ("semantic", roots.len(), None, None),
        Some(tag) => {
            let tag = one_line(&tag);
            let class = if tag == "defect" || tag == "round-trip" || tag.starts_with("defect:") {
                "defect"
            } else {
                "refused"
            };
            (class, roots.len(), Some(tag.clone()), Some(tag))
        }
    }
}

fn closure_round_trip(core_m: &Core, name: &str) -> Result<bool, String> {
    let sub = bridge::closure(core_m, name).ok_or_else(|| "defect".to_string())?;
    let form_m = match bridge::construct(&sub) {
        Ok(form_m) => form_m,
        Err(e) => return Err(e.tag().to_string()),
    };
    let semantic = bridge::body_break_rewrite();
    let scheduled = match bridge::schedule(&form_m) {
        Ok(scheduled) => scheduled,
        Err(e) => return Err(e.tag().to_string()),
    };
    match bridge::construct(&scheduled) {
        Ok(back) if fir::encode(&form_m) == fir::encode(&back) => Ok(semantic),
        Ok(_) => Err("round-trip".into()),
        Err(e) => Err(e.tag().to_string()),
    }
}

fn program_root(name: &str) -> bool {
    name.split(" = ").any(|n| {
        n.starts_with("fn ") || n.starts_with("region ") || n == "delta" || n == "initial state"
    })
}

fn root_names(module: &Core) -> Vec<String> {
    let mut names = Vec::new();
    let Some(fns) = module.names.get("fns").and_then(|v| v.as_object()) else {
        return names;
    };
    for values in fns.values() {
        let Some(array) = values.as_array() else {
            continue;
        };
        for value in array {
            let Some(text) = value.as_str() else { continue };
            if let Some(part) = text.split(" = ").find(|part| program_root(part)) {
                names.push(part.to_string());
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

fn core_observation(module: &Core) -> (&'static str, bool, String) {
    let once = catch_unwind(AssertUnwindSafe(|| fingerprint(module)));
    let twice = catch_unwind(AssertUnwindSafe(|| fingerprint(module)));
    match (once, twice) {
        (Ok(Ok(a)), Ok(Ok(b))) if a == b => (
            "unverified",
            true,
            "the core evaluator agreed with itself; that is not agreement with the source and the Form".into(),
        ),
        (Ok(Ok(_)), Ok(Ok(_))) => (
            "defect",
            true,
            "two core evaluations of the same module differed".into(),
        ),
        (Ok(Err(e)), _) | (_, Ok(Err(e))) => (
            "process_error",
            true,
            one_line(&format!("core evaluator: {e}")),
        ),
        _ => ("process_error", true, "the core evaluator panicked".into()),
    }
}

fn fingerprint(module: &Core) -> Result<String, String> {
    let run = eval::run(module)?;
    let halt = format!("{:?}", run.halt);
    let mut hasher = Sha256::new();
    hasher.update(&run.out);
    hasher.update(halt.as_bytes());
    Ok(hex(&hasher.finalize()))
}

fn module_json(id: &str, exam: &Exam) -> Value {
    let output_status = if exam.whole_tag.as_deref().is_some_and(is_out_tag) {
        "refused"
    } else if exam.status == "unverified" {
        "unverified"
    } else {
        exam.status
    };
    json!({
        "id": id,
        "path": format!("spec/examples/{id}"),
        "admitted": exam.admitted,
        "status": exam.status,
        "floor_tag": exam.floor_tag,
        "program_functions": {
            "class": exam.fn_class,
            "roots": exam.fn_roots,
            "tag": exam.fn_tag,
        },
        "whole_module": {
            "status": exam.whole_status,
            "tag": exam.whole_tag,
            "bytes_equal": exam.bytes_equal,
            "results_equal": Value::Null,
            "compared": [],
        },
        "output_semantics": {
            "hard": true,
            "status": output_status,
            "compared": [],
        },
        "source_trace_sha256": exam.source_trace_sha256,
    })
}

fn is_out_tag(tag: &str) -> bool {
    tag == "out" || tag.starts_with("out:") || tag == "emit" || tag.starts_with("emit:")
}

fn case_json(id: &str, status: &str, executed: bool, detail: &str) -> Value {
    json!({
        "module": id,
        "status": status,
        "exit_equal": Value::Null,
        "outputs_equal": Value::Null,
        "state_equal": Value::Null,
        "refusal_equal": Value::Null,
        "compared": [],
        "executed": executed,
        "detail": detail,
    })
}

fn backends(exams: &[(String, Exam)]) -> Value {
    let mut form = Vec::new();
    let mut scheduled = Vec::new();
    let mut core = Vec::new();
    let mut absent = Vec::new();
    for (id, exam) in exams {
        form.push(case_json(
            id,
            exam.status,
            exam.status != "unverified",
            "form bridge construct",
        ));
        scheduled.push(case_json(
            id,
            if exam.bytes_equal == Some(true) {
                "unverified"
            } else {
                exam.status
            },
            exam.bytes_equal.is_some(),
            "schedule of the constructed Form",
        ));
        core.push(case_json(
            id,
            exam.core_status,
            exam.core_executed,
            &exam.core_detail,
        ));
        absent.push(case_json(
            id,
            "unverified",
            false,
            "hand-emitter result parity was not run; emit success would not be result agreement",
        ));
    }
    json!({
        "form": {"cases": form},
        "scheduled": {"cases": scheduled},
        "core": {"cases": core},
        "rust": {"cases": absent.clone()},
        "python": {"cases": absent.clone()},
        "js": {"cases": absent.clone()},
        "scala": {"cases": absent},
    })
}

fn equivalence(exams: &[(String, Exam)]) -> Vec<Value> {
    exams
        .iter()
        .map(|(id, exam)| {
            json!({
                "module": id,
                "status": if exam.bytes_equal == Some(true) { "unverified" } else { exam.status },
                "outputs_equal": Value::Null,
                "state_equal": Value::Null,
                "refusal_equal": Value::Null,
                "compared": [],
            })
        })
        .collect()
}

fn construct_bucket(tag: &str) -> Option<&'static str> {
    let head = tag.split(':').next().unwrap_or(tag).trim();
    match head {
        "break" => Some("break"),
        "catch" => Some("catch"),
        "FG8" | "bound" => Some("pure-count"),
        _ => None,
    }
}

fn constructs(exams: &[(String, Exam)]) -> Value {
    let mut counts = BTreeMap::from([("break", 0), ("catch", 0), ("pure-count", 0)]);
    for (_, exam) in exams {
        let mut seen = BTreeMap::new();
        for tag in exam.floor_tag.iter().chain(exam.whole_tag.iter()) {
            if let Some(bucket) = construct_bucket(tag) {
                seen.insert(bucket, ());
            }
        }
        for bucket in seen.keys() {
            *counts.get_mut(*bucket).unwrap() += 1;
        }
    }
    let mut object = serde_json::Map::new();
    for (name, count) in counts {
        object.insert(
            name.into(),
            json!({
                "status": if count > 0 { "refused" } else { "unverified" },
                "modules": count,
                "counted_as_support": false,
            }),
        );
    }
    Value::Object(object)
}

fn globals_outside(exams: &[(String, Exam)]) -> usize {
    exams
        .iter()
        .filter(|(_, exam)| {
            exam.whole_tag
                .as_deref()
                .is_some_and(|t| t == "global" || t.starts_with("global:"))
                || exam
                    .floor_tag
                    .as_deref()
                    .is_some_and(|t| t == "global" || t.starts_with("global:"))
        })
        .count()
}

fn refusals(exams: &[(String, Exam)]) -> Vec<Value> {
    let mut rows = Vec::new();
    for (id, exam) in exams {
        if exam.status == "supported" {
            continue;
        }
        let class = match exam.status {
            "defect" => "defect",
            "process_error" => "process_error",
            "unverified" => "unverified",
            _ => "unsupported",
        };
        let tag = exam
            .floor_tag
            .clone()
            .or_else(|| exam.whole_tag.clone())
            .unwrap_or_else(|| exam.status.to_string());
        rows.push(json!({
            "module": id,
            "tag": tag,
            "class": class,
            "counted_as_support": false,
        }));
    }
    rows
}

fn tag_counts<'a>(tags: impl Iterator<Item = &'a str>) -> Value {
    let mut counts: BTreeMap<&str, i64> = BTreeMap::new();
    for tag in tags {
        *counts.entry(tag).or_default() += 1;
    }
    let mut object = serde_json::Map::new();
    for (tag, count) in counts {
        object.insert(tag.to_string(), json!(count));
    }
    Value::Object(object)
}

fn floor_json(exams: &[(String, Exam)]) -> Value {
    let mut syntactic = 0;
    let mut semantic = 0;
    let mut function_refused = 0;
    let mut function_defect = 0;
    let mut function_process_error = 0;
    let mut function_unverified = 0;
    let mut function_not_admitted = 0;
    let mut whole = 0;
    let mut supported = 0;
    let mut refused = 0;
    let mut unverified = 0;
    let mut defect = 0;
    let mut process_error = 0;
    let mut admitted = 0;
    for (_, exam) in exams {
        match exam.fn_class {
            "syntactic" => syntactic += 1,
            "semantic" => semantic += 1,
            "refused" => function_refused += 1,
            "defect" => function_defect += 1,
            "process_error" => function_process_error += 1,
            "not-admitted" => function_not_admitted += 1,
            _ => function_unverified += 1,
        }
        match exam.status {
            "supported" => supported += 1,
            "refused" => refused += 1,
            "defect" => defect += 1,
            "process_error" => process_error += 1,
            _ => unverified += 1,
        }
        if exam.bytes_equal == Some(true) {
            whole += 1;
        }
        if exam.admitted {
            admitted += 1;
        }
    }
    json!({
        "denominator": exams.len(),
        "syntactic_program_examples": syntactic,
        "semantic_option_b_examples": semantic,
        "function_refused": function_refused,
        "function_defect": function_defect,
        "function_process_error": function_process_error,
        "function_unverified": function_unverified,
        "function_not_admitted": function_not_admitted,
        "whole_modules_constructed": whole,
        "supported_modules": supported,
        "refused_modules": refused,
        "unverified_modules": unverified,
        "defect_modules": defect,
        "process_error_modules": process_error,
        "admitted_programs": admitted,
    })
}

fn read_manifest(path: &Path) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut ids = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if !line.is_empty() {
            ids.push(line.to_string());
        }
    }
    if ids.is_empty() {
        return Err("the corpus manifest is empty".into());
    }
    Ok(ids)
}

fn git_head(root: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn one_line(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else {
        "panic".into()
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    out
}
