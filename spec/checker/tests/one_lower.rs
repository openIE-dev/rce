//! The lowering of the one emitter (spec/proposals/one-emitter.md ONE-4,
//! section 3.5; order of work step 3), tested differentially on
//! spec/examples: for every program the lowering covers, the trace printed
//! by evaluating its core (`rce emit trace`) equals the reference
//! interpreter's trace byte for byte. Every example is listed below, either
//! as covered or with the first construct the lowering does not yet cover;
//! a program that is not covered is refused with that construct named, and
//! never printed differently.
//!
//! The conformance suite is tested the same way in one_conformance.rs.
use rce_spec_check::one;
use std::path::PathBuf;

/// The programs of spec/examples whose core prints the reference's trace.
const COVERED: &[&str] = &[
    "01_counter", "02_hello_main", "03_fib_loop", "04_bank_account", "05_kind_match", "06_option", "16_multi_region",
    "17_traffic_light", "18_while_bound", "19_floats", "22_tuples", "23_patterns", "25_labels", "27_bitops",
    "30_state_machine_guarded", "31_inventory", "32_comptime_block", "35_nested_if", "38_hex_bin", "42_f32", "46_u64",
    "49_loop_limit", "53_capability", "54_secret", "57_float_vectors", "59_dim_redundant_field", "60_dim_hidden_period",
    "61_dim_minimal",
];

/// The programs the lowering does not yet cover, and the construct it
/// names first. Each is refused with that construct; none prints a trace.
const NOT_COVERED: &[(&str, &str)] = &[
    ("07_error_union", "the method isdigit of text"),
    ("08_traits", "a trait"),
    ("09_generics", "the generic function"),
    ("10_comptime", "the generic function"),
    ("11_arena", "the method words of text"),
    ("12_borrow", "the &mut parameter"),
    ("13_spawn_chan", "chan"),
    ("14_select", "chan"),
    ("15_actor", "chan"),
    ("20_strings", "the method strip of text"),
    ("21_maps", "the expression at line"),
    ("24_lambda", "the method map"),
    ("28_slices", "the expression at line"),
    ("29_nested_kind", "a recursive type"),
    ("33_in_operator", "the operator in"),
    ("34_chan_types", "chan"),
    ("36_multiline_brackets", "the method fold"),
    ("37_actor_spawn", "chan"),
    ("39_structural_recursion", "a recursive type"),
    ("40_extern", "a foreign function"),
    ("41_arena_bound_wrapping", "an arena with a bound"),
    ("43_simd_lanes", "List[f64] * List[f64]"),
    ("44_device_region", "an arena with a bound, a device region or a marker region"),
    ("45_byte_text", "the expression at line"),
    ("47_select_fair", "chan"),
    ("48_linear_handle", "a foreign function"),
    ("50_iterators", "the iterator function"),
    ("51_refined_input", "a refined type"),
    ("52_context_text", "the name Html"),
    ("55_rcedb", "an indexed assignment"),
    ("56_utf8", "the method utf8"),
    ("58_float_keys", "the method get of Map"),
];

/// The covered programs whose probes (the extra observations of the
/// emitter tests, `emit::with_probes`, which call every function on sample
/// arguments) also lower with identical traces.
const COVERED_WITH_PROBES: &[&str] = &[
    "01_counter", "02_hello_main", "03_fib_loop", "04_bank_account", "05_kind_match", "06_option", "17_traffic_light",
    "18_while_bound", "19_floats", "22_tuples", "23_patterns", "25_labels", "27_bitops", "30_state_machine_guarded",
    "32_comptime_block", "35_nested_if", "38_hex_bin", "46_u64", "49_loop_limit", "53_capability", "54_secret",
    "57_float_vectors", "59_dim_redundant_field", "60_dim_hidden_period", "61_dim_minimal",
];

fn examples() -> Vec<(String, String)> {
    let mut v: Vec<PathBuf> = std::fs::read_dir("../examples")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    v.sort();
    v.into_iter().map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(&p).unwrap())).collect()
}

fn module(src: &str) -> rce_spec_check::ast::Module {
    assert!(rce_spec_check::cli::diagnostics(src, &Default::default()).is_empty());
    rce_spec_check::program(src).unwrap()
}

fn first_diff(want: &str, got: &str) -> String {
    let (w, g): (Vec<&str>, Vec<&str>) = (want.lines().collect(), got.lines().collect());
    let k = w.iter().zip(&g).position(|(a, b)| a != b).unwrap_or(w.len().min(g.len()));
    format!("line {}: want {:?}, got {:?}", k + 1, w.get(k), g.get(k))
}

/// The differential test of the lowering on spec/examples.
#[test]
fn examples_lower_to_the_reference_trace() {
    let all = examples();
    assert_eq!(all.len(), COVERED.len() + NOT_COVERED.len(), "every example is listed once");
    let mut same = 0;
    for (stem, src) in &all {
        let m = module(src);
        let want = one::trace::reference(&m);
        let got = one::trace_of(&m);
        if COVERED.contains(&stem.as_str()) {
            let got = got.unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(got == want, "{stem}: {}", first_diff(&want, &got));
            same += 1;
        } else {
            let (_, why) = NOT_COVERED.iter().find(|(n, _)| n == stem).unwrap_or_else(|| panic!("{stem} is not listed"));
            let e = got.err().unwrap_or_else(|| panic!("{stem} is now covered; move it to COVERED"));
            assert!(e.starts_with("not covered by the lowering: ") && e.contains(why), "{stem}: {e}");
        }
    }
    eprintln!("examples: {}; identical traces: {same}; not yet covered: {}", all.len(), all.len() - same);
}

/// The same with the probe observations, which reach every function body.
#[test]
fn examples_with_probes_lower_to_the_reference_trace() {
    for (stem, src) in examples() {
        let m = rce_spec_check::emit::with_probes(&module(&src));
        let want = one::trace::reference(&m);
        match one::trace_of(&m) {
            Ok(got) => {
                assert!(COVERED_WITH_PROBES.contains(&stem.as_str()), "{stem} with probes is now covered; list it");
                assert!(got == want, "{stem} with probes: {}", first_diff(&want, &got));
            }
            Err(e) => {
                assert!(!COVERED_WITH_PROBES.contains(&stem.as_str()), "{stem} with probes: {e}");
                assert!(e.starts_with("not covered by the lowering: "), "{stem} with probes: {e}");
            }
        }
    }
}

/// 3.4 on lowered programs: the JSON form reads back to the same module and
/// its text is canonical; `rce emit trace` of that JSON prints the same
/// trace, and `rce emit core` of it prints it again (ONE-1: FILE may be a
/// core module).
#[test]
fn core_json_round_trip() {
    for (stem, src) in examples() {
        if !COVERED.contains(&stem.as_str()) {
            continue;
        }
        let m = module(&src);
        let c = one::core_of(&m).unwrap();
        let text = one::json::module_text(&c);
        let back = one::json::module_from_text(&text).unwrap();
        assert!(back == c, "{stem}: the JSON does not read back to the same module");
        assert_eq!(one::json::module_text(&back), text, "{stem}: the text is not canonical");
        assert!(text.starts_with("{\"defs\":{"), "{stem}: keys in sorted order");
        let t = rce_spec_check::cli::emit("trace", &text).unwrap();
        assert_eq!(t, one::trace::reference(&m), "{stem}: the trace of the core JSON");
        assert_eq!(rce_spec_check::cli::emit("core", &text).unwrap(), text + "\n");
    }
}

/// The lowering is deterministic: the same program gives the same bytes.
#[test]
fn core_is_deterministic() {
    let src = std::fs::read_to_string("../examples/04_bank_account.rce").unwrap();
    let a = rce_spec_check::cli::emit("core", &src).unwrap();
    let b = rce_spec_check::cli::emit("core", &src).unwrap();
    assert_eq!(a, b);
}

/// `rce emit core FILE` and `rce emit trace FILE` through the command line:
/// status 0 with the bytes on standard output; a program the lowering does
/// not cover is status 2 with the construct named; a program the checker
/// refuses is status 1.
#[test]
fn emit_core_and_trace_commands() {
    let rce = |args: &[&str]| {
        let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let (mut out, mut err) = (vec![], vec![]);
        let code = rce_spec_check::cli::main(&a, &mut &b""[..], &mut out, &mut err);
        (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
    };
    let counter = "../examples/01_counter.rce";
    let (c, out, _) = rce(&["emit", "trace", counter]);
    assert_eq!(c, 0);
    assert_eq!(out, one::trace::reference(&module(&std::fs::read_to_string(counter).unwrap())));
    assert!(out.starts_with("example "), "{out}");
    let (c, core, _) = rce(&["emit", "core", counter]);
    assert_eq!(c, 0);
    let v: serde_json::Value = serde_json::from_str(&core).unwrap();
    assert_eq!(v["format"], "rce-core/1");
    let dir = std::env::temp_dir().join(format!("rce-one-lower-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("counter.core.json");
    std::fs::write(&p, &core).unwrap();
    let ps = p.to_string_lossy().into_owned();
    assert_eq!(rce(&["emit", "trace", &ps]), (0, out.clone(), String::new()));
    assert_eq!(rce(&["emit", "core", &ps]), (0, core.clone(), String::new()));
    // a core module that is not well formed is refused when it is read
    let bad = core.replacen("\"ret\"", "\"halt\"", 1);
    std::fs::write(&p, &bad).unwrap();
    assert_eq!(rce(&["emit", "trace", &ps]).0, 1);
    let (c, _, err) = rce(&["emit", "trace", "../examples/13_spawn_chan.rce"]);
    assert_eq!(c, 2);
    assert!(err.starts_with("not covered by the lowering: "), "{err}");
    let q = dir.join("bad.rce");
    std::fs::write(&q, "fn f() -> i64\n    \"text\"\n").unwrap();
    assert_eq!(rce(&["emit", "trace", &q.to_string_lossy()]).0, 1);
    let _ = std::fs::remove_dir_all(&dir);
}
