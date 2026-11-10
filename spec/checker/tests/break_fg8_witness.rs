//! Witnesses for the block-break and `fn table` measure contract.
//!
//! The contract is `spec/proposals/break-fg8-impl-contract.md`. These tests
//! call the public bridge and the core evaluator. A refusal is the reason
//! the bridge returns on `410bfd40`. A case that now constructs is checked
//! against the core: the form returns the same value, and so does the
//! scheduled core when the schedule builds. Nothing here builds a second Form.
//!
//! `410bfd40` is main after #157 (`d77075a2`, the residual law inside a
//! recognised loop) and #158 (`34c915fd`, library functions kept in the
//! core). The pins name that commit.
use rce_spec_check::form::bridge::{self, Change};
use rce_spec_check::form::{self, eval as feval, ir as fir};
use rce_spec_check::one::build::{self, FB};
use rce_spec_check::one::eval::{self, Flow, Val};
use rce_spec_check::one::ir::{Atom, Fn as CoreFn, FnRef, Lit, Module as CoreModule, Stmt, Ty};
use rce_spec_check::one::json::{canonical_group, group_key};
use rce_spec_check::one::wf;
use std::collections::BTreeMap;

fn i64ty() -> Ty {
    Ty::i64()
}

fn module(name: &str, f: CoreFn) -> CoreModule {
    let mut defs = BTreeMap::new();
    let mut names = serde_json::Map::new();
    let (g, _) = canonical_group(std::slice::from_ref(&f));
    let key = group_key(&g);
    names.insert(key.clone(), serde_json::json!([name]));
    defs.insert(key.clone(), g);
    let entry = if f.params.is_empty() {
        key
    } else {
        let main = CoreFn {
            params: vec![],
            ret: i64ty(),
            body: vec![Stmt::Ret(build::lit(0))],
        };
        let (mg, _) = canonical_group(std::slice::from_ref(&main));
        let mk = group_key(&mg);
        names.insert(mk.clone(), serde_json::json!(["main"]));
        defs.insert(mk.clone(), mg);
        mk
    };
    CoreModule {
        globals: vec![],
        defs,
        entry: (entry, 0),
        names: serde_json::json!({"fns": names, "globals": []}),
        graphs: BTreeMap::new(),
    }
}

fn tick_parts() -> (CoreFn, String) {
    let mut tick = FB::new(vec![Ty::u64()], Ty::Bool);
    let n = tick.arg(0);
    let cur = tick.let_global(0, Ty::u64());
    let c = tick.int_cmp("int.lt", cur.clone(), n.clone()).unwrap();
    tick.guard_halt(c, |b| b.trap_value(Atom::bytes(b"comptime fuel exhausted")))
        .unwrap();
    let r = tick.sub_in(cur, n, Ty::u64()).unwrap();
    tick.setg(0, r);
    tick.ret(Atom::bool(true));
    let tick = tick.finish();
    let (tg, _) = canonical_group(std::slice::from_ref(&tick));
    (tick, group_key(&tg))
}

fn assemble(tick: CoreFn, tkey: String, name: &str, user: CoreFn) -> CoreModule {
    let (tg, _) = canonical_group(std::slice::from_ref(&tick));
    let (ug, _) = canonical_group(std::slice::from_ref(&user));
    let ukey = group_key(&ug);
    let main = CoreFn {
        params: vec![],
        ret: i64ty(),
        body: vec![Stmt::Ret(build::lit(0))],
    };
    let (mg, _) = canonical_group(std::slice::from_ref(&main));
    let mkey = group_key(&mg);
    CoreModule {
        globals: vec![(Ty::u64(), Some(Lit::Int(feval::FUEL as i128)))],
        defs: BTreeMap::from([(tkey.clone(), tg), (ukey.clone(), ug), (mkey.clone(), mg)]),
        entry: (mkey.clone(), 0),
        names: serde_json::json!({
            "fns": {tkey: ["rt.tick"], ukey: [name], mkey: ["main"]},
            "globals": ["fuel"]
        }),
        graphs: BTreeMap::new(),
    }
}

fn by_arity(m: &CoreModule, n: usize) -> (String, u32) {
    let mut best: Option<(String, u32, usize)> = None;
    for (k, g) in &m.defs {
        for (i, f) in g.fns.iter().enumerate() {
            let name = m.names["fns"][k][i].as_str().unwrap_or("");
            if matches!(name, "rt.tick" | "rt.charge" | "rt.try_charge" | "main") {
                continue;
            }
            if f.params.len() == n && best.as_ref().is_none_or(|b| f.body.len() > b.2) {
                best = Some((k.clone(), i as u32, f.body.len()));
            }
        }
    }
    let (k, i, _) = best.unwrap_or_else(|| panic!("no core function of {n} parameters"));
    (k, i)
}

fn core_eval(m: &CoreModule, args: Vec<Val>) -> (Result<Val, String>, Option<Val>) {
    let (k, i) = by_arity(m, args.len());
    let mut ev = eval::Eval::new(m);
    let r = match ev.call_key(&k, i, args) {
        Ok(v) => Ok(v),
        Err(Flow::Halt(h)) => Err(format!("halt: {h:?}")),
        Err(Flow::Bad(e)) => Err(e),
    };
    let fuel = ev.global(0).cloned();
    (r, fuel)
}

fn refuse(m: &CoreModule) -> Change {
    wf::check(m).unwrap_or_else(|e| panic!("core is not well formed: {e}"));
    bridge::construct(m).expect_err("the bridge admitted a case the contract keeps refused")
}

/// A break that is not inside a count or a measure. `d77075a2` carries a
/// residual only inside a recognised loop, and this sentence is the refusal
/// for the rest. Measured on `410bfd40`.
const BREAK_OUTSIDE: &str =
    "break: a break outside a recognised loop is an exit this slice does not restructure";

/// An inner loop that is not a count and not a measure. The residual law is
/// not applied there. Measured on `410bfd40`.
const UNRECOGNISED_LOOP: &str = "loop: a loop has none of the two-region shapes the schedule writes (a measure, or a count bound). A lowering loop with a different statement order requires change";

fn assert_break_block(m: &CoreModule) {
    let e = refuse(m);
    assert_eq!(e.tag(), "break", "{}", e.line());
    assert_eq!(e.reason, BREAK_OUTSIDE, "{}", e.line());
}

fn assert_applied(f: &fir::Module) {
    assert!(
        matches!(
            form::check(fir::encode(f).as_bytes(), None),
            form::Outcome::Applied(_)
        ),
        "the constructed form is not applied"
    );
}

fn assert_values(m: &CoreModule, f: &fir::Module, args: Vec<Val>) {
    let (core_v, _) = core_eval(m, args.clone());
    let core_v = core_v.expect("core eval");
    let (k, i) = form_key(f, args.len());
    let form_v = feval::apply(f, &k, i, args).unwrap_or_else(|e| panic!("form eval: {e}"));
    assert_eq!(form_v, vec![core_v], "form eval and core eval differ");
}

fn assert_scheduled_values(origin: &CoreModule, scheduled: &CoreModule, args: Vec<Val>) {
    let (core_v, _) = core_eval(origin, args.clone());
    let (k, i) = by_arity(scheduled, args.len());
    let mut ev = eval::Eval::new(scheduled);
    let got = match ev.call_key(&k, i, args) {
        Ok(v) => Ok(v),
        Err(Flow::Halt(h)) => Err(format!("halt: {h:?}")),
        Err(Flow::Bad(e)) => Err(e),
    };
    assert_eq!(
        got, core_v,
        "the scheduled core and the original core differ"
    );
}

fn schedule_fg8(f: &fir::Module, at: &str) {
    let s = bridge::schedule(f).unwrap_or_else(|e| panic!("schedule: {}", e.line()));
    wf::check(&s).unwrap_or_else(|e| panic!("schedule is not well formed: {e}"));
    let e = bridge::construct(&s).expect_err("the scheduled measure was applied");
    assert_eq!(e.tag(), "FG8", "{}", e.line());
    assert_eq!(e.at, at, "{}", e.line());
    assert!(
        e.reason.contains(
            "the body region does not return i raised by exactly one whenever it continues"
        ),
        "{}",
        e.line()
    );
}

fn program_root(name: &str) -> bool {
    name.split(" = ").any(|n| {
        n.starts_with("fn ") || n.starts_with("region ") || n == "delta" || n == "initial state"
    })
}

fn root_names(m: &CoreModule) -> Vec<String> {
    let mut v = vec![];
    let Some(fns) = m.names.get("fns").and_then(|x| x.as_object()) else {
        return v;
    };
    for arr in fns.values() {
        let Some(a) = arr.as_array() else { continue };
        for n in a {
            let Some(s) = n.as_str() else { continue };
            if let Some(part) = s.split(" = ").find(|p| program_root(p)) {
                v.push(part.to_string());
            }
        }
    }
    v.sort();
    v.dedup();
    v
}

fn closure_tag(core_m: &CoreModule, name: &str) -> Result<bool, String> {
    let sub = bridge::closure(core_m, name).ok_or_else(|| "defect".to_string())?;
    let f = match bridge::construct(&sub) {
        Ok(f) => f,
        Err(e) => return Err(e.tag().to_string()),
    };
    let semantic = bridge::body_break_rewrite();
    let s = match bridge::schedule(&f) {
        Ok(s) => s,
        Err(e) => return Err(e.tag().to_string()),
    };
    match bridge::construct(&s) {
        Ok(back) if fir::encode(&f) == fir::encode(&back) => Ok(semantic),
        Ok(_) => Err("round-trip".into()),
        Err(e) => Err(e.tag().to_string()),
    }
}

fn load_example(file: &str) -> CoreModule {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../examples")
            .join(file),
    )
    .unwrap();
    let ast = rce_spec_check::program(&src).unwrap();
    rce_spec_check::one::core_of(&ast).unwrap_or_else(|e| panic!("{file}: {e}"))
}

#[test]
fn corpus_baseline_on_50294e05() {
    // The name records the commit the contract first measured (section 1).
    // The rows below are measured on 410bfd40, main after #157 (d77075a2)
    // and #158 (34c915fd).
    //
    // `whole` is the census definition in form_bridge_census.txt: the core
    // constructs, the schedule constructs, and the bytes are equal.
    // whole-entry-only is that set restricted to a core with no program
    // root (`fn`, `region`, `delta`, `initial state`). whole-with-user is
    // the rest. A program root's first refusal is the file's root tag, as
    // in examples_are_covered_or_require_change.
    //
    // Counting `construct` alone, with no schedule, accepts 18 modules on
    // this commit. That is not the census `whole` of 2. One of the 18,
    // `09_generics.rce`, is entry-only. The other 17 contain a user
    // function, and 16 of them do not schedule back to the same bytes.
    // The 18 entry-only cores in the #158 note were the count before
    // library functions were kept in the core. This 18 is the gate
    // accepting the Form after that change.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    files.sort();
    let mut programs = 0usize;
    let mut covered = Vec::new();
    let mut semantic = Vec::new();
    let mut root_tags: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut construct_tag: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut lowering: BTreeMap<String, String> = BTreeMap::new();
    let mut whole = Vec::new();
    let mut entry_only = Vec::new();
    let mut with_user = Vec::new();
    let mut construct_ok = Vec::new();
    for p in &files {
        let src = std::fs::read_to_string(p).unwrap();
        if rce_spec_check::program(&src).is_err() {
            continue;
        }
        let ast = rce_spec_check::program(&src).unwrap();
        programs += 1;
        let file = p.file_name().unwrap().to_string_lossy().to_string();
        let core_m = match rce_spec_check::one::core_of(&ast) {
            Ok(c) => c,
            Err(e) => {
                lowering.insert(file, e);
                continue;
            }
        };
        let roots = root_names(&core_m);
        let entry = roots.is_empty();
        match bridge::construct(&core_m) {
            Err(e) => construct_tag
                .entry(e.tag().to_string())
                .or_default()
                .push(file.clone()),
            Ok(f) => {
                construct_ok.push(file.clone());
                match bridge::schedule(&f) {
                    Err(e) => {
                        construct_tag
                            .entry(format!("schedule {}", e.tag()))
                            .or_default()
                            .push(file.clone());
                    }
                    Ok(s) => match bridge::construct(&s) {
                        Err(e) => {
                            construct_tag
                                .entry(format!("schedule {}", e.tag()))
                                .or_default()
                                .push(file.clone());
                        }
                        Ok(back) if fir::encode(&f) == fir::encode(&back) => {
                            whole.push(file.clone());
                            if entry {
                                entry_only.push(file.clone());
                            } else {
                                with_user.push(file.clone());
                            }
                        }
                        Ok(_) => construct_tag
                            .entry("round-trip".into())
                            .or_default()
                            .push(file.clone()),
                    },
                }
            }
        }
        if roots.is_empty() {
            root_tags
                .entry("no program root".into())
                .or_default()
                .push(file);
            continue;
        }
        let mut bad: Option<String> = None;
        let mut rewrote = false;
        for name in &roots {
            match closure_tag(&core_m, name) {
                Ok(sem) => rewrote |= sem,
                Err(e) => {
                    if bad.is_none() {
                        bad = Some(e);
                    }
                }
            }
        }
        match bad {
            None if !rewrote => covered.push(file),
            None => semantic.push(file),
            Some(tag) => root_tags.entry(tag).or_default().push(file),
        }
    }
    assert_eq!(programs, 60, "spec/examples gained or lost a program");
    assert_eq!(
        whole,
        ["09_generics.rce", "35_nested_if.rce"],
        "census whole, measured on 410bfd40"
    );
    assert_eq!(entry_only, ["09_generics.rce"], "whole-entry-only");
    assert_eq!(with_user, ["35_nested_if.rce"], "whole-with-user");
    assert_eq!(
        construct_ok,
        [
            "03_fib_loop.rce",
            "08_traits.rce",
            "09_generics.rce",
            "11_arena.rce",
            "12_borrow.rce",
            "18_while_bound.rce",
            "22_tuples.rce",
            "24_lambda.rce",
            "25_labels.rce",
            "27_bitops.rce",
            "28_slices.rce",
            "35_nested_if.rce",
            "36_multiline_brackets.rce",
            "40_extern.rce",
            "41_arena_bound_wrapping.rce",
            "43_simd_lanes.rce",
            "48_linear_handle.rce",
            "49_loop_limit.rce",
        ],
        "construct alone, before the schedule; not census whole"
    );
    assert_eq!(
        covered,
        [
            "12_borrow.rce",
            "35_nested_if.rce",
            "40_extern.rce",
            "42_f32.rce",
            "48_linear_handle.rce",
        ],
        "the syntactic floor moved"
    );
    assert!(
        semantic.is_empty(),
        "an example was admitted only as a body-break rewrite: {semantic:?}"
    );
    let names = |tag: &str| root_tags.get(tag).map(|v| v.as_slice()).unwrap_or(&[]);
    assert_eq!(
        names("break"),
        [
            "01_counter.rce",
            "04_bank_account.rce",
            "05_kind_match.rce",
            "16_multi_region.rce",
            "17_traffic_light.rce",
            "23_patterns.rce",
            "30_state_machine_guarded.rce",
            "44_device_region.rce",
            "51_refined_input.rce",
            "52_context_text.rce",
            "53_capability.rce",
            "54_secret.rce",
            "56_utf8.rce",
            "57_float_vectors.rce",
            "58_float_keys.rce",
            "59_dim_redundant_field.rce",
            "60_dim_hidden_period.rce",
            "61_dim_minimal.rce",
        ]
    );
    assert_eq!(names("FG8"), ["33_in_operator.rce"]);
    assert!(
        names("catch").is_empty(),
        "a program root's first tag is catch: {:?}",
        names("catch")
    );
    assert_eq!(names("global"), ["32_comptime_block.rce"]);
    assert_eq!(names("loop"), ["50_iterators.rce", "55_rcedb.rce"]);
    assert_eq!(names("exit"), ["02_hello_main.rce", "31_inventory.rce"]);
    assert_eq!(
        names("defect"),
        ["06_option.rce", "07_error_union.rce", "20_strings.rce"]
    );
    assert_eq!(
        names("round-trip"),
        [
            "03_fib_loop.rce",
            "08_traits.rce",
            "10_comptime.rce",
            "11_arena.rce",
            "18_while_bound.rce",
            "19_floats.rce",
            "22_tuples.rce",
            "24_lambda.rce",
            "25_labels.rce",
            "27_bitops.rce",
            "28_slices.rce",
            "36_multiline_brackets.rce",
            "41_arena_bound_wrapping.rce",
            "43_simd_lanes.rce",
            "45_byte_text.rce",
            "46_u64.rce",
            "49_loop_limit.rce",
        ]
    );
    assert_eq!(
        names("no program root"),
        ["09_generics.rce", "38_hex_bin.rce"]
    );
    assert_eq!(
        lowering,
        BTreeMap::from([
            ("13_spawn_chan.rce".into(), "not covered by the lowering: the type chan i64".into()),
            ("14_select.rce".into(), "not covered by the lowering: the type chan text".into()),
            ("15_actor.rce".into(), "not covered by the lowering: the type chan i64".into()),
            (
                "21_maps.rce".into(),
                "not covered by the lowering: an integer operation checked against ranges that differ at run time ([\"lit\", \"u32\"])".into(),
            ),
            ("29_nested_kind.rce".into(), "not covered by the lowering: a recursive type".into()),
            (
                "34_chan_types.rce".into(),
                "not covered by the lowering: the type chan (u64, text)".into(),
            ),
            ("37_actor_spawn.rce".into(), "not covered by the lowering: the type chan u64".into()),
            (
                "39_structural_recursion.rce".into(),
                "not covered by the lowering: a recursive type".into(),
            ),
            ("47_select_fair.rce".into(), "not covered by the lowering: the type chan u64".into()),
        ])
    );
    let tagged = |tag: &str| construct_tag.get(tag).map(|v| v.as_slice()).unwrap_or(&[]);
    assert_eq!(
        tagged("break"),
        [
            "01_counter.rce",
            "04_bank_account.rce",
            "05_kind_match.rce",
            "06_option.rce",
            "17_traffic_light.rce",
            "19_floats.rce",
            "23_patterns.rce",
            "30_state_machine_guarded.rce",
            "38_hex_bin.rce",
            "42_f32.rce",
            "44_device_region.rce",
            "51_refined_input.rce",
            "52_context_text.rce",
            "53_capability.rce",
            "54_secret.rce",
            "57_float_vectors.rce",
            "59_dim_redundant_field.rce",
            "60_dim_hidden_period.rce",
            "61_dim_minimal.rce",
        ]
    );
    assert_eq!(
        tagged("catch"),
        ["16_multi_region.rce", "56_utf8.rce", "58_float_keys.rce"]
    );
    assert_eq!(
        tagged("defect"),
        ["07_error_union.rce", "20_strings.rce", "33_in_operator.rce"]
    );
    assert_eq!(tagged("exit"), ["02_hello_main.rce", "31_inventory.rce"]);
    assert_eq!(
        tagged("global"),
        [
            "10_comptime.rce",
            "32_comptime_block.rce",
            "45_byte_text.rce",
            "46_u64.rce",
            "50_iterators.rce",
        ]
    );
    assert_eq!(tagged("loop"), ["55_rcedb.rce"]);
    assert_eq!(tagged("schedule defect"), ["22_tuples.rce"]);
    assert_eq!(tagged("schedule FG8"), ["43_simd_lanes.rce"]);
    assert_eq!(
        tagged("round-trip"),
        [
            "03_fib_loop.rce",
            "08_traits.rce",
            "11_arena.rce",
            "12_borrow.rce",
            "18_while_bound.rce",
            "24_lambda.rce",
            "25_labels.rce",
            "27_bitops.rce",
            "28_slices.rce",
            "36_multiline_brackets.rce",
            "40_extern.rce",
            "41_arena_bound_wrapping.rce",
            "48_linear_handle.rce",
            "49_loop_limit.rce",
        ]
    );
    assert!(
        !construct_tag.contains_key("out"),
        "out of bytes is emit, so out is not a whole-module tag: {:?}",
        construct_tag.get("out")
    );
    assert_eq!(
        root_tags.keys().cloned().collect::<Vec<_>>(),
        [
            "FG8",
            "break",
            "defect",
            "exit",
            "global",
            "loop",
            "no program root",
            "round-trip",
        ]
    );
    assert_eq!(
        construct_tag.keys().cloned().collect::<Vec<_>>(),
        [
            "break",
            "catch",
            "defect",
            "exit",
            "global",
            "loop",
            "round-trip",
            "schedule FG8",
            "schedule defect",
        ]
    );
    let tagged_files: usize = root_tags.values().map(|v| v.len()).sum();
    assert_eq!(
        lowering.len() + covered.len() + semantic.len() + tagged_files,
        programs,
        "a program is missing from the root census"
    );
}

#[test]
fn fn_table_constructs_and_the_schedule_is_refused_by_fg8() {
    // Contract section 4. The first form is applied. The schedule is not.
    // On 410bfd40 the file does not lower: 34c915fd keeps `fn freq`, and
    // `(m.get(w) catch 0) + 1` mixes a literal with u32. The census records
    // that refusal. `fn table` is the measure this test checks, so the
    // function is lowered from its own line in the same file.
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/21_maps.rce"),
    )
    .unwrap();
    let ast = rce_spec_check::program(&src).unwrap();
    let err = rce_spec_check::one::core_of(&ast).expect_err("21_maps lowered");
    assert_eq!(
        err,
        "not covered by the lowering: an integer operation checked against ranges that differ at run time ([\"lit\", \"u32\"])"
    );
    let line = src
        .lines()
        .find(|l| l.starts_with("fn table"))
        .expect("fn table");
    let ast = rce_spec_check::program(&format!("{line}\n")).unwrap();
    let maps = rce_spec_check::one::core_of(&ast).unwrap_or_else(|e| panic!("fn table: {e}"));
    let sub = bridge::closure(&maps, "fn table").expect("fn table");
    let f = bridge::construct(&sub).unwrap_or_else(|e| panic!("first construct: {}", e.line()));
    assert!(
        bridge::body_break_rewrite(),
        "the search loop's break was not restructured into a flag"
    );
    assert!(matches!(
        form::check(fir::encode(&f).as_bytes(), None),
        form::Outcome::Applied(_)
    ));
    let (value, _) = core_eval(&sub, vec![]);
    assert_eq!(
        value,
        Ok(Val::List(vec![
            Val::Tup(vec![Val::bytes(b"a"), Val::Int(1)]),
            Val::Tup(vec![Val::bytes(b"b"), Val::Int(2)]),
        ])),
        "core eval of fn table is the value a later three-way check has to match"
    );
    let s = bridge::schedule(&f).unwrap_or_else(|e| panic!("schedule: {}", e.line()));
    wf::check(&s).unwrap_or_else(|e| panic!("schedule is not well formed: {e}"));
    let e = bridge::construct(&s).expect_err("the scheduled measure was applied");
    assert_eq!(e.tag(), "FG8", "{}", e.line());
    assert_eq!(e.at, "node 3", "{}", e.line());
    assert!(
        e.reason.contains(
            "the body region does not return i raised by exactly one whenever it continues"
        ),
        "{}",
        e.line()
    );
    // A future applied schedule matches `f` byte for byte and matches `value`.
    // It does not enter the syntactic floor while `body_break_rewrite` is set.
}

#[test]
fn a_break_out_of_a_block_is_refused_and_the_continuation_runs() {
    // Contract section 3.1. Depth 0 leaves the block. The return after it runs.
    let mut b = FB::new(vec![Ty::Bool], i64ty());
    let c = b.arg(0);
    let r = b.var(i64ty(), build::lit(0));
    b.open();
    b.if_(
        c,
        |b| {
            b.brk(0);
            Ok(())
        },
        |_| Ok(()),
    )
    .unwrap();
    b.set(r, build::lit(1));
    let inner = b.close();
    b.push(Stmt::Block(inner));
    let rv = b.let_(Atom::V(r));
    b.ret(rv);
    let m = module("block", b.finish());
    assert_break_block(&m);
    let (t, _) = core_eval(&m, vec![Val::Bool(true)]);
    let (f, _) = core_eval(&m, vec![Val::Bool(false)]);
    assert_eq!(t, Ok(Val::Int(0)), "the break arm skipped the store");
    assert_eq!(
        f,
        Ok(Val::Int(1)),
        "the fall arm stored, and the continuation returned it"
    );

    // Ticks: one before the block, one after the break inside it, one after the block.
    let (tick, tkey) = tick_parts();
    let key = tkey.clone();
    let mut b = FB::new(vec![Ty::Bool], i64ty());
    let c = b.arg(0);
    let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
    b.open();
    b.if_(
        c,
        |b| {
            b.brk(0);
            Ok(())
        },
        |_| Ok(()),
    )
    .unwrap();
    let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
    let inner = b.close();
    b.push(Stmt::Block(inner));
    let _ = b.call(FnRef::Key(key, 0), Ty::Bool, vec![build::lit(1)]);
    b.ret(build::lit(0));
    let m = assemble(tick, tkey, "ticks", b.finish());
    assert_break_block(&m);
    let (_t, fuel_t) = core_eval(&m, vec![Val::Bool(true)]);
    let (_f, fuel_f) = core_eval(&m, vec![Val::Bool(false)]);
    let spent = |g: Option<Val>| match g {
        Some(Val::Int(n)) => feval::FUEL as i128 - n,
        other => panic!("fuel {other:?}"),
    };
    // Break path: the tick before the block and the tick after it. Not the tick after the break.
    assert_eq!(spent(fuel_t), 2, "a tick after the break was charged");
    // Fall path: all three ticks.
    assert_eq!(spent(fuel_f), 3, "the fall path did not charge every tick");
}

#[test]
fn a_break_that_names_the_loop_from_inside_a_block_is_refused() {
    // Contract section 3.2. `break 1` inside the body block names the loop.
    // On 410bfd40 the first form is that flag: it is applied, and it
    // returns the core's value. The schedule is still refused by FG8.
    // The node is 1, measured on this commit; the reason is unchanged.
    // The increment does not run on the break path.
    let build = |with_tick: bool| {
        let (tick, tkey) = tick_parts();
        let key = tkey.clone();
        let t = Ty::u64();
        let mut b = FB::new(vec![t.clone(), Ty::Bool], t.clone());
        let n = b.arg(0);
        let stop = b.arg(1);
        let i = b.var(t.clone(), build::lit(0));
        b.loop_(|b| {
            let iv = b.let_(Atom::V(i));
            let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
            b.exit_unless(more)?;
            b.open();
            b.when(stop.clone(), |b| {
                b.brk(1);
                Ok(())
            })?;
            if with_tick {
                let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
            }
            let inner = b.close();
            b.push(Stmt::Block(inner));
            if with_tick {
                let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
            }
            let i1 = b.add_in(iv, build::lit(1), t.clone())?;
            b.set(i, i1);
            Ok(())
        })
        .unwrap();
        let r = b.let_(Atom::V(i));
        b.ret(r);
        if with_tick {
            assemble(tick, tkey, "depth", b.finish())
        } else {
            module("depth", b.finish())
        }
    };
    let m = build(false);
    let f = bridge::construct(&m).unwrap_or_else(|e| panic!("first construct: {}", e.line()));
    assert!(
        bridge::body_break_rewrite(),
        "break 1 was not restructured into the loop flag"
    );
    assert_applied(&f);
    assert_values(&m, &f, vec![Val::Int(0), Val::Bool(true)]);
    assert_values(&m, &f, vec![Val::Int(4), Val::Bool(true)]);
    assert_values(&m, &f, vec![Val::Int(4), Val::Bool(false)]);
    let (z, _) = core_eval(&m, vec![Val::Int(0), Val::Bool(true)]);
    let (one, _) = core_eval(&m, vec![Val::Int(4), Val::Bool(true)]);
    let (many, _) = core_eval(&m, vec![Val::Int(4), Val::Bool(false)]);
    assert_eq!(z, Ok(Val::Int(0)), "zero iterations");
    assert_eq!(one, Ok(Val::Int(0)), "the break skipped the increment");
    assert_eq!(many, Ok(Val::Int(4)), "four iterations step the index");
    schedule_fg8(&f, "node 1");

    let m = build(true);
    let f = bridge::construct(&m).unwrap_or_else(|e| panic!("first construct: {}", e.line()));
    assert!(bridge::body_break_rewrite());
    assert_applied(&f);
    assert_values(&m, &f, vec![Val::Int(4), Val::Bool(true)]);
    assert_values(&m, &f, vec![Val::Int(4), Val::Bool(false)]);
    assert_values(&m, &f, vec![Val::Int(0), Val::Bool(false)]);
    schedule_fg8(&f, "node 1");
    let spent = |args: Vec<Val>| {
        let (_, g) = core_eval(&m, args);
        match g {
            Some(Val::Int(n)) => feval::FUEL as i128 - n,
            other => panic!("fuel {other:?}"),
        }
    };
    // Stop on entry to the first iteration: the tick inside the block is after
    // the break, and the tick after the block is the loop continuation. Neither runs.
    assert_eq!(spent(vec![Val::Int(4), Val::Bool(true)]), 0);
    // No stop: each of 4 iterations charges the inner tick and the tick after the block.
    assert_eq!(spent(vec![Val::Int(4), Val::Bool(false)]), 8);
    assert_eq!(spent(vec![Val::Int(0), Val::Bool(false)]), 0);
}

#[test]
fn a_flat_measure_with_a_body_break_is_refused_by_fg8() {
    // Contract section 4.3 and shortcut 1's neighbour: the continue path does
    // raise the index, and the first construction still fails FG8. `mset`
    // shows the closed shape can accept that meaning. This statement order
    // does not reach it. The refusal stays until it does.
    let build = |with_tick: bool| {
        let (tick, tkey) = tick_parts();
        let key = tkey.clone();
        let t = Ty::u64();
        let mut b = FB::new(vec![t.clone(), Ty::Bool], t.clone());
        let n = b.arg(0);
        let stop = b.arg(1);
        let i = b.var(t.clone(), build::lit(0));
        b.loop_(|b| {
            let iv = b.let_(Atom::V(i));
            let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
            b.exit_unless(more)?;
            if with_tick {
                let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
            }
            b.when(stop.clone(), |b| {
                b.brk(0);
                Ok(())
            })?;
            if with_tick {
                let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
            }
            let i1 = b.add_in(iv, build::lit(1), t.clone())?;
            b.set(i, i1);
            Ok(())
        })
        .unwrap();
        let rv = b.let_(Atom::V(i));
        b.ret(rv);
        if with_tick {
            assemble(tick, tkey, "flat", b.finish())
        } else {
            module("flat", b.finish())
        }
    };
    let m = build(false);
    let e = refuse(&m);
    assert_eq!(e.tag(), "FG8", "{}", e.line());
    assert!(
        e.reason
            .contains("raised by exactly one whenever it continues"),
        "{}",
        e.line()
    );
    assert_ne!(
        e.tag(),
        "bound",
        "a body break was rewritten into a pure count"
    );
    let (z, _) = core_eval(&m, vec![Val::Int(0), Val::Bool(false)]);
    let (stop, _) = core_eval(&m, vec![Val::Int(4), Val::Bool(true)]);
    let (one, _) = core_eval(&m, vec![Val::Int(1), Val::Bool(false)]);
    let (many, _) = core_eval(&m, vec![Val::Int(4), Val::Bool(false)]);
    assert_eq!(z, Ok(Val::Int(0)));
    assert_eq!(stop, Ok(Val::Int(0)));
    assert_eq!(one, Ok(Val::Int(1)));
    assert_eq!(many, Ok(Val::Int(4)));

    let m = build(true);
    let e = refuse(&m);
    assert_eq!(e.tag(), "FG8", "{}", e.line());
    let spent = |args: Vec<Val>| {
        let (_, g) = core_eval(&m, args);
        match g {
            Some(Val::Int(n)) => feval::FUEL as i128 - n,
            other => panic!("fuel {other:?}"),
        }
    };
    assert_eq!(
        spent(vec![Val::Int(0), Val::Bool(false)]),
        0,
        "zero iterations"
    );
    // One iteration, break before the second tick and before the increment.
    assert_eq!(spent(vec![Val::Int(4), Val::Bool(true)]), 1);
    assert_eq!(spent(vec![Val::Int(1), Val::Bool(false)]), 2);
    assert_eq!(spent(vec![Val::Int(4), Val::Bool(false)]), 8);
}

#[test]
fn a_nested_return_outside_a_loop_round_trips() {
    // Contract section 3.3. This shape is already a Branch. Both selectors agree.
    let mut b = FB::new(vec![Ty::Bool], i64ty());
    let c = b.arg(0);
    b.if_(
        c,
        |b| {
            b.ret(build::lit(1));
            Ok(())
        },
        |_| Ok(()),
    )
    .unwrap();
    b.ret(build::lit(0));
    let m = module("ret", b.finish());
    wf::check(&m).unwrap();
    let f = bridge::construct(&m).unwrap_or_else(|e| panic!("{}", e.line()));
    assert!(!bridge::body_break_rewrite());
    let s = bridge::schedule(&f).unwrap_or_else(|e| panic!("{}", e.line()));
    let back = bridge::construct(&s).unwrap_or_else(|e| panic!("{}", e.line()));
    assert_eq!(fir::encode(&f), fir::encode(&back));
    for (bit, want) in [(true, 1), (false, 0)] {
        let (got, _) = core_eval(&m, vec![Val::Bool(bit)]);
        assert_eq!(got, Ok(Val::Int(want)));
        let (k, i) = by_arity(&m, 1);
        let _ = (k, i);
        let got = feval::apply(
            &f,
            &form_key(&f, 1).0,
            form_key(&f, 1).1,
            vec![Val::Bool(bit)],
        )
        .unwrap();
        assert_eq!(got, vec![Val::Int(want)]);
        let (sg, _) = core_eval(&s, vec![Val::Bool(bit)]);
        assert_eq!(sg, Ok(Val::Int(want)));
    }
}

fn form_key(m: &fir::Module, nargs: usize) -> (String, u32) {
    let mut best: Option<(String, u32, usize)> = None;
    for (k, g) in &m.defs {
        for (i, f) in g.fns.iter().enumerate() {
            let n = f.ins.iter().filter(|t| **t != fir::BTy::Order).count();
            if n == nargs {
                let size = f.body.nodes.len();
                if best.as_ref().is_none_or(|b| size > b.2) {
                    best = Some((k.clone(), i as u32, size));
                }
            }
        }
    }
    let (k, i, _) = best.unwrap();
    (k, i)
}

#[test]
fn a_return_inside_a_loop_is_a_defect_of_the_arm_widths() {
    // Contract section 3.3. The bridge reports a defect of the arm widths.
    // A return is not the loop flag, and the core results stay the contract.
    let t = Ty::u64();
    let mut b = FB::new(vec![t.clone(), Ty::Bool], t.clone());
    let n = b.arg(0);
    let stop = b.arg(1);
    let i = b.var(t.clone(), build::lit(0));
    b.loop_(|b| {
        let iv = b.let_(Atom::V(i));
        let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
        b.exit_unless(more)?;
        b.when(stop.clone(), |b| {
            b.ret(build::lit(7));
            Ok(())
        })?;
        let i1 = b.add_in(iv, build::lit(1), t.clone())?;
        b.set(i, i1);
        Ok(())
    })
    .unwrap();
    let rv = b.let_(Atom::V(i));
    b.ret(rv);
    let m = module("retloop", b.finish());
    let e = refuse(&m);
    assert_eq!(e.tag(), "defect", "{}", e.line());
    assert_eq!(
        e.reason,
        "defect: branch arms return 1, 2 values for 1 outputs",
        "{}",
        e.line()
    );
    let (stop_v, _) = core_eval(&m, vec![Val::Int(4), Val::Bool(true)]);
    let (run, _) = core_eval(&m, vec![Val::Int(4), Val::Bool(false)]);
    let (z, _) = core_eval(&m, vec![Val::Int(0), Val::Bool(true)]);
    assert_eq!(stop_v, Ok(Val::Int(7)));
    assert_eq!(run, Ok(Val::Int(4)));
    assert_eq!(z, Ok(Val::Int(0)));
}

#[test]
fn an_owned_list_through_a_block_break_is_refused() {
    // Contract section 3.1 and shortcut 8. Both arms move the list. The
    // continuation returns the arm's list. The bridge refuses the break.
    let e = i64ty();
    let lt = Ty::list(e.clone());
    let mut b = FB::new(vec![Ty::Bool, lt.clone()], lt.clone());
    let c = b.arg(0);
    let l = b.var(lt, b.arg(1));
    b.open();
    b.if_(
        c,
        |b| {
            let cur = b.take(l);
            let p = b.p("list.push", vec![e.clone()], vec![cur, build::lit(1)])?;
            b.set(l, p);
            b.brk(0);
            Ok(())
        },
        |b| {
            let cur = b.take(l);
            let p = b.p("list.push", vec![e.clone()], vec![cur, build::lit(2)])?;
            b.set(l, p);
            Ok(())
        },
    )
    .unwrap();
    let inner = b.close();
    b.push(Stmt::Block(inner));
    let rv = b.take(l);
    b.ret(rv);
    let m = module("owned", b.finish());
    assert_break_block(&m);
    let start = Val::List(vec![Val::Int(7)]);
    let (yes, _) = core_eval(&m, vec![Val::Bool(true), start.clone()]);
    let (no, _) = core_eval(&m, vec![Val::Bool(false), start]);
    assert_eq!(yes, Ok(Val::List(vec![Val::Int(7), Val::Int(1)])));
    assert_eq!(no, Ok(Val::List(vec![Val::Int(7), Val::Int(2)])));
}

#[test]
fn shortcuts_that_admit_the_wrong_program_stay_refused() {
    // Contract section 5. None of these is an applied measure or an applied get.
    let fg8_not_count = |m: CoreModule, what: &str| {
        let e = refuse(&m);
        assert_eq!(e.tag(), "FG8", "{what}: {}", e.line());
        assert!(
            e.reason.contains("raised by exactly one"),
            "{what}: {}",
            e.line()
        );
        assert_ne!(e.tag(), "bound", "{what} became a count");
    };

    // Continue path that does not always raise the index. Not a break.
    let t = Ty::u64();
    let mut b = FB::new(vec![Ty::Bool, t.clone()], t.clone());
    let c = b.arg(0);
    let n = b.arg(1);
    let i = b.var(t.clone(), build::lit(0));
    b.loop_(|b| {
        let iv = b.let_(Atom::V(i));
        let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
        b.exit_unless(more)?;
        b.when(c.clone(), |b| {
            let i1 = b.add_in(iv.clone(), build::lit(1), t.clone())?;
            b.set(i, i1);
            Ok(())
        })?;
        Ok(())
    })
    .unwrap();
    let rv = b.let_(Atom::V(i));
    b.ret(rv);
    fg8_not_count(module("cond", b.finish()), "conditional increment");

    // The end is stored. The loop still terminates. It is not a measure.
    let mut b = FB::new(vec![t.clone()], t.clone());
    let n = b.arg(0);
    let end = b.var(t.clone(), n);
    let i = b.var(t.clone(), build::lit(0));
    b.loop_(|b| {
        let iv = b.let_(Atom::V(i));
        let ev = b.let_(Atom::V(end));
        let more = b.int_cmp("int.lt", iv.clone(), ev)?;
        b.exit_unless(more)?;
        b.set(end, iv.clone());
        let i1 = b.add_in(iv, build::lit(1), t.clone())?;
        b.set(i, i1);
        Ok(())
    })
    .unwrap();
    let rv = b.let_(Atom::V(i));
    b.ret(rv);
    let m = module("mutend", b.finish());
    let e = refuse(&m);
    assert_eq!(e.tag(), "loop", "mutated end: {}", e.line());
    assert!(
        e.reason.contains("none of the two-region shapes"),
        "mutated end: {}",
        e.line()
    );
    let (z, _) = core_eval(&m, vec![Val::Int(0)]);
    let (n4, _) = core_eval(&m, vec![Val::Int(4)]);
    assert_eq!(z, Ok(Val::Int(0)));
    assert_eq!(n4, Ok(Val::Int(1)), "the mutated end stops the next test");

    // A pure count that breaks. The flag is not a use of ω. O8.
    let ct = build::count_t();
    let mut b = FB::new(vec![ct.clone()], ct.clone());
    let bound = b.arg(0);
    let i = b.var(ct.clone(), build::lit(0));
    let cnt = b.counter();
    b.loop_(|b| {
        b.count(cnt, bound)?;
        let iv = b.let_(Atom::V(i));
        let stop = b.int_cmp("int.eq", iv, build::lit(0))?;
        b.exit_if(stop)?;
        Ok(())
    })
    .unwrap();
    let rv = b.let_(Atom::V(i));
    b.ret(rv);
    let m = module("pure", b.finish());
    let e = refuse(&m);
    assert_eq!(e.tag(), "bound", "{}", e.line());
    assert!(
        !bridge::body_break_rewrite(),
        "a refused pure count is not an admission"
    );

    // A depth that names nothing. The core rejects it.
    let mut b = FB::new(vec![], i64ty());
    b.loop_(|b| {
        b.brk(2);
        Ok(())
    })
    .unwrap();
    b.ret(build::lit(0));
    let m = module("orphan", b.finish());
    match wf::check(&m) {
        Err(e) => assert!(e.contains("W2") || e.starts_with("W"), "{e}"),
        Ok(()) => {
            let e = bridge::construct(&m).expect_err("orphan break");
            assert_eq!(e.tag(), "break", "{}", e.line());
        }
    }

    // Missing guard, stale index, and a guard on a different list.
    let bad = |body: Vec<Stmt>, label: &str| {
        let f = CoreFn {
            params: vec![],
            ret: i64ty(),
            body,
        };
        let m = module(label, f);
        match wf::check(&m) {
            Err(e) => assert!(
                e.contains(
                    "W4: list.get is not dominated by a test that the index is below the length"
                ),
                "{label}: {e}"
            ),
            Ok(()) => {
                let e = bridge::construct(&m).expect_err(label);
                assert_eq!(e.tag(), "FG8", "{label}: {}", e.line());
            }
        }
    };
    let e = i64ty();
    // list.get with no test.
    bad(
        vec![
            Stmt::Prim("list.new".into(), vec![e.clone()], vec![]),
            Stmt::Prim(
                "list.push".into(),
                vec![e.clone()],
                vec![Atom::V(0), build::lit(7)],
            ),
            Stmt::Prim(
                "list.get".into(),
                vec![e.clone()],
                vec![Atom::V(1), build::lit(0)],
            ),
            Stmt::Ret(Atom::V(2)),
        ],
        "missing guard",
    );
    // Guard 0 < len, then get at 1.
    let mut b = FB::new(vec![], e.clone());
    let xs = b.p("list.new", vec![e.clone()], vec![]).unwrap();
    let xs = b
        .p("list.push", vec![e.clone()], vec![xs, build::lit(7)])
        .unwrap();
    let n = b.p("list.len", vec![e.clone()], vec![xs.clone()]).unwrap();
    let ok = b.int_cmp("int.lt", build::lit(0), n).unwrap();
    b.guard_halt_unless(ok, "index").unwrap();
    let i1 = b.add_in(build::lit(0), build::lit(1), Ty::u64()).unwrap();
    let g = b.p("list.get", vec![e.clone()], vec![xs, i1]).unwrap();
    b.ret(g);
    let m = module("stale", b.finish());
    let err = wf::check(&m).expect_err("a stale index was well formed");
    assert!(
        err.contains("W4: list.get is not dominated by a test that the index is below the length"),
        "{err}"
    );
    // Two list parameters. The length test is on the first. The get is of
    // the second, at a literal index. Equal elements are not the question:
    // the lists are different values, and a just-built list of one element
    // would be in domain by DF1 without any guard.
    let lt = Ty::list(e.clone());
    let mut b = FB::new(vec![lt.clone(), lt], e.clone());
    let a = b.arg(0);
    let c = b.arg(1);
    let n = b.p("list.len", vec![e.clone()], vec![a.clone()]).unwrap();
    let ok = b.int_cmp("int.lt", build::lit(0), n).unwrap();
    b.guard_halt_unless(ok, "index").unwrap();
    let g = b
        .p("list.get", vec![e.clone()], vec![c.clone(), build::lit(0)])
        .unwrap();
    b.drop(a).unwrap();
    b.drop(c).unwrap();
    b.ret(g);
    let m = module("foreign", b.finish());
    let err = wf::check(&m).expect_err("a guard on one list was accepted for a get of another");
    assert!(
        err.contains("W4: list.get is not dominated by a test that the index is below the length"),
        "{err}"
    );

    // The same list, read back by list.len, guards a get of that list.
    let mut b = FB::new(vec![], e.clone());
    let xs = b.p("list.new", vec![e.clone()], vec![]).unwrap();
    let xs = b
        .p("list.push", vec![e.clone()], vec![xs, build::lit(7)])
        .unwrap();
    let v = b.list_at(xs.clone(), build::lit(0)).unwrap();
    b.drop(xs).unwrap();
    b.ret(v);
    let m = module("same", b.finish());
    wf::check(&m).unwrap_or_else(|err| panic!("same-value get is not well formed: {err}"));
    let f = bridge::construct(&m).unwrap_or_else(|err| panic!("same-value get: {}", err.line()));
    assert!(matches!(
        form::check(fir::encode(&f).as_bytes(), None),
        form::Outcome::Applied(_)
    ));
    let (got, _) = core_eval(&m, vec![]);
    assert_eq!(got, Ok(Val::Int(7)));
}

#[test]
fn hello_and_dot4_keep_their_break_sentences() {
    // Contract section 1. On 410bfd40 `fn main`'s first refusal is the
    // return inside `rt.escaped`, not the break in `rt.debug`. A return is
    // not the loop flag. `fn dot4` still refuses with the break sentence.
    let hello = load_example("02_hello_main.rce");
    let sub = bridge::closure(&hello, "fn main").unwrap();
    let e = bridge::construct(&sub).expect_err("hello");
    assert_eq!(e.tag(), "exit", "{}", e.line());
    assert_eq!(e.at, "rt.escaped", "{}", e.line());
    assert_eq!(
        e.reason,
        "exit: a loop body does not fall through",
        "{}",
        e.line()
    );

    let dev = load_example("44_device_region.rce");
    let sub = bridge::closure(&dev, "fn dot4").unwrap();
    let e = bridge::construct(&sub).expect_err("dot4");
    assert_eq!(e.tag(), "break", "{}", e.line());
    assert_eq!(e.at, "rt.f64.parts", "{}", e.line());
    assert_eq!(e.reason, BREAK_OUTSIDE, "{}", e.line());
}

#[test]
fn a_break_that_leaves_two_loops_is_refused() {
    // Contract section 3.2. The inner loop's `break 1` names the outer loop
    // as well. The bridge refuses it. It does not build a measure.
    let mut b = FB::new(vec![build::count_t()], build::count_t());
    let bound = b.arg(0);
    let cnt = b.counter();
    let (tick, tkey) = tick_parts();
    let key = tkey.clone();
    b.loop_(|b| {
        b.count(cnt, bound)?;
        let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
        b.loop_(|b| {
            b.brk(1);
            Ok(())
        })?;
        Ok(())
    })
    .unwrap();
    b.ret(build::lit(0));
    let m = assemble(tick, tkey, "nest", b.finish());
    let e = refuse(&m);
    // The inner loop is not a count or a measure, so it is not a recognised
    // loop. d77075a2 removed the "leaves more than one loop" sentence when
    // a recognised loop started carrying the residual. This shape stays
    // refused. Measured on 410bfd40.
    assert_eq!(e.tag(), "loop", "{}", e.line());
    assert_eq!(e.reason, UNRECOGNISED_LOOP, "{}", e.line());
}

fn loop_bound(r: &Result<Val, String>) -> bool {
    match r {
        Err(e) => e.contains("[108, 111, 111, 112, 32, 98, 111, 117, 110, 100]"),
        Ok(_) => false,
    }
}

fn spent(fuel: Option<Val>) -> i128 {
    match fuel {
        Some(Val::Int(n)) => feval::FUEL as i128 - n,
        other => panic!("fuel {other:?}"),
    }
}

/// Two loops, `break 1` from the inner one. The return is
/// `after_inner + 2*after_outer`. A count at the top of the outer loop is
/// this iteration's bound work; the next iteration's count must not run
/// when the break leaves the outer loop.
fn two_loop_exit(with_tick: bool, ret_counter: bool) -> CoreModule {
    let (tick, tkey) = tick_parts();
    let key = tkey.clone();
    let ct = build::count_t();
    let ret = if ret_counter { ct.clone() } else { i64ty() };
    let mut b = FB::new(vec![ct.clone(), Ty::Bool], ret);
    let bound = b.arg(0);
    let stop = b.arg(1);
    let cnt = b.counter();
    let after_inner = b.var(i64ty(), build::lit(0));
    let after_outer = b.var(i64ty(), build::lit(0));
    b.loop_(|b| {
        b.count(cnt, bound.clone())?;
        if with_tick {
            let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
        }
        b.loop_(|b| {
            b.if_(
                stop.clone(),
                |b| {
                    b.brk(1);
                    Ok(())
                },
                |_| Ok(()),
            )?;
            if with_tick {
                let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
            }
            b.brk(0);
            Ok(())
        })?;
        b.set(after_inner, build::lit(1));
        if with_tick {
            let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
        }
        Ok(())
    })
    .unwrap();
    b.set(after_outer, build::lit(1));
    if with_tick {
        let _ = b.call(FnRef::Key(key, 0), Ty::Bool, vec![build::lit(1)]);
    }
    if ret_counter {
        let c = b.let_(Atom::V(cnt));
        b.ret(c);
        return module("counter", b.finish());
    }
    let inner = b.let_(Atom::V(after_inner));
    let outer = b.let_(Atom::V(after_outer));
    let outer2 = b.add_in(outer.clone(), outer, i64ty()).unwrap();
    let sum = b.add_in(inner, outer2, i64ty()).unwrap();
    b.ret(sum);
    if with_tick {
        assemble(tick, tkey, "markers", b.finish())
    } else {
        module("markers", b.finish())
    }
}

#[test]
fn two_loop_exit_core_results() {
    // The refusal test above does not say what the core does. These results
    // are that witness. The bridge refusal is unchanged.
    let markers = two_loop_exit(false, false);
    let e = refuse(&markers);
    assert_eq!(e.tag(), "loop", "{}", e.line());
    assert_eq!(e.reason, UNRECOGNISED_LOOP, "{}", e.line());
    let (leave, _) = core_eval(&markers, vec![Val::Int(1), Val::Bool(true)]);
    let (stay, _) = core_eval(&markers, vec![Val::Int(1), Val::Bool(false)]);
    let (early, _) = core_eval(&markers, vec![Val::Int(0), Val::Bool(true)]);
    // after_inner is 0, after_outer is 1. The continuation after the outer loop ran.
    assert_eq!(leave, Ok(Val::Int(2)));
    // Inner break 0 only. The outer loop reaches its next count and halts.
    assert!(loop_bound(&stay), "{stay:?}");
    // The count stands before the break, so bound 0 halts before the exit.
    assert!(loop_bound(&early), "{early:?}");

    let counter = two_loop_exit(false, true);
    let (once, _) = core_eval(&counter, vec![Val::Int(1), Val::Bool(true)]);
    let (again, _) = core_eval(&counter, vec![Val::Int(1), Val::Bool(false)]);
    let (wide, _) = core_eval(&counter, vec![Val::Int(10), Val::Bool(true)]);
    assert_eq!(once, Ok(Val::Int(1)), "one count, then both loops are left");
    assert!(loop_bound(&again), "{again:?}");
    assert_eq!(
        wide,
        Ok(Val::Int(1)),
        "a wide bound still does not start a second count"
    );

    let fuelled = two_loop_exit(true, false);
    let (leave, fuel_leave) = core_eval(&fuelled, vec![Val::Int(1), Val::Bool(true)]);
    let (stay, fuel_stay) = core_eval(&fuelled, vec![Val::Int(1), Val::Bool(false)]);
    let (wide, fuel_wide) = core_eval(&fuelled, vec![Val::Int(10), Val::Bool(true)]);
    assert_eq!(leave, Ok(Val::Int(2)));
    // Tick before the break, and the tick after the outer loop. Not the tick
    // after the break inside the inner loop, and not the tick after the inner loop.
    assert_eq!(spent(fuel_leave), 2);
    assert!(loop_bound(&stay), "{stay:?}");
    // First iteration charges three ticks, then the next count halts before its tick.
    assert_eq!(spent(fuel_stay), 3);
    assert_eq!(wide, Ok(Val::Int(2)));
    assert_eq!(spent(fuel_wide), 2);
}

/// `loop { if { block { break depth } } }`. The packed result is
/// `tail + 2*after_block + 4*after_if + 8*after_loop + 16*steps`.
fn block_if_loop(depth: u32) -> CoreModule {
    let t = Ty::u64();
    let mut b = FB::new(vec![t.clone(), Ty::Bool, Ty::Bool], i64ty());
    let n = b.arg(0);
    let gate = b.arg(1);
    let take = b.arg(2);
    let steps = b.var(t.clone(), build::lit(0));
    let tail = b.var(i64ty(), build::lit(0));
    let after_block = b.var(i64ty(), build::lit(0));
    let after_if = b.var(i64ty(), build::lit(0));
    let after_loop = b.var(i64ty(), build::lit(0));
    b.loop_(|b| {
        let iv = b.let_(Atom::V(steps));
        let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
        b.exit_unless(more)?;
        b.if_(
            gate.clone(),
            |b| {
                b.open();
                b.if_(
                    take.clone(),
                    |b| {
                        b.brk(depth);
                        Ok(())
                    },
                    |_| Ok(()),
                )?;
                b.set(tail, build::lit(1));
                let inner = b.close();
                b.push(Stmt::Block(inner));
                b.set(after_block, build::lit(1));
                Ok(())
            },
            |_| Ok(()),
        )?;
        b.set(after_if, build::lit(1));
        let s1 = b.add_in(iv, build::lit(1), t.clone())?;
        b.set(steps, s1);
        Ok(())
    })
    .unwrap();
    b.set(after_loop, build::lit(1));
    let t1 = b.let_(Atom::V(tail));
    let a1 = b.let_(Atom::V(after_block));
    let a1 = b.add_in(a1.clone(), a1, i64ty()).unwrap();
    let a2 = b.let_(Atom::V(after_if));
    let a2 = b.add_in(a2.clone(), a2.clone(), i64ty()).unwrap();
    let a2 = b.add_in(a2.clone(), a2, i64ty()).unwrap();
    let a3 = b.let_(Atom::V(after_loop));
    let a3 = b.add_in(a3.clone(), a3.clone(), i64ty()).unwrap();
    let a3 = b.add_in(a3.clone(), a3.clone(), i64ty()).unwrap();
    let a3 = b.add_in(a3.clone(), a3, i64ty()).unwrap();
    let s0 = b.let_(Atom::V(steps));
    let s0 = b.cast_in(s0, i64ty()).unwrap();
    let s = b.add_in(s0.clone(), s0, i64ty()).unwrap();
    let s = b.add_in(s.clone(), s.clone(), i64ty()).unwrap();
    let s = b.add_in(s.clone(), s.clone(), i64ty()).unwrap();
    let s = b.add_in(s.clone(), s, i64ty()).unwrap();
    let acc = b.add_in(t1, a1, i64ty()).unwrap();
    let acc = b.add_in(acc, a2, i64ty()).unwrap();
    let acc = b.add_in(acc, a3, i64ty()).unwrap();
    let acc = b.add_in(acc, s, i64ty()).unwrap();
    b.ret(acc);
    module("bifl", b.finish())
}

#[test]
fn block_if_loop_distinguishes_depth_from_residual() {
    // Source depth 0 names the block. The if is not a level. Depth 1 names
    // the loop. Depth 2 names nothing: the core rejects it before it runs.
    // On 410bfd40 depth 0 and depth 1 construct. The form and the scheduled
    // core return the core's value. Depth 1's schedule is still FG8.
    let args = vec![Val::Int(1), Val::Bool(true), Val::Bool(true)];
    let fall_args = vec![Val::Int(1), Val::Bool(true), Val::Bool(false)];
    let depth0 = block_if_loop(0);
    let f = bridge::construct(&depth0).unwrap_or_else(|e| panic!("depth 0: {}", e.line()));
    assert!(
        bridge::body_break_rewrite(),
        "break 0 was not the block's local exit"
    );
    assert_applied(&f);
    assert_values(&depth0, &f, args.clone());
    assert_values(&depth0, &f, fall_args.clone());
    let s = bridge::schedule(&f).unwrap_or_else(|e| panic!("schedule: {}", e.line()));
    wf::check(&s).unwrap_or_else(|e| panic!("schedule is not well formed: {e}"));
    let back = bridge::construct(&s).unwrap_or_else(|e| panic!("schedule construct: {}", e.line()));
    assert_applied(&back);
    assert_scheduled_values(&depth0, &s, args.clone());
    assert_scheduled_values(&depth0, &s, fall_args);
    let (take, _) = core_eval(&depth0, args.clone());
    let (fall, _) = core_eval(
        &depth0,
        vec![Val::Int(1), Val::Bool(true), Val::Bool(false)],
    );
    // Break 0: tail skipped, after_block, after_if, after_loop, one step.
    assert_eq!(take, Ok(Val::Int(30)));
    // No break taken: the tail runs as well. Same continuations, different control.
    assert_eq!(fall, Ok(Val::Int(31)));

    let depth1 = block_if_loop(1);
    let f = bridge::construct(&depth1).unwrap_or_else(|e| panic!("depth 1: {}", e.line()));
    assert!(
        bridge::body_break_rewrite(),
        "depth 1 was not the loop flag"
    );
    assert_applied(&f);
    assert_values(&depth1, &f, args.clone());
    schedule_fg8(&f, "node 1");
    let (leave, _) = core_eval(&depth1, args);
    // Only after_loop. The if did not consume a level. steps stayed 0.
    assert_eq!(leave, Ok(Val::Int(8)));

    let deep = block_if_loop(2);
    let err = wf::check(&deep).expect_err("break 2");
    assert!(
        err.contains("W2: break 2 names no enclosing block or loop"),
        "{err}"
    );
    // A residual that still names a construct the function does not have.
    // The evaluator's sentence for a break that leaves the function is
    // "break outside a block or loop". W2 rejects these cores first, so
    // that sentence is not what a well-formed module returns.
    let mut b = FB::new(vec![], i64ty());
    b.open();
    b.brk(1);
    let inner = b.close();
    b.push(Stmt::Block(inner));
    b.ret(build::lit(1));
    let err = wf::check(&module("esc", b.finish())).unwrap_err();
    assert!(
        err.contains("W2: break 1 names no enclosing block or loop"),
        "{err}"
    );
    let mut b = FB::new(vec![], i64ty());
    b.loop_(|b| {
        b.brk(1);
        Ok(())
    })
    .unwrap();
    b.ret(build::lit(1));
    let err = wf::check(&module("esc2", b.finish())).unwrap_err();
    assert!(
        err.contains("W2: break 1 names no enclosing block or loop"),
        "{err}"
    );
}
