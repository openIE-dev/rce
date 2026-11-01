//! `rce fmt nameless` (10.6, 6.6 concept record): the projection is
//! accepted whenever the source is, has the same Form up to names (same
//! log hashes, T7), is idempotent, and renames the declared names.
//! Not yet run in the session that wrote it (no toolchain there).
use rce_spec_check::cli;
use std::path::{Path, PathBuf};

fn examples() -> Vec<PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("examples");
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    v.sort();
    v
}

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap())
}

fn tmp(name: &str, text: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-nameless-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn counter_is_renamed_by_kind() {
    let (code, out) = rce(&["fmt", "nameless", "../examples/01_counter.rce"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("state r0"), "{out}");
    assert!(out.contains("event e0") && out.contains("event e1"), "{out}");
    assert!(out.contains("error x0"), "{out}");
    assert!(out.contains("v0: u64 = 0"), "{out}");
    assert!(out.contains("a counter that never goes negative"), "intent text is kept: {out}");
    assert!(!out.contains("Counter") && !out.contains("Underflow"), "{out}");
}

#[test]
fn projection_is_accepted_and_has_the_same_trace_hashes() {
    let mut checked = 0;
    for p in examples() {
        let file = p.to_string_lossy().into_owned();
        let (c0, _) = rce(&["check", &file]);
        if c0 != 0 {
            continue; // the corpus has files that are only accepted as a package
        }
        let (c1, text) = rce(&["fmt", "nameless", &file]);
        assert_eq!(c1, 0, "{file}: {text}");
        let t = tmp(p.file_name().unwrap().to_str().unwrap(), &text);
        let (c2, why) = rce(&["check", &t]);
        assert_eq!(c2, 0, "{file} nameless requires change: {why}\n{text}");
        let (_, a) = rce(&["trace", &file]);
        let (_, b) = rce(&["trace", &t]);
        // Same shape of every trace: step count and outcome per step, and the
        // same state hashes (T7: 6.1 hashes the positional form of the state).
        let hashes = |s: &str| -> Vec<String> {
            s.match_indices("\"state_hash\":\"").map(|(i, _)| s[i + 14..i + 14 + 64].to_string()).collect()
        };
        assert_eq!(hashes(&a), hashes(&b), "{file}: state hashes differ under renaming");
        let outcomes = |s: &str| -> Vec<String> {
            s.match_indices("\"outcome\":").map(|(i, _)| s[i..s[i..].find('}').map_or(s.len(), |j| i + j)].to_string()).collect()
        };
        let oa = outcomes(&a);
        let ob = outcomes(&b);
        assert_eq!(oa.len(), ob.len(), "{file}: step count");
        for (x, y) in oa.iter().zip(&ob) {
            // the outcome of a kept step names the error; compare up to the renamed error
            assert_eq!(x.starts_with("\"outcome\":\"kept"), y.starts_with("\"outcome\":\"kept"), "{file}: {x} vs {y}");
        }
        checked += 1;
    }
    assert!(checked >= 40, "only {checked} examples checked");
}

#[test]
fn projection_is_idempotent() {
    let (_, once) = rce(&["fmt", "nameless", "../examples/04_bank_account.rce"]);
    let t = tmp("bank_once.rce", &once);
    let (_, twice) = rce(&["fmt", "nameless", &t]);
    assert_eq!(once, twice);
}

/// T7 as modelled (`RCE/NameFree.lean` hashes De Bruijn keys): since the
/// LEDGER decision of 2026-10-23 (resolution A), `state_hash` and the entry
/// hash are over the positional forms of 6.1, so a renamed program has the
/// same log hashes. Before that decision this test was ignored and failing
/// (finding of 2026-10-22: the hashes were over name-keyed JSON).
#[test]
fn state_hashes_are_name_free() {
    let (_, a) = rce(&["trace", "../examples/01_counter.rce"]);
    let (_, text) = rce(&["fmt", "nameless", "../examples/01_counter.rce"]);
    let t = tmp("counter_nameless.rce", &text);
    let (_, b) = rce(&["trace", &t]);
    let hashes = |s: &str| -> Vec<String> {
        s.match_indices("\"state_hash\":\"").map(|(i, _)| s[i + 14..i + 14 + 64].to_string()).collect()
    };
    assert_eq!(hashes(&a), hashes(&b), "T7 as modelled: log hashes do not depend on names");
    let entry_hashes = |s: &str| -> Vec<String> {
        s.match_indices("\"hash\":\"").map(|(i, _)| s[i + 8..i + 8 + 64].to_string()).collect()
    };
    assert_eq!(entry_hashes(&a), entry_hashes(&b), "the entry hashes cover the positional event");
    assert!(!entry_hashes(&a).is_empty());
}

/// HC-21 under T7: the host's entries for kept steps name no declaration (the
/// cause, not the error's name, is in D), so a renamed program run on the
/// host gives a D with the same hashes, kept steps included.
#[test]
fn kept_entries_are_name_free() {
    use rce_spec_check::ast::Item;
    use rce_spec_check::host::{Host, MemStore};
    let src = "state Counter\n    n: u64 = 0\n\nevent Inc\nevent Dec\n\nerror Underflow\n\nstep Counter on Inc -> n + 1\nstep Counter on Dec -> try n.checked_sub(1) else Underflow\n";
    let f = tmp("kept_names.rce", src);
    let (_, projected) = rce(&["fmt", "nameless", &f]);
    assert!(!projected.contains("Underflow"), "{projected}");
    let run = |text: &str| -> (Vec<String>, String) {
        let m = rce_spec_check::parse::parse(text).unwrap();
        let names: Vec<String> = m.items.iter().filter_map(|i| if let Item::Event { name, .. } = i { Some(name.clone()) } else { None }).collect();
        let (inc, dec) = (names[0].clone(), names[1].clone());
        let mut st = MemStore::default();
        let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
        let mut causes = vec![];
        for ev in [&dec, &inc, &dec, &dec] {
            causes.extend(h.admit_event(ev, vec![]).unwrap().iter().filter_map(|a| a.cause));
        }
        assert_eq!(causes, vec!["fail", "fail"]);
        drop(h);
        let hashes = st.lines.iter().map(|l| rce_spec_check::host::Entry::parse_line(l).unwrap().hash).collect();
        (hashes, st.lines.join("\n"))
    };
    let (a, da) = run(src);
    let (b, db) = run(&projected);
    assert!(da.contains(r#"{"payload":{"cause":"fail","seq":2},"type":"host.StepKept"}"#), "{da}");
    assert!(db.contains(r#"{"payload":{"cause":"fail","seq":2},"type":"host.StepKept"}"#), "{db}");
    assert_eq!(a, b, "D hashes do not depend on names, entries of kept steps included");
}

/// 6.6: the concept record is printed by `rce emit form` under `names`,
/// keyed by declaration index, and is the only place the source names
/// survive in a nameless projection's Form.
#[test]
fn emit_form_carries_the_concept_record() {
    let (code, out) = rce(&["emit", "form", "../examples/01_counter.rce"]);
    assert_eq!(code, 0, "{out}");
    let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    let items = v["names"]["items"].as_array().unwrap();
    assert_eq!(items[0]["key"], "#1");
    assert_eq!(items[0]["name"], "Counter");
    assert_eq!(v["names"]["intent"][0], "a counter that never goes negative");
    let (_, text) = rce(&["fmt", "nameless", "../examples/01_counter.rce"]);
    let t = tmp("counter_cr.rce", &text);
    let (_, out2) = rce(&["emit", "form", &t]);
    let w: serde_json::Value = serde_json::from_str(out2.trim()).unwrap();
    assert_eq!(w["names"]["items"][0]["key"], "#1");
    assert_eq!(w["names"]["items"][0]["name"], "r0");
    assert_eq!(w["names"]["intent"], v["names"]["intent"], "intent text survives the projection");
}

#[test]
fn a_field_named_like_a_builtin_method_does_not_rename_the_method() {
    // T7 sweep finding (apps/chat.rce, 24 October 2026): a state field `keys`
    // renamed the Map method `.keys()` and the projection required change
    let src = "intent \"keys\"\n    example Put(k: \"a\") -> n == 1\n\nstate S\n    keys: Map[text, u64] = {}\n    n: usize = 0\n\nevent Put(k: text)\n\nstep S on Put(k)\n    keys[k] = 1\n    n = keys.keys().len()\n";
    let f = tmp("keys_src", src);
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 0, "{out}");
    let (_, projected) = rce(&["fmt", "nameless", &f]);
    assert!(projected.contains(".keys()"), "{projected}");
    let g = tmp("keys_nameless", &projected);
    let (st, out) = rce(&["check", &g]);
    assert_eq!(st, 0, "{out}");
    let (st, out) = rce(&["equiv", "fuel", "200", &f, &g]);
    assert_eq!(st, 0, "{out}");
}

#[test]
fn float_exponent_is_not_a_name() {
    // A program that declares a name `e`: the exponent of `1.5e-3` and of
    // `2E+4` is part of the literal, not a use of the name.
    let src = "fn e(x: f64) -> f64\n    return x * 1.5e-3 + 2E+4\n\nfn h() -> i64\n    return 0x1e\n\nfn main()\n    emit e(2.0)\n    emit h()\n";
    let p = tmp("float_exp.rce", src);
    let (code, out) = rce(&["fmt", "nameless", &p]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("1.5e-3") && out.contains("2E+4") && out.contains("0x1e"), "{out}");
    let q = tmp("float_exp_nameless.rce", &out);
    let (c2, o2) = rce(&["check", &q]);
    assert_eq!(c2, 0, "{o2}");
}

#[test]
fn use_paths_keep_their_names() {
    // stdlib/lib/num/stats.rce uses `num.linalg` and declares names that the
    // projection renames; the module path must not change.
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../stdlib/lib/num/stats.rce");
    let (code, out) = rce(&["fmt", "nameless", p.to_str().unwrap()]);
    assert_eq!(code, 0, "{out}");
    for l in out.lines().filter(|l| l.trim_start().starts_with("use ")) {
        assert!(!l.contains(" v") || l.contains("use num") || l.contains("use core"), "{l}");
    }
    assert!(out.contains("use num.linalg"), "{out}");
}

/// Project `src`, check the projection and run both: the projection is
/// admitted and prints what the source prints.
fn projects_and_runs_alike(name: &str, src: &str) -> String {
    let f = tmp(&format!("{name}.rce"), src);
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 0, "{out}");
    let (st, projected) = rce(&["fmt", "nameless", &f]);
    assert_eq!(st, 0, "{projected}");
    let g = tmp(&format!("{name}_nameless.rce"), &projected);
    let (st, out) = rce(&["check", &g]);
    assert_eq!(st, 0, "{projected}\n{out}");
    assert_eq!(rce(&["run", &f]), rce(&["run", &g]), "{projected}");
    projected
}

#[test]
fn a_builtin_method_keeps_its_name_beside_a_function_of_that_name() {
    // stdlib/lib/math/math.rce declares `pub fn sqrt` and calls `x.sqrt()`,
    // the f64 method: `x.m(…)` reaches a function of the module only as a
    // method of an `impl` block (5.15), so the call keeps its name
    let p = projects_and_runs_alike(
        "sqrt",
        "fn sqrt(x: f64) -> f64\n    return x.sqrt() + 1.0\n\nfn main()\n    emit sqrt(4.0)\n",
    );
    assert!(p.contains("v0.sqrt()") && p.contains("emit f0(4.0)"), "{p}");
}

#[test]
fn fields_of_an_imported_type_keep_their_names() {
    // stdlib/lib/num/stats.rce reads `a.m` and `a.n` of a `linalg.Mat` and
    // also declares locals `m` and `n`: the locals are renamed, the fields of
    // the imported type are not, in accesses and in literal labels alike, and
    // a range `0..n` is not a member access
    let p = projects_and_runs_alike(
        "imported_fields",
        "use num.linalg\n\nstruct Shape\n    n: usize\n\nfn size(a: &linalg.Mat, s: &Shape) -> usize\n    let m = a.m\n    var t = 0\n    for _i in 0..m\n        t = t + 1\n    return t * a.n + s.n\n\nfn main()\n    let a = linalg.Mat(m: 2, n: 3, a: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])\n    let s = Shape(n: 1)\n    emit size(&a, &s)\n",
    );
    assert!(p.contains(".m\n") && p.contains(".n + ") && p.contains("Mat(m: 2, n: 3, a: ["), "{p}");
    assert!(!p.contains("let m ") && !p.contains("fn size"), "{p}");
}

#[test]
fn template_holes_are_projected_like_code() {
    // spec/examples-0.9/ok/15_templates.rce: the expressions in a template's
    // `{…}` name the module's own locals and fields, so they are renamed with
    // them; the template's text and its `{{` `}}` stay as written
    let p = projects_and_runs_alike(
        "templates",
        "# rce-version: 0.9\nstruct User\n    name: text\n\nfn greet(u: &User, n: i64) -> text\n    return text.t\"Hi {u.name}, {n} new {{ok}}\"\n\nfn main()\n    let u = User(name: \"ann\")\n    emit greet(&u, 3)\n",
    );
    assert!(p.contains("t\"Hi {v1.v0}, {v2} new {{ok}}\""), "{p}");
}

#[test]
fn the_authority_structs_keep_their_names() {
    // spec/examples-0.9/ok/07_dir_caps.rce: the checker reads DirRead,
    // DirWrite and their field `root` by name (F.5.1, W-CAP-3), so they keep
    // them; a local named `root` is still renamed
    let f = tmp("caps.rce", "# rce-version: 0.9\nstruct DirRead\n    root: text\n\nextern \"C\" fn file_len(w: &World, dir: Cap[DirRead], path: &text) -> i64\n\nfn reader(w: &World, root: Cap[DirRead], out: chan i64)\n    let sub = root.narrow(\"logs\")\n    let name = \"app.log\"\n    out <- file_len(w, sub, &name)\n\nfn main()\n    emit 0\n");
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 0, "{out}");
    let (st, p) = rce(&["fmt", "nameless", &f]);
    assert_eq!(st, 0, "{p}");
    assert!(p.contains("struct DirRead\n    root: text") && p.contains("Cap[DirRead]") && !p.contains("root: Cap"), "{p}");
    let g = tmp("caps_nameless.rce", &p);
    let (st, out) = rce(&["check", &g]);
    assert_eq!(st, 0, "{p}\n{out}");
}

/// T7 on one program (scripts/t7-equiv.sh): the projection is admitted, is
/// its own projection, and `rce equiv` finds it equivalent to the source.
fn t7_equivalent(name: &str, src: &str) -> String {
    let f = tmp(&format!("{name}.rce"), src);
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 0, "{out}");
    let (st, projected) = rce(&["fmt", "nameless", &f]);
    assert_eq!(st, 0, "{projected}");
    let g = tmp(&format!("{name}_nameless.rce"), &projected);
    let (st, out) = rce(&["check", &g]);
    assert_eq!(st, 0, "{projected}\n{out}");
    let (_, again) = rce(&["fmt", "nameless", &g]);
    assert_eq!(again, projected, "the projection of the projection differs");
    let (st, out) = rce(&["equiv", "fuel", "200", &f, &g]);
    assert_eq!(st, 0, "{projected}\n{out}");
    projected
}

#[test]
fn a_state_field_and_a_struct_field_of_one_name_are_renamed_apart() {
    // T7 sweep finding (programs/systems/ls.rce, 1 November 2026): the state
    // field `at` beside a struct field `at`, in a program that uses `time`,
    // whose error BadTime has a field `at`. The struct field keeps the name,
    // as a field of an imported type would; the state field, used by its bare
    // name in the step, was also left as written while its uses were renamed.
    let src = "intent \"a state field and a struct field share a name\"\n    example Mark(n: 2) -> count == 1 and at[0] == 2\n\nuse time\n\nstruct LocalOffset\n    req: u64\n    at: i64\n\nevent Mark(n: i64)\n\nstate Book\n    at: Map[u64, i64] = {}\n    count: u64 = 0\n\nstep Book on Mark(n: k)\n    at[count] = k\n    count += 1\n    emit LocalOffset(req: count, at: time.fmod(k, 7))\n";
    let p = t7_equivalent("state_struct_at", src);
    assert!(p.contains("struct t0\n    v1: u64\n    at: i64"), "{p}");
    assert!(p.contains("state r0\n    v2: Map") && p.contains("    v2[v3] = v4") && p.contains("at: time.fmod"), "{p}");
    // without `time` no field is imported: both are renamed, to one name
    let p = t7_equivalent("state_struct_at_local", &src.replace("use time\n", "").replace("time.fmod(k, 7)", "k"));
    assert!(p.contains("struct t0\n    v1: u64\n    v2: i64") && p.contains("state r0\n    v2: Map") && p.contains("t0(v1: v3, v2: v4)"), "{p}");
}

#[test]
fn an_event_field_and_a_state_field_of_one_name_are_renamed_apart() {
    // the event's field keeps the imported field name in its declaration,
    // its pattern and its example; the state field is renamed with its uses
    let p = t7_equivalent(
        "event_state_at",
        "intent \"an event field and a state field share a name\"\n    example Mark(at: 2) -> at == 2\n\nuse time\n\nevent Mark(at: i64)\n\nstate Book\n    at: i64 = 0\n\nstep Book on Mark(at: a)\n    at = time.fmod(a, 7)\n",
    );
    assert!(p.contains("example e0(at: 2) -> v0 == 2") && p.contains("event e0(at: i64)") && p.contains("    v0: i64 = 0"), "{p}");
    assert!(p.contains("on e0(at: v1)\n    v0 = time.fmod(v1, 7)"), "{p}");
}

#[test]
fn parameters_and_locals_named_like_a_field_are_renamed_on_every_line() {
    // a parameter and a local `at` beside a struct field `at` named like an
    // imported field, with labels on lines of their own inside parentheses:
    // the parameter is renamed where it is declared, used and passed, and
    // the struct field keeps its name in its literal and its access
    let p = projects_and_runs_alike(
        "param_at",
        "use time\n\nstruct P\n    at: i64\n\nfn shift(\n    at: i64,\n    by: i64,\n) -> i64\n    let p = P(\n        at: at + by,\n    )\n    return time.fmod(p.at, 100)\n\nfn main()\n    let at = 3\n    emit shift(\n        at: at,\n        by: 4,\n    )\n",
    );
    assert!(p.contains("fn f0(\n    v0: i64,\n    v1: i64,\n)") && p.contains("        at: v0 + v1,") && p.contains("v2.at"), "{p}");
    assert!(p.contains("    let v0 = 3\n    emit f0(\n        v0: v0,\n        v1: 4,"), "{p}");
    let g = tmp("param_at_nameless.rce", &p);
    assert_eq!(rce(&["fmt", "nameless", &g]).1, p, "the projection of the projection differs");
}

#[test]
fn a_map_key_inside_a_call_is_a_name_not_a_label() {
    // R7: `{a: 1}` is a map whose key is the variable `a`, also after `,`
    // inside a call's parentheses, so the key is renamed with the local
    let p = projects_and_runs_alike(
        "map_key",
        "fn main()\n    let a = 1\n    let b = 2\n    var xs: List[Map[i64, i64]] = []\n    xs.push({a: 10, b: 20})\n    emit xs[0][b]\n",
    );
    assert!(p.contains("v2.push({v0: 10, v1: 20})"), "{p}");
}
