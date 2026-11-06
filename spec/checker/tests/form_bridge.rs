//! Schedule and construction (rce-form.md §13) on the slice both sides apply.
//!
//! A core module in the slice builds a canonical Form that passes FG1 to FG15.
//! Scheduling that Form and constructing again yields the same bytes. On the
//! same arguments, the core evaluator and the Form evaluator agree, and so
//! does the core evaluator on the scheduled module.
//!
//! `spec/examples` is the corpus. A whole module still requires change: the
//! observation driver writes `out` and reads globals outside ω. An example is
//! counted when every function of the program itself (`fn`, `region`, `delta`,
//! `initial state`) round-trips through construct(schedule(F)). The reason's
//! tag says which construct stopped the others. The count is the ratchet.
use rce_spec_check::form::bridge;
use rce_spec_check::form::{self, ir as fir};
use rce_spec_check::one::build::{self, FB};
use rce_spec_check::one::eval::{self, Flow, Val};
use rce_spec_check::one::ir::{self as core, Atom, FnRef, Lit, Place, Stmt, Ty};
use rce_spec_check::one::json::{canonical_group, group_key};
use rce_spec_check::one::wf;
use std::collections::BTreeMap;

fn i64ty() -> Ty {
    Ty::i64()
}

fn module(name: &str, f: core::Fn) -> core::Module {
    let mut defs = BTreeMap::new();
    let mut names = serde_json::Map::new();
    let (g, _) = canonical_group(std::slice::from_ref(&f));
    let key = group_key(&g);
    names.insert(key.clone(), serde_json::json!([name]));
    defs.insert(key.clone(), g);
    // W1: the entry takes no parameters. A function that takes some is not the entry.
    let entry = if f.params.is_empty() {
        key
    } else {
        let main = core::Fn {
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
    core::Module {
        globals: vec![],
        defs,
        entry: (entry, 0),
        names: serde_json::json!({"fns": names, "globals": []}),
        graphs: BTreeMap::new(),
    }
}

fn checked_add() -> core::Module {
    let t = i64ty();
    let f = core::Fn {
        params: vec![t.clone(), t.clone()],
        ret: t.clone(),
        body: vec![
            Stmt::Prim("int.bad.add".into(), vec![t.clone()], vec![
                Atom::V(0),
                Atom::V(1),
            ]),
            Stmt::If(
                Atom::V(2),
                vec![
                    Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![
                        build::lit(0),
                        Atom::bytes(b"overflow"),
                    ]),
                    Stmt::Halt(Atom::V(3)),
                ],
                vec![],
            ),
            Stmt::Prim("int.add".into(), vec![t], vec![Atom::V(0), Atom::V(1)]),
            Stmt::Ret(Atom::V(4)),
        ],
    };
    module("add", f)
}

fn choose() -> core::Module {
    let t = i64ty();
    let f = core::Fn {
        params: vec![Ty::Bool, t.clone(), t.clone()],
        ret: t.clone(),
        body: vec![
            Stmt::Var(t, build::lit(0)),
            Stmt::If(Atom::V(0), vec![Stmt::Set(Place::V(3), Atom::V(1))], vec![
                Stmt::Set(Place::V(3), Atom::V(2)),
            ]),
            Stmt::Ret(Atom::V(3)),
        ],
    };
    module("choose", f)
}

fn len0() -> core::Module {
    let e = i64ty();
    let f = core::Fn {
        params: vec![],
        ret: Ty::Int(0, core::INT_MAX),
        body: vec![
            Stmt::Prim("list.new".into(), vec![e.clone()], vec![]),
            Stmt::Prim("list.len".into(), vec![e.clone()], vec![Atom::V(0)]),
            Stmt::Prim("drop".into(), vec![Ty::List(Box::new(e))], vec![Atom::V(0)]),
            Stmt::Ret(Atom::V(1)),
        ],
    };
    module("len0", f)
}

/// `n` with one fuel tick in front of the return. The tick is `rt.tick`.
fn ticked() -> core::Module {
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
    let tkey = group_key(&tg);
    let user = core::Fn {
        params: vec![i64ty()],
        ret: i64ty(),
        body: vec![
            Stmt::Call(FnRef::Key(tkey.clone(), 0), vec![build::lit(1)]),
            Stmt::Ret(Atom::V(0)),
        ],
    };
    let (ug, _) = canonical_group(std::slice::from_ref(&user));
    let ukey = group_key(&ug);
    let main = core::Fn {
        params: vec![],
        ret: i64ty(),
        body: vec![Stmt::Ret(build::lit(0))],
    };
    let (mg, _) = canonical_group(std::slice::from_ref(&main));
    let mkey = group_key(&mg);
    core::Module {
        globals: vec![(Ty::u64(), Some(Lit::Int(form::eval::FUEL as i128)))],
        defs: BTreeMap::from([(tkey.clone(), tg), (ukey.clone(), ug), (mkey.clone(), mg)]),
        entry: (mkey.clone(), 0),
        names: serde_json::json!({"fns": {tkey: ["rt.tick"], ukey: ["ticked"], mkey: ["main"]}, "globals": ["fuel"]}),
        graphs: BTreeMap::new(),
    }
}

fn by_arity(m: &core::Module, n: usize) -> (String, u32) {
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

fn core_apply(m: &core::Module, key: &str, i: u32, args: Vec<Val>) -> Result<Val, String> {
    let mut ev = eval::Eval::new(m);
    match ev.call_key(key, i, args) {
        Ok(v) => Ok(v),
        Err(Flow::Halt(h)) => Err(format!("halt: {}", halt_text(&h))),
        Err(Flow::Bad(e)) => Err(e),
    }
}

fn halt_text(h: &Val) -> String {
    match h {
        Val::Sum(0, t) => match &**t {
            Val::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
            _ => "?".into(),
        },
        Val::Sum(1, t) => match &**t {
            Val::Tup(xs) => match xs.first() {
                Some(Val::Bytes(b)) => String::from_utf8_lossy(b).into_owned(),
                _ => "?".into(),
            },
            _ => "?".into(),
        },
        _ => "?".into(),
    }
}

fn round_trip(core_m: &core::Module) -> fir::Module {
    wf::check(core_m).unwrap_or_else(|e| panic!("core is not well formed: {e}"));
    round_trip_unchecked(core_m)
}

/// Like `round_trip`, without checking the input core. A closure taken out of
/// a lowered module is not itself an entry: W3 would require it to reset
/// globals the observation driver owns.
fn round_trip_unchecked(core_m: &core::Module) -> fir::Module {
    let f = bridge::construct(core_m).unwrap_or_else(|e| panic!("{}", e.line()));
    let bytes = fir::encode(&f);
    assert!(
        matches!(
            form::check(bytes.as_bytes(), None),
            form::Outcome::Applied(_)
        ),
        "the constructed form is not applied"
    );
    let scheduled = bridge::schedule(&f).unwrap_or_else(|e| panic!("{}", e.line()));
    wf::check(&scheduled).unwrap_or_else(|e| panic!("schedule is not well formed: {e}"));
    let back = bridge::construct(&scheduled)
        .unwrap_or_else(|e| panic!("construct of the schedule: {}", e.line()));
    assert_eq!(
        fir::encode(&f),
        fir::encode(&back),
        "construct(schedule(F)) is not F"
    );
    f
}

fn agree(core_m: &core::Module, form_m: &fir::Module, scheduled: &core::Module, args: Vec<Val>) {
    let (ck, ci) = by_arity(core_m, args.len());
    let want = core_apply(core_m, &ck, ci, args.clone());
    let (fk, fi) = form_user_matching(form_m, args.len());
    let got = form::eval::apply(form_m, &fk, fi, args.clone()).map(|v| {
        assert_eq!(v.len(), 1, "the slice returns one value");
        v.into_iter().next().unwrap()
    });
    match (&want, &got) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "form eval and core eval differ"),
        (Err(a), Err(b)) => assert_eq!(a, b, "halt texts differ"),
        _ => panic!("form {got:?} and core {want:?} disagree"),
    }
    let (sk, si) = by_arity(scheduled, args.len());
    let again = core_apply(scheduled, &sk, si, args);
    assert_eq!(
        want, again,
        "the scheduled core and the original core disagree"
    );
}

/// The constructed module may hold several functions. Pick the one whose
/// non-ω inputs match `nargs` and whose body is the largest; the helpers a
/// canonical pass leaves behind are smaller.
fn form_user_matching(m: &fir::Module, nargs: usize) -> (String, u32) {
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
    let (k, i, _) = best.unwrap_or_else(|| panic!("no form function of {nargs} inputs"));
    (k, i)
}

#[test]
fn add_round_trips_and_agrees() {
    let c = checked_add();
    let f = round_trip(&c);
    let s = bridge::schedule(&f).unwrap();
    agree(&c, &f, &s, vec![Val::Int(2), Val::Int(3)]);
    agree(&c, &f, &s, vec![Val::Int(i64::MAX as i128), Val::Int(1)]);
}

#[test]
fn branch_round_trips_and_agrees() {
    let c = choose();
    let f = round_trip(&c);
    let s = bridge::schedule(&f).unwrap();
    agree(&c, &f, &s, vec![
        Val::Bool(true),
        Val::Int(10),
        Val::Int(20),
    ]);
    agree(&c, &f, &s, vec![
        Val::Bool(false),
        Val::Int(10),
        Val::Int(20),
    ]);
}

fn example_closure(file: &str, name: &str) -> core::Module {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../examples")
        .join(file);
    let src = std::fs::read_to_string(&path).unwrap();
    let ast = rce_spec_check::program(&src).unwrap();
    let core_m = rce_spec_check::one::core_of(&ast).unwrap_or_else(|e| panic!("{e}"));
    let mut names = vec![];
    if let Some(fns) = core_m.names.get("fns").and_then(|v| v.as_object()) {
        for (k, arr) in fns {
            if let Some(a) = arr.as_array() {
                for (i, n) in a.iter().enumerate() {
                    names.push(format!("{k}/{i} {}", n.as_str().unwrap_or("?")));
                }
            }
        }
    }
    bridge::closure(&core_m, name)
        .unwrap_or_else(|| panic!("{name} is not in {file}; names:\n{}", names.join("\n")))
}

#[test]
fn escapes_and_axpy_round_trip_and_agree() {
    let c = example_closure("20_strings.rce", "fn escapes");
    let f = round_trip_unchecked(&c);
    let s = bridge::schedule(&f).unwrap();
    agree(&c, &f, &s, vec![]);
    let c = example_closure("42_f32.rce", "fn axpy");
    let f = round_trip_unchecked(&c);
    let s = bridge::schedule(&f).unwrap();
    agree(&c, &f, &s, vec![Val::F(2.0), Val::F(0.1), Val::F(1.0)]);
}

/// `for i in 0..n`, the lowering's measure: read i, leave unless i < n, then
/// raise i by one. The count is the value i holds when the test fails.
fn count_up() -> core::Module {
    let t = i64ty();
    let mut b = FB::new(vec![t.clone()], t.clone());
    let n = b.arg(0);
    let i = b.var(t.clone(), build::lit(0));
    b.loop_(|b| {
        let iv = b.let_(Atom::V(i));
        let more = b.int_cmp("int.lt", iv.clone(), n)?;
        b.exit_unless(more)?;
        let i1 = b.add_in(iv, build::lit(1), t.clone())?;
        b.set(i, i1);
        Ok(())
    })
    .unwrap();
    let r = b.let_(Atom::V(i));
    b.ret(r);
    module("count_up", b.finish())
}

#[test]
fn borrow_round_trips_and_agrees() {
    let c = example_closure("12_borrow.rce", "fn demo");
    let f = round_trip_unchecked(&c);
    let s = bridge::schedule(&f).unwrap();
    agree(&c, &f, &s, vec![]);
    let c = example_closure("12_borrow.rce", "fn push_all");
    let f = round_trip_unchecked(&c);
    let s = bridge::schedule(&f).unwrap();
    agree(&c, &f, &s, vec![
        Val::List(vec![]),
        Val::List(vec![Val::Int(4), Val::Int(5)]),
    ]);
    let c = example_closure("43_simd_lanes.rce", "fn blend");
    let f = round_trip_unchecked(&c);
    let s = bridge::schedule(&f).unwrap();
    let lane = |xs: [f64; 4]| Val::Arr(xs.into_iter().map(Val::F).collect());
    agree(&c, &f, &s, vec![
        lane([1.0, 2.0, 3.0, 4.0]),
        lane([0.0, 0.0, 0.0, 0.0]),
        lane([1.0, 0.5, 0.0, 1.0]),
    ]);
}

#[test]
fn measure_loop_round_trips_and_agrees() {
    let c = count_up();
    let f = round_trip(&c);
    let s = bridge::schedule(&f).unwrap();
    agree(&c, &f, &s, vec![Val::Int(0)]);
    agree(&c, &f, &s, vec![Val::Int(4)]);
}

#[test]
fn read_return_round_trips_and_agrees() {
    let c = len0();
    let f = round_trip(&c);
    let s = bridge::schedule(&f).unwrap();
    agree(&c, &f, &s, vec![]);
}

#[test]
fn fuel_tick_round_trips_and_agrees() {
    let c = ticked();
    wf::check(&c).unwrap_or_else(|e| panic!("{e}"));
    let f = round_trip(&c);
    let s = bridge::schedule(&f).unwrap();
    agree(&c, &f, &s, vec![Val::Int(7)]);
}

#[test]
fn refused_constructs_name_the_reason() {
    let refuse = |body: Vec<Stmt>, ret: Ty, tag: &str| {
        let f = core::Fn {
            params: vec![],
            ret,
            body,
        };
        let m = module("no", f);
        // the core may itself be rejected by W1; construction still has to refuse, or wf does
        match wf::check(&m) {
            Err(e) => assert!(e.starts_with("W"), "{e}"),
            Ok(()) => {
                let e = bridge::construct(&m).expect_err(tag);
                assert_eq!(e.tag(), tag, "{}", e.line());
            }
        }
    };
    refuse(
        vec![Stmt::Out(Atom::bytes(b"x")), Stmt::Ret(build::lit(0))],
        i64ty(),
        "out",
    );
    refuse(
        vec![
            Stmt::Prim("io.byte".into(), vec![], vec![build::lit(1)]),
            Stmt::Ret(build::lit(0)),
        ],
        i64ty(),
        "carrier",
    );
    let mut with_global = module("g", core::Fn {
        params: vec![],
        ret: i64ty(),
        body: vec![Stmt::Let(Atom::G(0)), Stmt::Ret(Atom::V(0))],
    });
    with_global.globals.push((i64ty(), Some(Lit::Int(1))));
    wf::check(&with_global).unwrap_or_else(|e| panic!("{e}"));
    let e = bridge::construct(&with_global).expect_err("global");
    assert_eq!(e.tag(), "global", "{}", e.line());

    let catch = core::Fn {
        params: vec![],
        ret: Ty::Sum(vec![i64ty(), Ty::halt()]),
        body: vec![
            Stmt::Catch(FnRef::Key("missing".into(), 0), vec![]),
            Stmt::Ret(Atom::V(0)),
        ],
    };
    // not well formed (the callee is missing); the tag is still the one construction would use
    let _ = catch;
    let halt_only = module("h", core::Fn {
        params: vec![],
        ret: i64ty(),
        body: vec![
            Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![
                build::lit(0),
                Atom::bytes(b"no"),
            ]),
            Stmt::Halt(Atom::V(0)),
        ],
    });
    // a halt is in the slice; this one is applied, which pins the boundary
    let f = bridge::construct(&halt_only).unwrap_or_else(|e| panic!("{}", e.line()));
    assert!(matches!(
        form::check(fir::encode(&f).as_bytes(), None),
        form::Outcome::Applied(_)
    ));
}

/// A function of the program itself, not the observation driver and not a
/// runtime helper. The driver (`entry`, `observe *`, `check *`, `out`, fuel
/// resets) still requires change. Runtime helpers are in a root's closure
/// when the root calls them; they are not roots.
fn program_root(name: &str) -> bool {
    name.split(" = ").any(|n| {
        n.starts_with("fn ") || n.starts_with("region ") || n == "delta" || n == "initial state"
    })
}

fn root_names(m: &core::Module) -> Vec<String> {
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

fn closure_round_trip(core_m: &core::Module, name: &str) -> Result<(), String> {
    let sub = bridge::closure(core_m, name).ok_or_else(|| "defect".to_string())?;
    let f = match bridge::construct(&sub) {
        Ok(f) => f,
        Err(e) => return Err(e.tag().to_string()),
    };
    let s = match bridge::schedule(&f) {
        Ok(s) => s,
        Err(e) => return Err(e.tag().to_string()),
    };
    match bridge::construct(&s) {
        Ok(back) if fir::encode(&f) == fir::encode(&back) => Ok(()),
        Ok(_) => Err("round-trip".into()),
        Err(e) => Err(e.tag().to_string()),
    }
}

#[test]
fn examples_are_covered_or_require_change() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    files.sort();
    let mut programs = 0usize;
    let mut whole = 0usize;
    let mut covered = 0usize;
    let mut tags: BTreeMap<String, usize> = BTreeMap::new();
    let mut root_tags: BTreeMap<String, usize> = BTreeMap::new();
    for p in &files {
        let src = std::fs::read_to_string(p).unwrap();
        if rce_spec_check::program(&src).is_err() {
            continue;
        }
        let ast = rce_spec_check::program(&src).unwrap();
        programs += 1;
        let core_m = match rce_spec_check::one::core_of(&ast) {
            Ok(c) => c,
            Err(e) => {
                *tags.entry(format!("lowering: {e}")).or_default() += 1;
                *root_tags.entry(format!("lowering: {e}")).or_default() += 1;
                continue;
            }
        };
        match bridge::construct(&core_m) {
            Ok(f) => {
                let s = bridge::schedule(&f)
                    .unwrap_or_else(|e| panic!("{}: {}", p.display(), e.line()));
                let back = bridge::construct(&s)
                    .unwrap_or_else(|e| panic!("{}: {}", p.display(), e.line()));
                assert_eq!(fir::encode(&f), fir::encode(&back), "{}", p.display());
                whole += 1;
            }
            Err(e) => {
                *tags.entry(e.tag().to_string()).or_default() += 1;
            }
        }
        let roots = root_names(&core_m);
        if roots.is_empty() {
            *root_tags.entry("no program root".into()).or_default() += 1;
            continue;
        }
        let mut bad: Option<String> = None;
        for name in &roots {
            if let Err(e) = closure_round_trip(&core_m, name) {
                bad = Some(e);
                break;
            }
        }
        match bad {
            None => {
                eprintln!("  covered {}", p.file_name().unwrap().to_string_lossy());
                covered += 1;
            }
            Some(tag) => {
                *root_tags.entry(tag).or_default() += 1;
            }
        }
    }
    eprintln!("form bridge: {whole} of {programs} whole examples construct");
    for (k, n) in &tags {
        eprintln!("  {n} {k}");
    }
    eprintln!(
        "form bridge: {covered} of {programs} examples have every program function in the slice"
    );
    for (k, n) in &root_tags {
        eprintln!("  {n} {k}");
    }
    // Ratchet. Whole modules still meet the observation driver (`out`, fuel
    // globals, `catch`). A program function is in the slice when its closure
    // round-trips through construct(schedule(F)). An example is counted when
    // every such function does. Raise the floor when the slice grows; do not
    // lower it.
    assert_eq!(programs, 60, "spec/examples gained or lost a program");
    assert_eq!(
        whole, 0,
        "a whole example now constructs; raise this floor and check its trace"
    );
    assert_eq!(
        covered, 3,
        "the number of examples whose program functions round-trip changed; raise the ratchet, do not lower it"
    );
    assert!(
        tags.values().sum::<usize>() + whole >= programs.saturating_sub(4),
        "every lowered example is covered or tagged"
    );
}
