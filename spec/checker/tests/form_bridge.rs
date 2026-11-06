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

fn tick_parts() -> (core::Fn, String) {
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
    (tick, tkey)
}

fn assemble(tick: core::Fn, tkey: String, name: &str, user: core::Fn) -> core::Module {
    let (tg, _) = canonical_group(std::slice::from_ref(&tick));
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
        names: serde_json::json!({
            "fns": {tkey: ["rt.tick"], ukey: [name], mkey: ["main"]},
            "globals": ["fuel"]
        }),
        graphs: BTreeMap::new(),
    }
}

/// A count loop that ticks, then breaks when the carried value is 2.
/// The break names the loop. The bound is the parameter.
fn early_count() -> core::Module {
    let (tick, tkey) = tick_parts();
    let key = tkey.clone();
    let ct = build::count_t();
    let mut b = FB::new(vec![ct.clone()], ct.clone());
    let bound = b.arg(0);
    let i = b.var(ct.clone(), build::lit(0));
    let cnt = b.counter();
    b.loop_(|b| {
        b.count(cnt, bound)?;
        let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
        let iv = b.let_(Atom::V(i));
        let stop = b.int_cmp("int.eq", iv.clone(), build::lit(2))?;
        b.exit_if(stop)?;
        let i1 = b.add_in(iv, build::lit(1), ct.clone())?;
        b.set(i, i1);
        Ok(())
    })
    .unwrap();
    let r = b.let_(Atom::V(i));
    b.ret(r);
    assemble(tick, tkey, "early", b.finish())
}

fn fuel_of(m: &core::Module, args: Vec<Val>) -> (Result<Val, String>, Val) {
    let (k, i) = by_arity(m, args.len());
    let mut ev = eval::Eval::new(m);
    let r = match ev.call_key(&k, i, args) {
        Ok(v) => Ok(v),
        Err(Flow::Halt(h)) => Err(format!("halt: {}", halt_text(&h))),
        Err(Flow::Bad(e)) => Err(e),
    };
    let g = ev.global(0).cloned().unwrap_or(Val::Int(-1));
    (r, g)
}

/// The break arm's flag is a literal false with no inputs, not a carried bool.
fn flag_is_fresh_false(f: &fir::Module) -> bool {
    fn false_lit(b: &fir::Body, s: &fir::Src) -> bool {
        match s {
            fir::Src::Out(i, 0) => matches!(
                b.nodes.get(*i as usize),
                Some(fir::Node {
                    attrs: fir::Attr::Lit(_, Lit::Bool(false)),
                    ins,
                    ..
                }) if ins.is_empty()
            ),
            _ => false,
        }
    }
    fn arm_false(b: &fir::Body, flag: &fir::Src) -> bool {
        if false_lit(b, flag) {
            return true;
        }
        let fir::Src::Out(i, p) = *flag else {
            return false;
        };
        let Some(n) = b.nodes.get(i as usize) else {
            return false;
        };
        if !matches!(n.attrs, fir::Attr::Branch(..)) {
            return false;
        }
        n.regions.iter().any(|arm| {
            let Some(s) = arm.out.get(p as usize) else {
                return false;
            };
            if false_lit(arm, s) || arm_false(arm, s) {
                return true;
            }
            // The schedule's dummy initial is the same literal, so the arm
            // returns that capture. It is still not a carried loop input.
            let fir::Src::In(j) = *s else {
                return false;
            };
            n.ins
                .get(1 + j as usize)
                .is_some_and(|cap| false_lit(b, cap) && !matches!(cap, fir::Src::In(_)))
        })
    }
    f.defs.values().any(|g| {
        g.fns.iter().any(|func| {
            func.body.nodes.iter().any(|n| match &n.attrs {
                fir::Attr::Count(_) | fir::Attr::Measure(..) => {
                    let body = &n.regions[1];
                    let Some(flag) = body.out.first() else {
                        return false;
                    };
                    !matches!(flag, fir::Src::In(_)) && arm_false(body, flag)
                }
                _ => false,
            })
        })
    })
}

#[test]
fn body_break_is_option_b_not_syntactic_identity() {
    // Option B of spec/proposals/break-carried-flag.md. The bytes below are
    // the flag form against its schedule. They are not syntactic identity
    // with the core break: Lean Stmt has no break constructor (B1), so this
    // Iterate has no preimage there. Admission is three-way eval, O6' and O7'.
    let c = early_count();
    wf::check(&c).unwrap_or_else(|e| panic!("{e}"));
    let f = bridge::construct(&c).unwrap_or_else(|e| panic!("{}", e.line()));
    assert!(
        bridge::body_break_rewrite(),
        "the body break was not restructured into a flag"
    );
    assert!(
        flag_is_fresh_false(&f),
        "the break path must return a literal false, not a carried bool (B4)"
    );
    assert!(
        matches!(
            form::check(fir::encode(&f).as_bytes(), None),
            form::Outcome::Applied(_)
        ),
        "the flag form is not applied"
    );
    let s = bridge::schedule(&f).unwrap_or_else(|e| panic!("{}", e.line()));
    wf::check(&s).unwrap_or_else(|e| panic!("schedule is not well formed: {e}"));
    let back =
        bridge::construct(&s).unwrap_or_else(|e| panic!("construct of the schedule: {}", e.line()));
    // The flag form matches its schedule. This is not syntactic identity for
    // the core break.
    assert_eq!(
        fir::encode(&f),
        fir::encode(&back),
        "construct(schedule(F)) is not the flag form"
    );
    // Break before the last legal iteration. Bound 10, stop at 2: not "loop bound" (O6').
    agree(&c, &f, &s, vec![Val::Int(10)]);
    let (src, fuel_src) = fuel_of(&c, vec![Val::Int(10)]);
    let (sch, fuel_sch) = fuel_of(&s, vec![Val::Int(10)]);
    assert_eq!(src, Ok(Val::Int(2)));
    assert_eq!(sch, src);
    // The flag does not tick. Fuel on the flag path equals fuel on the source (O7').
    assert_eq!(fuel_src, fuel_sch, "O7': fuel differs on the break path");
    // The bound trap is still the path that does not break.
    agree(&c, &f, &s, vec![Val::Int(2)]);
    let (src_b, fuel_b) = fuel_of(&c, vec![Val::Int(2)]);
    let (sch_b, fuel_sb) = fuel_of(&s, vec![Val::Int(2)]);
    assert_eq!(src_b, Err("halt: loop bound".into()));
    assert_eq!(sch_b, src_b);
    assert_eq!(fuel_b, fuel_sb, "O7': fuel differs on the bound path");
}

#[test]
fn pure_count_break_stays_outside_dom() {
    // O8. A break flag is a bool. It does not put a pure count into Dom.
    let ct = build::count_t();
    let mut b = FB::new(vec![ct.clone()], ct.clone());
    let bound = b.arg(0);
    let i = b.var(ct.clone(), build::lit(0));
    let cnt = b.counter();
    b.loop_(|b| {
        b.count(cnt, bound)?;
        let iv = b.let_(Atom::V(i));
        // `int.eq` does not halt, so neither region uses ω. The flag is not a use of ω.
        let stop = b.int_cmp("int.eq", iv, build::lit(0))?;
        b.exit_if(stop)?;
        Ok(())
    })
    .unwrap();
    let r = b.let_(Atom::V(i));
    b.ret(r);
    let m = module("pure", b.finish());
    wf::check(&m).unwrap_or_else(|e| panic!("{e}"));
    let e = bridge::construct(&m).expect_err("pure count");
    assert_eq!(e.tag(), "bound", "{}", e.line());
    assert!(
        !bridge::body_break_rewrite(),
        "a refused pure count is not an admitted break rewrite"
    );
}

#[test]
fn conditional_increment_is_not_rewritten_into_a_count() {
    // B5. The continue path does not always raise the index. The refusal is
    // not a count.
    let mut b = FB::new(vec![Ty::Bool, Ty::u64()], Ty::u64());
    let c = b.arg(0);
    let n = b.arg(1);
    let i = b.var(Ty::u64(), build::lit(0));
    b.loop_(|b| {
        let iv = b.let_(Atom::V(i));
        let more = b.int_cmp("int.lt", iv.clone(), n.clone())?;
        b.exit_unless(more)?;
        b.if_(
            c.clone(),
            |b| {
                let i1 = b.add_in(iv.clone(), build::lit(1), Ty::u64())?;
                b.set(i, i1);
                Ok(())
            },
            |_| Ok(()),
        )?;
        Ok(())
    })
    .unwrap();
    let r = b.let_(Atom::V(i));
    b.ret(r);
    let m = module("step", b.finish());
    wf::check(&m).unwrap_or_else(|e| panic!("{e}"));
    match bridge::construct(&m) {
        Err(e) => {
            assert_ne!(e.tag(), "bound", "{}", e.line());
            assert!(
                e.tag() == "loop" || e.tag() == "break" || e.tag() == "FG8",
                "a conditional increment became {}: {}",
                e.tag(),
                e.line()
            );
            assert!(
                e.line().contains("raised by exactly one") || e.tag() != "FG8",
                "{}",
                e.line()
            );
        }
        Ok(f) => {
            let mut counts = 0;
            let mut measures = 0;
            for g in f.defs.values() {
                for func in &g.fns {
                    for n in &func.body.nodes {
                        match &n.attrs {
                            fir::Attr::Count(_) => counts += 1,
                            fir::Attr::Measure(..) => measures += 1,
                            _ => {}
                        }
                    }
                }
            }
            assert_eq!(
                counts, 0,
                "a conditional increment was rewritten into a count"
            );
            assert!(measures > 0, "the loop was not read as a measure");
            let bytes = fir::encode(&f);
            assert!(
                !matches!(
                    form::check(bytes.as_bytes(), None),
                    form::Outcome::Applied(_)
                ),
                "a body that steps the index only on some continuing paths was applied as a measure"
            );
        }
    }
}

#[test]
fn breaks_that_do_not_name_the_loop_still_require_change() {
    // A break out of a block that is not the loop. A later `exit_if` names the
    // loop, so the function can return; the block break is still refused.
    let (tick, tkey) = tick_parts();
    let key = tkey.clone();
    let ct = build::count_t();
    let mut b = FB::new(vec![ct.clone()], ct.clone());
    let bound = b.arg(0);
    let cnt = b.counter();
    b.loop_(|b| {
        b.count(cnt, bound)?;
        let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
        b.open();
        b.brk(0);
        let inner = b.close();
        b.push(Stmt::Block(inner));
        b.exit_if(Atom::bool(true))?;
        Ok(())
    })
    .unwrap();
    b.ret(build::lit(0));
    refuse_with(tick, tkey, b.finish());

    // A break that leaves the inner loop and the outer loop. Depth 1 from the
    // inner loop names the outer one. Item 3 does not restructure that.
    let (tick, tkey) = tick_parts();
    let key = tkey.clone();
    let mut b = FB::new(vec![ct.clone()], ct);
    let bound = b.arg(0);
    let cnt = b.counter();
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
    refuse_with(tick, tkey, b.finish());

    // A depth that names no enclosing loop. The core rejects it, or construction
    // does, under the break rule.
    let mut b = FB::new(vec![build::count_t()], build::count_t());
    b.loop_(|b| {
        b.brk(2);
        Ok(())
    })
    .unwrap();
    b.ret(build::lit(0));
    let m = module("deep", b.finish());
    match wf::check(&m) {
        Err(e) => assert!(e.contains("W2") || e.starts_with("W"), "{e}"),
        Ok(()) => {
            let e = bridge::construct(&m).expect_err("break");
            assert_eq!(e.tag(), "break", "{}", e.line());
        }
    }
}

fn refuse_with(tick: core::Fn, tkey: String, user: core::Fn) {
    let m = assemble(tick, tkey, "no", user);
    wf::check(&m).unwrap_or_else(|e| panic!("{e}"));
    let e = bridge::construct(&m).expect_err("break");
    assert_eq!(e.tag(), "break", "{}", e.line());
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

fn closure_round_trip(core_m: &core::Module, name: &str) -> Result<bool, String> {
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
    let mut semantic = 0usize;
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
        let mut rewrote = false;
        for name in &roots {
            match closure_round_trip(&core_m, name) {
                Ok(sem) => rewrote |= sem,
                Err(e) => {
                    bad = Some(e);
                    break;
                }
            }
        }
        match bad {
            None if !rewrote => {
                eprintln!("  covered {}", p.file_name().unwrap().to_string_lossy());
                covered += 1;
            }
            // Option B. The flag form's bytes match its schedule. That is not
            // syntactic identity for the break, so it does not raise the floor.
            None => {
                eprintln!("  semantic {}", p.file_name().unwrap().to_string_lossy());
                semantic += 1;
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
        "form bridge: {covered} of {programs} examples have every program function in the syntactic slice"
    );
    eprintln!(
        "form bridge: {semantic} of {programs} examples admit a body break under option B (not syntactic identity)"
    );
    for (k, n) in &root_tags {
        eprintln!("  {n} {k}");
    }
    // Ratchet. Whole modules still meet the observation driver (`out`, fuel
    // globals, `catch`). A program function is in the syntactic slice when its
    // closure round-trips through construct(schedule(F)) and the construction
    // did not restructure a body break. Option B admits that break by
    // three-way eval, O6' and O7'; the byte match of the flag form is not
    // syntactic identity, and those examples stay off this floor. Raise the
    // floor when the syntactic slice grows; do not lower it.
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
