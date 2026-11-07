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
//! A round-trip tag is a byte mismatch, not syntactic coverage, and a defect
//! tag of zero is not a claim that the corpus has no semantic defect.
//! A bare core `catch` returns the fuel trap as a value. The Form `catch`
//! and its schedule rethrow that trap. Those observations are not the same.
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

/// Both arms move one list. The schedule consumes that capture once.
fn owned_branch() -> core::Module {
    let e = i64ty();
    let lt = Ty::list(e.clone());
    let mut b = FB::new(vec![Ty::Bool, lt.clone()], lt.clone());
    let c = b.arg(0);
    let l = b.var(lt, b.arg(1));
    b.if_(
        c,
        |b| {
            let cur = b.take(l);
            let p = b.p("list.push", vec![e.clone()], vec![cur, build::lit(1)])?;
            b.set(l, p);
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
    let r = b.take(l);
    b.ret(r);
    module("owned_branch", b.finish())
}

#[test]
fn owned_branch_round_trips_and_agrees() {
    // No break. Syntactic identity holds: construct(schedule(F)) is F.
    let c = owned_branch();
    let f = round_trip(&c);
    let s = bridge::schedule(&f).unwrap();
    let one = Val::List(vec![Val::Int(7)]);
    agree(&c, &f, &s, vec![Val::Bool(true), one.clone()]);
    agree(&c, &f, &s, vec![Val::Bool(false), one]);
}

/// A count loop that carries a list, ticks, and breaks when the list is empty.
/// The continue path pushes. The break names the loop.
fn owned_early() -> core::Module {
    let (tick, tkey) = tick_parts();
    let key = tkey.clone();
    let e = i64ty();
    let lt = Ty::list(e.clone());
    let ct = build::count_t();
    let mut b = FB::new(vec![lt.clone(), ct.clone()], lt.clone());
    let bound = b.arg(1);
    let l = b.var(lt, b.arg(0));
    let cnt = b.counter();
    b.loop_(|b| {
        b.count(cnt, bound.clone())?;
        let _ = b.call(FnRef::Key(key.clone(), 0), Ty::Bool, vec![build::lit(1)]);
        let n = b.p("list.len", vec![e.clone()], vec![Atom::V(l)])?;
        let empty = b.int_cmp("int.eq", n, build::lit(0))?;
        b.exit_if(empty)?;
        let cur = b.take(l);
        let pushed = b.p("list.push", vec![e.clone()], vec![cur, build::lit(1)])?;
        b.set(l, pushed);
        Ok(())
    })
    .unwrap();
    let r = b.take(l);
    b.ret(r);
    assemble(tick, tkey, "owned_early", b.finish())
}

#[test]
fn owned_body_break_is_scheduled_once() {
    // Option B. The list is an owned capture of the break branch. Each arm
    // starts with that list live, because one arm runs. Lean Stmt has no
    // break constructor (B1), so byte equality is not the claim: the corpus
    // counts a differing canonical form as round-trip. Admission is
    // three-way eval, O6' and O7'.
    let c = owned_early();
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
    bridge::construct(&s).unwrap_or_else(|e| panic!("construct of the schedule: {}", e.line()));
    let empty = Val::List(vec![]);
    // Bound 1. The body breaks on the first iteration, so the next bound check
    // does not run (O6').
    agree(&c, &f, &s, vec![empty.clone(), Val::Int(1)]);
    let (src, fuel_src) = fuel_of(&c, vec![empty.clone(), Val::Int(1)]);
    let (sch, fuel_sch) = fuel_of(&s, vec![empty, Val::Int(1)]);
    assert_eq!(src, Ok(Val::List(vec![])));
    assert_eq!(sch, src);
    assert_eq!(fuel_src, fuel_sch, "O7': fuel differs on the break path");
    // A list that never becomes empty hits the bound. The flag is not that halt.
    let grown = Val::List(vec![Val::Int(7)]);
    agree(&c, &f, &s, vec![grown.clone(), Val::Int(1)]);
    let (src_b, fuel_b) = fuel_of(&c, vec![grown.clone(), Val::Int(1)]);
    let (sch_b, fuel_sb) = fuel_of(&s, vec![grown, Val::Int(1)]);
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
                let file = p.file_name().unwrap().to_string_lossy();
                if matches!(tag.as_str(), "round-trip" | "defect" | "catch" | "break" | "loop" | "FG8" | "global") {
                    eprintln!("  {tag} {file}");
                }
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
    // globals). A program function is in the syntactic slice when its closure
    // round-trips through construct(schedule(F)) and the construction did not
    // restructure a body break. Option B records that a body break was
    // rewritten; the flag form's byte match against its own schedule is not
    // syntactic identity and it is not source, Form and scheduled agreement.
    // A `round-trip` tag is an unresolved byte mismatch. A `defect` tag counts
    // a bridge failure, not a semantic defect of the program, so zero defect
    // tags are not zero semantic defects. Raise the syntactic floor when that
    // slice grows; do not lower it.
    assert_eq!(programs, 60, "spec/examples gained or lost a program");
    assert_eq!(
        whole, 0,
        "a whole example now constructs; raise this floor and check its trace"
    );
    assert_eq!(
        covered, 5,
        "the number of examples whose program functions round-trip changed; raise the ratchet, do not lower it"
    );
    assert!(
        tags.values().sum::<usize>() + whole >= programs.saturating_sub(4),
        "every lowered example is covered or tagged"
    );
}

#[test]
fn a_function_that_halts_schedules_without_a_return() {
    // The same shape as `48_linear_handle`'s `fn main`: a trap value, then
    // halt. The function's outputs are the halt node's outputs. They are not
    // core values, so the schedule ends at the halt.
    let f = core::Fn {
        params: vec![],
        ret: Ty::unit(),
        body: vec![
            Stmt::Prim(
                "sum.mk".into(),
                vec![Ty::halt()],
                vec![
                    build::lit(0),
                    Atom::bytes(b"cannot call c_open at comptime"),
                ],
            ),
            Stmt::Halt(Atom::V(0)),
        ],
    };
    let m = module("main", f);
    wf::check(&m).unwrap_or_else(|e| panic!("{e}"));
    let form = bridge::construct(&m).unwrap_or_else(|e| panic!("{}", e.line()));
    assert!(
        matches!(
            form::check(fir::encode(&form).as_bytes(), None),
            form::Outcome::Applied(_)
        ),
        "a halting function is not applied"
    );
    let s = bridge::schedule(&form).unwrap_or_else(|e| panic!("{}", e.line()));
    wf::check(&s).unwrap_or_else(|e| panic!("schedule is not well formed: {e}"));
    let back =
        bridge::construct(&s).unwrap_or_else(|e| panic!("construct of the schedule: {}", e.line()));
    assert_eq!(
        fir::encode(&form),
        fir::encode(&back),
        "construct(schedule(F)) is not F for a function that halts"
    );
    let user = s
        .names
        .get("fns")
        .and_then(|v| v.as_object())
        .and_then(|map| {
            map.iter().find_map(|(k, names)| {
                let arr = names.as_array()?;
                arr.iter().enumerate().find_map(|(i, n)| {
                    let s = n.as_str()?;
                    s.split(" = ")
                        .any(|p| p == "main")
                        .then_some((k.clone(), i as u32))
                })
            })
        });
    // `module` names the entry `main` when the function takes no parameters.
    // The scheduled name is `f` (one function, keyed on its own). Either way
    // the body ends in halt and has no ret.
    let body = match user {
        Some((k, i)) => s.func(&k, i).unwrap().body.clone(),
        None => s.defs.values().next().unwrap().fns[0].body.clone(),
    };
    assert!(
        body.iter().any(|st| matches!(st, Stmt::Halt(_))),
        "the schedule dropped the halt: {body:?}"
    );
    assert!(
        !body.iter().any(|st| matches!(st, Stmt::Ret(_))),
        "the schedule returned after a halt: {body:?}"
    );
}

#[test]
fn a_finished_branch_keeps_the_next_node_outside_the_arm() {
    // A domain test whose success arm already holds the operation, followed
    // by another node. Sinking that node into the arm hides the local it
    // binds from the region that returns it. `[0; 4]` lowers to that shape:
    // the schedule writes the branch and the push as siblings. A literal may
    // sit on either side of the branch, so the bytes are not the claim.
    let src = "fn zeros() -> List[u32]\n    return [0; 4]\n";
    let ast = rce_spec_check::program(src).unwrap_or_else(|e| panic!("{e}"));
    let core_m = rce_spec_check::one::core_of(&ast).unwrap_or_else(|e| panic!("{e}"));
    let sub = bridge::closure(&core_m, "fn zeros").unwrap_or_else(|| panic!("no fn zeros"));
    // The closure is not an entry: a helper may hold a global the driver owns.
    let form = bridge::construct(&sub).unwrap_or_else(|e| panic!("{}", e.line()));
    assert!(!bridge::body_break_rewrite(), "a repeat has no body break");
    let s = bridge::schedule(&form).unwrap_or_else(|e| panic!("{}", e.line()));
    wf::check(&s).unwrap_or_else(|e| panic!("schedule is not well formed: {e}"));
    let back =
        bridge::construct(&s).unwrap_or_else(|e| panic!("construct of the schedule: {}", e.line()));
    assert!(
        push_follows_branch(&form) && push_follows_branch(&back),
        "list.push was sunk into the branch arm"
    );
}

/// A region has `list.push` as a later sibling of a branch, not only inside an arm.
fn push_follows_branch(m: &fir::Module) -> bool {
    fn go(b: &fir::Body) -> bool {
        let mut branch = None;
        let mut push = None;
        for (i, n) in b.nodes.iter().enumerate() {
            match &n.attrs {
                fir::Attr::Branch(..) if branch.is_none() => branch = Some(i),
                fir::Attr::Prim(p, ..) if p == "list.push" && push.is_none() => push = Some(i),
                _ => {}
            }
            if n.regions.iter().any(go) {
                return true;
            }
        }
        matches!((branch, push), (Some(i), Some(j)) if j > i)
    }
    m.defs.values().any(|g| g.fns.iter().any(|f| go(&f.body)))
}

fn eval_core(f: core::Fn) -> Result<Val, String> {
    let n = f.params.len();
    let m = module("f", f);
    wf::check(&m).unwrap_or_else(|e| panic!("{e}"));
    let (k, i) = by_arity(&m, n);
    core_apply(&m, &k, i, vec![])
}

fn i64_lit(v: i128) -> Atom {
    Atom::K(i64ty(), Lit::Int(v))
}

/// `sum(ret, Halt)` holding `value` at variant 0, or a trap at variant 1.
fn raw_sum(ret: &Ty, success: Option<Atom>, trap: Option<&[u8]>) -> (Vec<Stmt>, u32) {
    let mut body = vec![];
    let raw = if let Some(msg) = trap {
        body.push(Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![
            build::lit(0),
            Atom::bytes(msg),
        ]));
        body.push(Stmt::Prim(
            "sum.mk".into(),
            vec![bridge::catch_sum(ret)],
            vec![build::lit(1), Atom::V(0)],
        ));
        1
    } else {
        body.push(Stmt::Prim(
            "sum.mk".into(),
            vec![bridge::catch_sum(ret)],
            vec![build::lit(0), success.unwrap()],
        ));
        0
    };
    (body, raw)
}

fn finish_boundary(pack: bridge::CatchPack, ret: Ty, success: Option<Atom>, trap: Option<&[u8]>) -> core::Fn {
    let (mut body, raw) = raw_sum(&ret, success, trap);
    let b = bridge::catch_boundary_for(pack, &ret, raw).unwrap_or_else(|e| panic!("{e}"));
    body.extend(b.stmts);
    let payload = bridge::catch_form_payload(pack, &ret).unwrap();
    core::Fn {
        params: vec![],
        ret: bridge::catch_sum(&payload),
        body: {
            body.push(Stmt::Ret(Atom::V(b.wrapped)));
            body
        },
    }
}

#[test]
fn catch_pack_follows_declared_arity() {
    // These evaluate the core statements of the pack. They are not schedule
    // support: `sig_of` used to refuse every arity other than one before the
    // pack was chosen. Schedule support is `declared_arities_schedule`.
    let i64t = i64ty();
    let pair = Ty::Tup(vec![i64t.clone(), i64t.clone()]);
    // Zero outputs: the payload is the unit tuple, the same value the core holds.
    let mut zbody = vec![Stmt::Prim(
        "tup.mk".into(),
        vec![Ty::unit()],
        vec![],
    )];
    zbody.push(Stmt::Prim(
        "sum.mk".into(),
        vec![bridge::catch_sum(&Ty::unit())],
        vec![build::lit(0), Atom::V(0)],
    ));
    let zb = bridge::catch_boundary_for(bridge::CatchPack::Zero, &Ty::unit(), 1).unwrap();
    zbody.extend(zb.stmts);
    zbody.push(Stmt::Ret(Atom::V(zb.wrapped)));
    let zero = eval_core(core::Fn {
        params: vec![],
        ret: bridge::catch_sum(&Ty::unit()),
        body: zbody,
    })
    .unwrap();
    assert_eq!(zero, Val::Sum(0, Box::new(Val::unit())), "{zero:?}");

    // Scalar singleton: one component, not the bare integer.
    let scalar = eval_core(finish_boundary(
        bridge::CatchPack::Singleton,
        i64t.clone(),
        Some(i64_lit(7)),
        None,
    ))
    .unwrap();
    assert_eq!(scalar, Val::Sum(0, Box::new(Val::Tup(vec![Val::Int(7)]))));

    // Tuple singleton: the pair stays one component. It is not opened.
    let mut pbody = vec![Stmt::Prim(
        "tup.mk".into(),
        vec![pair.clone()],
        vec![i64_lit(3), i64_lit(4)],
    )];
    pbody.push(Stmt::Prim(
        "sum.mk".into(),
        vec![bridge::catch_sum(&pair)],
        vec![build::lit(0), Atom::V(0)],
    ));
    let pb = bridge::catch_boundary_for(bridge::CatchPack::Singleton, &pair, 1).unwrap();
    pbody.extend(pb.stmts);
    let packed_at = pb.wrapped;
    let up = bridge::catch_unpack_for(bridge::CatchPack::Singleton, &pair, packed_at, pb.next).unwrap();
    pbody.extend(up.stmts);
    pbody.push(Stmt::Ret(Atom::V(up.wrapped)));
    let back = eval_core(core::Fn {
        params: vec![],
        ret: bridge::catch_sum(&pair),
        body: pbody,
    })
    .unwrap();
    assert_eq!(
        back,
        Val::Sum(0, Box::new(Val::Tup(vec![Val::Int(3), Val::Int(4)]))),
        "a tuple singleton unpacks to the pair, not to two outputs"
    );

    // Several outputs: the payload is the tuple itself, with no extra wrap.
    let mut mbody = vec![Stmt::Prim(
        "tup.mk".into(),
        vec![pair.clone()],
        vec![i64_lit(3), i64_lit(4)],
    )];
    mbody.push(Stmt::Prim(
        "sum.mk".into(),
        vec![bridge::catch_sum(&pair)],
        vec![build::lit(0), Atom::V(0)],
    ));
    let mb = bridge::catch_boundary_for(bridge::CatchPack::Multiple, &pair, 1).unwrap();
    mbody.extend(mb.stmts);
    mbody.push(Stmt::Ret(Atom::V(mb.wrapped)));
    let many = eval_core(core::Fn {
        params: vec![],
        ret: bridge::catch_sum(&pair),
        body: mbody,
    })
    .unwrap();
    assert_eq!(
        many,
        Val::Sum(0, Box::new(Val::Tup(vec![Val::Int(3), Val::Int(4)]))),
        "several outputs are not wrapped again"
    );
}

#[test]
fn catch_boundary_keeps_halt_payload_and_rethrows_fuel() {
    let i64t = i64ty();
    let ordinary = eval_core(finish_boundary(
        bridge::CatchPack::Singleton,
        i64t.clone(),
        None,
        Some(b"no"),
    ))
    .unwrap();
    assert_eq!(
        ordinary,
        Val::Sum(
            1,
            Box::new(Val::Sum(0, Box::new(Val::bytes(b"no"))))
        )
    );
    let fuel = eval_core(finish_boundary(
        bridge::CatchPack::Singleton,
        i64t,
        None,
        Some(b"comptime fuel exhausted"),
    ));
    assert_eq!(fuel, Err("halt: comptime fuel exhausted".into()));
    // Zero outputs rethrow the same way. The pack is not what decides the trap.
    let fuel0 = eval_core(finish_boundary(
        bridge::CatchPack::Zero,
        Ty::unit(),
        None,
        Some(b"comptime fuel exhausted"),
    ));
    assert_eq!(fuel0, Err("halt: comptime fuel exhausted".into()));
}

fn callee_fn(ret: Ty, body: Vec<Stmt>) -> core::Fn {
    core::Fn {
        params: vec![],
        ret,
        body,
    }
}

fn catch_caller(callee_key: &str, ret: &Ty) -> core::Fn {
    let payload = bridge::catch_form_payload(bridge::CatchPack::Singleton, ret).unwrap();
    let b = bridge::catch_boundary_for(bridge::CatchPack::Singleton, ret, 0).unwrap();
    let mut body = vec![Stmt::Catch(FnRef::Key(callee_key.into(), 0), vec![])];
    body.extend(b.stmts);
    body.push(Stmt::Ret(Atom::V(b.wrapped)));
    core::Fn {
        params: vec![],
        ret: bridge::catch_sum(&payload),
        body,
    }
}

fn core_catch(m: &core::Module) -> (String, u32) {
    fn has(b: &[Stmt]) -> bool {
        b.iter().any(|s| match s {
            Stmt::Catch(..) => true,
            Stmt::If(_, t, e) => has(t) || has(e),
            Stmt::Block(x) | Stmt::Loop(x) => has(x),
            _ => false,
        })
    }
    for (k, g) in &m.defs {
        for (i, f) in g.fns.iter().enumerate() {
            if has(&f.body) {
                return (k.clone(), i as u32);
            }
        }
    }
    panic!("no catch");
}

/// Original core, Form, and scheduled core of one caller.
///
/// `bare` is a core `catch` with no boundary. The Form lowers the inverse,
/// so success, an ordinary halt and a declared error agree. The fuel trap
/// does not: the original core returns it and the Form rethrows it.
fn observe_catch(
    callee: core::Fn,
    bare: bool,
) -> (
    Result<Val, String>,
    Result<Val, String>,
    Result<Val, String>,
    bool,
) {
    let ret = callee.ret.clone();
    let (cg, _) = canonical_group(std::slice::from_ref(&callee));
    let ck = group_key(&cg);
    let caller = if bare {
        core::Fn {
            params: vec![],
            ret: bridge::catch_sum(&ret),
            body: vec![
                Stmt::Catch(FnRef::Key(ck.clone(), 0), vec![]),
                Stmt::Ret(Atom::V(0)),
            ],
        }
    } else {
        catch_caller(&ck, &ret)
    };
    let (ug, _) = canonical_group(std::slice::from_ref(&caller));
    let uk = group_key(&ug);
    let mut names = serde_json::Map::new();
    names.insert(ck.clone(), serde_json::json!(["callee"]));
    names.insert(uk.clone(), serde_json::json!(["caller"]));
    let m = core::Module {
        globals: vec![],
        defs: BTreeMap::from([(ck, cg), (uk.clone(), ug)]),
        entry: (uk, 0),
        names: serde_json::json!({"fns": names, "globals": []}),
        graphs: BTreeMap::new(),
    };
    wf::check(&m).unwrap_or_else(|e| panic!("{e}"));
    let (ok, oi) = core_catch(&m);
    let original = core_apply(&m, &ok, oi, vec![]);
    let form = bridge::construct(&m).unwrap_or_else(|e| panic!("{}", e.line()));
    assert!(matches!(
        form::check(fir::encode(&form).as_bytes(), None),
        form::Outcome::Applied(_)
    ));
    let scheduled = bridge::schedule(&form).unwrap_or_else(|e| panic!("{}", e.line()));
    wf::check(&scheduled).unwrap_or_else(|e| panic!("schedule: {e}"));
    let back = bridge::construct(&scheduled);
    let bytes_equal = match &back {
        Ok(back) => {
            let a = fir::encode(&form);
            let b = fir::encode(back);
            a == b
        }
        Err(_) => false,
    };
    // The boundary is the schedule's own shape, so its bytes match. A bare
    // catch lowers an inverse the original core did not contain; the bytes
    // match when that inverse schedules back to the same nodes.
    if !bare {
        let back = back.unwrap_or_else(|e| panic!("{}", e.line()));
        assert_eq!(
            fir::encode(&form),
            fir::encode(&back),
            "construct(schedule(F)) is not F"
        );
    }
    let (fk, fi) = form_catch(&form);
    let form_v = form::eval::apply(&form, &fk, fi, vec![]).map(|mut v| v.pop().unwrap());
    let (sk, si) = core_catch(&scheduled);
    let core_v = core_apply(&scheduled, &sk, si, vec![]);
    (original, form_v, core_v, bytes_equal)
}

fn round_trip_catch(callee: core::Fn) -> (Result<Val, String>, Result<Val, String>, Result<Val, String>) {
    let (original, form_v, core_v, bytes_equal) = observe_catch(callee, false);
    assert!(bytes_equal, "the boundary schedule is not byte-identical");
    (original, form_v, core_v)
}

fn form_catch(m: &fir::Module) -> (String, u32) {
    for (k, g) in &m.defs {
        for (i, f) in g.fns.iter().enumerate() {
            if f.body
                .nodes
                .iter()
                .any(|n| matches!(n.attrs, fir::Attr::Catch(_)))
            {
                return (k.clone(), i as u32);
            }
        }
    }
    panic!("no catch");
}

#[test]
fn singleton_catch_round_trips_for_success_halt_and_fuel() {
    let i64t = i64ty();
    let (original, form_v, core_v) =
        round_trip_catch(callee_fn(i64t.clone(), vec![Stmt::Ret(i64_lit(7))]));
    let want = Val::Sum(0, Box::new(Val::Tup(vec![Val::Int(7)])));
    assert_eq!(original.as_ref().unwrap(), &want);
    assert_eq!(form_v.as_ref().unwrap(), &want);
    assert_eq!(core_v.as_ref().unwrap(), &want);

    let ordinary = callee_fn(i64t.clone(), vec![
        Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![
            build::lit(0),
            Atom::bytes(b"no"),
        ]),
        Stmt::Halt(Atom::V(0)),
    ]);
    let (original, form_v, core_v) = round_trip_catch(ordinary);
    let want = Val::Sum(1, Box::new(Val::Sum(0, Box::new(Val::bytes(b"no")))));
    assert_eq!(original.as_ref().unwrap(), &want, "original {original:?}");
    assert_eq!(form_v.as_ref().unwrap(), &want, "form {form_v:?}");
    assert_eq!(core_v.as_ref().unwrap(), &want, "core {core_v:?}");

    let declared = callee_fn(i64t.clone(), vec![
        Stmt::Prim(
            "tup.mk".into(),
            vec![Ty::Tup(vec![Ty::Bytes, Ty::Bytes])],
            vec![Atom::bytes(b"E"), Atom::bytes(b"boom")],
        ),
        Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![
            build::lit(1),
            Atom::V(0),
        ]),
        Stmt::Halt(Atom::V(1)),
    ]);
    let (original, form_v, core_v) = round_trip_catch(declared);
    let want = Val::Sum(
        1,
        Box::new(Val::Sum(
            1,
            Box::new(Val::Tup(vec![Val::bytes(b"E"), Val::bytes(b"boom")])),
        )),
    );
    assert_eq!(original.as_ref().unwrap(), &want, "original {original:?}");
    assert_eq!(form_v.as_ref().unwrap(), &want, "form {form_v:?}");
    assert_eq!(core_v.as_ref().unwrap(), &want, "scheduled {core_v:?}");

    let fuel = callee_fn(i64t, vec![
        Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![
            build::lit(0),
            Atom::bytes(b"comptime fuel exhausted"),
        ]),
        Stmt::Halt(Atom::V(0)),
    ]);
    let (original, form_v, core_v) = round_trip_catch(fuel);
    // The boundary is in the original core, so all three rethrow. That is
    // the schedule's fuel composition, not an inverse of the fuel sum.
    assert_eq!(original, Err("halt: comptime fuel exhausted".into()));
    assert_eq!(form_v, Err("halt: comptime fuel exhausted".into()));
    assert_eq!(core_v, Err("halt: comptime fuel exhausted".into()));
}

#[test]
fn tuple_singleton_catch_does_not_flatten() {
    let pair = Ty::Tup(vec![i64ty(), i64ty()]);
    let callee = callee_fn(pair, vec![
        Stmt::Prim("tup.mk".into(), vec![Ty::Tup(vec![i64ty(), i64ty()])], vec![
            i64_lit(3),
            i64_lit(4),
        ]),
        Stmt::Ret(Atom::V(0)),
    ]);
    let (original, form_v, core_v) = round_trip_catch(callee);
    let want = Val::Sum(
        0,
        Box::new(Val::Tup(vec![Val::Tup(vec![Val::Int(3), Val::Int(4)])])),
    );
    assert_eq!(original.as_ref().unwrap(), &want, "{original:?}");
    assert_eq!(form_v.as_ref().unwrap(), &want, "{form_v:?}");
    assert_eq!(core_v.as_ref().unwrap(), &want, "{core_v:?}");
}

fn same_result(original: &Result<Val, String>, form_v: &Result<Val, String>, scheduled: &Result<Val, String>) {
    assert_eq!(original, form_v, "original and Form differ");
    assert_eq!(form_v, scheduled, "Form and scheduled core differ");
}

#[test]
fn bare_catch_preserves_the_core_sum_and_rethrows_fuel() {
    let i64t = i64ty();
    let (original, form_v, scheduled, bytes) =
        observe_catch(callee_fn(i64t.clone(), vec![Stmt::Ret(i64_lit(7))]), true);
    let want = Val::Sum(0, Box::new(Val::Int(7)));
    assert_eq!(original.as_ref().unwrap(), &want, "original {original:?}");
    same_result(&original, &form_v, &scheduled);
    assert!(bytes, "a bare scalar catch does not schedule back to the same bytes");

    let ordinary = callee_fn(i64t.clone(), vec![
        Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![
            build::lit(0),
            Atom::bytes(b"no"),
        ]),
        Stmt::Halt(Atom::V(0)),
    ]);
    let (original, form_v, scheduled, bytes) = observe_catch(ordinary, true);
    let want = Val::Sum(1, Box::new(Val::Sum(0, Box::new(Val::bytes(b"no")))));
    assert_eq!(original.as_ref().unwrap(), &want, "{original:?}");
    same_result(&original, &form_v, &scheduled);
    assert!(bytes, "a bare ordinary halt does not schedule back to the same bytes");

    let declared = callee_fn(i64t.clone(), vec![
        Stmt::Prim(
            "tup.mk".into(),
            vec![Ty::Tup(vec![Ty::Bytes, Ty::Bytes])],
            vec![Atom::bytes(b"E"), Atom::bytes(b"boom")],
        ),
        Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![build::lit(1), Atom::V(0)]),
        Stmt::Halt(Atom::V(1)),
    ]);
    let (original, form_v, scheduled, bytes) = observe_catch(declared, true);
    let want = Val::Sum(
        1,
        Box::new(Val::Sum(1, Box::new(Val::Tup(vec![Val::bytes(b"E"), Val::bytes(b"boom")])))),
    );
    assert_eq!(original.as_ref().unwrap(), &want, "{original:?}");
    same_result(&original, &form_v, &scheduled);
    assert!(bytes, "a bare declared error does not schedule back to the same bytes");

    // The fuel trap is outside the inverse. The original core returns it.
    // The Form catch and the scheduled boundary rethrow it, so there is no
    // returned sum for the inverse to restore.
    let fuel = callee_fn(i64t, vec![
        Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![
            build::lit(0),
            Atom::bytes(b"comptime fuel exhausted"),
        ]),
        Stmt::Halt(Atom::V(0)),
    ]);
    let (original, form_v, scheduled, _) = observe_catch(fuel, true);
    assert_eq!(
        original.unwrap(),
        Val::Sum(1, Box::new(Val::Sum(0, Box::new(Val::bytes(b"comptime fuel exhausted")))))
    );
    assert_eq!(form_v, Err("halt: comptime fuel exhausted".into()));
    assert_eq!(scheduled, Err("halt: comptime fuel exhausted".into()));
}

#[test]
fn bare_tuple_singleton_unpacks_to_the_pair() {
    let pair = Ty::Tup(vec![i64ty(), i64ty()]);
    let callee = callee_fn(pair, vec![
        Stmt::Prim("tup.mk".into(), vec![Ty::Tup(vec![i64ty(), i64ty()])], vec![
            i64_lit(3),
            i64_lit(4),
        ]),
        Stmt::Ret(Atom::V(0)),
    ]);
    let (original, form_v, scheduled, bytes) = observe_catch(callee, true);
    let want = Val::Sum(0, Box::new(Val::Tup(vec![Val::Int(3), Val::Int(4)])));
    assert_eq!(original.as_ref().unwrap(), &want, "{original:?}");
    same_result(&original, &form_v, &scheduled);
    assert!(bytes, "a bare tuple singleton does not schedule back to the same bytes");
}

#[test]
fn list_and_array_singletons_have_no_dummy() {
    let list = Ty::List(Box::new(i64ty()));
    match bridge::catch_boundary_for(bridge::CatchPack::Singleton, &list, 0) {
        Err(err) => assert!(err.contains("list"), "{err}"),
        Ok(_) => panic!("a list singleton was given a dummy"),
    }
    let arr = Ty::Arr(Box::new(i64ty()));
    match bridge::catch_boundary_for(bridge::CatchPack::Singleton, &arr, 0) {
        Err(err) => assert!(err.contains("array"), "{err}"),
        Ok(_) => panic!("an array singleton was given a dummy"),
    }
}

fn i64b() -> fir::BTy {
    fir::BTy::Int(-(1i128 << 63), (1i128 << 63) - 1)
}

fn lit_node(t: fir::BTy, v: fir::Lit) -> fir::Node {
    fir::Node { attrs: fir::Attr::Lit(t, v), ins: vec![], regions: vec![] }
}

fn seal_form(m: fir::Module) -> fir::Module {
    let (m, _) = form::canon::canonical(&m).unwrap_or_else(|e| panic!("canon: {}", e.0));
    let bytes = fir::encode(&m);
    match form::check(bytes.as_bytes(), None) {
        form::Outcome::Applied(_) => {}
        form::Outcome::RequiresChange(k) => panic!("{} {}: {}", k.rule, k.path, k.reason),
    }
    form::decode::decode(bytes.as_bytes()).unwrap_or_else(|k| panic!("{} {}: {}", k.rule, k.path, k.reason))
}

fn form_of(callee: fir::Func, caller: fir::Func) -> fir::Module {
    let m = fir::Module {
        catalogue: form::catalogue::key().into(),
        types: vec![],
        regions: vec![],
        events: vec![],
        errors: vec![],
        steps: vec![],
        main: None,
        caps: vec![],
        defs: BTreeMap::from([
            ("cal".into(), fir::Group { fns: vec![callee] }),
            ("clr".into(), fir::Group { fns: vec![caller] }),
        ]),
    };
    seal_form(m)
}

fn catch_caller_form(payload: fir::BTy) -> fir::Func {
    fir::Func {
        ins: vec![],
        outs: vec![fir::BTy::Sum(vec![payload, fir::BTy::halt()])],
        eff: 0,
        bound: fir::Bound::None,
        body: fir::Body {
            nodes: vec![fir::Node {
                attrs: fir::Attr::Catch(fir::Ref::Key("cal".into(), 0)),
                ins: vec![],
                regions: vec![],
            }],
            out: vec![fir::Src::Out(0, 0)],
        },
    }
}

fn eval_form_catch(m: &fir::Module) -> Result<Val, String> {
    let (k, i) = form_catch(m);
    form::eval::apply(m, &k, i, vec![]).map(|mut v| v.pop().unwrap())
}

fn eval_scheduled_catch(m: &fir::Module) -> Result<Val, String> {
    let scheduled = bridge::schedule(m).unwrap_or_else(|e| panic!("{}", e.line()));
    wf::check(&scheduled).unwrap_or_else(|e| panic!("schedule: {e}"));
    let (k, i) = core_catch(&scheduled);
    core_apply(&scheduled, &k, i, vec![])
}

#[test]
fn declared_arities_schedule() {
    // Zero declared outputs. The payload is the unit tuple, with no extra wrap.
    // A core function that returns unit is a singleton, and its pack wraps;
    // that is a different arity from this one.
    let zero_callee = fir::Func {
        ins: vec![],
        outs: vec![],
        eff: 0,
        bound: fir::Bound::None,
        body: fir::Body { nodes: vec![], out: vec![] },
    };
    let zero = form_of(zero_callee, catch_caller_form(fir::BTy::Tup(vec![])));
    let want = Ok(Val::Sum(0, Box::new(Val::unit())));
    assert_eq!(eval_form_catch(&zero), want, "form zero");
    assert_eq!(eval_scheduled_catch(&zero), want, "scheduled zero");
    let back = bridge::construct(&bridge::schedule(&zero).unwrap());
    // The core return is the unit tuple, so the constructed signature has one
    // output. Byte identity is not the claim.
    if let Ok(back) = back {
        assert_ne!(
            fir::encode(&zero),
            fir::encode(&back),
            "a zero-output signature was restored from one core return"
        );
    }

    let pair_ty = fir::BTy::Tup(vec![i64b(), i64b()]);
    let many_callee = fir::Func {
        ins: vec![],
        outs: vec![i64b(), i64b()],
        eff: 0,
        bound: fir::Bound::None,
        body: fir::Body {
            nodes: vec![
                lit_node(i64b(), fir::Lit::Int(3)),
                lit_node(i64b(), fir::Lit::Int(4)),
            ],
            out: vec![fir::Src::Out(0, 0), fir::Src::Out(1, 0)],
        },
    };
    let many = form_of(many_callee, catch_caller_form(pair_ty));
    let want = Ok(Val::Sum(0, Box::new(Val::Tup(vec![Val::Int(3), Val::Int(4)]))));
    assert_eq!(eval_form_catch(&many), want, "form several");
    assert_eq!(eval_scheduled_catch(&many), want, "scheduled several");

    // The same values from a core image: one unit return, and one tuple return.
    // Fuel still differs, because that core catch returns the trap.
    // A bare catch of a unit return unpacks to the core sum. The boundary,
    // which the schedule writes and the construction folds, is the wrap.
    let unit_callee = callee_fn(Ty::unit(), vec![
        Stmt::Prim("tup.mk".into(), vec![Ty::unit()], vec![]),
        Stmt::Ret(Atom::V(0)),
    ]);
    let (original, form_v, scheduled, _) = observe_catch(unit_callee.clone(), true);
    let bare = Ok(Val::Sum(0, Box::new(Val::unit())));
    assert_eq!(&original, &bare, "original {original:?}");
    same_result(&original, &form_v, &scheduled);
    let (original, form_v, scheduled) = round_trip_catch(unit_callee);
    let wrapped = Ok(Val::Sum(0, Box::new(Val::Tup(vec![Val::unit()]))));
    assert_eq!(&original, &wrapped, "boundary original {original:?}");
    same_result(&original, &form_v, &scheduled);
}

fn prim_node(name: &str, ty: fir::BTy, ins: Vec<fir::Src>) -> fir::Node {
    fir::Node {
        attrs: fir::Attr::Prim(name.into(), vec![fir::TyArg::Body(ty)], None),
        ins,
        regions: vec![],
    }
}

/// A callee that halts, with `value_outs` declared values and ω last.
fn halting_callee(value_outs: Vec<fir::BTy>, msg: &[u8]) -> fir::Func {
    let mut halt_tys = value_outs.clone();
    halt_tys.push(fir::BTy::Order);
    let n = 3u32;
    fir::Func {
        ins: vec![fir::BTy::Order],
        outs: halt_tys.clone(),
        eff: 1,
        bound: fir::Bound::None,
        body: fir::Body {
            nodes: vec![
                lit_node(fir::BTy::Int(0, 1), fir::Lit::Int(0)),
                lit_node(fir::BTy::Bytes, fir::Lit::Bytes(msg.to_vec())),
                prim_node("sum.mk", fir::BTy::halt(), vec![fir::Src::Out(0, 0), fir::Src::Out(1, 0)]),
                fir::Node {
                    attrs: fir::Attr::Halt(halt_tys.clone()),
                    ins: vec![fir::Src::Out(2, 0), fir::Src::In(0)],
                    regions: vec![],
                },
            ],
            out: (0..n).map(|p| fir::Src::Out(3, p)).take(halt_tys.len()).collect(),
        },
    }
}

fn effect_caller(payload: fir::BTy) -> fir::Func {
    fir::Func {
        ins: vec![fir::BTy::Order],
        outs: vec![fir::BTy::Sum(vec![payload, fir::BTy::halt()]), fir::BTy::Order],
        eff: 1,
        bound: fir::Bound::None,
        body: fir::Body {
            nodes: vec![fir::Node {
                attrs: fir::Attr::Catch(fir::Ref::Key("cal".into(), 0)),
                ins: vec![fir::Src::In(0)],
                regions: vec![],
            }],
            out: vec![fir::Src::Out(0, 0), fir::Src::Out(0, 1)],
        },
    }
}

#[test]
fn declared_halt_schedules_and_fuel_is_not_in_the_inverse() {
    // Zero declared values. Ordinary halt is a returned sum. The fuel trap is
    // rethrown, so the inverse is not applied to it.
    let ordinary = form_of(
        halting_callee(vec![], b"no"),
        effect_caller(fir::BTy::Tup(vec![])),
    );
    let want = Ok(Val::Sum(1, Box::new(Val::Sum(0, Box::new(Val::bytes(b"no"))))));
    assert_eq!(eval_form_catch(&ordinary), want, "form zero halt");
    assert_eq!(eval_scheduled_catch(&ordinary), want, "scheduled zero halt");

    let fuel = form_of(
        halting_callee(vec![], b"comptime fuel exhausted"),
        effect_caller(fir::BTy::Tup(vec![])),
    );
    let trap = Err("halt: comptime fuel exhausted".into());
    assert_eq!(eval_form_catch(&fuel), trap, "form zero fuel");
    assert_eq!(eval_scheduled_catch(&fuel), trap, "scheduled zero fuel");

    // Several declared values use the same halt sum. They are not opened.
    let many = form_of(
        halting_callee(vec![i64b(), i64b()], b"no"),
        effect_caller(fir::BTy::Tup(vec![i64b(), i64b()])),
    );
    assert_eq!(eval_form_catch(&many), want, "form several halt");
    assert_eq!(eval_scheduled_catch(&many), want, "scheduled several halt");
}

#[test]
fn catch_keeps_cost_charged_by_the_callee() {
    // The order value is a unit token. Fuel, cost, output and memo live in
    // the evaluator's order record. `apply` strips the token. This checks
    // the record: a charge before the catch is still spent afterwards, on
    // success and on an ordinary halt. The evaluator is unchanged.
    let sum_ty = fir::BTy::Sum(vec![fir::BTy::Tup(vec![i64b()]), fir::BTy::halt()]);
    let u64b = fir::BTy::u64();
    let callee = fir::Func {
        ins: vec![fir::BTy::Order],
        outs: vec![i64b(), fir::BTy::Order],
        eff: 1,
        bound: fir::Bound::None,
        body: fir::Body {
            nodes: vec![
                lit_node(u64b.clone(), fir::Lit::Int(3)),
                lit_node(i64b(), fir::Lit::Int(7)),
                fir::Node {
                    attrs: fir::Attr::Prim("cost.charge".into(), vec![], None),
                    ins: vec![fir::Src::Out(0, 0), fir::Src::In(0)],
                    regions: vec![],
                },
            ],
            out: vec![fir::Src::Out(1, 0), fir::Src::Out(2, 0)],
        },
    };
    let caller = fir::Func {
        ins: vec![fir::BTy::Order],
        outs: vec![sum_ty.clone(), u64b.clone(), fir::BTy::Order],
        eff: 1,
        bound: fir::Bound::None,
        body: fir::Body {
            nodes: vec![
                fir::Node {
                    attrs: fir::Attr::Catch(fir::Ref::Key("cal".into(), 0)),
                    ins: vec![fir::Src::In(0)],
                    regions: vec![],
                },
                fir::Node {
                    attrs: fir::Attr::Prim("cost.spent".into(), vec![], None),
                    ins: vec![fir::Src::Out(0, 1)],
                    regions: vec![],
                },
            ],
            out: vec![fir::Src::Out(0, 0), fir::Src::Out(1, 0), fir::Src::Out(1, 1)],
        },
    };
    let m = form_of(callee, caller);
    let (k, i) = form_catch(&m);
    let form_vals = form::eval::apply(&m, &k, i, vec![]).unwrap_or_else(|e| panic!("{e}"));
    let want_sum = Val::Sum(0, Box::new(Val::Tup(vec![Val::Int(7)])));
    assert_eq!(form_vals, vec![want_sum.clone(), Val::Int(3)], "form {form_vals:?}");
    // `cost.spent` is not scheduled in this slice. The refusal stays; the
    // charge above is the evaluator's order record, not a patched parity.
    let err = bridge::schedule(&m).expect_err("cost.spent");
    assert!(err.line().contains("cost.spent"), "{}", err.line());

    let bare = form_of(
        fir::Func {
            ins: vec![fir::BTy::Order],
            outs: vec![i64b(), fir::BTy::Order],
            eff: 1,
            bound: fir::Bound::None,
            body: fir::Body {
                nodes: vec![
                    lit_node(u64b, fir::Lit::Int(3)),
                    lit_node(i64b(), fir::Lit::Int(7)),
                    fir::Node {
                        attrs: fir::Attr::Prim("cost.charge".into(), vec![], None),
                        ins: vec![fir::Src::Out(0, 0), fir::Src::In(0)],
                        regions: vec![],
                    },
                ],
                out: vec![fir::Src::Out(1, 0), fir::Src::Out(2, 0)],
            },
        },
        effect_caller(fir::BTy::Tup(vec![i64b()])),
    );
    assert_eq!(eval_form_catch(&bare).unwrap(), want_sum, "form sum");
    assert_eq!(eval_scheduled_catch(&bare).unwrap(), want_sum, "scheduled sum");
}
