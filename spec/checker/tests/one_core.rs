//! The core `rce-core/1` of the one emitter (spec/proposals/one-emitter.md,
//! order of work step 2): its JSON form, its well-formedness rules W1 to
//! W6, its reference evaluator, and the reference trace every printed
//! program must reproduce (ONE-5).
use rce_spec_check::one::{self, ir::*};
use std::path::PathBuf;

fn lit(v: i128) -> Atom {
    Atom::K(Ty::Int(v, v), Lit::Int(v))
}

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

/// ONE-5: the reference trace is what today's emitted Python prints under
/// `--trace --fuel-trace`, byte for byte, without the entries of D. The
/// emitted program prints the entry of each event (`d applied HASH BODY`
/// or `d kept HASH BODY`) before its line, and its whole output is the
/// reference's whole trace (`emit::observe_trace`); the core's trace does
/// not carry the entries yet.
#[test]
fn reference_trace_is_the_emitted_trace() {
    if std::process::Command::new("python3").arg("--version").output().is_err() {
        eprintln!("python3 is not available; skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("rce-one-ref-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut n = 0;
    for (stem, src) in examples() {
        let m = module(&src);
        let Ok(code) = rce_spec_check::emit::emit(&m, rce_spec_check::emit::Lang::Python) else { continue };
        let f = dir.join(format!("{stem}.py"));
        std::fs::write(&f, code).unwrap();
        let o = std::process::Command::new("python3").arg(&f).arg("--trace").arg("--fuel-trace").output().unwrap();
        let got = String::from_utf8_lossy(&o.stdout);
        let mut whole = rce_spec_check::emit::observe_trace(&m).join("\n");
        if !whole.is_empty() {
            whole.push('\n');
        }
        assert_eq!(got, whole, "{stem}: the whole trace");
        let without_d: String = got.lines().filter(|l| !l.starts_with("d applied ") && !l.starts_with("d kept ")).map(|l| format!("{l}\n")).collect();
        assert_eq!(without_d, one::trace::reference(&m), "{stem}");
        n += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(n >= 55, "{n} programs compared");
}

// ---- W1 to W6 on small modules written by hand

/// A JSON round trip of every hand-written module.
fn round_trip(m: &Module) {
    let t = one::json::module_text(m);
    assert!(one::json::module_from_text(&t).unwrap() == *m);
}

/// One group of `fns` in canonical order (W5), entered at the first of them.
fn module_of(fns: Vec<Fn>, globals: Vec<(Ty, Option<Lit>)>) -> Module {
    let (g, class) = one::json::canonical_group(&fns);
    let key = one::json::group_key(&g);
    Module { globals, defs: [(key.clone(), g)].into_iter().collect(), entry: (key, class[0]), names: serde_json::json!({}) }
}

fn entry(body: Vec<Stmt>) -> Fn {
    Fn { params: vec![], ret: Ty::unit(), body }
}

/// `v_n = tup.mk[tup()](); ret v_n`
fn unit_ret(n: u32) -> Vec<Stmt> {
    vec![Stmt::Prim("tup.mk".into(), vec![Ty::unit()], vec![]), Stmt::Ret(Atom::V(n))]
}

fn wf_err(m: &Module) -> String {
    round_trip(m);
    one::wf::check(m).err().unwrap_or_default()
}

fn int9() -> Ty {
    Ty::Int(0, 9)
}

#[test]
fn w_rules() {
    // a well-formed module
    let ok = module_of(vec![entry(unit_ret(0))], vec![]);
    assert_eq!(one::wf::check(&ok), Ok(()));

    // W1: an application of the wrong type, a ret of the wrong type
    let mut b = vec![Stmt::Prim("int.add".into(), vec![int9()], vec![Atom::bool(true), lit(1)])];
    b.extend(unit_ret(1));
    assert!(wf_err(&module_of(vec![entry(b)], vec![])).contains("W1"));
    let bad_ret = module_of(vec![entry(vec![Stmt::Ret(lit(3))])], vec![]);
    assert!(wf_err(&bad_ret).contains("W1"), "{}", wf_err(&bad_ret));

    // W2: a path without ret, a break with no enclosing loop
    let no_ret = module_of(vec![entry(vec![Stmt::Prim("tup.mk".into(), vec![Ty::unit()], vec![]), Stmt::Prim("drop".into(), vec![Ty::unit()], vec![Atom::V(0)])])], vec![]);
    assert!(wf_err(&no_ret).contains("W2"), "{}", wf_err(&no_ret));
    let mut b = vec![Stmt::Break(0)];
    b.extend(unit_ret(0));
    assert!(wf_err(&module_of(vec![entry(b)], vec![])).contains("W2"));

    // W3: an owned local never consumed, and one consumed twice
    let mut b = vec![Stmt::Prim("list.new".into(), vec![Ty::Bool], vec![])];
    b.extend(unit_ret(1));
    assert!(wf_err(&module_of(vec![entry(b)], vec![])).contains("W3"));
    let twice = module_of(vec![entry(vec![Stmt::Prim("tup.mk".into(), vec![Ty::unit()], vec![]), Stmt::Let(Atom::V(0)), Stmt::Ret(Atom::V(0))])], vec![]);
    assert!(wf_err(&twice).contains("W3"), "{}", wf_err(&twice));
    // ... and the paths of an if must agree
    let mut b = vec![
        Stmt::Prim("list.new".into(), vec![Ty::Bool], vec![]),
        Stmt::If(Atom::bool(true), vec![Stmt::Prim("drop".into(), vec![Ty::list(Ty::Bool)], vec![Atom::V(0)])], vec![]),
    ];
    b.extend(unit_ret(1));
    assert!(wf_err(&module_of(vec![entry(b)], vec![])).contains("W3"));

    // W4: a checked primitive without its domain test, then with it
    let mut b = vec![Stmt::Prim("int.add".into(), vec![int9()], vec![lit(5), lit(5)])];
    b.extend(unit_ret(1));
    assert!(wf_err(&module_of(vec![entry(b)], vec![])).contains("W4"));
    let halt = |t: &str| vec![Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![lit(0), Atom::bytes(t.as_bytes())]), Stmt::Halt(Atom::V(1))];
    let mut b = vec![
        Stmt::Prim("int.bad.add".into(), vec![int9()], vec![lit(5), lit(5)]),
        Stmt::If(Atom::V(0), halt("overflow"), vec![]),
        Stmt::Prim("int.add".into(), vec![int9()], vec![lit(4), lit(5)]),
    ];
    b.extend(unit_ret(3));
    assert!(wf_err(&module_of(vec![entry(b)], vec![])).contains("W4"), "a test of other operands does not count");
    let mut b = vec![
        Stmt::Prim("int.bad.add".into(), vec![int9()], vec![lit(4), lit(5)]),
        Stmt::If(Atom::V(0), halt("overflow"), vec![]),
        Stmt::Prim("int.add".into(), vec![int9()], vec![lit(4), lit(5)]),
    ];
    b.extend(unit_ret(3));
    assert_eq!(one::wf::check(&module_of(vec![entry(b)], vec![])), Ok(()));
    // sum.get needs a test of the tag
    let s = Ty::opt(Ty::Bool);
    let mut b = vec![
        Stmt::Prim("tup.mk".into(), vec![Ty::unit()], vec![]),
        Stmt::Prim("sum.mk".into(), vec![s.clone()], vec![lit(0), Atom::V(0)]),
        Stmt::Prim("sum.get".into(), vec![s.clone()], vec![Atom::V(1), lit(1)]),
    ];
    b.extend(unit_ret(3));
    assert!(wf_err(&module_of(vec![entry(b)], vec![])).contains("W4"));

    // W5: a key that is not the hash of the group, a rec outside its group
    let mut m = ok.clone();
    let g = m.defs.values().next().unwrap().clone();
    m.defs = [("00".repeat(32), g)].into_iter().collect();
    m.entry.0 = "00".repeat(32);
    assert!(wf_err(&m).contains("W5"));
    let mut b = vec![Stmt::Call(FnRef::Rec(3), vec![])];
    b.extend(unit_ret(1));
    assert!(wf_err(&module_of(vec![entry(b)], vec![])).contains("W5"));

    // W6: a global as an operand of call, a set of a parameter
    let callee = Fn { params: vec![Ty::Bool], ret: Ty::Bool, body: vec![Stmt::Ret(Atom::V(0))] };
    let mut b = vec![Stmt::Call(FnRef::Rec(1), vec![Atom::G(0)])];
    b.extend(unit_ret(1));
    let m = module_of(vec![entry(b), callee], vec![(Ty::Bool, Some(Lit::Bool(true)))]);
    assert!(wf_err(&m).contains("W6"), "{}", wf_err(&m));
    let setp = Fn { params: vec![Ty::Bool], ret: Ty::Bool, body: vec![Stmt::Set(Place::V(0), Atom::bool(false)), Stmt::Ret(Atom::V(0))] };
    let mut b = vec![Stmt::Call(FnRef::Rec(1), vec![Atom::bool(true)])];
    b.extend(unit_ret(1));
    assert!(wf_err(&module_of(vec![entry(b), setp], vec![])).contains("W6"));
}

/// W4 and globals: a fact about a global is a fact about the
/// value it held, and a call may change any global. The first three modules
/// below were accepted before this rule held, and running each applies
/// `int.add` outside its domain; each now requires change under W4. The same loop
/// without the call is accepted, since nothing in it changes the global,
/// and so is a local that read the global before a call, tested and used.
/// The last module is the same defect for memory: two reads of one address
/// with a `mem.set` between them were taken as one value.
#[test]
fn w4_facts_about_globals_end_at_a_call() {
    let g9 = Ty::Int(0, 9);
    let halt_at = |n: u32| vec![Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![lit(0), Atom::bytes(b"overflow")]), Stmt::Halt(Atom::V(n))];
    let set_to = |v: i128| Fn { params: vec![], ret: Ty::Bool, body: vec![Stmt::Set(Place::G(0), lit(v)), Stmt::Ret(Atom::bool(true))] };
    let add = |a: Atom| Stmt::Prim("int.add".into(), vec![g9.clone()], vec![a, lit(1)]);
    let test = Stmt::Prim("int.bad.add".into(), vec![g9.clone()], vec![Atom::G(0), lit(1)]);
    let kept = |m: &Module, why: &str| {
        let e = wf_err(m);
        assert!(e.contains("W4: int.add is not dominated by its test int.bad.add"), "{why}: {e}");
        let run = one::eval::run(m);
        assert!(run.is_err(), "{why}: the run should apply int.add outside its domain");
    };

    // a loop whose body calls a function that sets the global: the test
    // before the loop holds on the first iteration only
    let in_loop = |call: bool| {
        let mut l = vec![add(Atom::G(0))];
        if call {
            l.push(Stmt::Call(FnRef::Rec(1), vec![]));
        }
        vec![test.clone(), Stmt::If(Atom::V(0), halt_at(1), vec![]), Stmt::Loop(l)]
    };
    let m = module_of(vec![entry(in_loop(true)), set_to(9)], vec![(g9.clone(), Some(Lit::Int(0)))]);
    kept(&m, "a call in a loop");
    let m = module_of(vec![entry(in_loop(false)), set_to(9)], vec![(g9.clone(), Some(Lit::Int(0)))]);
    assert_eq!(one::wf::check(&m), Ok(()), "a loop that changes no global keeps the fact");

    // a block left by a break after a call: the fact before the block does
    // not hold after it
    let mut b = vec![
        test.clone(),
        Stmt::If(Atom::V(0), halt_at(1), vec![]),
        Stmt::Block(vec![Stmt::Call(FnRef::Rec(1), vec![]), Stmt::If(Atom::V(2), vec![Stmt::Break(0)], vec![])]),
        add(Atom::G(0)),
    ];
    b.extend(unit_ret(4));
    let m = module_of(vec![entry(b), set_to(9)], vec![(g9.clone(), Some(Lit::Int(0)))]);
    kept(&m, "a block with a call");

    // a local bound to a global's value before a call keeps the old value,
    // so a test of the global after the call says nothing about it
    let mut b = vec![
        Stmt::Let(Atom::G(0)),
        Stmt::Call(FnRef::Rec(1), vec![]),
        test.clone(),
        Stmt::If(Atom::V(2), halt_at(3), vec![]),
        add(Atom::V(0)),
    ];
    b.extend(unit_ret(5));
    let m = module_of(vec![entry(b), set_to(0)], vec![(g9.clone(), Some(Lit::Int(9)))]);
    kept(&m, "a local bound before a call");

    // a local that read the global before a call keeps the facts tested on
    // the value it holds
    let mut b = vec![
        Stmt::Let(Atom::G(0)),
        Stmt::Prim("int.bad.add".into(), vec![g9.clone()], vec![Atom::V(0), lit(1)]),
        Stmt::Call(FnRef::Rec(1), vec![]),
        Stmt::If(Atom::V(1), halt_at(3), vec![]),
        add(Atom::V(0)),
    ];
    b.extend(unit_ret(5));
    let m = module_of(vec![entry(b), set_to(9)], vec![(g9.clone(), Some(Lit::Int(0)))]);
    assert_eq!(wf_err(&m), "", "the value read before the call is the value tested");
    assert!(one::eval::run(&m).unwrap().halt.is_none());

    // memory is state outside every value: a fact about what `mem.get` read
    // says nothing about what it reads after a `mem.set`
    let get = |v: i128| Stmt::Prim("mem.get".into(), vec![g9.clone()], vec![lit(v)]);
    let mut b = vec![
        get(0),
        Stmt::Prim("int.bad.add".into(), vec![g9.clone()], vec![Atom::V(0), lit(1)]),
        Stmt::If(Atom::V(1), halt_at(2), vec![]),
        Stmt::Prim("mem.set".into(), vec![g9.clone()], vec![lit(0), lit(9)]),
        get(0),
        add(Atom::V(3)),
    ];
    b.extend(unit_ret(5));
    kept(&module_of(vec![entry(b)], vec![]), "a read of memory after a mem.set");
}

/// The evaluator: output bytes, a halt, and the primitives at the edges of
/// their types.
#[test]
fn evaluator() {
    let mut b = vec![Stmt::Out(Atom::bytes(b"hi\n"))];
    b.extend(unit_ret(0));
    let r = one::eval::run(&module_of(vec![entry(b)], vec![])).unwrap();
    assert_eq!(r.out, b"hi\n");
    assert!(r.halt.is_none());
    let b = vec![
        Stmt::Out(Atom::bytes(b"a")),
        Stmt::Prim("sum.mk".into(), vec![Ty::halt()], vec![lit(0), Atom::bytes(b"stop")]),
        Stmt::Halt(Atom::V(0)),
    ];
    let r = one::eval::run(&module_of(vec![entry(b)], vec![])).unwrap();
    assert_eq!(r.out, b"a");
    assert!(r.halt.is_some());

    use one::eval::{Val, prim};
    let i = |v: i128| Val::Int(v);
    let u8t = vec![Ty::u8()];
    let i8t = vec![Ty::Int(-128, 127)];
    let p = |n: &str, t: &[Ty], a: Vec<Val>| prim(n, t, a).unwrap();
    assert_eq!(p("int.wadd", &u8t, vec![i(250), i(10)]), vec![i(4)]);
    assert_eq!(p("int.wsub", &i8t, vec![i(-128), i(1)]), vec![i(127)]);
    assert_eq!(p("int.sadd", &u8t, vec![i(250), i(10)]), vec![i(255)]);
    assert_eq!(p("int.wmul", &[Ty::u64()], vec![i(u64::MAX as i128), i(u64::MAX as i128)]), vec![i(1)]);
    assert_eq!(p("int.quo", &i8t, vec![i(-7), i(2)]), vec![i(-3)]);
    assert_eq!(p("int.rem", &i8t, vec![i(-7), i(2)]), vec![i(-1)]);
    assert_eq!(p("int.bad.quo", &i8t, vec![i(-128), i(-1)]), vec![Val::Bool(true)]);
    assert_eq!(p("int.bad.add", &u8t, vec![i(200), i(56)]), vec![Val::Bool(true)]);
    assert_eq!(p("int.bad.add", &u8t, vec![i(200), i(55)]), vec![Val::Bool(false)]);
    assert_eq!(p("int.shl", &i8t, vec![i(1), i(7)]), vec![i(-128)]);
    assert_eq!(p("int.shl", &u8t, vec![i(-1), i(3)]), vec![i(248)]);
    assert_eq!(p("int.bad.shift", &u8t, vec![i(1), i(8)]), vec![Val::Bool(true)]);
    assert_eq!(p("int.not", &u8t, vec![i(5)]), vec![i(250)]);
    assert_eq!(p("int.not", &i8t, vec![i(5)]), vec![i(-6)]);
    assert!(prim("int.add", &u8t, vec![i(200), i(56)]).is_err(), "outside the domain");
    assert_eq!(p("f64.round", &[], vec![Val::F(-2.5)]), vec![Val::F(-3.0)]);
    assert_eq!(p("f64.bad.div", &[], vec![Val::F(1.0), Val::F(0.0)]), vec![Val::Bool(true)]);
    // f64.min and f64.max on signed zeros (6.3 N5): by IEEE order the two
    // are equal, and the first operand, the receiver, is returned on a tie;
    // compared by their bits, since -0.0 == 0.0
    let bits = |n: &str, x: f64, y: f64| match p(n, &[], vec![Val::F(x), Val::F(y)]).as_slice() {
        [Val::F(r)] => r.to_bits(),
        r => panic!("{n}: {r:?}"),
    };
    for n in ["f64.min", "f64.max"] {
        for (x, y) in [(0.0, -0.0), (-0.0, 0.0), (-0.0, -0.0), (0.0, 0.0)] {
            assert_eq!(bits(n, x, y), f64::to_bits(x), "{n}({x:?}, {y:?})");
        }
    }
    assert_eq!(bits("f64.min", 1.5, -0.0), (-0.0f64).to_bits());
    assert_eq!(bits("f64.max", -0.0, -1.5), (-0.0f64).to_bits());
}

/// 3.4: every type, atom and statement has one JSON form, read back exactly;
/// the text is canonical; a group's key is the SHA-256 of its JSON; text
/// that is not of the form requires change when it is read.
#[test]
fn json_form() {
    use sha2::{Digest, Sha256};
    let tys = vec![
        Ty::Bool, Ty::Int(-5, 300), Ty::F64, Ty::F32, Ty::Bytes, Ty::Arr(Box::new(Ty::Bool)), Ty::list(Ty::u8()),
        Ty::Tup(vec![Ty::Bytes, Ty::unit()]), Ty::Sum(vec![Ty::unit(), Ty::i64()]),
    ];
    let body = vec![
        Stmt::Let(Atom::K(Ty::F64, Lit::F64((-0.0f64).to_bits()))),
        Stmt::Let(Atom::K(Ty::F32, Lit::F32(1.5f64.to_bits()))),
        Stmt::Let(Atom::bytes(b"a\"\n")),
        Stmt::Var(Ty::Int(-5, 300), lit(-5)),
        Stmt::Set(Place::V(3), lit(300)),
        Stmt::If(Atom::bool(true), vec![Stmt::Block(vec![Stmt::Break(0)])], vec![Stmt::Loop(vec![Stmt::Break(0)])]),
        Stmt::Out(Atom::V(2)),
        Stmt::Prim("tup.mk".into(), vec![Ty::unit()], vec![]),
        Stmt::Ret(Atom::V(4)),
    ];
    let f = Fn { params: tys.clone(), ret: Ty::unit(), body };
    let g = Group { fns: vec![f] };
    let key = one::json::group_key(&g);
    let want: String = Sha256::digest(one::json::canon(&one::json::group_json(&g)).as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(key, want);
    let m = Module { globals: vec![(Ty::Bool, Some(Lit::Bool(false))), (Ty::list(Ty::Bytes), None)], defs: [(key.clone(), g)].into_iter().collect(), entry: (key, 0), names: serde_json::json!({"fns": {}}) };
    let t = one::json::module_text(&m);
    assert!(one::json::module_from_text(&t).unwrap() == m);
    assert!(t.contains(r#"["int","-5","300"]"#) && t.contains(r#"["k",["f64"],"8000000000000000"]"#) && t.contains(r#""61220a""#), "{t}");
    // forms that require change
    for (bad, why) in [
        (t.replace(r#""-5","300""#, r#""-05","300""#), "canonical decimal"),
        (t.replace(r#"["bool"]"#, r#"["boolean"]"#), "unknown type"),
        (t.replace(r#""tup.mk""#, r#""tup.make""#), "unknown primitive"),
        (t.replace("rce-core/1", "rce-core/2"), "format"),
        (t.replace(r#"["ret","#, r#"["return","#), "unknown statement"),
    ] {
        let e = one::json::module_from_text(&bad).err().unwrap_or_default();
        assert!(e.contains(why), "{why}: {e}");
    }
}

/// Every loop of the runtime written in the core (`rt`, `rtf`, `rtt`) has
/// a bounded shape: a measure, or a count bound computed before the loop
/// (`one::shape`). Every runtime function is built here, from the
/// `BUILDERS` table of each module, whether or not a program of the corpus
/// calls it; every public builder of the three modules must be in a table,
/// so a new builder is checked without a change here. The runtime has 102
/// loops: a change of that number is a change of the runtime to state here.
#[test]
fn every_runtime_loop_has_a_bounded_shape() {
    use rce_spec_check::one::build::{Fns, sym_id};
    use rce_spec_check::one::{rt, rtf, rtt, shape};
    let mut f = Fns::default();
    let mut missing = vec![];
    for (file, table) in [("src/one/rt.rs", rt::BUILDERS), ("src/one/rtf.rs", rtf::BUILDERS), ("src/one/rtt.rs", rtt::BUILDERS)] {
        for line in std::fs::read_to_string(file).unwrap().lines() {
            let Some(rest) = line.strip_prefix("pub fn ") else { continue };
            let (name, sig) = rest.split_once('(').unwrap();
            if sig.starts_with("f: &mut Fns") && sig.contains("-> R<usize>") && !table.iter().any(|(n, _)| n.split(' ').next() == Some(name)) {
                missing.push(format!("{file}: {name}"));
            }
        }
        for (name, build) in table {
            build(&mut f).unwrap_or_else(|e| panic!("{file}: {name}: {e}"));
        }
    }
    assert!(missing.is_empty(), "builders in no BUILDERS table:\n{}", missing.join("\n"));
    let globals: Vec<Ty> = f.globals.iter().map(|g| g.0.clone()).collect();
    let callee = |r: &FnRef| sym_id(r).map(|i| f.sigs[i].1.clone());
    let (mut loops, mut bad) = (0, vec![]);
    for (id, func) in f.fns.iter().enumerate() {
        let shapes = shape::loops(func.as_ref().unwrap(), &globals, &callee).unwrap();
        for (k, r) in shapes.into_iter().enumerate() {
            loops += 1;
            if let Err(e) = r {
                bad.push(format!("{} loop {k}: {e}", f.names[id]));
            }
        }
    }
    assert!(bad.is_empty(), "loops without a bounded shape:\n{}", bad.join("\n"));
    eprintln!("runtime functions: {}; loops: {loops}", f.fns.len());
    assert_eq!(loops, 102, "the runtime's loops");
}

/// The classifier of `one::shape` itself: a loop that doubles its variable
/// until it reaches an end has neither bounded shape; the same loop with a
/// count bound (`FB::count`) has the count shape, and a loop over a range
/// (`FB::for_n`) is a measure.
#[test]
fn loop_shapes() {
    use rce_spec_check::one::build::FB;
    use rce_spec_check::one::shape::{self, Shape};
    let t = Ty::Int(1, 1000);
    let doubling = |bounded: bool| {
        let mut b = FB::new(vec![], Ty::unit());
        let x = b.var(t.clone(), lit(1));
        let cnt = b.counter();
        b.loop_(|b| {
            let xv = b.take(x);
            let more = b.int_cmp("int.lt", xv.clone(), lit(100))?;
            b.exit_unless(more)?;
            if bounded {
                // x doubles from 1 and is below 100 here: at most 7 times
                b.count(cnt, lit(7))?;
            }
            let x2 = b.mul_in(xv, lit(2), t.clone())?;
            b.set(x, x2);
            Ok(())
        })
        .unwrap();
        let u = b.unit().unwrap();
        b.ret(u);
        b.finish()
    };
    let shapes = |f: &Fn| shape::loops(f, &[], &|_: &FnRef| None).unwrap();
    let r = shapes(&doubling(false));
    assert!(matches!(r.as_slice(), [Err(why)] if why.starts_with("neither a count nor a measure")), "{r:?}");
    assert_eq!(shapes(&doubling(true)), vec![Ok(Shape::Count)]);
    let mut b = FB::new(vec![], Ty::unit());
    b.for_n(lit(5), |_, _| Ok(())).unwrap();
    let u = b.unit().unwrap();
    b.ret(u);
    assert_eq!(shapes(&b.finish()), vec![Ok(Shape::Measure)]);
}

/// A ring of functions: member i writes its text and calls member
/// `spec[i].2`; they are declared in the order `order` and the entry calls
/// member 0. Every member reaches every other, so they form one group.
fn ring(spec: &[(&str, &str, usize)], order: &[usize]) -> Module {
    use rce_spec_check::one::build::{FB, Fns, sym};
    let mut fs = Fns::default();
    let entry = fs.declare("entry", vec![], Ty::unit());
    let mut ids = vec![0; spec.len()];
    for &k in order {
        ids[k] = fs.declare(spec[k].0, vec![], Ty::Bool);
    }
    for (k, (_, text, next)) in spec.iter().enumerate() {
        let mut b = FB::new(vec![], Ty::Bool);
        b.out(Atom::bytes(text.as_bytes()));
        let r = b.call(sym(ids[*next]), Ty::Bool, vec![]);
        b.ret(r);
        fs.define(ids[k], b.finish());
    }
    let mut b = FB::new(vec![], Ty::unit());
    b.call(sym(ids[0]), Ty::Bool, vec![]);
    let u = b.unit().unwrap();
    b.ret(u);
    fs.define(entry, b.finish());
    let m = fs.finish(entry).unwrap();
    assert_eq!(one::wf::check(&m), Ok(()));
    round_trip(&m);
    m
}

/// The members of a recursive group are in canonical order (W5): ordered
/// by an iterated digest, so the group's key, and every byte of the
/// module, do not depend on the order in which the functions were first
/// used. Before this order, members were in the order the lowering
/// declared them, so the two declaration orders of the first ring gave two
/// keys. Members that are the same function up to the members they call
/// are one function.
#[test]
fn recursive_groups_in_canonical_order() {
    let perms: [&[usize]; 6] = [&[0, 1, 2], &[0, 2, 1], &[1, 0, 2], &[1, 2, 0], &[2, 0, 1], &[2, 1, 0]];
    let group = |m: &Module| m.defs.values().find(|g| g.fns.len() > 1 || g.fns[0].ret == Ty::Bool).unwrap().clone();
    // two members that differ in their text
    let two = [("f", "f", 1), ("g", "g", 0)];
    let a = ring(&two, &[0, 1]);
    let b = ring(&two, &[1, 0]);
    assert_eq!(one::json::module_text(&a), one::json::module_text(&b));
    assert_eq!(group(&a).fns.len(), 2);
    // three members, two with the same text: the first round tells them
    // apart from the third only, and the second round by what they call
    let three = [("a", "x", 1), ("b", "x", 2), ("c", "y", 0)];
    let first = one::json::module_text(&ring(&three, perms[0]));
    for p in perms {
        let m = ring(&three, p);
        assert_eq!(one::json::module_text(&m), first, "declared in the order {p:?}");
        assert_eq!(group(&m).fns.len(), 3);
    }
    // three members of the same text that call around the ring are one
    // function that calls itself, named by all three
    let same = [("a", "x", 1), ("b", "x", 2), ("c", "x", 0)];
    for p in perms {
        let m = ring(&same, p);
        let g = group(&m);
        assert_eq!(g.fns.len(), 1, "declared in the order {p:?}");
        assert_eq!(g.fns[0].body[1], Stmt::Call(FnRef::Rec(0), vec![]));
        let key = one::json::group_key(&g);
        assert_eq!(m.names["fns"][&key][0], "a = b = c");
    }
}

/// W5 is checked on every core module read, not only on modules the
/// builder wrote: a group whose members are out of canonical order, or
/// that repeats a member, would give one group a second key, so wf reports
/// it (as `rce emit core FILE` and `rce run-core FILE` do on such input).
#[test]
fn wf_requires_change_for_a_group_out_of_canonical_order() {
    fn swap_rec(b: &mut [Stmt]) {
        for s in b {
            match s {
                Stmt::Call(FnRef::Rec(i), _) | Stmt::Catch(FnRef::Rec(i), _) => *i = 1 - *i,
                Stmt::If(_, t, e) => {
                    swap_rec(t);
                    swap_rec(e);
                }
                Stmt::Block(x) | Stmt::Loop(x) => swap_rec(x),
                _ => {}
            }
        }
    }
    let m = ring(&[("f", "f", 1), ("g", "g", 0)], &[0, 1]);
    let g = m.defs.values().find(|g| g.fns.len() == 2).unwrap().clone();
    // the same two members in the other order, beside the canonical group
    let mut swapped = Group { fns: vec![g.fns[1].clone(), g.fns[0].clone()] };
    for f in &mut swapped.fns {
        swap_rec(&mut f.body);
    }
    let mut bad = m.clone();
    bad.defs.insert(one::json::group_key(&swapped), swapped);
    let e = one::wf::check(&bad).unwrap_err();
    assert!(e.starts_with("W5: member 0 of group") && e.contains("canonical order"), "{e}");
    // a member repeated: one function that calls itself, written twice
    let one_fn = ring(&[("a", "x", 1), ("b", "x", 2), ("c", "x", 0)], &[0, 1, 2]);
    let f = one_fn.defs.values().find(|g| g.fns[0].ret == Ty::Bool).unwrap().fns[0].clone();
    let twice = Group { fns: vec![f.clone(), f] };
    let mut bad = one_fn.clone();
    bad.defs.insert(one::json::group_key(&twice), twice);
    let e = one::wf::check(&bad).unwrap_err();
    assert!(e.starts_with("W5: member 1 of group") && e.contains("repeats a member"), "{e}");
}
