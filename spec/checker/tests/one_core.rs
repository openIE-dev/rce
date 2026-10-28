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
/// `--trace --fuel-trace`, byte for byte.
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
        assert_eq!(String::from_utf8_lossy(&o.stdout), one::trace::reference(&m), "{stem}");
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

fn module_of(fns: Vec<Fn>, globals: Vec<(Ty, Option<Lit>)>) -> Module {
    let g = Group { fns };
    let key = one::json::group_key(&g);
    Module { globals, defs: [(key.clone(), g)].into_iter().collect(), entry: (key, 0), names: serde_json::json!({}) }
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
/// that is not of the form is refused when it is read.
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
    // refused forms
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
