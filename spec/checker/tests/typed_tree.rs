//! The typed tree (spec/proposals/one-emitter.md ONE-4, order of work
//! step 1): after the checker accepts a module, every expression node the
//! reference evaluates has a recorded type, every name its binding and
//! every call its resolution.

fn module(src: &str) -> rce_spec_check::ast::Module {
    assert!(rce_spec_check::cli::diagnostics(src, &Default::default()).is_empty());
    rce_spec_check::program(src).unwrap()
}

/// A type for every expression node, the binding of every name and the
/// resolution of every call; an integer literal is untyped until a declared
/// type is given to it, as in the reference.
#[test]
fn types_bindings_and_calls() {
    use rce_spec_check::typed::{self, Binding, Callee, RType};
    let src = "state S\n    n: u8 = 0\n\nevent Inc\n\nfn twice(x: u8) -> u8 = x + x\n\nstep S on Inc\n    let k = 1\n    n = twice(n + k)\n\nexample Inc -> 2\n";
    let m = module(src);
    let t = typed::build(&m);
    let mut ints = 0;
    let mut calls = 0;
    let mut names = 0;
    fn walk(e: &rce_spec_check::ast::Expr, f: &mut dyn FnMut(&rce_spec_check::ast::Expr)) {
        f(e);
        rce_spec_check::check::walk_expr(e, &mut |x| if !std::ptr::eq(x, e) { f(x) });
    }
    for it in &m.items {
        let exprs: Vec<&rce_spec_check::ast::Expr> = match it {
            rce_spec_check::ast::Item::Fn(f) => match &f.body {
                rce_spec_check::ast::Body::Arrow(e) => vec![e],
                _ => vec![],
            },
            rce_spec_check::ast::Item::Step { body: rce_spec_check::ast::Body::Block(b), .. } => b
                .iter()
                .filter_map(|s| match s {
                    rce_spec_check::ast::Stmt::Let(_, _, e, _) => Some(e),
                    rce_spec_check::ast::Stmt::Assign(_, _, e) => Some(e),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        };
        for e in exprs {
            walk(e, &mut |x| {
                let n = t.node(x).unwrap_or_else(|| panic!("no node for {x:?}"));
                if let RType::Int(i) = &n.ty {
                    ints += 1;
                    assert!(!i.tags.names().is_empty());
                }
                if matches!(n.bind, Some(Binding::Local(_))) {
                    names += 1;
                }
                if matches!(n.callee, Some(Callee::Fn(ref f)) if f == "twice") {
                    calls += 1;
                    // the call's result has its declared type
                    assert_eq!(n.ty.show(), "int{u8}[0, 255]");
                }
                if let rce_spec_check::ast::Expr::Int(1, _) = x {
                    assert_eq!(n.ty.show(), "int{lit}[1, 1]");
                }
                if let rce_spec_check::ast::Expr::Bin(op, ..) = x
                    && op == "+"
                {
                    // u8 + lit is checked against u8 (the left operand's name)
                    assert!(n.ty.show().starts_with("int{u8}"), "{}", n.ty.show());
                }
            });
        }
    }
    assert!(ints >= 5 && calls == 1 && names >= 4, "{ints} {calls} {names}");
}

/// A variable's type follows the program: after `x = y` with `y: u8`, a
/// variable that held an untyped literal holds a u8, and around a loop the
/// type is the one at which the loop's types settle.
#[test]
fn types_follow_assignments_and_loops() {
    use rce_spec_check::typed::{self, RType};
    let src = "fn f(y: u8) -> u8\n    var x = 0\n    x = y\n    let z = x + 1\n    return z\n\nfn g(n: u32) -> u64\n    var a: u64 = 0\n    for _i in 0..n\n        a = a + 1\n    return a\n";
    let m = module(src);
    let t = typed::build(&m);
    let mut zs = vec![];
    for n in t.nodes.values() {
        if let RType::Int(i) = &n.ty {
            zs.push(i.tags.names().join(","));
        }
    }
    // the sum `x + 1` after `x = y` is checked against u8
    let rce_spec_check::ast::Item::Fn(f) = &m.items[0] else { panic!() };
    let rce_spec_check::ast::Body::Block(b) = &f.body else { panic!() };
    let rce_spec_check::ast::Stmt::Let(_, _, e, _) = &b[2] else { panic!() };
    assert_eq!(t.ty(e).show(), "int{u8}[0, 255]");
    // the variable itself holds an untyped literal or a u8 over its scope
    let x = t.locals.iter().find(|l| l.name == "x").unwrap();
    assert_eq!(x.ty.show(), "int{lit,u8}[0, 255]");
    // in g the loop settles with `a` a u64
    let a = t.locals.iter().find(|l| l.name == "a").unwrap();
    assert_eq!(a.ty.show(), "int{u64}[0, 18446744073709551615]");
    let i = t.locals.iter().find(|l| l.name == "_i").unwrap();
    assert!(i.ty.show().starts_with("int{u32}"), "{}", i.ty.show());
    assert!(!zs.is_empty());
}
