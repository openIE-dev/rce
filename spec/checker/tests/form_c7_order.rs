//! C7 on a count-bounded Iterate's carried order.
//!
//! Erratum of 2026-11-13: a count keeps its carried order even when both
//! regions only pass it through. The old rule removed that order. The
//! module it produced does not comply with FG7, because a count may halt
//! and then does not consume and return order. An unused carried value that
//! is not that order is still removed. A measure's unused order is still
//! removed.
use rce_spec_check::form::canon::canonical;
use rce_spec_check::form::ir::{Attr, BTy, Body, Bound, Func, Group, Lit, Module, Node, Src};
use rce_spec_check::form::{self, Outcome};
use std::collections::BTreeMap;

fn lit(t: BTy, v: Lit) -> Node {
    Node { attrs: Attr::Lit(t, v), ins: vec![], regions: vec![] }
}

/// A region that returns true and then each carried input, unchanged.
fn pass(n: u32) -> Body {
    Body {
        nodes: vec![lit(BTy::Bool, Lit::Bool(true))],
        out: std::iter::once(Src::Out(0, 0)).chain((0..n).map(Src::In)).collect(),
    }
}

fn module_of(f: Func) -> Module {
    let g = Group { fns: vec![f] };
    let key = rce_spec_check::form::ir::group_key(&g);
    Module {
        catalogue: rce_spec_check::form::catalogue::key().to_string(),
        types: vec![],
        regions: vec![],
        events: vec![],
        errors: vec![],
        steps: vec![],
        main: None,
        caps: vec![],
        defs: BTreeMap::from([(key, g)]),
    }
}

fn counts(m: &Module) -> Vec<Vec<BTy>> {
    let mut out = vec![];
    fn walk(b: &Body, out: &mut Vec<Vec<BTy>>) {
        for n in &b.nodes {
            match &n.attrs {
                Attr::Count(ts) => out.push(ts.clone()),
                Attr::Measure(_, _, ts) => out.push(ts.clone()),
                _ => {}
            }
            for r in &n.regions {
                walk(r, out);
            }
        }
    }
    for g in m.defs.values() {
        for f in &g.fns {
            walk(&f.body, &mut out);
        }
    }
    out
}

fn i100() -> BTy {
    BTy::Int(0, 100)
}

/// Count carrying order and an unused integer. Both regions pass both through.
fn count_with_unused_int() -> Module {
    let carried = vec![BTy::Order, i100()];
    let count = Node {
        attrs: Attr::Count(carried),
        ins: vec![Src::Out(0, 0), Src::In(1), Src::In(0)],
        regions: vec![pass(2), pass(2)],
    };
    let f = Func {
        ins: vec![i100(), BTy::Order],
        outs: vec![i100(), BTy::Order],
        eff: 1,
        bound: Bound::None,
        body: Body {
            nodes: vec![lit(BTy::u64(), Lit::Int(2)), count],
            out: vec![Src::Out(1, 1), Src::Out(1, 0)],
        },
    };
    module_of(f)
}

#[test]
fn a_count_keeps_unused_carried_order_and_drops_another_unused_value() {
    let m = count_with_unused_int();
    let (c, first) = canonical(&m).unwrap_or_else(|e| panic!("{}", e.0));
    assert_eq!(first, Some("C7"), "the unused integer was not removed");
    assert_eq!(
        counts(&c),
        vec![vec![BTy::Order]],
        "the old rule removed the carried order as well"
    );
    let (again, second) = canonical(&c).unwrap_or_else(|e| panic!("{}", e.0));
    assert_eq!(second, None, "the kept order is not a fixed point");
    assert_eq!(counts(&again), vec![vec![BTy::Order]]);
    let bytes = rce_spec_check::form::ir::encode(&c);
    match form::check(bytes.as_bytes(), None) {
        Outcome::Applied(_) => {}
        Outcome::RequiresChange(k) => panic!("the count that keeps order is not applied: {}", k.line()),
    }
}

#[test]
fn the_old_count_without_carried_order_requires_change() {
    // The bytes the old C7 produced for a count whose only job for order
    // was to carry it: the order is gone, the integer is still carried
    // because the body changes it, and the count may still halt.
    let wide = BTy::Int(0, 101);
    let one = lit(BTy::Int(1, 1), Lit::Int(1));
    let add = Node {
        attrs: Attr::Prim("int.add".into(), vec![rce_spec_check::form::ir::TyArg::Body(wide.clone())], None),
        ins: vec![Src::In(0), Src::Out(0, 0)],
        regions: vec![],
    };
    let body = Body {
        nodes: vec![one, add, lit(BTy::Bool, Lit::Bool(true))],
        out: vec![Src::Out(2, 0), Src::Out(1, 0)],
    };
    let count = Node {
        attrs: Attr::Count(vec![wide.clone()]),
        ins: vec![Src::Out(0, 0), Src::In(0)],
        regions: vec![pass(1), body],
    };
    let f = Func {
        ins: vec![i100(), BTy::Order],
        outs: vec![wide, BTy::Order],
        eff: 1,
        bound: Bound::None,
        body: Body {
            nodes: vec![lit(BTy::u64(), Lit::Int(2)), count],
            out: vec![Src::Out(1, 0), Src::In(1)],
        },
    };
    let m = module_of(f);
    let (c, _) = canonical(&m).unwrap_or_else(|e| panic!("{}", e.0));
    assert!(
        counts(&c).iter().any(|ts| !ts.is_empty() && !ts.contains(&BTy::Order)),
        "the stripped count did not stay a count without order: {:?}",
        counts(&c)
    );
    let bytes = rce_spec_check::form::ir::encode(&c);
    match form::check(bytes.as_bytes(), None) {
        Outcome::RequiresChange(k) => {
            assert_eq!(k.rule, "FG7", "{}", k.line());
            assert!(
                k.reason.contains("does not consume and return"),
                "{}",
                k.line()
            );
        }
        Outcome::Applied(_) => panic!("a count that lost its carried order was applied"),
    }
}

#[test]
fn a_measure_still_drops_unused_carried_order() {
    let span = i100();
    let carried = vec![BTy::Order, span.clone(), span.clone()];
    let measure = Node {
        attrs: Attr::Measure(1, 2, carried),
        ins: vec![Src::In(2), Src::In(0), Src::In(1)],
        regions: vec![pass(3), pass(3)],
    };
    let f = Func {
        ins: vec![span.clone(), span.clone(), BTy::Order],
        outs: vec![span.clone(), span, BTy::Order],
        eff: 1,
        bound: Bound::None,
        body: Body {
            nodes: vec![measure],
            out: vec![Src::Out(0, 1), Src::Out(0, 2), Src::Out(0, 0)],
        },
    };
    let (c, first) = canonical(&module_of(f)).unwrap_or_else(|e| panic!("{}", e.0));
    assert_eq!(first, Some("C7"));
    let ts = counts(&c);
    assert_eq!(ts.len(), 1, "{ts:?}");
    assert!(
        !ts[0].contains(&BTy::Order),
        "a measure kept an unused carried order: {ts:?}"
    );
}
