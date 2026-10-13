//! Static energy model over the Form (lesson from SPEAR-Viewer): count the
//! constructs in each function and price them with a per-class cost table.
//! The result feeds E_sim before anything runs. Loops and recursion have
//! unknown trip counts at compile time, so a loop body is multiplied by
//! `Table::loop_trips` (an assumption, reported as such), and calls to
//! other functions add the callee's own static cost once (no recursion).
//! Default nanojoule prices are placeholders until `telemetry calibrate`
//! fits them on a real meter; any number from the defaults is synthetic.
use rce_spec_check::ast::{Arg, Body, Expr, Item, Module, SelArm, Stmt};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Op { Arith, Compare, Branch, Call, Alloc, Load, Chan, Spawn, Device }

#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    pub nj: BTreeMap<Op, f64>,
    pub loop_trips: f64,
    /// True while the prices are defaults rather than a calibration fit.
    pub synthetic: bool,
}

impl Default for Table {
    fn default() -> Self {
        use Op::*;
        let nj = [(Arith, 0.5), (Compare, 0.5), (Branch, 1.0), (Call, 5.0), (Alloc, 50.0),
                  (Load, 2.0), (Chan, 200.0), (Spawn, 5000.0), (Device, 20000.0)]
            .into_iter().collect();
        Table { nj, loop_trips: 16.0, synthetic: true }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FnCost {
    pub ops: BTreeMap<Op, f64>,
    pub calls: Vec<String>,
    pub line: usize,
}

impl FnCost {
    pub fn local_nj(&self, t: &Table) -> f64 {
        self.ops.iter().map(|(o, n)| n * t.nj.get(o).copied().unwrap_or(0.0)).sum()
    }
}

struct W<'a> { c: &'a mut FnCost, mul: f64, trips: f64 }

impl W<'_> {
    fn op(&mut self, o: Op) { *self.c.ops.entry(o).or_default() += self.mul; }
    fn args(&mut self, a: &[Arg]) { for x in a { self.e(&x.e) } }
    fn e(&mut self, e: &Expr) {
        use Expr::*;
        match e {
            Name(..) | SelfRef(_) | Int(..) | Float(..) | Str(..) | Bool(..) | None(_) | Unit(_) => {}
            Tuple(v, _) => v.iter().for_each(|x| self.e(x)),
            List(v, _) => { self.op(Op::Alloc); v.iter().for_each(|x| self.e(x)) }
            ListRepeat(a, b, _) => { self.op(Op::Alloc); self.e(a); self.e(b) }
            ListComp(a, _, b, g, _) => {
                self.op(Op::Alloc); self.e(b);
                let m = self.mul; self.mul *= self.trips; self.e(a);
                if let Some(g) = g { self.op(Op::Branch); self.e(g) }
                self.mul = m;
            }
            Map(v, _) => { self.op(Op::Alloc); for (a, b) in v { self.e(a); self.e(b) } }
            ChanNew(_, x, _) => { self.op(Op::Chan); if let Some(x) = x { self.e(x) } }
            Call(f, a, _) => {
                self.op(Op::Call);
                if let Name(n, _) = &**f { self.c.calls.push(n.clone()) } else { self.e(f) }
                self.args(a)
            }
            Index(a, v, _) => { self.op(Op::Load); self.e(a); v.iter().for_each(|x| self.e(x)) }
            Slice(a, b, c, _) => { self.op(Op::Load); self.e(a); for x in [b, c].into_iter().flatten() { self.e(x) } }
            Member(a, ..) | TupleMember(a, ..) => { self.op(Op::Load); self.e(a) }
            Neg(a, _) | BitNot(a, _) | Not(a, _) => { self.op(Op::Arith); self.e(a) }
            Borrow(_, a, _) | Comptime(a, _) | Cast(a, _, _) => self.e(a),
            Recv(a, _) => { self.op(Op::Chan); self.e(a) }
            Bin(o, a, b, _) => {
                self.op(if ["==", "!=", "<", ">", "<=", ">="].contains(&o.as_str()) { Op::Compare } else { Op::Arith });
                self.e(a); self.e(b)
            }
            Catch(a, _, b, _) => { self.op(Op::Branch); self.e(a); self.e(b) }
            Try(a, b, _) => { self.op(Op::Branch); self.e(a); if let Some(b) = b { self.e(b) } }
            If(c, a, b, _) => { self.op(Op::Branch); self.e(c); self.e(a); self.e(b) }
            Lambda(_, b, _) => self.e(b),
        }
    }
    fn loop_body(&mut self, b: &[Stmt]) {
        let m = self.mul; self.mul *= self.trips; self.op(Op::Branch); self.ss(b); self.mul = m;
    }
    fn ss(&mut self, v: &[Stmt]) { v.iter().for_each(|s| self.s(s)) }
    fn s(&mut self, s: &Stmt) {
        use Stmt::*;
        match s {
            Expr(e) | Emit(e) | Fail(e) | Check(e) | Yield(e, _) => self.e(e),
            Assign(_, a, b) => { self.op(Op::Load); self.e(a); self.e(b) }
            Send(a, b) => { self.op(Op::Chan); self.e(a); self.e(b) }
            Let(_, _, e, _) | Var(_, _, e, _) => self.e(e),
            If(arms, els) => {
                for (c, b) in arms { self.op(Op::Branch); self.e(c); self.ss(b) }
                if let Some(b) = els { self.ss(b) }
            }
            While(c, inv, b) => { self.e(c); if let Some(i) = inv { self.e(i) } self.loop_body(b) }
            Loop(_, b) => self.loop_body(b),
            For(_, it, b, _) => { self.e(it); self.loop_body(b) }
            Match(e, arms, _) => {
                self.e(e);
                for a in arms { self.op(Op::Branch); if let Some(g) = &a.guard { self.e(g) } self.ss(&a.body) }
            }
            Return(e, _) => { if let Some(e) = e { self.e(e) } }
            Break(_) | Continue(_) | Pass => {}
            Spawn(e) => { self.op(Op::Spawn); self.e(e) }
            Select(arms, _) => {
                self.op(Op::Chan);
                for a in arms {
                    match a {
                        SelArm::Recv(_, e, b) | SelArm::After(e, b) => { self.e(e); self.ss(b) }
                        SelArm::Send(x, y, b) => { self.e(x); self.e(y); self.ss(b) }
                        SelArm::Default(b) => self.ss(b),
                    }
                }
            }
            Arena(_, e, b, _) => { self.op(Op::Alloc); if let Some(e) = e { self.e(e) } self.ss(b) }
            Device(_, e, b, _) => { self.op(Op::Device); if let Some(e) = e { self.e(e) } self.ss(b) }
            Comptime(_) => {}
        }
    }
    fn body(&mut self, b: &Body) { match b { Body::Arrow(e) => self.e(e), Body::Block(v) => self.ss(v) } }
}

/// Per-function static op counts for one module, keyed by function name
/// (`Type.method` for impl members, `Actor.handler#i` for handlers).
pub fn analyze(m: &Module, t: &Table) -> BTreeMap<String, FnCost> {
    let mut out = BTreeMap::new();
    let mut one = |name: String, line: usize, b: &Body| {
        let mut c = FnCost { line, ..Default::default() };
        W { c: &mut c, mul: 1.0, trips: t.loop_trips }.body(b);
        out.insert(name, c);
    };
    for it in &m.items {
        match it {
            Item::Fn(f) => one(f.sig.name.clone(), f.sig.line, &f.body),
            Item::Impl { members, .. } => for f in members { one(f.sig.name.clone(), f.sig.line, &f.body) },
            Item::Step { region, body, line, .. } => one(format!("step:{region}:{line}"), *line, body),
            Item::Actor { name, handlers, fns, .. } => {
                for (i, h) in handlers.iter().enumerate() { one(format!("{name}.handler#{i}"), h.line, &h.body) }
                for f in fns { one(format!("{name}.{}", f.sig.name), f.sig.line, &f.body) }
            }
            _ => {}
        }
    }
    out
}

/// E_sim in joules per function: local cost plus each distinct callee's
/// total once, without following cycles.
pub fn e_sim(costs: &BTreeMap<String, FnCost>, t: &Table) -> BTreeMap<String, f64> {
    fn tot(n: &str, c: &BTreeMap<String, FnCost>, t: &Table, seen: &mut Vec<String>) -> f64 {
        let Some(f) = c.get(n) else { return 0.0 };
        if seen.iter().any(|s| s == n) { return 0.0 }
        seen.push(n.to_string());
        let mut j = f.local_nj(t);
        let mut callees = f.calls.clone(); callees.sort(); callees.dedup();
        for k in callees { j += tot(&k, c, t, seen) }
        seen.pop();
        j
    }
    costs.keys().map(|k| (k.clone(), tot(k, costs, t, &mut Vec::new()) * 1e-9)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_spec_check::parse::parse;

    #[test]
    fn loops_cost_more_and_calls_roll_up() {
        let src = "fn leaf(x: i64) -> i64\n    return x + 1\n\nfn hot(n: i64) -> i64\n    var s: i64 = 0\n    for i in 0..n\n        s = s + leaf(i)\n    return s\n";
        let m = parse(src).expect("parse");
        let t = Table::default();
        let c = analyze(&m, &t);
        assert_eq!(c["leaf"].ops[&Op::Arith], 1.0);
        let e = e_sim(&c, &t);
        assert!(e["hot"] > e["leaf"] * 10.0, "{e:?}");
        assert!(c["hot"].calls.contains(&"leaf".to_string()));
    }

    #[test]
    fn recursion_terminates() {
        let src = "fn f(n: i64) -> i64\n    return f(n - 1)\n";
        let m = parse(src).expect("parse");
        let t = Table::default();
        let e = e_sim(&analyze(&m, &t), &t);
        assert!(e["f"] > 0.0);
    }
}
