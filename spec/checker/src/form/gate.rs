//! The gate on `rce-form/1` (rce-form.md §11): FG1 to FG15, in order. The
//! module is kept at the first rule that fails, with the rule, the group
//! key, the function index and the node path; a module that passes every
//! rule is applied: it complies with the Form.
use super::canon::{self, Defer};
use super::catalogue::{self, Entry};
use super::ir::*;
use super::types::{Tx, fshow, lies_within, show};
use super::walk::{Cx, Visit, walk};
use super::{Kept, decode, eval};
use serde_json::Value as J;
use std::collections::{BTreeMap, BTreeSet};

type G = Result<(), Kept>;
type W = Result<(), String>;
/// A failed check of one function: the node path and the reason.
type At = Result<(), (String, String)>;
/// A call to a member of the caller's group: caller, callee, and for each
/// argument the parameter it is and whether it is a component of it.
type RecCall = (u32, u32, Vec<Option<(u32, bool)>>);

fn kept(rule: &'static str, group: Option<&str>, func: Option<u32>, path: impl Into<String>, reason: impl Into<String>) -> Kept {
    Kept { rule, group: group.map(str::to_string), func, path: path.into(), reason: reason.into() }
}

fn frame(rule: &'static str, path: impl Into<String>, reason: impl Into<String>) -> Kept {
    kept(rule, None, None, path, reason)
}

/// Run a check on every function, groups in key order, functions in order.
fn per_fn(m: &Module, rule: &'static str, check: &dyn Fn(&Module, &str, u32, &Func) -> At) -> G {
    for (k, g) in &m.defs {
        for (i, f) in g.fns.iter().enumerate() {
            if let Err((p, r)) = check(m, k, i as u32, f) {
                return Err(kept(rule, Some(k), Some(i as u32), p, r));
            }
        }
    }
    Ok(())
}

/// Walk every function with a visitor that `mk` makes for it.
fn per_walk<'a, V: Visit<'a>>(m: &'a Module, rule: &'static str, mk: &dyn Fn() -> V) -> G {
    for (k, g) in &m.defs {
        for (i, f) in g.fns.iter().enumerate() {
            let mut v = mk();
            if let Err((p, r)) = walk(m, k, i as u32, f, &mut v) {
                return Err(kept(rule, Some(k), Some(i as u32), p, r));
            }
        }
    }
    Ok(())
}

// ---- FG2 Keys

fn refs_in(b: &Body, out: &mut Vec<(String, Ref)>, path: &str) {
    for (i, n) in b.nodes.iter().enumerate() {
        let p = if path.is_empty() { format!("node {i}") } else { format!("{path} node {i}") };
        if let Some(r) = n.attrs.refs() {
            out.push((p.clone(), r.clone()));
        }
        for (k, r) in n.regions.iter().enumerate() {
            refs_in(r, out, &format!("{p} region {k}"));
        }
    }
}

fn fg2(m: &Module, raw: &J) -> G {
    for (k, g) in raw["defs"].as_object().into_iter().flatten() {
        let h = sha(g);
        if &h != k {
            return Err(kept("FG2", Some(k), None, "", format!("the group key is not the SHA-256 of the group's bytes, which is {h}")));
        }
    }
    let resolves = |r: &Ref, group: Option<&str>| -> Option<String> {
        match (r, group) {
            (Ref::Key(k, i), _) => match m.defs.get(k) {
                None => Some(format!("[{k}, {i}] names no group of defs")),
                Some(g) if *i as usize >= g.fns.len() => Some(format!("group {k} has no function {i}")),
                _ => None,
            },
            (Ref::Rec(i), Some(g)) => (*i as usize >= m.defs[g].fns.len()).then(|| format!("[\"rec\", {i}] names no function of its group")),
            (Ref::Rec(i), None) => Some(format!("[\"rec\", {i}] is written outside a group")),
        }
    };
    for (k, g) in &m.defs {
        for (fi, f) in g.fns.iter().enumerate() {
            let mut rs = vec![];
            refs_in(&f.body, &mut rs, "");
            for (p, r) in rs {
                if let Some(e) = resolves(&r, Some(k)) {
                    return Err(kept("FG2", Some(k), Some(fi as u32), p, e));
                }
            }
        }
    }
    for (i, s) in m.steps.iter().enumerate() {
        for r in s.guard.iter().chain([&s.body]) {
            if let Some(e) = resolves(r, None) {
                return Err(frame("FG2", format!("steps[{i}]"), e));
            }
        }
    }
    if let Some(r) = &m.main
        && let Some(e) = resolves(r, None)
    {
        return Err(frame("FG2", "main", e));
    }
    for (i, t) in m.types.iter().enumerate() {
        if let TypeDecl::Refined(_, r) = t
            && let Some(e) = resolves(r, None)
        {
            return Err(frame("FG2", format!("types[{i}]"), e));
        }
    }
    // the graph of keys is acyclic
    let mut state: BTreeMap<&str, u8> = BTreeMap::new();
    fn visit<'a>(m: &'a Module, k: &'a str, st: &mut BTreeMap<&'a str, u8>) -> Option<String> {
        match st.get(k) {
            Some(2) => return None,
            Some(_) => return Some(k.to_string()),
            None => {}
        }
        st.insert(k, 1);
        for c in canon::group_refs(&m.defs[k]) {
            if let Some((ck, _)) = m.defs.get_key_value(&c)
                && let Some(x) = visit(m, ck, st)
            {
                return Some(x);
            }
        }
        st.insert(k, 2);
        None
    }
    for k in m.defs.keys() {
        if let Some(c) = visit(m, k, &mut state) {
            return Err(kept("FG2", Some(&c), None, "", "the graph of keys has a cycle through this group"));
        }
    }
    Ok(())
}

// ---- FG3 Canonical

fn fg3(bytes: &[u8], m: &Module) -> G {
    let (c, first) = match canon::canonical(m) {
        Ok(x) => x,
        // no canonical form: a source names no node, or sources form a cycle; FG4 keeps it
        Err(Defer(_)) => return Ok(()),
    };
    if encode(&c).as_bytes() == bytes {
        return Ok(());
    }
    let group = m.defs.keys().find(|k| !c.defs.contains_key(*k)).map(String::as_str);
    Err(match first {
        Some(rule) => kept("FG3", group, None, "", format!("{rule}: decoding and encoding again does not give the same bytes; the canonical form differs by {rule}")),
        None => kept("FG3", None, None, "", "6.1.1: the bytes are not the canonical JSON of the module"),
    })
}

// ---- FG4 Ports

struct Fg4;

impl<'a> Visit<'a> for Fg4 {
    fn region(&mut self, cx: &Cx<'a>) -> W {
        let fr = cx.top();
        let nin = fr.ins.len();
        let nodes = &fr.body.nodes;
        let ok = |s: &Src, before: usize| -> W {
            match s {
                Src::In(j) if (*j as usize) < nin => Ok(()),
                Src::In(j) => Err(format!("input {j} of the region does not exist")),
                Src::Out(i, p) => {
                    let i = *i as usize;
                    if i >= nodes.len() {
                        return Err(format!("node {i} does not exist"));
                    }
                    if i >= before {
                        return Err(format!("node {i} does not precede its consumer"));
                    }
                    if let Some(n) = nout(cx, i)
                        && *p as usize >= n
                    {
                        return Err(format!("node {i} has no output {p}"));
                    }
                    Ok(())
                }
            }
        };
        for (i, n) in nodes.iter().enumerate() {
            for s in &n.ins {
                ok(s, i).map_err(|e| format!("node {i}: {e}"))?;
            }
        }
        for s in &fr.body.out {
            ok(s, nodes.len()).map_err(|e| format!("output: {e}"))?;
        }
        Ok(())
    }

    fn node(&mut self, cx: &Cx<'a>, i: usize) -> W {
        let n = &cx.top().body.nodes[i];
        let want_ins = |k: usize| if n.ins.len() == k { Ok(()) } else { Err(format!("takes {k} inputs, and has {}", n.ins.len())) };
        let want_regions = |k: usize| if n.regions.len() == k { Ok(()) } else { Err(format!("takes {k} regions, and has {}", n.regions.len())) };
        let region_outs = |k: usize| -> W {
            for (q, r) in n.regions.iter().enumerate() {
                if r.out.len() != k {
                    return Err(format!("region {q} has {} outputs, and must have {k}", r.out.len()));
                }
            }
            Ok(())
        };
        match &n.attrs {
            Attr::Lit(..) => {
                want_ins(0)?;
                want_regions(0)
            }
            Attr::Prim(name, args, _) => {
                let e = catalogue::entry(name).ok_or("an unknown entry")?;
                if args.len() != e.params.len() {
                    return Err(format!("`{name}` takes {} type arguments, and has {}", e.params.len(), args.len()));
                }
                if let Ok(ports) = cx.tx.operand_ports(e, args) {
                    want_ins(ports.len())?;
                }
                want_regions(0)
            }
            Attr::Call(r) | Attr::Catch(r) => {
                let f = cx.tx.callee(cx.group, r).ok_or("an unresolved reference")?;
                want_ins(f.ins.len())?;
                want_regions(0)
            }
            Attr::Cap(_) => want_regions(0),
            Attr::Branch(k, ts) => {
                if n.ins.is_empty() {
                    return Err("a Branch takes a selector".into());
                }
                want_regions(*k as usize)?;
                region_outs(ts.len())
            }
            Attr::Count(ts) => {
                want_ins(1 + ts.len())?;
                want_regions(2)?;
                region_outs(1 + ts.len())
            }
            Attr::Measure(x, y, ts) => {
                want_ins(ts.len())?;
                if *x as usize >= ts.len() || *y as usize >= ts.len() || x == y {
                    return Err(format!("measure positions [{x}, {y}] are not two carried values"));
                }
                want_regions(2)?;
                region_outs(1 + ts.len())
            }
            Attr::Halt(_) => {
                want_ins(2)?;
                want_regions(0)
            }
        }
    }
}

fn nout(cx: &Cx, i: usize) -> Option<usize> {
    cx.top().outs.get(i)?.as_ref().map(|v| v.len())
}

fn fg4(m: &Module) -> G {
    per_fn(m, "FG4", &|_, _, _, f| {
        if f.body.out.len() != f.outs.len() {
            return Err(("body".into(), format!("the body has {} outputs, and the signature {}", f.body.out.len(), f.outs.len())));
        }
        Ok(())
    })?;
    per_walk(m, "FG4", &|| Fg4)
}

// ---- FG5 Types

struct Fg5;

impl<'a> Visit<'a> for Fg5 {
    fn region(&mut self, cx: &Cx<'a>) -> W {
        let fr = cx.top();
        let want: Vec<BTy> = match fr.owner.map(|n| &n.attrs) {
            None => cx.f.outs.clone(),
            Some(Attr::Branch(_, ts)) => ts.clone(),
            Some(Attr::Count(ts) | Attr::Measure(_, _, ts)) => std::iter::once(BTy::Bool).chain(ts.iter().cloned()).collect(),
            _ => return Ok(()),
        };
        for (q, (s, w)) in fr.body.out.iter().zip(&want).enumerate() {
            if let Some(t) = cx.ty(s)
                && !lies_within(&t, w)
            {
                return Err(format!("output {q}: {} does not lie within {}", show(&t), show(w)));
            }
        }
        Ok(())
    }

    fn node(&mut self, cx: &Cx<'a>, i: usize) -> W {
        let d = cx.depth();
        let n = &cx.top().body.nodes[i];
        let need = |k: usize, w: &BTy| -> W {
            match n.ins.get(k).and_then(|s| cx.ty(s)) {
                Some(t) if !lies_within(&t, w) => Err(format!("input {k}: {} does not match {}", show(&t), show(w))),
                _ => Ok(()),
            }
        };
        let order_first = |ts: &[BTy]| -> W {
            match ts.iter().position(|t| *t == BTy::Order) {
                Some(p) if p > 0 => Err("ω is carried, and not as the first carried value".into()),
                _ => Ok(()),
            }
        };
        match &n.attrs {
            Attr::Prim(name, args, _) => {
                let e = catalogue::entry(name).ok_or("an unknown entry")?;
                for (k, (p, a)) in e.params.iter().zip(args).enumerate() {
                    let ok = match (p.as_str(), a) {
                        ("body", TyArg::Body(t)) => !t.has_order(),
                        ("sum", TyArg::Body(t)) => matches!(cx.tx.unfold(t), Some(BTy::Sum(_))),
                        ("tup", TyArg::Body(t)) => matches!(cx.tx.unfold(t), Some(BTy::Tup(_))),
                        ("frame", TyArg::Frame(_)) => true,
                        // integer kinds are FG12's
                        ("int" | "int-bitwise" | "int-zero", TyArg::Body(_)) => true,
                        _ => false,
                    };
                    if !ok {
                        return Err(format!("`{name}`: type argument {k} is not a {p} type"));
                    }
                }
                let lits: Vec<Option<i128>> = n.ins.iter().map(|s| cx.lit(d, s)).collect();
                let ports = cx.tx.operand_ports(e, args).map_err(|e| format!("`{name}`: {e}"))?;
                for (k, (_, port)) in ports.iter().enumerate() {
                    let Some(t) = n.ins.get(k).and_then(|s| cx.ty(s)) else { continue };
                    if !cx.tx.port_takes(port, &t, args, &lits, lits[k]).map_err(|x| format!("`{name}` operand {k}: {x}"))? {
                        return Err(format!("`{name}` operand {k}: {} does not match its port", show(&t)));
                    }
                }
                cx.tx.results(e, args, &lits).map_err(|x| format!("`{name}`: {x}"))?;
                Ok(())
            }
            Attr::Call(r) | Attr::Catch(r) => {
                let f = cx.tx.callee(cx.group, r).ok_or("an unresolved reference")?;
                for (k, w) in f.ins.iter().enumerate() {
                    need(k, w)?;
                }
                Ok(())
            }
            Attr::Branch(k, _) => match n.ins.first().and_then(|s| cx.ty(s)) {
                Some(BTy::Bool) if *k == 2 => Ok(()),
                Some(BTy::Int(..)) | None => Ok(()),
                Some(t) => Err(format!("the selector of {} is not bool for 2 arms or an integer", show(&t))),
            },
            Attr::Count(ts) => {
                need(0, &BTy::u64())?;
                for (k, w) in ts.iter().enumerate() {
                    need(1 + k, w)?;
                }
                order_first(ts)
            }
            Attr::Measure(_, _, ts) => {
                for (k, w) in ts.iter().enumerate() {
                    need(k, w)?;
                }
                order_first(ts)
            }
            Attr::Halt(_) => {
                match n.ins.first().and_then(|s| cx.ty(s)) {
                    Some(t) if t != BTy::halt() => return Err(format!("a Halt takes a value of type halt, and is given {}", show(&t))),
                    _ => {}
                }
                need(1, &BTy::Order)
            }
            Attr::Lit(..) | Attr::Cap(_) => Ok(()),
        }
    }
}

fn fg5(m: &Module) -> G {
    per_walk(m, "FG5", &|| Fg5)
}

// ---- FG6 Linearity

struct Fg6;

impl<'a> Visit<'a> for Fg6 {
    fn region(&mut self, cx: &Cx<'a>) -> W {
        let fr = cx.top();
        let mut uses: BTreeMap<Src, usize> = BTreeMap::new();
        for n in &fr.body.nodes {
            for s in &n.ins {
                *uses.entry(*s).or_default() += 1;
            }
        }
        for s in &fr.body.out {
            *uses.entry(*s).or_default() += 1;
        }
        let check = |s: Src, t: &BTy, what: String| -> W {
            let u = uses.get(&s).copied().unwrap_or(0);
            if *t == BTy::Order && u != 1 {
                return Err(format!("{what} is ω and has {u} consumers; ω is used exactly once"));
            }
            if *t != BTy::Order && cx.tx.owned(t) && u > 1 {
                return Err(format!("{what} is owned ({}) and has {u} consumers", show(t)));
            }
            Ok(())
        };
        for (j, t) in fr.ins.iter().enumerate() {
            if let Some(t) = t {
                check(Src::In(j as u32), t, format!("input {j}"))?;
            }
        }
        for (i, ts) in fr.outs.iter().enumerate() {
            // a Halt's outputs are never produced
            if matches!(fr.body.nodes[i].attrs, Attr::Halt(_)) {
                continue;
            }
            for (p, t) in ts.iter().flatten().enumerate() {
                check(Src::Out(i as u32, p as u32), t, format!("node {i} output {p}"))?;
            }
        }
        Ok(())
    }
}

fn resolve(group: &str, r: &Ref) -> (String, u32) {
    match r {
        Ref::Key(k, i) => (k.clone(), *i),
        Ref::Rec(i) => (group.to_string(), *i),
    }
}

type CellId = (bool, String, u32);

/// The memo cells taken full and not yet filled again, along the ω chain; a
/// call or a return with one outstanding breaks FG6. Each cell is held with
/// the source of the value `memo.get` gave, while it is a source of the
/// region walked: in arm 0 of a Branch on `sum.tag` of that value the cell
/// was taken empty. Returns the cells outstanding at the end of the region,
/// none when it ends in a Halt.
fn cells(group: &str, b: &Body, mut taken: BTreeMap<CellId, Option<Src>>, path: &str) -> Result<BTreeMap<CellId, Option<Src>>, (String, String)> {
    for (i, n) in b.nodes.iter().enumerate() {
        let p = format!("{path}node {i}");
        match &n.attrs {
            Attr::Prim(name, _, Some(c)) => {
                let (k, f) = resolve(group, c.r());
                let id = (matches!(c, Cell::Memo(_)), k, f);
                if name == "memo.get" {
                    taken.insert(id, Some(Src::Out(i as u32, 0)));
                } else {
                    taken.remove(&id);
                }
            }
            Attr::Call(_) | Attr::Catch(_) if !taken.is_empty() => {
                return Err((p, "a call while a memo cell taken full by memo.get is not yet filled again by memo.put".into()));
            }
            Attr::Halt(_) => return Ok(BTreeMap::new()),
            _ => {}
        }
        if n.regions.is_empty() {
            continue;
        }
        // the value whose tag selects the arm, when the selector is `sum.tag` of it
        let tagged = match (&n.attrs, n.ins.first()) {
            (Attr::Branch(..), Some(Src::Out(t, 0))) => match b.nodes.get(*t as usize) {
                Some(Node { attrs: Attr::Prim(x, _, _), ins, .. }) if x == "sum.tag" => ins.first().copied(),
                _ => None,
            },
            _ => None,
        };
        let inner: BTreeMap<CellId, Option<Src>> = taken.keys().map(|k| (k.clone(), None)).collect();
        let mut after: BTreeMap<CellId, Option<Src>> = BTreeMap::new();
        if matches!(n.attrs, Attr::Branch(..)) {
            for (k, r) in n.regions.iter().enumerate() {
                let mut start = inner.clone();
                if k == 0 {
                    start.retain(|id, _| !(tagged.is_some() && taken.get(id).copied().flatten() == tagged));
                }
                after.extend(cells(group, r, start, &format!("{p} region {k} "))?);
            }
            taken = after;
        } else {
            let t = cells(group, &n.regions[0], inner.clone(), &format!("{p} region 0 "))?;
            let mut s = inner.clone();
            s.extend(t.clone());
            let bd = cells(group, &n.regions[1], s, &format!("{p} region 1 "))?;
            taken.extend(inner);
            taken.extend(t);
            taken.extend(bd);
        }
    }
    Ok(taken)
}

fn fg6(m: &Module) -> G {
    per_walk(m, "FG6", &|| Fg6)?;
    per_fn(m, "FG6", &|_, g, _, f| {
        let left = cells(g, &f.body, BTreeMap::new(), "")?;
        if !left.is_empty() {
            return Err(("body".into(), "the function returns while a memo cell taken by memo.get is not yet filled again".into()));
        }
        Ok(())
    })
}

// ---- FG7 Effects

/// The least effect class of a node (§6): 0 total, 1 may halt, 2 emits, 3 host.
fn node_class(tx: &Tx, group: &str, n: &Node) -> u8 {
    let inner = n.regions.iter().flat_map(|r| r.nodes.iter()).map(|x| node_class(tx, group, x)).max().unwrap_or(0);
    match &n.attrs {
        Attr::Lit(..) => 0,
        Attr::Prim(name, _, _) => catalogue::entry(name).map(|e| e.effect).unwrap_or(0),
        Attr::Call(r) | Attr::Catch(r) => tx.callee(group, r).map(|f| f.eff).unwrap_or(0),
        Attr::Cap(_) => 3,
        Attr::Branch(..) | Attr::Measure(..) => inner,
        Attr::Count(_) => inner.max(1),
        Attr::Halt(_) => 1,
    }
}

struct Fg7;

impl<'a> Visit<'a> for Fg7 {
    fn node(&mut self, cx: &Cx<'a>, i: usize) -> W {
        let n = &cx.top().body.nodes[i];
        let c = node_class(&cx.tx, cx.group, n);
        if c > cx.f.eff {
            return Err(format!("the node has effect class {c}, above the function's class {}", cx.f.eff));
        }
        if c >= 1 && !matches!(n.attrs, Attr::Cap(_)) {
            let takes = n.ins.iter().any(|s| cx.ty(s) == Some(BTy::Order));
            let gives = matches!(n.attrs, Attr::Halt(_)) || cx.top().outs[i].as_ref().is_some_and(|o| o.contains(&BTy::Order));
            if !takes || !gives {
                return Err(format!("the node has effect class {c} and does not consume and return ω"));
            }
        }
        Ok(())
    }
}

fn fg7(m: &Module) -> G {
    per_fn(m, "FG7", &|_, _, _, f| {
        let takes = f.ins.last() == Some(&BTy::Order) && f.outs.last() == Some(&BTy::Order);
        let others = f.ins.iter().rev().skip(usize::from(takes)).chain(f.outs.iter().rev().skip(usize::from(takes))).any(|t| t.has_order());
        if f.eff >= 1 && !takes {
            return Err(("sig".into(), format!("a function of class {} takes ω as its last input and returns it as its last output", f.eff)));
        }
        if others || (f.eff == 0 && takes) {
            return Err(("sig".into(), format!("ω appears in the signature of a function of class {} other than as its last input and output", f.eff)));
        }
        Ok(())
    })?;
    per_walk(m, "FG7", &|| Fg7)?;
    // the context table of 5.2: a step body emits (class 2), a guard is total
    // or may halt (0 or 1), main is in host context (3)
    let tx = Tx::new(m);
    let class = |r: &Ref| tx.callee("", r).map(|f| f.eff);
    for (i, s) in m.steps.iter().enumerate() {
        if let Some(c) = class(&s.body)
            && c != 2
        {
            return Err(frame("FG7", format!("steps[{i}]"), format!("a step body has effect class 2 (5.2), and this one has {c}")));
        }
        if let Some(g) = &s.guard
            && let Some(c) = class(g)
            && c > 1
        {
            return Err(frame("FG7", format!("steps[{i}]"), format!("a guard has effect class 0 or 1, and this one has {c}")));
        }
    }
    if let Some(r) = &m.main
        && let Some(c) = class(r)
        && c != 3
    {
        return Err(frame("FG7", "main", format!("main has effect class 3 (host context, 5.2), and has {c}")));
    }
    Ok(())
}

// ---- FG8 Totality

fn same(cx: &Cx, d1: usize, a: &Src, d2: usize, b: &Src) -> bool {
    cx.trace(d1, *a) == cx.trace(d2, *b)
}

fn is_zero(cx: &Cx, d: usize, s: &Src) -> bool {
    cx.lit(d, s) == Some(0)
}

/// Whether the source is the length (`bytes.len`, `arr.len`, `list.len`) of the same value as `of`.
fn is_len(cx: &Cx, d: usize, s: &Src, dn: usize, of: &Src) -> bool {
    let (sd, ss) = cx.trace(d, *s);
    let Some(n) = cx.node_at(sd, &ss) else { return false };
    matches!(&n.attrs, Attr::Prim(x, _, _) if x == "bytes.len" || x == "arr.len" || x == "list.len")
        && matches!(ss, Src::Out(_, 0))
        && n.ins.first().is_some_and(|x| same(cx, sd, x, dn, of))
}

fn int_range(t: &Option<BTy>) -> Option<(i128, i128)> {
    match t {
        Some(BTy::Int(lo, hi)) => Some((*lo, *hi)),
        _ => None,
    }
}

/// DF1: the operand types imply the domain test `test` of a node.
fn df1(cx: &Cx, n: &Node, test: &str) -> bool {
    let ty = |k: usize| n.ins.get(k).and_then(|s| cx.ty(s));
    let nonneg = |k: usize| int_range(&ty(k)).is_some_and(|(lo, _)| lo >= 0);
    let tau = match &n.attrs {
        Attr::Prim(_, args, _) => match args.first() {
            Some(TyArg::Body(BTy::Int(lo, hi))) => Some((*lo, *hi)),
            _ => None,
        },
        _ => None,
    };
    let within = |r: Option<(i128, i128)>| match (r, tau) {
        (Some((a, b)), Some((lo, hi))) => lo <= a && b <= hi,
        _ => false,
    };
    let (a, b) = (int_range(&ty(0)), int_range(&ty(1)));
    match test {
        "index-not-negative" | "start-not-negative" => nonneg(1),
        "length-not-negative" => nonneg(0),
        "divisor-not-zero" => b.is_some_and(|(lo, hi)| lo > 0 || hi < 0),
        "tag-equals-index" => {
            let one = matches!(ty(0).and_then(|t| cx.tx.unfold(&t)), Some(BTy::Sum(v)) if v.len() == 1);
            one && n.ins.get(1).and_then(|s| cx.lit(cx.depth(), s)) == Some(0)
        }
        "int.bad.add" => within(a.zip(b).and_then(|(a, b)| Some((a.0.checked_add(b.0)?, a.1.checked_add(b.1)?)))),
        "int.bad.sub" => within(a.zip(b).and_then(|(a, b)| Some((a.0.checked_sub(b.1)?, a.1.checked_sub(b.0)?)))),
        "int.bad.mul" => within(a.zip(b).and_then(|(a, b)| {
            let ps = [a.0.checked_mul(b.0)?, a.0.checked_mul(b.1)?, a.1.checked_mul(b.0)?, a.1.checked_mul(b.1)?];
            Some((*ps.iter().min()?, *ps.iter().max()?))
        })),
        "int.bad.neg" => within(a.map(|a| (-a.1, -a.0))),
        "int.bad.abs" => within(a.map(|(lo, hi)| {
            let m = lo.abs().max(hi.abs());
            if lo <= 0 && 0 <= hi { (0, m) } else { (lo.abs().min(hi.abs()), m) }
        })),
        "int.bad.cast" => within(a),
        "int.bad.quo" => within(a.zip(b).and_then(|(a, (blo, bhi))| {
            let ds: Vec<i128> = [blo, bhi, -1, 1].into_iter().filter(|x| *x != 0 && blo <= *x && *x <= bhi).collect();
            let qs: Vec<i128> = ds.iter().flat_map(|d| [a.0 / d, a.1 / d]).collect();
            Some((*qs.iter().min()?, *qs.iter().max()?))
        })),
        "int.bad.shift" => tau.zip(b).is_some_and(|((lo, hi), (klo, khi))| klo >= 0 && khi < crate::one::ir::bits(lo, hi) as i128),
        _ => false,
    }
}

/// Whether node `n` (at the top of the walk) is guarded by `test` (§7): it
/// lies inside the arm of a Branch on that test that the in-domain outcome
/// selects, the test reading the same values as its operands.
fn guarded(cx: &Cx, n: &Node, e: &Entry, test: &str) -> bool {
    let dn = cx.depth();
    let op = |k: usize| n.ins.get(k).copied();
    for d in 1..=dn {
        let fr = &cx.frames[d];
        let Some(b) = fr.arm_of() else { continue };
        let arm = fr.arm;
        let (sd, ss) = cx.trace(d - 1, b.ins[0]);
        let Some(t) = cx.node_at(sd, &ss) else { continue };
        if !matches!(ss, Src::Out(_, 0)) {
            continue;
        }
        let Attr::Prim(tid, targs, _) = &t.attrs else { continue };
        let ti = |k: usize| t.ins.get(k).copied();
        let sv = |a: Option<Src>, b: Option<Src>| matches!((a, b), (Some(a), Some(b)) if same(cx, sd, &a, dn, &b));
        let z = |a: Option<Src>| a.is_some_and(|a| is_zero(cx, sd, &a));
        let len = |a: Option<Src>| matches!((a, op(0)), (Some(a), Some(o)) if is_len(cx, sd, &a, dn, &o));
        let ok = match test {
            "divisor-not-zero" => tid == "int.eq" && arm == 0 && ((sv(ti(0), op(1)) && z(ti(1))) || (z(ti(0)) && sv(ti(1), op(1)))),
            "index-below-length" => tid == "int.lt" && arm == 1 && sv(ti(0), op(1)) && len(ti(1)),
            "end-not-after-length" => tid == "int.le" && arm == 1 && sv(ti(0), op(2)) && len(ti(1)),
            "start-not-after-end" => tid == "int.le" && arm == 1 && sv(ti(0), op(1)) && sv(ti(1), op(2)),
            "index-not-negative" | "start-not-negative" | "length-not-negative" => {
                let k = if test == "length-not-negative" { 0 } else { 1 };
                (tid == "int.le" && arm == 1 && z(ti(0)) && sv(ti(1), op(k))) || (tid == "int.lt" && arm == 0 && sv(ti(0), op(k)) && z(ti(1)))
            }
            "length-above-zero" => (tid == "int.lt" && arm == 1 && z(ti(0)) && len(ti(1))) || (tid == "int.eq" && arm == 0 && ((len(ti(0)) && z(ti(1))) || (z(ti(0)) && len(ti(1))))),
            "tag-equals-index" => tid == "sum.tag" && sv(ti(0), op(0)) && op(1).and_then(|s| cx.lit(dn, &s)) == Some(arm as i128),
            id => {
                let Attr::Prim(_, nargs, _) = &n.attrs else { return false };
                tid == id && arm == 0 && targs == nargs && t.ins.len() == n.ins.len() && {
                    let mut a: Vec<(usize, Src)> = t.ins.iter().map(|s| cx.trace(sd, *s)).collect();
                    let mut b: Vec<(usize, Src)> = n.ins.iter().map(|s| cx.trace(dn, *s)).collect();
                    if e.symmetric.len() == 2 {
                        a.sort();
                        b.sort();
                    }
                    a == b
                }
            }
        };
        if ok {
            return true;
        }
    }
    false
}

/// The closed shape of a measure (§6).
fn measure_shape(n: &Node, i: u32, j: u32) -> Result<(), String> {
    let (test, body) = (&n.regions[0], &n.regions[1]);
    let (iu, ju) = (1 + i as usize, 1 + j as usize);
    if test.out.get(iu) != Some(&Src::In(i)) || test.out.get(ju) != Some(&Src::In(j)) {
        return Err("the test region does not return carried values i and j unchanged".into());
    }
    let is_lt = |s: &Src| match s {
        Src::Out(x, 0) => matches!(test.nodes.get(*x as usize), Some(Node { attrs: Attr::Prim(name, _, _), ins, .. }) if name == "int.lt" && ins[..] == [Src::In(i), Src::In(j)]),
        _ => false,
    };
    let first = test.out.first().ok_or("the test region has no outputs")?;
    let ok = is_lt(first)
        || matches!(first, Src::Out(x, 0) if matches!(test.nodes.get(*x as usize), Some(Node { attrs: Attr::Prim(name, _, _), ins, .. }) if name == "bool.and" && ins.iter().any(is_lt)));
    if !ok {
        return Err("the test region's first output is not int.lt of i and j, or bool.and with it as an operand".into());
    }
    if body.out.get(ju) != Some(&Src::In(j)) {
        return Err("the body region does not return j unchanged".into());
    }
    let is_i = |s: &Src| *s == Src::In(i);
    if !raised(body, body.out.first(), &body.out[iu], &is_i, &|_: &Src| false) {
        return Err("the body region does not return i raised by exactly one whenever it continues".into());
    }
    Ok(())
}

/// Whether `val` is i raised by exactly one (`int.add` of i and the literal 1)
/// whenever the loop continues. `flag` is the continue flag, when it is a
/// source of this region; an arm that halts never continues.
fn raised(b: &Body, flag: Option<&Src>, val: &Src, is_i: &dyn Fn(&Src) -> bool, is_one: &dyn Fn(&Src) -> bool) -> bool {
    let node = |s: &Src| match s {
        Src::Out(x, _) => b.nodes.get(*x as usize),
        _ => None,
    };
    let one = |s: &Src| is_one(s) || matches!(node(s), Some(Node { attrs: Attr::Lit(_, Lit::Int(1)), .. }));
    if let (Src::Out(_, 0), Some(Node { attrs: Attr::Prim(name, _, _), ins, .. })) = (val, node(val))
        && name == "int.add"
        && ins.len() == 2
        && ((is_i(&ins[0]) && one(&ins[1])) || (one(&ins[0]) && is_i(&ins[1])))
    {
        return true;
    }
    if flag.is_some_and(|f| matches!(node(f), Some(Node { attrs: Attr::Lit(_, Lit::Bool(false)), .. }))) {
        return true;
    }
    let Src::Out(x, q) = val else { return false };
    let Some(br) = node(val) else { return false };
    if !matches!(br.attrs, Attr::Branch(..)) {
        return false;
    }
    // the flag, when the same Branch gives it, is followed into each arm
    let r = match flag {
        Some(Src::Out(y, r)) if y == x => Some(*r),
        _ => None,
    };
    br.regions.iter().all(|arm| {
        if arm.nodes.iter().any(|n| matches!(n.attrs, Attr::Halt(_))) {
            return true;
        }
        let cap = |s: &Src| match s {
            Src::In(j) => br.ins.get(1 + *j as usize).copied(),
            _ => None,
        };
        let Some(v) = arm.out.get(*q as usize) else { return false };
        let f = r.and_then(|r| arm.out.get(r as usize));
        raised(arm, f, v, &|s| cap(s).is_some_and(|o| is_i(&o)), &|s| cap(s).is_some_and(|o| one(&o)))
    })
}

struct Fg8;

impl<'a> Visit<'a> for Fg8 {
    fn node(&mut self, cx: &Cx<'a>, i: usize) -> W {
        let n = &cx.top().body.nodes[i];
        match &n.attrs {
            Attr::Prim(name, _, _) => {
                let e = catalogue::entry(name).ok_or("an unknown entry")?;
                if let Some((fact, tests)) = &e.domain {
                    for t in tests {
                        if !df1(cx, n, t) && !guarded(cx, n, e, t) {
                            return Err(format!("`{name}` is partial, and no fact establishes its domain: neither DF1 nor {fact} gives the test `{t}` on the same values as its operands"));
                        }
                    }
                }
                Ok(())
            }
            Attr::Measure(x, y, _) => measure_shape(n, *x, *y),
            Attr::Branch(k, _) => match n.ins.first().and_then(|s| cx.ty(s)) {
                Some(BTy::Int(lo, hi)) if lo < 0 || hi >= *k as i128 => {
                    Err(format!("the selector ranges over [{lo}, {hi}], and the Branch has a region for 0 to {} only", k - 1))
                }
                _ => Ok(()),
            },
            _ => Ok(()),
        }
    }
}

/// The calls a function makes to members of its own group, with, for each
/// argument, the parameter it is and whether it is a component of it.
struct Calls {
    caller: u32,
    out: Vec<RecCall>,
}

fn classify(cx: &Cx, d: usize, s: &Src) -> Option<(u32, bool)> {
    let (dd, ss) = cx.trace(d, *s);
    match ss {
        Src::In(i) if dd == 0 => Some((i, false)),
        Src::Out(x, p) => {
            let n = cx.frames[dd].body.nodes.get(x as usize)?;
            let Attr::Prim(name, _, _) = &n.attrs else { return None };
            let comp = match name.as_str() {
                "sum.get" | "tup.get" => p == 0,
                "tup.open" => true,
                _ => false,
            };
            if !comp {
                return None;
            }
            classify(cx, dd, n.ins.first()?).map(|(i, _)| (i, true))
        }
        _ => None,
    }
}

impl<'a> Visit<'a> for Calls {
    fn node(&mut self, cx: &Cx<'a>, i: usize) -> W {
        let n = &cx.top().body.nodes[i];
        if let Attr::Call(Ref::Rec(g)) | Attr::Catch(Ref::Rec(g)) = &n.attrs {
            let d = cx.depth();
            self.out.push((self.caller, *g, n.ins.iter().map(|s| classify(cx, d, s)).collect()));
        }
        Ok(())
    }
}

/// Whether the calls among `members` form a cycle.
fn cyclic(edges: &[(u32, u32)], members: &BTreeSet<u32>) -> bool {
    let mut st: BTreeMap<u32, u8> = BTreeMap::new();
    fn go(v: u32, e: &[(u32, u32)], ms: &BTreeSet<u32>, st: &mut BTreeMap<u32, u8>) -> bool {
        match st.get(&v) {
            Some(1) => return true,
            Some(_) => return false,
            None => {}
        }
        st.insert(v, 1);
        for (a, b) in e {
            if *a == v && ms.contains(b) && go(*b, e, ms, st) {
                return true;
            }
        }
        st.insert(v, 2);
        false
    }
    members.iter().any(|v| go(*v, edges, members, &mut st))
}

/// The members that reach `v` and that `v` reaches: its strongly connected component.
fn scc(v: u32, edges: &[(u32, u32)], n: u32) -> BTreeSet<u32> {
    let reach = |from: u32, fwd: bool| {
        let mut seen = BTreeSet::from([from]);
        let mut stack = vec![from];
        while let Some(x) = stack.pop() {
            for (a, b) in edges {
                let (s, t) = if fwd { (a, b) } else { (b, a) };
                if *s == x && *t < n && seen.insert(*t) {
                    stack.push(*t);
                }
            }
        }
        seen
    };
    reach(v, true).intersection(&reach(v, false)).copied().collect()
}

fn fg8_group(m: &Module, k: &str, g: &Group) -> G {
    let n = g.fns.len() as u32;
    let mut calls: Vec<RecCall> = vec![];
    for (fi, f) in g.fns.iter().enumerate() {
        let mut c = Calls { caller: fi as u32, out: vec![] };
        let _ = walk(m, k, fi as u32, f, &mut c);
        calls.extend(c.out);
    }
    let edges: Vec<(u32, u32)> = calls.iter().map(|c| (c.0, c.1)).collect();
    for fi in 0..n {
        let comp = scc(fi, &edges, n);
        let recursive = edges.iter().any(|(a, b)| comp.contains(a) && comp.contains(b));
        let f = &g.fns[fi as usize];
        if !recursive {
            if f.bound != Bound::None {
                return Err(kept("FG8", Some(k), Some(fi), "bound", "a function that is not recursive has bound null"));
            }
            continue;
        }
        if fi != *comp.iter().next().unwrap() {
            continue;
        }
        // layered measures: every member carries one certificate that holds on the graph
        let certs: Vec<&Bound> = comp.iter().map(|x| &g.fns[*x as usize].bound).collect();
        if let Bound::Layers(l) = certs[0]
            && certs.iter().all(|c| *c == certs[0])
            && layers_hold(l, &comp, &calls, n)
        {
            continue;
        }
        // declared bounds: every cycle of calls passes through a bounded function
        let unbounded: BTreeSet<u32> = comp.iter().copied().filter(|x| !matches!(g.fns[*x as usize].bound, Bound::N(_))).collect();
        if cyclic(&edges, &unbounded) {
            return Err(kept(
                "FG8",
                Some(k),
                Some(fi),
                "bound",
                "a cycle of calls passes through no function with a declared bound, and no layered-measure certificate holds on the graph",
            ));
        }
    }
    Ok(())
}

/// The layered-measure certificate of 6.7 on the graph: layer k gives each
/// member f of the cycle the parameter positions `l[k][f]`.
fn layers_hold(l: &[Vec<Vec<u32>>], comp: &BTreeSet<u32>, calls: &[RecCall], n: u32) -> bool {
    let mut left: Vec<&RecCall> = calls.iter().filter(|c| comp.contains(&c.0) && comp.contains(&c.1)).collect();
    for layer in l {
        if layer.len() != n as usize {
            return false;
        }
        // keeps: the positions of the callee map one to one into the caller's,
        // each argument the parameter at its image or a component of it
        let judge = |c: &RecCall| -> Option<bool> {
            let (pf, pg) = (&layer[c.0 as usize], &layer[c.1 as usize]);
            let mut img = BTreeSet::new();
            let mut shrinks = false;
            for j in pg {
                let (i, comp) = c.2.get(*j as usize).copied().flatten()?;
                if !pf.contains(&i) || !img.insert(i) {
                    return None;
                }
                shrinks |= comp;
            }
            Some(shrinks)
        };
        let mut done = vec![];
        for (x, c) in left.iter().enumerate() {
            match judge(c) {
                None => return false,
                Some(true) => done.push(x),
                Some(false) => {}
            }
        }
        if done.is_empty() {
            return false;
        }
        left = left.into_iter().enumerate().filter(|(x, _)| !done.contains(x)).map(|(_, c)| c).collect();
    }
    let edges: Vec<(u32, u32)> = left.iter().map(|c| (c.0, c.1)).collect();
    !cyclic(&edges, comp)
}

fn fg8(m: &Module) -> G {
    per_walk(m, "FG8", &|| Fg8)?;
    for (k, g) in &m.defs {
        fg8_group(m, k, g)?;
    }
    Ok(())
}

// ---- FG9 Acyclic regions

fn acyclic(b: &Body, path: &str) -> At {
    // a region is acyclic when it has a topological order; its index order is
    // one when every source precedes its consumer
    for (i, n) in b.nodes.iter().enumerate() {
        for s in &n.ins {
            if let Src::Out(x, _) = s
                && *x as usize >= i
            {
                return Err((format!("{path}node {i}"), "the region is not a directed acyclic graph in its numbering".into()));
            }
        }
        for (k, r) in n.regions.iter().enumerate() {
            acyclic(r, &format!("{path}node {i} region {k} "))?;
        }
    }
    Ok(())
}

fn fg9(m: &Module) -> G {
    per_fn(m, "FG9", &|_, _, _, f| acyclic(&f.body, ""))
}

// ---- FG10 State and FG11 Events

/// The canonical order of Form values (6.1).
fn order(a: &J, b: &J) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let rank = |v: &J| match v {
        J::Null => 0,
        J::Bool(_) => 1,
        J::Number(n) if !n.is_f64() => 2,
        J::Number(_) => 3,
        J::String(_) => 4,
        J::Array(_) => 5,
        J::Object(_) => 6,
    };
    match (a, b) {
        _ if rank(a) != rank(b) => rank(a).cmp(&rank(b)),
        (J::Bool(x), J::Bool(y)) => x.cmp(y),
        (J::Number(x), J::Number(y)) if !x.is_f64() => {
            let v = |n: &serde_json::Number| n.as_i64().map(i128::from).or(n.as_u64().map(i128::from)).unwrap_or(0);
            v(x).cmp(&v(y))
        }
        (J::Number(x), J::Number(y)) => x.as_f64().unwrap_or(0.0).total_cmp(&y.as_f64().unwrap_or(0.0)),
        (J::String(x), J::String(y)) => x.cmp(y),
        (J::Array(x), J::Array(y)) => {
            for (p, q) in x.iter().zip(y) {
                let o = order(p, q);
                if o != Ordering::Equal {
                    return o;
                }
            }
            x.len().cmp(&y.len())
        }
        (J::Object(x), J::Object(y)) => order(&J::Array(x.iter().map(|(k, v)| J::Array(vec![J::from(k.clone()), v.clone()])).collect()), &J::Array(y.iter().map(|(k, v)| J::Array(vec![J::from(k.clone()), v.clone()])).collect())),
        _ => Ordering::Equal,
    }
}

/// Whether a positional canonical JSON value (6.1) is a value of a frame type.
pub fn value_of(m: &Module, t: &FTy, v: &J) -> Result<(), String> {
    let bad = || Err(format!("{} is not a value of {}", canon(v), fshow(t)));
    let int = |v: &J| -> Option<i128> {
        match v {
            J::Number(n) if !n.is_f64() => n.as_i64().map(i128::from).or(n.as_u64().map(i128::from)),
            _ => None,
        }
    };
    let ascending = |xs: &[J], key: &dyn Fn(&J) -> J| xs.windows(2).all(|w| order(&key(&w[0]), &key(&w[1])) == std::cmp::Ordering::Less);
    match (t, v) {
        (FTy::Bool, J::Bool(_)) | (FTy::Text, J::String(_)) | (FTy::Unit, J::Null) => Ok(()),
        (FTy::Int(lo, hi), _) => match int(v) {
            Some(x) if *lo <= x && x <= *hi => Ok(()),
            _ => bad(),
        },
        (FTy::F64, J::Number(n)) if n.is_f64() => Ok(()),
        (FTy::F32, J::Number(n)) if n.is_f64() => {
            let x = n.as_f64().unwrap_or(0.0);
            if (x as f32) as f64 == x { Ok(()) } else { bad() }
        }
        (FTy::Tup(xs), J::Array(a)) if xs.len() == a.len() => xs.iter().zip(a).try_for_each(|(x, y)| value_of(m, x, y)),
        (FTy::Arr(x, n), J::Array(a)) if a.len() as u64 == *n => a.iter().try_for_each(|y| value_of(m, x, y)),
        (FTy::List(x), J::Array(a)) => a.iter().try_for_each(|y| value_of(m, x, y)),
        (FTy::Set(x), J::Array(a)) => {
            a.iter().try_for_each(|y| value_of(m, x, y))?;
            if ascending(a, &|v| v.clone()) { Ok(()) } else { Err(format!("{} is not sorted by canonical order without duplicates", canon(v))) }
        }
        (FTy::Map(k, w), J::Array(a)) => {
            for p in a {
                match p.as_array().map(|x| &x[..]) {
                    Some([x, y]) => {
                        value_of(m, k, x)?;
                        value_of(m, w, y)?;
                    }
                    _ => return bad(),
                }
            }
            if ascending(a, &|p| p[0].clone()) { Ok(()) } else { Err(format!("{} is not sorted by canonical order of its keys", canon(v))) }
        }
        (FTy::Opt(_), J::Null) => Ok(()),
        (FTy::Opt(x), _) => value_of(m, x, v),
        (FTy::Res(..), _) => Err(format!("{}: §3 and 6.1 state no positional form of a result", fshow(t))),
        (FTy::Decl(i), _) => match &m.types[*i as usize] {
            TypeDecl::Record(fs) => match v.as_array() {
                Some(a) if a.len() == fs.len() => fs.iter().zip(a).try_for_each(|(x, y)| value_of(m, x, y)),
                _ => bad(),
            },
            TypeDecl::Kind(vs) => match v.as_array().map(|a| &a[..]) {
                Some([k, J::Array(fields)]) => match int(k).and_then(|k| vs.get(usize::try_from(k).ok()?)) {
                    Some(fs) if fs.len() == fields.len() => fs.iter().zip(fields).try_for_each(|(x, y)| value_of(m, x, y)),
                    _ => bad(),
                },
                _ => bad(),
            },
            TypeDecl::Refined(x, r) => {
                value_of(m, x, v)?;
                match eval::predicate(m, r, x, v) {
                    Ok(true) => Ok(()),
                    Ok(false) => Err(format!("{} does not satisfy the predicate of {}", canon(v), fshow(t))),
                    Err(e) => Err(format!("the predicate of {} on {}: {e}", fshow(t), canon(v))),
                }
            }
        },
        _ => bad(),
    }
}

/// The payload types of a step's event: a declared event, or a built-in one.
fn payload(m: &Module, ev: &Ev) -> Option<Vec<FTy>> {
    match ev {
        Ev::Idx(i) => m.events.get(*i as usize).cloned(),
        Ev::Name(n) if n == "Tick" => Some(vec![FTy::Int(0, (1i128 << 64) - 1)]),
        Ev::Name(_) => None,
    }
}

fn reprs(tx: &Tx, ts: &[FTy]) -> Option<Vec<BTy>> {
    ts.iter().map(|t| tx.repr(t)).collect()
}

fn fg10(m: &Module) -> G {
    let tx = Tx::new(m);
    for (i, r) in m.regions.iter().enumerate() {
        for (k, (t, v)) in r.fields.iter().enumerate() {
            value_of(m, t, v).map_err(|e| frame("FG10", format!("regions[{i}] field {k}"), format!("the initial value is not a comptime value of the field's type: {e}")))?;
        }
    }
    for (i, s) in m.steps.iter().enumerate() {
        let at = format!("steps[{i}]");
        let Some(r) = m.regions.get(s.region as usize) else {
            return Err(frame("FG10", at, format!("region {} does not exist", s.region)));
        };
        let fields: Vec<FTy> = r.fields.iter().map(|f| f.0.clone()).collect();
        let Some(rf) = reprs(&tx, &fields) else {
            return Err(frame("FG10", at, "a field's type has no body type that represents it"));
        };
        let nf = rf.len();
        let body = tx.callee("", &s.body).unwrap();
        let mut want_out = rf.clone();
        want_out.push(BTy::Order);
        if body.ins.len() < nf || body.ins[..nf] != rf[..] || body.outs != want_out {
            return Err(frame("FG10", at, "the step body's signature is not its region's fields: inputs (fields, payload, ω), outputs (fields, ω)"));
        }
        if let Some(g) = &s.guard {
            let g = tx.callee("", g).unwrap();
            if g.ins.len() < nf || g.ins[..nf] != rf[..] {
                return Err(frame("FG10", at, "the guard's inputs do not start with its region's fields"));
            }
        }
    }
    Ok(())
}

fn fg11(m: &Module) -> G {
    let tx = Tx::new(m);
    for (i, s) in m.steps.iter().enumerate() {
        let at = format!("steps[{i}]");
        let Some(p) = payload(m, &s.ev) else {
            return Err(frame("FG11", at, format!("the step names no declared or built-in event: {}", match &s.ev { Ev::Idx(i) => i.to_string(), Ev::Name(n) => n.clone() })));
        };
        let Some(rp) = reprs(&tx, &p) else {
            return Err(frame("FG11", at, "a payload field's type has no body type that represents it"));
        };
        let nf = m.regions[s.region as usize].fields.len();
        let body = tx.callee("", &s.body).unwrap();
        let mut want = body.ins[..nf].to_vec();
        want.extend(rp.iter().cloned());
        want.push(BTy::Order);
        if body.ins != want {
            return Err(frame("FG11", at, "the step body's inputs are not (fields, payload, ω)"));
        }
        if let Some(g) = &s.guard {
            let g = tx.callee("", g).unwrap();
            let mut gw = body.ins[..nf + rp.len()].to_vec();
            let mut go = vec![BTy::Bool];
            if g.eff >= 1 {
                gw.push(BTy::Order);
                go.push(BTy::Order);
            }
            if g.ins != gw || g.outs != go {
                return Err(frame("FG11", at, "the guard's signature is not (fields, payload) to bool, with ω when it may halt"));
            }
        }
    }
    if let Some(r) = &m.main {
        // `Main` is the module's one region, and `Start` carries the arguments (4.2),
        // or arguments, environment and standard input as byte lists (§9)
        let f = tx.callee("", r).unwrap();
        if m.regions.len() != 1 || m.regions[0].actor {
            return Err(frame("FG11", "main", "a module with main has one state region, Main"));
        }
        let rf = reprs(&tx, &m.regions[0].fields.iter().map(|f| f.0.clone()).collect::<Vec<_>>()).ok_or_else(|| frame("FG11", "main", "a field of Main has no body type"))?;
        let bytes = BTy::List(Box::new(BTy::u8()));
        let starts = [vec![BTy::List(Box::new(BTy::Bytes))], vec![BTy::List(Box::new(bytes.clone())), BTy::List(Box::new(BTy::Tup(vec![bytes.clone(), bytes.clone()]))), bytes]];
        let mut outs = rf.clone();
        outs.push(BTy::Order);
        let ok = starts.iter().any(|st| {
            let mut ins = rf.clone();
            ins.extend(st.iter().cloned());
            ins.push(BTy::Order);
            f.ins == ins
        });
        if !ok || f.outs != outs {
            return Err(frame("FG11", "main", "main's signature is not (Main's fields, Start's payload, ω) to (Main's fields, ω)"));
        }
    }
    Ok(())
}

// ---- FG12 Numbers

fn bitwise(lo: i128, hi: i128) -> bool {
    (1..=64u32).any(|k| (lo == 0 && hi == (1i128 << k) - 1) || (lo == -(1i128 << (k - 1)) && hi == (1i128 << (k - 1)) - 1))
}

struct Fg12;

impl<'a> Visit<'a> for Fg12 {
    fn node(&mut self, cx: &Cx<'a>, i: usize) -> W {
        let n = &cx.top().body.nodes[i];
        match &n.attrs {
            Attr::Prim(name, args, _) => {
                let e = catalogue::entry(name).ok_or("an unknown entry")?;
                for (k, (p, a)) in e.params.iter().zip(args).enumerate() {
                    let r = match a {
                        TyArg::Body(BTy::Int(lo, hi)) => Some((*lo, *hi)),
                        _ => None,
                    };
                    let ok = match p.as_str() {
                        "int" => r.is_some(),
                        "int-bitwise" => r.is_some_and(|(lo, hi)| bitwise(lo, hi)),
                        "int-zero" => r.is_some_and(|(lo, hi)| lo <= 0 && 0 <= hi),
                        _ => true,
                    };
                    if !ok {
                        return Err(format!("`{name}` is not typed at a width: type argument {k} is not an integer type of kind {p}"));
                    }
                }
                Ok(())
            }
            Attr::Lit(_, Lit::F64(b)) if !f64::from_bits(*b).is_finite() => Err(format!("the f64 literal {b:016x} is not finite (F4)")),
            Attr::Lit(_, Lit::F32(b)) => {
                let x = f64::from_bits(*b);
                if !x.is_finite() || (x as f32) as f64 != x { Err(format!("the f32 literal {b:016x} is not a finite binary32 value (F4, 6.3 N3)")) } else { Ok(()) }
            }
            _ => Ok(()),
        }
    }
}

fn fg12(m: &Module) -> G {
    per_walk(m, "FG12", &|| Fg12)
}

// ---- FG13 Boundary

fn caps_in(b: &Body, out: &mut Vec<(String, String)>, path: &str) {
    for (i, n) in b.nodes.iter().enumerate() {
        let p = format!("{path}node {i}");
        if let Attr::Cap(k) = &n.attrs {
            out.push((p.clone(), k.clone()));
        }
        for (k, r) in n.regions.iter().enumerate() {
            caps_in(r, out, &format!("{p} region {k} "));
        }
    }
}

fn fg13(m: &Module) -> G {
    // no `rce-cap/1` signature is known to the checker: the format is named
    // in §21 and defined nowhere yet, so every capability is kept
    per_fn(m, "FG13", &|_, _, _, f| {
        let mut cs = vec![];
        caps_in(&f.body, &mut cs, "");
        match cs.first() {
            Some((p, k)) => Err((p.clone(), format!("capability {k} has no rce-cap/1 signature"))),
            None => Ok(()),
        }
    })?;
    if let Some(k) = m.caps.first() {
        return Err(frame("FG13", "caps", format!("capability {k} has no rce-cap/1 signature")));
    }
    Ok(())
}

// ---- FG14 Frame types

/// Whether the positional form of a type can be null (§3).
fn nullable(m: &Module, t: &FTy, depth: usize) -> bool {
    match t {
        FTy::Unit | FTy::Opt(_) => true,
        FTy::Decl(i) if depth <= m.types.len() => matches!(&m.types[*i as usize], TypeDecl::Refined(x, _) if nullable(m, x, depth + 1)),
        _ => false,
    }
}

fn frame_ty(m: &Module, t: &FTy) -> Result<(), String> {
    let mut stack = vec![t];
    while let Some(t) = stack.pop() {
        match t {
            FTy::Opt(x) => {
                if nullable(m, x, 0) {
                    return Err(format!("{}: an option of a type whose positional form can be null shares a positional form between two values", fshow(t)));
                }
                stack.push(x);
            }
            FTy::Tup(xs) => stack.extend(xs),
            FTy::Arr(x, _) | FTy::List(x) | FTy::Set(x) | FTy::Chan(x) | FTy::Res(x, _) => stack.push(x),
            FTy::Map(k, v) => {
                stack.push(k);
                stack.push(v);
            }
            _ => {}
        }
    }
    Ok(())
}

fn fg14(m: &Module) -> G {
    let tx = Tx::new(m);
    let mut all: Vec<(String, &FTy)> = vec![];
    for (i, d) in m.types.iter().enumerate() {
        match d {
            TypeDecl::Record(fs) => all.extend(fs.iter().map(|f| (format!("types[{i}]"), f))),
            TypeDecl::Kind(vs) => all.extend(vs.iter().flatten().map(|f| (format!("types[{i}]"), f))),
            TypeDecl::Refined(f, r) => {
                all.push((format!("types[{i}]"), f));
                let p = tx.callee("", r).unwrap();
                let want = tx.repr(f).map(|x| vec![x]);
                if Some(&p.ins) != want.as_ref() || p.outs != [BTy::Bool] || p.eff != 0 {
                    return Err(frame("FG14", format!("types[{i}]"), "a refined type names a class-0 definition of signature [[τ],[[\"bool\"]],0]"));
                }
            }
        }
    }
    for (i, r) in m.regions.iter().enumerate() {
        all.extend(r.fields.iter().map(|f| (format!("regions[{i}]"), &f.0)));
    }
    for (name, tab) in [("events", &m.events), ("errors", &m.errors)] {
        for (i, e) in tab.iter().enumerate() {
            all.extend(e.iter().map(|f| (format!("{name}[{i}]"), f)));
        }
    }
    for (p, t) in all {
        frame_ty(m, t).map_err(|e| frame("FG14", p, e))?;
    }
    per_walk(m, "FG14", &|| Fg14)
}

struct Fg14;

impl<'a> Visit<'a> for Fg14 {
    fn node(&mut self, cx: &Cx<'a>, i: usize) -> W {
        if let Attr::Prim(_, args, _) = &cx.top().body.nodes[i].attrs {
            for a in args {
                if let TyArg::Frame(t) = a {
                    frame_ty(cx.tx.m, t)?;
                }
            }
        }
        Ok(())
    }
}

// ---- the gate

/// FG1 to FG15 on a module's bytes and, when given, its admission vectors.
pub fn gate(bytes: &[u8], vectors: Option<&[u8]>) -> Result<Module, Kept> {
    let m = decode::decode(bytes)?;
    let raw: J = serde_json::from_slice(bytes).map_err(|e| frame("FG1", "", format!("C1: {e}")))?;
    fg2(&m, &raw)?;
    fg3(bytes, &m)?;
    fg4(&m)?;
    fg5(&m)?;
    fg6(&m)?;
    fg7(&m)?;
    fg8(&m)?;
    fg9(&m)?;
    fg10(&m)?;
    fg11(&m)?;
    fg12(&m)?;
    fg13(&m)?;
    fg14(&m)?;
    if let Some(v) = vectors {
        eval::admission(&m, bytes, v).map_err(|(p, r)| frame("FG15", p, r))?;
    }
    Ok(m)
}
