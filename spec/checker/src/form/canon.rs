//! The canonical rules C2 to C11 of rce-form.md §10, and group and module
//! keys. `canonical` makes a decoded module canonical by passes, one per
//! rule, repeated until none changes it; the first pass that changes a
//! module names the rule it breaks (FG3).
//!
//! The Merkle digest (C2) is computed over canonical JSON, a digest written
//! as its lower-case hexadecimal text: md(input j) = SHA-256 of `["in", j]`;
//! md(output p of node i) = SHA-256 of `["out", md(node i), p]`; md(node) =
//! SHA-256 of `[k, attrs, [md of each source], [md of each region]]`;
//! md(region) = SHA-256 of `["region", [md of each output]]`.
use super::catalogue;
use super::ir::*;
use super::types::{Tx, node_outs};
use serde_json::{Value as J, json};
use std::collections::{BTreeMap, HashMap};

/// A region whose canonical form does not exist: a source names no node, or
/// its sources form a cycle. FG4 decides such a module.
#[derive(Debug)]
pub struct Defer(pub String);

type R<T> = Result<T, Defer>;

fn md_in(j: u32) -> String {
    sha(&json!(["in", j]))
}

fn md_src(s: &Src, md: &[Option<String>]) -> String {
    match s {
        Src::In(j) => md_in(*j),
        Src::Out(i, p) => sha(&json!(["out", md[*i as usize].as_deref().unwrap_or("?"), p])),
    }
}

type Hook<'h> = &'h dyn Fn(&mut Node, &dyn Fn(&Src) -> String);

/// The Merkle digests of a region's nodes and of the region (C2), visiting
/// nodes so that every source is digested before its consumer. `hook` may
/// rewrite a node (its sources' digests known) before it is digested; it is
/// applied in every region inside as well. Returns the rewritten region.
fn md_region(body: &Body, hook: Hook) -> R<(Body, String, Vec<String>)> {
    let n = body.nodes.len();
    let mut nodes = body.nodes.clone();
    let mut md: Vec<Option<String>> = vec![None; n];
    let mut state = vec![0u8; n];
    for root in 0..n {
        if state[root] == 2 {
            continue;
        }
        state[root] = 1;
        let mut stack = vec![(root, 0usize)];
        while let Some(top) = stack.last_mut() {
            let (i, k) = (top.0, top.1);
            if k < nodes[i].ins.len() {
                top.1 += 1;
                if let Src::Out(j, _) = nodes[i].ins[k] {
                    let j = j as usize;
                    if j >= n {
                        return Err(Defer(format!("node {i} names node {j}, which does not exist")));
                    }
                    match state[j] {
                        0 => {
                            state[j] = 1;
                            stack.push((j, 0));
                        }
                        1 => return Err(Defer(format!("the sources of node {j} form a cycle"))),
                        _ => {}
                    }
                }
                continue;
            }
            stack.pop();
            let mut node = nodes[i].clone();
            hook(&mut node, &|s| md_src(s, &md));
            let mut rmd = vec![];
            for r in &mut node.regions {
                let (b, d, _) = md_region(r, hook)?;
                *r = b;
                rmd.push(d);
            }
            let srcs: Vec<String> = node.ins.iter().map(|s| md_src(s, &md)).collect();
            md[i] = Some(sha(&json!([node.attrs.kind(), attrs_json(&node.attrs), srcs, rmd])));
            nodes[i] = node;
            state[i] = 2;
        }
    }
    for s in &body.out {
        if let Src::Out(i, _) = s
            && *i as usize >= n
        {
            return Err(Defer(format!("an output names node {i}, which does not exist")));
        }
    }
    let rmd = sha(&json!(["region", body.out.iter().map(|s| md_src(s, &md)).collect::<Vec<_>>()]));
    Ok((Body { nodes, out: body.out.clone() }, rmd, md.into_iter().map(|x| x.unwrap_or_default()).collect()))
}

/// The Merkle digest of a region (C2).
pub fn region_digest(body: &Body) -> R<String> {
    Ok(md_region(body, &|_, _| {})?.1)
}

/// Rewrite a region's own sources (not those of regions inside), then
/// remove the nodes `gone` marks, renumbering the rest in their order.
fn rewire(body: &Body, subst: &dyn Fn(Src) -> Src, gone: &[bool]) -> Body {
    let mut map = vec![u32::MAX; body.nodes.len()];
    let mut k = 0;
    for (i, g) in gone.iter().enumerate() {
        if !g {
            map[i] = k;
            k += 1;
        }
    }
    let fix = |s: Src| match subst(s) {
        Src::Out(i, p) => Src::Out(*map.get(i as usize).unwrap_or(&u32::MAX), p),
        x => x,
    };
    Body {
        nodes: body
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| !gone[*i])
            .map(|(_, n)| Node { attrs: n.attrs.clone(), ins: n.ins.iter().map(|s| fix(*s)).collect(), regions: n.regions.clone() })
            .collect(),
        out: body.out.iter().map(|s| fix(*s)).collect(),
    }
}

/// Apply `f` to every function of every group, then bring the keys up to date.
fn each_fn(m: &Module, f: &dyn Fn(&Tx, &str, &Func) -> R<Func>) -> R<Module> {
    let tx = Tx::new(m);
    let mut out = m.clone();
    for (k, g) in &m.defs {
        for (i, fun) in g.fns.iter().enumerate() {
            let nf = f(&tx, k, fun)?;
            out.defs.get_mut(k).unwrap().fns[i] = nf;
        }
    }
    Ok(refresh_keys(&out, &HashMap::new()))
}

// ---- C6 reachability

fn c6_body(b: &Body) -> R<Body> {
    let n = b.nodes.len();
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = vec![];
    let push = |s: &Src, seen: &mut Vec<bool>, stack: &mut Vec<usize>| -> R<()> {
        if let Src::Out(i, _) = s {
            let i = *i as usize;
            if i >= n {
                return Err(Defer(format!("a source names node {i}, which does not exist")));
            }
            if !seen[i] {
                seen[i] = true;
                stack.push(i);
            }
        }
        Ok(())
    };
    for s in &b.out {
        push(s, &mut seen, &mut stack)?;
    }
    while let Some(i) = stack.pop() {
        for s in &b.nodes[i].ins {
            push(s, &mut seen, &mut stack)?;
        }
    }
    let gone: Vec<bool> = seen.iter().map(|x| !x).collect();
    let mut nb = rewire(b, &|s| s, &gone);
    for node in &mut nb.nodes {
        for r in &mut node.regions {
            *r = c6_body(r)?;
        }
    }
    Ok(nb)
}

fn c6(m: &Module) -> R<Module> {
    each_fn(m, &|_, _, f| Ok(Func { body: c6_body(&f.body)?, ..f.clone() }))
}

// ---- C5 numbering

fn c5_body(b: &Body) -> R<Body> {
    let n = b.nodes.len();
    let mut order = vec![];
    let mut state = vec![0u8; n];
    for s in &b.out {
        let Src::Out(root, _) = *s else { continue };
        let root = root as usize;
        if root >= n {
            return Err(Defer(format!("an output names node {root}, which does not exist")));
        }
        if state[root] != 0 {
            continue;
        }
        state[root] = 1;
        let mut stack = vec![(root, 0usize)];
        while let Some(top) = stack.last_mut() {
            let (i, k) = (top.0, top.1);
            if k < b.nodes[i].ins.len() {
                top.1 += 1;
                if let Src::Out(j, _) = b.nodes[i].ins[k] {
                    let j = j as usize;
                    if j >= n {
                        return Err(Defer(format!("node {i} names node {j}, which does not exist")));
                    }
                    match state[j] {
                        0 => {
                            state[j] = 1;
                            stack.push((j, 0));
                        }
                        1 => return Err(Defer(format!("the sources of node {j} form a cycle"))),
                        _ => {}
                    }
                }
                continue;
            }
            stack.pop();
            state[i] = 2;
            order.push(i);
        }
    }
    // nodes no output reaches keep their order after the others (C6 removes them)
    for (i, s) in state.iter().enumerate() {
        if *s == 0 {
            order.push(i);
        }
    }
    let mut pos = vec![0u32; n];
    for (k, i) in order.iter().enumerate() {
        pos[*i] = k as u32;
    }
    let fix = |s: &Src| match s {
        Src::Out(i, p) => Src::Out(pos[*i as usize], *p),
        x => *x,
    };
    let mut nodes = vec![];
    for i in order {
        let nd = &b.nodes[i];
        nodes.push(Node { attrs: nd.attrs.clone(), ins: nd.ins.iter().map(fix).collect(), regions: nd.regions.iter().map(c5_body).collect::<R<_>>()? });
    }
    Ok(Body { nodes, out: b.out.iter().map(fix).collect() })
}

fn c5(m: &Module) -> R<Module> {
    each_fn(m, &|_, _, f| Ok(Func { body: c5_body(&f.body)?, ..f.clone() }))
}

// ---- C3 symmetric operands

fn c3_hook(node: &mut Node, md: &dyn Fn(&Src) -> String) {
    if let Attr::Prim(name, _, _) = &node.attrs
        && let Some(e) = catalogue::entry(name)
        && let [a, b] = e.symmetric[..]
        && a < node.ins.len()
        && b < node.ins.len()
        && md(&node.ins[b]) < md(&node.ins[a])
    {
        node.ins.swap(a, b);
    }
}

fn c3(m: &Module) -> R<Module> {
    each_fn(m, &|_, _, f| Ok(Func { body: md_region(&f.body, &c3_hook)?.0, ..f.clone() }))
}

// ---- C7 captures and carried values

/// Whether a region uses its input j other than as output `except`.
fn uses_input(b: &Body, j: u32, except: Option<usize>) -> bool {
    b.nodes.iter().any(|n| n.ins.contains(&Src::In(j))) || b.out.iter().enumerate().any(|(k, s)| *s == Src::In(j) && Some(k) != except)
}

/// Renumber a region's own inputs.
fn map_inputs(b: &Body, f: &dyn Fn(u32) -> Src) -> Body {
    let fix = |s: &Src| match s {
        Src::In(j) => f(*j),
        x => *x,
    };
    Body {
        nodes: b.nodes.iter().map(|n| Node { attrs: n.attrs.clone(), ins: n.ins.iter().map(fix).collect(), regions: n.regions.clone() }).collect(),
        out: b.out.iter().map(fix).collect(),
    }
}

fn c7_hook(node: &mut Node, md: &dyn Fn(&Src) -> String) {
    if !matches!(node.attrs, Attr::Branch(..)) || node.ins.is_empty() {
        return;
    }
    let caps = node.ins[1..].to_vec();
    // the outer sources the arms use, each once, in ascending digest
    let mut used: Vec<Src> = vec![];
    for (j, s) in caps.iter().enumerate() {
        if node.regions.iter().any(|r| uses_input(r, j as u32, None)) && !used.contains(s) {
            used.push(*s);
        }
    }
    used.sort_by_key(|s| md(s));
    let pos = |j: u32| -> Src {
        match caps.get(j as usize).and_then(|s| used.iter().position(|u| u == s)) {
            Some(k) => Src::In(k as u32),
            None => Src::In(j),
        }
    };
    for r in &mut node.regions {
        *r = map_inputs(r, &pos);
    }
    node.ins.truncate(1);
    node.ins.extend(used);
}

/// Remove the carried values of an Iterate that neither region uses and
/// both return unchanged; the Iterate's output for one becomes its initial value.
fn c7_carried(b: &Body) -> Body {
    let mut b = b.clone();
    loop {
        let mut found = None;
        'nodes: for (i, n) in b.nodes.iter().enumerate() {
            let (first, ts) = match &n.attrs {
                Attr::Count(ts) => (1, ts),
                Attr::Measure(x, y, ts) => {
                    let _ = (x, y);
                    (0, ts)
                }
                _ => continue,
            };
            if n.regions.len() != 2 || n.ins.len() != first + ts.len() {
                continue;
            }
            for k in 0..ts.len() {
                if let Attr::Measure(x, y, _) = n.attrs
                    && (x as usize == k || y as usize == k)
                {
                    continue;
                }
                let keep = n.regions.iter().any(|r| r.out.get(1 + k) != Some(&Src::In(k as u32)) || uses_input(r, k as u32, Some(1 + k)));
                if !keep {
                    found = Some((i, k, first));
                    break 'nodes;
                }
            }
        }
        let Some((i, k, first)) = found else { return b };
        let init = b.nodes[i].ins[first + k];
        let n = &mut b.nodes[i];
        match &mut n.attrs {
            Attr::Count(ts) => {
                ts.remove(k);
            }
            Attr::Measure(x, y, ts) => {
                ts.remove(k);
                if *x as usize > k {
                    *x -= 1;
                }
                if *y as usize > k {
                    *y -= 1;
                }
            }
            _ => {}
        }
        n.ins.remove(first + k);
        for r in &mut n.regions {
            r.out.remove(1 + k);
            *r = map_inputs(r, &|j| if j as usize > k { Src::In(j - 1) } else { Src::In(j) });
        }
        let it = i as u32;
        b = rewire(
            &b,
            &|s| match s {
                Src::Out(x, p) if x == it && p as usize == k => init,
                Src::Out(x, p) if x == it && p as usize > k => Src::Out(x, p - 1),
                s => s,
            },
            &vec![false; b.nodes.len()],
        );
    }
}

fn c7_body(b: &Body) -> R<Body> {
    let mut nb = c7_carried(b);
    for n in &mut nb.nodes {
        for r in &mut n.regions {
            *r = c7_body(r)?;
        }
    }
    Ok(md_region(&nb, &c7_hook)?.0)
}

fn c7(m: &Module) -> R<Module> {
    each_fn(m, &|_, _, f| Ok(Func { body: c7_body(&f.body)?, ..f.clone() }))
}

// ---- C4 sharing

/// A region's scope: its body, the Branch it is an arm of, and the scope around it.
pub struct Scope<'a> {
    pub body: &'a Body,
    pub arm_of: Option<&'a Node>,
    pub parent: Option<&'a Scope<'a>>,
}

/// The integer literal a source holds, following captures outward.
pub fn lit_of(sc: &Scope, s: &Src) -> Option<i128> {
    match s {
        Src::Out(i, 0) => match &sc.body.nodes.get(*i as usize)?.attrs {
            Attr::Lit(_, Lit::Int(v)) => Some(*v),
            _ => None,
        },
        Src::In(j) => {
            let b = sc.arm_of?;
            lit_of(sc.parent?, b.ins.get(1 + *j as usize)?)
        }
        _ => None,
    }
}

/// The types of a region's sources: of its inputs, and of each node's outputs.
pub fn source_types(tx: &Tx, group: &str, sc: &Scope) -> Vec<Option<Vec<BTy>>> {
    sc.body
        .nodes
        .iter()
        .map(|n| {
            let lits: Vec<Option<i128>> = n.ins.iter().map(|s| lit_of(sc, s)).collect();
            node_outs(tx, group, n, &lits)
        })
        .collect()
}

fn src_ty(s: &Src, ins: &[Option<BTy>], outs: &[Option<Vec<BTy>>]) -> Option<BTy> {
    match s {
        Src::In(j) => ins.get(*j as usize).cloned().flatten(),
        Src::Out(i, p) => outs.get(*i as usize)?.as_ref()?.get(*p as usize).cloned(),
    }
}

/// The input types of the regions of a node.
fn region_ins(n: &Node, ins: &[Option<BTy>], outs: &[Option<Vec<BTy>>]) -> Vec<Vec<Option<BTy>>> {
    match &n.attrs {
        Attr::Branch(..) => {
            let caps: Vec<Option<BTy>> = n.ins.iter().skip(1).map(|s| src_ty(s, ins, outs)).collect();
            n.regions.iter().map(|_| caps.clone()).collect()
        }
        Attr::Count(ts) | Attr::Measure(_, _, ts) => n.regions.iter().map(|_| ts.iter().cloned().map(Some).collect()).collect(),
        _ => n.regions.iter().map(|_| vec![]).collect(),
    }
}

fn c4_body(tx: &Tx, group: &str, b: &Body, ins: &[Option<BTy>], parent: Option<(&Scope, &Node)>) -> R<Body> {
    // the regions inside first, each in its own scope
    let sc = Scope { body: b, arm_of: parent.map(|p| p.1), parent: parent.map(|p| p.0) };
    let outs = source_types(tx, group, &sc);
    let mut nb = b.clone();
    for (i, n) in b.nodes.iter().enumerate() {
        let rins = region_ins(n, ins, &outs);
        for (k, (r, ri)) in n.regions.iter().zip(rins).enumerate() {
            let arm = if matches!(n.attrs, Attr::Branch(..)) { Some((&sc, n)) } else { None };
            nb.nodes[i].regions[k] = c4_body(tx, group, r, &ri, arm)?;
        }
    }
    let (_, _, md) = md_region(&nb, &|_, _| {})?;
    let plain = |t: &Option<BTy>| t.as_ref().is_some_and(|t| !tx.owned(t));
    let mut first: HashMap<&str, u32> = HashMap::new();
    let mut rep: Vec<Option<u32>> = vec![None; nb.nodes.len()];
    for (i, n) in b.nodes.iter().enumerate() {
        let ok = n.ins.iter().all(|s| plain(&src_ty(s, ins, &outs))) && outs[i].as_ref().is_some_and(|o| o.iter().all(|t| !tx.owned(t)));
        if !ok || matches!(n.attrs, Attr::Halt(_)) {
            continue;
        }
        match first.get(md[i].as_str()) {
            Some(r) => rep[i] = Some(*r),
            None => {
                first.insert(&md[i], i as u32);
            }
        }
    }
    let gone: Vec<bool> = rep.iter().map(|r| r.is_some()).collect();
    Ok(rewire(
        &nb,
        &|s| match s {
            Src::Out(i, p) => match rep.get(i as usize).copied().flatten() {
                Some(r) => Src::Out(r, p),
                None => s,
            },
            s => s,
        },
        &gone,
    ))
}

fn c4(m: &Module) -> R<Module> {
    each_fn(m, &|tx, g, f| {
        let ins: Vec<Option<BTy>> = f.ins.iter().cloned().map(Some).collect();
        Ok(Func { body: c4_body(tx, g, &f.body, &ins, None)?, ..f.clone() })
    })
}

// ---- C11 guarded primitives inline

fn has_rec(b: &Body) -> bool {
    b.nodes.iter().any(|n| matches!(n.attrs.refs(), Some(Ref::Rec(_))) || n.regions.iter().any(has_rec))
}

/// The arm input j of a Branch, as the outer source it captures.
fn outer(branch: &Node, s: &Src) -> Option<Src> {
    match s {
        Src::In(j) => branch.ins.get(1 + *j as usize).copied(),
        _ => None,
    }
}

/// Whether a function's body is exactly a partial primitive's domain test,
/// the Branch around the primitive and a Halt (C11).
pub fn guarded_shape(f: &Func) -> bool {
    if f.bound != Bound::None || has_rec(&f.body) {
        return false;
    }
    let b = &f.body;
    let mut test = None;
    let mut branch = None;
    for (i, n) in b.nodes.iter().enumerate() {
        match &n.attrs {
            Attr::Lit(..) => {}
            Attr::Prim(..) if test.is_none() => test = Some(i),
            Attr::Branch(2, _) if branch.is_none() => branch = Some(i),
            _ => return false,
        }
    }
    let (Some(t), Some(bi)) = (test, branch) else { return false };
    let (tn, bn) = (&b.nodes[t], &b.nodes[bi]);
    let Attr::Prim(tid, targs, None) = &tn.attrs else { return false };
    let Attr::Branch(_, outs) = &bn.attrs else { return false };
    if bn.ins.first() != Some(&Src::Out(t as u32, 0)) || bn.regions.len() != 2 {
        return false;
    }
    let want: Vec<Src> = (0..outs.len() as u32).map(|p| Src::Out(bi as u32, p)).collect();
    if b.out != want {
        return false;
    }
    let arm0 = &bn.regions[0];
    let [pn] = &arm0.nodes[..] else { return false };
    let Attr::Prim(pid, pargs, None) = &pn.attrs else { return false };
    let Some(p) = catalogue::entry(pid) else { return false };
    if !p.domain.as_ref().is_some_and(|d| d.1.iter().any(|x| x == tid)) || pargs != targs {
        return false;
    }
    let mut ops: Vec<Option<Src>> = pn.ins.iter().map(|s| outer(bn, s)).collect();
    let mut tops: Vec<Option<Src>> = tn.ins.iter().copied().map(Some).collect();
    if p.symmetric.len() == 2 {
        ops.sort();
        tops.sort();
    }
    if ops != tops {
        return false;
    }
    let arm1 = &bn.regions[1];
    let Some(last) = arm1.nodes.last() else { return false };
    matches!(last.attrs, Attr::Halt(_))
        && arm1.nodes[..arm1.nodes.len() - 1].iter().all(|n| matches!(n.attrs, Attr::Lit(..) | Attr::Prim(..) | Attr::Call(_)))
}

fn c11_body(tx: &Tx, group: &str, b: &Body) -> Body {
    let mut b = b.clone();
    for n in &mut b.nodes {
        for r in &mut n.regions {
            *r = c11_body(tx, group, r);
        }
    }
    loop {
        let found = b.nodes.iter().enumerate().find_map(|(i, n)| match &n.attrs {
            Attr::Call(r @ Ref::Key(..)) => {
                let f = tx.callee(group, r)?;
                (f.ins.len() == n.ins.len() && guarded_shape(f)).then_some((i, f))
            }
            _ => None,
        });
        let Some((ci, f)) = found else { return b };
        let off = b.nodes.len() as u32;
        let call_ins = b.nodes[ci].ins.clone();
        let inl = |s: &Src| match s {
            Src::In(j) => call_ins[*j as usize],
            Src::Out(i, p) => Src::Out(off + i, *p),
        };
        for n in &f.body.nodes {
            b.nodes.push(Node { attrs: n.attrs.clone(), ins: n.ins.iter().map(inl).collect(), regions: n.regions.clone() });
        }
        let outs: Vec<Src> = f.body.out.iter().map(inl).collect();
        let mut gone = vec![false; b.nodes.len()];
        gone[ci] = true;
        let c = ci as u32;
        b = rewire(
            &b,
            &|s| match s {
                Src::Out(x, p) if x == c => outs.get(p as usize).copied().unwrap_or(s),
                s => s,
            },
            &gone,
        );
    }
}

fn c11(m: &Module) -> R<Module> {
    each_fn(m, &|tx, g, f| Ok(Func { body: c11_body(tx, g, &f.body), ..f.clone() }))
}

// ---- C8 declared types

fn fty_with(t: &FTy, d: &dyn Fn(u32) -> J) -> J {
    match t {
        FTy::Decl(i) => json!(["decl", d(*i)]),
        FTy::Tup(xs) => {
            let mut v = vec![J::from("tup")];
            v.extend(xs.iter().map(|x| fty_with(x, d)));
            J::Array(v)
        }
        FTy::Arr(x, n) => json!(["arr", fty_with(x, d), n.to_string()]),
        FTy::List(x) => json!(["list", fty_with(x, d)]),
        FTy::Set(x) => json!(["set", fty_with(x, d)]),
        FTy::Map(k, v) => json!(["map", fty_with(k, d), fty_with(v, d)]),
        FTy::Opt(x) => json!(["opt", fty_with(x, d)]),
        FTy::Chan(x) => json!(["chan", fty_with(x, d)]),
        FTy::Res(x, es) => json!(["res", fty_with(x, d), es]),
        t => fty_json(t),
    }
}

fn decl_with(t: &TypeDecl, d: &dyn Fn(u32) -> J) -> J {
    match t {
        TypeDecl::Record(fs) => json!(["record", fs.iter().map(|f| fty_with(f, d)).collect::<Vec<_>>()]),
        TypeDecl::Kind(vs) => json!(["kind", vs.iter().map(|fs| fs.iter().map(|f| fty_with(f, d)).collect::<Vec<_>>()).collect::<Vec<_>>()]),
        TypeDecl::Refined(f, r) => json!(["refined", fty_with(f, d), ref_json(r)]),
    }
}

/// An iterated digest over members that refer to each other by index: in
/// round 0 each reference is written "?"; in each later round it is the
/// digest of the member it names in the round before; after as many rounds
/// as there are members, two members have one digest exactly when they are
/// bisimilar. Returns, for each new position, the member written there, and
/// for each member its new position.
/// A digest of member i, with each reference to member j written as `d(j)`.
type Digest<'a> = &'a dyn Fn(usize, &dyn Fn(u32) -> J) -> String;

fn iterated(n: usize, digest: Digest) -> (Vec<usize>, Vec<u32>) {
    let mut d: Vec<String> = (0..n).map(|i| digest(i, &|_| json!("?"))).collect();
    for _ in 1..n {
        let prev = d.clone();
        d = (0..n).map(|i| digest(i, &|j| json!(prev.get(j as usize).cloned().unwrap_or_default()))).collect();
    }
    let mut distinct: Vec<&String> = d.iter().collect();
    distinct.sort();
    distinct.dedup();
    let class: Vec<u32> = d.iter().map(|x| distinct.binary_search(&x).unwrap() as u32).collect();
    let order = (0..distinct.len() as u32).map(|j| class.iter().position(|c| *c == j).unwrap()).collect();
    (order, class)
}

fn bty_map(t: &BTy, f: &dyn Fn(u32) -> u32) -> BTy {
    match t {
        BTy::Decl(i) => BTy::Decl(f(*i)),
        BTy::Arr(x) => BTy::Arr(Box::new(bty_map(x, f))),
        BTy::List(x) => BTy::List(Box::new(bty_map(x, f))),
        BTy::Tup(xs) => BTy::Tup(xs.iter().map(|x| bty_map(x, f)).collect()),
        BTy::Sum(xs) => BTy::Sum(xs.iter().map(|x| bty_map(x, f)).collect()),
        t => t.clone(),
    }
}

fn fty_map(t: &FTy, f: &dyn Fn(u32) -> u32) -> FTy {
    let b = |x: &FTy| Box::new(fty_map(x, f));
    match t {
        FTy::Decl(i) => FTy::Decl(f(*i)),
        FTy::Tup(xs) => FTy::Tup(xs.iter().map(|x| fty_map(x, f)).collect()),
        FTy::Arr(x, n) => FTy::Arr(b(x), *n),
        FTy::List(x) => FTy::List(b(x)),
        FTy::Set(x) => FTy::Set(b(x)),
        FTy::Map(k, v) => FTy::Map(b(k), b(v)),
        FTy::Opt(x) => FTy::Opt(b(x)),
        FTy::Chan(x) => FTy::Chan(b(x)),
        FTy::Res(x, es) => FTy::Res(b(x), es.clone()),
        t => t.clone(),
    }
}

fn body_types_map(b: &Body, f: &dyn Fn(u32) -> u32) -> Body {
    let bm = |ts: &Vec<BTy>| ts.iter().map(|t| bty_map(t, f)).collect::<Vec<_>>();
    Body {
        nodes: b
            .nodes
            .iter()
            .map(|n| Node {
                attrs: match &n.attrs {
                    Attr::Lit(t, l) => Attr::Lit(bty_map(t, f), l.clone()),
                    Attr::Prim(name, args, c) => Attr::Prim(
                        name.clone(),
                        args.iter()
                            .map(|a| match a {
                                TyArg::Body(t) => TyArg::Body(bty_map(t, f)),
                                TyArg::Frame(t) => TyArg::Frame(fty_map(t, f)),
                            })
                            .collect(),
                        c.clone(),
                    ),
                    Attr::Branch(k, ts) => Attr::Branch(*k, bm(ts)),
                    Attr::Count(ts) => Attr::Count(bm(ts)),
                    Attr::Measure(i, j, ts) => Attr::Measure(*i, *j, bm(ts)),
                    Attr::Halt(ts) => Attr::Halt(bm(ts)),
                    a => a.clone(),
                },
                ins: n.ins.clone(),
                regions: n.regions.iter().map(|r| body_types_map(r, f)).collect(),
            })
            .collect(),
        out: b.out.clone(),
    }
}

fn c8(m: &Module) -> R<Module> {
    let n = m.types.len();
    if n == 0 {
        return Ok(m.clone());
    }
    let (order, class) = iterated(n, &|i, d| sha(&decl_with(&m.types[i], d)));
    let f = |i: u32| class.get(i as usize).copied().unwrap_or(i);
    let fm = |ts: &Vec<FTy>| ts.iter().map(|t| fty_map(t, &f)).collect::<Vec<_>>();
    let mut out = m.clone();
    out.types = order
        .iter()
        .map(|i| match &m.types[*i] {
            TypeDecl::Record(fs) => TypeDecl::Record(fm(fs)),
            TypeDecl::Kind(vs) => TypeDecl::Kind(vs.iter().map(fm).collect()),
            TypeDecl::Refined(t, r) => TypeDecl::Refined(fty_map(t, &f), r.clone()),
        })
        .collect();
    out.regions = m.regions.iter().map(|r| RegionDecl { actor: r.actor, fields: r.fields.iter().map(|(t, v)| (fty_map(t, &f), v.clone())).collect() }).collect();
    out.events = m.events.iter().map(fm).collect();
    out.errors = m.errors.iter().map(fm).collect();
    for g in out.defs.values_mut() {
        for fun in &mut g.fns {
            fun.ins = fun.ins.iter().map(|t| bty_map(t, &f)).collect();
            fun.outs = fun.outs.iter().map(|t| bty_map(t, &f)).collect();
            fun.body = body_types_map(&fun.body, &f);
        }
    }
    Ok(refresh_keys(&out, &HashMap::new()))
}

// ---- C9 recursive groups

fn subst_rec(j: &J, d: &dyn Fn(u32) -> J) -> J {
    match j {
        J::Array(a) if a.len() == 2 && a[0] == "rec" && a[1].is_u64() => json!(["rec", d(a[1].as_u64().unwrap() as u32)]),
        J::Array(a) => J::Array(a.iter().map(|x| subst_rec(x, d)).collect()),
        J::Object(o) => J::Object(o.iter().map(|(k, v)| (k.clone(), subst_rec(v, d))).collect()),
        x => x.clone(),
    }
}

fn map_refs(b: &Body, f: &dyn Fn(&Ref) -> Ref) -> Body {
    Body {
        nodes: b
            .nodes
            .iter()
            .map(|n| {
                let mut attrs = n.attrs.clone();
                if let Some(r) = attrs.refs_mut() {
                    *r = f(r);
                }
                Node { attrs, ins: n.ins.clone(), regions: n.regions.iter().map(|r| map_refs(r, f)).collect() }
            })
            .collect(),
        out: b.out.clone(),
    }
}

fn c9(m: &Module) -> R<Module> {
    let mut out = m.clone();
    let mut pos: HashMap<(String, u32), u32> = HashMap::new();
    for (k, g) in &m.defs {
        let n = g.fns.len();
        if n < 2 {
            continue;
        }
        let jsons: Vec<J> = g.fns.iter().map(fn_json).collect();
        let (order, class) = iterated(n, &|i, d| sha(&subst_rec(&jsons[i], d)));
        let f = |r: &Ref| match r {
            Ref::Rec(i) => Ref::Rec(class.get(*i as usize).copied().unwrap_or(*i)),
            r => r.clone(),
        };
        let fns = order.iter().map(|i| Func { body: map_refs(&g.fns[*i].body, &f), ..g.fns[*i].clone() }).collect();
        out.defs.insert(k.clone(), Group { fns });
        for (i, c) in class.iter().enumerate() {
            pos.insert((k.clone(), i as u32), *c);
        }
    }
    Ok(refresh_keys(&out, &pos))
}

// ---- keys

fn body_keys(b: &Body, out: &mut Vec<String>) {
    for n in &b.nodes {
        if let Some(Ref::Key(k, _)) = n.attrs.refs() {
            out.push(k.clone());
        }
        for r in &n.regions {
            body_keys(r, out);
        }
    }
}

/// The keys a group names.
pub fn group_refs(g: &Group) -> Vec<String> {
    let mut v = vec![];
    for f in &g.fns {
        body_keys(&f.body, &mut v);
    }
    v.sort();
    v.dedup();
    v
}

/// Bring every group key and every reference up to date with the groups'
/// contents, callees before callers; `pos` moves function i of a group to
/// a new position (C9).
pub fn refresh_keys(m: &Module, pos: &HashMap<(String, u32), u32>) -> Module {
    let mut order = vec![];
    let mut state: BTreeMap<&str, u8> = BTreeMap::new();
    fn visit<'a>(m: &'a Module, k: &'a str, st: &mut BTreeMap<&'a str, u8>, order: &mut Vec<&'a str>) {
        if st.get(k).is_some() {
            return;
        }
        st.insert(k, 1);
        if let Some(g) = m.defs.get(k) {
            for f in &g.fns {
                let mut ks = vec![];
                body_keys(&f.body, &mut ks);
                for c in ks {
                    if let Some((ck, _)) = m.defs.get_key_value(&c) {
                        visit(m, ck, st, order);
                    }
                }
            }
        }
        st.insert(k, 2);
        order.push(k);
    }
    for k in m.defs.keys() {
        visit(m, k, &mut state, &mut order);
    }
    let mut newk: HashMap<String, String> = HashMap::new();
    let mut defs = BTreeMap::new();
    for k in order {
        let g = &m.defs[k];
        let fix = |r: &Ref| match r {
            Ref::Key(c, i) => Ref::Key(newk.get(c).cloned().unwrap_or(c.clone()), pos.get(&(c.clone(), *i)).copied().unwrap_or(*i)),
            r => r.clone(),
        };
        let ng = Group { fns: g.fns.iter().map(|f| Func { body: map_refs(&f.body, &fix), ..f.clone() }).collect() };
        let nk = group_key(&ng);
        newk.insert(k.to_string(), nk.clone());
        defs.insert(nk, ng);
    }
    let fix = |r: &Ref| match r {
        Ref::Key(c, i) => Ref::Key(newk.get(c).cloned().unwrap_or(c.clone()), pos.get(&(c.clone(), *i)).copied().unwrap_or(*i)),
        r => r.clone(),
    };
    let mut out = m.clone();
    out.defs = defs;
    for s in &mut out.steps {
        s.guard = s.guard.as_ref().map(fix);
        s.body = fix(&s.body);
    }
    out.main = out.main.as_ref().map(fix);
    for t in &mut out.types {
        if let TypeDecl::Refined(_, r) = t {
            *r = fix(r);
        }
    }
    out
}

/// The module key: the SHA-256 of the module's canonical JSON.
pub fn module_key(m: &Module) -> String {
    crate::interp::d_sha(&encode(m))
}

/// The passes, in the order they run, each named by its rule.
type Pass = fn(&Module) -> R<Module>;

const PASSES: [(&str, Pass); 8] =
    [("C8", c8), ("C11", c11), ("C6", c6), ("C7", c7), ("C4", c4), ("C3", c3), ("C5", c5), ("C9", c9)];

/// The canonical form of a module, and the rule of the first pass that changed it.
pub fn canonical(m: &Module) -> R<(Module, Option<&'static str>)> {
    let mut cur = m.clone();
    let mut first = None;
    for _ in 0..64 {
        let mut changed = false;
        for (name, pass) in PASSES {
            let next = pass(&cur)?;
            if module_json(&next) != module_json(&cur) {
                changed = true;
                first.get_or_insert(name);
                cur = next;
            }
        }
        if !changed {
            return Ok((cur, first));
        }
    }
    Err(Defer("the canonical passes did not reach a fixed point".into()))
}
