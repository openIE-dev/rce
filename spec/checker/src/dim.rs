//! `rce dim` (spec 6.10, 10.11): the number of behaviours a program can be
//! in and the redundancy of its state.
//!
//! Exploration: breadth-first from s₀ over every declared event, with
//! payloads drawn from the boundary set of 6.3 per field type (the T35
//! refinement fragment is not read yet: a refined field gets its base type's
//! boundary set). Partition refinement: Moore's iterative algorithm on the
//! explored graph with the per-step output of λ as the initial partition.
//! It is quadratic in the worst case where Hopcroft is n log n; it is kept
//! because its order of operations is obviously deterministic and the graphs
//! `--fuel` admits are small. States not explored (bound reached) share one
//! class, so the count is then a lower bound on dim(P), never an upper one.
//!
//! Status: written against the interpreter API (`Interp::new`,
//! `initial_state`, `construct`, `delta`, `state_json`); not yet built or
//! run in the session that wrote it. `--control` is reserved, not
//! implemented (W-RES-1 wording): the symbolic quotient of 6.10 needs the
//! refinement checker of T35 wired to a predicate-equivalence oracle.

use crate::ast::{Arg, Expr, Item, Module, Ty};
use crate::interp::{state_json, why, Interp, Region, V};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// One concrete event instance: its name and the literal arguments used to
/// construct it, kept so the report can name a distinguishing event.
#[derive(Clone)]
struct EvInst {
    name: String,
    args: Vec<Arg>,
    show: String,
}

/// The result of `rce dim` on one program.
pub struct Dim {
    pub states: usize,
    pub bound_reached: bool,
    pub behaviours: usize,
    pub redundancy_bits: f64,
    pub regions: Vec<(String, usize, usize, f64)>, // name, distinct values, needed values, bits
    pub events_skipped: Vec<String>,
}

fn lit_int(i: i128) -> Expr {
    Expr::Int(i, 0)
}

/// The boundary set of 6.3 for a payload field type: the values an
/// exploration tries. A type this table does not know yields `None` and the
/// event is skipped and reported.
fn boundary(t: &Ty) -> Option<Vec<Expr>> {
    match t {
        Ty::Path(p, args) if args.is_empty() && p.len() == 1 => match p[0].as_str() {
            "bool" => Some(vec![Expr::Bool(false, 0), Expr::Bool(true, 0)]),
            "text" => Some(vec![Expr::Str(String::new(), 0), Expr::Str("a".into(), 0)]),
            "i8" => Some(vec![lit_int(-128), lit_int(-1), lit_int(0), lit_int(1), lit_int(127)]),
            "i16" => Some(vec![lit_int(-32768), lit_int(-1), lit_int(0), lit_int(1), lit_int(32767)]),
            "i32" => Some(vec![lit_int(-(1 << 31)), lit_int(-1), lit_int(0), lit_int(1), lit_int((1 << 31) - 1)]),
            "i64" => Some(vec![lit_int(i64::MIN as i128), lit_int(-1), lit_int(0), lit_int(1), lit_int(i64::MAX as i128)]),
            "u8" => Some(vec![lit_int(0), lit_int(1), lit_int(255)]),
            "u16" => Some(vec![lit_int(0), lit_int(1), lit_int(65535)]),
            "u32" => Some(vec![lit_int(0), lit_int(1), lit_int((1u128 << 32) as i128 - 1)]),
            "u64" | "usize" => Some(vec![lit_int(0), lit_int(1), lit_int(u64::MAX as i128)]),
            "f64" | "f32" => Some(vec![Expr::Float(-1.0, 0), Expr::Float(0.0, 0), Expr::Float(1.0, 0)]),
            _ => None,
        },
        Ty::Unit => Some(vec![Expr::Unit(0)]),
        Ty::Opt(inner) => {
            let mut v = vec![Expr::None(0)];
            for e in boundary(inner)? {
                v.push(Expr::Call(Box::new(Expr::Name("some".into(), 0)), vec![Arg { label: None, e }], 0, false));
            }
            Some(v)
        }
        Ty::Path(p, args) if p.len() == 1 && p[0] == "List" && args.len() == 1 => {
            let mut v = vec![Expr::List(vec![], 0)];
            if let Some(first) = boundary(&args[0]).and_then(|b| b.into_iter().nth(1)) {
                v.push(Expr::List(vec![first], 0));
            }
            Some(v)
        }
        _ => None,
    }
}

/// Every concrete instance of the declared events, in the canonical order of
/// 6.1 (events by name, payloads by the order of their boundary sets).
fn instances(m: &Module) -> (Vec<EvInst>, Vec<String>) {
    let mut out = vec![];
    let mut skipped = vec![];
    let mut decls: Vec<(&String, &Vec<(String, Ty)>)> = m
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Event { name, payload, .. } => Some((name, payload)),
            _ => None,
        })
        .collect();
    decls.sort_by(|a, b| a.0.cmp(b.0));
    'ev: for (name, payload) in decls {
        let mut columns: Vec<Vec<Expr>> = vec![];
        for (_, t) in payload {
            match boundary(t) {
                Some(b) => columns.push(b),
                None => {
                    skipped.push(format!("{name}: no boundary set for a field of type {t:?}"));
                    continue 'ev;
                }
            }
        }
        // Cartesian product, capped so one wide event cannot exhaust the fuel.
        let mut rows: Vec<Vec<Expr>> = vec![vec![]];
        for col in &columns {
            let mut next = vec![];
            for r in &rows {
                for e in col {
                    let mut r2 = r.clone();
                    r2.push(e.clone());
                    next.push(r2);
                }
            }
            rows = next;
            if rows.len() > 4096 {
                rows.truncate(4096);
            }
        }
        for r in rows {
            let args: Vec<Arg> = r
                .iter()
                .zip(payload)
                .map(|(e, (f, _))| Arg { label: Some(f.clone()), e: e.clone() })
                .collect();
            let show = format!("{}({})", name, args.iter().map(|a| expr_show(&a.e)).collect::<Vec<_>>().join(", "));
            out.push(EvInst { name: name.clone(), args, show });
        }
    }
    (out, skipped)
}

fn expr_show(e: &Expr) -> String {
    match e {
        Expr::Int(i, _) => i.to_string(),
        Expr::Float(f, _) => format!("{f:?}"),
        Expr::Str(s, _) => format!("{s:?}"),
        Expr::Bool(b, _) => b.to_string(),
        Expr::None(_) => "none".into(),
        Expr::Unit(_) => "()".into(),
        Expr::List(xs, _) => format!("[{}]", xs.iter().map(expr_show).collect::<Vec<_>>().join(", ")),
        Expr::Call(_, a, _, _) => format!("some({})", a.iter().map(|x| expr_show(&x.e)).collect::<Vec<_>>().join(", ")),
        _ => "?".into(),
    }
}

/// The label of one step: the canonical text of its outputs, or its refusal.
fn step_label(r: &Result<(Vec<Region>, Vec<V>), String>) -> String {
    match r {
        Ok((_, outs)) => format!("ok:{}", outs.iter().map(|o| o.show()).collect::<Vec<_>>().join("|")),
        Err(x) => format!("refused:{x}"),
    }
}

/// Explore and refine. `fuel` bounds the number of explored states.
pub fn dim(m: &Module, fuel: usize) -> Result<Dim, String> {
    let mut it = Interp::new(m);
    let s0 = it.initial_state().map_err(why)?;
    let (evs, skipped) = instances(m);
    if evs.is_empty() {
        return Err("no event has a boundary set; nothing to explore".into());
    }
    // Build the event values once; `construct` reads the scopes of a fresh frame.
    let mut evals: Vec<(EvInst, V)> = vec![];
    for e in &evs {
        it.scopes = vec![HashMap::new()];
        match it.construct(&e.name, &e.args) {
            Ok(Some(v)) => evals.push((e.clone(), v)),
            Ok(None) => return Err(format!("`{}` is not an event", e.name)),
            Err(f) => return Err(format!("{}: {}", e.show, why(f))),
        }
    }
    let k = evals.len();

    // Breadth-first exploration. Key = canonical JSON of the state (6.1).
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut states: Vec<Vec<Region>> = vec![];
    let mut succ: Vec<Vec<Option<usize>>> = vec![]; // None = unexplored (bound reached)
    let mut label: Vec<Vec<String>> = vec![];
    let mut queue = VecDeque::new();
    let key0 = state_json(&s0, m);
    index.insert(key0, 0);
    states.push(s0);
    queue.push_back(0usize);
    let mut bound_reached = false;
    while let Some(i) = queue.pop_front() {
        let s = states[i].clone();
        let mut row = Vec::with_capacity(k);
        let mut lab = Vec::with_capacity(k);
        for (_, ev) in &evals {
            it.scopes = vec![HashMap::new()];
            let r = it.delta(&s, ev);
            lab.push(step_label(&r));
            match r {
                Ok((s2, _)) => {
                    let key = state_json(&s2, m);
                    if let Some(&j) = index.get(&key) {
                        row.push(Some(j));
                    } else if states.len() < fuel {
                        let j = states.len();
                        index.insert(key, j);
                        states.push(s2);
                        queue.push_back(j);
                        row.push(Some(j));
                    } else {
                        bound_reached = true;
                        row.push(None);
                    }
                }
                Err(_) => row.push(Some(i)), // T8: a refused event leaves the state unchanged
            }
        }
        succ.push(row);
        label.push(lab);
    }
    let n = states.len();

    // Moore refinement: class = (output labels, successor classes); iterate to a fixpoint.
    // The unexplored sink is class `n` and never splits.
    let mut class: Vec<usize> = {
        let mut sig: BTreeMap<Vec<String>, usize> = BTreeMap::new();
        label.iter().map(|l| {
            let next = sig.len();
            *sig.entry(l.clone()).or_insert(next)
        }).collect()
    };
    loop {
        let mut sig: BTreeMap<(usize, Vec<usize>), usize> = BTreeMap::new();
        let next: Vec<usize> = (0..n)
            .map(|i| {
                let key = (class[i], succ[i].iter().map(|t| t.map_or(n, |j| class[j])).collect::<Vec<_>>());
                let c = sig.len();
                *sig.entry(key).or_insert(c)
            })
            .collect();
        let before = class.iter().collect::<std::collections::BTreeSet<_>>().len();
        let after = sig.len();
        class = next;
        if after == before {
            break;
        }
    }
    let behaviours = class.iter().collect::<std::collections::BTreeSet<_>>().len();
    let bits = |x: usize| if x == 0 { 0.0 } else { (x as f64).log2() };
    let redundancy_bits = bits(n) - bits(behaviours);

    // Per region: distinct values seen, and distinct values among one
    // representative per behaviour class (the smallest state key in the class).
    let mut rep: BTreeMap<usize, usize> = BTreeMap::new();
    for i in 0..n {
        let c = class[i];
        let better = match rep.get(&c) {
            None => true,
            Some(&r) => state_json(&states[i], m) < state_json(&states[r], m),
        };
        if better {
            rep.insert(c, i);
        }
    }
    let region_names: Vec<String> = states[0].iter().map(|r| r.0.clone()).collect();
    let mut regions = vec![];
    for (ri, rn) in region_names.iter().enumerate() {
        let val = |s: &Vec<Region>| state_json(std::slice::from_ref(&s[ri]), m);
        let all: std::collections::BTreeSet<String> = states.iter().map(val).collect();
        let needed: std::collections::BTreeSet<String> = rep.values().map(|&i| val(&states[i])).collect();
        regions.push((rn.clone(), all.len(), needed.len(), bits(all.len()) - bits(needed.len())));
    }
    Ok(Dim { states: n, bound_reached, behaviours, redundancy_bits, regions, events_skipped: skipped })
}

/// The text report of 10.11.
pub fn report(d: &Dim) -> String {
    let mut s = String::new();
    s.push_str(&format!("states explored     {:<8} bound reached: {}\n", d.states, if d.bound_reached { "yes" } else { "no" }));
    s.push_str(&format!(
        "behaviours          {:<8} dim(P) ({})\n",
        d.behaviours,
        if d.bound_reached { "lower bound" } else { "exact" }
    ));
    s.push_str(&format!("redundancy          {:<8.2} bits: log2(states) - log2(behaviours)\n", d.redundancy_bits));
    for (n, all, needed, b) in &d.regions {
        s.push_str(&format!("  region {:<12} {:<8.2} ({all} values, {needed} distinguish behaviours)\n", n, b));
    }
    for e in &d.events_skipped {
        s.push_str(&format!("skipped             {e}\n"));
    }
    s
}

/// The JSON report (`rce-dim/1`, compact, sorted keys).
pub fn report_json(d: &Dim) -> String {
    let v = serde_json::json!({
        "format": "rce-dim/1",
        "states": d.states,
        "bound_reached": d.bound_reached,
        "behaviours": d.behaviours,
        "redundancy_bits": d.redundancy_bits,
        "regions": d.regions.iter().map(|(n, a, b, bits)| serde_json::json!({"name": n, "values": a, "needed": b, "bits": bits})).collect::<Vec<_>>(),
        "skipped": d.events_skipped,
    });
    crate::interp::canon_json(&v)
}
