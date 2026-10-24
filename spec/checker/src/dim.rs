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
//! run in the session that wrote it; later built and run (2026-10-23).
//!
//! `--control` computes the control quotient of 6.10 on the explored
//! graph: the same states and transitions, with each step labelled only by
//! its control outcome (which regions changed, whether and how it was
//! refused, the constructors of what it emitted) instead of the values.
//! That is a coarser output function on the same deterministic machine, so
//! its minimal size dim_c is at most dim, exactly as 6.10 states, and the
//! predicate-equivalence oracle is concrete evaluation on reach(P): exact
//! on the explored set, a lower bound when the fuel bound is reached. The
//! canonical control string numbers the quotient's states by first visit
//! in breadth-first order from s0 with events in canonical order
//! (Reis, Moreira and Almeida 2005) and `shape_ctl` is its SHA-256.

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
    /// `--control`: dim_c and the hash of the canonical control string
    pub control: Option<(usize, String)>,
    /// Some explored step has a λ output (6.1): an emitted value, or the
    /// record of a refusal. A program with no such step has one behaviour by
    /// definition (6.10): its state is in D and readable, but no future
    /// output depends on it.
    pub has_output: bool,
    /// Some explored step emits a value.
    pub emits: bool,
    /// Some explored step refuses its event.
    pub refuses: bool,
    /// Distinct output sequences over the explored steps (the text of a
    /// step's emitted values, or its refusal; a silent step is excluded): 1
    /// means every output is the same, so the program has one behaviour as
    /// well.
    pub distinct_outputs: usize,
}

fn lit_int(i: i128) -> Expr {
    Expr::Int(i, 0)
}

/// The boundary set of 6.3 for a payload field type: the values an
/// exploration tries. A type this table does not know yields `None` and the
/// event is skipped and reported.
fn boundary(t: &Ty, aliases: &HashMap<String, Ty>) -> Option<Vec<Expr>> {
    match t {
        Ty::Path(p, args) if args.is_empty() && p.len() == 1 && aliases.contains_key(&p[0]) => boundary(&aliases[&p[0]], aliases),
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
            for e in boundary(inner, aliases)? {
                v.push(Expr::Call(Box::new(Expr::Name("some".into(), 0)), vec![Arg { label: None, e }], 0, false));
            }
            Some(v)
        }
        Ty::Path(p, args) if p.len() == 1 && p[0] == "List" && args.len() == 1 => {
            let mut v = vec![Expr::List(vec![], 0)];
            if let Some(first) = boundary(&args[0], aliases).and_then(|b| b.into_iter().nth(1)) {
                v.push(Expr::List(vec![first], 0));
            }
            Some(v)
        }
        _ => None,
    }
}

/// Every concrete instance of the declared events, in declaration order
/// (the name-free key of 6.6), payloads in the order of their boundary sets.
fn instances(m: &Module) -> (Vec<EvInst>, Vec<String>) {
    let mut out = vec![];
    let mut skipped = vec![];
    // type aliases, refined or not (5.1): a refined field gets its base type's
    // boundary set, and the refusal of an out-of-range value is an observed step
    let aliases: HashMap<String, Ty> = m
        .items
        .iter()
        .filter_map(|i| match i {
            Item::TypeAlias { name, generics, ty, .. } if generics.is_empty() => Some((name.clone(), ty.clone())),
            _ => None,
        })
        .collect();
    // Declaration order, never name order: the exploration order is part of
    // the canonical control string and must not change under renaming (T61).
    let decls: Vec<(&String, &Vec<(String, Ty)>)> = m
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Event { name, payload, .. } => Some((name, payload)),
            _ => None,
        })
        .collect();
    'ev: for (name, payload) in decls {
        let mut columns: Vec<Vec<Expr>> = vec![];
        for (_, t) in payload {
            match boundary(t, &aliases) {
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

/// Declaration index of every named item, in declaration order: the
/// name-free key of 6.6 for control labels, so that `shape_ctl` does not
/// depend on names (T61, T7).
fn decl_index(m: &Module) -> HashMap<String, usize> {
    // index within kind: the k-th error, the k-th struct, the k-th kind
    // variant; an `intent` or a function elsewhere in the file must not shift
    // the key of an error (the labels must agree across projections and lifts)
    let mut idx = HashMap::new();
    let (mut errors, mut structs, mut variants) = (0usize, 0usize, 0usize);
    for it in &m.items {
        match it {
            Item::Error { name, .. } => {
                idx.entry(name.clone()).or_insert(errors);
                errors += 1;
            }
            Item::Struct { name, .. } => {
                idx.entry(name.clone()).or_insert(structs);
                structs += 1;
            }
            Item::Kind { variants: vs, .. } => {
                for (vn, _) in vs {
                    idx.entry(vn.clone()).or_insert(variants);
                    variants += 1;
                }
            }
            _ => {}
        }
    }
    idx
}

/// The control label of one step (6.10 control projection): the refusal's
/// error (by declaration index) if refused, else the constructors of the
/// emitted values (by declaration index, or a type class); never a data
/// value and never a name. It is a function of the step's output alone, so
/// the control quotient is a coarsening of the behaviour quotient and
/// dim_c <= dim holds by construction. (An earlier draft also recorded
/// which regions changed; a change with no output is invisible to λ, so
/// that label was not a coarsening and the pilot showed dim_c > dim on
/// example 51.)
/// The name-free output label of a step (T7): emitted values in their
/// positional form (6.1), a refusal by the error's declaration index. Two
/// programs are compared on these, so a renaming never distinguishes them.
fn pos_label(m: &Module, idx: &HashMap<String, usize>, r: &Result<(Vec<Region>, Vec<V>), String>) -> String {
    match r {
        Ok((_, outs)) => format!("ok:{}", outs.iter().map(|o| crate::interp::form_value_pos(o, m).json()).collect::<Vec<_>>().join("|")),
        Err(x) => {
            let name = x.split('(').next().unwrap_or(x).trim();
            match idx.get(name) {
                Some(i) => format!("refused:err#{i}"),
                None => format!("refused:{x}"),
            }
        }
    }
}

fn control_label(idx: &HashMap<String, usize>, r: &Result<(Vec<Region>, Vec<V>), String>) -> String {
    match r {
        Ok((_, outs)) => {
            let emitted: Vec<String> = outs
                .iter()
                .map(|o| match o {
                    V::Rec(n, _) => idx.get(n).map_or("rec".to_string(), |i| format!("rec#{i}")),
                    V::Int(..) => "int".into(),
                    V::F(_) | V::F32(_) => "float".into(),
                    V::Bool(_) => "bool".into(),
                    V::Text(_) => "text".into(),
                    _ => "value".into(),
                })
                .collect();
            format!("ok:{}", emitted.join(","))
        }
        Err(x) => {
            let name = x.split('(').next().unwrap_or(x).trim();
            match idx.get(name) {
                Some(i) => format!("refused:err#{i}"),
                None => format!("refused:{name}"), // a trap: its text carries no declared name
            }
        }
    }
}

/// Hopcroft's partition refinement on a deterministic labelled graph
/// (n log n in the number of transitions). Returns the class of each state.
/// `succ[i][j]` is the successor on event j (`None` = unexplored: one shared
/// sink, class index `n`, never split), `label[i][j]` its output label.
/// Classes are numbered in order of first state, so the result is a
/// deterministic function of the input.
fn refine(n: usize, succ: &[Vec<Option<usize>>], label: &[Vec<String>]) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let k = succ[0].len();
    // initial partition: by the full output label vector
    let mut class: Vec<usize> = {
        let mut sig: BTreeMap<&Vec<String>, usize> = BTreeMap::new();
        let mut order = vec![];
        for l in label {
            if !sig.contains_key(l) {
                sig.insert(l, order.len());
                order.push(l);
            }
        }
        label.iter().map(|l| sig[l]).collect()
    };
    // blocks as index lists; the sink is block `n_blocks` conceptually and
    // has no members, so it never needs splitting
    let mut nblocks = class.iter().copied().max().map_or(0, |m| m + 1);
    let mut blocks: Vec<Vec<usize>> = vec![vec![]; nblocks];
    for (i, &c) in class.iter().enumerate() {
        blocks[c].push(i);
    }
    // inverse transitions per event
    let mut pred: Vec<Vec<Vec<usize>>> = vec![vec![vec![]; n]; k];
    for i in 0..n {
        for (j, t) in succ[i].iter().enumerate() {
            if let Some(x) = t {
                pred[j][*x].push(i);
            }
        }
    }
    let mut work: VecDeque<(usize, usize)> = VecDeque::new(); // (block, event)
    for b in 0..nblocks {
        for j in 0..k {
            work.push_back((b, j));
        }
    }
    while let Some((b, j)) = work.pop_front() {
        // states with an event-j successor in block b
        let splitter: std::collections::BTreeSet<usize> = blocks[b].iter().flat_map(|&x| pred[j][x].iter().copied()).collect();
        if splitter.is_empty() {
            continue;
        }
        // which blocks does the splitter cut?
        let mut touched: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &s in &splitter {
            touched.entry(class[s]).or_default().push(s);
        }
        for (c, inside) in touched {
            if inside.len() == blocks[c].len() {
                continue; // whole block is in the splitter: no split
            }
            let inside_set: std::collections::BTreeSet<usize> = inside.iter().copied().collect();
            let (yes, no): (Vec<usize>, Vec<usize>) = blocks[c].iter().partition(|x| inside_set.contains(x));
            // the smaller part becomes a new block (Hopcroft's rule); its
            // members keep deterministic numbering by first state
            let (keep, new) = if yes.len() <= no.len() { (no, yes) } else { (yes, no) };
            let nb = nblocks;
            nblocks += 1;
            for &x in &new {
                class[x] = nb;
            }
            blocks[c] = keep;
            blocks.push(new);
            for e in 0..k {
                work.push_back((nb, e));
            }
        }
    }
    // renumber classes by first occurrence so the output is canonical
    let mut renum: BTreeMap<usize, usize> = BTreeMap::new();
    class
        .iter()
        .map(|&c| {
            let next = renum.len();
            *renum.entry(c).or_insert(next)
        })
        .collect()
}

/// The canonical string of a quotient machine: states renumbered by first
/// visit in breadth-first order from the class of state 0, events in their
/// canonical order; one line per (state, event) with the successor's
/// canonical number and the label. Unique for the isomorphism class.
fn canonical_string(n: usize, class: &[usize], succ: &[Vec<Option<usize>>], label: &[Vec<String>]) -> String {
    // representative concrete state per class: the first explored (BFS order) one
    let mut rep: BTreeMap<usize, usize> = BTreeMap::new();
    for i in 0..n {
        rep.entry(class[i]).or_insert(i);
    }
    let mut number: BTreeMap<usize, usize> = BTreeMap::new();
    let mut order = vec![];
    let mut queue = VecDeque::new();
    number.insert(class[0], 0);
    queue.push_back(class[0]);
    while let Some(c) = queue.pop_front() {
        order.push(c);
        let i = rep[&c];
        for t in &succ[i] {
            if let Some(j) = t {
                let cj = class[*j];
                if !number.contains_key(&cj) {
                    let k = number.len();
                    number.insert(cj, k);
                    queue.push_back(cj);
                }
            }
        }
    }
    let mut s = String::new();
    for c in order {
        let i = rep[&c];
        for (j, t) in succ[i].iter().enumerate() {
            let target = match t {
                Some(x) => number[&class[*x]].to_string(),
                None => "?".into(),
            };
            s.push_str(&format!("{} {} {} {}\n", number[&c], j, target, label[i][j]));
        }
    }
    s
}

/// Explore and refine. `fuel` bounds the number of explored states.
pub fn dim(m: &Module, fuel: usize) -> Result<Dim, String> {
    dim_with(m, fuel, false)
}

/// Breadth-first exploration of a program from its initial state over the
/// boundary instances of its events (10.11), to at most `fuel` states.
pub fn explore(m: &Module, fuel: usize) -> Result<Graph, String> {
    let mut it = Interp::new(m);
    let s0 = it.initial_state().map_err(why)?;
    let (evs, skipped) = instances(m);
    let has_state = m.items.iter().any(|i| matches!(i, Item::State { .. } | Item::Actor { .. }));
    if !has_state {
        return Err("a command (no state region) is one behaviour; nothing to explore".into());
    }
    if evs.is_empty() {
        return Err(format!("no event has a boundary set; nothing to explore ({})", skipped.join("; ")));
    }
    // Build the event values once; `construct` reads the scopes of a fresh frame.
    let mut evals: Vec<(EvInst, V)> = vec![];
    let mut skipped = skipped;
    for e in &evs {
        it.scopes = vec![HashMap::new()];
        match it.construct(&e.name, &e.args) {
            Ok(Some(v)) => evals.push((e.clone(), v)),
            Ok(None) => return Err(format!("`{}` is not an event", e.name)),
            // a boundary value outside a refined type traps at construction
            // (T35): that instance cannot be an event, so it is not explored
            Err(f) => skipped.push(format!("{}: {}", e.show, why(f))),
        }
    }
    if evals.is_empty() {
        return Err(format!("no event instance can be constructed ({})", skipped.join("; ")));
    }
    let k = evals.len();
    let idx = decl_index(m);

    // Breadth-first exploration. Key = canonical JSON of the state (6.1).
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut states: Vec<Vec<Region>> = vec![];
    let mut succ: Vec<Vec<Option<usize>>> = vec![]; // None = unexplored (bound reached)
    let mut label: Vec<Vec<String>> = vec![];
    let mut clabel: Vec<Vec<String>> = vec![];
    let mut plabel: Vec<Vec<String>> = vec![];
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
        let mut clab = Vec::with_capacity(k);
        let mut plab = Vec::with_capacity(k);
        for (_, ev) in &evals {
            it.scopes = vec![HashMap::new()];
            let r = it.delta(&s, ev);
            lab.push(step_label(&r));
            plab.push(pos_label(m, &idx, &r));
            clab.push(control_label(&idx, &r));
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
        clabel.push(clab);
        plabel.push(plab);
    }
    let keys = evals
        .iter()
        .map(|(e, _)| format!("{}({})", crate::interp::event_index(m, &e.name).map(|i| i.to_string()).unwrap_or_else(|| e.name.clone()), e.args.iter().map(|a| expr_show(&a.e)).collect::<Vec<_>>().join(", ")))
        .collect();
    Ok(Graph { states, succ, label, clabel, plabel, bound_reached, skipped, alphabet: evals.iter().map(|(e, _)| e.show.clone()).collect(), keys })
}

/// The explored graph of a program: states keyed by canonical JSON, one
/// successor and one output label per state and event instance.
pub struct Graph {
    pub states: Vec<Vec<Region>>,
    pub succ: Vec<Vec<Option<usize>>>,
    pub label: Vec<Vec<String>>,
    pub clabel: Vec<Vec<String>>,
    /// name-free output labels (T7), the labels `equiv` compares
    pub plabel: Vec<Vec<String>>,
    pub bound_reached: bool,
    pub skipped: Vec<String>,
    /// the event instances explored, in canonical order, as printed
    pub alphabet: Vec<String>,
    /// the same instances with the event named by its declaration index
    /// (T7): the key two programs are compared on
    pub keys: Vec<String>,
}

/// `dim`, optionally with the control quotient (`--control`).
pub fn dim_with(m: &Module, fuel: usize, control: bool) -> Result<Dim, String> {
    let Graph { states, succ, label, clabel, bound_reached, skipped, .. } = explore(m, fuel)?;
    let _ = &states;
    let n = states.len();
    let has_output = label.iter().flatten().any(|l| l != "ok:");
    let emits = label.iter().flatten().any(|l| l.starts_with("ok:") && l != "ok:");
    let refuses = label.iter().flatten().any(|l| l.starts_with("refused:"));
    let distinct_outputs = label.iter().flatten().filter(|l| *l != "ok:").collect::<std::collections::BTreeSet<_>>().len();

    // Moore refinement with the full output labels (dim) and, if asked, the
    // control labels (dim_c), on the same explored graph.
    let class = refine(n, &succ, &label);
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
    let control = if control {
        let cclass = refine(n, &succ, &clabel);
        let dim_c = cclass.iter().collect::<std::collections::BTreeSet<_>>().len();
        let cs = canonical_string(n, &cclass, &succ, &clabel);
        use sha2::{Digest, Sha256};
        let h = Sha256::digest(crate::interp::canon_json(&serde_json::Value::String(cs)).as_bytes());
        Some((dim_c, h.iter().map(|b| format!("{b:02x}")).collect::<String>()))
    } else {
        None
    };
    Ok(Dim { states: n, bound_reached, behaviours, redundancy_bits, regions, events_skipped: skipped, control, has_output, emits, refuses, distinct_outputs })
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
    let yn = |b: bool| if b { "yes" } else { "no" };
    s.push_str(&format!("outputs             {:<8} some explored step has a λ output (emits: {}, refuses: {})\n", yn(d.has_output), yn(d.emits), yn(d.refuses)));
    s.push_str(&format!("distinct outputs    {:<8} distinct output sequences over the explored steps (silent steps excluded)\n", d.distinct_outputs));
    for (n, all, needed, b) in &d.regions {
        s.push_str(&format!("  region {:<12} {:<8.2} ({all} values, {needed} distinguish behaviours)\n", n, b));
    }
    if let Some((c, h)) = &d.control {
        s.push_str(&format!("control states      {:<8} dim_c(P) ({})\n", c, if d.bound_reached { "lower bound" } else { "exact" }));
        s.push_str(&format!("shape_ctl           sha256:{h}\n"));
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
        "has_output": d.has_output,
        "emits": d.emits,
        "refuses": d.refuses,
        "distinct_outputs": d.distinct_outputs,
        "regions": d.regions.iter().map(|(n, a, b, bits)| serde_json::json!({"name": n, "values": a, "needed": b, "bits": bits})).collect::<Vec<_>>(),
        "skipped": d.events_skipped,
        "control_states": d.control.as_ref().map(|c| c.0),
        "shape_ctl": d.control.as_ref().map(|c| c.1.clone()),
    });
    crate::interp::canon_json(&v)
}

/// The result of `rce equiv A B`: behaviour equivalence of two programs on
/// the explored alphabet (6.10, T58), with a shortest distinguishing event
/// sequence when they differ.
pub struct Equiv {
    pub equivalent: bool,
    /// neither exploration reached its bound: the verdict is over the whole
    /// reachable graph of both programs on this alphabet
    pub exact: bool,
    pub states_a: usize,
    pub states_b: usize,
    pub alphabet: Vec<String>,
    /// event instances explored by one program and not the other
    pub only_a: Vec<String>,
    pub only_b: Vec<String>,
    /// (event sequence, A's output at its end, B's output at its end)
    pub witness: Option<(Vec<String>, String, String)>,
}

/// Behaviour equivalence of two programs (6.10): explore both over the same
/// event instances, refine the disjoint union by output labels, and read
/// whether the two initial states fall in one class (T58: equivalence is
/// bisimilarity). The verdict is over the explored alphabet; T59 bounds what
/// trace agreement to a depth proves.
pub fn equiv(a: &Module, b: &Module, fuel: usize) -> Result<Equiv, String> {
    let ga = explore(a, fuel)?;
    let gb = explore(b, fuel)?;
    let sa: std::collections::BTreeSet<&String> = ga.keys.iter().collect();
    let sb: std::collections::BTreeSet<&String> = gb.keys.iter().collect();
    let only_a: Vec<String> = ga.keys.iter().zip(&ga.alphabet).filter(|(k, _)| !sb.contains(k)).map(|(_, s)| s.clone()).collect();
    let only_b: Vec<String> = gb.keys.iter().zip(&gb.alphabet).filter(|(k, _)| !sa.contains(k)).map(|(_, s)| s.clone()).collect();
    if !only_a.is_empty() || !only_b.is_empty() || ga.keys != gb.keys {
        return Ok(Equiv { equivalent: false, exact: !ga.bound_reached && !gb.bound_reached, states_a: ga.states.len(), states_b: gb.states.len(), alphabet: ga.alphabet.clone(), only_a, only_b, witness: None });
    }
    let na = ga.states.len();
    let nb = gb.states.len();
    let mut succ = ga.succ.clone();
    succ.extend(gb.succ.iter().map(|row| row.iter().map(|x| x.map(|j| j + na)).collect::<Vec<_>>()));
    let mut label = ga.plabel.clone();
    label.extend(gb.plabel.iter().cloned());
    let class = refine(na + nb, &succ, &label);
    let equivalent = class[0] == class[na];
    // shortest distinguishing sequence: breadth-first over state pairs
    let mut witness = None;
    if !equivalent {
        let k = ga.alphabet.len();
        let mut seen = std::collections::HashSet::new();
        let mut q = VecDeque::new();
        q.push_back((0usize, 0usize, Vec::<usize>::new()));
        seen.insert((0usize, 0usize));
        'bfs: while let Some((i, j, path)) = q.pop_front() {
            for e in 0..k {
                if ga.plabel[i][e] != gb.plabel[j][e] {
                    let mut p: Vec<String> = path.iter().map(|&x| ga.alphabet[x].clone()).collect();
                    p.push(ga.alphabet[e].clone());
                    witness = Some((p, ga.label[i][e].clone(), gb.label[j][e].clone()));
                    break 'bfs;
                }
                if let (Some(i2), Some(j2)) = (ga.succ[i][e], gb.succ[j][e]) {
                    if seen.insert((i2, j2)) {
                        let mut p = path.clone();
                        p.push(e);
                        q.push_back((i2, j2, p));
                    }
                }
            }
        }
    }
    Ok(Equiv { equivalent, exact: !ga.bound_reached && !gb.bound_reached, states_a: na, states_b: nb, alphabet: ga.alphabet.clone(), only_a, only_b, witness })
}

fn show_output(l: &str) -> String {
    if l == "ok:" { "silent".into() } else if let Some(x) = l.strip_prefix("ok:") { format!("emits {x}") } else if let Some(x) = l.strip_prefix("refused:") { format!("refused: {x}") } else { l.to_string() }
}

/// The text report of `rce equiv`.
pub fn equiv_report(e: &Equiv) -> String {
    let mut s = String::new();
    s.push_str(&format!("alphabet            {:<8} event instances explored\n", e.alphabet.len()));
    s.push_str(&format!("states              {} / {}\n", e.states_a, e.states_b));
    if !e.only_a.is_empty() || !e.only_b.is_empty() {
        s.push_str("verdict             different  the programs explore different event instances\n");
        for x in &e.only_a { s.push_str(&format!("  only A            {x}\n")); }
        for x in &e.only_b { s.push_str(&format!("  only B            {x}\n")); }
        return s;
    }
    if e.equivalent {
        s.push_str(&format!("verdict             equivalent  same behaviour on the explored alphabet ({})\n", if e.exact { "exact" } else { "to the explored bound" }));
    } else if let Some((seq, la, lb)) = &e.witness {
        s.push_str("verdict             different\n");
        s.push_str(&format!("after               {}\n", seq.join(", ")));
        s.push_str(&format!("  A                 {}\n", show_output(la)));
        s.push_str(&format!("  B                 {}\n", show_output(lb)));
    } else {
        s.push_str("verdict             different  (no finite witness within the explored graph)\n");
    }
    s
}

/// The JSON report of `rce equiv` (format `rce-equiv/1`).
pub fn equiv_json(e: &Equiv) -> String {
    let v = serde_json::json!({
        "format": "rce-equiv/1",
        "equivalent": e.equivalent,
        "exact": e.exact,
        "states": [e.states_a, e.states_b],
        "alphabet": e.alphabet,
        "only_a": e.only_a,
        "only_b": e.only_b,
        "witness": e.witness.as_ref().map(|(seq, la, lb)| serde_json::json!({"after": seq, "a": la, "b": lb})),
    });
    crate::interp::canon_json(&v)
}
