//! Definition hashes and `rce api-diff` (spec F.4.3, P10).
//!
//! A *definition hash* is the SHA-256 of the canonical text of an item with
//! its own name and the names of the items it references replaced: a
//! reference to another item becomes that item's definition hash, a
//! reference inside a mutually recursive group becomes the member's rank in
//! the group. Comments, layout and declaration order are not hashed (the
//! text is the unparse of the parsed item, which parenthesizes every
//! subexpression). Renaming an item and its references everywhere gives the
//! same hash (T7 for items, stated and unproved as T50).
//!
//! Limits, stated: a reference is found by a name token in the item's text
//! outside string literals, so a field, label or method that is spelled like
//! an item of the module is treated as a reference (the hash stays a
//! function of the text; it only changes more often than it must). `impl`
//! blocks and `step`s are not individually hashed. Contracts, measures,
//! ghost clauses and labels are in the hash by construction once the
//! constructs exist (F.6.2, F.4.2); they do not exist in this checker yet.
use crate::ast::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub fn member_hash(m: &Module, f: &FnDecl) -> String {
    sha(&format!("rce-def/0.9\n{}\n{}", module_version(m), one(m, &Item::Fn(f.clone()))))
}

fn sha(s: &str) -> String {
    Sha256::digest(s.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

pub fn kind_of(it: &Item) -> &'static str {
    match it {
        Item::Const { .. } => "const",
        Item::TypeAlias { .. } => "type",
        Item::Struct { .. } => "struct",
        Item::State { .. } => "state",
        Item::Event { .. } => "event",
        Item::Error { .. } => "error",
        Item::Kind { .. } => "kind",
        Item::Trait { .. } => "trait",
        Item::Actor { .. } => "actor",
        Item::Protocol { .. } => "protocol",
        Item::Node { .. } => "node",
        Item::Supervisor { .. } => "supervisor",
        Item::Fn(_) => "fn",
        Item::Extern { .. } => "extern",
        _ => "",
    }
}

/// One named definition of a module.
#[derive(Debug, Clone)]
pub struct Def {
    pub name: String,
    pub kind: &'static str,
    pub exported: bool,
    pub text: String,
    pub hash: String,
    /// the signature text of a function (empty body), or the shape of a type
    pub sig: String,
    item: Item,
}

fn module_version(m: &Module) -> &'static str {
    if m.v09 {
        "0.9"
    } else if m.v08 {
        "0.8"
    } else if m.v07 {
        "0.7"
    } else if m.draft06 {
        "0.6"
    } else {
        "0.5"
    }
}

fn one(m: &Module, it: &Item) -> String {
    let mm = Module {
        path: None,
        items: vec![it.clone()],
        pubs: BTreeSet::new(),
        draft06: m.draft06,
        v07: m.v07,
        v08: m.v08,
        v09: m.v09,
        v09_hygiene: vec![],
        x09: Default::default(),
    };
    crate::unparse::unparse(&mm).lines().filter(|l| !l.starts_with("# rce-version:")).collect::<Vec<_>>().join("\n")
}

fn strip_strings(s: &str) -> String {
    let mut o = String::new();
    let mut in_s = false;
    let mut esc = false;
    for c in s.chars() {
        if in_s {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_s = false;
                o.push('"');
            }
        } else {
            if c == '"' {
                in_s = true;
            }
            o.push(c);
        }
    }
    o
}

/// Replace every identifier token of `text` found in `map`.
fn mask(text: &str, map: &HashMap<String, String>) -> String {
    let mut out = String::new();
    let cs: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut in_s = false;
    let mut esc = false;
    while i < cs.len() {
        let c = cs[i];
        if in_s {
            out.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_s = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_s = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            let w: String = cs[st..i].iter().collect();
            // not after a `.` (a member) or a `:` label is still masked: stated limit
            match map.get(&w) {
                Some(r) => out.push_str(r),
                None => out.push_str(&w),
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn idents(text: &str) -> Vec<String> {
    let t = strip_strings(text);
    let mut v = vec![];
    let mut cur = String::new();
    for c in t.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_alphanumeric() || c == '_' {
            cur.push(c);
        } else if !cur.is_empty() {
            if !cur.chars().next().unwrap().is_ascii_digit() {
                v.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    v
}

fn sig_of(m: &Module, it: &Item) -> String {
    match it {
        Item::Fn(f) => {
            let mut g = f.clone();
            g.sig.line = 0;
            g.body = Body::Block(vec![Stmt::Pass]);
            one(m, &Item::Fn(g))
        }
        _ => one(m, it),
    }
}

/// Every named definition of a module with its definition hash.
pub fn defs(m: &Module) -> Vec<Def> {
    let named: Vec<(String, &Item)> = m
        .items
        .iter()
        .filter_map(|it| crate::link::item_name(it).map(|n| (n, it)))
        .collect();
    let names: BTreeMap<String, usize> = named.iter().enumerate().map(|(i, (n, _))| (n.clone(), i)).collect();
    let texts: Vec<String> = named.iter().map(|(_, it)| one(m, it)).collect();
    let refs: Vec<BTreeSet<usize>> = texts
        .iter()
        .map(|t| idents(t).into_iter().filter_map(|w| names.get(&w).copied()).collect())
        .collect();
    // Tarjan SCC over the reference graph
    let n = named.len();
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0; n];
    let mut on = vec![false; n];
    let mut st: Vec<usize> = vec![];
    let mut comps: Vec<Vec<usize>> = vec![];
    let mut idx = 0;
    fn strong(v: usize, refs: &[BTreeSet<usize>], index: &mut Vec<usize>, low: &mut Vec<usize>, on: &mut Vec<bool>, st: &mut Vec<usize>, comps: &mut Vec<Vec<usize>>, idx: &mut usize) {
        index[v] = *idx;
        low[v] = *idx;
        *idx += 1;
        st.push(v);
        on[v] = true;
        for &w in &refs[v] {
            if index[w] == usize::MAX {
                strong(w, refs, index, low, on, st, comps, idx);
                low[v] = low[v].min(low[w]);
            } else if on[w] {
                low[v] = low[v].min(index[w]);
            }
        }
        if low[v] == index[v] {
            let mut c = vec![];
            loop {
                let w = st.pop().unwrap();
                on[w] = false;
                c.push(w);
                if w == v {
                    break;
                }
            }
            comps.push(c);
        }
    }
    for v in 0..n {
        if index[v] == usize::MAX {
            strong(v, &refs, &mut index, &mut low, &mut on, &mut st, &mut comps, &mut idx);
        }
    }
    // Tarjan emits components in reverse topological order of the reference
    // graph: a component comes after every component it references
    let ver = module_version(m);
    let mut hash: Vec<String> = vec![String::new(); n];
    for comp in &comps {
        let in_comp: BTreeSet<usize> = comp.iter().copied().collect();
        let mut base: HashMap<String, String> = HashMap::new();
        for (name, &j) in &names {
            if !in_comp.contains(&j) {
                if !hash[j].is_empty() {
                    base.insert(name.clone(), format!("#{}", hash[j]));
                }
            }
        }
        // rank members by their text with the group's own names all masked alike
        let mut flat = base.clone();
        for &j in comp {
            flat.insert(named[j].0.clone(), "@".into());
        }
        let mut order: Vec<(String, usize)> = comp.iter().map(|&j| (mask(&texts[j], &flat), j)).collect();
        order.sort();
        let mut ranked = base.clone();
        for (k, (_, j)) in order.iter().enumerate() {
            ranked.insert(named[*j].0.clone(), format!("@{k}"));
        }
        let masked: Vec<String> = order.iter().map(|(_, j)| mask(&texts[*j], &ranked)).collect();
        let group = sha(&format!("rce-def/0.9\n{ver}\n{}", masked.join("\n---\n")));
        for (k, (_, j)) in order.iter().enumerate() {
            hash[*j] = if comp.len() == 1 && !refs[*j].contains(j) {
                group.clone()
            } else {
                sha(&format!("{group}#{k}"))
            };
        }
    }
    named
        .iter()
        .enumerate()
        .map(|(i, (name, it))| Def {
            name: name.clone(),
            kind: kind_of(it),
            exported: m.pubs.contains(name) || matches!(it, Item::Event { .. } | Item::Error { .. } | Item::State { .. }),
            text: texts[i].clone(),
            hash: hash[i].clone(),
            sig: sig_of(m, it),
            item: (*it).clone(),
        })
        .collect()
}

/// A row of the api-diff table (F.4.3).
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub breaking: bool,
    pub kind: &'static str,
    pub name: String,
    pub what: String,
}

impl Row {
    pub fn show(&self) -> String {
        format!("{}\t{} {}\t{}", if self.breaking { "breaking" } else { "compatible" }, self.kind, self.name, self.what)
    }
}

fn mode(t: &Ty) -> &'static str {
    match t {
        Ty::Ref(false, _) => "borrow",
        Ty::Ref(true, _) => "&mut",
        _ => "consume",
    }
}

fn strip_mode(t: &Ty) -> Ty {
    match t {
        Ty::Ref(_, x) => (**x).clone(),
        o => o.clone(),
    }
}

fn shape(it: &Item) -> Vec<(String, String)> {
    match it {
        Item::Struct { fields, .. } | Item::State { fields, .. } => fields.iter().map(|f| (f.name.clone(), format!("{:?}", strip_lines_ty(&f.ty)))).collect(),
        Item::Kind { variants, .. } => variants.iter().map(|(n, ps)| (n.clone(), format!("{:?}", ps.iter().map(|(a, t)| (a.clone(), strip_lines_ty(t))).collect::<Vec<_>>()))).collect(),
        Item::Event { payload, .. } | Item::Error { payload, .. } => payload.iter().map(|(n, t)| (n.clone(), format!("{:?}", strip_lines_ty(t)))).collect(),
        _ => vec![],
    }
}

fn strip_lines_ty(t: &Ty) -> String {
    // types carry no line numbers except inside array length expressions
    format!("{t:?}")
}

/// Compare two module sets by exported definition. Rows are sorted by name.
pub fn diff(old: &[Module], new: &[Module]) -> Vec<Row> {
    let collect = |ms: &[Module]| -> BTreeMap<String, Def> {
        let mut o = BTreeMap::new();
        for m in ms {
            for d in defs(m) {
                if d.exported {
                    o.insert(d.name.clone(), d);
                }
            }
        }
        o
    };
    let (a, b) = (collect(old), collect(new));
    let mut rows = vec![];
    for (n, d) in &a {
        let Some(e) = b.get(n) else {
            rows.push(Row { breaking: true, kind: d.kind, name: n.clone(), what: "export removed or renamed".into() });
            continue;
        };
        if d.kind != e.kind {
            rows.push(Row { breaking: true, kind: e.kind, name: n.clone(), what: format!("kind changed from {} to {}", d.kind, e.kind) });
            continue;
        }
        if d.hash == e.hash {
            continue;
        }
        match (&d.item, &e.item) {
            (Item::Fn(f), Item::Fn(g)) => {
                if d.sig == e.sig {
                    rows.push(Row { breaking: false, kind: "fn", name: n.clone(), what: "body changed under an unchanged signature".into() });
                    continue;
                }
                let (pf, pg) = (&f.sig.params, &g.sig.params);
                let named = |ps: &Vec<Param>| ps.iter().filter_map(|p| if let Param::Named { ty, .. } = p { Some(ty.clone()) } else { None }).collect::<Vec<_>>();
                let (tf, tg) = (named(pf), named(pg));
                let mut explained = false;
                if tf.len() == tg.len() && f.sig.ret == g.sig.ret && f.sig.generics == g.sig.generics {
                    for (i, (x, y)) in tf.iter().zip(&tg).enumerate() {
                        if x != y && strip_mode(x) == strip_mode(y) && mode(x) != mode(y) {
                            rows.push(Row { breaking: true, kind: "fn", name: n.clone(), what: format!("parameter {} changed from {} to {}", i + 1, mode(x), mode(y)) });
                            explained = true;
                        }
                    }
                }
                if !explained {
                    rows.push(Row { breaking: true, kind: "fn", name: n.clone(), what: "signature changed".into() });
                }
            }
            (x, y) => {
                let (sx, sy) = (shape(x), shape(y));
                if matches!(x, Item::Struct { .. } | Item::Kind { .. } | Item::Event { .. } | Item::Error { .. } | Item::State { .. }) {
                    let added: Vec<&(String, String)> = sy.iter().filter(|p| !sx.iter().any(|q| q.0 == p.0)).collect();
                    let kept_same = sx.iter().all(|p| sy.contains(p));
                    if kept_same && !added.is_empty() {
                        let what = if matches!(x, Item::Kind { .. }) { "enum variant added" } else { "record field added" };
                        for a in added {
                            rows.push(Row { breaking: true, kind: d.kind, name: n.clone(), what: format!("{what}: {}", a.0) });
                        }
                        continue;
                    }
                }
                rows.push(Row { breaking: true, kind: d.kind, name: n.clone(), what: "shape or type changed".into() });
            }
        }
    }
    for (n, e) in &b {
        if !a.contains_key(n) {
            rows.push(Row { breaking: false, kind: e.kind, name: n.clone(), what: "export added".into() });
        }
    }
    rows.sort_by(|x, y| (x.name.as_str(), &x.what).cmp(&(y.name.as_str(), &y.what)));
    rows
}
