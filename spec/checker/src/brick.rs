//! Brick catalog (`rce index`) and type-directed search (`rce search`).
//!
//! A brick is a top-level function of a module (the `pub` ones when the
//! module marks any as `pub`, else all). Its record holds the module, the
//! name, the declared type, a normalized type key, the SHA-256 of its
//! canonical text (`unparse` of the function alone, so layout and line
//! numbers do not change it), the checker's verdict on its module, and its
//! lowering to a Form IR term (spec D.7) or the reason there is none.
//!
//! The catalog is a JSON object with sorted keys, entries sorted by
//! (module, name); its `hash` is the SHA-256 of the canonical JSON of the
//! entries, so equal libraries give equal catalogs byte for byte.
//!
//! Library dependencies (`# uses:`, `#!lib`, and a `#!lib` inferred from
//! unknown names) are resolved before checking; the D.7 lowering is in
//! `brick_lower.rs` and the interpreter's vectors in `brick_vec.rs`.
use crate::ast::*;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub const FORMAT: &str = "rce-catalog/1";
/// Deepest program a catalog holds. Readers parse catalogs without
/// serde_json's default limit of 128 (`read_catalog` here,
/// `rce::form::brick::read_catalog` in the engine, which allows 1024).
pub const MAX_DEPTH: usize = 1000;

/// Parse a catalog past serde_json's default depth limit of 128.
pub fn read_catalog(text: &str) -> Result<Value, String> {
    use serde::Deserialize;
    let mut de = serde_json::Deserializer::from_str(text);
    de.disable_recursion_limit();
    let v = Value::deserialize(&mut de).map_err(|e| e.to_string())?;
    de.end().map_err(|e| e.to_string())?;
    Ok(v)
}

fn sha(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Canonical JSON text (serde_json maps are sorted by key).
/// SHA-256 of the canonical JSON of a term: the key of a Form IR format 2
/// definition (`rce::form::ir::def_key`).
pub fn digest(v: &Value) -> String {
    sha(&canon(v))
}

pub fn canon(v: &Value) -> String {
    serde_json::to_string(v).expect("json")
}

// ---------- types as trees ----------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum N {
    Var(String),
    Con(String, Vec<N>),
}

fn to_n(t: &Ty, vars: &BTreeSet<String>) -> N {
    let c = |n: &str, a: Vec<N>| N::Con(n.into(), a);
    match t {
        Ty::Path(p, a) if a.is_empty() && p.len() == 1 && vars.contains(&p[0]) => {
            N::Var(p[0].clone())
        }
        Ty::Path(p, a) => c(&p.join("."), a.iter().map(|x| to_n(x, vars)).collect()),
        Ty::Unit => c("()", vec![]),
        Ty::Tuple(v) => c("Tuple", v.iter().map(|x| to_n(x, vars)).collect()),
        Ty::Slice(x) => c("Slice", vec![to_n(x, vars)]),
        Ty::Array(x, n) => c(
            &format!("Array;{}", crate::unparse::expr(n)),
            vec![to_n(x, vars)],
        ),
        Ty::Chan(x) => c("Chan", vec![to_n(x, vars)]),
        Ty::Iter(x) => c("Iter", vec![to_n(x, vars)]),
        Ty::Fn(a, r) => {
            let mut v: Vec<N> = a.iter().map(|x| to_n(x, vars)).collect();
            v.push(to_n(r, vars));
            c("Fn", v)
        }
        Ty::Ref(m, x) => c(if *m { "RefMut" } else { "Ref" }, vec![to_n(x, vars)]),
        Ty::Opt(x) => c("Opt", vec![to_n(x, vars)]),
        Ty::Union(a, b) => c("Union", vec![to_n(a, vars), to_n(b, vars)]),
        Ty::Region(x, r) => c(&format!("Region;{r}"), vec![to_n(x, vars)]),
    }
}

/// Rename variables to t0, t1, ... in order of first occurrence.
fn alpha(ts: &[N]) -> Vec<N> {
    fn go(n: &N, m: &mut BTreeMap<String, String>) -> N {
        match n {
            N::Var(v) => {
                let k = m.len();
                N::Var(
                    m.entry(v.clone())
                        .or_insert_with(|| format!("t{k}"))
                        .clone(),
                )
            }
            N::Con(c, a) => N::Con(c.clone(), a.iter().map(|x| go(x, m)).collect()),
        }
    }
    let mut m = BTreeMap::new();
    ts.iter().map(|t| go(t, &mut m)).collect()
}

fn show(n: &N) -> String {
    match n {
        N::Var(v) => v.clone(),
        N::Con(c, a) if a.is_empty() => c.clone(),
        N::Con(c, a) => format!("{c}[{}]", a.iter().map(show).collect::<Vec<_>>().join(", ")),
    }
}

/// The normalized key "(A, B) -> R" with variables renamed t0, t1, ...
fn key(params: &[N], ret: &N) -> String {
    let mut all = params.to_vec();
    all.push(ret.clone());
    let a = alpha(&all);
    let (r, p) = a.split_last().unwrap();
    format!(
        "({}) -> {}",
        p.iter().map(show).collect::<Vec<_>>().join(", "),
        show(r)
    )
}

// ---------- index ----------

/// Options of `rce index`.
#[derive(Default, Clone)]
pub struct Opts {
    /// generated inputs per brick, run through the reference interpreter
    pub vectors: usize,
}

fn lib_directives(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|l| l.strip_prefix("#!lib "))
        .flat_map(|r| r.split_whitespace().map(String::from))
        .collect()
}

fn uses_directives(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|l| l.strip_prefix("# uses: "))
        .flat_map(|r| r.split_whitespace().map(String::from))
        .collect()
}

/// Add the files named by `# uses: a/b` that were not given: `a/b.rce`
/// under the nearest ancestor directory of the naming file that has it.
fn resolve_uses(files: &mut Vec<(String, String)>) {
    let mut i = 0;
    while i < files.len() {
        let (f, src) = files[i].clone();
        for u in uses_directives(&src) {
            let rel = format!("{u}.rce");
            if files
                .iter()
                .any(|(g, _)| g.ends_with(&format!("/{rel}")) || *g == rel)
            {
                continue;
            }
            let mut dir = std::path::Path::new(&f).parent().map(|p| p.to_path_buf());
            while let Some(d) = dir {
                let cand = d.join(&rel);
                if let Ok(t) = std::fs::read_to_string(&cand) {
                    files.push((cand.to_string_lossy().into_owned(), t));
                    break;
                }
                dir = d.parent().map(|p| p.to_path_buf());
            }
        }
        i += 1;
    }
}

/// `m` with the items of `libs` in front (their checks, examples, intents
/// and module lines dropped; an item `m` declares itself wins), as the rce
/// binary does for `#!lib NAME`.
fn flatten(m: &Module, libs: &[&Module]) -> Module {
    let own: HashSet<String> = m.items.iter().filter_map(crate::link::item_name).collect();
    let mut items = vec![];
    let mut seen = own.clone();
    for l in libs {
        for it in &l.items {
            if matches!(it, Item::Check(_) | Item::Example(_) | Item::Intent(..)) {
                continue;
            }
            if let Some(n) = crate::link::item_name(it) {
                if !seen.insert(n) {
                    continue;
                }
            }
            items.push(it.clone());
        }
    }
    items.extend(m.items.iter().cloned());
    Module { items, ..m.clone() }
}

struct Checked {
    errs: Vec<String>,
    linked: Module,
}

fn check_all(mods: &[(String, Module)]) -> Vec<Checked> {
    let known: HashSet<String> = mods
        .iter()
        .filter_map(|(_, m)| m.path.as_ref().map(|q| q.join(".")))
        .collect();
    let refs: Vec<&Module> = mods.iter().map(|(_, m)| m).collect();
    mods.iter()
        .map(|(_, m)| {
            let lk = crate::link::link(m, &refs);
            let mut c = crate::check::Checker::new();
            c.imported_fns = lk.imported_fns;
            c.imported_types = lk.imported_types;
            for e in lk.errs {
                c.e(e);
            }
            c.check_module(&lk.module, &known);
            if c.errs.is_empty() {
                c.errs = crate::interp::admit(&lk.module);
            }
            Checked {
                errs: c.errs,
                linked: lk.module,
            }
        })
        .collect()
}

fn unknown_names(errs: &[String]) -> Option<BTreeSet<String>> {
    let re = regex::Regex::new(r"W-NAME-1 unknown name `([^`]+)`").unwrap();
    let mut v = BTreeSet::new();
    for e in errs {
        for part in e.split("; ") {
            if let Some(c) = re.captures(part) {
                v.insert(c[1].to_string());
            }
        }
    }
    if v.is_empty() { None } else { Some(v) }
}

/// Build the catalog of `files` (name, source), checked together as one program.
pub fn index(files: &[(String, String)]) -> Value {
    index_with(files, &Opts::default())
}

pub fn index_with(given: &[(String, String)], o: &Opts) -> Value {
    let mut files = given.to_vec();
    resolve_uses(&mut files);
    let given_names: HashSet<&String> = given.iter().map(|(f, _)| f).collect();
    let mut parse_errs = Map::new();
    let mut orig: Vec<(String, Module, Vec<String>)> = vec![];
    for (f, src) in &files {
        match crate::parse::parse(src) {
            Ok(m) => orig.push((f.clone(), m, lib_directives(src))),
            Err(e) => {
                parse_errs.insert(f.clone(), json!(e));
            }
        }
    }
    let by_path: HashMap<String, usize> = orig
        .iter()
        .enumerate()
        .filter_map(|(i, (_, m, _))| m.path.as_ref().map(|p| (p.join("."), i)))
        .collect();
    let mut libs: Vec<Vec<String>> = orig.iter().map(|(_, _, d)| d.clone()).collect();
    let mut how: Vec<&str> = libs
        .iter()
        .map(|d| if d.is_empty() { "" } else { "directive" })
        .collect();
    let build = |libs: &Vec<Vec<String>>| -> Vec<(String, Module)> {
        orig.iter()
            .zip(libs)
            .map(|((f, m, _), ls)| {
                let deps: Vec<&Module> = ls
                    .iter()
                    .filter_map(|n| by_path.get(n))
                    .map(|&i| &orig[i].1)
                    .collect();
                (f.clone(), flatten(m, &deps))
            })
            .collect()
    };
    let mut mods = build(&libs);
    let mut checked = check_all(&mods);
    // A file refused only for want of names that one flat library module
    // declares is checked again with that module as its `#!lib`.
    let mut changed = false;
    for i in 0..orig.len() {
        let Some(unk) = unknown_names(&checked[i].errs) else {
            continue;
        };
        let cands: Vec<usize> = (0..orig.len())
            .filter(|&j| {
                j != i
                    && orig[j].1.path.as_ref().is_some_and(|p| p.len() == 1)
                    && !libs[i].contains(&orig[j].1.path.as_ref().unwrap()[0])
            })
            .filter(|&j| {
                // its own names: a module flattened with a library does not offer it
                let names: HashSet<String> = orig[j]
                    .1
                    .items
                    .iter()
                    .filter_map(crate::link::item_name)
                    .collect();
                unk.iter().all(|n| names.contains(n))
            })
            .collect();
        if cands.len() == 1 {
            libs[i].push(orig[cands[0]].1.path.as_ref().unwrap()[0].clone());
            how[i] = "inferred";
            changed = true;
        }
    }
    if changed {
        mods = build(&libs);
        checked = check_all(&mods);
    }
    let mut entries = vec![];
    let mut vectors = Map::new();
    let mut refused_files = parse_errs;
    let mut resolved = Map::new();
    for (i, (file, m, _)) in orig.iter().enumerate() {
        if !given_names.contains(file) {
            continue;
        }
        if !libs[i].is_empty() {
            resolved.insert(file.clone(), json!({"lib": libs[i], "how": how[i]}));
        }
        let ok = checked[i].errs.is_empty();
        if !ok {
            refused_files.insert(
                file.clone(),
                json!(
                    checked[i]
                        .errs
                        .join("; ")
                        .chars()
                        .take(300)
                        .collect::<String>()
                ),
            );
        }
        let module = m.path.as_ref().map(|p| p.join(".")).unwrap_or_default();
        // A file with no `# rce-version` line has no declared version. With the
        // `conc06` feature the parser reads it as 0.6 and `unparse` would write a
        // 0.6 header, so a brick hash would depend on the build. The hash uses the
        // declared version only.
        let undeclared = files
            .iter()
            .find(|(f, _)| f == file)
            .is_some_and(|(_, src)| matches!(crate::parse::declared_version(src), Ok(None)));
        let linked = &checked[i].linked;
        let fns: Vec<&FnDecl> = m
            .items
            .iter()
            .filter_map(|i| if let Item::Fn(d) = i { Some(d) } else { None })
            .collect();
        let any_pub = fns.iter().any(|d| m.pubs.contains(&d.sig.name));
        // constants as values, by the interpreter (only when the file admits)
        let mut consts = HashMap::new();
        if ok {
            let mut it = crate::interp::Interp::new(linked);
            for item in &linked.items {
                if let Item::Const { name, .. } = item
                    && let Ok(v) = it.const_value(name)
                    && let Ok(j) =
                        serde_json::from_str::<Value>(&crate::interp::form_value(&v, linked).json())
                {
                    consts.insert(name.clone(), j);
                }
            }
        }
        let mut lw = crate::brick_lower::Lowerer::new(linked, &module, consts);
        for d in fns {
            if any_pub && !m.pubs.contains(&d.sig.name) {
                continue;
            }
            let vars: BTreeSet<String> = d.sig.generics.iter().map(|g| g.name.clone()).collect();
            let mut pnames = vec![];
            let mut pts = vec![];
            let mut ptys = vec![];
            let mut has_self = false;
            for p in &d.sig.params {
                match p {
                    Param::Named { name, ty, .. } => {
                        pnames.push(name.clone());
                        ptys.push(crate::unparse::ty(ty));
                        pts.push(to_n(ty, &vars));
                    }
                    _ => has_self = true,
                }
            }
            if has_self {
                continue;
            }
            let ret = d.sig.ret.clone().unwrap_or(Ty::Unit);
            let rn = to_n(&ret, &vars);
            let one = Module {
                path: None,
                items: vec![Item::Fn(d.clone())],
                pubs: BTreeSet::new(),
                draft06: m.draft06 && !undeclared,
                v07: m.v07,
                v08: m.v08,
            };
            let text = crate::unparse::unparse(&one);
            let hash = sha(&format!("{module}\n{text}"));
            let lowering = if !ok {
                json!({"refused": "its file is refused by the checker"})
            } else {
                match lw.lower_fn(&d.sig.name) {
                    // a reader of JSON nests at most 128 deep (serde_json)
                    Ok(t) if depth(&t) > MAX_DEPTH => {
                        json!({"refused": format!("its program nests {} deep, over {MAX_DEPTH}", depth(&t))})
                    }
                    Ok(t) => json!({"term": t}),
                    Err(e) => json!({"refused": e}),
                }
            };
            if ok && o.vectors > 0 && lowering.get("term").is_some() {
                let vs = crate::brick_vec::vectors(linked, d, o.vectors);
                if !vs.is_empty() {
                    vectors.insert(hash.clone(), Value::Array(vs));
                }
            }
            entries.push(json!({
                "module": module,
                "name": d.sig.name,
                "file": file.rsplit('/').next().unwrap_or(file),
                "line": d.sig.line,
                "generics": vars.iter().cloned().collect::<Vec<_>>(),
                "params": pnames.iter().zip(&ptys).map(|(n, t)| json!([n, t])).collect::<Vec<_>>(),
                "ret": crate::unparse::ty(&ret),
                "key": key(&pts, &rn),
                "hash": hash,
                "checked": ok,
                "lowering": lowering,
            }));
        }
    }
    entries.sort_by(|a, b| {
        (a["module"].as_str(), a["name"].as_str()).cmp(&(b["module"].as_str(), b["name"].as_str()))
    });
    let ents = Value::Array(entries);
    let lowered = ents
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["lowering"].get("term").is_some())
        .count();
    let mut out = json!({
        "format": FORMAT,
        "hash": sha(&canon(&ents)),
        "count": ents.as_array().unwrap().len(),
        "lowered": lowered,
        "refused_files": Value::Object(refused_files),
        "resolved": Value::Object(resolved),
        "entries": ents,
    });
    if o.vectors > 0 {
        out["vectors"] = Value::Object(vectors);
    }
    out
}

// ---------- search ----------

/// Parse a query "A -> R" or "(A, B) -> R"; lowercase one-letter names are variables.
pub fn parse_query(q: &str) -> Result<(Vec<N>, N), String> {
    let split_top = |s: &str, sep: &str| -> Vec<String> {
        let (mut d, mut out, mut cur) = (0i32, vec![], String::new());
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let ch = b[i] as char;
            if "([".contains(ch) {
                d += 1
            }
            if ")]".contains(ch) {
                d -= 1
            }
            if d == 0 && s[i..].starts_with(sep) {
                out.push(std::mem::take(&mut cur));
                i += sep.len();
                continue;
            }
            cur.push(ch);
            i += 1;
        }
        out.push(cur);
        out
    };
    let parts = split_top(q.trim(), "->");
    if parts.len() < 2 {
        return Err("query must be `A -> R` or `(A, B) -> R`".into());
    }
    let ret_s = parts.last().unwrap().trim().to_string();
    let mut args_s = parts[..parts.len() - 1].join("->").trim().to_string();
    if args_s.starts_with('(')
        && args_s.ends_with(')')
        && split_top(&args_s[1..args_s.len() - 1], ",").len() >= 1
    {
        let inner = &args_s[1..args_s.len() - 1];
        // `(A, B)` is the argument list; `()` is no arguments
        args_s = inner.to_string();
    }
    let ty = |s: &str| -> Result<N, String> {
        let m = crate::parse::parse(&format!("type Q__ = {}\n", s.trim()))
            .map_err(|e| format!("type `{}`: {e}", s.trim()))?;
        let Some(Item::TypeAlias { ty, .. }) = m.items.first() else {
            return Err(format!("type `{s}`"));
        };
        let mut vars = BTreeSet::new();
        collect_vars(ty, &mut vars);
        Ok(to_n(ty, &vars))
    };
    let args = if args_s.trim().is_empty() {
        vec![]
    } else {
        split_top(&args_s, ",")
            .iter()
            .map(|a| ty(a))
            .collect::<Result<_, _>>()?
    };
    Ok((args, ty(&ret_s)?))
}

fn collect_vars(t: &Ty, v: &mut BTreeSet<String>) {
    match t {
        Ty::Path(p, a) => {
            if a.is_empty()
                && p.len() == 1
                && p[0].len() == 1
                && p[0].chars().all(|c| c.is_ascii_lowercase())
            {
                v.insert(p[0].clone());
            }
            a.iter().for_each(|x| collect_vars(x, v));
        }
        Ty::Tuple(xs) => xs.iter().for_each(|x| collect_vars(x, v)),
        Ty::Fn(xs, r) => {
            xs.iter().for_each(|x| collect_vars(x, v));
            collect_vars(r, v)
        }
        Ty::Slice(x)
        | Ty::Array(x, _)
        | Ty::Chan(x)
        | Ty::Iter(x)
        | Ty::Ref(_, x)
        | Ty::Opt(x)
        | Ty::Region(x, _) => collect_vars(x, v),
        Ty::Union(a, b) => {
            collect_vars(a, v);
            collect_vars(b, v)
        }
        Ty::Unit => {}
    }
}

/// Rebuild an entry's type tree from its catalog record.
fn entry_sig(e: &Value) -> Option<(Vec<N>, N)> {
    let vars: BTreeSet<String> = e["generics"]
        .as_array()?
        .iter()
        .filter_map(|x| x.as_str().map(String::from))
        .collect();
    let ty = |s: &str| -> Option<N> {
        let m = crate::parse::parse(&format!("type Q__ = {s}\n")).ok()?;
        let Some(Item::TypeAlias { ty, .. }) = m.items.first() else {
            return None;
        };
        Some(to_n(ty, &vars))
    };
    let ps = e["params"]
        .as_array()?
        .iter()
        .map(|p| ty(p[1].as_str()?))
        .collect::<Option<Vec<_>>>()?;
    Some((ps, ty(e["ret"].as_str()?)?))
}

/// First-order unification; variables of the brick are prefixed `b:`, of the query `q:`.
fn unify(a: &N, b: &N, s: &mut BTreeMap<String, N>) -> bool {
    fn walk(n: &N, s: &BTreeMap<String, N>) -> N {
        match n {
            N::Var(v) => s.get(v).map(|x| walk(x, s)).unwrap_or_else(|| n.clone()),
            _ => n.clone(),
        }
    }
    fn occurs(v: &str, n: &N, s: &BTreeMap<String, N>) -> bool {
        match walk(n, s) {
            N::Var(w) => w == v,
            N::Con(_, a) => a.iter().any(|x| occurs(v, x, s)),
        }
    }
    let (a, b) = (walk(a, s), walk(b, s));
    match (&a, &b) {
        (N::Var(x), N::Var(y)) if x == y => true,
        (N::Var(x), t) | (t, N::Var(x)) => {
            if occurs(x, t, s) {
                return false;
            }
            s.insert(x.clone(), t.clone());
            true
        }
        (N::Con(c, xs), N::Con(d, ys)) => {
            c == d && xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| unify(x, y, s))
        }
    }
}

fn prefix(n: &N, p: &str) -> N {
    match n {
        N::Var(v) => N::Var(format!("{p}{v}")),
        N::Con(c, a) => N::Con(c.clone(), a.iter().map(|x| prefix(x, p)).collect()),
    }
}

fn perms(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut out = vec![];
    for p in perms(n - 1) {
        for i in 0..=p.len() {
            let mut q = p.clone();
            q.insert(i, n - 1);
            out.push(q);
        }
    }
    out.sort();
    out
}

/// Search the catalog for bricks of type `q`. Rank (lower is better):
/// 0 equal up to renaming of variables; 1 an instance (variables bound);
/// +2 when the arguments match only in another order. Arity above 5 is
/// matched in declared order only. Ties break by (module, name).
pub fn search(catalog: &Value, q: &str) -> Result<Vec<(u32, String, String, String)>, String> {
    let (qa, qr) = parse_query(q)?;
    let qkey = key(&qa, &qr);
    let mut hits = vec![];
    for e in catalog["entries"].as_array().ok_or("not a catalog")? {
        let Some((ba, br)) = entry_sig(e) else {
            continue;
        };
        if ba.len() != qa.len() {
            continue;
        }
        let orders = if ba.len() <= 5 {
            perms(ba.len())
        } else {
            vec![(0..ba.len()).collect()]
        };
        let mut best: Option<u32> = None;
        for (oi, ord) in orders.iter().enumerate() {
            let mut s = BTreeMap::new();
            let ok = ord
                .iter()
                .enumerate()
                .all(|(qi, &bi)| unify(&prefix(&ba[bi], "b:"), &prefix(&qa[qi], "q:"), &mut s))
                && unify(&prefix(&br, "b:"), &prefix(&qr, "q:"), &mut s);
            if !ok {
                continue;
            }
            let permuted: Vec<N> = ord.iter().map(|&i| ba[i].clone()).collect();
            let r = if key(&permuted, &br) == qkey { 0 } else { 1 }
                + if ord.iter().enumerate().all(|(i, &j)| i == j) {
                    0
                } else {
                    2
                };
            let _ = oi;
            best = Some(best.map_or(r, |b: u32| b.min(r)));
        }
        if let Some(r) = best {
            hits.push((
                r,
                e["module"].as_str().unwrap_or("").to_string(),
                e["name"].as_str().unwrap_or("").to_string(),
                e["key"].as_str().unwrap_or("").to_string(),
            ));
        }
    }
    hits.sort();
    Ok(hits)
}

/// Nesting depth of a JSON value.
pub fn depth(v: &Value) -> usize {
    match v {
        Value::Array(a) => 1 + a.iter().map(depth).max().unwrap_or(0),
        Value::Object(m) => 1 + m.values().map(depth).max().unwrap_or(0),
        _ => 0,
    }
}
