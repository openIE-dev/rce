//! The RCE Form as the code graph.
//!
//! The workspace program is parsed and checked by the RCE reference checker
//! (`rce-spec-check`, the same code as `rce check`). From its AST we derive
//! definitions, references, call edges and an Aider-style ranked repo map.
//! These are caches: [`FormIndex::build`] is a pure function of the program,
//! and [`IndexCache`] keys them by the Form version hash, so a cache entry is
//! either exact for that hash or rebuilt.
use rce_spec_check::ast::{Arg, Body, Expr, FnDecl, Item, Pat, ProtoStep, SelArm, Stmt, Ty};
use rce_spec_check::{check_program, parse};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Bumped when the derived index layout changes; part of the cache key.
pub const INDEX_SCHEMA: &str = "rce-memory.index.v1";

/// BLAKE3 over the sorted (path, source) pairs: the Form version hash.
pub fn form_hash(program: &[(String, String)]) -> String {
    let mut files: Vec<&(String, String)> = program.iter().collect();
    files.sort();
    let mut h = blake3::Hasher::new();
    for (p, s) in files {
        h.update(p.as_bytes());
        h.update(&[0]);
        h.update(s.as_bytes());
        h.update(&[0]);
    }
    h.finalize().to_hex().to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Def {
    /// Qualified name: `Name`, or `Owner.method` for impl and actor members.
    pub name: String,
    pub kind: String,
    pub file: String,
    pub line: usize,
    /// Concurrency annotation from RCE 0.6/0.6b, when present:
    /// `replicated`, `interrupt priority P`, `deadline D ticks`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

/// A short source-like rendering of a small expression (priority, deadline).
fn short(e: &Expr) -> String {
    match e {
        Expr::Int(n, _) => n.to_string(),
        Expr::Name(n, _) => n.clone(),
        _ => "expr".into(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ref {
    /// Index into `defs` of the item the reference sits in.
    pub from: usize,
    /// Index into `defs` of the item it names.
    pub to: usize,
    pub call: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormIndex {
    pub schema: String,
    pub form_hash: String,
    pub defs: Vec<Def>,
    pub refs: Vec<Ref>,
    /// Per file: `None` when `rce check` admitted it, else the "requires change" report.
    pub verdicts: BTreeMap<String, Option<String>>,
    /// PageRank score per def (uniform personalization).
    pub rank: Vec<f64>,
}

/// Collects every name mentioned in an item body.
#[derive(Default)]
struct Names {
    names: Vec<(String, bool)>,
}

impl Names {
    fn ty(&mut self, t: &Ty) {
        match t {
            Ty::Path(p, args) => {
                if let Some(l) = p.last() {
                    self.names.push((l.clone(), false));
                }
                args.iter().for_each(|a| self.ty(a));
            }
            Ty::Unit => {}
            Ty::Tuple(v) => v.iter().for_each(|a| self.ty(a)),
            Ty::Slice(a) | Ty::Chan(a) | Ty::Iter(a) | Ty::Ref(_, a) | Ty::Opt(a) => self.ty(a),
            Ty::Region(a, _) => self.ty(a),
            Ty::Array(a, e) => {
                self.ty(a);
                self.expr(e)
            }
            Ty::Fn(ps, r) => {
                ps.iter().for_each(|a| self.ty(a));
                self.ty(r)
            }
            Ty::Union(a, b) => {
                self.ty(a);
                self.ty(b)
            }
        }
    }
    fn pat(&mut self, p: &Pat) {
        match p {
            Pat::Wild => {}
            Pat::Path(q) => {
                if let Some(l) = q.first() {
                    self.names.push((l.clone(), false));
                }
                if let Some(l) = q.last() {
                    self.names.push((l.clone(), false));
                }
            }
            Pat::Ctor(q, fs) => {
                if let Some(l) = q.last() {
                    self.names.push((l.clone(), false));
                }
                fs.iter().for_each(|(_, p)| self.pat(p));
            }
            Pat::Lit(e) => self.expr(e),
            Pat::Range(a, _, b) => {
                self.expr(a);
                self.expr(b)
            }
            Pat::Tuple(v) | Pat::Alt(v) => v.iter().for_each(|p| self.pat(p)),
        }
    }
    fn args(&mut self, a: &[Arg]) {
        a.iter().for_each(|x| self.expr(&x.e));
    }
    fn expr(&mut self, e: &Expr) {
        use Expr::*;
        match e {
            Name(n, _) => self.names.push((n.clone(), false)),
            SelfRef(_) | Int(..) | Float(..) | Str(..) | Bool(..) | None(_) | Unit(_) => {}
            Tuple(v, _) | List(v, _) => v.iter().for_each(|x| self.expr(x)),
            ListRepeat(a, b, _) => {
                self.expr(a);
                self.expr(b)
            }
            ListComp(a, p, b, c, _) => {
                self.expr(a);
                self.pat(p);
                self.expr(b);
                if let Some(c) = c {
                    self.expr(c)
                }
            }
            Map(kv, _) => kv.iter().for_each(|(k, v)| {
                self.expr(k);
                self.expr(v)
            }),
            ChanNew(t, c, _) => {
                self.ty(t);
                if let Some(c) = c {
                    self.expr(c)
                }
            }
            Call(f, a, _, _) => {
                match &**f {
                    Name(n, _) => self.names.push((n.clone(), true)),
                    Member(r, n, _) => {
                        self.expr(r);
                        self.names.push((n.clone(), true))
                    }
                    other => self.expr(other),
                }
                self.args(a)
            }
            Index(a, v, _) => {
                self.expr(a);
                v.iter().for_each(|x| self.expr(x))
            }
            Slice(a, b, c, _) => {
                self.expr(a);
                if let Some(b) = b {
                    self.expr(b)
                }
                if let Some(c) = c {
                    self.expr(c)
                }
            }
            Member(a, _, _) | TupleMember(a, _, _) => self.expr(a),
            Neg(a, _)
            | BitNot(a, _)
            | Borrow(_, a, _)
            | Recv(a, _)
            | Comptime(a, _)
            | Not(a, _) => self.expr(a),
            Cast(a, t, _) => {
                self.expr(a);
                self.ty(t)
            }
            Bin(_, a, b, _) => {
                self.expr(a);
                self.expr(b)
            }
            Catch(a, _, b, _) => {
                self.expr(a);
                self.expr(b)
            }
            Try(a, b, _) => {
                self.expr(a);
                if let Some(b) = b {
                    self.expr(b)
                }
            }
            If(a, b, c, _) => {
                self.expr(a);
                self.expr(b);
                self.expr(c)
            }
            Lambda(ps, b, _) => {
                self.params(ps);
                self.expr(b)
            }
        }
    }
    fn params(&mut self, ps: &[rce_spec_check::ast::Param]) {
        for p in ps {
            if let rce_spec_check::ast::Param::Named { ty, .. } = p {
                self.ty(ty)
            }
        }
    }
    fn stmts(&mut self, v: &[Stmt]) {
        v.iter().for_each(|s| self.stmt(s));
    }
    fn stmt(&mut self, s: &Stmt) {
        use Stmt::*;
        match s {
            Expr(e) | Emit(e) | Fail(e) | Spawn(e) | Check(e) | Yield(e, _) => self.expr(e),
            Assign(_, a, b) | Send(a, b) => {
                self.expr(a);
                self.expr(b)
            }
            Let(p, t, e, _) => {
                self.pat(p);
                if let Some(t) = t {
                    self.ty(t)
                }
                self.expr(e)
            }
            Var(_, t, e, _) => {
                if let Some(t) = t {
                    self.ty(t)
                }
                self.expr(e)
            }
            If(arms, els) => {
                for (c, b) in arms {
                    self.expr(c);
                    self.stmts(b)
                }
                if let Some(b) = els {
                    self.stmts(b)
                }
            }
            While(c, l, b) => {
                self.expr(c);
                if let Some(l) = l {
                    self.expr(l)
                }
                self.stmts(b)
            }
            Loop(l, b) => {
                if let Some(l) = l {
                    self.expr(l)
                }
                self.stmts(b)
            }
            For(p, e, b, _) => {
                self.pat(p);
                self.expr(e);
                self.stmts(b)
            }
            Match(e, arms, _) => {
                self.expr(e);
                for a in arms {
                    self.pat(&a.pat);
                    if let Some(g) = &a.guard {
                        self.expr(g)
                    }
                    self.stmts(&a.body)
                }
            }
            Return(e, _) => {
                if let Some(e) = e {
                    self.expr(e)
                }
            }
            Break(_) | Continue(_) | Pass => {}
            Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(p, e, b) => {
                            self.pat(p);
                            self.expr(e);
                            self.stmts(b)
                        }
                        SelArm::Send(a, e, b) => {
                            self.expr(a);
                            self.expr(e);
                            self.stmts(b)
                        }
                        SelArm::After(e, b) => {
                            self.expr(e);
                            self.stmts(b)
                        }
                        SelArm::Default(b) => self.stmts(b),
                    }
                }
            }
            Arena(_, e, b, _) | Device(_, e, b, _) => {
                if let Some(e) = e {
                    self.expr(e)
                }
                self.stmts(b)
            }
            Comptime(b) => self.stmts(b),
        }
    }
    fn body(&mut self, b: &Body) {
        match b {
            Body::Arrow(e) => self.expr(e),
            Body::Block(v) => self.stmts(v),
        }
    }
    fn func(&mut self, f: &FnDecl) {
        self.params(&f.sig.params);
        if let Some(r) = &f.sig.ret {
            self.ty(r)
        }
        self.body(&f.body);
    }
}

fn ty_name(t: &Ty) -> String {
    match t {
        Ty::Path(p, _) => p.join("."),
        other => format!("{other:?}"),
    }
}

/// (def, names used inside it) for one item; impl and actor members are
/// their own defs.
fn item_defs(file: &str, it: &Item, out: &mut Vec<(Def, Names)>) {
    let mut n = Names::default();
    let d = |name: String, kind: &str, line: usize| Def {
        name,
        kind: kind.into(),
        file: file.into(),
        line,
        note: String::new(),
    };
    let def = match it {
        Item::Use(..) => return,
        Item::Intent(text, ex, checks, _) => {
            for e in ex {
                for (ev, a) in &e.events {
                    n.names.push((ev.clone(), false));
                    n.args(a);
                }
                if let Some(r) = &e.kept {
                    n.names.push((r.clone(), false));
                }
            }
            checks.iter().for_each(|c| n.expr(c));
            let line = ex.first().map(|e| e.line.saturating_sub(1)).unwrap_or(0);
            d(format!("intent \"{text}\""), "intent", line)
        }
        Item::Example(_) | Item::Check(_) | Item::Law(_) => return,
        Item::Const { name, ty, e, line } => {
            if let Some(t) = ty {
                n.ty(t)
            }
            n.expr(e);
            d(name.clone(), "const", *line)
        }
        Item::TypeAlias { name, ty, line, .. } => {
            n.ty(ty);
            d(name.clone(), "type", *line)
        }
        Item::Struct {
            name, fields, line, ..
        } => {
            fields.iter().for_each(|f| n.ty(&f.ty));
            d(name.clone(), "struct", *line)
        }
        Item::State { name, fields, line, replicated } => {
            for f in fields {
                n.ty(&f.ty);
                if let Some(e) = &f.init {
                    n.expr(e)
                }
            }
            let mut x = d(name.clone(), "state", *line);
            if *replicated {
                x.note = "replicated".into()
            }
            x
        }
        Item::Event {
            name,
            payload,
            line,
            interrupt,
            ..
        } => {
            payload.iter().for_each(|(_, t)| n.ty(t));
            let mut x = d(name.clone(), "event", *line);
            if let Some(p) = interrupt {
                n.expr(p);
                x.note = format!("interrupt priority {}", short(p));
            }
            x
        }
        Item::Error {
            name,
            payload,
            line,
        } => {
            payload.iter().for_each(|(_, t)| n.ty(t));
            d(name.clone(), "error", *line)
        }
        Item::Kind {
            name,
            variants,
            line,
            ..
        } => {
            for (_, fs) in variants {
                fs.iter().for_each(|(_, t)| n.ty(t));
            }
            d(name.clone(), "kind", *line)
        }
        Item::Trait { name, line, .. } => d(name.clone(), "trait", *line),
        Item::Impl {
            ty,
            trait_,
            members,
            line,
            ..
        } => {
            let owner = ty_name(ty);
            let mut head = Names::default();
            head.ty(ty);
            if let Some(t) = trait_ {
                head.ty(t)
            }
            for m in members {
                let mut mn = Names::default();
                mn.names.push((owner.clone(), false));
                mn.func(m);
                out.push((d(format!("{owner}.{}", m.sig.name), "fn", m.sig.line), mn));
            }
            n.names.extend(head.names);
            d(format!("impl {owner}@{line}"), "impl", *line)
        }
        Item::Fn(f) => {
            n.func(f);
            d(f.sig.name.clone(), "fn", f.sig.line)
        }
        Item::Extern { sig, line, .. } => {
            n.params(&sig.params);
            d(sig.name.clone(), "extern", *line)
        }
        Item::Step {
            region,
            pat,
            guard,
            body,
            line,
            deadline,
        } => {
            n.names.push((region.clone(), false));
            n.pat(pat);
            if let Some(g) = guard {
                n.expr(g)
            }
            n.body(body);
            let mut x = d(format!("step {region}@{line}"), "step", *line);
            if let Some(dl) = deadline {
                n.expr(dl);
                x.note = format!("deadline {} ticks", short(dl));
            }
            x
        }
        Item::Protocol { name, steps, line } => {
            fn walk(n: &mut Names, v: &[ProtoStep]) -> usize {
                v.iter().map(|s| match s {
                    ProtoStep::Send(t) | ProtoStep::Recv(t) => { n.ty(t); 1 }
                    ProtoStep::Offer(b) | ProtoStep::Choose(b) => 1 + b.iter().map(|(_, x)| walk(n, x)).sum::<usize>(),
                    ProtoStep::Loop(x) => 1 + walk(n, x),
                    ProtoStep::Break => 1,
                }).sum()
            }
            let k = walk(&mut n, steps);
            let mut x = d(name.clone(), "protocol", *line);
            x.note = format!("session type, {k} steps");
            x
        }
        Item::Node { name, actors, line } => {
            actors.iter().for_each(|a| n.names.push((a.clone(), false)));
            let mut x = d(name.clone(), "node", *line);
            x.note = format!("partition: {}", actors.join(", "));
            x
        }
        Item::Supervisor { name, strategy, children, line } => {
            children.iter().for_each(|c| n.names.push((c.ty.clone(), false)));
            let mut x = d(name.clone(), "supervisor", *line);
            x.note = format!("supervisor {strategy}, {} children", children.len());
            x
        }
        Item::Actor {
            name,
            fields,
            handlers,
            fns,
            line,
        } => {
            fields.iter().for_each(|f| n.ty(&f.ty));
            for h in handlers {
                n.pat(&h.pat);
                if let Some(g) = &h.guard {
                    n.expr(g)
                }
                n.body(&h.body);
            }
            for f in fns {
                let mut mn = Names::default();
                mn.names.push((name.clone(), false));
                mn.func(f);
                out.push((d(format!("{name}.{}", f.sig.name), "fn", f.sig.line), mn));
            }
            d(name.clone(), "actor", *line)
        }
    };
    out.push((def, n));
}

impl FormIndex {
    /// Parse and check `program` with the reference checker and derive the
    /// index. Files that do not parse contribute nothing but their report.
    pub fn build(program: &[(String, String)]) -> FormIndex {
        let mut files: Vec<(String, String)> = program.to_vec();
        files.sort();
        let verdicts: BTreeMap<String, Option<String>> = check_program(&files, false)
            .into_iter()
            .map(|(f, v)| (f, v.err()))
            .collect();
        let mut raw: Vec<(Def, Names)> = vec![];
        for (f, src) in &files {
            if let Ok(m) = parse::parse(src) {
                for it in &m.items {
                    item_defs(f, it, &mut raw);
                }
            }
        }
        let defs: Vec<Def> = raw.iter().map(|(d, _)| d.clone()).collect();
        // Resolve by full name, then by last segment (methods by bare name).
        let mut by_name: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, d) in defs.iter().enumerate() {
            by_name.entry(d.name.as_str()).or_default().push(i);
            if let Some((_, last)) = d.name.rsplit_once('.') {
                by_name.entry(last).or_default().push(i);
            }
        }
        let mut seen = BTreeSet::new();
        let mut refs = vec![];
        for (from, (_, names)) in raw.iter().enumerate() {
            for (n, call) in &names.names {
                if let Some(ts) = by_name.get(n.as_str()) {
                    for &to in ts {
                        if to != from && seen.insert((from, to, *call)) {
                            refs.push(Ref {
                                from,
                                to,
                                call: *call,
                            });
                        }
                    }
                }
            }
        }
        let mut ix = FormIndex {
            schema: INDEX_SCHEMA.into(),
            form_hash: form_hash(program),
            defs,
            refs,
            verdicts,
            rank: vec![],
        };
        ix.rank = ix.pagerank(&[]);
        ix
    }

    pub fn find(&self, name: &str) -> Vec<usize> {
        (0..self.defs.len())
            .filter(|&i| {
                let d = &self.defs[i].name;
                d == name || d.rsplit_once('.').is_some_and(|(_, l)| l == name)
            })
            .collect()
    }

    /// Defs that `i` references (or calls, with `calls_only`).
    pub fn out_edges(&self, i: usize, calls_only: bool) -> Vec<usize> {
        self.refs
            .iter()
            .filter(|r| r.from == i && (!calls_only || r.call))
            .map(|r| r.to)
            .collect()
    }

    /// Defs that reference (or call) `i`.
    pub fn in_edges(&self, i: usize, calls_only: bool) -> Vec<usize> {
        self.refs
            .iter()
            .filter(|r| r.to == i && (!calls_only || r.call))
            .map(|r| r.from)
            .collect()
    }

    /// Personalized PageRank over the reference graph (Aider's repo map
    /// idea, on Form defs instead of tree-sitter tags). `focus` def indexes
    /// get the teleport mass; empty means uniform. Calls weigh 2, other
    /// references 1. Deterministic: 50 iterations, damping 0.85.
    pub fn pagerank(&self, focus: &[usize]) -> Vec<f64> {
        let n = self.defs.len();
        if n == 0 {
            return vec![];
        }
        let mut tele = vec![0.0; n];
        if focus.is_empty() {
            tele.iter_mut().for_each(|t| *t = 1.0 / n as f64);
        } else {
            for &f in focus {
                tele[f] += 1.0 / focus.len() as f64;
            }
        }
        let mut outw = vec![0.0; n];
        for r in &self.refs {
            outw[r.from] += if r.call { 2.0 } else { 1.0 };
        }
        let mut pr = tele.clone();
        for _ in 0..50 {
            let mut next: Vec<f64> = tele.iter().map(|t| 0.15 * t).collect();
            let mut dangling = 0.0;
            for i in 0..n {
                if outw[i] == 0.0 {
                    dangling += pr[i];
                }
            }
            for (i, t) in tele.iter().enumerate() {
                next[i] += 0.85 * dangling * t;
            }
            for r in &self.refs {
                let w = if r.call { 2.0 } else { 1.0 };
                next[r.to] += 0.85 * pr[r.from] * w / outw[r.from];
            }
            pr = next;
        }
        pr
    }

    /// Ranked repo map: defs by personalized rank, grouped by file, cut at
    /// `budget_chars`. Returns the text and the def indexes included.
    pub fn repo_map(&self, focus: &[usize], budget_chars: usize) -> (String, Vec<usize>) {
        let rank = if focus.is_empty() {
            self.rank.clone()
        } else {
            self.pagerank(focus)
        };
        let mut order: Vec<usize> = (0..self.defs.len()).collect();
        order.sort_by(|&a, &b| {
            rank[b]
                .total_cmp(&rank[a])
                .then(self.defs[a].name.cmp(&self.defs[b].name))
        });
        let mut picked = vec![];
        let mut used = 0usize;
        for i in order {
            let d = &self.defs[i];
            let cost = d.file.len() + d.kind.len() + d.name.len() + 16;
            if used + cost > budget_chars {
                break;
            }
            used += cost;
            picked.push(i);
        }
        let mut by_file: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for &i in &picked {
            by_file.entry(&self.defs[i].file).or_default().push(i);
        }
        let mut s = String::new();
        for (f, mut is) in by_file {
            is.sort_by_key(|&i| self.defs[i].line);
            s.push_str(f);
            s.push_str(":\n");
            for i in is {
                let d = &self.defs[i];
                s.push_str(&format!("  {:>4}: {} {}\n", d.line, d.kind, d.name));
            }
        }
        (s, picked)
    }
}

/// On-disk cache of derived indexes, one JSON file per Form hash.
pub struct IndexCache {
    pub dir: PathBuf,
}

impl IndexCache {
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(IndexCache {
            dir: dir.to_path_buf(),
        })
    }
    fn path(&self, hash: &str) -> PathBuf {
        self.dir.join(format!("{hash}.json"))
    }
    /// Cached index for this exact Form hash, if present and valid.
    pub fn get(&self, hash: &str) -> Option<FormIndex> {
        let s = std::fs::read_to_string(self.path(hash)).ok()?;
        let ix: FormIndex = serde_json::from_str(&s).ok()?;
        (ix.form_hash == hash && ix.schema == INDEX_SCHEMA).then_some(ix)
    }
    /// Return the cached index or build and store it. The bool is true on a hit.
    pub fn load(&self, program: &[(String, String)]) -> std::io::Result<(FormIndex, bool)> {
        let h = form_hash(program);
        if let Some(ix) = self.get(&h) {
            return Ok((ix, true));
        }
        let ix = FormIndex::build(program);
        let tmp = self.dir.join(format!(".{h}.tmp"));
        std::fs::write(&tmp, serde_json::to_vec(&ix).expect("index json"))?;
        std::fs::rename(&tmp, self.path(&h))?;
        Ok((ix, false))
    }
}
