//! Reference type and well-formedness checker (spec sections 4, 5, 7).
//! Every diagnostic names the rule it enforces.
use crate::ast::*;
use crate::types::*;
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug)]
pub enum TyDef {
    Struct {
        generics: Vec<GParam>,
        fields: Vec<Field>,
    },
    Kind {
        generics: Vec<GParam>,
        variants: Vec<(String, Vec<(String, Ty)>)>,
    },
    State {
        fields: Vec<Field>,
    },
    Actor {
        fields: Vec<Field>,
    },
    Event {
        payload: Vec<(String, Ty)>,
    },
    Error {
        payload: Vec<(String, Ty)>,
    },
    Alias {
        generics: Vec<GParam>,
        ty: Ty,
    },
    Trait {
        members: Vec<TraitMember>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cx {
    Pure,
    Task,
    Main,
    Comptime,
    Step,
    Handler,
}

#[derive(Clone, Debug)]
pub struct FnInfo {
    pub decl: FnDecl,
    pub owner: Option<(Ty, Option<String>, Vec<GParam>)>,
    pub cx: Cx,
}

#[derive(Clone, Debug)]
struct Local {
    ty: T,
    mutable: bool,
    moved: bool,
    depth: usize,
    param: bool,
    field: bool,
}

pub struct Checker {
    /// explicit type arguments of the call being checked, `f[T](x)` (R5)
    targs: Option<Vec<Ty>>,
    pub types: HashMap<String, TyDef>,
    pub fns: HashMap<String, FnInfo>,
    pub methods: Vec<(String, FnInfo)>,
    pub impls: Vec<(String, Option<String>)>,
    pub consts: HashMap<String, (Option<Ty>, Expr)>,
    pub variants: HashMap<String, (String, Vec<(String, Ty)>)>,
    pub errs: Vec<String>,
    pub sub: Subst,
    pub bin_types: HashMap<usize, T>,
    pub calls: HashMap<String, BTreeSet<String>>,
    scopes: Vec<HashMap<String, Local>>,
    pub(crate) cx: Cx,
    ret: T,
    gbounds: HashMap<String, Vec<String>>,
    self_ty: Option<T>,
    arenas: Vec<(String, usize)>,
    loops: usize,
    cur_fn: String,
    item_names: HashSet<String>,
    tailpos: bool,
    lits: Vec<(T, i128, usize)>,
    pub externs: HashMap<String, (String, usize)>,
    pub linear: HashSet<String>,
    /// Every local binding the checker declared: (name, line, type). The
    /// language server reads it for hover and go-to-definition (10.7).
    pub decls: Vec<(String, usize, T)>,
    pub(crate) device: usize,
    iter_elem: Option<T>,
    iter_ok: bool,
    /// draft 0.6b state (see check06b.rs)
    pub(crate) b6: crate::check06b::State06b,
    /// the module under check is language version 0.6 (spec 2.7)
    pub v06: bool,
    /// version 0.8 module: indexed values of 5.16 are in scope
    pub v08: bool,
    /// version 0.9 module (Appendix F, src/v09.rs)
    pub v09: bool,
    /// structs and kinds that transitively hold a Secret (5.16, W-SEC-1)
    pub secret_structs: HashSet<String>,
    /// functions and impl heads linked from imported modules (src/link.rs)
    pub imported_fns: HashSet<String>,
    pub imported_types: HashSet<String>,
}

const SAFE_REFUSED: &[&str] = &["unsafe", "Rc", "Arc", "RefCell", "transmute", "static"];
const BUILTIN_TRAITS: &[&str] = &["Eq", "Ord", "Hash", "Show", "Copy", "Data", "Merge"];
const STD_MODULES: &[&str] = &["std.text", "std.collections", "std.math"];

fn d<X: std::fmt::Display>(line: usize, rule: &str, m: X) -> String {
    format!("line {line}: {rule} {m}")
}

fn fn_cx(sig: &FnSig, owner_is_impl: bool, al: &HashMap<String, Ty>) -> Cx {
    if sig.comptime {
        return Cx::Comptime;
    }
    if sig.name == "main" && !owner_is_impl {
        return Cx::Main;
    }
    let has_chan = sig
        .params
        .iter()
        .any(|p| matches!(p, Param::Named { ty, .. } if ty_mentions_chan(ty, al)));
    if has_chan { Cx::Task } else { Cx::Pure }
}
/// Whether a type is or holds a channel. A type alias `type A = τ` is
/// transparent (5.1): A is τ, so `fn f(c: A)` with τ a channel type is a
/// task function. `al` maps alias names to their types; the depth cap
/// stops a cyclic alias (refused elsewhere) from looping.
fn ty_mentions_chan(t: &Ty, al: &HashMap<String, Ty>) -> bool {
    fn go(t: &Ty, al: &HashMap<String, Ty>, depth: usize) -> bool {
        match t {
            Ty::Chan(_) => true,
            Ty::Path(p, _) if p.len() == 1 && p[0] == "session" => true,
            Ty::Path(p, a) => {
                a.iter().any(|x| go(x, al, depth))
                    || (depth < 32 && p.len() == 1 && al.get(&p[0]).is_some_and(|x| go(x, al, depth + 1)))
            }
            Ty::Ref(_, x) | Ty::Opt(x) | Ty::Slice(x) | Ty::Region(x, _) => go(x, al, depth),
            Ty::Tuple(v) => v.iter().any(|x| go(x, al, depth)),
            _ => false,
        }
    }
    go(t, al, 0)
}

impl Default for Checker {
    fn default() -> Self {
        Self::new()
    }
}

impl Checker {
    pub fn new() -> Self {
        Checker {
            targs: None,
            types: HashMap::new(),
            fns: HashMap::new(),
            methods: vec![],
            impls: vec![],
            consts: HashMap::new(),
            variants: HashMap::new(),
            errs: vec![],
            sub: Subst::default(),
            bin_types: HashMap::new(),
            calls: HashMap::new(),
            externs: HashMap::new(),
            scopes: vec![],
            cx: Cx::Pure,
            ret: T::Unit,
            gbounds: HashMap::new(),
            self_ty: None,
            arenas: vec![],
            loops: 0,
            cur_fn: String::new(),
            item_names: HashSet::new(),
            tailpos: false,
            lits: vec![],
            linear: HashSet::new(),
            decls: vec![],
            device: 0,
            iter_elem: None,
            iter_ok: false,
            b6: Default::default(),
            v06: false,
            v08: false,
            v09: false,
            secret_structs: HashSet::new(),
            imported_fns: HashSet::new(),
            imported_types: HashSet::new(),
        }
    }
    pub(crate) fn push_scope_pub(&mut self) {
        self.scopes.push(HashMap::new());
    }
    pub(crate) fn pop_scope_pub(&mut self) {
        self.scopes.pop();
    }
    pub(crate) fn e(&mut self, s: String) {
        if !self.errs.contains(&s) {
            self.errs.push(s);
        }
    }

    fn alias_types(&self) -> HashMap<String, Ty> {
        self.types
            .iter()
            .filter_map(|(n, t)| if let TyDef::Alias { ty, .. } = t { Some((n.clone(), ty.clone())) } else { None })
            .collect()
    }

    pub fn check_module(&mut self, m: &Module, known_modules: &HashSet<String>) {
        self.v06 = m.draft06;
        self.v08 = m.v08;
        self.v09 = m.v09;
        for e in &m.v09_hygiene {
            self.e(e.clone());
        }
        if m.v08 {
            for e in crate::index08::check(m) {
                self.e(e);
            }
            self.secret_structs = crate::index08::struct_secret(m);
        }
        self.pre06b(m);
        // type aliases of this module and of the modules it links, for the
        // task-function test (5.1: an alias is the type it names)
        let mut al: HashMap<String, Ty> = self.alias_types();
        for it in &m.items {
            if let Item::TypeAlias { name, ty, .. } = it {
                al.insert(name.clone(), ty.clone());
            }
        }
        let mut seen: HashMap<String, usize> = HashMap::new();
        for it in &m.items {
            let mut decl = |c: &mut Checker, n: &str, line: usize| {
                if SAFE_REFUSED.contains(&n) {
                    c.e(d(
                        line,
                        "W-SAFE-1",
                        format!("`{n}` is refused: memory safety is by construction"),
                    ));
                }
                if seen.insert(n.to_string(), line).is_some() {
                    c.e(d(line, "W-NAME-2", format!("duplicate declaration `{n}`")));
                }
            };
            match it {
                Item::Use(p, _, line) => {
                    let s = p.join(".");
                    if !STD_MODULES.contains(&s.as_str()) && !known_modules.contains(&s) {
                        self.e(d(*line, "W-NAME-1", format!("unknown module `{s}`")));
                    }
                    if m.path.as_ref().map(|q| q.join(".")) == Some(s.clone()) {
                        self.e(d(*line, "W-MOD-2", "a module imports itself"));
                    }
                }
                Item::Const { name, ty, e, line } => {
                    decl(self, name, *line);
                    self.consts.insert(name.clone(), (ty.clone(), e.clone()));
                }
                Item::TypeAlias {
                    name,
                    generics,
                    ty,
                    pred,
                    line,
                } => {
                    decl(self, name, *line);
                    if let Some(p) = pred {
                        // W-REF-1 (5.1): a refined type is not generic, refines an
                        // integer, text or collection type, and its predicate is in
                        // the decidable fragment.
                        if !generics.is_empty() {
                            self.e(d(*line, "W-REF-1", "a refined type has no type parameters"));
                        }
                        match crate::refine::base_of(ty) {
                            None => self.e(d(*line, "W-REF-1", format!("a refined type refines an integer type, text or a collection, not {}", crate::interp::ty_str(ty)))),
                            Some(b) => {
                                if let Err(m) = crate::refine::compile(p, b) {
                                    self.e(d(*line, "W-REF-1", m));
                                }
                            }
                        }
                    }
                    self.types.insert(
                        name.clone(),
                        TyDef::Alias {
                            generics: generics.clone(),
                            ty: ty.clone(),
                        },
                    );
                }
                Item::Struct {
                    name,
                    generics,
                    fields,
                    line,
                    linear,
                } => {
                    decl(self, name, *line);
                    if *linear {
                        self.linear.insert(name.clone());
                    }
                    self.types.insert(
                        name.clone(),
                        TyDef::Struct {
                            generics: generics.clone(),
                            fields: fields.clone(),
                        },
                    );
                }
                Item::State { name, fields, line, .. } => {
                    decl(self, name, *line);
                    if fields.is_empty() {
                        self.e(d(*line, "W-STATE-1", "state has no fields"));
                    }
                    self.types.insert(
                        name.clone(),
                        TyDef::State {
                            fields: fields.clone(),
                        },
                    );
                }
                Item::Event {
                    name,
                    payload,
                    line,
                    ..
                } => {
                    decl(self, name, *line);
                    self.types.insert(
                        name.clone(),
                        TyDef::Event {
                            payload: payload.clone(),
                        },
                    );
                }
                Item::Error {
                    name,
                    payload,
                    line,
                } => {
                    decl(self, name, *line);
                    self.types.insert(
                        name.clone(),
                        TyDef::Error {
                            payload: payload.clone(),
                        },
                    );
                }
                Item::Kind {
                    name,
                    generics,
                    variants,
                    line,
                } => {
                    decl(self, name, *line);
                    for (v, p) in variants {
                        if self
                            .variants
                            .insert(v.clone(), (name.clone(), p.clone()))
                            .is_some()
                        {
                            self.e(d(*line, "W-NAME-2", format!("duplicate variant `{v}`")));
                        }
                    }
                    self.types.insert(
                        name.clone(),
                        TyDef::Kind {
                            generics: generics.clone(),
                            variants: variants.clone(),
                        },
                    );
                }
                Item::Trait {
                    name,
                    members,
                    line,
                    ..
                } => {
                    decl(self, name, *line);
                    for mb in members {
                        if let TraitMember::Fn(s) = mb
                            && s.bound.is_some()
                        {
                            self.e(d(s.line, "W-REC-1", "`bound` applies to a function with a body (6.7), not to a trait signature"));
                        }
                    }
                    self.types.insert(
                        name.clone(),
                        TyDef::Trait {
                            members: members.clone(),
                        },
                    );
                }
                Item::Actor {
                    name,
                    fields,
                    fns,
                    line,
                    ..
                } => {
                    decl(self, name, *line);
                    self.types.insert(
                        name.clone(),
                        TyDef::Actor {
                            fields: fields.clone(),
                        },
                    );
                    for f in fns {
                        self.methods.push((
                            name.clone(),
                            FnInfo {
                                decl: f.clone(),
                                owner: Some((Ty::Path(vec![name.clone()], vec![]), None, vec![])),
                                cx: Cx::Pure,
                            },
                        ));
                    }
                }
                Item::Fn(f) => {
                    decl(self, &f.sig.name, f.sig.line);
                    if f.sig.bound.is_some() && matches!(f.sig.ret, Some(Ty::Iter(_))) {
                        self.e(d(f.sig.line, "W-REC-1", "`bound` applies to a function, not to an iterator (6.7, 5.14)"));
                    }
                    let cx = fn_cx(&f.sig, false, &al);
                    self.fns.insert(
                        f.sig.name.clone(),
                        FnInfo {
                            decl: f.clone(),
                            owner: None,
                            cx,
                        },
                    );
                }
                Item::Extern { abi, sig, line } => {
                    decl(self, &sig.name, *line);
                    if sig.bound.is_some() {
                        self.e(d(*line, "W-REC-1", "`bound` applies to a function with a body (6.7), not to a foreign function"));
                    }
                    self.externs.insert(sig.name.clone(), (abi.clone(), *line));
                    self.fns.insert(
                        sig.name.clone(),
                        FnInfo {
                            decl: FnDecl {
                                sig: sig.clone(),
                                body: Body::Block(vec![]),
                            },
                            owner: None,
                            cx: Cx::Task,
                        },
                    );
                }
                Item::Impl {
                    generics,
                    ty,
                    trait_,
                    members,
                    ..
                } => {
                    let head = ty_head(ty);
                    let tr = trait_.as_ref().map(ty_head);
                    self.impls.push((head.clone(), tr.clone()));
                    for f in members {
                        let cx = fn_cx(&f.sig, true, &al);
                        self.methods.push((
                            head.clone(),
                            FnInfo {
                                decl: f.clone(),
                                owner: Some((ty.clone(), tr.clone(), generics.clone())),
                                cx,
                            },
                        ));
                    }
                }
                _ => {}
            }
        }
        self.item_names = seen.keys().cloned().collect();
        for it in &m.items {
            self.check_decl_types(it);
        }
        self.check_impls(m);
        for it in &m.items {
            if let Item::Const { name, ty, e, line } = it {
                self.begin(Cx::Comptime, T::Unit, None, name);
                let t = self.infer(e, false);
                if let Some(ty) = ty {
                    let want = self.lower(ty, *line);
                    self.expect(&t, &want, *line);
                }
                if name == "LOOP_LIMIT" {
                    let ok = matches!(ty, Some(Ty::Path(p, a)) if a.is_empty() && p.len() == 1 && p[0] == "u64");
                    let pos = matches!(e, Expr::Int(n, _) if *n > 0);
                    if !ok || !pos {
                        self.e(d(*line, "W-LOOP-1", "LOOP_LIMIT must be declared `const LOOP_LIMIT: u64 = n` with a positive literal n"));
                    }
                }
                self.end();
            }
        }
        let mut fns: Vec<FnInfo> = self.fns.values().cloned().collect();
        fns.sort_by_key(|a| a.decl.sig.line);
        for f in fns {
            self.check_fn(&f);
        }
        for (_, f) in self.methods.clone() {
            self.check_fn(&f);
        }
        self.check_form(m);
        for it in &m.items {
            match it {
                Item::Check(e) => self.check_comptime_bool(e),
                Item::Intent(_, ex, ck) => {
                    for e in ck {
                        self.check_comptime_bool(e);
                    }
                    for x in ex {
                        self.check_example(x, m);
                    }
                }
                Item::Example(x) => self.check_example(x, m),
                _ => {}
            }
        }
        self.post06b(m);
        self.check_recursion();
        // W-LEX-3: every integer literal fits the type inference gives it (default i64)
        for (t, v, l) in self.lits.clone() {
            let n = match self.sub.resolve(&t) {
                T::Int(n) => n,
                T::F64 => continue,
                _ => "i64",
            };
            let (lo, hi) = int_range(n);
            if v < lo || v > hi {
                self.e(d(l, "W-LEX-3", format!("literal {v} is outside {n}")));
            }
        }
    }

    fn check_comptime_bool(&mut self, e: &Expr) {
        self.begin(Cx::Comptime, T::Unit, None, "check");
        let t = self.infer(e, false);
        self.expect(&t, &T::Bool, e.line());
        self.end();
    }

    fn data_fields(&mut self, fs: &[(String, Ty)], line: usize, what: &str, allow_chan: bool) {
        let mut names = HashSet::new();
        for (n, t) in fs {
            if !names.insert(n.clone()) {
                self.e(d(line, "W-NAME-2", format!("duplicate field `{n}`")));
            }
            let lt = self.lower(t, line);
            if self.mentions_linear(&lt) {
                self.e(d(
                    line,
                    "W-LIN-1",
                    format!("{what} `{n}` cannot hold a linear value"),
                ));
            }
            self.data_type(&lt, line, what, allow_chan);
        }
    }

    fn check_decl_types(&mut self, it: &Item) {
        match it {
            Item::Struct {
                fields,
                line,
                generics,
                ..
            } => {
                self.gbounds = generics.iter().map(|g| (g.name.clone(), vec![])).collect();
                let fs: Vec<(String, Ty)> = fields
                    .iter()
                    .map(|f| (f.name.clone(), f.ty.clone()))
                    .collect();
                self.data_fields(&fs, *line, "struct field", false);
                for f in fields {
                    if let Some(e) = &f.init {
                        self.begin(Cx::Comptime, T::Unit, None, "struct");
                        let t = self.infer(e, false);
                        let w = self.lower(&f.ty, f.line);
                        self.expect(&t, &w, f.line);
                        self.end();
                    }
                }
                self.gbounds.clear();
            }
            Item::State { fields, line, .. } | Item::Actor { fields, line, .. } => {
                let fs: Vec<(String, Ty)> = fields
                    .iter()
                    .map(|f| (f.name.clone(), f.ty.clone()))
                    .collect();
                self.data_fields(&fs, *line, "state field (W-STATE-3)", false);
                for f in fields {
                    match &f.init {
                        None => self.e(d(
                            f.line,
                            "W-STATE-2",
                            format!("field `{}` needs a comptime initial value", f.name),
                        )),
                        Some(e) => {
                            self.begin(Cx::Comptime, T::Unit, None, "state");
                            let t = self.infer(e, false);
                            let want = self.lower(&f.ty, f.line);
                            self.expect(&t, &want, f.line);
                            self.end();
                        }
                    }
                }
            }
            Item::Event { payload, line, .. } => {
                self.data_fields(payload, *line, "event payload", true)
            }
            Item::Error { payload, line, .. } => {
                self.data_fields(payload, *line, "error payload", false)
            }
            Item::Kind {
                variants,
                line,
                generics,
                ..
            } => {
                self.gbounds = generics.iter().map(|g| (g.name.clone(), vec![])).collect();
                for (_, p) in variants {
                    self.data_fields(p, *line, "variant payload", false);
                }
                self.gbounds.clear();
            }
            _ => {}
        }
    }

    fn data_type(&mut self, t: &T, line: usize, what: &str, allow_chan: bool) {
        let bad = match t {
            T::Ref(..) => Some("a borrow"),
            T::Region(..) => Some("a region type"),
            T::Fn(..) => Some("a function type"),
            T::Slice(_) => Some("an unborrowed slice"),
            T::Chan(_) if !allow_chan => Some("a channel"),
            _ => None,
        };
        if let Some(b) = bad {
            // a state field is checked under its own rule (W-STATE-3, 9)
            let code = if what.starts_with("state field") { "W-STATE-3" } else { "W-TY-2" };
            self.e(d(line, code, format!("{what} cannot hold {b}")));
            return;
        }
        match t {
            T::Tuple(v) | T::Named(_, v) => {
                for x in v {
                    self.data_type(x, line, what, allow_chan)
                }
            }
            T::Array(a, _) | T::List(a) | T::Set(a) | T::Opt(a) => {
                self.data_type(a, line, what, allow_chan)
            }
            T::Map(a, b) => {
                self.data_type(a, line, what, allow_chan);
                self.data_type(b, line, what, allow_chan)
            }
            _ => {}
        }
    }

    fn check_impls(&mut self, m: &Module) {
        for it in &m.items {
            if let Item::Impl {
                ty,
                trait_: Some(tr),
                members,
                line,
                ..
            } = it
            {
                let tn = ty_head(tr);
                let local_trait = matches!(self.types.get(&tn), Some(TyDef::Trait { .. }));
                let local_type = self.types.contains_key(&ty_head(ty));
                if !local_trait && !BUILTIN_TRAITS.contains(&tn.as_str()) {
                    self.e(d(*line, "W-NAME-1", format!("unknown trait `{tn}`")));
                    continue;
                }
                if !local_trait && !local_type {
                    self.e(d(
                        *line,
                        "W-TRAIT-2",
                        "orphan impl: neither the trait nor the type is declared in this module",
                    ));
                }
                if let Some(TyDef::Trait { members: tm }) = self.types.get(&tn).cloned() {
                    for mem in &tm {
                        if let TraitMember::Fn(sig) = mem {
                            match members.iter().find(|f| f.sig.name == sig.name) {
                                None => self.e(d(
                                    *line,
                                    "W-TRAIT-1",
                                    format!("impl is missing `{}`", sig.name),
                                )),
                                Some(f) => {
                                    if !sig_matches(&f.sig, sig, ty) {
                                        self.e(d(
                                            f.sig.line,
                                            "W-TRAIT-1",
                                            format!(
                                                "`{}` does not match the trait signature",
                                                sig.name
                                            ),
                                        ))
                                    }
                                }
                            }
                        }
                    }
                    for f in members {
                        if !tm
                            .iter()
                            .any(|mm| matches!(mm, TraitMember::Fn(s) if s.name == f.sig.name))
                        {
                            self.e(d(
                                f.sig.line,
                                "W-TRAIT-1",
                                format!("`{}` is not a member of trait `{tn}`", f.sig.name),
                            ));
                        }
                    }
                }
            }
        }
        let mut seen = HashSet::new();
        for (h, t) in self.impls.clone() {
            if let Some(t) = t
                && !seen.insert((h.clone(), t.clone()))
            {
                self.e(format!("line 0: W-TRAIT-2 two impls of `{t}` for `{h}`"));
            }
        }
    }

    pub fn lower(&mut self, t: &Ty, line: usize) -> T {
        match t {
            Ty::Path(p, args) if p.len() == 1 && (p[0] == "session" || p[0] == "lvar") => {
                self.lower06b(&p[0], args, line)
            }
            Ty::Path(p, args)
                if self.v08 && p.len() == 1 && crate::index08::INDEXED.contains(&p[0].as_str()) =>
            {
                // 5.16 indexed types (well-formedness is W-CTX-1/W-SEC-1/W-CAP-1 in index08)
                let name = |t: &Ty| match t {
                    Ty::Path(q, _) if q.len() == 1 => q[0].clone(),
                    _ => String::new(),
                };
                match p[0].as_str() {
                    "Text" => T::Named("Text".into(), vec![T::Named(args.first().map(name).unwrap_or_default(), vec![])]),
                    "Secret" => {
                        let i = args.first().map(|a| self.lower(a, line)).unwrap_or(T::Never);
                        T::Named("Secret".into(), vec![i])
                    }
                    "Cap" => T::Named("Cap".into(), vec![T::Named(args.first().map(name).unwrap_or_default(), vec![])]),
                    // Guarded[τ, A] is τ for typing; writes are W-CAP-2
                    _ => args.first().map(|a| self.lower(a, line)).unwrap_or(T::Never),
                }
            }
            Ty::Path(p, args) => {
                let largs: Vec<T> = args.iter().map(|a| self.lower(a, line)).collect();
                if p.len() == 2 && p[0] == "Self" {
                    return T::Param(format!("Self.{}", p[1]));
                }
                if p.len() != 1 {
                    self.e(d(
                        line,
                        "W-NAME-1",
                        format!("unknown type `{}`", p.join(".")),
                    ));
                    return T::Never;
                }
                let n = p[0].as_str();
                if let Some(i) = INTS.iter().find(|x| **x == n) {
                    return T::Int(i);
                }
                let nargs = largs.len();
                let mut want = |c: &mut Checker, k: usize| {
                    if nargs != k {
                        c.e(d(line, "W-TY-1", format!("`{n}` takes {k} type arguments")));
                    }
                };
                let g = |i: usize| largs.get(i).cloned().unwrap_or(T::Never);
                match n {
                    "bool" => T::Bool,
                    "f64" => T::F64,
                    "f32" => T::F32,
                    "text" => T::Text,
                    "List" | "Set" | "Map" if largs.iter().any(|a| self.mentions_linear(a)) => {
                        self.e(d(
                            line,
                            "W-LIN-1",
                            format!("a linear value cannot sit inside {n}"),
                        ));
                        T::Never
                    }
                    "List" => {
                        want(self, 1);
                        T::List(Box::new(g(0)))
                    }
                    "Set" => {
                        want(self, 1);
                        T::Set(Box::new(g(0)))
                    }
                    "Map" => {
                        want(self, 2);
                        T::Map(Box::new(g(0)), Box::new(g(1)))
                    }
                    "Box" => {
                        want(self, 1);
                        T::Named("Box".into(), largs.clone())
                    }
                    "Self" => self.self_ty.clone().unwrap_or_else(|| {
                        self.e(d(line, "W-NAME-1", "`Self` outside an impl or trait"));
                        T::Never
                    }),
                    _ if SAFE_REFUSED.contains(&n) => {
                        self.e(d(
                            line,
                            "W-SAFE-1",
                            format!("`{n}` is refused: memory safety is by construction"),
                        ));
                        T::Never
                    }
                    _ if self.gbounds.contains_key(n) => T::Param(n.into()),
                    _ => match self.types.get(n).cloned() {
                        Some(TyDef::Alias { ty, .. }) => self.lower(&ty, line),
                        Some(TyDef::Trait { .. }) => {
                            self.e(d(line, "W-TY-1", format!("trait `{n}` used as a type")));
                            T::Never
                        }
                        Some(_) => T::Named(n.into(), largs.clone()),
                        None => {
                            self.e(d(line, "W-NAME-1", format!("unknown type `{n}`")));
                            T::Never
                        }
                    },
                }
            }
            Ty::Unit => T::Unit,
            Ty::Tuple(v) => T::Tuple(v.iter().map(|x| self.lower(x, line)).collect()),
            Ty::Slice(x) => T::Slice(Box::new(self.lower(x, line))),
            Ty::Array(x, n) => {
                let len = self.const_len(n);
                let t = self.lower(x, line);
                if self.mentions_linear(&t) {
                    self.e(d(
                        line,
                        "W-LIN-1",
                        "a linear value cannot sit inside an array",
                    ));
                }
                T::Array(Box::new(t), len)
            }
            Ty::Chan(x) => {
                let t = self.lower(x, line);
                if self.mentions_linear(&t) {
                    self.e(d(
                        line,
                        "W-LIN-1",
                        "a linear value cannot be sent on a channel",
                    ));
                }
                T::Chan(Box::new(t))
            }
            Ty::Iter(x) => T::Iter(Box::new(self.lower(x, line))),
            Ty::Fn(a, r) => T::Fn(
                a.iter().map(|x| self.lower(x, line)).collect(),
                Box::new(self.lower(r, line)),
            ),
            Ty::Ref(m, x) => {
                let inner = self.lower(x, line);
                if matches!(inner, T::Ref(..)) {
                    self.e(d(line, "W-TY-2", "borrow of a borrow"));
                }
                T::Ref(*m, Box::new(inner))
            }
            Ty::Opt(x) => {
                let inner = self.lower(x, line);
                if matches!(inner, T::Opt(_)) {
                    self.e(d(line, "W-TY-2", "nested option"));
                }
                T::Opt(Box::new(inner))
            }
            Ty::Union(a, b) => {
                let ok = self.lower(a, line);
                let err = self.lower(b, line);
                let is_err = match &err {
                    T::Named(n, _) => matches!(
                        self.types.get(n),
                        Some(TyDef::Error { .. }) | Some(TyDef::Kind { .. })
                    ),
                    T::Never => true,
                    _ => false,
                };
                if !is_err {
                    self.e(d(
                        line,
                        "W-TY-1",
                        format!("`{err}` is not an error or a kind of errors"),
                    ));
                }
                T::Union(Box::new(ok), Box::new(err))
            }
            Ty::Region(x, r) => {
                if !self.arenas.iter().any(|(a, _)| a == r) {
                    self.e(d(line, "W-NAME-1", format!("no open arena `{r}`")));
                }
                T::Region(Box::new(self.lower(x, line)), r.clone())
            }
        }
    }
    fn const_len(&self, e: &Expr) -> Option<u64> {
        match e {
            Expr::Int(n, _) => Some(*n as u64),
            Expr::Name(n, _) => match self.consts.get(n) {
                Some((_, Expr::Int(v, _))) => Some(*v as u64),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn is_copy(&self, t: &T) -> bool {
        match self.sub.resolve(t) {
            T::Named(n, _) if self.linear.contains(&n) => false,
            T::Iter(_) => false,
            T::Bool
            | T::Int(_)
            | T::F64
            | T::F32
            | T::Unit
            | T::Chan(_)
            | T::Fn(..)
            | T::Var(_)
            | T::Never => true,
            T::Ref(m, _) => !m,
            T::Tuple(v) => v.iter().all(|x| self.is_copy(x)),
            T::Array(a, _) | T::Opt(a) | T::Region(a, _) => self.is_copy(&a),
            T::Named(n, args) => match self.types.get(&n) {
                Some(TyDef::Actor { .. }) => true,
                Some(TyDef::Struct { fields, .. }) if args.is_empty() => {
                    fields.iter().all(|f| self.ty_copy_syntactic(&f.ty))
                }
                Some(TyDef::Kind { variants, .. }) if args.is_empty() => variants
                    .iter()
                    .all(|(_, p)| p.iter().all(|(_, t)| self.ty_copy_syntactic(t))),
                Some(TyDef::Event { payload }) | Some(TyDef::Error { payload }) => {
                    payload.iter().all(|(_, t)| self.ty_copy_syntactic(t))
                }
                _ => false,
            },
            _ => false,
        }
    }
    fn ty_copy_syntactic(&self, t: &Ty) -> bool {
        match t {
            Ty::Path(p, a) if p.len() == 1 => {
                let n = p[0].as_str();
                if INTS.contains(&n) || n == "bool" || n == "f64" {
                    return true;
                }
                if ["text", "List", "Map", "Set", "Box"].contains(&n) || !a.is_empty() {
                    return false;
                }
                match self.types.get(n) {
                    Some(TyDef::Struct { fields, .. }) => {
                        fields.iter().all(|f| self.ty_copy_syntactic(&f.ty))
                    }
                    Some(TyDef::Kind { variants, .. }) => variants
                        .iter()
                        .all(|(_, p)| p.iter().all(|(_, t)| self.ty_copy_syntactic(t))),
                    Some(TyDef::Alias { ty, .. }) => self.ty_copy_syntactic(ty),
                    Some(TyDef::Actor { .. }) => true,
                    _ => false,
                }
            }
            Ty::Unit | Ty::Chan(_) | Ty::Fn(..) => true,
            Ty::Tuple(v) => v.iter().all(|x| self.ty_copy_syntactic(x)),
            Ty::Array(x, _) | Ty::Opt(x) => self.ty_copy_syntactic(x),
            _ => false,
        }
    }
}

impl Checker {
    fn begin(&mut self, cx: Cx, ret: T, self_ty: Option<T>, name: &str) {
        self.scopes = vec![HashMap::new()];
        self.cx = cx;
        self.ret = ret;
        self.self_ty = self_ty;
        self.arenas.clear();
        self.loops = 0;
        self.cur_fn = name.to_string();
    }
    fn end(&mut self) {
        self.scopes.clear();
        self.gbounds.clear();
        self.self_ty = None;
    }
    fn depth(&self) -> usize {
        self.scopes.len()
    }
    fn mentions_linear(&self, t: &T) -> bool {
        match t {
            T::Named(n, a) => self.linear.contains(n) || a.iter().any(|x| self.mentions_linear(x)),
            T::Tuple(v) => v.iter().any(|x| self.mentions_linear(x)),
            T::Array(a, _)
            | T::List(a)
            | T::Set(a)
            | T::Slice(a)
            | T::Opt(a)
            | T::Chan(a)
            | T::Region(a, _)
            | T::Iter(a) => self.mentions_linear(a),
            T::Map(a, b) | T::Union(a, b) => self.mentions_linear(a) || self.mentions_linear(b),
            _ => false,
        }
    }
    /// W-LIN-1: every linear local declared at depth >= `from` must be consumed here.
    fn linear_live(&mut self, line: usize, from: usize, wh: &str) {
        let mut bad = vec![];
        for (i, s) in self.scopes.iter().enumerate() {
            if i + 1 < from {
                continue;
            }
            for (n, l) in s {
                if !l.moved
                    && !matches!(l.ty, T::Ref(..))
                    && self.mentions_linear(&self.sub.resolve(&l.ty))
                {
                    bad.push(n.clone());
                }
            }
        }
        bad.sort();
        for n in bad {
            self.e(d(
                line,
                "W-LIN-1",
                format!("linear `{n}` is not consumed {wh}"),
            ));
        }
    }
    /// W-LIN-1: all non-diverging branches must consume the same linear locals.
    fn linear_branches(&mut self, sets: &[HashSet<String>], line: usize) {
        if sets.len() < 2 {
            return;
        }
        let all: HashSet<String> = sets.iter().flatten().cloned().collect();
        let mut bad: Vec<String> = all
            .into_iter()
            .filter(|n| {
                !sets.iter().all(|s| s.contains(n))
                    && self
                        .lookup(n)
                        .is_some_and(|l| self.mentions_linear(&self.sub.resolve(&l.ty)))
            })
            .collect();
        bad.sort();
        for n in bad {
            self.e(d(
                line,
                "W-LIN-1",
                format!("linear `{n}` is consumed on some branches but not others"),
            ));
        }
    }
    /// W-DEV-1: values in a device region are fixed-size.
    fn device_value(&mut self, t: &T, line: usize) {
        if self.device == 0 {
            return;
        }
        fn growable(t: &T) -> bool {
            match t {
                T::List(_)
                | T::Map(..)
                | T::Set(_)
                | T::Text
                | T::Slice(_)
                | T::Chan(_)
                | T::Iter(_) => true,
                T::Tuple(v) | T::Named(_, v) => v.iter().any(growable),
                T::Array(a, _) | T::Opt(a) | T::Ref(_, a) | T::Region(a, _) => growable(a),
                T::Union(a, b) => growable(a) || growable(b),
                _ => false,
            }
        }
        let t = self.sub.resolve(t);
        if growable(&t) {
            self.e(d(
                line,
                "W-DEV-1",
                format!("a device region holds fixed-size values only, found {t}"),
            ));
        }
    }
    pub(crate) fn declare(&mut self, n: &str, ty: T, mutable: bool, line: usize, param: bool) {
        if SAFE_REFUSED.contains(&n) {
            self.e(d(line, "W-SAFE-1", format!("`{n}` is refused")));
        }
        if self.variants.contains_key(n)
            || matches!(
                self.types.get(n),
                Some(TyDef::Event { .. }) | Some(TyDef::Error { .. })
            )
        {
            self.e(d(
                line,
                "W-PAT-1",
                format!("binding `{n}` has the name of a variant, event or error"),
            ));
        }
        let last = self.scopes.len() - 1;
        for (i, s) in self.scopes.iter().enumerate() {
            if let Some(l) = s.get(n)
                && (l.param || l.field || i == last)
            {
                self.e(d(
                    line,
                    "W-NAME-2",
                    format!("`{n}` shadows a parameter, field or binding in the same scope"),
                ));
                break;
            }
        }
        if self.fns.contains_key(n) || self.consts.contains_key(n) {
            self.e(d(
                line,
                "W-NAME-2",
                format!("`{n}` shadows a module-level name"),
            ));
        }
        let depth = self.depth();
        self.decls.push((n.to_string(), line, ty.clone()));
        self.scopes.last_mut().unwrap().insert(
            n.to_string(),
            Local {
                ty,
                mutable,
                moved: false,
                depth,
                param,
                field: false,
            },
        );
    }
    fn lookup(&self, n: &str) -> Option<Local> {
        self.scopes.iter().rev().find_map(|s| s.get(n).cloned())
    }
    fn set_moved(&mut self, n: &str, moved: bool) {
        for s in self.scopes.iter_mut().rev() {
            if let Some(l) = s.get_mut(n) {
                l.moved = moved;
                return;
            }
        }
    }
    fn moved_set(&self) -> HashSet<String> {
        self.scopes
            .iter()
            .flat_map(|s| s.iter().filter(|(_, l)| l.moved).map(|(n, _)| n.clone()))
            .collect()
    }
    fn restore_moved(&mut self, m: &HashSet<String>) {
        for s in self.scopes.iter_mut() {
            for (n, l) in s.iter_mut() {
                l.moved = m.contains(n);
            }
        }
    }

    fn check_extern(&mut self, sig: &FnSig, abi: &str, line: usize) {
        // F.5.5: a version 0.9 module also declares `extern "model"`, a C-ABI
        // function whose cost is chosen by the outside
        if abi != "C" && !(self.v09 && abi == "model") {
            self.e(d(
                line,
                "W-FFI-1",
                format!("unknown ABI \"{abi}\" (only \"C\")"),
            ));
        }
        if sig.comptime || !sig.generics.is_empty() {
            self.e(d(
                line,
                "W-FFI-1",
                "a foreign function cannot be comptime or generic",
            ));
        }
        let lin1: HashSet<String> = self.linear.iter().filter(|n| matches!(self.types.get(*n), Some(TyDef::Struct { fields, generics }) if generics.is_empty() && fields.len() == 1 && matches!(&fields[0].ty, Ty::Path(p, a) if a.is_empty() && p.len() == 1 && (INTS.contains(&p[0].as_str()) || p[0] == "bool")))).cloned().collect();
        let scalar = |t: &T| {
            matches!(t, T::Bool | T::Int(_) | T::F64 | T::F32 | T::Unit)
                || matches!(t, T::Named(n, a) if a.is_empty() && lin1.contains(n))
        };
        for p in &sig.params {
            match p {
                Param::Named { ty, comptime, .. } => {
                    if *comptime {
                        self.e(d(line, "W-FFI-1", "a foreign parameter cannot be comptime"));
                    }
                    let t = self.lower(ty, line);
                    let ok = match &t {
                        t if scalar(t) && !matches!(t, T::Unit) => true,
                        T::Ref(false, x) => {
                            matches!(&**x, T::Text) || matches!(&**x, T::Slice(e) if scalar(e))
                        }
                        T::Ref(true, x) => matches!(&**x, T::Slice(e) if scalar(e)),
                        _ => false,
                    };
                    if !ok {
                        self.e(d(line, "W-FFI-1", format!("foreign parameter type {t} is not bool, an integer, f64, &text, &[scalar] or &mut [scalar]")));
                    }
                }
                _ => self.e(d(line, "W-FFI-1", "a foreign function has no self")),
            }
        }
        if let Some(r) = &sig.ret {
            let t = self.lower(r, line);
            if !scalar(&t) {
                self.e(d(
                    line,
                    "W-FFI-1",
                    format!("foreign result type {t} is not bool, an integer, f64 or ()"),
                ));
            }
        }
    }

    fn check_fn(&mut self, f: &FnInfo) {
        let sig = f.decl.sig.clone();
        // linked from an imported module: checked there (src/link.rs)
        match &f.owner {
            None if self.imported_fns.contains(&sig.name) => return,
            Some((t, _, _)) if self.imported_types.contains(&ty_head(t)) => return,
            _ => {}
        }
        if let Some((abi, line)) = self.externs.get(&sig.name).cloned()
            && f.owner.is_none()
        {
            self.check_extern(&sig, &abi, line);
            return;
        }
        self.gbounds.clear();
        if let Some((_, _, g)) = &f.owner {
            for x in g {
                self.gbounds
                    .insert(x.name.clone(), x.bounds.iter().map(ty_head).collect());
            }
        }
        for g in &sig.generics {
            self.gbounds
                .insert(g.name.clone(), g.bounds.iter().map(ty_head).collect());
        }
        let self_t = match &f.owner {
            Some((ty, _, _)) => Some(self.lower(ty, sig.line)),
            None => None,
        };
        self.self_ty = self_t.clone();
        let mut ret = sig
            .ret
            .as_ref()
            .map(|r| self.lower(r, sig.line))
            .unwrap_or(T::Unit);
        self.iter_elem = None;
        if let T::Iter(e) = ret.clone() {
            if f.cx != Cx::Pure {
                self.e(d(
                    sig.line,
                    "W-ITER-1",
                    format!("iterator function `{}` must be pure", sig.name),
                ));
            }
            if self.mentions_linear(&e) {
                self.e(d(
                    sig.line,
                    "W-LIN-1",
                    "an iterator cannot yield linear values",
                ));
            }
            self.iter_elem = Some(*e);
            ret = T::Unit;
            if matches!(f.decl.body, Body::Arrow(_)) {
                self.e(d(
                    sig.line,
                    "W-ITER-1",
                    "an iterator function needs a block body with yield",
                ));
            }
        }
        let gb = self.gbounds.clone();
        self.begin(
            f.cx,
            ret.clone(),
            self_t.clone(),
            &match &f.owner {
                Some((ty, _, _)) => format!("{}.{}", ty_head(ty), sig.name),
                None => sig.name.clone(),
            },
        );
        self.gbounds = gb;
        if f.cx == Cx::Main && (!sig.params.is_empty() || sig.ret.is_some()) {
            self.e(d(
                sig.line,
                "W-TY-1",
                "`main` takes no parameters and returns nothing",
            ));
        }
        for g in &sig.generics {
            if g.comptime {
                self.declare(&g.name.clone(), T::Int("usize"), false, sig.line, true);
            }
        }
        for p in &sig.params {
            match p {
                Param::Named { name, ty, .. } => {
                    let t = self.lower(ty, sig.line);
                    self.declare(name, t, false, sig.line, true);
                }
                Param::SelfVal => {
                    let t = self_t.clone().unwrap_or(T::Never);
                    self.declare("self", t, false, sig.line, true)
                }
                Param::SelfRef(m) => {
                    let t = T::Ref(*m, Box::new(self_t.clone().unwrap_or(T::Never)));
                    self.declare("self", t, false, sig.line, true)
                }
            }
        }
        if matches!(ret, T::Ref(..))
            && !sig.params.iter().any(|p| {
                matches!(
                    p,
                    Param::Named {
                        ty: Ty::Ref(..),
                        ..
                    } | Param::SelfRef(_)
                )
            })
        {
            self.e(d(
                sig.line,
                "W-OWN-3",
                "a returned borrow must derive from a borrowed parameter (R-RET)",
            ));
        }
        match &f.decl.body {
            Body::Arrow(e) => {
                self.check_return_expr(e, &ret, sig.line);
                let t = self.infer(e, true);
                self.expect(&t, &ret, sig.line);
            }
            Body::Block(b) => {
                let div = self.block_t(b, true);
                if !div && !matches!(self.sub.resolve(&ret), T::Unit) {
                    self.e(d(
                        sig.line,
                        "W-TY-1",
                        format!("`{}` can reach its end without returning {ret}", sig.name),
                    ));
                }
                if !div {
                    self.linear_live(sig.line, 0, "at the end of the function");
                }
            }
        }
        self.iter_elem = None;
        self.end();
    }

    fn check_return_expr(&mut self, e: &Expr, ret: &T, line: usize) {
        if matches!(ret, T::Ref(..)) {
            match place_root(e).and_then(|r| self.lookup(&r)) {
                Some(l) if l.param && matches!(l.ty, T::Ref(..)) => {}
                _ => self.e(d(
                    line,
                    "W-OWN-3",
                    "returned borrow is not derived from a borrowed parameter",
                )),
            }
        }
        if !self.arenas.is_empty() {
            let t = self.infer_quiet(e);
            let t = self.sub.resolve(&t);
            for (r, _) in self.arenas.clone() {
                if contains_region(&t, &r) {
                    self.e(d(
                        line,
                        "W-REG-1",
                        format!("a value in arena `{r}` escapes by return"),
                    ));
                }
            }
        }
    }
    pub(crate) fn infer_quiet(&mut self, e: &Expr) -> T {
        let (n, mv) = (self.errs.len(), self.moved_set());
        let t = self.infer(e, false);
        self.errs.truncate(n);
        self.restore_moved(&mv);
        t
    }

    fn check_form(&mut self, m: &Module) {
        let mut events: Vec<String> = self
            .types
            .iter()
            .filter(|(_, t)| matches!(t, TyDef::Event { .. }))
            .map(|(n, _)| n.clone())
            .collect();
        events.sort();
        // where each event is declared, for diagnostics about it (an
        // imported event has no line in this module: line 1)
        let ev_line: HashMap<String, usize> = m
            .items
            .iter()
            .filter_map(|it| match it {
                Item::Event { name, line, .. } => Some((name.clone(), *line)),
                _ => None,
            })
            .collect();
        let at = |ev: &String| ev_line.get(ev).copied().unwrap_or(1).max(1);
        let mut handled: HashSet<String> = HashSet::new();
        let mut unguarded: HashMap<(String, String), usize> = HashMap::new();
        let mut step_events: HashSet<String> = HashSet::new();
        for it in &m.items {
            match it {
                Item::Step {
                    region,
                    pat,
                    guard,
                    body,
                    line,
                    ..
                } => {
                    let fields = match self.types.get(region).cloned() {
                        Some(TyDef::State { fields }) => fields,
                        _ => {
                            self.e(d(
                                *line,
                                "W-STEP-1",
                                format!("`{region}` is not a declared state"),
                            ));
                            continue;
                        }
                    };
                    if let Some(ev) =
                        self.check_handler(region, &fields, pat, guard, body, *line, Cx::Step)
                    {
                        handled.insert(ev.clone());
                        step_events.insert(ev.clone());
                        if let Some(prev) = unguarded.get(&(region.clone(), ev.clone())) {
                            self.e(d(*line, "W-STEP-4", format!("unreachable: the unguarded step at line {prev} already handles {region} on {ev}")));
                        }
                        if guard.is_none() && pat_irrefutable_payload(pat) {
                            unguarded.insert((region.clone(), ev.clone()), *line);
                        }
                    }
                }
                Item::Actor {
                    name,
                    fields,
                    handlers,
                    ..
                } => {
                    for h in handlers {
                        if let Some(ev) = self.check_handler(
                            name,
                            fields,
                            &h.pat,
                            &h.guard,
                            &h.body,
                            h.line,
                            Cx::Handler,
                        ) {
                            handled.insert(ev);
                        }
                    }
                }
                _ => {}
            }
        }
        for ev in &events {
            if !handled.contains(ev) {
                self.e(d(
                    at(ev),
                    "W-STEP-2",
                    format!("event `{ev}` is handled by no step or actor: δ would not be total"),
                ));
            }
            if let Some(TyDef::Event { payload }) = self.types.get(ev).cloned()
                && payload.iter().any(|(_, t)| ty_mentions_chan(t, &self.alias_types()))
                && step_events.contains(ev)
            {
                self.e(d(
                    at(ev),
                    "W-TY-2",
                    format!("event `{ev}` carries a channel but is handled by a state step"),
                ));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_handler(
        &mut self,
        region: &str,
        fields: &[Field],
        pat: &Pat,
        guard: &Option<Expr>,
        body: &Body,
        line: usize,
        cx: Cx,
    ) -> Option<String> {
        self.begin(cx, T::Unit, None, &format!("step {region}"));
        for f in fields {
            let t = self.lower(&f.ty, f.line);
            self.scopes[0].insert(
                f.name.clone(),
                Local {
                    ty: t,
                    mutable: true,
                    moved: false,
                    depth: 1,
                    param: false,
                    field: true,
                },
            );
        }
        self.scopes.push(HashMap::new());
        let ev = match pat {
            Pat::Path(p) | Pat::Ctor(p, _)
                if p.len() == 1 && matches!(self.types.get(&p[0]), Some(TyDef::Event { .. })) =>
            {
                p[0].clone()
            }
            _ => {
                self.e(d(line, "W-STEP-1", "a step must match a declared event"));
                self.end();
                return None;
            }
        };
        self.bind_pat(pat, &T::Named(ev.clone(), vec![]), line, false);
        if let Some(g) = guard {
            let t = self.infer(g, false);
            self.expect(&t, &T::Bool, line);
        }
        match body {
            Body::Arrow(e) => {
                if fields.len() != 1 {
                    self.e(d(
                        line,
                        "W-STEP-3",
                        "the arrow form needs a region with exactly one field",
                    ));
                }
                let t = self.infer(e, true);
                if let Some(f) = fields.first() {
                    let ft = self.lower(&f.ty, line);
                    self.expect(&t, &ft, line);
                }
            }
            Body::Block(b) => {
                self.block(b);
            }
        }
        self.end();
        Some(ev)
    }

    fn check_example(&mut self, x: &Example, m: &Module) {
        for (ev, args) in &x.events {
            match self.types.get(ev).cloned() {
                Some(TyDef::Event { payload }) => {
                    self.begin(Cx::Comptime, T::Unit, None, "example");
                    // 5.16: in an example the host writes an attested capability
                    // as `()` and a secret as its plain value.
                    let host = |t: &Ty| match t {
                        Ty::Path(p, a) if self.v08 && p.len() == 1 && p[0] == "Cap" && a.len() == 1 => Ty::Unit,
                        Ty::Path(p, a) if self.v08 && p.len() == 1 && p[0] == "Secret" && a.len() == 1 => a[0].clone(),
                        t => t.clone(),
                    };
                    let fs: Vec<(String, Ty, bool)> = payload
                        .iter()
                        .map(|(n, t)| (n.clone(), host(t), false))
                        .collect();
                    self.ctor_args(ev, &fs, args, x.line);
                    self.end();
                }
                _ => self.e(d(
                    x.line,
                    "W-NAME-1",
                    format!("`{ev}` is not a declared event (names are case-sensitive)"),
                )),
            }
        }
        if let Some(r) = &x.refused
            && !r.starts_with("trap(")
            && !matches!(self.types.get(r), Some(TyDef::Error { .. }))
        {
            self.e(d(
                x.line,
                "W-NAME-1",
                format!("`{r}` is not a declared error"),
            ));
        }
        if let Some(v) = &x.value {
            let states: Vec<Vec<Field>> = m
                .items
                .iter()
                .filter_map(|i| {
                    if let Item::State { fields, .. } = i {
                        Some(fields.clone())
                    } else {
                        None
                    }
                })
                .collect();
            self.begin(Cx::Comptime, T::Unit, None, "example");
            for fs in &states {
                for f in fs {
                    let t = self.lower(&f.ty, f.line);
                    self.scopes[0].insert(
                        f.name.clone(),
                        Local {
                            ty: t,
                            mutable: false,
                            moved: false,
                            depth: 1,
                            param: false,
                            field: true,
                        },
                    );
                }
            }
            let t = self.infer(v, false);
            let single = if states.len() == 1 && states[0].len() == 1 {
                Some(self.lower(&states[0][0].ty, x.line))
            } else {
                None
            };
            match single {
                Some(ft) if self.coerce(&t, &ft) => {}
                _ => self.expect(&t, &T::Bool, x.line),
            }
            self.end();
        }
    }

    fn check_recursion(&mut self) {
        let mut g = self.calls.clone();
        // every recursive cycle, a function calling itself included: a
        // lexicographic tuple of summed measures and a rank (6.7)
        for scc in sccs(&g) {
            if scc.len() == 1 && !g.get(&scc[0]).is_some_and(|e| e.contains(&scc[0])) {
                continue;
            }
            let vs = self.variants.clone();
            let members: BTreeSet<String> = scc.iter().cloned().collect();
            let mut fs = vec![];
            for n in &scc {
                let Some(fi) = self.fns.get(n).cloned() else { continue };
                let arity = fi.decl.sig.params.iter().filter(|p| matches!(p, Param::Named { .. })).count();
                fs.push((n.clone(), arity, rec_sites(&fi.decl, &|c| members.contains(c), &|c| vs.contains_key(c))));
            }
            // a cycle through something other than plain functions (trait
            // methods) is left to the cycle report below
            if fs.len() != scc.len() {
                continue;
            }
            // `bound n` (6.7): every cycle of calls through an unbounded
            // function only would need the measure; one through a bounded
            // function ends by its count
            let decls: Vec<FnDecl> = scc.iter().filter_map(|n| self.fns.get(n).map(|f| f.decl.clone())).collect();
            let bounded: Vec<bool> = decls.iter().map(|f| f.sig.bound.is_some()).collect();
            if let Err(l) = lex_measure(&fs) {
                let mut free: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); scc.len()];
                for (k, n) in scc.iter().enumerate() {
                    if bounded[k] {
                        continue;
                    }
                    for m in g.get(n).into_iter().flatten() {
                        if let Some(j) = scc.iter().position(|x| x == m)
                            && !bounded[j]
                        {
                            free[k].insert(j);
                        }
                    }
                }
                if !acyclic(&free) {
                    self.e(d(l, "W-REC-1", format!("recursion through {} has no decreasing measure (6.7): no layers of summed measures, each kept by every call not yet done and shrunk by at least one, leave the remaining calls without a cycle, and a cycle of its calls passes through no function declared with `bound`", scc.join(", "))));
                }
            }
            if bounded.iter().any(|b| *b) {
                for f in &decls {
                    let mut hit = None;
                    if let Body::Block(b) = &f.body {
                        walk_stmts(b, &mut |s| {
                            if let Stmt::Arena(r, None, _, l) = s
                                && (r == "$par" || r == "$forpar")
                                && hit.is_none()
                            {
                                hit = Some(*l);
                            }
                        });
                    }
                    if let Some(l) = hit {
                        self.e(d(l, "W-REC-1", format!("`par` in `{}`, which is in a recursive cycle with a function declared with `bound` (6.7): the count of active calls is per task and branch", f.sig.name)));
                    }
                }
            }
            for n in &scc {
                if let Some(es) = g.get_mut(n) {
                    es.retain(|m| !members.contains(m));
                }
            }
        }
        let mut st: HashMap<String, u8> = HashMap::new();
        fn dfs(
            n: &str,
            g: &HashMap<String, BTreeSet<String>>,
            st: &mut HashMap<String, u8>,
            path: &mut Vec<String>,
        ) -> Option<Vec<String>> {
            match st.get(n) {
                Some(1) => {
                    let i = path.iter().position(|x| x == n).unwrap_or(0);
                    let mut c = path[i..].to_vec();
                    c.push(n.to_string());
                    return Some(c);
                }
                Some(2) => return None,
                _ => {}
            }
            st.insert(n.to_string(), 1);
            path.push(n.to_string());
            if let Some(es) = g.get(n) {
                for m in es {
                    if let Some(c) = dfs(m, g, st, path) {
                        return Some(c);
                    }
                }
            }
            path.pop();
            st.insert(n.to_string(), 2);
            None
        }
        let mut keys: Vec<String> = g.keys().cloned().collect();
        keys.sort();
        for k in keys {
            if let Some(c) = dfs(&k, &g, &mut st, &mut vec![]) {
                self.e(format!(
                    "line 0: W-REC-1 recursion through {}",
                    c.join(" -> ")
                ));
                break;
            }
        }
    }

    pub(crate) fn expect(&mut self, actual: &T, expected: &T, line: usize) {
        let a = self.sub.shallow(actual);
        let x = self.sub.shallow(expected);
        let ok = match (&a, &x) {
            (T::Never, _) => true,
            (T::Union(..), T::Union(..)) => self.sub.unify(&a, &x),
            // A literal's variable (integer or list literal) can only become
            // an integer or a list, never a union, so it is the success
            // type (`return 0` in a `-> i64 ! E` function). Other variables
            // may still resolve to a union and are unified as is.
            (_, T::Union(t, _))
                if match a {
                    T::Var(i) => self.sub.intlit[i] || self.sub.listlit[i].is_some(),
                    _ => true,
                } =>
            {
                self.coerce(&a, t)
            }
            (T::Region(t, _), _) if !matches!(x, T::Region(..)) => self.coerce(t, &x),
            (_, T::Region(t, _)) if !matches!(a, T::Region(..)) => self.coerce(&a, t),
            _ => self.coerce(&a, &x),
        };
        if !ok {
            let code = self.mix_code(&a, &x, true);
            let (a, x) = (self.sub.resolve(&a), self.sub.resolve(&x));
            self.e(d(line, code, format!("expected {x}, found {a}")));
        }
    }
    fn coerce(&mut self, a: &T, x: &T) -> bool {
        let a = self.sub.shallow(a);
        let x = self.sub.shallow(x);
        match (&a, &x) {
            (T::Ref(m2, inner), T::Ref(m, xs)) if !*m || *m2 => {
                let inner = self.sub.shallow(inner);
                let xs = self.sub.shallow(xs);
                if let (T::Var(i), false) = (&inner, matches!(xs, T::Array(..) | T::Var(_)))
                    && let Some((e, _)) = self.sub.listlit[*i].clone()
                {
                    self.sub.unify(&inner, &T::List(Box::new(e)));
                }
                let inner = self.sub.shallow(&inner);
                match (&inner, &xs) {
                    (T::List(e) | T::Array(e, _) | T::Slice(e), T::Slice(f)) => {
                        self.sub.unify(e, f)
                    }
                    (T::Named(n, b), _)
                        if n == "Box" && !matches!(&xs, T::Named(m, _) if m == "Box") =>
                    {
                        let b0 = b[0].clone();
                        self.sub.unify(&b0, &xs)
                    }
                    _ => self.sub.unify(&inner, &xs),
                }
            }
            _ => self.sub.unify(&a, &x),
        }
    }
}

impl Checker {
    /// Returns true if the block always diverges (return, fail, break, continue).
    pub(crate) fn block(&mut self, b: &[Stmt]) -> bool {
        self.block_t(b, false)
    }
    /// `tail`: the block is in tail position of a function body (spec 5.3 [Tail]).
    fn block_t(&mut self, b: &[Stmt], tail: bool) -> bool {
        self.scopes.push(HashMap::new());
        let mut div = false;
        for (i, s) in b.iter().enumerate() {
            if div {
                self.e(d(
                    crate::check06b::stmt_line(s).max(1),
                    "W-MATCH-2",
                    "unreachable statement after return, fail, break or continue",
                ));
                break;
            }
            self.tailpos = tail && i + 1 == b.len();
            div = self.stmt(s);
        }
        self.tailpos = false;
        if !div {
            let dd = self.depth();
            self.linear_live(0, dd, "at the end of its block");
        }
        self.scopes.pop();
        div
    }

    fn assignable(&mut self, place: &Expr, line: usize) -> Option<String> {
        let root = match place_root(place) {
            Some(r) if is_place(place) => r,
            _ => {
                self.e(d(
                    line,
                    "W-PLACE",
                    "left side of an assignment is not a place",
                ));
                return None;
            }
        };
        match self.lookup(&root) {
            Some(l) => {
                if !l.mutable && !matches!(l.ty, T::Ref(true, _)) {
                    self.e(d(
                        line,
                        "W-PLACE",
                        format!("`{root}` is not mutable (declare it with var)"),
                    ));
                }
                if self.cx == Cx::Comptime && l.field {
                    self.e(d(line, "W-CT-1", "comptime code cannot write state"));
                }
                Some(root)
            }
            None => {
                self.e(d(
                    line,
                    "W-PLACE",
                    format!("`{root}` is not a local variable or field"),
                ));
                None
            }
        }
    }

    fn region_escape(&mut self, root: &str, t: &T, line: usize) {
        if let Some(l) = self.lookup(root) {
            let t = self.sub.resolve(t);
            for (r, depth) in self.arenas.clone() {
                if l.depth < depth && contains_region(&t, &r) {
                    self.e(d(
                        line,
                        "W-REG-1",
                        format!("a value in arena `{r}` escapes into `{root}`"),
                    ));
                }
            }
        }
    }

    pub(crate) fn stmt(&mut self, s: &Stmt) -> bool {
        let tail = std::mem::take(&mut self.tailpos);
        match s {
            Stmt::Expr(e) if tail && !matches!(self.sub.resolve(&self.ret.clone()), T::Unit) => {
                let ret = self.ret.clone();
                self.check_return_expr(e, &ret, e.line());
                let t = self.infer(e, true);
                self.expect(&t, &ret, e.line());
                true
            }
            Stmt::Expr(e) => {
                let t = self.infer(e, false);
                if matches!(self.sub.resolve(&t), T::Union(..)) {
                    self.e(d(
                        e.line(),
                        "W-TY-1",
                        "an error union must be handled with try or catch",
                    ));
                } else if self.v09
                    && !matches!(self.sub.resolve(&t), T::Unit | T::Never | T::Var(_))
                {
                    self.e(d(
                        e.line(),
                        "W-DROP-1",
                        format!(
                            "an expression statement whose value is {} is refused; bind it to `_` to drop it on purpose",
                            self.sub.resolve(&t)
                        ),
                    ));
                }
                false
            }
            Stmt::Assign(op, place, rhs) => {
                let line = rhs.line();
                let root = self.assignable(place, line);
                let pt = if let Expr::Name(n, _) = place {
                    self.lookup(n).map(|l| l.ty).unwrap_or(T::Never)
                } else {
                    self.infer(place, false)
                };
                let target = match self.sub.resolve(&pt) {
                    T::Ref(true, x) => *x,
                    o => o,
                };
                let rt = self.infer(rhs, true);
                if op == "=" {
                    self.expect(&rt, &target, line);
                } else {
                    self.arith_op(&op[..op.len() - 1], &target, &rt, line, rhs);
                }
                if contains_ref(&self.sub.resolve(&rt))
                    && !matches!(self.sub.resolve(&target), T::Ref(..))
                {
                    self.e(d(line, "W-OWN-3", "a borrow cannot be stored"));
                }
                if let Some(r) = root {
                    self.region_escape(&r, &rt, line);
                    if let Expr::Name(n, _) = place {
                        self.set_moved(n, false);
                    }
                }
                false
            }
            Stmt::Send(_, v) if self.device > 0 => {
                self.e(d(v.line(), "W-DEV-1", "send inside a device region"));
                false
            }
            Stmt::Send(ch, v) => {
                let line = v.line();
                if matches!(self.cx, Cx::Pure | Cx::Comptime) {
                    self.e(d(line, "W-EFF-1", "send in a pure function"));
                }
                let ct = self.infer(ch, false);
                let vt = self.infer(v, true);
                match self.sub.resolve(&peel(&ct)) {
                    T::Chan(e) => self.expect(&vt, &e, line),
                    T::Named(s, _) if s == "session" => self.session_send_ty(ch, &vt, line),
                    T::Named(a, _) if matches!(self.types.get(&a), Some(TyDef::Actor { .. })) => {
                        match self.sub.resolve(&vt) {
                            T::Named(ev, _)
                                if matches!(self.types.get(&ev), Some(TyDef::Event { .. })) => {}
                            other => self.e(d(
                                line,
                                "W-TY-1",
                                format!("an actor receives events, found {other}"),
                            )),
                        }
                    }
                    other => self.e(d(
                        line,
                        "W-TY-1",
                        format!("`<-` needs a channel or actor, found {other}"),
                    )),
                }
                if contains_ref(&self.sub.resolve(&vt)) {
                    self.e(d(line, "W-OWN-3", "a message cannot hold a borrow"));
                }
                false
            }
            Stmt::Let(p, ty, e, line) => {
                let mut t = self.infer(e, true);
                if let Some(ty) = ty {
                    let want = self.lower(ty, *line);
                    self.expect(&t, &want, *line);
                    t = want;
                }
                if matches!(self.sub.resolve(&t), T::Union(..)) {
                    self.e(d(
                        *line,
                        "W-TY-1",
                        "an error union must be handled with try or catch",
                    ));
                }
                if !pat_irrefutable(p, self) {
                    self.e(d(*line, "W-MATCH-1", "let needs an irrefutable pattern"));
                }
                self.device_value(&t, *line);
                self.bind_pat(p, &t, *line, false);
                false
            }
            Stmt::Var(n, ty, e, line) => {
                let mut t = self.infer(e, true);
                if let Some(ty) = ty {
                    let want = self.lower(ty, *line);
                    self.expect(&t, &want, *line);
                    t = want;
                }
                if contains_ref(&self.sub.resolve(&t)) {
                    self.e(d(*line, "W-OWN-3", "a var cannot hold a borrow"));
                }
                self.device_value(&t, *line);
                self.declare(n, t, true, *line, false);
                false
            }
            Stmt::If(arms, els) => {
                let mut out: HashSet<String> = HashSet::new();
                let mut all_div = true;
                let mut sets: Vec<HashSet<String>> = vec![];
                for (c, b) in arms {
                    let t = self.infer(c, false);
                    self.expect(&peel(&t), &T::Bool, c.line());
                    let after_c = self.moved_set();
                    let div = self.block_t(b, tail);
                    if !div {
                        out.extend(self.moved_set());
                        sets.push(self.moved_set());
                    }
                    all_div &= div;
                    self.restore_moved(&after_c);
                }
                match els {
                    Some(b) => {
                        let div = self.block_t(b, tail);
                        if !div {
                            out.extend(self.moved_set());
                            sets.push(self.moved_set());
                        }
                        all_div &= div;
                    }
                    None => {
                        all_div = false;
                        out.extend(self.moved_set());
                        sets.push(self.moved_set());
                    }
                }
                let line0 = arms.first().map(|a| a.0.line()).unwrap_or(0);
                self.linear_branches(&sets, line0);
                self.restore_moved(&out);
                all_div
            }
            Stmt::While(c, b, body) => {
                let t = self.infer(c, false);
                self.expect(&peel(&t), &T::Bool, c.line());
                if let Some(b) = b {
                    let bt = self.infer(b, false);
                    self.expect(&bt, &T::Int("u64"), b.line());
                }
                self.loop_body(body);
                false
            }
            Stmt::Loop(b, body) => {
                if let Some(b) = b {
                    let bt = self.infer(b, false);
                    self.expect(&bt, &T::Int("u64"), b.line());
                }
                self.loop_body(body);
                false
            }
            Stmt::For(p, it, body, line) => {
                self.iter_ok = true;
                let t = self.infer(it, false);
                self.iter_ok = false;
                let elem = self.elem_type(&t, *line, is_place(it));
                self.scopes.push(HashMap::new());
                if !pat_irrefutable(p, self) {
                    self.e(d(*line, "W-MATCH-1", "for needs an irrefutable pattern"));
                }
                self.bind_pat(p, &elem, *line, false);
                if let Some(r) = place_root(it)
                    && writes_root(body, &r)
                {
                    self.e(d(
                        *line,
                        "W-OWN-2",
                        format!(
                            "`{r}` is borrowed by the for loop and cannot be modified inside it"
                        ),
                    ));
                }
                let outer = self.depth() - 1;
                self.loop_body_d(body, outer);
                self.scopes.pop();
                false
            }
            Stmt::Match(e, arms, line) => {
                let t0 = self.infer(e, false);
                let t = if is_place(e)
                    && !self.is_copy(&t0)
                    && !matches!(self.sub.resolve(&t0), T::Ref(..))
                {
                    T::Ref(false, Box::new(t0))
                } else {
                    t0
                };
                let before = self.moved_set();
                let mut out: HashSet<String> = HashSet::new();
                let mut all_div = true;
                let mut msets: Vec<HashSet<String>> = vec![];
                for a in arms {
                    self.scopes.push(HashMap::new());
                    self.bind_pat(&a.pat, &t, a.line, true);
                    if let Some(g) = &a.guard {
                        let gt = self.infer(g, false);
                        self.expect(&peel(&gt), &T::Bool, a.line);
                    }
                    let div = self.block_t(&a.body, tail);
                    self.scopes.pop();
                    if !div {
                        out.extend(self.moved_set());
                        msets.push(self.moved_set());
                    }
                    all_div &= div;
                    self.restore_moved(&before);
                }
                self.linear_branches(&msets, *line);
                self.restore_moved(&out);
                self.exhaustive(&t, arms, *line);
                all_div && !arms.is_empty()
            }
            Stmt::Return(e, line) => {
                if matches!(self.cx, Cx::Step | Cx::Handler) {
                    self.e(d(*line, "W-TY-1", "return in a step body"));
                }
                let ret = self.ret.clone();
                match e {
                    Some(e) => {
                        self.check_return_expr(e, &ret, *line);
                        let t = self.infer(e, true);
                        self.expect(&t, &ret, *line);
                    }
                    None => self.expect(&T::Unit, &ret, *line),
                }
                self.linear_live(*line, 0, "at return");
                true
            }
            Stmt::Break(l) | Stmt::Continue(l) => {
                if self.loops == 0 {
                    self.e(d(*l, "W-SYN-1", "break or continue outside a loop"));
                }
                true
            }
            Stmt::Pass => false,
            Stmt::Emit(e) if self.device > 0 => {
                self.e(d(e.line(), "W-DEV-1", "emit inside a device region"));
                false
            }
            Stmt::Spawn(e) if self.device > 0 => {
                self.e(d(e.line(), "W-DEV-1", "spawn inside a device region"));
                false
            }
            Stmt::Select(_, line) if self.device > 0 => {
                self.e(d(*line, "W-DEV-1", "select inside a device region"));
                false
            }
            Stmt::Emit(e) => {
                if matches!(self.cx, Cx::Pure | Cx::Comptime) {
                    self.e(d(e.line(), "W-EFF-1", "emit in a pure function"));
                }
                let t = self.infer(e, true);
                if contains_ref(&self.sub.resolve(&t)) {
                    self.e(d(e.line(), "W-OWN-3", "emitted values cannot hold borrows"));
                }
                if self.v08 && self.holds_secret(&self.sub.resolve(&t)) {
                    self.e(d(e.line(), "W-SEC-1", "an emitted value cannot hold a secret (5.16)"));
                }
                false
            }
            Stmt::Fail(e) => {
                let line = e.line();
                let t = self.infer(e, true);
                let t = self.sub.resolve(&t);
                match &t {
                    T::Named(n, _)
                        if matches!(
                            self.types.get(n),
                            Some(TyDef::Error { .. }) | Some(TyDef::Kind { .. })
                        ) => {}
                    _ => self.e(d(line, "W-TY-1", format!("fail needs an error, found {t}"))),
                }
                let ret = self.sub.resolve(&self.ret.clone());
                match (self.cx, ret) {
                    (Cx::Step | Cx::Handler | Cx::Main | Cx::Comptime, _) => {}
                    (_, T::Union(_, et)) => {
                        if !self.sub.unify(&t, &et) {
                            self.e(d(
                                line,
                                "W-TY-1",
                                format!("`{t}` is not in this function's error set {et}"),
                            ));
                        }
                    }
                    _ => self.e(d(
                        line,
                        "W-TY-1",
                        "fail in a function that does not return an error union",
                    )),
                }
                self.linear_live(line, 0, "at fail");
                true
            }
            Stmt::Spawn(e) => {
                let line = e.line();
                if !matches!(self.cx, Cx::Main | Cx::Task) {
                    self.e(d(line, "W-EFF-1", "spawn outside main or a task function"));
                } else if self.v06 && self.cx == Cx::Task && self.b6.scope_depth == 0 {
                    self.e(d(line, "W-SCOPE-1", "spawn outside a scope (only fn main is an implicit scope)"));
                }
                if self.b6.atomic_depth > 0 {
                    self.e(d(line, "W-ATOM-1", "spawn inside atomic"));
                }
                match e {
                    Expr::Call(f, args, _, _) => {
                        if let Expr::Name(n, _) = &**f {
                            if self.externs.contains_key(n) {
                                self.e(d(
                                    line,
                                    "W-SPAWN-1",
                                    format!("`{n}` is a foreign function and cannot be spawned"),
                                ));
                            } else if !matches!(self.fns.get(n).map(|f| f.cx), Some(Cx::Task)) {
                                self.e(d(
                                    line,
                                    "W-SPAWN-1",
                                    format!("`{n}` is not a task function (it takes no channel)"),
                                ));
                            }
                        }
                        for a in args {
                            if matches!(a.e, Expr::Borrow(..)) {
                                self.e(d(
                                    line,
                                    "W-SPAWN-1",
                                    "spawned tasks cannot receive borrows",
                                ));
                            }
                            let t = self.infer_quiet(&a.e);
                            let t = self.sub.resolve(&t);
                            if contains_ref(&t)
                                || self.arenas.iter().any(|(r, _)| contains_region(&t, r))
                            {
                                self.e(d(
                                    line,
                                    "W-SPAWN-1",
                                    "spawned tasks cannot receive borrows or arena values",
                                ));
                            }
                        }
                        let _ = self.infer(e, true);
                    }
                    _ => self.e(d(line, "W-SYN-1", "spawn needs a call")),
                }
                false
            }
            Stmt::Select(arms, line) => {
                if !matches!(self.cx, Cx::Main | Cx::Task) {
                    self.e(d(
                        *line,
                        "W-EFF-1",
                        "select outside main or a task function",
                    ));
                }
                let (mut nd, mut na) = (0, 0);
                let before = self.moved_set();
                for a in arms {
                    self.scopes.push(HashMap::new());
                    let body = match a {
                        SelArm::Recv(p, c, b) => {
                            let ct = self.infer(c, false);
                            match self.sub.resolve(&peel(&ct)) {
                                T::Chan(e) => self.bind_pat(p, &T::Opt(e), *line, true),
                                o => self.e(d(
                                    *line,
                                    "W-TY-1",
                                    format!("recv needs a channel, found {o}"),
                                )),
                            }
                            b
                        }
                        SelArm::Send(v, c, b) => {
                            let ct = self.infer(c, false);
                            let vt = self.infer(v, true);
                            match self.sub.resolve(&peel(&ct)) {
                                T::Chan(e) => self.expect(&vt, &e, *line),
                                o => self.e(d(
                                    *line,
                                    "W-TY-1",
                                    format!("send needs a channel, found {o}"),
                                )),
                            }
                            b
                        }
                        SelArm::After(n, b) => {
                            na += 1;
                            let t = self.infer(n, false);
                            self.expect(&t, &T::Int("u64"), *line);
                            b
                        }
                        SelArm::Default(b) => {
                            nd += 1;
                            b
                        }
                    };
                    self.block(body);
                    self.scopes.pop();
                    self.restore_moved(&before);
                }
                if nd > 1 || na > 1 {
                    self.e(d(
                        *line,
                        "W-SYN-1",
                        "select takes at most one default and one after arm",
                    ));
                }
                false
            }
            Stmt::Yield(e, line) => {
                if self.device > 0 {
                    self.e(d(*line, "W-DEV-1", "yield inside a device region"));
                }
                match self.iter_elem.clone() {
                    None => {
                        self.e(d(*line, "W-ITER-1", "yield outside an iterator function"));
                        let _ = self.infer(e, true);
                    }
                    Some(el) => {
                        let t = self.infer(e, true);
                        self.expect(&t, &el, *line);
                        if contains_ref(&self.sub.resolve(&t)) {
                            self.e(d(*line, "W-OWN-3", "a yielded value cannot hold a borrow"));
                        }
                    }
                }
                false
            }
            Stmt::Device(r, bnd, body, line) => {
                if let Some(b) = bnd {
                    let t = self.infer(b, false);
                    self.expect(&t, &T::Int("u64"), *line);
                }
                let depth = self.depth() + 1;
                self.arenas.push((r.clone(), depth));
                self.device += 1;
                let div = self.block(body);
                self.device -= 1;
                self.arenas.pop();
                div
            }
            Stmt::Arena(r, None, body, line) if r.starts_with('$') => {
                self.conc06(r, body, *line)
            }
            Stmt::Arena(r, bnd, body, line) => {
                if let Some(b) = bnd {
                    let t = self.infer(b, false);
                    self.expect(&t, &T::Int("u64"), *line);
                }
                let depth = self.depth() + 1;
                self.arenas.push((r.clone(), depth));
                let div = self.block(body);
                self.arenas.pop();
                div
            }
            Stmt::Comptime(body) => {
                let cx = self.cx;
                self.cx = Cx::Comptime;
                let div = self.block(body);
                self.cx = cx;
                div
            }
            Stmt::Check(Expr::Call(f, a, _, _)) if a.len() == 2 && matches!(&**f, Expr::Name(n, _) if n == "@contract") => {
                let t = self.infer(&a[1].e, false);
                self.expect(&t, &T::Bool, a[1].e.line());
                false
            }
            Stmt::Check(e) => {
                let t = self.infer(e, false);
                self.expect(&t, &T::Bool, e.line());
                false
            }
        }
    }

    fn loop_body(&mut self, body: &[Stmt]) {
        let d0 = self.depth();
        self.loop_body_d(body, d0)
    }
    /// Names declared at depth >= `outer` are fresh per iteration and are not checked.
    fn loop_body_d(&mut self, body: &[Stmt], outer: usize) {
        let before = self.moved_set();
        self.loops += 1;
        self.block(body);
        self.loops -= 1;
        let after = self.moved_set();
        for n in after.difference(&before) {
            if self
                .lookup(n)
                .is_some_and(|l| l.depth <= outer && self.mentions_linear(&l.ty))
            {
                self.e(format!(
                    "line 0: W-LIN-1 linear `{n}` is moved inside a loop"
                ));
                continue;
            }
            if self.lookup(n).is_some_and(|l| l.depth <= outer)
                && mentions_name(body, n)
                && !assigns_name(body, n)
            {
                self.e(format!("line 0: W-OWN-1 `{n}` is moved inside a loop and used again in the next iteration"));
            }
        }
    }

    fn elem_type(&mut self, t: &T, line: usize, borrowed: bool) -> T {
        let rt = self.sub.resolve(t);
        let by_ref = borrowed || matches!(rt, T::Ref(..));
        let wrap = |c: &Checker, e: T| {
            if by_ref && !c.is_copy(&e) {
                T::Ref(false, Box::new(e))
            } else {
                e
            }
        };
        match peel(&rt) {
            T::List(e) | T::Array(e, _) | T::Slice(e) | T::Set(e) => wrap(self, *e),
            T::Map(k, v) => wrap(self, T::Tuple(vec![*k, *v])),
            T::Text => {
                if by_ref {
                    T::Ref(false, Box::new(T::Text))
                } else {
                    T::Text
                }
            }
            T::Named(n, a) if n == "Range" => a[0].clone(),
            T::Iter(e) => *e,
            T::Var(_) => self.sub.fresh(),
            other => {
                self.e(d(line, "W-TY-1", format!("cannot iterate over {other}")));
                T::Never
            }
        }
    }

    #[allow(clippy::only_used_in_recursion)] // kept in the signature: the rule (spec 4) is stated per binding mode
    fn bind_pat(&mut self, p: &Pat, t: &T, line: usize, through_ref: bool) {
        let rt = self.sub.resolve(t);
        let by_ref = matches!(rt, T::Ref(..));
        let comp = |c: &Checker, x: T| {
            if by_ref && !c.is_copy(&x) {
                T::Ref(false, Box::new(x))
            } else {
                x
            }
        };
        let base = peel(&rt);
        match p {
            Pat::Wild => {}
            Pat::Lit(e) => {
                let lt = self.infer(e, false);
                if !self.coerce(&lt, &base) {
                    let l = self.sub.resolve(&lt);
                    self.e(d(
                        line,
                        "W-TY-1",
                        format!("pattern of type {l} cannot match {base}"),
                    ));
                }
            }
            Pat::Range(a, _, b) => {
                let at = self.infer(a, false);
                let bt = self.infer(b, false);
                if !(self.sub.unify(&at, &base) && self.sub.unify(&bt, &base)) {
                    self.e(d(line, "W-TY-1", "range pattern type mismatch"));
                }
            }
            Pat::Tuple(ps) => match base {
                T::Tuple(ts) if ts.len() == ps.len() => {
                    for (q, x) in ps.iter().zip(ts) {
                        let x = comp(self, x);
                        self.bind_pat(q, &x, line, through_ref)
                    }
                }
                T::Var(_) => {
                    let ts: Vec<T> = ps.iter().map(|_| self.sub.fresh()).collect();
                    self.sub.unify(&base, &T::Tuple(ts.clone()));
                    for (q, x) in ps.iter().zip(ts) {
                        self.bind_pat(q, &x, line, through_ref)
                    }
                }
                other => self.e(d(
                    line,
                    "W-TY-1",
                    format!("tuple pattern cannot match {other}"),
                )),
            },
            Pat::Alt(ps) => {
                for q in ps {
                    if pat_binds(q, self) {
                        self.e(d(line, "W-TY-1", "alternatives cannot bind names"));
                    }
                    self.bind_pat(q, t, line, through_ref);
                }
            }
            Pat::Path(path) if path.len() == 1 && self.label_pat(&base, &path[0], line) => {}
            Pat::Path(path) => {
                let n = &path[path.len() - 1];
                if path.len() == 1 && !self.is_ctor_name(n) {
                    if n == "some" {
                        self.e(d(line, "W-TY-1", "`some` needs a pattern"));
                        return;
                    }
                    self.declare(n, rt.clone(), false, line, false);
                    return;
                }
                if let Some(p) = self.payload_of(n)
                    && !p.is_empty()
                {
                    self.e(d(line, "W-TY-1", format!("`{n}` has fields; write {n}(…)")));
                }
                let ct = self.ctor_type(n, line, &base);
                if !self.sub.unify(&ct, &base) {
                    let b = self.sub.resolve(&base);
                    self.e(d(line, "W-TY-1", format!("pattern `{n}` cannot match {b}")));
                }
            }
            Pat::Ctor(path, args) => {
                let n = &path[path.len() - 1];
                if n == "some" && path.len() == 1 {
                    let inner = match base {
                        T::Opt(x) => *x,
                        T::Var(_) => {
                            let v = self.sub.fresh();
                            self.sub.unify(&base, &T::Opt(Box::new(v.clone())));
                            v
                        }
                        other => {
                            self.e(d(line, "W-TY-1", format!("`some` cannot match {other}")));
                            T::Never
                        }
                    };
                    if args.len() != 1 {
                        self.e(d(line, "W-TY-1", "`some` takes one pattern"));
                        return;
                    }
                    let inner = comp(self, inner);
                    self.bind_pat(&args[0].1, &inner, line, through_ref);
                    return;
                }
                let payload = match self.payload_of(n) {
                    Some(p) => p,
                    None => {
                        self.e(d(
                            line,
                            "W-NAME-1",
                            format!("`{n}` is not a variant, event or error"),
                        ));
                        return;
                    }
                };
                let ct = self.ctor_type(n, line, &base);
                if !self.sub.unify(&ct, &base) {
                    let b = self.sub.resolve(&base);
                    self.e(d(line, "W-TY-1", format!("pattern `{n}` cannot match {b}")));
                }
                if args.len() != payload.len() {
                    self.e(d(
                        line,
                        "W-TY-1",
                        format!(
                            "`{n}` has {} fields, pattern has {}",
                            payload.len(),
                            args.len()
                        ),
                    ));
                }
                for (i, (label, q)) in args.iter().enumerate() {
                    let field = match label {
                        Some(l) => payload.iter().find(|(f, _)| f == l).cloned(),
                        None => payload.get(i).cloned(),
                    };
                    match field {
                        Some((_, fty)) => {
                            let ft = self.lower(&fty, line);
                            let ft = comp(self, ft);
                            self.bind_pat(q, &ft, line, through_ref)
                        }
                        None => self.e(d(
                            line,
                            "W-NAME-1",
                            format!("`{n}` has no field `{}`", label.clone().unwrap_or_default()),
                        )),
                    }
                }
            }
        }
    }
    pub fn is_ctor_name(&self, n: &str) -> bool {
        self.variants.contains_key(n)
            || matches!(
                self.types.get(n),
                Some(TyDef::Event { .. }) | Some(TyDef::Error { .. })
            )
    }
    fn payload_of(&self, n: &str) -> Option<Vec<(String, Ty)>> {
        if let Some((_, p)) = self.variants.get(n) {
            return Some(p.clone());
        }
        match self.types.get(n) {
            Some(TyDef::Event { payload }) | Some(TyDef::Error { payload }) => {
                Some(payload.clone())
            }
            _ => None,
        }
    }
    fn ctor_type(&mut self, n: &str, line: usize, hint: &T) -> T {
        if let Some((k, _)) = self.variants.get(n).cloned() {
            let ngen = match self.types.get(&k) {
                Some(TyDef::Kind { generics, .. }) => generics.len(),
                _ => 0,
            };
            let args = match hint {
                T::Named(m, a) if *m == k => a.clone(),
                _ => (0..ngen).map(|_| self.sub.fresh()).collect(),
            };
            return T::Named(k, args);
        }
        if self.types.contains_key(n) {
            return T::Named(n.to_string(), vec![]);
        }
        self.e(d(line, "W-NAME-1", format!("unknown constructor `{n}`")));
        T::Never
    }

    fn exhaustive(&mut self, t: &T, arms: &[Arm], line: usize) {
        let base = peel(&self.sub.resolve(t));
        // labels of a session offer: coverage is checked by the session pass
        if matches!(&base, T::Named(n, _) if n.starts_with("$labels:")) {
            return;
        }
        let mut covered_all = false;
        let mut covered: HashSet<String> = HashSet::new();
        for a in arms {
            if covered_all {
                self.e(d(a.line, "W-MATCH-2", "unreachable arm"));
                continue;
            }
            if a.guard.is_some() {
                continue;
            }
            let alts: Vec<&Pat> = match &a.pat {
                Pat::Alt(v) => v.iter().collect(),
                p => vec![p],
            };
            for p in alts {
                match p {
                    Pat::Wild => covered_all = true,
                    Pat::Path(path) if path.len() == 1 && !self.is_ctor_name(&path[0]) => {
                        covered_all = true
                    }
                    Pat::Path(path) => {
                        if !covered.insert(path.last().unwrap().clone()) {
                            self.e(d(a.line, "W-MATCH-2", "unreachable arm"));
                        }
                    }
                    Pat::Ctor(path, args) if args.iter().all(|(_, q)| pat_irrefutable(q, self)) => {
                        if !covered.insert(path.last().unwrap().clone()) {
                            self.e(d(a.line, "W-MATCH-2", "unreachable arm"));
                        }
                    }
                    Pat::Lit(Expr::None(_)) => {
                        covered.insert("none".into());
                    }
                    Pat::Lit(Expr::Bool(b, _)) => {
                        covered.insert(b.to_string());
                    }
                    Pat::Tuple(ps) if ps.iter().all(|q| pat_irrefutable(q, self)) => {
                        covered_all = true
                    }
                    _ => {}
                }
            }
        }
        if covered_all {
            return;
        }
        let need: Vec<String> = match &base {
            T::Named(k, _) => match self.types.get(k) {
                Some(TyDef::Kind { variants, .. }) => {
                    variants.iter().map(|v| v.0.clone()).collect()
                }
                _ => vec![k.clone()],
            },
            T::Opt(_) => vec!["some".into(), "none".into()],
            T::Bool => vec!["true".into(), "false".into()],
            _ => vec!["_".into()],
        };
        let missing: Vec<String> = need.into_iter().filter(|n| !covered.contains(n)).collect();
        if !missing.is_empty() {
            self.e(d(
                line,
                "W-MATCH-1",
                format!("match is not exhaustive: missing {}", missing.join(", ")),
            ));
        }
    }
}

pub fn ty_head(t: &Ty) -> String {
    match t {
        Ty::Path(p, _) => p.join("."),
        Ty::Slice(_) => "Slice".into(),
        Ty::Ref(_, x) => ty_head(x),
        _ => "_".into(),
    }
}
pub fn place_root(e: &Expr) -> Option<String> {
    match e {
        Expr::Name(n, _) => Some(n.clone()),
        Expr::SelfRef(_) => Some("self".into()),
        Expr::Member(x, _, _)
        | Expr::TupleMember(x, _, _)
        | Expr::Index(x, _, _)
        | Expr::Slice(x, _, _, _) => place_root(x),
        Expr::Borrow(_, x, _) => place_root(x),
        _ => None,
    }
}
pub fn is_place(e: &Expr) -> bool {
    match e {
        Expr::Name(..) | Expr::SelfRef(_) => true,
        Expr::Member(x, _, _) | Expr::TupleMember(x, _, _) | Expr::Index(x, _, _) => is_place(x),
        _ => false,
    }
}
/// Structural recursion (spec 6.7, rule W-REC-1 as revised in draft 0.4):
/// a self-call is admitted when, for some parameter position i, the argument
/// is (a borrow of) a name bound inside a constructor pattern of a `match`
/// whose scrutinee is parameter i or a name already known to be smaller
/// than it. Returns the line of the first self-call that is not structural.
pub(crate) fn stmt_direct_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) | Stmt::Spawn(e) | Stmt::Check(e) => vec![e],
        Stmt::Assign(_, a, b) | Stmt::Send(a, b) => vec![a, b],
        Stmt::Let(_, _, e, _)
        | Stmt::Var(_, _, e, _)
        | Stmt::For(_, e, _, _)
        | Stmt::Match(e, _, _) => vec![e],
        Stmt::If(arms, _) => arms.iter().map(|(c, _)| c).collect(),
        Stmt::While(c, b, _) => {
            let mut v = vec![c];
            if let Some(b) = b {
                v.push(b);
            }
            v
        }
        Stmt::Loop(Some(b), _) => vec![b],
        Stmt::Return(Some(e), _) => vec![e],
        Stmt::Arena(_, Some(b), _, _) | Stmt::Device(_, Some(b), _, _) => vec![b],
        Stmt::Yield(e, _) => vec![e],
        Stmt::Select(arms, _) => arms
            .iter()
            .flat_map(|a| match a {
                SelArm::Recv(_, e, _) | SelArm::After(e, _) => vec![e],
                SelArm::Send(a, b, _) => vec![a, b],
                SelArm::Default(_) => vec![],
            })
            .collect(),
        _ => vec![],
    }
}
fn pat_binders(p: &Pat, under_ctor: bool, out: &mut Vec<String>, is_ctor: &dyn Fn(&str) -> bool) {
    match p {
        Pat::Path(v) if v.len() == 1 && !is_ctor(&v[0]) => {
            if under_ctor {
                out.push(v[0].clone());
            }
        }
        Pat::Ctor(_, a) => {
            for (_, q) in a {
                pat_binders(q, true, out, is_ctor)
            }
        }
        Pat::Tuple(v) | Pat::Alt(v) => {
            for q in v {
                pat_binders(q, under_ctor, out, is_ctor)
            }
        }
        _ => {}
    }
}
fn strip_borrow(e: &Expr) -> &Expr {
    match e {
        Expr::Borrow(_, x, _) => strip_borrow(x),
        _ => e,
    }
}
pub(crate) fn walk_stmts(b: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in b {
        f(s);
        match s {
            Stmt::If(arms, els) => {
                for (_, x) in arms {
                    walk_stmts(x, f);
                }
                if let Some(x) = els {
                    walk_stmts(x, f);
                }
            }
            Stmt::While(_, _, x)
            | Stmt::Loop(_, x)
            | Stmt::For(_, _, x, _)
            | Stmt::Arena(_, _, x, _)
            | Stmt::Device(_, _, x, _)
            | Stmt::Comptime(x) => walk_stmts(x, f),
            Stmt::Match(_, arms, _) => {
                for a in arms {
                    walk_stmts(&a.body, f)
                }
            }
            Stmt::Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(_, _, x)
                        | SelArm::Send(_, _, x)
                        | SelArm::After(_, x)
                        | SelArm::Default(x) => walk_stmts(x, f),
                    }
                }
            }
            _ => {}
        }
    }
}
pub(crate) fn mutating_method(m: &str) -> bool {
    [
        "append", "insert", "remove", "pop", "clear", "push", "close", "sort",
    ]
    .contains(&m)
}
fn expr_writes(e: &Expr, r: &str) -> bool {
    let mut hit = false;
    walk_expr(e, &mut |x| {
        if let Expr::Call(f, _, _, _) = x
            && let Expr::Member(o, m, _) = &**f
            && mutating_method(m)
            && place_root(o).as_deref() == Some(r)
        {
            hit = true;
        }
    });
    hit
}
pub fn writes_root(b: &[Stmt], r: &str) -> bool {
    let mut hit = false;
    walk_stmts(b, &mut |s| match s {
        Stmt::Assign(_, p, e) => {
            if place_root(p).as_deref() == Some(r) || expr_writes(e, r) {
                hit = true;
            }
        }
        Stmt::Expr(e) if expr_writes(e, r) => {
            hit = true;
        }
        _ => {}
    });
    hit
}
pub fn walk_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    use Expr::*;
    match e {
        Tuple(v, _) | List(v, _) => {
            for x in v {
                walk_expr(x, f)
            }
        }
        ListRepeat(a, b, _) | Bin(_, a, b, _) | Catch(a, _, b, _) => {
            walk_expr(a, f);
            walk_expr(b, f)
        }
        ListComp(a, _, b, c, _) => {
            walk_expr(a, f);
            walk_expr(b, f);
            if let Some(c) = c {
                walk_expr(c, f)
            }
        }
        Map(v, _) => {
            for (a, b) in v {
                walk_expr(a, f);
                walk_expr(b, f)
            }
        }
        ChanNew(_, Some(x), _) => walk_expr(x, f),
        Call(c, args, _, _) => {
            walk_expr(c, f);
            for a in args {
                walk_expr(&a.e, f)
            }
        }
        Index(x, v, _) => {
            walk_expr(x, f);
            for y in v {
                walk_expr(y, f)
            }
        }
        Slice(x, a, b, _) => {
            walk_expr(x, f);
            if let Some(a) = a {
                walk_expr(a, f)
            }
            if let Some(b) = b {
                walk_expr(b, f)
            }
        }
        Member(x, _, _)
        | TupleMember(x, _, _)
        | Neg(x, _)
        | BitNot(x, _)
        | Borrow(_, x, _)
        | Recv(x, _)
        | Comptime(x, _)
        | Cast(x, _, _)
        | Not(x, _)
        | Lambda(_, x, _) => walk_expr(x, f),
        Try(a, b, _) => {
            walk_expr(a, f);
            if let Some(b) = b {
                walk_expr(b, f)
            }
        }
        If(a, b, c, _) => {
            walk_expr(a, f);
            walk_expr(b, f);
            walk_expr(c, f)
        }
        _ => {}
    }
}
pub(crate) fn stmt_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) | Stmt::Spawn(e) | Stmt::Check(e) => vec![e],
        Stmt::Assign(_, a, b) | Stmt::Send(a, b) => vec![a, b],
        Stmt::Let(_, _, e, _)
        | Stmt::Var(_, _, e, _)
        | Stmt::For(_, e, _, _)
        | Stmt::Match(e, _, _) => vec![e],
        Stmt::If(arms, _) => arms.iter().map(|(c, _)| c).collect(),
        Stmt::While(c, b, _) => {
            let mut v = vec![c];
            if let Some(b) = b {
                v.push(b)
            }
            v
        }
        Stmt::Return(Some(e), _) => vec![e],
        _ => vec![],
    }
}
pub fn mentions_name(b: &[Stmt], n: &str) -> bool {
    let mut hit = false;
    walk_stmts(b, &mut |s| {
        for e in stmt_exprs(s) {
            walk_expr(e, &mut |x| {
                if matches!(x, Expr::Name(m, _) if m == n) {
                    hit = true
                }
            })
        }
    });
    hit
}
pub fn assigns_name(b: &[Stmt], n: &str) -> bool {
    let mut hit = false;
    walk_stmts(b, &mut |s| {
        if matches!(s, Stmt::Assign(op, Expr::Name(m, _), _) if op == "=" && m == n) {
            hit = true
        }
    });
    hit
}
fn pat_irrefutable(p: &Pat, c: &Checker) -> bool {
    match p {
        Pat::Wild => true,
        Pat::Path(v) => v.len() == 1 && !c.is_ctor_name(&v[0]) && v[0] != "some",
        Pat::Tuple(ps) => ps.iter().all(|q| pat_irrefutable(q, c)),
        _ => false,
    }
}
fn pat_irrefutable_payload(p: &Pat) -> bool {
    match p {
        Pat::Ctor(_, args) => args.iter().all(|(_, q)| matches!(q, Pat::Wild) || matches!(q, Pat::Path(v) if v.len() == 1 && v[0].chars().next().is_some_and(|c| c.is_lowercase()))),
        _ => true,
    }
}
fn pat_binds(p: &Pat, c: &Checker) -> bool {
    match p {
        Pat::Path(v) => v.len() == 1 && !c.is_ctor_name(&v[0]),
        Pat::Ctor(_, a) => a.iter().any(|(_, q)| pat_binds(q, c)),
        Pat::Tuple(v) | Pat::Alt(v) => v.iter().any(|q| pat_binds(q, c)),
        _ => false,
    }
}

impl Checker {
    fn use_name(&mut self, n: &str, consume: bool, line: usize) -> Option<T> {
        let l = self.lookup(n)?;
        if l.moved {
            self.e(d(
                line,
                "W-OWN-1",
                format!("`{n}` is used after it was moved"),
            ));
        }
        if consume && !self.is_copy(&l.ty) {
            if l.field && matches!(self.cx, Cx::Step | Cx::Handler) { /* fields are moved out of the pre-state copy */
            } else if l.param && matches!(l.ty, T::Ref(..)) {
            } else {
                self.set_moved(n, true);
            }
        }
        Some(l.ty)
    }
    fn effect_call(&mut self, callee: &str, cx: Cx, line: usize) {
        let me = self.cur_fn.clone();
        if !me.is_empty() {
            self.calls.entry(me).or_default().insert(callee.to_string());
        }
        if self.externs.contains_key(callee) && self.device > 0 {
            self.e(d(
                line,
                "W-DEV-1",
                format!("`{callee}` is a foreign function and cannot be called in a device region"),
            ));
        }
        if self.externs.contains_key(callee)
            && matches!(self.cx, Cx::Pure | Cx::Comptime | Cx::Step | Cx::Handler)
        {
            self.e(d(
                line,
                "W-EFF-1",
                format!(
                    "`{callee}` is a foreign function: only main and task functions may call it"
                ),
            ));
        } else if cx == Cx::Task
            && matches!(self.cx, Cx::Pure | Cx::Comptime | Cx::Step | Cx::Handler)
        {
            self.e(d(
                line,
                "W-EFF-1",
                format!("`{callee}` is a task function and cannot be called here"),
            ));
        }
        if cx == Cx::Main {
            self.e(d(line, "W-EFF-1", "`main` cannot be called"));
        }
    }

    pub fn infer(&mut self, e: &Expr, consume: bool) -> T {
        let ok = std::mem::take(&mut self.iter_ok);
        let t = self.infer_inner(e, consume);
        if !ok
            && matches!(e, Expr::Call(..) | Expr::Name(..))
            && let T::Iter(_) = self.sub.resolve(&t)
        {
            self.e(d(
                e.line(),
                "W-ITER-1",
                "an iterator may only be the operand of `for` or the receiver of `.collect()`",
            ));
        }
        if let Expr::Bin(_, _, _, l) | Expr::Neg(_, l) | Expr::Cast(_, _, l) = e {
            self.bin_types
                .insert(*l * 1000 + (self.bin_types.len() % 1000), t.clone());
        }
        t
    }

    fn infer_inner(&mut self, e: &Expr, consume: bool) -> T {
        use Expr as E;
        match e {
            E::Name(n, l) => {
                if let Some(t) = self.use_name(n, consume, *l) {
                    return t;
                }
                if let Some((ty, ce)) = self.consts.get(n).cloned() {
                    return match ty {
                        Some(ty) => self.lower(&ty, *l),
                        None => {
                            let cx = self.cx;
                            let t = self.infer(&ce, false);
                            self.cx = cx;
                            t
                        }
                    };
                }
                if let Some(f) = self.fns.get(n).cloned() {
                    // 5.11: a foreign function is not a value. Only a direct
                    // call is checked against the context (W-EFF-1), so a
                    // value would let a pure function call foreign code.
                    if self.externs.contains_key(n) {
                        self.e(d(*l, "W-FFI-1", format!("`{n}` is a foreign function: it can only be called directly, not used as a value")));
                    }
                    self.effect_call(n, f.cx, *l);
                    let ps: Vec<T> = f
                        .decl
                        .sig
                        .params
                        .iter()
                        .filter_map(|p| {
                            if let Param::Named { ty, .. } = p {
                                Some(ty.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .iter()
                        .map(|t| self.lower(t, *l))
                        .collect();
                    let r = f
                        .decl
                        .sig
                        .ret
                        .as_ref()
                        .map(|r| self.lower(r, *l))
                        .unwrap_or(T::Unit);
                    return T::Fn(ps, Box::new(r));
                }
                if self.is_ctor_name(n) {
                    if self.payload_of(n).is_some_and(|p| !p.is_empty()) {
                        self.e(d(*l, "W-TY-1", format!("`{n}` needs its fields")));
                    }
                    let h = T::Never;
                    return self.ctor_type(n, *l, &h);
                }
                if let Some(TyDef::Actor { .. }) = self.types.get(n) {
                    return T::Fn(vec![], Box::new(T::Named(n.clone(), vec![])));
                }
                self.e(d(*l, "W-NAME-1", format!("unknown name `{n}`")));
                T::Never
            }
            E::SelfRef(l) => self.use_name("self", consume, *l).unwrap_or_else(|| {
                self.e(d(*l, "W-NAME-1", "`self` outside a method"));
                T::Never
            }),
            E::Int(v, l) => {
                let t = self.sub.fresh_int();
                self.lits.push((t.clone(), *v, *l));
                t
            }
            E::Float(..) => T::F64,
            E::Str(..) => T::Text,
            E::Bool(..) => T::Bool,
            E::None(_) => {
                let v = self.sub.fresh();
                T::Opt(Box::new(v))
            }
            E::Unit(_) => T::Unit,
            E::Tuple(v, _) => T::Tuple(v.iter().map(|x| self.infer(x, consume)).collect()),
            E::List(v, l) => {
                let el = self.sub.fresh();
                for x in v {
                    let t = self.infer(x, true);
                    if contains_ref(&self.sub.resolve(&t)) {
                        self.e(d(*l, "W-OWN-3", "a list cannot hold borrows"));
                    }
                    self.expect(&t, &el, *l);
                }
                self.sub.fresh_list(el, v.len() as u64)
            }
            E::ListRepeat(x, n, l) => {
                let t = self.infer(x, true);
                if !self.is_copy(&t) {
                    self.e(d(*l, "W-TY-1", "repeated element must be Copy"));
                }
                let nt = self.infer(n, false);
                self.expect(&nt, &T::Int("usize"), *l);
                T::Array(Box::new(t), self.const_len(n))
            }
            E::ListComp(body, p, it, cond, l) => {
                let itt = self.infer(it, false);
                let elem = self.elem_type(&itt, *l, is_place(it));
                self.scopes.push(HashMap::new());
                self.bind_pat(p, &elem, *l, false);
                if let Some(c) = cond {
                    let ct = self.infer(c, false);
                    self.expect(&peel(&ct), &T::Bool, *l);
                }
                let bt = self.infer(body, true);
                self.scopes.pop();
                if contains_ref(&self.sub.resolve(&bt)) {
                    self.e(d(
                        *l,
                        "W-OWN-3",
                        "a list cannot hold borrows (copy the element)",
                    ));
                }
                T::List(Box::new(bt))
            }
            E::Map(v, l) => {
                let (k, val) = (self.sub.fresh(), self.sub.fresh());
                for (a, b) in v {
                    let at = self.infer(a, true);
                    self.expect(&at, &k, *l);
                    let bt = self.infer(b, true);
                    self.expect(&bt, &val, *l);
                }
                T::Map(Box::new(k), Box::new(val))
            }
            E::ChanNew(ty, cap, l) => {
                let t = self.lower(ty, *l);
                if contains_ref(&t) {
                    self.e(d(*l, "W-OWN-3", "a channel cannot carry borrows"));
                }
                if let Some(c) = cap {
                    let ct = self.infer(c, false);
                    self.expect(&ct, &T::Int("usize"), *l);
                }
                if let T::Named(s, a) = &t
                    && s == "session"
                {
                    return self.session_pair(a);
                }
                T::Chan(Box::new(t))
            }
            E::Call(f, args, l, _) => self.call(f, args, *l),
            E::Index(x, idx, l) => {
                let xt = self.infer(x, false);
                let base = peel(&self.sub.resolve(&xt));
                if idx.len() != 1 {
                    self.e(d(*l, "W-TY-1", "one index expected"));
                }
                let it = self.infer(&idx[0], false);
                let out = match base {
                    T::List(e) | T::Array(e, _) | T::Slice(e) => {
                        self.expect(&peel(&it), &T::Int("usize"), *l);
                        *e
                    }
                    T::Map(k, v) => {
                        self.expect(&peel(&it), &k, *l);
                        *v
                    }
                    T::Text => {
                        self.expect(&peel(&it), &T::Int("usize"), *l);
                        T::Int("u8")
                    }
                    T::Var(_) => self.sub.fresh(),
                    other => {
                        self.e(d(*l, "W-TY-1", format!("cannot index {other}")));
                        T::Never
                    }
                };
                if consume && !self.is_copy(&out) {
                    self.e(d(
                        *l,
                        "W-OWN-1",
                        "cannot move out of an indexed element (use copy)",
                    ));
                }
                out
            }
            E::Slice(x, a, b, l) => {
                let xt = self.infer(x, false);
                for y in [a, b].into_iter().flatten() {
                    let yt = self.infer(y, false);
                    if !self.coerce(&yt, &T::Int("i64")) && !self.coerce(&yt, &T::Int("usize")) {
                        self.e(d(*l, "W-TY-1", "slice bounds are integers"));
                    }
                }
                match peel(&self.sub.resolve(&xt)) {
                    T::List(e) | T::Array(e, _) | T::Slice(e) => {
                        T::Ref(false, Box::new(T::Slice(e)))
                    }
                    T::Text => T::Text,
                    other => {
                        self.e(d(*l, "W-TY-1", format!("cannot slice {other}")));
                        T::Never
                    }
                }
            }
            E::Member(x, m, l) => {
                let xt = self.infer(x, false);
                let out = self.field_type(&xt, m, *l);
                if consume
                    && !self.is_copy(&out)
                    && is_place(x)
                    && let Some(r) = place_root(x)
                    && let Some(lc) = self.lookup(&r)
                    && !matches!(lc.ty, T::Ref(..))
                    && !(lc.field)
                {
                    self.set_moved(&r, true);
                }
                out
            }
            E::TupleMember(x, i, l) => {
                let xt = self.infer(x, false);
                match peel(&self.sub.resolve(&xt)) {
                    T::Tuple(v) if *i < v.len() => v[*i].clone(),
                    other => {
                        self.e(d(*l, "W-TY-1", format!("{other} has no element .{i}")));
                        T::Never
                    }
                }
            }
            E::Neg(x, l) => {
                let t = self.infer(x, false);
                if let E::Int(v, _) = &**x
                    && let Some(last) = self.lits.last_mut()
                {
                    last.1 = -*v;
                }
                match self.sub.resolve(&peel(&t)) {
                    T::Int(n) if n.starts_with('u') => {
                        self.e(d(*l, "W-TY-1", "negation of an unsigned integer"));
                        t
                    }
                    T::Int(_) | T::F64 | T::Var(_) => peel(&t),
                    o => {
                        self.e(d(*l, "W-TY-1", format!("cannot negate {o}")));
                        T::Never
                    }
                }
            }
            E::BitNot(x, l) => {
                let t = self.infer(x, false);
                match self.sub.resolve(&peel(&t)) {
                    T::Int(_) | T::Var(_) => peel(&t),
                    o => {
                        self.e(d(*l, "W-TY-1", format!("`~` needs an integer, found {o}")));
                        T::Never
                    }
                }
            }
            E::Not(x, l) => {
                let t = self.infer(x, false);
                self.expect(&peel(&t), &T::Bool, *l);
                T::Bool
            }
            E::Borrow(m, x, l) => {
                if *m {
                    match place_root(x).and_then(|r| self.lookup(&r)) {
                        Some(lc) if lc.mutable || matches!(lc.ty, T::Ref(true, _)) => {}
                        _ => self.e(d(
                            *l,
                            "W-OWN-2",
                            "`&mut` needs a mutable place (declare it with var)",
                        )),
                    }
                }
                if !is_place(x) && !matches!(**x, E::Slice(..)) {
                    self.e(d(*l, "W-PLACE", "only places can be borrowed"));
                }
                let t = self.infer(x, false);
                match self.sub.resolve(&t) {
                    T::Ref(_, inner) => T::Ref(*m, inner),
                    t => T::Ref(*m, Box::new(t)),
                }
            }
            E::Recv(x, l) => {
                if !matches!(self.cx, Cx::Main | Cx::Task) {
                    self.e(d(*l, "W-EFF-1", "receive outside main or a task function"));
                }
                let t = self.infer(x, false);
                match self.sub.resolve(&peel(&t)) {
                    T::Chan(e) => T::Opt(e),
                    T::Named(s, _) if s == "session" => self.session_recv_ty(x, *l),
                    o => {
                        self.e(d(*l, "W-TY-1", format!("`<-` needs a channel, found {o}")));
                        T::Never
                    }
                }
            }
            E::Comptime(x, _) => {
                let cx = self.cx;
                self.cx = Cx::Comptime;
                let t = self.infer(x, consume);
                self.cx = cx;
                t
            }
            E::Cast(x, ty, l) => {
                let t = self.infer(x, false);
                let to = self.lower(ty, *l);
                let from = self.sub.resolve(&peel(&t));
                let num = |t: &T| matches!(t, T::Int(_) | T::F64 | T::F32 | T::Var(_));
                if !(num(&from) && num(&to)) && !(from == T::Bool && matches!(to, T::Int(_))) {
                    self.e(d(*l, "W-TY-1", format!("cannot cast {from} to {to}")));
                }
                to
            }
            E::Bin(op, a, b, l) => self.binary(op, a, b, *l),
            E::Catch(x, bind, h, l) => {
                let t = self.infer(x, consume);
                match self.sub.resolve(&t) {
                    T::Union(ok, err) => {
                        self.scopes.push(HashMap::new());
                        if let Some(n) = bind {
                            self.declare(n, *err, false, *l, false);
                        }
                        let ht = self.infer(h, true);
                        self.scopes.pop();
                        self.expect(&ht, &ok, *l);
                        *ok
                    }
                    o => {
                        self.e(d(
                            *l,
                            "W-TY-1",
                            format!("catch needs an error union, found {o}"),
                        ));
                        T::Never
                    }
                }
            }
            E::Try(x, els, l) => {
                let t = self.infer(x, consume);
                let ret = self.sub.resolve(&self.ret.clone());
                let in_step = matches!(self.cx, Cx::Step | Cx::Handler | Cx::Main | Cx::Comptime);
                let (ok, err) = match (self.sub.resolve(&t), els) {
                    (T::Union(ok, err), None) => (*ok, *err),
                    (T::Opt(ok), Some(ex)) | (T::Union(ok, _), Some(ex)) => {
                        let et = self.infer(ex, true);
                        (*ok, self.sub.resolve(&et))
                    }
                    (T::Opt(_), None) => {
                        self.e(d(*l, "W-TY-1", "try on an option needs `else <error>`"));
                        return T::Never;
                    }
                    (o, _) => {
                        self.e(d(
                            *l,
                            "W-TY-1",
                            format!("try needs an error union or option, found {o}"),
                        ));
                        return T::Never;
                    }
                };
                if !in_step {
                    match ret {
                        T::Union(_, re) => {
                            if !self.sub.unify(&err, &re) {
                                self.e(d(
                                    *l,
                                    "W-TY-1",
                                    format!("error {err} is not in this function's error set {re}"),
                                ));
                            }
                        }
                        _ => self.e(d(
                            *l,
                            "W-TY-1",
                            "try in a function that does not return an error union",
                        )),
                    }
                }
                ok
            }
            E::If(c, a, b, l) => {
                let ct = self.infer(c, false);
                self.expect(&peel(&ct), &T::Bool, *l);
                let before = self.moved_set();
                let at = self.infer(a, consume);
                let m1 = self.moved_set();
                self.restore_moved(&before);
                let bt = self.infer(b, consume);
                let mut m2 = self.moved_set();
                m2.extend(m1);
                self.restore_moved(&m2);
                self.expect(&bt, &at, *l);
                at
            }
            E::Lambda(ps, body, l) => {
                self.scopes.push(HashMap::new());
                let mut ts = vec![];
                for p in ps {
                    if let Param::Named { name, ty, .. } = p {
                        let t = self.lower(ty, *l);
                        ts.push(t.clone());
                        self.declare(name, t, false, *l, true);
                    }
                }
                let r = self.infer(body, true);
                self.scopes.pop();
                T::Fn(ts, Box::new(r))
            }
        }
    }
}

impl Checker {
    /// Whether a value of type t can hold a secret (5.16, W-SEC-1).
    fn holds_secret(&self, t: &T) -> bool {
        match t {
            T::Named(n, a) => n == "Secret" || self.secret_structs.contains(n) || a.iter().any(|x| self.holds_secret(x)),
            T::Tuple(v) => v.iter().any(|x| self.holds_secret(x)),
            T::List(x) | T::Set(x) | T::Opt(x) | T::Array(x, _) | T::Slice(x) | T::Ref(_, x) | T::Iter(x) | T::Chan(x) | T::Region(x, _) => self.holds_secret(x),
            T::Map(a, b) | T::Union(a, b) => self.holds_secret(a) || self.holds_secret(b),
            _ => false,
        }
    }
    fn has_trait(&self, t: &T, tr: &str) -> bool {
        match self.sub.resolve(&peel(t)) {
            T::Param(p) => self
                .gbounds
                .get(&p)
                .is_some_and(|b| b.iter().any(|x| x == tr)),
            T::Named(n, _) => {
                self.impls
                    .iter()
                    .any(|(h, t)| *h == n && t.as_deref() == Some(tr))
                    || matches!(tr, "Eq" | "Show" | "Data")
                        && !matches!(self.types.get(&n), Some(TyDef::Actor { .. }))
            }
            T::F64 => tr != "Hash",
            T::Fn(..) | T::Chan(_) => false,
            _ => true,
        }
    }
    /// W-TY-3 (9): integer/float mixing without `as`, or an implicit
    /// narrowing of an integer (`found` does not fit in `expected`); any
    /// other mismatch is W-TY-1.
    fn mix_code(&self, found: &T, expected: &T, narrowing: bool) -> &'static str {
        let cls = |t: &T| match self.sub.resolve(&peel(t)) {
            T::Int(n) => Some(Some(n)),
            T::Var(i) if self.sub.intlit[i] => Some(None),
            T::F64 | T::F32 => Some(Some("float")),
            _ => None,
        };
        match (cls(found), cls(expected)) {
            (Some(Some("float")), Some(Some(x))) | (Some(Some(x)), Some(Some("float"))) if x != "float" => "W-TY-3",
            (Some(None), Some(Some("float"))) | (Some(Some("float")), Some(None)) => "W-TY-3",
            (Some(Some(f)), Some(Some(x))) if narrowing && f != x && f != "float" && x != "float" => {
                let ((fl, fh), (xl, xh)) = (crate::types::int_range(f), crate::types::int_range(x));
                if fl < xl || fh > xh { "W-TY-3" } else { "W-TY-1" }
            }
            _ => "W-TY-1",
        }
    }

    fn num(&self, t: &T) -> bool {
        matches!(self.sub.resolve(t), T::Int(_) | T::F64 | T::F32 | T::Var(_))
    }

    pub fn arith_op(&mut self, op: &str, a: &T, b: &T, line: usize, _rhs: &Expr) -> T {
        let (sa, sb) = (self.sub.shallow(&peel(a)), self.sub.shallow(&peel(b)));
        match (&sa, &sb) {
            (T::Array(..), T::Var(_)) => {
                self.sub.unify(&sb, &sa);
            }
            (T::Var(_), T::Array(..)) => {
                self.sub.unify(&sa, &sb);
            }
            _ => {}
        }
        let (a, b) = (peel(&self.sub.resolve(a)), peel(&self.sub.resolve(b)));
        match op {
            "+" if a == T::Text || b == T::Text => {
                self.expect(&b, &a, line);
                T::Text
            }
            "+" | "-" | "*" | "/" | "%"
                if matches!(self.sub.resolve(&a), T::Array(..))
                    || matches!(self.sub.resolve(&b), T::Array(..)) =>
            {
                match (self.sub.resolve(&a), self.sub.resolve(&b)) {
                    (T::Array(x, n), T::Array(y, m)) => {
                        if n.is_none() || n != m {
                            self.e(d(line, "W-SIMD-1", format!("lane-wise `{op}` needs two arrays of one fixed length, found {a} and {b}")));
                            return T::Never;
                        }
                        let xs = self.sub.resolve(&x);
                        if !matches!(xs, T::Int(_) | T::F64 | T::F32 | T::Var(_)) {
                            self.e(d(
                                line,
                                "W-SIMD-1",
                                format!("lane-wise `{op}` needs numeric lanes, found {xs}"),
                            ));
                            return T::Never;
                        }
                        let el = self.arith_op(op, &x, &y, line, _rhs);
                        T::Array(Box::new(el), n)
                    }
                    _ => {
                        self.e(d(
                            line,
                            "W-SIMD-1",
                            format!("lane-wise `{op}` needs two arrays, found {a} and {b}"),
                        ));
                        T::Never
                    }
                }
            }
            "+" | "-" | "*" | "/" | "%" => {
                if !self.num(&a) || !self.num(&b) {
                    self.e(d(
                        line,
                        "W-TY-1",
                        format!("`{op}` needs numbers, found {a} and {b}"),
                    ));
                    return T::Never;
                }
                let code = self.mix_code(&a, &b, false);
                if !self.sub.unify(&a, &b) {
                    self.e(d(
                        line,
                        if code == "W-TY-3" { "W-TY-3" } else { "W-TY-1" },
                        format!("`{op}` on {a} and {b}: no implicit conversion"),
                    ));
                }
                if op == "%" && matches!(self.sub.resolve(&a), T::F64 | T::F32) {
                    self.e(d(line, "W-TY-1", "`%` needs integers"));
                }
                a
            }
            "&" | "|" | "^" | "<<" | ">>" => {
                let ok = |t: &T| matches!(t, T::Int(_) | T::Var(_));
                if !ok(&self.sub.resolve(&a)) || !ok(&self.sub.resolve(&b)) {
                    self.e(d(line, "W-TY-1", format!("`{op}` needs integers")));
                    return T::Never;
                }
                if op == "<<" || op == ">>" {
                    if let T::Var(_) = self.sub.resolve(&b) {
                        self.sub.unify(&b, &T::Int("u32"));
                    }
                } else if !self.sub.unify(&a, &b) {
                    self.e(d(line, "W-TY-1", format!("`{op}` on {a} and {b}")));
                }
                a
            }
            _ => {
                self.e(d(line, "W-TY-1", format!("unknown operator `{op}`")));
                T::Never
            }
        }
    }

    fn binary(&mut self, op: &str, a: &Expr, b: &Expr, l: usize) -> T {
        match op {
            "and" | "or" => {
                let at = self.infer(a, false);
                self.expect(&peel(&at), &T::Bool, l);
                let bt = self.infer(b, false);
                self.expect(&peel(&bt), &T::Bool, l);
                T::Bool
            }
            "==" | "!=" | "<" | "<=" | ">" | ">=" => {
                let at = self.infer(a, false);
                let bt = self.infer(b, false);
                let (pa, pb) = (peel(&self.sub.shallow(&at)), peel(&self.sub.shallow(&bt)));
                let (pa, pb) = (peel(&self.sub.shallow(&pa)), peel(&self.sub.shallow(&pb)));
                let code = self.mix_code(&pa, &pb, false);
                if !self.sub.unify(&pa, &pb) {
                    let (pa, pb) = (self.sub.resolve(&pa), self.sub.resolve(&pb));
                    self.e(d(l, code, format!("cannot compare {pa} with {pb}")));
                }
                let pa = self.sub.resolve(&pa);
                let tr = if op == "==" || op == "!=" {
                    "Eq"
                } else {
                    "Ord"
                };
                if !self.has_trait(&pa, tr) {
                    self.e(d(l, "W-TRAIT-3", format!("{pa} does not implement {tr}")));
                }
                T::Bool
            }
            "in" | "not in" => {
                let at = self.infer(a, false);
                let bt = self.infer(b, false);
                let pa = peel(&self.sub.resolve(&at));
                match peel(&self.sub.resolve(&bt)) {
                    T::List(e) | T::Array(e, _) | T::Slice(e) | T::Set(e) | T::Map(e, _) => {
                        if !self.sub.unify(&pa, &e) {
                            self.e(d(l, "W-TY-1", format!("{pa} is not the element type {e}")));
                        }
                    }
                    T::Text => self.expect(&pa, &T::Text, l),
                    T::Named(n, args) if n == "Range" => {
                        if !self.sub.unify(&pa, &args[0]) {
                            self.e(d(l, "W-TY-1", "range element type mismatch"));
                        }
                    }
                    o => self.e(d(
                        l,
                        "W-TY-1",
                        format!("`in` needs a collection, found {o}"),
                    )),
                }
                T::Bool
            }
            ".." | "..=" => {
                let at = self.infer(a, false);
                let bt = self.infer(b, false);
                let (pa, pb) = (peel(&at), peel(&bt));
                if !matches!(self.sub.resolve(&pa), T::Int(_) | T::Var(_))
                    || !self.sub.unify(&pa, &pb)
                {
                    self.e(d(l, "W-TY-1", "a range needs two integers of one type"));
                }
                T::Named("Range".into(), vec![pa])
            }
            _ => {
                let at = self.infer(a, false);
                let bt = self.infer(b, false);
                if self.v08 {
                    let (ra, rb) = (self.sub.resolve(&peel(&at)), self.sub.resolve(&peel(&bt)));
                    let ctx = |t: &T| matches!(t, T::Named(n, _) if n == "Text");
                    if ctx(&ra) || ctx(&rb) {
                        if op == "+" && ra == rb {
                            return ra;
                        }
                        self.e(d(l, "W-CTX-1", format!("`{op}` on {ra} and {rb}: context-typed text joins only with text of the same context (5.16)")));
                        return T::Never;
                    }
                    if matches!(&ra, T::Named(n, _) if n == "Secret" || n == "Cap") || matches!(&rb, T::Named(n, _) if n == "Secret" || n == "Cap") {
                        self.e(d(l, "W-SEC-1", format!("`{op}` on {ra} and {rb}: a secret or capability is opaque (5.16)")));
                        return T::Never;
                    }
                }
                self.arith_op(op, &at, &bt, l, b)
            }
        }
    }

    fn field_type(&mut self, t: &T, m: &str, l: usize) -> T {
        let rt = self.sub.resolve(t);
        let by_ref = matches!(rt, T::Ref(..));
        let mut_ref = matches!(rt, T::Ref(true, _));
        let base = peel(&rt);
        let found: Option<Ty> = match &base {
            T::Named(n, _) => match self.types.get(n) {
                Some(TyDef::Struct { fields, .. })
                | Some(TyDef::State { fields })
                | Some(TyDef::Actor { fields }) => {
                    fields.iter().find(|f| f.name == m).map(|f| f.ty.clone())
                }
                Some(TyDef::Event { payload }) | Some(TyDef::Error { payload }) => {
                    payload.iter().find(|f| f.0 == m).map(|f| f.1.clone())
                }
                Some(TyDef::Kind { variants, .. }) => {
                    let mut r = None;
                    if variants.len() == 1 {
                        r = variants[0].1.iter().find(|f| f.0 == m).map(|f| f.1.clone());
                    }
                    r
                }
                _ => None,
            },
            _ => None,
        };
        match found {
            Some(ty) => {
                let ft = self.lower(&ty, l);
                if by_ref && !self.is_copy(&ft) {
                    T::Ref(mut_ref, Box::new(ft))
                } else {
                    ft
                }
            }
            None => {
                self.e(d(l, "W-NAME-1", format!("{base} has no field `{m}`")));
                T::Never
            }
        }
    }

    pub fn ctor_args(&mut self, name: &str, fields: &[(String, Ty, bool)], args: &[Arg], l: usize) {
        let mut used = HashSet::new();
        for (i, a) in args.iter().enumerate() {
            let idx = match &a.label {
                Some(lb) => fields.iter().position(|f| &f.0 == lb),
                None => {
                    if fields.len() == 1 || i < fields.len() {
                        Some(i)
                    } else {
                        None
                    }
                }
            };
            match idx {
                Some(k) => {
                    if !used.insert(k) {
                        self.e(d(
                            l,
                            "W-TY-1",
                            format!("field `{}` given twice", fields[k].0),
                        ));
                    }
                    if a.label.is_none() && fields.len() > 1 {
                        self.e(d(
                            l,
                            "W-TY-1",
                            format!("`{name}` has several fields: label the arguments"),
                        ));
                    }
                    let at = self.infer(&a.e, true);
                    let ft = self.lower(&fields[k].1, l);
                    if contains_ref(&self.sub.resolve(&at)) {
                        self.e(d(l, "W-OWN-3", "a constructed value cannot hold borrows"));
                    }
                    self.expect(&at, &ft, l);
                }
                None => self.e(d(
                    l,
                    "W-NAME-1",
                    format!(
                        "`{name}` has no field `{}`",
                        a.label.clone().unwrap_or_else(|| i.to_string())
                    ),
                )),
            }
        }
        for (k, f) in fields.iter().enumerate() {
            if !used.contains(&k) && !f.2 {
                self.e(d(
                    l,
                    "W-TY-1",
                    format!("`{name}` is missing field `{}`", f.0),
                ));
            }
        }
    }

    fn call(&mut self, f: &Expr, args: &[Arg], l: usize) -> T {
        // R5: `f[T](x)` is an index of a generic function by type arguments,
        // which name resolution turns into instantiation
        if let Expr::Index(g, ix, _) = f
            && let Expr::Name(n, _) = &**g
            && self.fns.get(n).is_some_and(|fi| !fi.decl.sig.generics.is_empty())
        {
            match ix.iter().map(expr_as_ty).collect::<Option<Vec<Ty>>>() {
                Some(ts) => {
                    self.targs = Some(ts);
                    return self.call(g, args, l);
                }
                None => {
                    self.e(d(l, "W-TY-1", format!("`{n}[…]`: type arguments expected")));
                    return T::Never;
                }
            }
        }
        // aliasing of borrows among the arguments (W-OWN-2)
        let mut roots: Vec<(String, bool)> = vec![];
        for a in args {
            if let Expr::Borrow(m, x, _) = &a.e
                && let Some(r) = place_root(x)
            {
                roots.push((r, *m));
            }
        }
        for (i, (r, m)) in roots.iter().enumerate() {
            for (s, n) in &roots[i + 1..] {
                if r == s && (*m || *n) {
                    self.e(d(
                        l,
                        "W-OWN-2",
                        format!("`{r}` is borrowed mutably and again in one call"),
                    ));
                }
            }
        }
        if self.v08
            && let Expr::Member(o, m, _) = f
            && let Expr::Name(c, _) = &**o
            && crate::index08::is_ctx(c, self.v09)
            && self.lookup(c).is_none()
        {
            let out = T::Named("Text".into(), vec![T::Named(c.clone(), vec![])]);
            if c == "Shell" {
                // F.5.2: `Shell.cmd(program, args)`, a literal program and `Text[ShellArg]` arguments
                if m == "cmd" && args.len() == 2 && matches!(args[0].e, Expr::Str(..)) {
                    let at = self.infer(&args[1].e, false);
                    let want = T::List(Box::new(T::Named("Text".into(), vec![T::Named("ShellArg".into(), vec![])])));
                    self.expect(&peel(&at), &want, l);
                } else {
                    self.e(d(l, "W-CTX-1", "`Shell.cmd(program, args)` takes a string literal program and a `List[Text[ShellArg]]`; there is no other way to make a shell command (F.5.2)"));
                }
                return out;
            }
            match (m.as_str(), args) {
                ("lit", [a]) if matches!(a.e, Expr::Str(..)) => return out,
                ("lit", _) => {
                    self.e(d(l, "W-CTX-1", format!("`{c}.lit` takes one string literal: only program text is trusted (5.16)")));
                    return out;
                }
                ("encode", [a]) => {
                    let t = self.infer(&a.e, false);
                    self.expect(&peel(&t), &T::Text, l);
                    return out;
                }
                _ => {
                    self.e(d(l, "W-NAME-1", format!("`{c}.{m}`: a context has `lit` and `encode` (5.16)")));
                    return T::Never;
                }
            }
        }
        match f {
            Expr::Name(n, _) => {
                if self.lookup(n).is_none() {
                    match n.as_str() {
                        "lvar_new" => return self.lvar_new_ty(args, l),
                        // F.5.4: `ct_choose(c, a, b)` evaluates both `a` and `b`
                        "ct_choose" if self.v09 => {
                            if args.len() != 3 {
                                self.e(d(l, "W-TY-1", "`ct_choose` takes a condition and two values"));
                                return T::Never;
                            }
                            let c = self.infer(&args[0].e, false);
                            self.expect(&peel(&c), &T::Bool, l);
                            let a = self.infer(&args[1].e, true);
                            let b = self.infer(&args[2].e, true);
                            self.expect(&b, &a, l);
                            return a;
                        }
                        "set_of" => {
                            if args.len() != 1 {
                                self.e(d(l, "W-TY-1", "`set_of` takes one list"));
                                return T::Never;
                            }
                            let t = self.infer(&args[0].e, true);
                            let e = self.sub.fresh();
                            self.expect(&t, &T::List(Box::new(e.clone())), l);
                            return T::Set(Box::new(e));
                        }
                        "some" => {
                            if args.len() != 1 {
                                self.e(d(l, "W-TY-1", "`some` takes one value"));
                                return T::Never;
                            }
                            let t = self.infer(&args[0].e, true);
                            return T::Opt(Box::new(t));
                        }
                        "copy" => {
                            if args.len() != 1 {
                                self.e(d(l, "W-TY-1", "`copy` takes one value"));
                                return T::Never;
                            }
                            let t = self.infer(&args[0].e, false);
                            if self.mentions_linear(&self.sub.resolve(&t)) {
                                self.e(d(l, "W-LIN-1", "a linear value cannot be copied"));
                            }
                            return match peel(&self.sub.resolve(&t)) {
                                T::Slice(e) => T::List(e),
                                o => o,
                            };
                        }
                        "Box" => {
                            let t = self.infer(&args[0].e, true);
                            return T::Named("Box".into(), vec![t]);
                        }
                        _ => {}
                    }
                    if let Some(fi) = self.fns.get(n).cloned() {
                        self.effect_call(n, fi.cx, l);
                        return self.call_sig(&fi.decl.sig, args, l, None);
                    }
                    if let Some((k, p)) = self.variants.get(n).cloned() {
                        let fs: Vec<(String, Ty, bool)> = p
                            .iter()
                            .map(|(a, b)| (a.clone(), b.clone(), false))
                            .collect();
                        self.ctor_args(n, &fs, args, l);
                        let ngen = match self.types.get(&k) {
                            Some(TyDef::Kind { generics, .. }) => generics.len(),
                            _ => 0,
                        };
                        let ga = (0..ngen).map(|_| self.sub.fresh()).collect();
                        return T::Named(k, ga);
                    }
                    match self.types.get(n).cloned() {
                        Some(TyDef::Struct { fields, generics }) => {
                            let fs: Vec<(String, Ty, bool)> = fields
                                .iter()
                                .map(|f| (f.name.clone(), f.ty.clone(), f.init.is_some()))
                                .collect();
                            let saved = self.gbounds.clone();
                            for g in &generics {
                                self.gbounds.insert(g.name.clone(), vec![]);
                            }
                            self.ctor_args(n, &fs, args, l);
                            self.gbounds = saved;
                            let ga = generics.iter().map(|_| self.sub.fresh()).collect();
                            return T::Named(n.clone(), ga);
                        }
                        Some(TyDef::Event { payload }) | Some(TyDef::Error { payload }) => {
                            let fs: Vec<(String, Ty, bool)> = payload
                                .iter()
                                .map(|(a, b)| (a.clone(), b.clone(), false))
                                .collect();
                            self.ctor_args(n, &fs, args, l);
                            return T::Named(n.clone(), vec![]);
                        }
                        Some(TyDef::Actor { .. }) => {
                            if !matches!(self.cx, Cx::Main | Cx::Task) {
                                self.e(d(
                                    l,
                                    "W-EFF-1",
                                    "actors are started from main or a task function",
                                ));
                            }
                            if !args.is_empty() {
                                self.e(d(
                                    l,
                                    "W-TY-1",
                                    "an actor starts from its declared initial state",
                                ));
                            }
                            return T::Named(n.clone(), vec![]);
                        }
                        Some(TyDef::State { .. }) => {
                            self.e(d(
                                l,
                                "W-STATE-1",
                                format!("state `{n}` cannot be constructed"),
                            ));
                            return T::Never;
                        }
                        _ => {}
                    }
                }
                let ft = self.infer(f, false);
                self.call_value(&ft, args, l)
            }
            Expr::Member(obj, m, ml) => {
                // Type.method(...) static call
                if let Expr::Name(tn, _) = &**obj
                    && self.lookup(tn).is_none()
                    && self.types.contains_key(tn)
                {
                    let found = self
                        .methods
                        .iter()
                        .find(|(h, fi)| h == tn && fi.decl.sig.name == *m)
                        .map(|x| x.1.clone());
                    return match found {
                        Some(fi) => {
                            self.effect_call(&format!("{tn}.{m}"), fi.cx, l);
                            let st = T::Named(tn.clone(), vec![]);
                            self.call_sig(&fi.decl.sig, args, l, Some(st))
                        }
                        None => {
                            self.e(d(*ml, "W-NAME-1", format!("`{tn}` has no function `{m}`")));
                            T::Never
                        }
                    };
                }
                self.method_call(obj, m, args, l)
            }
            _ => {
                let ft = self.infer(f, false);
                self.call_value(&ft, args, l)
            }
        }
    }

    fn call_value(&mut self, ft: &T, args: &[Arg], l: usize) -> T {
        match self.sub.resolve(&peel(ft)) {
            T::Fn(ps, r) => {
                if ps.len() != args.len() {
                    self.e(d(
                        l,
                        "W-TY-1",
                        format!("expected {} arguments, found {}", ps.len(), args.len()),
                    ));
                }
                for (p, a) in ps.iter().zip(args) {
                    let at = self.infer(&a.e, true);
                    self.expect(&at, p, l);
                }
                *r
            }
            T::Never => T::Never,
            o => {
                self.e(d(l, "W-TY-1", format!("{o} is not callable")));
                T::Never
            }
        }
    }

    fn inst(&mut self, t: &T, m: &HashMap<String, T>) -> T {
        match t {
            T::Param(p) => m.get(p).cloned().unwrap_or_else(|| t.clone()),
            T::Tuple(v) => T::Tuple(v.iter().map(|x| self.inst(x, m)).collect()),
            T::Named(n, v) => T::Named(n.clone(), v.iter().map(|x| self.inst(x, m)).collect()),
            T::Array(a, n) => T::Array(Box::new(self.inst(a, m)), *n),
            T::List(a) => T::List(Box::new(self.inst(a, m))),
            T::Set(a) => T::Set(Box::new(self.inst(a, m))),
            T::Slice(a) => T::Slice(Box::new(self.inst(a, m))),
            T::Opt(a) => T::Opt(Box::new(self.inst(a, m))),
            T::Chan(a) => T::Chan(Box::new(self.inst(a, m))),
            T::Ref(b, a) => T::Ref(*b, Box::new(self.inst(a, m))),
            T::Region(a, r) => T::Region(Box::new(self.inst(a, m)), r.clone()),
            T::Map(a, b) => T::Map(Box::new(self.inst(a, m)), Box::new(self.inst(b, m))),
            T::Union(a, b) => T::Union(Box::new(self.inst(a, m)), Box::new(self.inst(b, m))),
            T::Fn(a, r) => T::Fn(
                a.iter().map(|x| self.inst(x, m)).collect(),
                Box::new(self.inst(r, m)),
            ),
            o => o.clone(),
        }
    }

    fn call_sig(&mut self, sig: &FnSig, args: &[Arg], l: usize, self_ty: Option<T>) -> T {
        let saved = self.gbounds.clone();
        let saved_self = self.self_ty.clone();
        if self_ty.is_some() {
            self.self_ty = self_ty.clone();
        }
        let mut m = HashMap::new();
        let mut bounds = vec![];
        let targs = self.targs.take();
        if let Some(ts) = &targs
            && ts.len() != sig.generics.len()
        {
            self.e(d(l, "W-TY-1", format!("`{}` takes {} type arguments, given {}", sig.name, sig.generics.len(), ts.len())));
        }
        for (gi, g) in sig.generics.iter().enumerate() {
            self.gbounds.insert(g.name.clone(), vec![]);
            let v = self.sub.fresh();
            if let Some(t) = targs.as_ref().and_then(|ts| ts.get(gi)) {
                let t = self.lower(t, l);
                self.sub.unify(&v, &t);
            }
            m.insert(g.name.clone(), v.clone());
            bounds.push((v, g.bounds.iter().map(ty_head).collect::<Vec<_>>()));
        }
        let params: Vec<(Option<String>, String, Ty)> = sig
            .params
            .iter()
            .filter_map(|p| {
                if let Param::Named {
                    label, name, ty, ..
                } = p
                {
                    Some((label.clone(), name.clone(), ty.clone()))
                } else {
                    None
                }
            })
            .collect();
        let pts: Vec<T> = params.iter().map(|p| self.lower(&p.2, l)).collect();
        let ret = sig
            .ret
            .as_ref()
            .map(|r| self.lower(r, l))
            .unwrap_or(T::Unit);
        self.gbounds = saved;
        self.self_ty = saved_self;
        let pts: Vec<T> = pts.iter().map(|t| self.inst(t, &m)).collect();
        let ret = self.inst(&ret, &m);
        if params.len() != args.len() {
            self.e(d(
                l,
                "W-TY-1",
                format!(
                    "`{}` takes {} arguments, found {}",
                    sig.name,
                    params.len(),
                    args.len()
                ),
            ));
        }
        for (i, (p, a)) in params.iter().zip(args).enumerate() {
            match (&p.0, &a.label) {
                (Some(lb), Some(al)) if lb != al => self.e(d(
                    l,
                    "W-TY-1",
                    format!("argument label `{al}` should be `{lb}`"),
                )),
                (Some(lb), None) => self.e(d(
                    l,
                    "W-TY-1",
                    format!(
                        "argument {} of `{}` needs the label `{lb}`",
                        i + 1,
                        sig.name
                    ),
                )),
                (None, Some(al)) if *al != p.1 => self.e(d(
                    l,
                    "W-TY-1",
                    format!("`{}` has no parameter `{al}`", sig.name),
                )),
                _ => {}
            }
            let at = self.infer(&a.e, true);
            self.expect(&at, &pts[i], l);
        }
        for (v, bs) in bounds {
            for b in bs {
                if !matches!(self.sub.resolve(&v), T::Var(_)) && !self.has_trait(&v, &b) {
                    let r = self.sub.resolve(&v);
                    self.e(d(l, "W-TRAIT-3", format!("{r} does not implement {b}")));
                }
            }
        }
        ret
    }
}

impl Checker {
    fn need_mut_receiver(&mut self, obj: &Expr, m: &str, l: usize) {
        let ok = match place_root(obj).and_then(|r| self.lookup(&r)) {
            Some(lc) => {
                lc.mutable
                    || matches!(lc.ty, T::Ref(true, _))
                    || (lc.field && matches!(self.cx, Cx::Step | Cx::Handler))
            }
            None => false,
        };
        if !ok {
            self.e(d(
                l,
                "W-PLACE",
                format!("`.{m}` modifies its receiver, which must be mutable"),
            ));
        }
    }

    fn method_call(&mut self, obj: &Expr, m: &str, args: &[Arg], l: usize) -> T {
        if m == "collect" {
            self.iter_ok = true;
        }
        let ot = self.infer(obj, false);
        self.iter_ok = false;
        if let T::Iter(e) = self.sub.resolve(&ot) {
            if m != "collect" {
                self.e(d(
                    l,
                    "W-ITER-1",
                    format!("`.{m}` is not defined on an iterator (only `.collect()`)"),
                ));
                return T::Never;
            }
            self.arity(m, args, 0, l);
            return T::List(e);
        }
        let rt = self.sub.resolve(&ot);
        let base = peel(&rt);
        if self.v08 && let T::Named(n, _) = &base && crate::index08::INDEXED.contains(&n.as_str()) {
            if n == "Text" && m == "raw" && args.is_empty() {
                return T::Text;
            }
            // F.5.1: `c.narrow(sub)` is a `Cap[A]` with a longer root; nothing widens
            if self.v09 && n == "Cap" && m == "narrow" && args.len() == 1 {
                let at = self.infer(&args[0].e, false);
                self.expect(&at, &T::Text, l);
                if let Expr::Str(s, _) = &args[0].e
                    && (s.is_empty() || s.starts_with('/') || s.split('/').any(|p| p == ".." || p == "."))
                {
                    self.e(d(l, "W-CAP-3", format!("`narrow({s:?})` is not a path extension: a narrowed root is relative, non-empty and has no `.` or `..` segment (F.5.1)")));
                }
                return rt.clone();
            }
            self.e(d(l, if n == "Text" { "W-CTX-1" } else if n == "Secret" { "W-SEC-1" } else { "W-CAP-1" }, format!("`.{m}` is not defined on {base} (5.16: context text has `.raw()`, secrets and capabilities are opaque)")));
            return T::Never;
        }
        // user methods
        let head = match &base {
            T::Named(n, _) => Some(n.clone()),
            T::List(_) => Some("List".into()),
            _ => None,
        };
        if let Some(h) = &head
            && let Some(fi) = self
                .methods
                .iter()
                .find(|(hh, fi)| hh == h && fi.decl.sig.name == m)
                .map(|x| x.1.clone())
        {
            match fi.decl.sig.params.first() {
                Some(Param::SelfRef(true)) => self.need_mut_receiver(obj, m, l),
                Some(Param::SelfVal) => {
                    if is_place(obj)
                        && !self.is_copy(&base)
                        && let Some(r) = place_root(obj)
                    {
                        self.set_moved(&r, true);
                    }
                }
                Some(Param::SelfRef(false)) => {}
                _ => {
                    self.e(d(
                        l,
                        "W-TY-1",
                        format!("`{m}` is not a method (call it as {h}.{m})"),
                    ));
                }
            }
            self.effect_call(&format!("{h}.{m}"), fi.cx, l);
            let mut sig = fi.decl.sig.clone();
            sig.params.retain(|p| matches!(p, Param::Named { .. }));
            // instantiate the impl's generics from the receiver
            let mut gb = self.gbounds.clone();
            if let Some((_, _, g)) = &fi.owner {
                for x in g {
                    gb.insert(x.name.clone(), vec![]);
                }
            }
            let saved = std::mem::replace(&mut self.gbounds, gb);
            let r = self.call_sig(&sig, args, l, Some(base.clone()));
            self.gbounds = saved;
            if let (Some((oty, _, g)), false) =
                (&fi.owner, fi.owner.as_ref().is_none_or(|o| o.2.is_empty()))
            {
                // map the impl's type parameters onto the receiver's arguments
                let mut mp = HashMap::new();
                if let Ty::Path(_, targs) = oty {
                    let rargs: Vec<T> = match &base {
                        T::List(e) => vec![(**e).clone()],
                        T::Named(_, a) => a.clone(),
                        _ => vec![],
                    };
                    for (ta, ra) in targs.iter().zip(rargs) {
                        if let Ty::Path(p, _) = ta
                            && g.iter().any(|x| x.name == p[0])
                        {
                            mp.insert(p[0].clone(), ra);
                        }
                    }
                }
                return self.inst(&r, &mp);
            }
            return r;
        }
        // trait methods through a generic bound
        if let T::Param(p) = &base {
            let bounds = self.gbounds.get(p).cloned().unwrap_or_default();
            for b in bounds {
                if let Some(TyDef::Trait { members }) = self.types.get(&b).cloned() {
                    for mem in members {
                        if let TraitMember::Fn(sig) = mem
                            && sig.name == m
                        {
                            let mut s2 = sig.clone();
                            s2.params.retain(|q| matches!(q, Param::Named { .. }));
                            let saved = self.self_ty.replace(base.clone());
                            let r = self.call_sig(&s2, args, l, Some(base.clone()));
                            self.self_ty = saved;
                            return r;
                        }
                    }
                }
            }
        }
        self.builtin_method(obj, &base, m, args, l)
    }

    fn arg_types(&mut self, args: &[Arg]) -> Vec<T> {
        args.iter().map(|a| self.infer(&a.e, true)).collect()
    }
    fn arity(&mut self, m: &str, args: &[Arg], n: usize, l: usize) -> bool {
        if args.len() != n {
            self.e(d(
                l,
                "W-TY-1",
                format!("`.{m}` takes {n} arguments, found {}", args.len()),
            ));
            false
        } else {
            true
        }
    }

    fn builtin_method(&mut self, obj: &Expr, base: &T, m: &str, args: &[Arg], l: usize) -> T {
        let usize_ = T::Int("usize");
        let unknown = |c: &mut Checker| {
            c.e(d(l, "W-NAME-1", format!("{base} has no method `{m}`")));
            T::Never
        };
        if let Some(t) = self.method06b(base, m, args, l) {
            return t;
        }
        // 5.15, read from spec-data.json: a method the table does not list
        // for this receiver does not exist (W-NAME-1)
        let recv = match base {
            T::Int(_) => Some("Int"),
            T::F64 => Some("F"),
            T::Bool => Some("Bool"),
            T::Text => Some("Text"),
            T::F32 => Some("F32"),
            T::Iter(_) => Some("Gen"),
            T::List(_) | T::Array(..) => Some("List"),
            T::Map(..) => Some("Map"),
            T::Set(_) => Some("Set"),
            T::Opt(_) => Some("Opt"),
            _ => None,
        };
        if let Some(r) = recv
            && !crate::specread::has_method(r, m)
        {
            return unknown(self);
        }
        match base {
            T::Int(_) => match m {
                "checked_add" | "checked_sub" | "checked_mul" | "checked_div" => {
                    if self.arity(m, args, 1, l) {
                        let a = self.infer(&args[0].e, true);
                        self.expect(&a, base, l);
                    }
                    T::Opt(Box::new(base.clone()))
                }
                "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add"
                | "saturating_sub" | "saturating_mul" => {
                    if self.arity(m, args, 1, l) {
                        let a = self.infer(&args[0].e, true);
                        self.expect(&a, base, l);
                    }
                    base.clone()
                }
                "tostr" => {
                    self.arity(m, args, 0, l);
                    T::Text
                }
                "abs" | "min" | "max" => {
                    let ts = self.arg_types(args);
                    for t in ts {
                        self.expect(&t, base, l);
                    }
                    base.clone()
                }
                _ => unknown(self),
            },
            T::Var(_)
                if self.sub.intlit[match base {
                    T::Var(i) => *i,
                    _ => 0,
                }] =>
            {
                self.sub.unify(base, &T::Int("i64"));
                self.builtin_method(obj, &T::Int("i64"), m, args, l)
            }
            T::F64 => match m {
                "sqrt" | "abs" | "floor" | "ceil" | "round" => {
                    self.arity(m, args, 0, l);
                    T::F64
                }
                "tostr" => T::Text,
                "min" | "max" => {
                    self.arity(m, args, 1, l);
                    let ts = self.arg_types(args);
                    for t in ts {
                        self.expect(&t, &T::F64, l);
                    }
                    T::F64
                }
                _ => unknown(self),
            },
            T::F32 => match m {
                "abs" => {
                    self.arity(m, args, 0, l);
                    T::F32
                }
                "tostr" => T::Text,
                _ => unknown(self),
            },
            T::Bool => match m {
                "tostr" => T::Text,
                _ => unknown(self),
            },
            T::Text => match m {
                "len" if self.v09 => {
                    self.e(d(l, "W-LEN-1", "text has no unqualified `len` in a 0.9 module: use `len_bytes` or `len_chars` (F.5.2)"));
                    usize_
                }
                "len_bytes" | "len_chars" if self.v09 => {
                    self.arity(m, args, 0, l);
                    usize_
                }
                "len" => {
                    self.arity(m, args, 0, l);
                    usize_
                }
                "strip" | "upper" | "lower" | "reverse" | "tostr" | "sha256" => {
                    self.arity(m, args, 0, l);
                    T::Text
                }
                "chars" | "words" | "lines" => {
                    self.arity(m, args, 0, l);
                    T::List(Box::new(T::Text))
                }
                "split" => {
                    let ts = self.arg_types(args);
                    for t in ts {
                        self.expect(&peel(&t), &T::Text, l);
                    }
                    T::List(Box::new(T::Text))
                }
                "isdigit" | "isalpha" | "isspace" | "isempty" => {
                    self.arity(m, args, 0, l);
                    T::Bool
                }
                "contains" | "startswith" | "endswith" => {
                    let ts = self.arg_types(args);
                    for t in ts {
                        self.expect(&peel(&t), &T::Text, l);
                    }
                    T::Bool
                }
                "toint" => {
                    self.arity(m, args, 0, l);
                    T::Int("i64")
                }
                "join" => {
                    if self.arity(m, args, 1, l) {
                        let t = self.infer(&args[0].e, true);
                        match peel(&self.sub.resolve(&t)) {
                            T::List(e) | T::Slice(e) | T::Array(e, _) => {
                                self.expect(&e, &T::Text, l)
                            }
                            o => self.e(d(
                                l,
                                "W-TY-1",
                                format!("`.join` needs a list of text, found {o}"),
                            )),
                        }
                    }
                    T::Text
                }
                _ => unknown(self),
            },
            T::List(e) | T::Array(e, _) | T::Slice(e) => {
                let e = (**e).clone();
                let elem_view = if matches!(base, T::List(_)) && !self.is_copy(&e) {
                    T::Ref(false, Box::new(e.clone()))
                } else {
                    e.clone()
                };
                let _ = elem_view;
                match m {
                    "len" => {
                        self.arity(m, args, 0, l);
                        usize_
                    }
                    "isempty" => T::Bool,
                    "append" | "push" => {
                        if !matches!(base, T::List(_)) {
                            self.e(d(l, "W-TY-1", "only a List can grow"));
                        }
                        self.need_mut_receiver(obj, m, l);
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, true);
                            if contains_ref(&self.sub.resolve(&t))
                                && !matches!(self.sub.resolve(&e), T::Ref(..))
                            {
                                self.e(d(
                                    l,
                                    "W-OWN-3",
                                    "a list cannot hold borrows (copy the element)",
                                ));
                            }
                            self.expect(&peel(&t), &e, l);
                            if let Some(r) = place_root(obj) {
                                self.region_escape(&r, &t, l);
                            }
                        }
                        T::Unit
                    }
                    "pop" => {
                        self.need_mut_receiver(obj, m, l);
                        T::Opt(Box::new(e))
                    }
                    "clear" | "sort" => {
                        self.need_mut_receiver(obj, m, l);
                        T::Unit
                    }
                    "with" => {
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, true);
                            self.expect(&t, &e, l);
                        }
                        T::List(Box::new(e))
                    }
                    "reverse" => T::List(Box::new(e)),
                    "contains" => {
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, false);
                            self.expect(&peel(&t), &e, l);
                        }
                        T::Bool
                    }
                    "get" => {
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, false);
                            self.expect(&t, &usize_, l);
                        }
                        T::Opt(Box::new(e))
                    }
                    "utf8" => {
                        self.arity(m, args, 0, l);
                        self.expect(&e, &T::Int("i64"), l);
                        T::Text
                    }
                    "sum" => {
                        if !self.num(&e) {
                            self.e(d(l, "W-TY-1", "`.sum` needs numbers"));
                        }
                        e
                    }
                    "map" => {
                        if self.arity(m, args, 1, l) {
                            let ft = self.infer(&args[0].e, true);
                            let r = self.sub.fresh();
                            self.expect(&ft, &T::Fn(vec![e.clone()], Box::new(r.clone())), l);
                            return T::List(Box::new(r));
                        }
                        T::Never
                    }
                    "filter" => {
                        if self.arity(m, args, 1, l) {
                            let ft = self.infer(&args[0].e, true);
                            self.expect(&ft, &T::Fn(vec![e.clone()], Box::new(T::Bool)), l);
                        }
                        T::List(Box::new(e))
                    }
                    "fold" => {
                        if self.arity(m, args, 2, l) {
                            let it = self.infer(&args[0].e, true);
                            let ft = self.infer(&args[1].e, true);
                            self.expect(
                                &ft,
                                &T::Fn(vec![it.clone(), e.clone()], Box::new(it.clone())),
                                l,
                            );
                            return it;
                        }
                        T::Never
                    }
                    _ => unknown(self),
                }
            }
            T::Set(e) => {
                let e = (**e).clone();
                match m {
                    "union" | "inter" => {
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, false);
                            self.expect(&peel(&t), base, l);
                        }
                        base.clone()
                    }
                    "insert" => {
                        self.need_mut_receiver(obj, m, l);
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, true);
                            self.expect(&t, &e, l);
                        }
                        T::Unit
                    }
                    "tolist" => {
                        self.arity(m, args, 0, l);
                        T::List(Box::new(e))
                    }
                    "len" => {
                        self.arity(m, args, 0, l);
                        usize_
                    }
                    "contains" => {
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, false);
                            self.expect(&peel(&t), &e, l);
                        }
                        T::Bool
                    }
                    _ => unknown(self),
                }
            }
            T::Map(k, v) => {
                let (k, v) = ((**k).clone(), (**v).clone());
                match m {
                    "merge" => {
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, false);
                            self.expect(&peel(&t), base, l);
                        }
                        // a key in both maps merges its values: v needs a Merge (C.11.5)
                        let vt = self.sub.resolve(&v);
                        if let Some(g) = (!matches!(vt, T::Var(_))).then(|| self.merge_gap(&vt)).flatten() {
                            self.e(d(l, "W-CRDT-1", format!("Map.merge needs a value type with Merge, found {vt} ({g})")));
                        }
                        base.clone()
                    }
                    "len" => usize_,
                    "get" => {
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, false);
                            self.expect(&peel(&t), &k, l);
                        }
                        T::Union(Box::new(v), Box::new(T::Named("Missing".into(), vec![])))
                    }
                    "contains" => {
                        if self.arity(m, args, 1, l) {
                            let t = self.infer(&args[0].e, false);
                            self.expect(&peel(&t), &k, l);
                        }
                        T::Bool
                    }
                    "insert" => {
                        self.need_mut_receiver(obj, m, l);
                        if self.arity(m, args, 2, l) {
                            let a = self.infer(&args[0].e, true);
                            self.expect(&a, &k, l);
                            let b = self.infer(&args[1].e, true);
                            self.expect(&b, &v, l);
                        }
                        T::Unit
                    }
                    "remove" => {
                        self.need_mut_receiver(obj, m, l);
                        if self.arity(m, args, 1, l) {
                            let a = self.infer(&args[0].e, false);
                            self.expect(&peel(&a), &k, l);
                        }
                        T::Opt(Box::new(v))
                    }
                    "keys" => T::List(Box::new(k)),
                    "values" => T::List(Box::new(v)),
                    _ => unknown(self),
                }
            }
            T::Chan(_) => match m {
                "close" => {
                    if !matches!(self.cx, Cx::Main | Cx::Task) {
                        self.e(d(l, "W-EFF-1", "close outside main or a task function"));
                    }
                    T::Unit
                }
                _ => unknown(self),
            },
            T::Opt(x) => match m {
                "isnone" | "issome" => T::Bool,
                "or" => {
                    if self.arity(m, args, 1, l) {
                        let t = self.infer(&args[0].e, true);
                        self.expect(&t, x, l);
                    }
                    (**x).clone()
                }
                _ => unknown(self),
            },
            T::Named(n, a) if n == "Box" => {
                let inner = a[0].clone();
                self.builtin_method(obj, &inner, m, args, l)
            }
            T::Never => T::Never,
            _ => unknown(self),
        }
    }
}

// ---- draft 0.6 concurrency (Appendix C, book/14-draft06-concurrency.md) ----
//
// The parser encodes the 0.6 statements as marker regions whose names
// start with `$` (not a lexable name), so the AST and the evaluator are
// unchanged: `$par` (children `$branch:NAME`), `$forpar`, `$budget`,
// `$deadline` (children n, `$dlbody`, optional `if false` else block),
// `$nondet`, `$priority`. Footprints are kept both per root variable
// (used by W-PAR-3/4 and the lvar rules) and as borrow paths (C.3, used
// by W-PAR-1): a path is a root followed by field (`.f`), tuple (`.0`) and
// index segments; an index is `[n]` for an integer literal and `[*]`
// otherwise. Two paths overlap when one is a prefix of the other, where
// `[*]` overlaps every index and distinct literals do not.

#[derive(Default)]
struct Foot {
    writes: std::collections::BTreeSet<String>,
    reads: std::collections::BTreeSet<String>,
    declared: std::collections::BTreeSet<String>,
    effect: Option<&'static str>,
    escape: bool,
    wpaths: std::collections::BTreeSet<Vec<String>>,
    rpaths: std::collections::BTreeSet<Vec<String>>,
}

/// The borrow path of a place expression (C.3), or None if `e` is not a place.
pub(crate) fn place_path(e: &Expr) -> Option<Vec<String>> {
    match e {
        Expr::Name(n, _) => Some(vec![n.clone()]),
        Expr::SelfRef(_) => Some(vec!["self".into()]),
        Expr::Member(x, m, _) => place_path(x).map(|mut p| {
            p.push(format!(".{m}"));
            p
        }),
        Expr::TupleMember(x, i, _) => place_path(x).map(|mut p| {
            p.push(format!(".{i}"));
            p
        }),
        Expr::Index(x, ix, _) => place_path(x).map(|mut p| {
            match ix.as_slice() {
                [Expr::Int(n, _)] => p.push(format!("[{n}]")),
                _ => p.push("[*]".into()),
            }
            p
        }),
        Expr::Slice(x, ..) => place_path(x).map(|mut p| {
            p.push("[*]".into());
            p
        }),
        Expr::Borrow(_, x, _) => place_path(x),
        _ => None,
    }
}

/// Prefix overlap of C.3 / 5.6.
pub(crate) fn paths_overlap(a: &[String], b: &[String]) -> bool {
    a.iter().zip(b).all(|(x, y)| {
        x == y || (x.starts_with('[') && y.starts_with('[') && (x == "[*]" || y == "[*]"))
    })
}

/// Read paths of `e`: maximal places read, plus the reads in their indices.
fn read_paths(e: &Expr, out: &mut std::collections::BTreeSet<Vec<String>>) {
    let mut total: std::collections::BTreeMap<Vec<String>, usize> = Default::default();
    let mut inner: std::collections::BTreeMap<Vec<String>, usize> = Default::default();
    walk_expr(e, &mut |x| {
        if let Some(p) = place_path(x)
            && !matches!(x, Expr::Borrow(..))
        {
            *total.entry(p).or_default() += 1;
            if let Expr::Member(o, _, _)
            | Expr::TupleMember(o, _, _)
            | Expr::Index(o, _, _)
            | Expr::Slice(o, ..) = x
                && let Some(q) = place_path(o)
            {
                *inner.entry(q).or_default() += 1;
            }
        }
    });
    for (p, n) in total {
        if n > inner.get(&p).copied().unwrap_or(0) {
            out.insert(p);
        }
    }
}

pub(crate) fn pat_names_pub(p: &Pat, out: &mut std::collections::BTreeSet<String>) {
    pat_names(p, out)
}

fn pat_names(p: &Pat, out: &mut std::collections::BTreeSet<String>) {
    match p {
        Pat::Path(v) if v.len() == 1 => {
            out.insert(v[0].clone());
        }
        Pat::Ctor(_, args) => args.iter().for_each(|(_, q)| pat_names(q, out)),
        Pat::Tuple(v) | Pat::Alt(v) => v.iter().for_each(|q| pat_names(q, out)),
        _ => {}
    }
}

fn foot_expr(e: &Expr, f: &mut Foot) {
    read_paths(e, &mut f.rpaths);
    walk_expr(e, &mut |x| match x {
        Expr::Name(n, _) => {
            f.reads.insert(n.clone());
        }
        Expr::Borrow(true, y, _) => {
            if let Some(r) = place_root(y) {
                f.writes.insert(r);
            }
            if let Some(p) = place_path(y) {
                f.wpaths.insert(p);
            }
        }
        Expr::Recv(..) => f.effect = f.effect.or(Some("receive")),
        Expr::Call(c, _, _, _) => {
            if let Expr::Member(o, m, _) = &**c
                && mutating_method(m)
                && let Some(r) = place_root(o)
            {
                f.writes.insert(r);
                if let Some(p) = place_path(o) {
                    f.wpaths.insert(p);
                }
            }
        }
        _ => {}
    });
}

fn foot_block(b: &[Stmt], loops: usize, f: &mut Foot) {
    for s in b {
        for e in stmt_exprs(s) {
            foot_expr(e, f);
        }
        match s {
            Stmt::Assign(_, p, _) => {
                if let Some(r) = place_root(p) {
                    f.writes.insert(r);
                }
                if let Some(q) = place_path(p) {
                    f.wpaths.insert(q);
                }
            }
            Stmt::Var(n, _, _, _) => {
                f.declared.insert(n.clone());
            }
            Stmt::Let(p, _, _, _) => pat_names(p, &mut f.declared),
            Stmt::Emit(_) => f.effect = f.effect.or(Some("emit")),
            Stmt::Send(..) => f.effect = f.effect.or(Some("send")),
            Stmt::Spawn(_) => f.effect = f.effect.or(Some("spawn")),
            Stmt::Select(..) => f.effect = f.effect.or(Some("select")),
            Stmt::Yield(..) => f.effect = f.effect.or(Some("yield")),
            Stmt::Return(..) => f.escape = true,
            Stmt::Break(_) | Stmt::Continue(_) if loops == 0 => f.escape = true,
            _ => {}
        }
        match s {
            Stmt::If(arms, els) => {
                for (_, x) in arms {
                    foot_block(x, loops, f);
                }
                if let Some(x) = els {
                    foot_block(x, loops, f);
                }
            }
            Stmt::While(_, b2, x) | Stmt::Loop(b2, x) => {
                if let Some(b2) = b2 {
                    foot_expr(b2, f);
                }
                foot_block(x, loops + 1, f)
            }
            Stmt::For(p, _, x, _) => {
                pat_names(p, &mut f.declared);
                foot_block(x, loops + 1, f)
            }
            Stmt::Arena(r, bnd, x, _) | Stmt::Device(r, bnd, x, _) => {
                if r.starts_with("$priority") || r.starts_with("$nondet") {
                    f.effect = f.effect.or(Some("spawn or select"));
                }
                if let Some(b2) = bnd {
                    foot_expr(b2, f);
                }
                foot_block(x, loops, f)
            }
            Stmt::Comptime(x) => foot_block(x, loops, f),
            Stmt::Match(_, arms, _) => {
                for a in arms {
                    pat_names(&a.pat, &mut f.declared);
                    if let Some(g) = &a.guard {
                        foot_expr(g, f);
                    }
                    foot_block(&a.body, loops, f)
                }
            }
            Stmt::Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(_, c, x) | SelArm::After(c, x) => {
                            foot_expr(c, f);
                            foot_block(x, loops, f)
                        }
                        SelArm::Send(v, c, x) => {
                            foot_expr(v, f);
                            foot_expr(c, f);
                            foot_block(x, loops, f)
                        }
                        SelArm::Default(x) => foot_block(x, loops, f),
                    }
                }
            }
            _ => {}
        }
    }
}

fn footprint(b: &[Stmt]) -> Foot {
    let mut f = Foot::default();
    foot_block(b, 0, &mut f);
    let decl = f.declared.clone();
    f.writes.retain(|n| !decl.contains(n));
    f.reads.retain(|n| !decl.contains(n));
    f.wpaths.retain(|p| !decl.contains(&p[0]));
    f.rpaths.retain(|p| !decl.contains(&p[0]));
    f
}

impl Checker {
    fn conc06(&mut self, r: &str, body: &[Stmt], line: usize) -> bool {
        match r {
            "$par" => self.par06(body, line),
            "$forpar" => self.forpar06(body, line),
            "$budget" => {
                if let Some(Stmt::Expr(n)) = body.first() {
                    let t = self.infer(n, false);
                    self.expect(&t, &T::Int("u64"), line);
                }
                let (rest, else_b) = match body.last() {
                    Some(Stmt::If(arms, _)) if arms.len() == 1
                        && matches!(&arms[0].0, Expr::Bool(false, _)) =>
                        (&body[1..body.len() - 1], Some(&arms[0].1[..])),
                    _ => (&body[1..], None),
                };
                let a = self.block(rest);
                match else_b {
                    Some(eb) => {
                        // W-BUD-1: a budget with an else is undone as a whole
                        // when it traps, so its block may hold only effects
                        // that can be undone (its own writes)
                        let mut put = None;
                        walk_stmts(rest, &mut |s| {
                            for e in stmt_exprs(s) {
                                walk_expr(e, &mut |x| {
                                    if let Expr::Call(c, _, _, _) = x
                                        && let Expr::Member(_, m, _) = &**c
                                        && (m == "put" || m == "freeze")
                                    {
                                        put = Some("lvar put or freeze");
                                    }
                                });
                            }
                        });
                        if let Some(eff) = footprint(rest).effect.or(put) {
                            self.e(d(
                                line,
                                "W-BUD-1",
                                format!("`with budget ... else` block has an effect ({eff}) that a trap cannot undo"),
                            ));
                        }
                        if footprint(rest).escape {
                            self.e(d(
                                line,
                                "W-BUD-1",
                                "`with budget ... else` block returns, breaks or continues out of it".to_string(),
                            ));
                        }
                        self.push_scope_pub();
                        self.declare("\u{1}__bn", T::Int("u64"), false, line, false);
                        self.declare("\u{1}__bu", T::Int("u64"), false, line, false);
                        let b = self.block(eb);
                        self.pop_scope_pub();
                        a && b
                    }
                    None => a,
                }
            }
            "$deadline" => {
                if !matches!(self.cx, Cx::Main | Cx::Task) {
                    self.e(d(
                        line,
                        "W-DL-1",
                        "with deadline outside main or a task function",
                    ));
                }
                if let Some(Stmt::Expr(n)) = body.first() {
                    let t = self.infer(n, false);
                    self.expect(&t, &T::Int("u64"), line);
                }
                let mut div = true;
                let before = self.moved_set();
                for s in &body[1..] {
                    match s {
                        Stmt::Arena(_, _, b, _) => div &= self.block(b),
                        Stmt::If(arms, _) => {
                            // the else block runs instead of the rest of the body
                            self.restore_moved(&before);
                            for (_, b) in arms {
                                div &= self.block(b);
                            }
                        }
                        _ => {}
                    }
                }
                if body.len() == 2 {
                    div = false;
                }
                div
            }
            "$nondet" => {
                if !matches!(self.cx, Cx::Main | Cx::Task) {
                    self.e(d(
                        line,
                        "W-ND-1",
                        "select nondet outside main or a task function",
                    ));
                }
                self.block(body)
            }
            "$priority" => {
                if let Some(Stmt::Expr(p)) = body.first() {
                    let t = self.infer(p, false);
                    self.expect(&t, &T::Int("u8"), line);
                }
                self.stmt(&body[1])
            }
            "$dlbody" => self.block(body),
            r if r.starts_with("$spawnas:")
                || r.starts_with("$group:")
                || matches!(r, "$scope" | "$atomic" | "$barrier") =>
            {
                self.marker06b(r, body, line)
            }
            _ => self.block(body),
        }
    }

    fn par06(&mut self, branches: &[Stmt], line: usize) -> bool {
        let mut names = std::collections::BTreeSet::new();
        let mut feet = vec![];
        for b in branches {
            let Stmt::Arena(n, _, body, bl) = b else { continue };
            let n = n.trim_start_matches("$branch:").to_string();
            if !names.insert(n.clone()) {
                self.e(d(*bl, "W-NAME-2", format!("branch `{n}` declared twice")));
            }
            self.block(body);
            let f = footprint(body);
            if let Some(eff) = f.effect {
                self.e(d(
                    *bl,
                    "W-PAR-2",
                    format!("branch `{n}` has an effect ({eff}); par branches are pure apart from their own writes"),
                ));
            }
            if f.escape {
                self.e(d(
                    *bl,
                    "W-PAR-2",
                    format!("branch `{n}` returns, breaks or continues out of the par"),
                ));
            }
            feet.push((n, f, *bl));
        }
        let bodies: Vec<&[Stmt]> = branches
            .iter()
            .filter_map(|b| match b {
                Stmt::Arena(_, _, body, _) => Some(&body[..]),
                _ => None,
            })
            .collect();
        self.lvar_freeze06(&bodies, line);
        for i in 0..feet.len() {
            for j in 0..feet.len() {
                if i == j {
                    continue;
                }
                let (ni, fi, li) = &feet[i];
                let (nj, fj, _) = &feet[j];
                let mut seen = std::collections::BTreeSet::new();
                for w in &fi.wpaths {
                    if let Some(o) = fj.wpaths.iter().chain(&fj.rpaths).find(|o| paths_overlap(w, o))
                        && seen.insert(w[0].clone())
                    {
                        let msg = format!(
                            "branch `{ni}` writes `{}`, which overlaps `{}` used by branch `{nj}` (race)",
                            w.concat(),
                            o.concat()
                        );
                        self.e(d(*li.max(&line), "W-PAR-1", msg));
                    }
                }
            }
        }
        false
    }

    fn forpar06(&mut self, body: &[Stmt], line: usize) -> bool {
        let Some(Stmt::For(p, it, b, _)) = body.first() else {
            return false;
        };
        self.stmt(&body[0]);
        self.lvar_freeze06(&[&b[..]], line);
        let f = footprint(b);
        if let Some(eff) = f.effect {
            self.e(d(line, "W-PAR-4", format!("for ... par body has an effect ({eff})")));
        }
        let mut esc = Foot::default();
        foot_block(b, 0, &mut esc);
        if esc.escape {
            self.e(d(
                line,
                "W-PAR-4",
                "for ... par body cannot return, break or continue",
            ));
        }
        let mut loopvar = std::collections::BTreeSet::new();
        pat_names(p, &mut loopvar);
        let range = matches!(it, Expr::Bin(op, _, _, _) if op == ".." || op == "..=");
        for w in &f.writes {
            if loopvar.contains(w) {
                continue;
            }
            let mut red = 0usize;
            let mut slot = 0usize;
            let mut other = 0usize;
            let mut uses = 0usize;
            walk_stmts(b, &mut |s| {
                if let Stmt::Assign(op, pl, _) = s {
                    match pl {
                        Expr::Name(n, _) if n == w => {
                            if ["+=", "*=", "&=", "|=", "^="].contains(&op.as_str()) {
                                red += 1
                            } else {
                                other += 1
                            }
                        }
                        Expr::Index(a, ix, _)
                            if matches!(&**a, Expr::Name(n, _) if n == w)
                                && op == "="
                                && range
                                && ix.len() == 1
                                && matches!(&ix[0], Expr::Name(i, _) if loopvar.contains(i)) =>
                        {
                            slot += 1
                        }
                        _ if place_root(pl).as_deref() == Some(w.as_str()) => other += 1,
                        _ => {}
                    }
                }
                for e in stmt_exprs(s) {
                    walk_expr(e, &mut |x| {
                        if matches!(x, Expr::Name(n, _) if n == w) {
                            uses += 1
                        }
                    });
                }
            });
            // each reduction or slot assignment mentions its target once (as a place)
            let ok = other == 0
                && ((red > 0 && slot == 0 && uses == red) || (slot > 0 && red == 0 && uses == slot));
            if !ok {
                self.e(d(
                    line,
                    "W-PAR-3",
                    format!(
                        "for ... par writes `{w}` other than as a reduction (+= *= &= |= ^=, never read) or as slot [i] of a range loop"
                    ),
                ));
            }
        }
        false
    }
}

/// A call inside a recursive cycle (spec 6.7, mutual recursion): the callee,
/// the line, and for each argument the caller's parameter positions it is
/// tied to: `(i, false)` when it is parameter i itself, `(i, true)` when it
/// is a name bound inside a constructor pattern of a `match` on parameter i
/// (or on such a name), as for structural recursion.
#[derive(Clone, Debug)]
pub struct RecSite {
    pub callee: String,
    pub line: usize,
    pub args: Vec<Vec<(usize, bool)>>,
}

#[derive(Clone)]
struct RecSt {
    sm: Vec<std::collections::HashSet<String>>,
    eq: Vec<bool>,
}

impl RecSt {
    fn unbind(&mut self, n: &str, params: &[String]) {
        for s in self.sm.iter_mut() {
            s.remove(n);
        }
        for (i, p) in params.iter().enumerate() {
            if p == n {
                self.eq[i] = false;
            }
        }
    }
}

pub fn rec_sites(f: &FnDecl, is_target: &dyn Fn(&str) -> bool, is_ctor: &dyn Fn(&str) -> bool) -> Vec<RecSite> {
    let params: Vec<String> = f
        .sig
        .params
        .iter()
        .filter_map(|p| if let Param::Named { name, .. } = p { Some(name.clone()) } else { None })
        .collect();
    fn sites_e(e: &Expr, params: &[String], st: &RecSt, is_target: &dyn Fn(&str) -> bool, out: &mut Vec<RecSite>) {
        walk_expr(e, &mut |x| {
            if let Expr::Call(g, args, l, _) = x
                && let Expr::Name(n, _) = &**g
                && is_target(n)
            {
                let args = args
                    .iter()
                    .map(|a| match strip_borrow(&a.e) {
                        Expr::Name(v, _) => (0..params.len())
                            .filter_map(|i| {
                                if st.sm[i].contains(v) {
                                    Some((i, true))
                                } else if st.eq[i] && params[i] == *v {
                                    Some((i, false))
                                } else {
                                    None
                                }
                            })
                            .collect(),
                        _ => vec![],
                    })
                    .collect();
                out.push(RecSite { callee: n.clone(), line: *l, args });
            }
        });
    }
    fn go(b: &[Stmt], params: &[String], st: &mut RecSt, is_target: &dyn Fn(&str) -> bool, is_ctor: &dyn Fn(&str) -> bool, out: &mut Vec<RecSite>) {
        for s in b {
            for e in stmt_direct_exprs(s) {
                sites_e(e, params, st, is_target, out);
            }
            match s {
                Stmt::Let(p, _, _, _) => {
                    let mut v = vec![];
                    pat_binders(p, true, &mut v, is_ctor);
                    for n in v {
                        st.unbind(&n, params);
                    }
                }
                Stmt::Var(n, _, _, _) | Stmt::Assign(_, Expr::Name(n, _), _) => st.unbind(n, params),
                Stmt::Match(scr, arms, _) => {
                    let root = match strip_borrow(scr) {
                        Expr::Name(n, _) => Some(n.clone()),
                        _ => None,
                    };
                    let idx: Vec<usize> = (0..params.len())
                        .filter(|&i| root.as_ref().is_some_and(|r| (*r == params[i] && st.eq[i]) || st.sm[i].contains(r)))
                        .collect();
                    for a in arms {
                        let mut st2 = st.clone();
                        let mut v = vec![];
                        pat_binders(&a.pat, false, &mut v, is_ctor);
                        let mut all = vec![];
                        pat_binders(&a.pat, true, &mut all, is_ctor);
                        for n in &all {
                            st2.unbind(n, params);
                        }
                        for &i in &idx {
                            for n in &v {
                                st2.sm[i].insert(n.clone());
                            }
                        }
                        if let Some(g) = &a.guard {
                            sites_e(g, params, &st2, is_target, out);
                        }
                        go(&a.body, params, &mut st2, is_target, is_ctor, out);
                    }
                }
                Stmt::If(arms, els) => {
                    for (_, x) in arms {
                        go(x, params, &mut st.clone(), is_target, is_ctor, out);
                    }
                    if let Some(x) = els {
                        go(x, params, &mut st.clone(), is_target, is_ctor, out);
                    }
                }
                Stmt::While(_, _, x) | Stmt::Loop(_, x) | Stmt::Arena(_, _, x, _) | Stmt::Device(_, _, x, _) | Stmt::Comptime(x) => {
                    go(x, params, &mut st.clone(), is_target, is_ctor, out)
                }
                Stmt::For(p, _, x, _) => {
                    let mut st2 = st.clone();
                    let mut v = vec![];
                    pat_binders(p, true, &mut v, is_ctor);
                    for n in v {
                        st2.unbind(&n, params);
                    }
                    go(x, params, &mut st2, is_target, is_ctor, out)
                }
                Stmt::Select(arms, _) => {
                    for a in arms {
                        match a {
                            SelArm::Recv(_, _, x) | SelArm::Send(_, _, x) | SelArm::After(_, x) | SelArm::Default(x) => {
                                go(x, params, &mut st.clone(), is_target, is_ctor, out)
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut st = RecSt { sm: vec![std::collections::HashSet::new(); params.len()], eq: vec![true; params.len()] };
    let mut out = vec![];
    match &f.body {
        Body::Arrow(e) => sites_e(e, &params, &st, is_target, &mut out),
        Body::Block(b) => go(b, &params, &mut st, is_target, is_ctor, &mut out),
    }
    out
}

/// Whether the designated positions `pg` of the callee map one to one into
/// the caller's designated positions `pf`, each argument tied to its image
/// (the same parameter, or a component of it) and at least one a strict
/// component (Lean `RCE/Mutual.lean`, `call_dec`): Some(true). Some(false)
/// when such a map exists with no strict component (the measure does not
/// grow), None when there is none.
fn site_decreases(site: &RecSite, pg: &[usize], pf: &[usize]) -> Option<bool> {
    let tie = |j: usize, i: usize| -> Option<bool> {
        site.args.get(j).and_then(|t| t.iter().filter(|(k, _)| *k == i).map(|(_, s)| *s).max())
    };
    // a matching of `rest` into `pf` minus `used`, by augmenting paths
    fn matching(rest: &[usize], pf: &[usize], used: usize, tie: &dyn Fn(usize, usize) -> Option<bool>) -> bool {
        let mut owner: Vec<Option<usize>> = vec![None; pf.len()];
        fn aug(j: usize, pf: &[usize], used: usize, tie: &dyn Fn(usize, usize) -> Option<bool>, owner: &mut Vec<Option<usize>>, seen: &mut Vec<bool>) -> bool {
            for b in 0..pf.len() {
                if b == used || seen[b] || tie(j, pf[b]).is_none() {
                    continue;
                }
                seen[b] = true;
                if owner[b].is_none_or(|o| aug(o, pf, used, tie, owner, seen)) {
                    owner[b] = Some(j);
                    return true;
                }
            }
            false
        }
        rest.iter().all(|&j| aug(j, pf, used, tie, &mut owner, &mut vec![false; pf.len()]))
    }
    for (a, &j) in pg.iter().enumerate() {
        for (b, &i) in pf.iter().enumerate() {
            if tie(j, i) == Some(true) {
                let rest: Vec<usize> = pg.iter().enumerate().filter(|(x, _)| *x != a).map(|(_, &y)| y).collect();
                if matching(&rest, pf, b, &tie) {
                    return Some(true);
                }
            }
        }
    }
    if matching(pg, pf, usize::MAX, &tie) { Some(false) } else { None }
}

/// Recursion (spec 6.7): a cycle of the call graph (one function calling
/// itself, or several) is admitted when its calls can be done in layers.
/// Each layer gives every function f a set P f of parameter positions; the
/// summed size at those positions must not grow along any call not yet
/// done (`site_decreases` is Some), and the calls where it shrinks
/// (Some(true)) are done; when the calls left form no cycle among the
/// functions, (μ₁, …, μ_k, rank) falls lexicographically at every call
/// (Lean `RCE/Mutual.lean`: `call_dec`, `call_keep`, `layer_dec`,
/// `wf_total`). Each layer takes, of all choices of sets tried in a fixed
/// order, the first that does the most calls; the search runs while the
/// cycle has at most MUTUAL_MAX_PARAMS parameters in total. Returns the
/// layers, or the first line of a call no layer does.
pub const MUTUAL_MAX_PARAMS: usize = 16;
pub fn lex_measure(fs: &[(String, usize, Vec<RecSite>)]) -> Result<Vec<Vec<Vec<usize>>>, usize> {
    let total: usize = fs.iter().map(|f| f.1).sum();
    let idx = |n: &str| fs.iter().position(|f| f.0 == n);
    let mut rest: Vec<(usize, usize, &RecSite)> = vec![];
    for (k, f) in fs.iter().enumerate() {
        for s in &f.2 {
            if let Some(g) = idx(&s.callee) {
                rest.push((k, g, s));
            }
        }
    }
    let first_line = |r: &[(usize, usize, &RecSite)]| r.iter().map(|x| x.2.line).min().unwrap_or(0);
    let mut layers = vec![];
    loop {
        let mut weak: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); fs.len()];
        for &(k, g, _) in &rest {
            weak[k].insert(g);
        }
        if acyclic(&weak) {
            return Ok(layers);
        }
        if total > MUTUAL_MAX_PARAMS {
            return Err(first_line(&rest));
        }
        let mut best: Option<(usize, Vec<Vec<usize>>, Vec<bool>)> = None;
        for mask in 1u64..(1u64 << total) {
            let mut sets = vec![];
            let mut off = 0;
            for f in fs {
                sets.push((0..f.1).filter(|i| mask >> (off + i) & 1 == 1).collect::<Vec<usize>>());
                off += f.1;
            }
            let mut strict = vec![];
            let ok = rest.iter().all(|&(k, g, s)| match site_decreases(s, &sets[g], &sets[k]) {
                Some(b) => {
                    strict.push(b);
                    true
                }
                None => false,
            });
            let n = strict.iter().filter(|b| **b).count();
            if ok && n > 0 && best.as_ref().is_none_or(|b| n > b.0) {
                best = Some((n, sets, strict));
            }
        }
        let Some((_, sets, strict)) = best else { return Err(first_line(&rest)) };
        let mut it = strict.iter();
        rest.retain(|_| !*it.next().unwrap());
        layers.push(sets);
    }
}

/// Strongly connected components of a call graph (Tarjan), each sorted,
/// in sorted order of their first member.
pub fn sccs(g: &HashMap<String, BTreeSet<String>>) -> Vec<Vec<String>> {
    struct T<'a> {
        g: &'a HashMap<String, BTreeSet<String>>,
        idx: HashMap<String, usize>,
        low: HashMap<String, usize>,
        on: BTreeSet<String>,
        st: Vec<String>,
        n: usize,
        out: Vec<Vec<String>>,
    }
    fn visit(t: &mut T, v: &str) {
        t.idx.insert(v.to_string(), t.n);
        t.low.insert(v.to_string(), t.n);
        t.n += 1;
        t.st.push(v.to_string());
        t.on.insert(v.to_string());
        let es: Vec<String> = t.g.get(v).map(|e| e.iter().cloned().collect()).unwrap_or_default();
        for w in es {
            if !t.idx.contains_key(&w) {
                visit(t, &w);
                let lw = t.low[&w];
                let lv = t.low.get_mut(v).unwrap();
                *lv = (*lv).min(lw);
            } else if t.on.contains(&w) {
                let iw = t.idx[&w];
                let lv = t.low.get_mut(v).unwrap();
                *lv = (*lv).min(iw);
            }
        }
        if t.low[v] == t.idx[v] {
            let mut c = vec![];
            loop {
                let w = t.st.pop().unwrap();
                t.on.remove(&w);
                let done = w == v;
                c.push(w);
                if done {
                    break;
                }
            }
            c.sort();
            t.out.push(c);
        }
    }
    let mut t = T { g, idx: HashMap::new(), low: HashMap::new(), on: BTreeSet::new(), st: vec![], n: 0, out: vec![] };
    let mut keys: Vec<&String> = g.keys().collect();
    keys.sort();
    for k in keys {
        if !t.idx.contains_key(k.as_str()) {
            visit(&mut t, k);
        }
    }
    t.out.sort();
    t.out
}

fn acyclic(g: &[BTreeSet<usize>]) -> bool {
    let n = g.len();
    let mut indeg = vec![0usize; n];
    for es in g {
        for &e in es {
            indeg[e] += 1;
        }
    }
    let mut q: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut seen = 0;
    while let Some(v) = q.pop() {
        seen += 1;
        for &e in &g[v] {
            indeg[e] -= 1;
            if indeg[e] == 0 {
                q.push(e);
            }
        }
    }
    seen == n
}

/// W-TRAIT-1: an impl method has the trait's signature, with `Self` read as
/// the implementing type: the same receiver, the same parameter types in
/// order (names may differ) and the same result type.
fn sig_matches(imp: &FnSig, tr: &FnSig, self_ty: &Ty) -> bool {
    fn subst(t: &Ty, s: &Ty) -> Ty {
        match t {
            Ty::Path(p, a) if p.len() == 1 && p[0] == "Self" && a.is_empty() => s.clone(),
            Ty::Path(p, a) => Ty::Path(p.clone(), a.iter().map(|x| subst(x, s)).collect()),
            Ty::Tuple(xs) => Ty::Tuple(xs.iter().map(|x| subst(x, s)).collect()),
            Ty::Slice(x) => Ty::Slice(Box::new(subst(x, s))),
            Ty::Chan(x) => Ty::Chan(Box::new(subst(x, s))),
            Ty::Iter(x) => Ty::Iter(Box::new(subst(x, s))),
            Ty::Opt(x) => Ty::Opt(Box::new(subst(x, s))),
            Ty::Ref(m, x) => Ty::Ref(*m, Box::new(subst(x, s))),
            Ty::Fn(xs, r) => Ty::Fn(xs.iter().map(|x| subst(x, s)).collect(), Box::new(subst(r, s))),
            Ty::Union(a, b) => Ty::Union(Box::new(subst(a, s)), Box::new(subst(b, s))),
            o => o.clone(),
        }
    }
    let ret = |r: &Option<Ty>| match r {
        None => Ty::Unit,
        Some(t) => subst(t, self_ty),
    };
    if imp.params.len() != tr.params.len() || ret(&imp.ret) != ret(&tr.ret) {
        return false;
    }
    imp.params.iter().zip(&tr.params).all(|(a, b)| match (a, b) {
        (Param::Named { ty: x, .. }, Param::Named { ty: y, .. }) => subst(x, self_ty) == subst(y, self_ty),
        (x, y) => x == y,
    })
}

/// A type written in expression position, as the index of `f[T]` (R5).
fn expr_as_ty(e: &Expr) -> Option<Ty> {
    match e {
        Expr::Name(n, _) => Some(Ty::Path(vec![n.clone()], vec![])),
        Expr::Member(b, m, _) => match expr_as_ty(b)? {
            Ty::Path(mut p, a) if a.is_empty() => {
                p.push(m.clone());
                Some(Ty::Path(p, vec![]))
            }
            _ => None,
        },
        Expr::Index(b, ix, _) => match expr_as_ty(b)? {
            Ty::Path(p, a) if a.is_empty() => Some(Ty::Path(p, ix.iter().map(expr_as_ty).collect::<Option<Vec<_>>>()?)),
            _ => None,
        },
        Expr::Tuple(xs, _) => Some(Ty::Tuple(xs.iter().map(expr_as_ty).collect::<Option<Vec<_>>>()?)),
        Expr::Unit(_) => Some(Ty::Unit),
        _ => None,
    }
}
