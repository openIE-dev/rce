//! The typed tree (one-emitter.md ONE-4; order of work, step 1).
//!
//! After the checker has accepted a module, this pass records for every
//! expression node of that module three facts, where the printer and the
//! lowering can read them:
//!
//! - its static type: the type of every value the reference interpreter
//!   (src/interp.rs) can give the node, written as an [`RType`];
//! - its binding, for a name: the local, the field of the state, the
//!   constant, the function or the constructor the name refers to;
//! - its resolved method, for a call: the user function, the method of an
//!   `impl`, the constructor or the built-in operation the call runs.
//!
//! The tables are keyed by the address of the node in the module that was
//! typed, so they describe exactly that module value, which the lowering
//! reads without copying it.
//!
//! The types are those of the reference interpreter's values, not only the
//! checker's inferred types: an integer value carries the name of its type
//! there, and an integer literal is untyped (`lit`) until a declared type is
//! given to it (6.3). So an integer type here is a set of those names
//! together with the range of values the node can take. That set decides
//! the range an arithmetic result is checked against: `x + 1` with `x: u8`
//! is checked against u8, while `a + b` with two untyped values is checked
//! against i64. A variable's type follows the program: it is computed at
//! each point, and around a loop until it no longer changes.
use crate::ast::*;
use crate::types::{INTS, int_range};
use std::collections::{BTreeMap, HashMap};

/// The names an integer value can carry: `lit` (an untyped literal, checked
/// as i64) and the nine integer types.
pub const TAG_NAMES: [&str; 10] = ["lit", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "usize"];

/// A set of integer type names, as a bit set over [`TAG_NAMES`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Tags(pub u16);

impl Tags {
    pub fn of(name: &str) -> Tags {
        Tags(1 << TAG_NAMES.iter().position(|t| *t == name).unwrap_or(0))
    }
    pub fn lit() -> Tags {
        Tags(1)
    }
    pub fn names(self) -> Vec<&'static str> {
        (0..10).filter(|k| self.0 & (1 << k) != 0).map(|k| TAG_NAMES[k]).collect()
    }
    pub fn single(self) -> Option<&'static str> {
        let n = self.names();
        if n.len() == 1 { Some(n[0]) } else { None }
    }
    pub fn has_lit(self) -> bool {
        self.0 & 1 != 0
    }
    pub fn union(self, o: Tags) -> Tags {
        Tags(self.0 | o.0)
    }
    /// The names a binary operation gives its result (interp `join_ty`):
    /// the left operand's name, unless it is `lit`.
    pub fn join_op(self, o: Tags) -> Tags {
        let mut r = Tags(0);
        for a in self.names() {
            for b in o.names() {
                r = r.union(Tags::of(if a == "lit" { b } else { a }));
            }
        }
        r
    }
}

/// The range a value with this name is checked against: `lit` is checked as i64.
pub fn tag_range(name: &str) -> (i128, i128) {
    int_range(if name == "lit" { "i64" } else { name })
}

/// The width in bits the interpreter gives a shift with this name.
pub fn tag_bits(name: &str) -> u32 {
    match name {
        "i8" | "u8" => 8,
        "i16" | "u16" => 16,
        "i32" | "u32" => 32,
        _ => 64,
    }
}

/// An integer type: the names its values can carry and the range they lie in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IntT {
    pub tags: Tags,
    pub lo: i128,
    pub hi: i128,
}

impl IntT {
    /// A value checked against its name's range.
    pub fn of_tags(tags: Tags) -> IntT {
        let (mut lo, mut hi) = (i128::MAX, i128::MIN);
        for n in tags.names() {
            let (a, b) = tag_range(n);
            lo = lo.min(a);
            hi = hi.max(b);
        }
        IntT { tags, lo, hi }
    }
    pub fn named(n: &str) -> IntT {
        IntT::of_tags(Tags::of(n))
    }
    pub fn join(self, o: IntT) -> IntT {
        IntT { tags: self.tags.union(o.tags), lo: self.lo.min(o.lo), hi: self.hi.max(o.hi) }
    }
    /// The range every name of the set checks against, when they agree.
    pub fn check_range(self) -> Option<(i128, i128)> {
        let rs: Vec<(i128, i128)> = self.tags.names().into_iter().map(tag_range).collect();
        if rs.windows(2).all(|w| w[0] == w[1]) { rs.first().copied() } else { None }
    }
}

/// The type of the values a node can have.
#[derive(Clone, PartialEq, Debug)]
pub enum RType {
    /// no value: the node is not reached, or it is an empty literal's element
    Never,
    Bool,
    Int(IntT),
    F64,
    F32,
    Text,
    Unit,
    Tuple(Vec<RType>),
    /// lists, arrays and slices
    List(Box<RType>),
    Set(Box<RType>),
    Map(Box<RType>, Box<RType>),
    Opt(Box<RType>),
    /// a value of a declared struct, event, error or kind whose fields have
    /// their declared types
    Named(String),
    /// `τ ! ε`: the ok value, or an error value
    Union(Box<RType>, Box<RType>),
    /// `a..b`
    Range(IntT),
    /// a named function as a value
    Fn(String),
    /// a value this pass does not type, and why
    Unknown(String),
}

impl RType {
    pub fn is_known(&self) -> bool {
        match self {
            RType::Unknown(_) => false,
            RType::Tuple(ts) => ts.iter().all(|t| t.is_known()),
            RType::List(t) | RType::Set(t) | RType::Opt(t) => t.is_known(),
            RType::Map(a, b) | RType::Union(a, b) => a.is_known() && b.is_known(),
            _ => true,
        }
    }
    pub fn unknown_reason(&self) -> Option<String> {
        match self {
            RType::Unknown(r) => Some(r.clone()),
            RType::Tuple(ts) => ts.iter().find_map(|t| t.unknown_reason()),
            RType::List(t) | RType::Set(t) | RType::Opt(t) => t.unknown_reason(),
            RType::Map(a, b) | RType::Union(a, b) => a.unknown_reason().or_else(|| b.unknown_reason()),
            _ => None,
        }
    }
    /// The least type holding the values of both.
    pub fn join(&self, o: &RType) -> RType {
        use RType::*;
        match (self, o) {
            (Never, x) | (x, Never) => x.clone(),
            (Unknown(r), _) | (_, Unknown(r)) => Unknown(r.clone()),
            (Int(a), Int(b)) => Int(a.join(*b)),
            (Range(a), Range(b)) => Range(a.join(*b)),
            (Tuple(a), Tuple(b)) if a.len() == b.len() => Tuple(a.iter().zip(b).map(|(x, y)| x.join(y)).collect()),
            (List(a), List(b)) => List(Box::new(a.join(b))),
            (Set(a), Set(b)) => Set(Box::new(a.join(b))),
            (Map(a, b), Map(c, d)) => Map(Box::new(a.join(c)), Box::new(b.join(d))),
            (Opt(a), Opt(b)) => Opt(Box::new(a.join(b))),
            (Union(a, b), Union(c, d)) => Union(Box::new(a.join(c)), Box::new(b.join(d))),
            (a, b) if a == b => a.clone(),
            (a, b) => Unknown(format!("a value that is {} or {}", a.show(), b.show())),
        }
    }
    pub fn show(&self) -> String {
        use RType::*;
        match self {
            Never => "never".into(),
            Bool => "bool".into(),
            Int(i) => format!("int{{{}}}[{}, {}]", i.tags.names().join(","), i.lo, i.hi),
            F64 => "f64".into(),
            F32 => "f32".into(),
            Text => "text".into(),
            Unit => "()".into(),
            Tuple(ts) => format!("({})", ts.iter().map(|t| t.show()).collect::<Vec<_>>().join(", ")),
            List(t) => format!("List[{}]", t.show()),
            Set(t) => format!("Set[{}]", t.show()),
            Map(a, b) => format!("Map[{}, {}]", a.show(), b.show()),
            Opt(t) => format!("{}?", t.show()),
            Named(n) => n.clone(),
            Union(a, b) => format!("{} ! {}", a.show(), b.show()),
            Range(i) => format!("Range[{}]", RType::Int(*i).show()),
            Fn(n) => format!("fn {n}"),
            Unknown(r) => format!("unknown ({r})"),
        }
    }
}

/// What a name refers to.
#[derive(Clone, PartialEq, Debug)]
pub enum Binding {
    /// a local: a parameter, `let`, `var`, pattern, `for` or `catch` binding,
    /// a state field inside a step, or `self`
    Local(u32),
    Const(String),
    Fn(String),
    /// a constructor without fields: a variant, an event or an error
    Ctor(String),
}

/// What a call runs.
#[derive(Clone, PartialEq, Debug)]
pub enum Callee {
    /// a function of the module
    Fn(String),
    /// a method of an `impl` on the receiver's type: (type head, name)
    Method(String, String),
    /// `T.m(…)`: a function of an `impl` called through its type
    Static(String, String),
    /// a struct, event, error or variant constructor
    Ctor(String),
    /// `some(x)`
    Some,
    /// `copy(x)` and `Box(x)`: the value itself
    Same,
    /// a built-in method on the receiver's type (5.15)
    Builtin(String),
    /// a call this pass does not resolve, and why
    Unknown(String),
}

/// The facts recorded for one expression node.
#[derive(Clone, Debug)]
pub struct Node {
    pub ty: RType,
    pub bind: Option<Binding>,
    pub callee: Option<Callee>,
}

/// One local of the module.
#[derive(Clone, Debug)]
pub struct LocalInfo {
    pub name: String,
    pub mutable: bool,
    /// the type of every value the local holds anywhere in its scope
    pub ty: RType,
}

/// The typed tree of one module.
#[derive(Default, Debug)]
pub struct Typed {
    pub nodes: HashMap<usize, Node>,
    pub locals: Vec<LocalInfo>,
    /// the local a binding pattern (`Pat::Path`) introduces
    pub pat_binds: HashMap<usize, u32>,
    /// the local a `var` statement introduces
    pub var_binds: HashMap<usize, u32>,
    /// the local the binder of `e catch |x| h` introduces
    pub catch_binds: HashMap<usize, u32>,
    /// the locals of a function's named parameters, and of its `self`
    pub params: HashMap<usize, (Vec<u32>, Option<u32>)>,
    /// the locals of the state fields inside a step, in declaration order
    pub step_fields: HashMap<usize, Vec<u32>>,
    /// the type a statement leaves in its place (the type of the value an
    /// assignment stores, the receiver of a mutating method)
    pub stores: HashMap<usize, RType>,
}

pub fn addr<T>(x: &T) -> usize {
    x as *const T as usize
}

impl Typed {
    pub fn node(&self, e: &Expr) -> Option<&Node> {
        self.nodes.get(&addr(e))
    }
    pub fn ty(&self, e: &Expr) -> RType {
        self.node(e).map(|n| n.ty.clone()).unwrap_or(RType::Unknown("an expression the typed tree did not reach".into()))
    }
}

/// The declarations of a module, as the interpreter resolves them.
pub struct Tables<'a> {
    pub fns: HashMap<String, &'a FnDecl>,
    pub methods: HashMap<(String, String), &'a FnDecl>,
    /// struct, state, event, error, actor and alias items by name
    pub types: HashMap<String, &'a Item>,
    /// variant name -> (kind name, index, fields)
    pub variants: HashMap<String, (String, usize, Vec<(String, Ty)>)>,
    pub kinds: HashMap<String, Vec<(String, Vec<(String, Ty)>)>>,
    pub consts: HashMap<String, (Option<Ty>, &'a Expr)>,
    /// the declared events, in declaration order
    pub events: Vec<String>,
}

impl<'a> Tables<'a> {
    pub fn new(m: &'a Module) -> Self {
        let mut t = Tables {
            fns: HashMap::new(),
            methods: HashMap::new(),
            types: HashMap::new(),
            variants: HashMap::new(),
            kinds: HashMap::new(),
            consts: HashMap::new(),
            events: vec![],
        };
        for i in &m.items {
            match i {
                Item::Fn(f) => {
                    t.fns.insert(f.sig.name.clone(), f);
                }
                Item::Impl { ty, members, .. } => {
                    let h = crate::check::ty_head(ty);
                    for f in members {
                        t.methods.insert((h.clone(), f.sig.name.clone()), f);
                    }
                }
                Item::Struct { name, .. } | Item::State { name, .. } | Item::Error { name, .. } | Item::Actor { name, .. } | Item::TypeAlias { name, .. } => {
                    t.types.insert(name.clone(), i);
                }
                Item::Event { name, .. } => {
                    t.types.insert(name.clone(), i);
                    t.events.push(name.clone());
                }
                Item::Kind { name, variants, .. } => {
                    t.types.insert(name.clone(), i);
                    t.kinds.insert(name.clone(), variants.clone());
                    for (k, (v, p)) in variants.iter().enumerate() {
                        t.variants.insert(v.clone(), (name.clone(), k, p.clone()));
                    }
                }
                Item::Const { name, ty, e, .. } => {
                    t.consts.insert(name.clone(), (ty.clone(), e));
                }
                _ => {}
            }
        }
        t
    }
    /// The declared fields of a struct, event or error, in declaration order.
    pub fn fields(&self, n: &str) -> Option<Vec<(String, Ty)>> {
        match self.types.get(n)? {
            Item::Struct { fields, .. } => Some(fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect()),
            Item::Event { payload, .. } | Item::Error { payload, .. } => Some(payload.clone()),
            _ => None,
        }
    }
    /// The fields of a constructor: a struct, event, error or variant.
    pub fn ctor_fields(&self, n: &str) -> Option<Vec<(String, Ty)>> {
        if let Some((_, _, p)) = self.variants.get(n) {
            return Some(p.clone());
        }
        self.fields(n)
    }
    /// The type a constructor builds: the kind of a variant, else the name itself.
    pub fn ctor_type(&self, n: &str) -> String {
        match self.variants.get(n) {
            Some((k, _, _)) => k.clone(),
            None => n.to_string(),
        }
    }
    /// A name a pattern tests rather than binds (interp `pmatch`).
    pub fn is_tested_name(&self, n: &str) -> bool {
        self.variants.contains_key(n) || matches!(self.types.get(n), Some(Item::Event { .. } | Item::Error { .. }))
    }
    pub fn is_ctor(&self, n: &str) -> bool {
        self.variants.contains_key(n) || matches!(self.types.get(n), Some(Item::Struct { .. } | Item::Event { .. } | Item::Error { .. }))
    }
    /// The declared type `t`, with aliases and erased wrappers resolved.
    pub fn alias(&self, t: &Ty) -> Ty {
        if let Ty::Path(p, a) = t
            && p.len() == 1
            && !self.types.contains_key(&p[0])
        {
            match (p[0].as_str(), a.as_slice()) {
                ("Guarded", [x, _]) | ("Secret", [x]) => return self.alias(x),
                ("Text", [_]) => return Ty::Path(vec!["text".into()], vec![]),
                ("Cap", [_]) => return Ty::Unit,
                _ => {}
            }
        }
        if let Ty::Path(p, _) = t
            && p.len() == 1
            && let Some(Item::TypeAlias { ty, .. }) = self.types.get(&p[0])
        {
            return self.alias(ty);
        }
        t.clone()
    }
    /// Whether `t` names a refined type (5.1), whose fit also tests a predicate.
    pub fn refined(&self, t: &Ty) -> bool {
        matches!(t, Ty::Path(p, _) if p.len() == 1 && matches!(self.types.get(&p[0]), Some(Item::TypeAlias { pred: Some(_), .. })))
    }
    /// The type of a value after the interpreter fits it to the declared type `t`.
    pub fn of_ty(&self, t: &Ty) -> RType {
        if self.refined(t) {
            return RType::Unknown("a refined type".into());
        }
        let t = self.alias(t);
        match &t {
            Ty::Path(p, a) if p.len() == 1 => {
                let n = p[0].as_str();
                if a.is_empty() && INTS.contains(&n) {
                    return RType::Int(IntT::named(n));
                }
                match (n, a.as_slice()) {
                    ("bool", []) => RType::Bool,
                    ("f64", []) => RType::F64,
                    ("f32", []) => RType::F32,
                    ("text", []) => RType::Text,
                    ("List", [x]) => RType::List(Box::new(self.of_ty(x))),
                    ("Set", [x]) => RType::Set(Box::new(self.of_ty(x))),
                    ("Map", [k, v]) => RType::Map(Box::new(self.of_ty(k)), Box::new(self.of_ty(v))),
                    ("Box", [x]) => self.of_ty(x),
                    _ if a.is_empty() && (self.kinds.contains_key(n) || self.fields(n).is_some()) => RType::Named(n.to_string()),
                    _ => RType::Unknown(format!("the type {}", crate::interp::ty_str(&t))),
                }
            }
            Ty::Unit => RType::Unit,
            Ty::Tuple(ts) => RType::Tuple(ts.iter().map(|x| self.of_ty(x)).collect()),
            Ty::Slice(x) | Ty::Array(x, _) => RType::List(Box::new(self.of_ty(x))),
            Ty::Ref(_, x) | Ty::Region(x, _) => self.of_ty(x),
            Ty::Opt(x) => RType::Opt(Box::new(self.of_ty(x))),
            Ty::Union(a, b) => RType::Union(Box::new(self.of_ty(a)), Box::new(self.of_ty(b))),
            o => RType::Unknown(format!("the type {}", crate::interp::ty_str(o))),
        }
    }
    /// The type a value of type `v` has after it is fitted to `t` (interp `fit`).
    pub fn fit(&self, v: &RType, t: &Ty) -> RType {
        let want = self.of_ty(t);
        match (v, &want) {
            (RType::Unknown(r), _) => RType::Unknown(r.clone()),
            // an error value keeps its type through a fit to the union
            _ => want,
        }
    }
    /// The type of field `f` of a value of declared type `n`.
    pub fn field_type(&self, n: &str, f: &str) -> RType {
        if let Some(fs) = self.fields(n) {
            return match fs.iter().find(|(x, _)| x == f) {
                Some((_, t)) => self.of_ty(t),
                None => RType::Unknown(format!("no field {f} in {n}")),
            };
        }
        if let Some(vs) = self.kinds.get(n) {
            let mut r = RType::Never;
            for (_, p) in vs {
                if let Some((_, t)) = p.iter().find(|(x, _)| x == f) {
                    r = r.join(&self.of_ty(t));
                }
            }
            return if r == RType::Never { RType::Unknown(format!("no field {f} in {n}")) } else { r };
        }
        RType::Unknown(format!("a field of {n}"))
    }
}

type Env = HashMap<u32, RType>;

fn env_join(a: &Option<Env>, b: &Option<Env>) -> Option<Env> {
    match (a, b) {
        (None, x) | (x, None) => x.clone(),
        (Some(a), Some(b)) => {
            let mut r = a.clone();
            for (k, v) in b {
                let j = match r.get(k) {
                    Some(x) => x.join(v),
                    None => v.clone(),
                };
                r.insert(*k, j);
            }
            Some(r)
        }
    }
}

/// The re-typing of an assignment (interp `exec` of `=`): a place holding an
/// integer with a declared name keeps that name, and the stored value is
/// checked against it; a place holding an untyped literal takes the stored
/// value's name.
pub fn retag(cur: &RType, v: &RType) -> RType {
    match (cur, v) {
        (RType::Int(c), RType::Int(x)) => {
            let mut tags = Tags(0);
            let (mut lo, mut hi) = (i128::MAX, i128::MIN);
            for n in c.tags.names() {
                if n == "lit" {
                    tags = tags.union(x.tags);
                    lo = lo.min(x.lo);
                    hi = hi.max(x.hi);
                } else {
                    tags = tags.union(Tags::of(n));
                    let (a, b) = tag_range(n);
                    lo = lo.min(a.max(x.lo));
                    hi = hi.max(b.min(x.hi));
                }
            }
            if lo > hi {
                // every store traps; keep the declared range
                return RType::Int(IntT::of_tags(tags));
            }
            RType::Int(IntT { tags, lo, hi })
        }
        _ => v.clone(),
    }
}

struct LoopAcc {
    breaks: Option<Env>,
    conts: Option<Env>,
}

struct Cx<'a, 'm> {
    t: &'a Tables<'m>,
    out: &'a mut Typed,
    scopes: Vec<HashMap<String, u32>>,
    env: Option<Env>,
    loops: Vec<LoopAcc>,
    consts: &'a mut HashMap<String, Option<RType>>,
}

impl<'a, 'm> Cx<'a, 'm> {
    fn new_local(&mut self, name: &str, mutable: bool, ty: RType) -> u32 {
        let id = self.out.locals.len() as u32;
        self.out.locals.push(LocalInfo { name: name.to_string(), mutable, ty: RType::Never });
        self.scopes.last_mut().unwrap().insert(name.to_string(), id);
        self.store(id, ty);
        id
    }
    /// The local a binding node introduces: the same local each time the
    /// node is typed again (around a loop).
    fn node_local(&mut self, key: usize, table: fn(&mut Typed) -> &mut HashMap<usize, u32>, name: &str, mutable: bool, ty: RType) -> u32 {
        if let Some(id) = table(self.out).get(&key).copied() {
            self.scopes.last_mut().unwrap().insert(name.to_string(), id);
            self.store(id, ty);
            return id;
        }
        let id = self.new_local(name, mutable, ty);
        table(self.out).insert(key, id);
        id
    }
    fn store(&mut self, id: u32, ty: RType) {
        let l = &mut self.out.locals[id as usize];
        l.ty = l.ty.join(&ty);
        if let Some(env) = self.env.as_mut() {
            env.insert(id, ty);
        }
    }
    fn lookup(&self, n: &str) -> Option<u32> {
        self.scopes.iter().rev().find_map(|s| s.get(n).copied())
    }
    fn cur(&self, id: u32) -> RType {
        match &self.env {
            Some(e) => e.get(&id).cloned().unwrap_or(RType::Never),
            None => RType::Never,
        }
    }
    fn record(&mut self, e: &Expr, ty: &RType, bind: Option<Binding>, callee: Option<Callee>) {
        let n = self.out.nodes.entry(addr(e)).or_insert(Node { ty: RType::Never, bind: None, callee: None });
        n.ty = n.ty.join(ty);
        if bind.is_some() {
            n.bind = bind;
        }
        if callee.is_some() {
            n.callee = callee;
        }
    }

    fn const_type(&mut self, n: &str) -> RType {
        if let Some(t) = self.consts.get(n) {
            return t.clone().unwrap_or(RType::Unknown(format!("the constant {n} refers to itself")));
        }
        let Some((ty, e)) = self.t.consts.get(n).cloned() else {
            return RType::Unknown(format!("unknown name {n}"));
        };
        self.consts.insert(n.to_string(), None);
        let saved_scopes = std::mem::replace(&mut self.scopes, vec![HashMap::new()]);
        let saved_env = std::mem::replace(&mut self.env, Some(HashMap::new()));
        let saved_loops = std::mem::take(&mut self.loops);
        let v = self.expr(e);
        self.scopes = saved_scopes;
        self.env = saved_env;
        self.loops = saved_loops;
        let v = match &ty {
            Some(t) => self.t.fit(&v, t),
            None => v,
        };
        self.consts.insert(n.to_string(), Some(v.clone()));
        v
    }

    // ---- expressions

    fn expr(&mut self, e: &Expr) -> RType {
        let (ty, bind, callee) = self.expr_inner(e);
        self.record(e, &ty, bind, callee);
        ty
    }

    fn expr_inner(&mut self, e: &Expr) -> (RType, Option<Binding>, Option<Callee>) {
        use Expr as E;
        let plain = |t: RType| (t, None, None);
        match e {
            E::Name(n, _) => {
                if let Some(id) = self.lookup(n) {
                    return (self.cur(id), Some(Binding::Local(id)), None);
                }
                if self.t.fns.contains_key(n) {
                    return (RType::Fn(n.clone()), Some(Binding::Fn(n.clone())), None);
                }
                if let Some((k, _, p)) = self.t.variants.get(n)
                    && p.is_empty()
                {
                    return (RType::Named(k.clone()), Some(Binding::Ctor(n.clone())), None);
                }
                if let Some(Item::Event { .. } | Item::Error { .. }) = self.t.types.get(n) {
                    return (RType::Named(n.clone()), Some(Binding::Ctor(n.clone())), None);
                }
                if self.t.consts.contains_key(n) {
                    let t = self.const_type(n);
                    return (t, Some(Binding::Const(n.clone())), None);
                }
                plain(RType::Unknown(format!("the name {n}")))
            }
            E::SelfRef(_) => match self.lookup("self") {
                Some(id) => (self.cur(id), Some(Binding::Local(id)), None),
                None => plain(RType::Unknown("self outside a method".into())),
            },
            E::Int(v, _) => plain(RType::Int(IntT { tags: Tags::lit(), lo: *v, hi: *v })),
            E::Float(..) => plain(RType::F64),
            E::Str(..) => plain(RType::Text),
            E::Bool(..) => plain(RType::Bool),
            E::None(_) => plain(RType::Opt(Box::new(RType::Never))),
            E::Unit(_) => plain(RType::Unit),
            E::Tuple(xs, _) => plain(RType::Tuple(xs.iter().map(|x| self.expr(x)).collect())),
            E::List(xs, _) => {
                let mut t = RType::Never;
                for x in xs {
                    let xt = self.expr(x);
                    t = t.join(&xt);
                }
                plain(RType::List(Box::new(t)))
            }
            E::ListRepeat(x, n, _) => {
                let t = self.expr(x);
                self.expr(n);
                plain(RType::List(Box::new(t)))
            }
            E::ListComp(..) => plain(RType::Unknown("a list comprehension".into())),
            E::Map(ps, _) => {
                let (mut k, mut v) = (RType::Never, RType::Never);
                for (a, b) in ps {
                    let at = self.expr(a);
                    let bt = self.expr(b);
                    k = k.join(&at);
                    v = v.join(&bt);
                }
                plain(RType::Map(Box::new(k), Box::new(v)))
            }
            E::ChanNew(..) => plain(RType::Unknown("a channel".into())),
            E::Call(f, args, _, _) => self.call(e, f, args),
            E::Index(x, idx, _) => {
                let xt = self.expr(x);
                for i in idx {
                    self.expr(i);
                }
                plain(match xt {
                    RType::List(t) => *t,
                    RType::Text => RType::Int(IntT::named("u8")),
                    RType::Map(_, v) => *v,
                    RType::Unknown(r) => RType::Unknown(r),
                    o => RType::Unknown(format!("an index into {}", o.show())),
                })
            }
            E::Slice(x, a, b, _) => {
                let xt = self.expr(x);
                if let Some(a) = a {
                    self.expr(a);
                }
                if let Some(b) = b {
                    self.expr(b);
                }
                plain(match xt {
                    RType::List(t) => RType::List(t),
                    RType::Text => RType::Text,
                    o => RType::Unknown(format!("a slice of {}", o.show())),
                })
            }
            E::Member(x, f, _) => {
                let xt = self.expr(x);
                plain(match xt {
                    RType::Named(n) => self.t.field_type(&n, f),
                    RType::Unknown(r) => RType::Unknown(r),
                    o => RType::Unknown(format!("the field {f} of {}", o.show())),
                })
            }
            E::TupleMember(x, i, _) => {
                let xt = self.expr(x);
                plain(match xt {
                    RType::Tuple(ts) if *i < ts.len() => ts[*i].clone(),
                    o => RType::Unknown(format!("component {i} of {}", o.show())),
                })
            }
            E::Neg(x, _) => {
                let xt = self.expr(x);
                plain(match xt {
                    RType::Int(i) => RType::Int(IntT::of_tags(i.tags)),
                    RType::F64 => RType::F64,
                    RType::F32 => RType::F32,
                    o => RType::Unknown(format!("the negation of {}", o.show())),
                })
            }
            E::BitNot(x, _) => {
                let xt = self.expr(x);
                plain(match xt {
                    RType::Int(i) => {
                        let (mut lo, mut hi) = (i128::MAX, i128::MIN);
                        for n in i.tags.names() {
                            let (tl, th) = tag_range(n);
                            let (a, b) = if tl == 0 { (th - i.hi, th - i.lo) } else { (-1 - i.hi, -1 - i.lo) };
                            lo = lo.min(a);
                            hi = hi.max(b);
                        }
                        RType::Int(IntT { tags: i.tags, lo, hi })
                    }
                    o => RType::Unknown(format!("~ of {}", o.show())),
                })
            }
            E::Not(x, _) => {
                self.expr(x);
                plain(RType::Bool)
            }
            E::Borrow(_, x, _) | E::Comptime(x, _) => {
                let t = self.expr(x);
                plain(t)
            }
            E::Recv(..) => plain(RType::Unknown("a channel receive".into())),
            E::Cast(x, ty, _) => {
                let xt = self.expr(x);
                let target = self.t.alias(ty);
                let to = self.t.of_ty(&target);
                plain(match (&xt, &to) {
                    (RType::Int(_) | RType::F64 | RType::F32 | RType::Bool, RType::Int(_)) => to.clone(),
                    (RType::Int(_) | RType::F64 | RType::F32, RType::F64 | RType::F32) => to.clone(),
                    (RType::Unknown(r), _) => RType::Unknown(r.clone()),
                    (a, b) => RType::Unknown(format!("a cast of {} to {}", a.show(), b.show())),
                })
            }
            E::Bin(op, a, b, _) => {
                let at = self.expr(a);
                let bt = self.expr(b);
                plain(bin_type(op, &at, &bt))
            }
            E::Catch(x, bind, h, _) => {
                let xt = self.expr(x);
                let (ok, err) = match &xt {
                    RType::Union(o, e) => ((**o).clone(), (**e).clone()),
                    o => (o.clone(), RType::Never),
                };
                let before = self.env.clone();
                self.scopes.push(HashMap::new());
                if let Some(n) = bind {
                    self.node_local(addr(e), |t| &mut t.catch_binds, n, false, err.clone());
                }
                let ht = self.expr(h);
                self.scopes.pop();
                self.env = env_join(&before, &self.env);
                plain(ok.join(&ht))
            }
            E::Try(x, els, _) => {
                let xt = self.expr(x);
                if let Some(e) = els {
                    let before = self.env.clone();
                    self.expr(e);
                    self.env = env_join(&before, &self.env);
                }
                plain(match xt {
                    RType::Union(ok, _) => match (*ok, els) {
                        (RType::Opt(t), Some(_)) => *t,
                        (o, _) => o,
                    },
                    RType::Opt(t) => {
                        if els.is_some() {
                            *t
                        } else {
                            RType::Opt(t)
                        }
                    }
                    o => o,
                })
            }
            E::If(c, a, b, _) => {
                self.expr(c);
                let before = self.env.clone();
                let at = self.expr(a);
                let after_a = std::mem::replace(&mut self.env, before);
                let bt = self.expr(b);
                self.env = env_join(&after_a, &self.env);
                plain(at.join(&bt))
            }
            E::Lambda(..) => plain(RType::Unknown("a lambda".into())),
        }
    }

    /// The arguments of a call; each is fitted to its parameter by the callee.
    fn args_fit(&mut self, args: &[Arg], _params: &[(String, Ty)]) {
        for a in args {
            self.expr(&a.e);
        }
    }

    fn fn_result(&self, f: &FnDecl) -> RType {
        if !f.sig.generics.is_empty() {
            return RType::Unknown(format!("a call of the generic function {}", f.sig.name));
        }
        match &f.sig.ret {
            Some(Ty::Iter(_)) => RType::Unknown("an iterator call".into()),
            Some(t) => self.t.of_ty(t),
            None => RType::Unit,
        }
    }

    fn named_params(f: &FnDecl) -> Vec<(String, Ty)> {
        f.sig
            .params
            .iter()
            .filter_map(|p| if let Param::Named { name, ty, .. } = p { Some((name.clone(), ty.clone())) } else { None })
            .collect()
    }

    fn call(&mut self, e: &Expr, f: &Expr, args: &[Arg]) -> (RType, Option<Binding>, Option<Callee>) {
        let _ = e;
        // the callee of a direct call is not evaluated; its node records what it names
        if let Expr::Name(n, _) = f
            && self.lookup(n).is_none()
        {
            let b = if self.t.fns.contains_key(n) {
                Some(Binding::Fn(n.clone()))
            } else if self.t.is_ctor(n) {
                Some(Binding::Ctor(n.clone()))
            } else {
                None
            };
            self.record(f, &RType::Fn(n.clone()), b, None);
        }
        if let Expr::Member(_, m, _) = f {
            self.record(f, &RType::Fn(m.clone()), None, None);
        }
        // `f[T](x)`: a generic call; types are erased at run time
        if let Expr::Index(g, _, _) = f
            && let Expr::Name(n, _) = &**g
            && self.t.fns.contains_key(n)
            && self.lookup(n).is_none()
        {
            for a in args {
                self.expr(&a.e);
            }
            return (RType::Unknown(format!("the generic call {n}[…]")), None, Some(Callee::Unknown("a generic call".into())));
        }
        match f {
            Expr::Name(n, _) if self.lookup(n).is_none() => {
                match n.as_str() {
                    "some" => {
                        let t = args.first().map(|a| self.expr(&a.e)).unwrap_or(RType::Never);
                        return (RType::Opt(Box::new(t)), None, Some(Callee::Some));
                    }
                    "copy" | "Box" => {
                        let t = args.first().map(|a| self.expr(&a.e)).unwrap_or(RType::Never);
                        return (t, None, Some(Callee::Same));
                    }
                    "ct_choose" | "set_of" | "lvar_new" => {
                        for a in args {
                            self.expr(&a.e);
                        }
                        return (RType::Unknown(format!("{n}")), None, Some(Callee::Unknown(n.clone())));
                    }
                    _ => {}
                }
                if let Some(fd) = self.t.fns.get(n).copied() {
                    let ps = Self::named_params(fd);
                    self.args_fit(args, &ps);
                    return (self.fn_result(fd), None, Some(Callee::Fn(n.clone())));
                }
                if let Some(fs) = self.t.ctor_fields(n) {
                    self.args_fit(args, &fs);
                    return (RType::Named(self.t.ctor_type(n)), None, Some(Callee::Ctor(n.clone())));
                }
                for a in args {
                    self.expr(&a.e);
                }
                (RType::Unknown(format!("a call of {n}")), None, Some(Callee::Unknown(format!("a call of {n}"))))
            }
            Expr::Member(obj, m, _) => {
                if let Expr::Name(tn, _) = &**obj
                    && self.lookup(tn).is_none()
                    && self.t.types.contains_key(tn)
                {
                    return match self.t.methods.get(&(tn.clone(), m.clone())).copied() {
                        Some(fd) => {
                            let ps = Self::named_params(fd);
                            self.args_fit(args, &ps);
                            (self.fn_result(fd), None, Some(Callee::Static(tn.clone(), m.clone())))
                        }
                        None => (RType::Unknown("no such function".into()), None, Some(Callee::Unknown("no such function".into()))),
                    };
                }
                let rt = self.expr(obj);
                let head = match &rt {
                    RType::Named(n) => Some(n.clone()),
                    RType::List(_) => Some("List".to_string()),
                    RType::Set(_) => Some("Set".to_string()),
                    _ => None,
                };
                if let Some(h) = head
                    && let Some(fd) = self.t.methods.get(&(h.clone(), m.clone())).copied()
                {
                    let ps = Self::named_params(fd);
                    self.args_fit(args, &ps);
                    let r = self.fn_result(fd);
                    if matches!(fd.sig.params.first(), Some(Param::SelfRef(true))) {
                        // a `&mut self` method writes its receiver back: the
                        // receiver keeps its declared type
                        self.place_store(obj, &rt);
                    }
                    return (r, None, Some(Callee::Method(h, m.clone())));
                }
                let at: Vec<RType> = args.iter().map(|a| self.expr(&a.e)).collect();
                let r = self.builtin(obj, &rt, m, &at);
                (r, None, Some(Callee::Builtin(m.clone())))
            }
            _ => {
                let ft = self.expr(f);
                for a in args {
                    self.expr(&a.e);
                }
                match ft {
                    RType::Fn(n) => match self.t.fns.get(&n).copied() {
                        Some(fd) => (self.fn_result(fd), None, Some(Callee::Fn(n))),
                        None => (RType::Unknown("a call of a value".into()), None, Some(Callee::Unknown("a call of a value".into()))),
                    },
                    _ => (RType::Unknown("a call of a value".into()), None, Some(Callee::Unknown("a call of a value".into()))),
                }
            }
        }
    }

    /// The result of a built-in method (interp `builtin`), and the new type
    /// of a receiver the method writes back.
    fn builtin(&mut self, obj: &Expr, rt: &RType, m: &str, a: &[RType]) -> RType {
        use RType as T;
        let arg = |k: usize| a.get(k).cloned().unwrap_or(T::Never);
        let usize_t = || T::Int(IntT::named("usize"));
        match (rt, m) {
            (T::Unknown(r), _) => T::Unknown(r.clone()),
            (T::Int(i), "checked_add" | "checked_sub" | "checked_mul" | "checked_div") => match arg(0) {
                T::Int(j) => T::Opt(Box::new(T::Int(IntT::of_tags(i.tags.join_op(j.tags))))),
                o => T::Unknown(format!("{m} of {}", o.show())),
            },
            (T::Int(i), "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub" | "saturating_mul") => {
                T::Int(IntT::of_tags(i.tags))
            }
            (T::Int(_), "tostr") | (T::F64, "tostr") | (T::Bool, "tostr") | (T::F32, "tostr") => T::Text,
            (T::Int(i), "abs") => T::Int(IntT::of_tags(i.tags)),
            (T::Int(i), "min" | "max") => match arg(0) {
                T::Int(j) => T::Int(IntT { tags: i.tags, lo: i.lo.min(j.lo), hi: i.hi.max(j.hi) }),
                o => T::Unknown(format!("{m} of {}", o.show())),
            },
            (T::F64, "sqrt" | "abs" | "min" | "max" | "floor" | "ceil" | "round") => T::F64,
            (T::F32, "abs") => T::F32,
            (T::Text, "len" | "len_bytes" | "len_chars") => usize_t(),
            (T::Text, "strip" | "upper" | "lower" | "reverse" | "tostr" | "raw" | "sha256" | "join") => T::Text,
            (T::Text, "chars" | "words" | "lines" | "split") => T::List(Box::new(T::Text)),
            (T::Text, "isdigit" | "isalpha" | "isspace" | "isempty" | "contains" | "startswith" | "endswith") => T::Bool,
            (T::Text, "toint") => T::Int(IntT::named("i64")),
            (T::List(_), "len") => usize_t(),
            (T::List(_), "isempty" | "contains") => T::Bool,
            (T::List(e), "append" | "push") => {
                let nt = T::List(Box::new(e.join(&arg(0))));
                self.place_store(obj, &nt);
                T::Unit
            }
            (T::List(e), "pop") => {
                self.place_store(obj, rt);
                T::Opt(e.clone())
            }
            (T::List(_), "clear" | "sort") => {
                self.place_store(obj, rt);
                T::Unit
            }
            (T::List(e), "with") => T::List(Box::new(e.join(&arg(0)))),
            (T::List(_), "reverse") => rt.clone(),
            (T::List(e), "get") => T::Opt(e.clone()),
            (T::List(_), "utf8") => T::Text,
            (T::List(e), "sum") => match &**e {
                T::Int(i) => T::Int(IntT::of_tags(Tags::lit().join_op(i.tags).union(Tags::lit()))),
                T::Never => T::Int(IntT::of_tags(Tags::lit())),
                o => T::Unknown(format!("sum of {}", o.show())),
            },
            (T::Set(_), "union" | "inter") => rt.clone(),
            (T::Set(e), "insert") => {
                let nt = T::Set(Box::new(e.join(&arg(0))));
                self.place_store(obj, &nt);
                T::Unit
            }
            (T::Set(e), "tolist") => T::List(e.clone()),
            (T::Set(_), "len") => usize_t(),
            (T::Set(_), "contains") => T::Bool,
            (T::Map(..), "len") => usize_t(),
            (T::Map(_, v), "get") => T::Union(v.clone(), Box::new(T::Named("Missing".into()))),
            (T::Map(..), "contains") => T::Bool,
            (T::Map(k, v), "insert") => {
                let nt = T::Map(Box::new(k.join(&arg(0))), Box::new(v.join(&arg(1))));
                self.place_store(obj, &nt);
                T::Unit
            }
            (T::Map(_, v), "remove") => {
                self.place_store(obj, rt);
                T::Opt(v.clone())
            }
            (T::Map(k, _), "keys") => T::List(k.clone()),
            (T::Map(_, v), "values") => T::List(v.clone()),
            (T::Opt(_), "isnone" | "issome") => T::Bool,
            (T::Opt(x), "or") => x.join(&arg(0)),
            (T::Unit, "narrow") => T::Unit,
            (r, _) => T::Unknown(format!("the method {m} of {}", r.show())),
        }
    }

    /// A value of type `ty` is written to the place `p` (an assignment, or a
    /// method that writes its receiver back).
    fn place_store(&mut self, p: &Expr, ty: &RType) {
        match p {
            Expr::Name(n, _) => {
                if let Some(id) = self.lookup(n) {
                    self.store(id, ty.clone());
                }
            }
            Expr::SelfRef(_) => {
                if let Some(id) = self.lookup("self") {
                    self.store(id, ty.clone());
                }
            }
            Expr::Borrow(_, x, _) => self.place_store(x, ty),
            Expr::Member(x, f, _) => {
                let xt = self.place_type(x);
                if let RType::Named(n) = &xt {
                    let ft = self.t.field_type(n, f);
                    if &ft == ty || ty == &RType::Never {
                        return;
                    }
                }
                self.place_store(x, &RType::Unknown(format!("a field of {} that differs from its declared type", xt.show())));
            }
            Expr::TupleMember(x, i, _) => {
                let xt = self.place_type(x);
                if let RType::Tuple(mut ts) = xt
                    && *i < ts.len()
                {
                    ts[*i] = ty.clone();
                    self.place_store(x, &RType::Tuple(ts));
                }
            }
            Expr::Index(x, _, _) => {
                let xt = self.place_type(x);
                let nt = match xt {
                    RType::List(e) => RType::List(Box::new(e.join(ty))),
                    RType::Map(k, v) => RType::Map(k, Box::new(v.join(ty))),
                    o => RType::Unknown(format!("an indexed store into {}", o.show())),
                };
                self.place_store(x, &nt);
            }
            _ => {}
        }
    }

    /// The current type of a place, without recording it as an evaluation.
    fn place_type(&self, p: &Expr) -> RType {
        match p {
            Expr::Name(n, _) => self.lookup(n).map(|id| self.cur(id)).unwrap_or(RType::Unknown(format!("the place {n}"))),
            Expr::SelfRef(_) => self.lookup("self").map(|id| self.cur(id)).unwrap_or(RType::Unknown("self".into())),
            Expr::Borrow(_, x, _) => self.place_type(x),
            Expr::Member(x, f, _) => match self.place_type(x) {
                RType::Named(n) => self.t.field_type(&n, f),
                o => RType::Unknown(format!("the field {f} of {}", o.show())),
            },
            Expr::TupleMember(x, i, _) => match self.place_type(x) {
                RType::Tuple(ts) if *i < ts.len() => ts[*i].clone(),
                o => RType::Unknown(format!("component {i} of {}", o.show())),
            },
            Expr::Index(x, _, _) => match self.place_type(x) {
                RType::List(e) => *e,
                RType::Map(_, v) => *v,
                o => RType::Unknown(format!("an element of {}", o.show())),
            },
            _ => RType::Unknown("not a place".into()),
        }
    }

    // ---- patterns

    /// Bind the names of pattern `p` against a value of type `t`, recording
    /// the literal and range expressions it evaluates.
    fn pat(&mut self, p: &Pat, t: &RType) {
        match p {
            Pat::Wild => {}
            Pat::Path(path) => {
                let n = &path[path.len() - 1];
                if path.len() == 1 && !self.t.is_tested_name(n) {
                    self.node_local(addr(p), |t| &mut t.pat_binds, n, false, t.clone());
                }
            }
            Pat::Ctor(path, args) => {
                let n = &path[path.len() - 1];
                if n == "some" && path.len() == 1 {
                    let inner = match t {
                        RType::Opt(x) => (**x).clone(),
                        o => RType::Unknown(format!("some(…) against {}", o.show())),
                    };
                    if let Some((_, q)) = args.first() {
                        self.pat(q, &inner);
                    }
                    return;
                }
                let fs = self.t.ctor_fields(n).unwrap_or_default();
                for (i, (lb, q)) in args.iter().enumerate() {
                    let ft = match lb {
                        Some(l) => fs.iter().find(|(f, _)| f == l),
                        None => fs.get(i),
                    };
                    let ft = match ft {
                        Some((_, ty)) => self.t.of_ty(ty),
                        None => RType::Unknown(format!("a field of {n}")),
                    };
                    self.pat(q, &ft);
                }
            }
            Pat::Lit(e) => {
                self.expr(e);
            }
            Pat::Range(a, _, b) => {
                self.expr(a);
                self.expr(b);
            }
            Pat::Tuple(ps) => {
                for (k, q) in ps.iter().enumerate() {
                    let ct = match t {
                        RType::Tuple(ts) if k < ts.len() => ts[k].clone(),
                        o => RType::Unknown(format!("a component of {}", o.show())),
                    };
                    self.pat(q, &ct);
                }
            }
            Pat::Alt(ps) => {
                for q in ps {
                    self.pat(q, t);
                }
            }
        }
    }

    // ---- statements

    fn block(&mut self, b: &[Stmt]) {
        self.scopes.push(HashMap::new());
        for s in b {
            self.stmt(s);
        }
        self.scopes.pop();
    }

    fn loop_fix(&mut self, run: &mut dyn FnMut(&mut Self)) {
        // iterate the body until the types at the loop head stop growing
        let entry = self.env.clone();
        let mut head = entry.clone();
        for _ in 0..32 {
            self.env = head.clone();
            self.loops.push(LoopAcc { breaks: None, conts: None });
            run(self);
            let acc = self.loops.pop().unwrap();
            let back = env_join(&self.env, &acc.conts);
            let next = env_join(&entry, &back);
            let done = next == head;
            head = next;
            if done {
                self.env = env_join(&head, &acc.breaks);
                return;
            }
        }
        // no fixpoint in 32 rounds: every type the loop touches is unknown
        let mut e = head.unwrap_or_default();
        for v in e.values_mut() {
            *v = RType::Unknown("a loop whose types do not settle".into());
        }
        self.env = Some(e);
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Expr(e) | Stmt::Emit(e) | Stmt::Fail(e) | Stmt::Check(e) | Stmt::Spawn(e) => {
                self.expr(e);
                if matches!(s, Stmt::Fail(_)) {
                    self.env = None;
                }
            }
            Stmt::Assign(op, place, rhs) => {
                let r = self.expr(rhs);
                let cur = self.expr(place);
                let v = if op == "=" {
                    r
                } else {
                    let a = bin_type(&op[..op.len() - 1], &cur, &r);
                    match (&cur, a) {
                        (RType::Int(c), RType::Int(_)) => RType::Int(IntT::of_tags(c.tags)),
                        (_, a) => a,
                    }
                };
                let stored = retag(&cur, &v);
                self.out.stores.insert(addr(s), stored.clone());
                self.place_store(place, &stored);
            }
            Stmt::Send(..) => {}
            Stmt::Let(p, ty, e, _) => {
                let v = self.expr(e);
                let v = match ty {
                    Some(t) => self.t.fit(&v, t),
                    None => v,
                };
                self.pat(p, &v);
            }
            Stmt::Var(n, ty, e, _) => {
                let v = self.expr(e);
                let v = match ty {
                    Some(t) => self.t.fit(&v, t),
                    None => v,
                };
                self.node_local(addr(s), |t| &mut t.var_binds, n, true, v);
            }
            Stmt::If(arms, els) => {
                let mut out: Option<Env> = None;
                for (c, b) in arms {
                    self.expr(c);
                    let before = self.env.clone();
                    self.block(b);
                    out = env_join(&out, &self.env);
                    self.env = before;
                }
                match els {
                    Some(b) => {
                        self.block(b);
                        out = env_join(&out, &self.env);
                    }
                    None => out = env_join(&out, &self.env),
                }
                self.env = out;
            }
            Stmt::While(c, b, body) => {
                if let Some(b) = b {
                    self.expr(b);
                }
                self.loop_fix(&mut |cx: &mut Self| {
                    cx.expr(c);
                    let exit = cx.env.clone();
                    cx.block(body);
                    // the exit when the condition is false is a break
                    let acc = cx.loops.last_mut().unwrap();
                    acc.breaks = env_join(&acc.breaks, &exit);
                });
            }
            Stmt::Loop(b, body) => {
                if let Some(b) = b {
                    self.expr(b);
                }
                self.loop_fix(&mut |cx: &mut Self| {
                    cx.block(body);
                });
            }
            Stmt::For(p, it, body, _) => {
                let itv = self.expr(it);
                let item = match &itv {
                    RType::List(t) | RType::Set(t) => (**t).clone(),
                    RType::Range(i) => RType::Int(*i),
                    RType::Text => RType::Text,
                    RType::Map(k, v) => RType::Tuple(vec![(**k).clone(), (**v).clone()]),
                    o => RType::Unknown(format!("iteration over {}", o.show())),
                };
                self.loop_fix(&mut |cx: &mut Self| {
                    let exit = cx.env.clone();
                    cx.scopes.push(HashMap::new());
                    cx.pat(p, &item);
                    cx.block(body);
                    cx.scopes.pop();
                    let acc = cx.loops.last_mut().unwrap();
                    acc.breaks = env_join(&acc.breaks, &exit);
                });
            }
            Stmt::Match(e, arms, _) => {
                let v = self.expr(e);
                let mut out: Option<Env> = None;
                for a in arms {
                    let before = self.env.clone();
                    self.scopes.push(HashMap::new());
                    self.pat(&a.pat, &v);
                    if let Some(g) = &a.guard {
                        self.expr(g);
                    }
                    let after_guard = self.env.clone();
                    self.block(&a.body);
                    self.scopes.pop();
                    out = env_join(&out, &self.env);
                    // a later arm sees the state after this arm's pattern and guard
                    self.env = env_join(&before, &after_guard);
                }
                self.env = out;
            }
            Stmt::Return(e, _) => {
                if let Some(e) = e {
                    self.expr(e);
                }
                self.env = None;
            }
            Stmt::Break(_) => {
                if let Some(acc) = self.loops.last_mut() {
                    acc.breaks = env_join(&acc.breaks, &self.env);
                }
                self.env = None;
            }
            Stmt::Continue(_) => {
                if let Some(acc) = self.loops.last_mut() {
                    acc.conts = env_join(&acc.conts, &self.env);
                }
                self.env = None;
            }
            Stmt::Pass => {}
            Stmt::Select(..) => {}
            Stmt::Arena(_, b, body, _) | Stmt::Device(_, b, body, _) => {
                if let Some(b) = b {
                    self.expr(b);
                }
                self.block(body);
            }
            Stmt::Comptime(body) => self.block(body),
            Stmt::Yield(e, _) => {
                self.expr(e);
            }
        }
    }
}

/// The type of `a op b` (interp `binary` and `arith`).
pub fn bin_type(op: &str, a: &RType, b: &RType) -> RType {
    use RType as T;
    match op {
        "and" | "or" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not in" => T::Bool,
        ".." | "..=" => match (a, b) {
            (T::Int(x), T::Int(y)) => T::Range(IntT::of_tags(x.tags.join_op(y.tags))),
            _ => T::Unknown("a range of non-integers".into()),
        },
        _ => match (a, b) {
            (T::Unknown(r), _) | (_, T::Unknown(r)) => T::Unknown(r.clone()),
            (T::Int(x), T::Int(y)) => T::Int(IntT::of_tags(x.tags.join_op(y.tags))),
            (T::F64, T::F64) | (T::F64, T::Int(_)) | (T::Int(_), T::F64) => T::F64,
            (T::F32, T::F32) => T::F32,
            (T::Text, T::Text) if op == "+" => T::Text,
            (T::List(x), T::List(y)) => T::List(Box::new(bin_type(op, x, y))),
            (x, y) => T::Unknown(format!("{} {op} {}", x.show(), y.show())),
        },
    }
}

/// Build the typed tree of a module the checker has accepted.
pub fn build(m: &Module) -> Typed {
    let t = Tables::new(m);
    let mut out = Typed::default();
    let mut consts: HashMap<String, Option<RType>> = HashMap::new();
    let fresh = |out: &mut Typed, consts: &mut HashMap<String, Option<RType>>, f: &mut dyn FnMut(&mut Cx)| {
        let mut cx = Cx { t: &t, out, scopes: vec![HashMap::new()], env: Some(HashMap::new()), loops: vec![], consts };
        f(&mut cx);
    };
    for item in &m.items {
        match item {
            Item::Const { name, .. } => fresh(&mut out, &mut consts, &mut |cx| {
                cx.const_type(name);
            }),
            Item::Check(e) => fresh(&mut out, &mut consts, &mut |cx| {
                cx.expr(e);
            }),
            Item::Intent(_, xs, cks) => {
                for e in cks {
                    fresh(&mut out, &mut consts, &mut |cx| {
                        cx.expr(e);
                    });
                }
                for x in xs {
                    example(&mut out, &mut consts, &t, x);
                }
            }
            Item::Example(x) => example(&mut out, &mut consts, &t, x),
            Item::Struct { fields, .. } | Item::State { fields, .. } => {
                for f in fields {
                    if let Some(e) = &f.init {
                        fresh(&mut out, &mut consts, &mut |cx| {
                            cx.expr(e);
                        });
                    }
                }
            }
            Item::Fn(f) => function(&mut out, &mut consts, &t, f, None),
            Item::Impl { ty, members, .. } => {
                let self_t = t.of_ty(ty);
                for f in members {
                    function(&mut out, &mut consts, &t, f, Some(&self_t));
                }
            }
            Item::Step { region, pat, guard, body, .. } => {
                let fields: Vec<(String, Ty)> = m
                    .items
                    .iter()
                    .find_map(|i| match i {
                        Item::State { name, fields, .. } if name == region => Some(fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let mut cx = Cx { t: &t, out: &mut out, scopes: vec![HashMap::new()], env: Some(HashMap::new()), loops: vec![], consts: &mut consts };
                let ids: Vec<u32> = fields.iter().map(|(n, ty)| cx.new_local(n, true, t.of_ty(ty))).collect();
                cx.scopes.push(HashMap::new());
                cx.pat(pat, &RType::Unknown("the whole event of a step".into()));
                if let Some(g) = guard {
                    cx.expr(g);
                }
                match body {
                    Body::Arrow(e) => {
                        let v = cx.expr(e);
                        if let Some(id) = ids.first() {
                            cx.store(*id, v);
                        }
                    }
                    Body::Block(b) => cx.block(b),
                }
                out.step_fields.insert(addr(item), ids);
            }
            _ => {}
        }
    }
    out
}

fn example(out: &mut Typed, consts: &mut HashMap<String, Option<RType>>, t: &Tables, x: &Example) {
    for (_, args) in &x.events {
        let mut cx = Cx { t, out, scopes: vec![HashMap::new()], env: Some(HashMap::new()), loops: vec![], consts };
        for a in args {
            cx.expr(&a.e);
        }
    }
}

fn function(out: &mut Typed, consts: &mut HashMap<String, Option<RType>>, t: &Tables, f: &FnDecl, self_t: Option<&RType>) {
    let mut cx = Cx { t, out, scopes: vec![HashMap::new()], env: Some(HashMap::new()), loops: vec![], consts };
    let mut ids = vec![];
    let mut self_id = None;
    for p in &f.sig.params {
        match p {
            Param::Named { name, ty, .. } => {
                let pt = if f.sig.generics.is_empty() { t.of_ty(ty) } else { RType::Unknown("a generic function".into()) };
                ids.push(cx.new_local(name, false, pt));
            }
            Param::SelfVal | Param::SelfRef(_) => {
                let st = self_t.cloned().unwrap_or(RType::Unknown("self".into()));
                self_id = Some(cx.new_local("self", matches!(p, Param::SelfRef(true)), st));
            }
        }
    }
    match &f.body {
        Body::Arrow(e) => {
            cx.expr(e);
        }
        Body::Block(b) => cx.block(b),
    }
    out.params.insert(addr(f), (ids, self_id));
}

/// The typed tree of a module with its declaration tables.
pub struct Program<'m> {
    pub m: &'m Module,
    pub t: Tables<'m>,
    pub typed: Typed,
}

impl<'m> Program<'m> {
    pub fn new(m: &'m Module) -> Self {
        Program { m, t: Tables::new(m), typed: build(m) }
    }
}

/// A summary of the typed tree for tests and tools: each node's line and type.
pub fn summary(m: &Module, typed: &Typed) -> BTreeMap<String, usize> {
    let _ = m;
    let mut r = BTreeMap::new();
    for n in typed.nodes.values() {
        *r.entry(if n.ty.is_known() { "known".to_string() } else { "unknown".to_string() }).or_insert(0) += 1;
    }
    r
}
