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
    /// a lambda made by a `let`, by the address of its expression: the
    /// value carries nothing the lowering keeps, since each call runs the
    /// lambda's body in the scope where it was made
    Closure(usize),
    /// a call of an iterator function (5.14) not yet run: the key of its
    /// instantiation and the type of the items it yields
    Gen(String, Box<RType>),
    /// a value this pass does not type, and why
    Unknown(String),
}

impl RType {
    pub fn is_known(&self) -> bool {
        match self {
            RType::Unknown(_) => false,
            RType::Tuple(ts) => ts.iter().all(|t| t.is_known()),
            RType::List(t) | RType::Set(t) | RType::Opt(t) | RType::Gen(_, t) => t.is_known(),
            RType::Map(a, b) | RType::Union(a, b) => a.is_known() && b.is_known(),
            _ => true,
        }
    }
    pub fn unknown_reason(&self) -> Option<String> {
        match self {
            RType::Unknown(r) => Some(r.clone()),
            RType::Tuple(ts) => ts.iter().find_map(|t| t.unknown_reason()),
            RType::List(t) | RType::Set(t) | RType::Opt(t) | RType::Gen(_, t) => t.unknown_reason(),
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
            Closure(_) => "a lambda".into(),
            Gen(_, t) => format!("iter {}", t.show()),
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
    /// `C.lit(t)` or `C.encode(t)` of a context C (5.16): (context, method)
    Ctx(String, String),
    /// `set_of(xs)`
    SetOf,
    /// a call of a lambda a `let` made, by the address of the lambda
    Closure(usize),
    /// a call the reference stops at, with this trap text: a foreign
    /// function, which has no definition to run (interp `call`)
    Trap(String),
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
    /// the local each parameter of a lambda introduces, keyed by the parameter
    pub lambda_binds: HashMap<usize, u32>,
    /// the locals of a function's named parameters, and of its `self`
    pub params: HashMap<usize, (Vec<u32>, Option<u32>)>,
    /// the locals of the state fields inside a step, in declaration order
    pub step_fields: HashMap<usize, Vec<u32>>,
    /// the type a statement leaves in its place (the type of the value an
    /// assignment stores, the receiver of a mutating method)
    pub stores: HashMap<usize, RType>,
    /// the instantiation a call runs, by the call's node, where the
    /// function is generic or its arguments are not of its declared types
    pub inst_of: HashMap<usize, String>,
    /// every instantiation, by key (in the tree of the module only)
    pub insts: HashMap<String, Inst>,
    /// the lambdas `let` makes, by address: a copy of the expression,
    /// whose body is typed at each call (the copy's nodes have addresses
    /// of their own, which the facts of the body are keyed by)
    pub lambdas: HashMap<usize, std::sync::Arc<Expr>>,
    /// the names in scope where each of those lambdas is made
    pub lambda_scopes: HashMap<usize, Vec<HashMap<String, u32>>>,
    /// the `let` locals bound to a list literal with at least one item,
    /// which are never empty
    pub nonempty: std::collections::HashSet<u32>,
    /// the two hidden locals the `else` of a budget sees (the limit, and
    /// the units the block spent), keyed by the address of the statement
    /// that holds the `else` and that address plus one
    pub hidden_binds: HashMap<usize, u32>,
}

/// One instantiation of a function at the types of its arguments (ONE-4:
/// generics are monomorphised). The reference fits an argument to its
/// parameter's declared type, and a type parameter fits every value
/// unchanged, so the body sees the arguments' own types there.
#[derive(Debug)]
pub struct Inst {
    /// the typed tree of the body at these types
    pub typed: Typed,
    /// the types of the parameters, self first
    pub params: Vec<RType>,
    /// the type of the result, fitted to the declared result type
    pub ret: RType,
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

/// Named fields with their declared types.
pub type Fields = Vec<(String, Ty)>;

/// The declarations of a module, as the interpreter resolves them.
pub struct Tables<'a> {
    pub fns: HashMap<String, &'a FnDecl>,
    pub methods: HashMap<(String, String), &'a FnDecl>,
    /// struct, state, event, error, actor and alias items by name
    pub types: HashMap<String, &'a Item>,
    /// variant name -> (kind name, index, fields)
    pub variants: HashMap<String, (String, usize, Fields)>,
    pub kinds: HashMap<String, Vec<(String, Fields)>>,
    pub consts: HashMap<String, (Option<Ty>, &'a Expr)>,
    /// the declared events, in declaration order
    pub events: Vec<String>,
    /// the type parameters of the impl of each method, by (type head, name)
    pub impl_generics: HashMap<(String, String), Vec<String>>,
    /// the foreign functions (`extern`) by name
    pub externs: std::collections::HashSet<String>,
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
            impl_generics: HashMap::new(),
            externs: Default::default(),
        };
        for i in &m.items {
            match i {
                Item::Fn(f) => {
                    t.fns.insert(f.sig.name.clone(), f);
                }
                Item::Impl { ty, members, generics, .. } => {
                    let h = crate::check::ty_head(ty);
                    for f in members {
                        t.methods.insert((h.clone(), f.sig.name.clone()), f);
                        t.impl_generics.insert((h.clone(), f.sig.name.clone()), generics.iter().map(|g| g.name.clone()).collect());
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
                Item::Extern { sig, .. } => {
                    t.externs.insert(sig.name.clone());
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
    /// The type of a value after the interpreter fits it to the declared type
    /// `t`; a refined type's values are those of its base type.
    pub fn of_ty(&self, t: &Ty) -> RType {
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
        fit_type(v, want)
    }
    /// interp `fit` where the type parameters `gs` fit every value
    /// unchanged; a value that does not have the declared type's form is
    /// also left as it is, as the interpreter leaves it.
    pub fn fit_generic(&self, v: &RType, t: &Ty, gs: &[String]) -> RType {
        fn mentions(t: &Ty, gs: &[String]) -> bool {
            match t {
                Ty::Path(p, a) => (p.len() == 1 && a.is_empty() && gs.contains(&p[0])) || a.iter().any(|x| mentions(x, gs)),
                Ty::Tuple(ts) => ts.iter().any(|x| mentions(x, gs)),
                Ty::Slice(x) | Ty::Array(x, _) | Ty::Ref(_, x) | Ty::Opt(x) | Ty::Region(x, _) => mentions(x, gs),
                Ty::Union(a, b) => mentions(a, gs) || mentions(b, gs),
                _ => false,
            }
        }
        if !mentions(t, gs) {
            return self.fit(v, t);
        }
        use RType as T;
        if let T::Unknown(r) = v {
            return T::Unknown(r.clone());
        }
        match t {
            Ty::Path(p, a) if p.len() == 1 && a.is_empty() && gs.contains(&p[0]) => v.clone(),
            Ty::Ref(_, x) | Ty::Region(x, _) => self.fit_generic(v, x, gs),
            Ty::Opt(i) => match v {
                T::Opt(x) => T::Opt(Box::new(self.fit_generic(x, i, gs))),
                _ => v.clone(),
            },
            Ty::Path(p, a) if p.len() == 1 && p[0] == "List" && a.len() == 1 => match v {
                T::List(x) => T::List(Box::new(self.fit_generic(x, &a[0], gs))),
                _ => v.clone(),
            },
            Ty::Path(p, a) if p.len() == 1 && p[0] == "Set" && a.len() == 1 => match v {
                T::Set(x) => T::Set(Box::new(self.fit_generic(x, &a[0], gs))),
                _ => v.clone(),
            },
            Ty::Path(p, a) if p.len() == 1 && p[0] == "Map" && a.len() == 2 => match v {
                T::Map(k, x) => T::Map(Box::new(self.fit_generic(k, &a[0], gs)), Box::new(self.fit_generic(x, &a[1], gs))),
                _ => v.clone(),
            },
            Ty::Array(e, _) => match v {
                T::List(x) => T::List(Box::new(self.fit_generic(x, e, gs))),
                _ => v.clone(),
            },
            Ty::Tuple(ts) => match v {
                T::Tuple(xs) if xs.len() == ts.len() => T::Tuple(xs.iter().zip(ts).map(|(x, t)| self.fit_generic(x, t, gs)).collect()),
                _ => v.clone(),
            },
            Ty::Union(ok, err) => match v {
                T::Union(a, e) => T::Union(Box::new(self.fit_generic(a, ok, gs)), e.clone()),
                _ => T::Union(Box::new(self.fit_generic(v, ok, gs)), Box::new(self.of_ty(err))),
            },
            _ => v.clone(),
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

/// The type of a value of type `v` fitted to a declared type whose values
/// have type `want` (interp `fit`): an integer takes the declared name, an
/// integer fitted to f64 or f32 becomes a float of that width, and a float
/// keeps its own width, since the interpreter does not convert a float
/// value to the other width when it fits it; lists, sets, maps, options and
/// tuples are fitted element by element.
pub fn fit_type(v: &RType, want: RType) -> RType {
    use RType as T;
    match (v, want) {
        (T::Unknown(r), _) => T::Unknown(r.clone()),
        (T::F64 | T::F32, T::F64 | T::F32) => v.clone(),
        (T::Opt(x), T::Opt(w)) => T::Opt(Box::new(fit_type(x, *w))),
        (T::List(x), T::List(w)) => T::List(Box::new(fit_type(x, *w))),
        (T::Set(x), T::Set(w)) => T::Set(Box::new(fit_type(x, *w))),
        (T::Map(a, b), T::Map(c, d)) => T::Map(Box::new(fit_type(a, *c)), Box::new(fit_type(b, *d))),
        (T::Tuple(xs), T::Tuple(ws)) if xs.len() == ws.len() => T::Tuple(xs.iter().zip(ws).map(|(x, w)| fit_type(x, w)).collect()),
        // an error value keeps its type through a fit to the union; the ok
        // value is fitted to the ok side
        (x, T::Union(ok, err)) if !matches!(x, T::Union(..)) => T::Union(Box::new(fit_type(x, *ok)), err),
        (_, w) => w,
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
    /// the instantiations typed so far (None while one is being typed)
    insts: &'a mut HashMap<String, Option<Inst>>,
    /// in the body of an instantiation: the types `return` gives
    rets: Option<RType>,
    /// in the body of an instantiation: the type parameters in scope
    gs: Vec<String>,
    /// in the body of an iterator function: the types its `yield`s give
    yields: Option<RType>,
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
        let saved_env = self.env.replace(HashMap::new());
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
            E::ListComp(body, p, it, cond, _) => {
                // interp: the iterable once, then for each item the pattern,
                // the condition and the body in a scope of their own
                let itv = self.expr(it);
                let item = match &itv {
                    RType::List(t) | RType::Set(t) => (**t).clone(),
                    RType::Range(i) => RType::Int(*i),
                    RType::Text => RType::Text,
                    RType::Map(k, v) => RType::Tuple(vec![(**k).clone(), (**v).clone()]),
                    RType::Gen(_, t) => (**t).clone(),
                    o => RType::Unknown(format!("iteration over {}", o.show())),
                };
                let mut bt = RType::Never;
                self.loop_fix(&mut |cx: &mut Self| {
                    let exit = cx.env.clone();
                    cx.scopes.push(HashMap::new());
                    cx.pat(p, &item);
                    if let Some(c) = cond {
                        cx.expr(c);
                    }
                    let t = cx.expr(body);
                    bt = bt.join(&t);
                    cx.scopes.pop();
                    let acc = cx.loops.last_mut().unwrap();
                    acc.breaks = env_join(&acc.breaks, &exit);
                });
                plain(RType::List(Box::new(bt)))
            }
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
            E::Lambda(..) => {
                // a lambda made here: its body is typed where it is called,
                // in the names in scope here (interp: the closure keeps its
                // environment)
                let a = addr(e);
                self.out.lambdas.entry(a).or_insert_with(|| std::sync::Arc::new(e.clone()));
                self.out.lambda_scopes.insert(a, self.scopes.clone());
                plain(RType::Closure(a))
            }
        }
    }

    /// The arguments of a call; each is fitted to its parameter by the callee.
    fn args_fit(&mut self, args: &[Arg], _params: &[(String, Ty)]) -> Vec<RType> {
        args.iter().map(|a| self.expr(&a.e)).collect()
    }

    /// The type of a call of `fd` run as an instantiation, when `fd` is
    /// generic (type parameters only) or an argument fitted to its parameter
    /// keeps a type other than the declared one (a float of the other
    /// width); the call's node records the instantiation.
    fn instance(&mut self, e: &Expr, fd: &FnDecl, self_rt: Option<RType>, impl_gs: &[String], at: &[RType]) -> Option<RType> {
        if fd.sig.generics.iter().any(|g| g.comptime) {
            return None;
        }
        let mut gs: Vec<String> = impl_gs.to_vec();
        gs.extend(fd.sig.generics.iter().map(|g| g.name.clone()));
        let named = Self::named_params(fd);
        if named.len() != at.len() {
            return None;
        }
        let mut params = vec![];
        let has_self = fd.sig.params.iter().any(|p| !matches!(p, Param::Named { .. }));
        if has_self {
            params.push(self_rt.clone()?);
        }
        let mut differs = !gs.is_empty();
        for ((_, ty), a) in named.iter().zip(at) {
            let p = self.t.fit_generic(a, ty, &gs);
            if p != self.t.of_ty(ty) {
                differs = true;
            }
            params.push(p);
        }
        if !differs {
            return None;
        }
        let head = match &self_rt {
            Some(RType::Named(n)) => format!("{n}."),
            Some(RType::List(_)) => "List.".into(),
            Some(RType::Set(_)) => "Set.".into(),
            _ => String::new(),
        };
        let key = format!("{head}{}({})", fd.sig.name, params.iter().map(|p| p.show()).collect::<Vec<_>>().join(", "));
        let ret = self.instantiate(&key, fd, &gs, params);
        self.out.inst_of.insert(addr(e), key);
        Some(ret)
    }

    /// A call of an iterator function: its body typed at the arguments'
    /// types, fitted as the reference fits them, which gives the type of
    /// the items it yields.
    fn iter_instance(&mut self, e: &Expr, fd: &FnDecl, at: &[RType]) -> RType {
        let named = Self::named_params(fd);
        if !fd.sig.generics.is_empty() || named.len() != at.len() || named.len() != fd.sig.params.len() {
            return RType::Unknown(format!("a call of the iterator function {}", fd.sig.name));
        }
        let params: Vec<RType> = named.iter().zip(at).map(|((_, ty), a)| self.t.fit_generic(a, ty, &[])).collect();
        let key = format!("iter {}({})", fd.sig.name, params.iter().map(|p| p.show()).collect::<Vec<_>>().join(", "));
        let item = self.instantiate(&key, fd, &[], params);
        self.out.inst_of.insert(addr(e), key.clone());
        RType::Gen(key, Box::new(item))
    }

    /// Type the body of `fd` with its parameters of types `params`.
    fn instantiate(&mut self, key: &str, fd: &FnDecl, gs: &[String], params: Vec<RType>) -> RType {
        if let Some(x) = self.insts.get(key) {
            return match x {
                Some(i) => i.ret.clone(),
                None => RType::Unknown("a recursive call of an instantiated function".into()),
            };
        }
        self.insts.insert(key.to_string(), None);
        let mut out = Typed::default();
        let iter = matches!(&fd.sig.ret, Some(Ty::Iter(_)));
        let ret = {
            let mut cx = Cx { t: self.t, out: &mut out, scopes: vec![HashMap::new()], env: Some(HashMap::new()), loops: vec![], consts: &mut *self.consts, insts: &mut *self.insts, rets: Some(RType::Never), gs: gs.to_vec(), yields: iter.then_some(RType::Never) };
            let mut ids = vec![];
            let mut self_id = None;
            for (k, p) in fd.sig.params.iter().enumerate() {
                match p {
                    Param::Named { name, .. } => ids.push(cx.new_local(name, false, params[k].clone())),
                    Param::SelfVal | Param::SelfRef(_) => self_id = Some(cx.new_local("self", matches!(p, Param::SelfRef(true)), params[k].clone())),
                }
            }
            let tail = match &fd.body {
                Body::Arrow(e) => cx.expr(e),
                Body::Block(b) => {
                    cx.block(b);
                    // a block that can end without `return` gives its tail, or ()
                    if cx.env.is_some() { tail_rtype(cx.out, b).unwrap_or(RType::Unit) } else { RType::Never }
                }
            };
            let body = cx.rets.take().unwrap_or(RType::Never).join(&tail);
            cx.out.params.insert(addr(fd), (ids, self_id));
            match &fd.sig.ret {
                // an iterator gives its items as `yield` gives them, unfitted
                _ if iter => cx.yields.take().unwrap_or(RType::Never),
                None => RType::Unit,
                Some(t) => self.t.fit_generic(&body, t, gs),
            }
        };
        self.insts.insert(key.to_string(), Some(Inst { typed: out, params, ret: ret.clone() }));
        ret
    }

    /// After a call of `fd`: each `&mut` argument's place takes the value
    /// the callee leaves in its parameter (interp `call_fn`'s copy out),
    /// which a union function's error leaves out (F.4.4).
    fn args_written_back(&mut self, fd: &FnDecl, args: &[Arg]) {
        let union = matches!(&fd.sig.ret, Some(Ty::Union(..)));
        for ((_, ty), a) in Self::named_params(fd).iter().zip(args) {
            if !matches!(ty, Ty::Ref(true, _)) {
                continue;
            }
            let place = match &a.e {
                Expr::Borrow(_, x, _) => &**x,
                e => e,
            };
            let pt = self.t.of_ty(ty);
            let t = if union { self.place_type(place).join(&pt) } else { pt };
            self.place_store(place, &t);
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
        // 5.16 context text: `C.lit(t)` and `C.encode(t)` (interp `call`)
        if let Expr::Member(o, m, _) = f
            && let Expr::Name(c, _) = &**o
            && crate::index08::is_ctx(c)
            && self.lookup(c).is_none()
            && !self.t.types.contains_key(c)
            && args.len() == 1
            && c != "Shell"
        {
            let at = self.expr(&args[0].e);
            let r = match at {
                RType::Text => RType::Text,
                RType::Unknown(r) => RType::Unknown(r),
                o => RType::Unknown(format!("{c}.{m} of {}", o.show())),
            };
            return (r, None, Some(Callee::Ctx(c.clone(), m.clone())));
        }
        // `f[T](x)`: a generic call; types are erased at run time
        if let Expr::Index(g, _, _) = f
            && let Expr::Name(n, _) = &**g
            && let Some(fd) = self.t.fns.get(n).copied()
            && self.lookup(n).is_none()
        {
            // the type arguments are erased at run time (interp `call`)
            self.record(g, &RType::Fn(n.clone()), Some(Binding::Fn(n.clone())), None);
            let ps = Self::named_params(fd);
            let at = self.args_fit(args, &ps);
            self.args_written_back(fd, args);
            if let Some(r) = self.instance(e, fd, None, &[], &at) {
                return (r, None, Some(Callee::Fn(n.clone())));
            }
            return (self.fn_result(fd), None, Some(Callee::Fn(n.clone())));
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
                    "set_of" if args.len() == 1 => {
                        let r = match self.expr(&args[0].e) {
                            RType::List(e) => RType::Set(e),
                            RType::Unknown(r) => RType::Unknown(r),
                            o => RType::Unknown(format!("set_of of {}", o.show())),
                        };
                        return (r, None, Some(Callee::SetOf));
                    }
                    "ct_choose" | "set_of" | "lvar_new" => {
                        for a in args {
                            self.expr(&a.e);
                        }
                        return (RType::Unknown(n.clone()), None, Some(Callee::Unknown(n.clone())));
                    }
                    _ => {}
                }
                if let Some(fd) = self.t.fns.get(n).copied() {
                    let ps = Self::named_params(fd);
                    let at = self.args_fit(args, &ps);
                    self.args_written_back(fd, args);
                    if matches!(&fd.sig.ret, Some(Ty::Iter(_))) {
                        return (self.iter_instance(e, fd, &at), None, Some(Callee::Fn(n.clone())));
                    }
                    if let Some(r) = self.instance(e, fd, None, &[], &at) {
                        return (r, None, Some(Callee::Fn(n.clone())));
                    }
                    return (self.fn_result(fd), None, Some(Callee::Fn(n.clone())));
                }
                if let Some(fs) = self.t.ctor_fields(n) {
                    self.args_fit(args, &fs);
                    return (RType::Named(self.t.ctor_type(n)), None, Some(Callee::Ctor(n.clone())));
                }
                if self.t.externs.contains(n) {
                    // interp `call`: a foreign function is not a function of
                    // the module, so the call traps before its arguments
                    return (RType::Never, None, Some(Callee::Trap(format!("cannot call {n} at comptime"))));
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
                            self.args_written_back(fd, args);
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
                    let at = self.args_fit(args, &ps);
                    self.args_written_back(fd, args);
                    let igs = self.t.impl_generics.get(&(h.clone(), m.clone())).cloned().unwrap_or_default();
                    let r = match self.instance(e, fd, Some(rt.clone()), &igs, &at) {
                        Some(r) => r,
                        None => self.fn_result(fd),
                    };
                    if matches!(fd.sig.params.first(), Some(Param::SelfRef(true))) {
                        // a `&mut self` method writes its receiver back: the
                        // receiver keeps its declared type
                        self.place_store(obj, &rt);
                    }
                    return (r, None, Some(Callee::Method(h, m.clone())));
                }
                if matches!(m.as_str(), "map" | "filter" | "fold")
                    && let RType::List(et) = &rt
                    && let Some(r) = self.hof(et, m, args)
                {
                    return (r, None, Some(Callee::Builtin(m.clone())));
                }
                let at: Vec<RType> = args.iter().map(|a| self.expr(&a.e)).collect();
                let r = self.builtin(obj, &rt, m, &at);
                (r, None, Some(Callee::Builtin(m.clone())))
            }
            _ => {
                let ft = self.expr(f);
                let at: Vec<RType> = args.iter().map(|a| self.expr(&a.e)).collect();
                if let RType::Closure(l) = ft {
                    return (self.closure_call(l, &at), None, Some(Callee::Closure(l)));
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

    /// A call of the lambda made at address l (interp `apply_closure`): its
    /// parameters, fitted to their declared types, in a scope of their own
    /// inside the names in scope where it was made, then its body.
    fn closure_call(&mut self, l: usize, at: &[RType]) -> RType {
        let (Some(lam), Some(scopes)) = (self.out.lambdas.get(&l).cloned(), self.out.lambda_scopes.get(&l).cloned()) else {
            return RType::Unknown("a lambda".into());
        };
        let Expr::Lambda(ps, body, _) = &*lam else { return RType::Unknown("a lambda".into()) };
        let mut named = vec![];
        for (k, p) in ps.iter().enumerate() {
            match p {
                Param::Named { name, ty, .. } => named.push((k, name.clone(), ty.clone())),
                _ => return RType::Unknown("a lambda with a self parameter".into()),
            }
        }
        if named.len() != at.len() {
            return RType::Unknown("a lambda called with another number of arguments".into());
        }
        let saved = std::mem::replace(&mut self.scopes, scopes);
        self.scopes.push(HashMap::new());
        for ((k, n, ty), v) in named.iter().zip(at) {
            let t = self.t.fit_generic(v, ty, &self.gs);
            self.node_local(addr(&ps[*k]), |t| &mut t.lambda_binds, n, false, t);
        }
        let r = self.expr(body);
        self.scopes = saved;
        r
    }

    /// Whether the list obj has at least one item: a list literal with
    /// items, or a `let` local bound to one.
    fn nonempty(&self, obj: &Expr) -> bool {
        match obj {
            Expr::List(xs, _) => !xs.is_empty(),
            Expr::Name(n, _) => self.lookup(n).is_some_and(|id| self.out.nonempty.contains(&id)),
            _ => false,
        }
    }

    /// `xs.map(f)`, `xs.filter(f)` and `xs.fold(init, f)` with a lambda f
    /// (interp `apply_closure`): the lambda's parameters take the items,
    /// fitted to their declared types, and its body is typed until the
    /// types settle. Another function value is not typed here.
    fn hof(&mut self, et: &RType, m: &str, args: &[Arg]) -> Option<RType> {
        let (init, fa) = match (m, args) {
            ("fold", [i, f]) => (Some(self.expr(&i.e)), f),
            ("map" | "filter", [f]) => (None, f),
            _ => return None,
        };
        let Expr::Lambda(ps, body, _) = &fa.e else { return None };
        let named: Vec<(usize, String, Ty)> = ps
            .iter()
            .enumerate()
            .filter_map(|(k, p)| if let Param::Named { name, ty, .. } = p { Some((k, name.clone(), ty.clone())) } else { None })
            .collect();
        let want = if m == "fold" { 2 } else { 1 };
        if named.len() != want {
            return None;
        }
        self.record(&fa.e, &RType::Unknown("a lambda".into()), None, None);
        let mut acc = init.clone().unwrap_or(RType::Never);
        let mut out = RType::Never;
        for _ in 0..16 {
            let before = (acc.clone(), out.clone());
            let saved = self.env.clone();
            self.scopes.push(HashMap::new());
            let ins: Vec<RType> = if m == "fold" { vec![acc.clone(), et.clone()] } else { vec![et.clone()] };
            for ((k, n, ty), v) in named.iter().zip(ins) {
                let t = self.t.fit_generic(&v, ty, &self.gs);
                self.node_local(addr(&ps[*k]), |t| &mut t.lambda_binds, n, false, t);
            }
            let t = self.expr(body);
            self.scopes.pop();
            self.env = env_join(&saved, &self.env);
            out = out.join(&t);
            if m == "fold" {
                acc = acc.join(&t);
            }
            if (acc.clone(), out.clone()) == before {
                break;
            }
        }
        Some(match m {
            "map" => RType::List(Box::new(out)),
            "filter" => RType::List(Box::new(et.clone())),
            _ => acc,
        })
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
                // interp: the sum starts at the integer 0, which the first
                // float turns into a float; an empty list gives the integer
                T::F64 if self.nonempty(obj) => T::F64,
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
            (T::Gen(_, e), "collect") => T::List(e.clone()),
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
                    Some(t) => self.t.fit_generic(&v, t, &self.gs),
                    None => v,
                };
                self.pat(p, &v);
                if let (Pat::Path(n), Expr::List(xs, _)) = (p, e)
                    && n.len() == 1
                    && !xs.is_empty()
                    && let Some(id) = self.lookup(&n[0])
                {
                    self.out.nonempty.insert(id);
                }
            }
            Stmt::Var(n, ty, e, _) => {
                let v = self.expr(e);
                let v = match ty {
                    Some(t) => self.t.fit_generic(&v, t, &self.gs),
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
                    RType::Gen(_, t) => (**t).clone(),
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
                let t = match e {
                    Some(e) => self.expr(e),
                    None => RType::Unit,
                };
                if let Some(r) = self.rets.as_mut() {
                    *r = r.join(&t);
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
            // `with budget n Block else Else` (interp `exec_marker`): the
            // limit, the block, and the else from the state before the block,
            // with the hidden limit and spent units in scope
            Stmt::Arena(name, None, body, _) if name == "$budget" && matches!(body.first(), Some(Stmt::Expr(_))) => {
                let Some(Stmt::Expr(n)) = body.first() else { unreachable!() };
                self.expr(n);
                let (rest, eb) = budget_parts(body);
                let before = self.env.clone();
                self.block(rest);
                if let Some((ifs, eb)) = eb {
                    let after = std::mem::replace(&mut self.env, before);
                    self.scopes.push(HashMap::new());
                    let u = RType::Int(IntT::named("u64"));
                    self.node_local(addr(ifs), |t| &mut t.hidden_binds, "\u{1}__bn", false, u.clone());
                    self.node_local(addr(ifs) + 1, |t| &mut t.hidden_binds, "\u{1}__bu", false, u);
                    self.block(eb);
                    self.scopes.pop();
                    self.env = env_join(&after, &self.env);
                }
            }
            Stmt::Arena(_, b, body, _) | Stmt::Device(_, b, body, _) => {
                if let Some(b) = b {
                    self.expr(b);
                }
                self.block(body);
            }
            Stmt::Comptime(body) => self.block(body),
            Stmt::Yield(e, _) => {
                let t = self.expr(e);
                if let Some(y) = &mut self.yields {
                    *y = y.join(&t);
                }
            }
        }
    }
}

/// The type of the value a statement list ends in, from its typed nodes.
pub fn tail_rtype(t: &Typed, b: &[Stmt]) -> Option<RType> {
    match b.last()? {
        Stmt::Expr(e) => Some(t.ty(e)),
        Stmt::If(arms, Some(els)) => {
            let mut r = tail_rtype(t, els)?;
            for (_, x) in arms {
                r = r.join(&tail_rtype(t, x)?);
            }
            Some(r)
        }
        Stmt::Match(_, arms, _) => {
            let mut r = RType::Never;
            for a in arms {
                r = r.join(&tail_rtype(t, &a.body)?);
            }
            Some(r)
        }
        _ => None,
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
            // interp `arith` has no rule for these operands: the operation
            // traps ("cannot apply …") and gives no value
            (x, y) if no_arith_rule(x, y) || (*x == T::Text && *y == T::Text) => T::Never,
            (x, y) => T::Unknown(format!("{} {op} {}", x.show(), y.show())),
        },
    }
}

/// The block of a budget and its `else` (interp `exec_marker`): a last
/// statement `if false` is the `else`, which the parser writes so.
pub fn budget_parts(body: &[Stmt]) -> (&[Stmt], Option<(&Stmt, &[Stmt])>) {
    match body.last() {
        Some(s @ Stmt::If(arms, _)) if arms.len() == 1 && matches!(&arms[0].0, Expr::Bool(false, _)) => (&body[1..body.len() - 1], Some((s, &arms[0].1[..]))),
        _ => (&body[1..], None),
    }
}

/// Whether interp `arith` has no rule for scalar operands of these types
/// (it has integers with integers, floats of one width, f64 with integers
/// and text with text), so that it traps naming both values.
pub fn no_arith_rule(a: &RType, b: &RType) -> bool {
    use RType as T;
    let scalar = |t: &RType| matches!(t, T::Int(_) | T::F64 | T::F32 | T::Bool | T::Text | T::Unit);
    scalar(a)
        && scalar(b)
        && !matches!(
            (a, b),
            (T::Int(_), T::Int(_)) | (T::F64, T::F64) | (T::F32, T::F32) | (T::F64, T::Int(_)) | (T::Int(_), T::F64) | (T::Text, T::Text)
        )
}

/// Build the typed tree of a module the checker has accepted.
pub fn build(m: &Module) -> Typed {
    let t = Tables::new(m);
    let mut out = Typed::default();
    let mut consts: HashMap<String, Option<RType>> = HashMap::new();
    let mut insts: HashMap<String, Option<Inst>> = HashMap::new();
    let fresh = |out: &mut Typed, consts: &mut HashMap<String, Option<RType>>, insts: &mut HashMap<String, Option<Inst>>, f: &mut dyn FnMut(&mut Cx)| {
        let mut cx = Cx { t: &t, out, scopes: vec![HashMap::new()], env: Some(HashMap::new()), loops: vec![], consts, insts, rets: None, gs: vec![], yields: None };
        f(&mut cx);
    };
    for item in &m.items {
        match item {
            Item::Const { name, .. } => fresh(&mut out, &mut consts, &mut insts, &mut |cx| {
                cx.const_type(name);
            }),
            Item::Check(e) => fresh(&mut out, &mut consts, &mut insts, &mut |cx| {
                cx.expr(e);
            }),
            Item::Intent(_, xs, cks, ..) => {
                for e in cks {
                    fresh(&mut out, &mut consts, &mut insts, &mut |cx| {
                        cx.expr(e);
                    });
                }
                for x in xs {
                    example(&mut out, &mut consts, &mut insts, &t, x);
                }
            }
            Item::Example(x) => example(&mut out, &mut consts, &mut insts, &t, x),
            Item::Struct { fields, .. } | Item::State { fields, .. } => {
                for f in fields {
                    if let Some(e) = &f.init {
                        fresh(&mut out, &mut consts, &mut insts, &mut |cx| {
                            cx.expr(e);
                        });
                    }
                }
            }
            Item::Fn(f) => function(&mut out, &mut consts, &mut insts, &t, f, None),
            Item::Impl { ty, members, .. } => {
                let self_t = t.of_ty(ty);
                for f in members {
                    function(&mut out, &mut consts, &mut insts, &t, f, Some(&self_t));
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
                let mut cx = Cx { t: &t, out: &mut out, scopes: vec![HashMap::new()], env: Some(HashMap::new()), loops: vec![], consts: &mut consts, insts: &mut insts, rets: None, gs: vec![], yields: None };
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
    out.insts = insts.into_iter().filter_map(|(k, v)| v.map(|v| (k, v))).collect();
    out
}

fn example(out: &mut Typed, consts: &mut HashMap<String, Option<RType>>, insts: &mut HashMap<String, Option<Inst>>, t: &Tables, x: &Example) {
    for (_, args) in &x.events {
        let mut cx = Cx { t, out, scopes: vec![HashMap::new()], env: Some(HashMap::new()), loops: vec![], consts, insts, rets: None, gs: vec![], yields: None };
        for a in args {
            cx.expr(&a.e);
        }
    }
}

fn function(out: &mut Typed, consts: &mut HashMap<String, Option<RType>>, insts: &mut HashMap<String, Option<Inst>>, t: &Tables, f: &FnDecl, self_t: Option<&RType>) {
    let mut cx = Cx { t, out, scopes: vec![HashMap::new()], env: Some(HashMap::new()), loops: vec![], consts, insts, rets: None, gs: vec![], yields: None };
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
