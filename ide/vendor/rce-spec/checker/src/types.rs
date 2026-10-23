//! Semantic types (spec 5.1) and unification.
use std::fmt;

pub const INTS: &[&str] = &[
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "usize",
];

pub fn int_range(n: &str) -> (i128, i128) {
    match n {
        "i8" => (-128, 127),
        "i16" => (-32768, 32767),
        "i32" => (i32::MIN as i128, i32::MAX as i128),
        "i64" => (i64::MIN as i128, i64::MAX as i128),
        "u8" => (0, 255),
        "u16" => (0, 65535),
        "u32" => (0, u32::MAX as i128),
        "u64" | "usize" => (0, u64::MAX as i128),
        _ => (i64::MIN as i128, i64::MAX as i128),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum T {
    Bool,
    Int(&'static str),
    F64,
    F32,
    Text,
    Unit,
    Iter(Box<T>),
    Tuple(Vec<T>),
    Array(Box<T>, Option<u64>),
    List(Box<T>),
    Map(Box<T>, Box<T>),
    Set(Box<T>),
    Slice(Box<T>),
    Opt(Box<T>),
    Union(Box<T>, Box<T>),
    Ref(bool, Box<T>),
    Chan(Box<T>),
    Fn(Vec<T>, Box<T>),
    Named(String, Vec<T>),
    Param(String),
    Region(Box<T>, String),
    Var(usize),
    Never,
}

impl fmt::Display for T {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let j = |v: &Vec<T>| {
            v.iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self {
            T::Bool => write!(f, "bool"),
            T::Int(n) => write!(f, "{n}"),
            T::F64 => write!(f, "f64"),
            T::F32 => write!(f, "f32"),
            T::Iter(t) => write!(f, "iter {t}"),
            T::Text => write!(f, "text"),
            T::Unit => write!(f, "()"),
            T::Tuple(v) => write!(f, "({})", j(v)),
            T::Array(t, n) => write!(
                f,
                "[{t}; {}]",
                n.map(|n| n.to_string()).unwrap_or("n".into())
            ),
            T::List(t) => write!(f, "List[{t}]"),
            T::Map(k, v) => write!(f, "Map[{k}, {v}]"),
            T::Set(t) => write!(f, "Set[{t}]"),
            T::Slice(t) => write!(f, "[{t}]"),
            T::Opt(t) => write!(f, "{t}?"),
            T::Union(t, e) => write!(f, "{t} ! {e}"),
            T::Ref(m, t) => write!(f, "&{}{t}", if *m { "mut " } else { "" }),
            T::Chan(t) => write!(f, "chan {t}"),
            T::Fn(a, r) => write!(f, "fn({}) -> {r}", j(a)),
            T::Named(n, a) => {
                if a.is_empty() {
                    write!(f, "{n}")
                } else {
                    write!(f, "{n}[{}]", j(a))
                }
            }
            T::Param(n) => write!(f, "{n}"),
            T::Region(t, r) => write!(f, "{t} @{r}"),
            T::Var(i) => write!(f, "?{i}"),
            T::Never => write!(f, "never"),
        }
    }
}

#[derive(Default)]
pub struct Subst {
    pub v: Vec<Option<T>>,
    pub intlit: Vec<bool>,
    pub listlit: Vec<Option<(T, u64)>>,
}

impl Subst {
    pub fn fresh(&mut self) -> T {
        self.v.push(None);
        self.intlit.push(false);
        self.listlit.push(None);
        T::Var(self.v.len() - 1)
    }
    pub fn fresh_int(&mut self) -> T {
        self.v.push(None);
        self.intlit.push(true);
        self.listlit.push(None);
        T::Var(self.v.len() - 1)
    }
    /// A list literal of `n` elements: a List unless it meets a fixed-size array type (5.3 [ListLit]).
    pub fn fresh_list(&mut self, e: T, n: u64) -> T {
        self.v.push(None);
        self.intlit.push(false);
        self.listlit.push(Some((e, n)));
        T::Var(self.v.len() - 1)
    }
    pub fn shallow(&self, t: &T) -> T {
        let mut t = t.clone();
        while let T::Var(i) = t {
            match &self.v[i] {
                Some(u) => t = u.clone(),
                None => break,
            }
        }
        t
    }
    pub fn resolve(&self, t: &T) -> T {
        let t = self.shallow(t);
        if let T::Var(i) = t
            && let Some((e, _)) = &self.listlit[i]
        {
            return T::List(Box::new(self.resolve(e)));
        }
        let r = |x: &T| Box::new(self.resolve(x));
        match t {
            T::Tuple(v) => T::Tuple(v.iter().map(|x| self.resolve(x)).collect()),
            T::Array(a, n) => T::Array(r(&a), n),
            T::List(a) => T::List(r(&a)),
            T::Map(a, b) => T::Map(r(&a), r(&b)),
            T::Set(a) => T::Set(r(&a)),
            T::Slice(a) => T::Slice(r(&a)),
            T::Opt(a) => T::Opt(r(&a)),
            T::Union(a, b) => T::Union(r(&a), r(&b)),
            T::Ref(m, a) => T::Ref(m, r(&a)),
            T::Chan(a) => T::Chan(r(&a)),
            T::Iter(a) => T::Iter(r(&a)),
            T::Fn(a, b) => T::Fn(a.iter().map(|x| self.resolve(x)).collect(), r(&b)),
            T::Named(n, a) => T::Named(n, a.iter().map(|x| self.resolve(x)).collect()),
            T::Region(a, s) => T::Region(r(&a), s),
            other => other,
        }
    }
    /// Default unconstrained integer-literal variables to i64 (spec 5.3 [Int]).
    pub fn default_ints(&mut self) {
        for i in 0..self.v.len() {
            if self.v[i].is_none() && self.intlit[i] {
                self.v[i] = Some(T::Int("i64"));
            }
        }
    }
    fn occurs(&self, i: usize, t: &T) -> bool {
        match self.resolve(t) {
            T::Var(j) => i == j,
            T::Tuple(v) | T::Named(_, v) => v.iter().any(|x| self.occurs(i, x)),
            T::Fn(a, b) => a.iter().any(|x| self.occurs(i, x)) || self.occurs(i, &b),
            T::Array(a, _)
            | T::List(a)
            | T::Set(a)
            | T::Slice(a)
            | T::Opt(a)
            | T::Ref(_, a)
            | T::Chan(a)
            | T::Iter(a)
            | T::Region(a, _) => self.occurs(i, &a),
            T::Map(a, b) | T::Union(a, b) => self.occurs(i, &a) || self.occurs(i, &b),
            _ => false,
        }
    }
    pub fn unify(&mut self, a: &T, b: &T) -> bool {
        let (a, b) = (self.shallow(a), self.shallow(b));
        match (&a, &b) {
            (T::Var(i), T::Var(j)) if i == j => true,
            (T::Var(i), other) | (other, T::Var(i)) => {
                let i = *i;
                if let Some((e, n)) = self.listlit[i].clone() {
                    let ok = match other {
                        T::List(f) => self.unify(&e, f),
                        T::Array(f, m) => (m.is_none() || *m == Some(n)) && self.unify(&e, f),
                        T::Var(j) => {
                            let j = *j;
                            if self.intlit[j] {
                                return false;
                            }
                            match self.listlit[j].clone() {
                                Some((f, m)) => m == n && self.unify(&e, &f),
                                None => {
                                    self.listlit[j] = Some((e, n));
                                    true
                                }
                            }
                        }
                        _ => false,
                    };
                    if !ok {
                        return false;
                    }
                    self.listlit[i] = None;
                    self.v[i] = Some(other.clone());
                    return true;
                }
                if self.intlit[i] {
                    match other {
                        T::Int(_) => {}
                        T::Var(j) => {
                            let j = *j;
                            self.intlit[j] = true;
                        }
                        _ => return false,
                    }
                }
                if self.occurs(i, other) {
                    return false;
                }
                self.v[i] = Some(other.clone());
                true
            }
            (T::Never, _) | (_, T::Never) => true,
            (T::Tuple(x), T::Tuple(y)) | (T::Named(_, x), T::Named(_, y)) if x.len() == y.len() => {
                if let (T::Named(n, _), T::Named(m, _)) = (&a, &b)
                    && n != m
                {
                    return false;
                }
                x.iter().zip(y.iter()).all(|(p, q)| self.unify(p, q))
            }
            (T::Array(x, n), T::Array(y, m)) => {
                (n.is_none() || m.is_none() || n == m) && self.unify(x, y)
            }
            (T::List(x), T::List(y))
            | (T::Set(x), T::Set(y))
            | (T::Slice(x), T::Slice(y))
            | (T::Opt(x), T::Opt(y))
            | (T::Chan(x), T::Chan(y))
            | (T::Iter(x), T::Iter(y)) => self.unify(x, y),
            (T::Map(x1, x2), T::Map(y1, y2)) | (T::Union(x1, x2), T::Union(y1, y2)) => {
                self.unify(x1, y1) && self.unify(x2, y2)
            }
            (T::Ref(m, x), T::Ref(n, y)) => m == n && self.unify(x, y),
            (T::Region(x, r), T::Region(y, s)) => r == s && self.unify(x, y),
            (T::Fn(x, r), T::Fn(y, s)) if x.len() == y.len() => {
                x.iter().zip(y.iter()).all(|(p, q)| self.unify(p, q)) && self.unify(r, s)
            }
            _ => a == b,
        }
    }
}

pub fn peel(t: &T) -> T {
    match t {
        T::Ref(_, x) | T::Region(x, _) => peel(x),
        o => o.clone(),
    }
}

pub fn contains_region(t: &T, r: &str) -> bool {
    match t {
        T::Region(x, s) => s == r || contains_region(x, r),
        T::Tuple(v) | T::Named(_, v) => v.iter().any(|x| contains_region(x, r)),
        T::Array(a, _)
        | T::List(a)
        | T::Set(a)
        | T::Slice(a)
        | T::Opt(a)
        | T::Ref(_, a)
        | T::Chan(a) => contains_region(a, r),
        T::Map(a, b) | T::Union(a, b) => contains_region(a, r) || contains_region(b, r),
        _ => false,
    }
}
pub fn contains_ref(t: &T) -> bool {
    match t {
        T::Ref(..) => true,
        T::Tuple(v) | T::Named(_, v) => v.iter().any(contains_ref),
        T::Array(a, _)
        | T::List(a)
        | T::Set(a)
        | T::Slice(a)
        | T::Opt(a)
        | T::Chan(a)
        | T::Region(a, _) => contains_ref(a),
        T::Map(a, b) | T::Union(a, b) => contains_ref(a) || contains_ref(b),
        _ => false,
    }
}
