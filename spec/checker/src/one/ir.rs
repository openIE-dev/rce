//! The data types of the core `rce-core/1` (one-emitter.md, section 3.2).
//!
//! The core is the only input of the printer. It has nine types, three
//! kinds of atom, thirteen statements, three declarations and one hundred
//! primitives. Every name has one meaning at its type argument, the form is
//! in A-normal form (every operand is an atom), and every partial primitive
//! is preceded by its domain test.
//!
//! Locals are numbered per function: the parameters are locals 0 to n - 1,
//! and in textual order each `let`, `prim`, `call`, `catch` and `var` binds
//! the next index or indices. A function is named by the key of its group
//! and its position in the group; a call inside a group names a member of
//! the same group as `rec i`.
use std::collections::BTreeMap;

/// The format name of a core module.
pub const FORMAT: &str = "rce-core/1";

/// The least and greatest integer a core integer type may hold.
pub const INT_MIN: i128 = -(1i128 << 63);
pub const INT_MAX: i128 = (1i128 << 64) - 1;

/// A core type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Ty {
    Bool,
    /// the integers from lo to hi, -2^63 <= lo <= hi <= 2^64 - 1
    Int(i128, i128),
    /// finite binary64 values; -0.0 is distinct from 0.0
    F64,
    /// binary32 values, held as binary64
    F32,
    /// immutable byte strings; text is UTF-8 bytes
    Bytes,
    /// arrays of fixed length
    Arr(Box<Ty>),
    /// growable sequences
    List(Box<Ty>),
    /// products
    Tup(Vec<Ty>),
    /// tagged unions; the tag is int[0, n - 1]
    Sum(Vec<Ty>),
}

impl Ty {
    /// Owned types are consumed exactly once (W3); plain types are copied freely.
    pub fn owned(&self) -> bool {
        matches!(self, Ty::Arr(_) | Ty::List(_) | Ty::Tup(_) | Ty::Sum(_))
    }
    pub fn unit() -> Ty {
        Ty::Tup(vec![])
    }
    /// `Halt = sum(bytes, tup(bytes, bytes))`: a trap text from the closed list
    /// of 6.2, or a declared error as its name and its display.
    pub fn halt() -> Ty {
        Ty::Sum(vec![Ty::Bytes, Ty::Tup(vec![Ty::Bytes, Ty::Bytes])])
    }
    pub fn int(lo: i128, hi: i128) -> Ty {
        Ty::Int(lo, hi)
    }
    pub fn u8() -> Ty {
        Ty::Int(0, 255)
    }
    pub fn u64() -> Ty {
        Ty::Int(0, (1i128 << 64) - 1)
    }
    pub fn i64() -> Ty {
        Ty::Int(-(1i128 << 63), (1i128 << 63) - 1)
    }
    pub fn list(t: Ty) -> Ty {
        Ty::List(Box::new(t))
    }
    /// `option τ` as the lowering holds it: `sum(tup(), τ)`.
    pub fn opt(t: Ty) -> Ty {
        Ty::Sum(vec![Ty::unit(), t])
    }
    /// The bounds of an integer type.
    pub fn bounds(&self) -> Option<(i128, i128)> {
        match self {
            Ty::Int(lo, hi) => Some((*lo, *hi)),
            _ => None,
        }
    }
    /// Whether every value of `self` is a value of `o`.
    pub fn within(&self, o: &Ty) -> bool {
        match (self, o) {
            (Ty::Int(a, b), Ty::Int(c, d)) => c <= a && b <= d,
            _ => self == o,
        }
    }
}

/// ceil(log2(hi - lo + 1)): the width in bits of an integer type, the `{bits}`
/// hole of a table row and the shift bound of `int.bad.shift`.
pub fn bits(lo: i128, hi: i128) -> u32 {
    let n = (hi - lo) as u128 + 1;
    if n <= 1 {
        return 0;
    }
    128 - (n - 1).leading_zeros()
}

/// A literal of a plain type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Lit {
    Bool(bool),
    Int(i128),
    /// the bits of a binary64 value
    F64(u64),
    /// a binary32 value held as the bits of a binary64 value
    F32(u64),
    Bytes(Vec<u8>),
}

/// An atom: a local, a global or a literal.
#[derive(Clone, Debug, PartialEq)]
pub enum Atom {
    V(u32),
    G(u32),
    K(Ty, Lit),
}

impl Atom {
    pub fn int(t: Ty, v: i128) -> Atom {
        Atom::K(t, Lit::Int(v))
    }
    pub fn bool(b: bool) -> Atom {
        Atom::K(Ty::Bool, Lit::Bool(b))
    }
    pub fn bytes(b: &[u8]) -> Atom {
        Atom::K(Ty::Bytes, Lit::Bytes(b.to_vec()))
    }
}

/// The callee of `call` and `catch`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FnRef {
    /// function i of the group with this key
    Key(String, u32),
    /// function i of the group being defined
    Rec(u32),
}

/// The target of `set`.
#[derive(Clone, Debug, PartialEq)]
pub enum Place {
    V(u32),
    G(u32),
}

/// A statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    /// `vN = a`
    Let(Atom),
    /// `vN… = p[ι…](a…)`
    Prim(String, Vec<Ty>, Vec<Atom>),
    /// `vN = call F(a…)`
    Call(FnRef, Vec<Atom>),
    /// `vN = catch F(a…)`: vN : sum(ret_F, Halt)
    Catch(FnRef, Vec<Atom>),
    /// `var vN: τ = a`
    Var(Ty, Atom),
    /// `set vN = a`, `set gN = a`
    Set(Place, Atom),
    If(Atom, Vec<Stmt>, Vec<Stmt>),
    Block(Vec<Stmt>),
    Loop(Vec<Stmt>),
    /// leave the d-th enclosing `block` or `loop`, 0 being the innermost
    Break(u32),
    Ret(Atom),
    Halt(Atom),
    /// append bytes to standard output, the only effect
    Out(Atom),
}

/// A function.
#[derive(Clone, Debug, PartialEq)]
pub struct Fn {
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub body: Vec<Stmt>,
}

/// A set of mutually recursive functions; its key is the SHA-256 of its
/// canonical JSON (W5).
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub fns: Vec<Fn>,
}

/// A core module.
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    /// each global's type and its initial literal; a global of an owned type
    /// has none and is set by the module's initialisation before it is read
    pub globals: Vec<(Ty, Option<Lit>)>,
    pub defs: BTreeMap<String, Group>,
    pub entry: (String, u32),
    /// the concept record: the RCE names each function and global came from;
    /// never hashed
    pub names: serde_json::Value,
}

impl Module {
    pub fn func(&self, key: &str, i: u32) -> Option<&Fn> {
        self.defs.get(key).and_then(|g| g.fns.get(i as usize))
    }
}

/// How many locals a binding statement introduces (`None` for statements
/// that bind nothing). For `prim` it depends on the primitive and its type
/// argument, so it is given by [`prim_results`].
pub fn binds(s: &Stmt) -> usize {
    match s {
        Stmt::Let(_) | Stmt::Call(..) | Stmt::Catch(..) | Stmt::Var(..) => 1,
        Stmt::Prim(p, ty, _) => prim_results(p, ty),
        _ => 0,
    }
}

/// The number of results a primitive binds.
pub fn prim_results(p: &str, targs: &[Ty]) -> usize {
    match p {
        "tup.open" => match targs.first() {
            Some(Ty::Tup(ts)) => ts.len(),
            _ => 0,
        },
        "list.pop" => 2,
        "io.byte" | "mem.set" | "drop" => 0,
        _ => 1,
    }
}

/// The names of the one hundred primitives, by family.
pub const PRIMS: &[&str] = &[
    // bool (4)
    "bool.not", "bool.and", "bool.or", "bool.eq",
    // int (33)
    "int.add", "int.sub", "int.mul", "int.neg", "int.abs", "int.quo", "int.rem",
    "int.eq", "int.lt", "int.le", "int.min", "int.max",
    "int.wadd", "int.wsub", "int.wmul", "int.sadd", "int.ssub", "int.smul",
    "int.and", "int.or", "int.xor", "int.not", "int.shl", "int.shr", "int.cast",
    "int.bad.add", "int.bad.sub", "int.bad.mul", "int.bad.neg", "int.bad.abs",
    "int.bad.quo", "int.bad.shift", "int.bad.cast",
    // f64 (25)
    "f64.add", "f64.sub", "f64.mul", "f64.div", "f64.sqrt",
    "f64.neg", "f64.abs", "f64.floor", "f64.ceil", "f64.round",
    "f64.min", "f64.max", "f64.eq", "f64.lt", "f64.le",
    "f64.of_int", "f64.trunc", "f64.to_f32",
    "f64.bad.add", "f64.bad.sub", "f64.bad.mul", "f64.bad.div", "f64.bad.sqrt",
    "f64.bad.trunc", "f64.bad.to_f32",
    // f32 (9)
    "f32.add", "f32.sub", "f32.mul", "f32.div", "f32.to_f64",
    "f32.bad.add", "f32.bad.sub", "f32.bad.mul", "f32.bad.div",
    // bytes (8)
    "bytes.len", "bytes.get", "bytes.slice", "bytes.cat", "bytes.eq", "bytes.lt",
    "bytes.of_list", "bytes.to_list",
    // arr (4)
    "arr.new", "arr.len", "arr.get", "arr.set",
    // list (6)
    "list.new", "list.len", "list.get", "list.set", "list.push", "list.pop",
    // tup (3)
    "tup.mk", "tup.get", "tup.open",
    // sum (3)
    "sum.mk", "sum.tag", "sum.get",
    // io (1)
    "io.byte",
    // mem (2)
    "mem.get", "mem.set",
    // generic (2)
    "copy", "drop",
];

#[cfg(test)]
mod tests {
    #[test]
    fn one_hundred_primitives() {
        let mut v = super::PRIMS.to_vec();
        v.sort();
        v.dedup();
        assert_eq!(v.len(), 100);
        assert_eq!(super::PRIMS.len(), 100);
    }
    #[test]
    fn widths() {
        assert_eq!(super::bits(0, 255), 8);
        assert_eq!(super::bits(-128, 127), 8);
        assert_eq!(super::bits(0, (1i128 << 64) - 1), 64);
        assert_eq!(super::bits(-(1i128 << 63), (1i128 << 64) - 1), 65);
        assert_eq!(super::bits(0, 1), 1);
        assert_eq!(super::bits(0, 0), 0);
    }
}
