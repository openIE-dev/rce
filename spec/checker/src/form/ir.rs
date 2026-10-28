//! The data of an `rce-form/1` module (rce-form.md §10) and its JSON form.
//! Encoding is total: every value of these types has one JSON form, and
//! `encode` writes it as canonical JSON (6.1.1).
pub use crate::one::ir::Lit;
use serde_json::{Value as J, json};
use std::collections::BTreeMap;

/// The format name of a module.
pub const FORMAT: &str = "rce-form/1";

/// A body type (§3).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BTy {
    Bool,
    Int(i128, i128),
    F64,
    F32,
    Bytes,
    Arr(Box<BTy>),
    List(Box<BTy>),
    Tup(Vec<BTy>),
    Sum(Vec<BTy>),
    Decl(u32),
    Order,
}

impl BTy {
    pub fn halt() -> BTy {
        BTy::Sum(vec![BTy::Bytes, BTy::Tup(vec![BTy::Bytes, BTy::Bytes])])
    }
    pub fn u64() -> BTy {
        BTy::Int(0, (1i128 << 64) - 1)
    }
    pub fn u8() -> BTy {
        BTy::Int(0, 255)
    }
    /// Whether `order` occurs in the type.
    pub fn has_order(&self) -> bool {
        match self {
            BTy::Order => true,
            BTy::Arr(t) | BTy::List(t) => t.has_order(),
            BTy::Tup(ts) | BTy::Sum(ts) => ts.iter().any(|t| t.has_order()),
            _ => false,
        }
    }
}

/// A frame type (§3).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FTy {
    Bool,
    Int(i128, i128),
    F64,
    F32,
    Text,
    Unit,
    Tup(Vec<FTy>),
    Arr(Box<FTy>, u64),
    List(Box<FTy>),
    Set(Box<FTy>),
    Map(Box<FTy>, Box<FTy>),
    Opt(Box<FTy>),
    Res(Box<FTy>, Vec<u32>),
    Decl(u32),
    Chan(Box<FTy>),
    Actor(u32),
}

/// A reference to a function: function i of a group, or of the group it is written in.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Ref {
    Key(String, u32),
    Rec(u32),
}

/// A memo cell of ω, named by its definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cell {
    Memo(Ref),
    Buffer(Ref),
}

impl Cell {
    pub fn r(&self) -> &Ref {
        match self {
            Cell::Memo(r) | Cell::Buffer(r) => r,
        }
    }
    pub fn r_mut(&mut self) -> &mut Ref {
        match self {
            Cell::Memo(r) | Cell::Buffer(r) => r,
        }
    }
}

/// A type argument: a body type, or a frame type where the entry takes one (`emit`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TyArg {
    Body(BTy),
    Frame(FTy),
}

/// The attributes of a node; the node's kind number follows from them.
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Lit(BTy, Lit),
    Prim(String, Vec<TyArg>, Option<Cell>),
    Call(Ref),
    Catch(Ref),
    Cap(String),
    Branch(u32, Vec<BTy>),
    Count(Vec<BTy>),
    Measure(u32, u32, Vec<BTy>),
    Halt(Vec<BTy>),
}

impl Attr {
    pub fn kind(&self) -> u8 {
        match self {
            Attr::Lit(..) => 0,
            Attr::Prim(..) | Attr::Call(_) | Attr::Catch(_) | Attr::Cap(_) => 1,
            Attr::Branch(..) => 3,
            Attr::Count(_) | Attr::Measure(..) => 4,
            Attr::Halt(_) => 13,
        }
    }
    /// The reference this node names, if it names one.
    pub fn refs_mut(&mut self) -> Option<&mut Ref> {
        match self {
            Attr::Call(r) | Attr::Catch(r) => Some(r),
            Attr::Prim(_, _, Some(c)) => Some(c.r_mut()),
            _ => None,
        }
    }
    pub fn refs(&self) -> Option<&Ref> {
        match self {
            Attr::Call(r) | Attr::Catch(r) => Some(r),
            Attr::Prim(_, _, Some(c)) => Some(c.r()),
            _ => None,
        }
    }
}

/// A source: output p of node i of the same region, or input j of the region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Src {
    Out(u32, u32),
    In(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub attrs: Attr,
    pub ins: Vec<Src>,
    pub regions: Vec<Body>,
}

/// A region: its nodes and its outputs; its inputs come from its owner.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Body {
    pub nodes: Vec<Node>,
    pub out: Vec<Src>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Bound {
    None,
    N(u128),
    Layers(Vec<Vec<Vec<u32>>>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Func {
    pub ins: Vec<BTy>,
    pub outs: Vec<BTy>,
    pub eff: u8,
    pub bound: Bound,
    pub body: Body,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub fns: Vec<Func>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeDecl {
    Record(Vec<FTy>),
    Kind(Vec<Vec<FTy>>),
    Refined(FTy, Ref),
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegionDecl {
    pub actor: bool,
    /// each field's type and its initial value, a positional canonical JSON value (6.1)
    pub fields: Vec<(FTy, J)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ev {
    Idx(u32),
    Name(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub region: u32,
    pub ev: Ev,
    pub guard: Option<Ref>,
    pub body: Ref,
    pub deadline: Option<u128>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub catalogue: String,
    pub types: Vec<TypeDecl>,
    pub regions: Vec<RegionDecl>,
    pub events: Vec<Vec<FTy>>,
    pub errors: Vec<Vec<FTy>>,
    pub steps: Vec<Step>,
    pub main: Option<Ref>,
    pub caps: Vec<String>,
    pub defs: BTreeMap<String, Group>,
}

// ---- JSON form

pub fn bty_json(t: &BTy) -> J {
    match t {
        BTy::Bool => json!(["bool"]),
        BTy::Int(lo, hi) => json!(["int", lo.to_string(), hi.to_string()]),
        BTy::F64 => json!(["f64"]),
        BTy::F32 => json!(["f32"]),
        BTy::Bytes => json!(["bytes"]),
        BTy::Arr(x) => json!(["arr", bty_json(x)]),
        BTy::List(x) => json!(["list", bty_json(x)]),
        BTy::Tup(xs) | BTy::Sum(xs) => {
            let mut v = vec![J::from(if matches!(t, BTy::Tup(_)) { "tup" } else { "sum" })];
            v.extend(xs.iter().map(bty_json));
            J::Array(v)
        }
        BTy::Decl(i) => json!(["decl", i]),
        BTy::Order => json!(["order"]),
    }
}

pub fn fty_json(t: &FTy) -> J {
    match t {
        FTy::Bool => json!(["bool"]),
        FTy::Int(lo, hi) => json!(["int", lo.to_string(), hi.to_string()]),
        FTy::F64 => json!(["f64"]),
        FTy::F32 => json!(["f32"]),
        FTy::Text => json!(["text"]),
        FTy::Unit => json!(["unit"]),
        FTy::Tup(xs) => {
            let mut v = vec![J::from("tup")];
            v.extend(xs.iter().map(fty_json));
            J::Array(v)
        }
        FTy::Arr(x, n) => json!(["arr", fty_json(x), n.to_string()]),
        FTy::List(x) => json!(["list", fty_json(x)]),
        FTy::Set(x) => json!(["set", fty_json(x)]),
        FTy::Map(k, v) => json!(["map", fty_json(k), fty_json(v)]),
        FTy::Opt(x) => json!(["opt", fty_json(x)]),
        FTy::Res(x, es) => json!(["res", fty_json(x), es]),
        FTy::Decl(i) => json!(["decl", i]),
        FTy::Chan(x) => json!(["chan", fty_json(x)]),
        FTy::Actor(r) => json!(["actor", r]),
    }
}

pub fn ref_json(r: &Ref) -> J {
    match r {
        Ref::Key(k, i) => json!([k, i]),
        Ref::Rec(i) => json!(["rec", i]),
    }
}

pub fn cell_json(c: &Cell) -> J {
    match c {
        Cell::Memo(r) => json!(["memo", ref_json(r)]),
        Cell::Buffer(r) => json!(["buffer", ref_json(r)]),
    }
}

pub fn src_json(s: &Src) -> J {
    match s {
        Src::Out(i, p) => json!([i, p]),
        Src::In(j) => json!([-1, j]),
    }
}

fn tys(ts: &[BTy]) -> J {
    J::Array(ts.iter().map(bty_json).collect())
}

pub fn attrs_json(a: &Attr) -> J {
    match a {
        Attr::Lit(t, l) => json!(["lit", bty_json(t), crate::one::json::lit_json(l)]),
        Attr::Prim(n, args, cell) => {
            let args: Vec<J> = args
                .iter()
                .map(|a| match a {
                    TyArg::Body(t) => bty_json(t),
                    TyArg::Frame(t) => fty_json(t),
                })
                .collect();
            match cell {
                None => json!(["prim", n, args]),
                Some(c) => json!(["prim", n, args, cell_json(c)]),
            }
        }
        Attr::Call(r) => json!(["call", ref_json(r)]),
        Attr::Catch(r) => json!(["catch", ref_json(r)]),
        Attr::Cap(k) => json!(["cap", k]),
        Attr::Branch(n, ts) => json!([n, tys(ts)]),
        Attr::Count(ts) => json!(["count", tys(ts)]),
        Attr::Measure(i, j, ts) => json!(["measure", [i, j], tys(ts)]),
        Attr::Halt(ts) => json!([tys(ts)]),
    }
}

pub fn node_json(n: &Node) -> J {
    json!([
        n.attrs.kind(),
        attrs_json(&n.attrs),
        n.ins.iter().map(src_json).collect::<Vec<_>>(),
        n.regions.iter().map(body_json).collect::<Vec<_>>()
    ])
}

pub fn body_json(b: &Body) -> J {
    json!({"nodes": b.nodes.iter().map(node_json).collect::<Vec<_>>(), "out": b.out.iter().map(src_json).collect::<Vec<_>>()})
}

pub fn bound_json(b: &Bound) -> J {
    match b {
        Bound::None => J::Null,
        Bound::N(n) => J::String(n.to_string()),
        Bound::Layers(l) => json!(["layers", l]),
    }
}

pub fn fn_json(f: &Func) -> J {
    json!({"sig": [tys(&f.ins), tys(&f.outs), f.eff], "bound": bound_json(&f.bound), "body": body_json(&f.body)})
}

pub fn group_json(g: &Group) -> J {
    json!({"fns": g.fns.iter().map(fn_json).collect::<Vec<_>>()})
}

pub fn decl_json(d: &TypeDecl) -> J {
    match d {
        TypeDecl::Record(fs) => json!(["record", fs.iter().map(fty_json).collect::<Vec<_>>()]),
        TypeDecl::Kind(vs) => {
            json!(["kind", vs.iter().map(|fs| fs.iter().map(fty_json).collect::<Vec<_>>()).collect::<Vec<_>>()])
        }
        TypeDecl::Refined(t, r) => json!(["refined", fty_json(t), ref_json(r)]),
    }
}

fn ftys(ts: &[FTy]) -> J {
    J::Array(ts.iter().map(fty_json).collect())
}

pub fn module_json(m: &Module) -> J {
    let defs: serde_json::Map<String, J> = m.defs.iter().map(|(k, g)| (k.clone(), group_json(g))).collect();
    json!({
        "format": FORMAT,
        "catalogue": m.catalogue,
        "types": m.types.iter().map(decl_json).collect::<Vec<_>>(),
        "regions": m.regions.iter().map(|r| json!([
            if r.actor { "actor" } else { "state" },
            r.fields.iter().map(|(t, v)| json!([fty_json(t), v])).collect::<Vec<_>>()
        ])).collect::<Vec<_>>(),
        "events": m.events.iter().map(|e| ftys(e)).collect::<Vec<_>>(),
        "errors": m.errors.iter().map(|e| ftys(e)).collect::<Vec<_>>(),
        "steps": m.steps.iter().map(|s| json!([
            s.region,
            match &s.ev { Ev::Idx(i) => json!(i), Ev::Name(n) => json!(n) },
            s.guard.as_ref().map(ref_json).unwrap_or(J::Null),
            ref_json(&s.body),
            s.deadline.map(|d| J::String(d.to_string())).unwrap_or(J::Null)
        ])).collect::<Vec<_>>(),
        "main": m.main.as_ref().map(ref_json).unwrap_or(J::Null),
        "caps": m.caps,
        "defs": J::Object(defs),
    })
}

/// Canonical JSON text (6.1.1) of a JSON value; floats, which only initial
/// values of the frame hold, in the form of 6.1.1.
pub fn canon(v: &J) -> String {
    crate::interp::canon_json(v)
}

/// SHA-256 of the canonical JSON of a value, in lower-case hex.
pub fn sha(v: &J) -> String {
    crate::interp::d_sha(&canon(v))
}

/// The canonical bytes of a module.
pub fn encode(m: &Module) -> String {
    canon(&module_json(m))
}

/// The key of a group (W5): the SHA-256 of its canonical JSON.
pub fn group_key(g: &Group) -> String {
    sha(&group_json(g))
}
