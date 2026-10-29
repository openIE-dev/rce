//! Types of sources (rce-form.md §3, §8 and FG5): declared types as their
//! declarations, the body type that represents a frame type, plain and owned,
//! and the catalogue's type patterns at a node's type arguments.
use super::catalogue::{Entry, Mode};
use super::ir::*;
use serde_json::Value as J;

pub type R<T> = Result<T, String>;

/// A port of an Apply node of an entry, after a `parts` pattern is spread.
#[derive(Clone, Debug)]
pub enum Port {
    Pat(J),
    Ty(BTy),
}

pub struct Tx<'a> {
    pub m: &'a Module,
}

impl<'a> Tx<'a> {
    pub fn new(m: &'a Module) -> Self {
        Tx { m }
    }

    /// A declared type as its declaration, one level: a record is a tuple of
    /// its fields, a kind a sum of tuples, a refined type the type it refines.
    /// Any other type is itself.
    pub fn unfold(&self, t: &BTy) -> Option<BTy> {
        let mut t = t.clone();
        for _ in 0..=self.m.types.len() {
            let BTy::Decl(i) = t else { return Some(t) };
            t = match self.m.types.get(i as usize)? {
                TypeDecl::Record(fs) => BTy::Tup(fs.iter().map(|f| self.repr(f)).collect::<Option<_>>()?),
                TypeDecl::Kind(vs) => BTy::Sum(
                    vs.iter().map(|fs| fs.iter().map(|f| self.repr(f)).collect::<Option<Vec<_>>>().map(BTy::Tup)).collect::<Option<_>>()?,
                ),
                TypeDecl::Refined(f, _) => self.repr(f)?,
            };
        }
        None
    }

    /// The body type that represents a frame type: the representation the
    /// lowering gives it (§3): text as bytes, records and tuples as tuples,
    /// options as `sum(tup(), τ)`, sets as lists, maps as lists of pairs.
    /// A result is `sum(τ, tup(fields of each error)…)`; channels and actor
    /// handles have none, since §3 states none.
    pub fn repr(&self, t: &FTy) -> Option<BTy> {
        Some(match t {
            FTy::Bool => BTy::Bool,
            FTy::Int(lo, hi) => BTy::Int(*lo, *hi),
            FTy::F64 => BTy::F64,
            FTy::F32 => BTy::F32,
            FTy::Text => BTy::Bytes,
            FTy::Unit => BTy::Tup(vec![]),
            FTy::Tup(xs) => BTy::Tup(xs.iter().map(|x| self.repr(x)).collect::<Option<_>>()?),
            FTy::Arr(x, _) => BTy::Arr(Box::new(self.repr(x)?)),
            FTy::List(x) | FTy::Set(x) => BTy::List(Box::new(self.repr(x)?)),
            FTy::Map(k, v) => BTy::List(Box::new(BTy::Tup(vec![self.repr(k)?, self.repr(v)?]))),
            FTy::Opt(x) => BTy::Sum(vec![BTy::Tup(vec![]), self.repr(x)?]),
            FTy::Res(x, es) => {
                let mut vs = vec![self.repr(x)?];
                for e in es {
                    let fs = self.m.errors.get(*e as usize)?;
                    vs.push(BTy::Tup(fs.iter().map(|f| self.repr(f)).collect::<Option<_>>()?));
                }
                BTy::Sum(vs)
            }
            FTy::Decl(i) => BTy::Decl(*i),
            FTy::Chan(_) | FTy::Actor(_) => return None,
        })
    }

    /// Owned values are used at most once (§8): arrays, lists, tuples, sums,
    /// the order value, and declared types as their declarations.
    pub fn owned(&self, t: &BTy) -> bool {
        match t {
            BTy::Arr(_) | BTy::List(_) | BTy::Tup(_) | BTy::Sum(_) | BTy::Order => true,
            BTy::Decl(_) => self.unfold(t).is_none_or(|u| self.owned(&u)),
            _ => false,
        }
    }

    /// The function a reference names, written in group `group`.
    pub fn callee(&self, group: &str, r: &Ref) -> Option<&'a Func> {
        match r {
            Ref::Key(k, i) => self.m.defs.get(k)?.fns.get(*i as usize),
            Ref::Rec(i) => self.m.defs.get(group)?.fns.get(*i as usize),
        }
    }

    fn arg(&self, args: &[TyArg], i: &J) -> R<BTy> {
        let i = i.as_u64().ok_or("pattern index")? as usize;
        match args.get(i) {
            Some(TyArg::Body(t)) => Ok(t.clone()),
            Some(TyArg::Frame(_)) => Err(format!("type argument {i} is a frame type where a body type is needed")),
            None => Err(format!("type argument {i} is missing")),
        }
    }

    fn frame_arg(&self, args: &[TyArg], i: &J) -> R<FTy> {
        let i = i.as_u64().ok_or("pattern index")? as usize;
        match args.get(i) {
            Some(TyArg::Frame(t)) => Ok(t.clone()),
            _ => Err(format!("type argument {i} is not a frame type")),
        }
    }

    /// The components of a tuple type argument or the variants of a sum.
    fn parts(&self, t: &BTy, want_sum: bool) -> R<Vec<BTy>> {
        match self.unfold(t) {
            Some(BTy::Tup(ts)) if !want_sum => Ok(ts),
            Some(BTy::Sum(ts)) if want_sum => Ok(ts),
            _ => Err(format!("{} is not a {} type", show(t), if want_sum { "sum" } else { "tuple" })),
        }
    }

    /// The component or variant of an argument that a tuple or sum entry
    /// picks: a tuple for `tup.*`, a sum for `sum.*`.
    fn part(&self, t: &BTy, k: i128) -> R<BTy> {
        let u = self.unfold(t).ok_or("an unknown declared type")?;
        let ts = match &u {
            BTy::Tup(ts) | BTy::Sum(ts) => ts,
            _ => return Err(format!("{} has no components", show(t))),
        };
        usize::try_from(k).ok().and_then(|k| ts.get(k)).cloned().ok_or_else(|| format!("{} has no component {k}", show(t)))
    }

    /// The type a (non-`parts`, non-`index`, non-`within`) pattern stands for.
    pub fn pat_ty(&self, p: &J, args: &[TyArg], lits: &[Option<i128>]) -> R<BTy> {
        let a = p.as_array().ok_or("pattern")?;
        let head = a.first().and_then(|h| h.as_str()).ok_or("pattern head")?;
        Ok(match head {
            "bool" => BTy::Bool,
            "f64" => BTy::F64,
            "f32" => BTy::F32,
            "bytes" => BTy::Bytes,
            "order" => BTy::Order,
            "int" => {
                let lo = a[1].as_str().and_then(|s| s.parse().ok()).ok_or("int pattern")?;
                let hi = a[2].as_str().and_then(|s| s.parse().ok()).ok_or("int pattern")?;
                BTy::Int(lo, hi)
            }
            "param" => self.arg(args, &a[1])?,
            "repr" => {
                let f = self.frame_arg(args, &a[1])?;
                self.repr(&f).ok_or_else(|| format!("no body type represents {}", fshow(&f)))?
            }
            "tag" => BTy::Int(0, self.parts(&self.arg(args, &a[1])?, true)?.len() as i128 - 1),
            "part" => {
                let t = self.arg(args, &a[1])?;
                let k = a[2].as_u64().ok_or("part operand")? as usize;
                let lit = lits.get(k).copied().flatten().ok_or_else(|| format!("operand {k} is not an integer literal"))?;
                self.part(&t, lit)?
            }
            "arr" => BTy::Arr(Box::new(self.pat_ty(&a[1], args, lits)?)),
            "list" => BTy::List(Box::new(self.pat_ty(&a[1], args, lits)?)),
            "tup" => BTy::Tup(a[1..].iter().map(|x| self.pat_ty(x, args, lits)).collect::<R<_>>()?),
            "sum" => BTy::Sum(a[1..].iter().map(|x| self.pat_ty(x, args, lits)).collect::<R<_>>()?),
            h => return Err(format!("the pattern `{h}` stands for no single type")),
        })
    }

    /// The operand ports of an entry at its type arguments.
    pub fn operand_ports(&self, e: &Entry, args: &[TyArg]) -> R<Vec<(Mode, Port)>> {
        let mut out = vec![];
        for (m, p) in &e.operands {
            if p[0] == "parts" {
                for t in self.parts(&self.arg(args, &p[1])?, false)? {
                    out.push((*m, Port::Ty(t)));
                }
            } else {
                out.push((*m, Port::Pat(p.clone())));
            }
        }
        Ok(out)
    }

    /// The result types of an entry at its type arguments and literal operands,
    /// with the operand each result returns.
    pub fn results(&self, e: &Entry, args: &[TyArg], lits: &[Option<i128>]) -> R<Vec<(Option<usize>, BTy)>> {
        let mut out = vec![];
        for (of, p) in &e.results {
            if p[0] == "parts" {
                for t in self.parts(&self.arg(args, &p[1])?, false)? {
                    out.push((*of, t));
                }
            } else {
                out.push((*of, self.pat_ty(p, args, lits)?));
            }
        }
        Ok(out)
    }

    /// Whether an operand port takes a source of type `t`. At the top of a
    /// port, it takes an integer type whose range lies inside the port's
    /// (integer-range inclusion, FG5); inside a type, `within` allows the same.
    pub fn port_takes(&self, port: &Port, t: &BTy, args: &[TyArg], lits: &[Option<i128>], own_lit: Option<i128>) -> R<bool> {
        match port {
            Port::Ty(x) => Ok(lies_within(t, x)),
            Port::Pat(p) => self.pattern_takes(p, t, args, lits, own_lit, true),
        }
    }

    fn pattern_takes(&self, p: &J, t: &BTy, args: &[TyArg], lits: &[Option<i128>], own_lit: Option<i128>, top: bool) -> R<bool> {
        let a = p.as_array().ok_or("pattern")?;
        let head = a.first().and_then(|h| h.as_str()).ok_or("pattern head")?;
        match head {
            "index" => {
                let n = match self.unfold(&self.arg(args, &a[1])?) {
                    Some(BTy::Tup(ts)) | Some(BTy::Sum(ts)) => ts.len() as i128,
                    _ => return Err(format!("type argument {} has no components", a[1])),
                };
                match own_lit {
                    Some(k) if matches!(t, BTy::Int(..)) => Ok(0 <= k && k < n),
                    _ => Err("the operand is not an integer literal, the output of a Bind node".into()),
                }
            }
            "within" => {
                let w = self.pat_ty(&a[1], args, lits)?;
                Ok(matches!((t, w), (BTy::Int(a, b), BTy::Int(c, d)) if c <= *a && *b <= d))
            }
            "arr" | "list" => match (head, t) {
                ("arr", BTy::Arr(x)) | ("list", BTy::List(x)) => self.pattern_takes(&a[1], x, args, lits, None, false),
                _ => Ok(false),
            },
            "tup" | "sum" => match (head, t) {
                ("tup", BTy::Tup(xs)) | ("sum", BTy::Sum(xs)) if xs.len() == a.len() - 1 => {
                    for (x, q) in xs.iter().zip(&a[1..]) {
                        if !self.pattern_takes(q, x, args, lits, None, false)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                _ => Ok(false),
            },
            _ => {
                let want = self.pat_ty(p, args, lits)?;
                Ok(if top { lies_within(t, &want) } else { *t == want })
            }
        }
    }
}

/// FG5's rule for one port: equal types, or two integer types with the
/// source's range inside the port's.
pub fn lies_within(t: &BTy, want: &BTy) -> bool {
    match (t, want) {
        (BTy::Int(a, b), BTy::Int(c, d)) => c <= a && b <= d,
        _ => t == want,
    }
}

pub fn show(t: &BTy) -> String {
    super::ir::canon(&super::ir::bty_json(t))
}

pub fn fshow(t: &FTy) -> String {
    super::ir::canon(&super::ir::fty_json(t))
}

/// The types of a node's outputs, given its input types and the integer
/// literals among its inputs; `None` when they cannot be derived.
pub fn node_outs(tx: &Tx, group: &str, n: &Node, lits: &[Option<i128>]) -> Option<Vec<BTy>> {
    match &n.attrs {
        Attr::Lit(t, _) => Some(vec![t.clone()]),
        Attr::Prim(name, args, _) => {
            let e = super::catalogue::entry(name)?;
            Some(tx.results(e, args, lits).ok()?.into_iter().map(|r| r.1).collect())
        }
        Attr::Call(r) => Some(tx.callee(group, r)?.outs.clone()),
        Attr::Catch(r) => Some(catch_outs(tx.callee(group, r)?)),
        Attr::Cap(_) => None,
        Attr::Branch(_, ts) | Attr::Count(ts) | Attr::Measure(_, _, ts) | Attr::Halt(ts) => Some(ts.clone()),
    }
}

/// `catch` returns `sum(tup(outputs…), halt)` and ω: the callee's outputs
/// other than ω, or its halt.
pub fn catch_outs(f: &Func) -> Vec<BTy> {
    let takes = f.outs.last() == Some(&BTy::Order);
    let outs = if takes { &f.outs[..f.outs.len() - 1] } else { &f.outs[..] };
    let mut v = vec![BTy::Sum(vec![BTy::Tup(outs.to_vec()), BTy::halt()])];
    if takes {
        v.push(BTy::Order);
    }
    v
}
