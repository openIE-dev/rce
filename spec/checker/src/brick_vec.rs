//! Test vectors of a brick: generated inputs and the reference
//! interpreter's results on them (`rce index vectors N`).
//!
//! Inputs are built from the parameter types by a fixed rule (no
//! randomness): i64 from a pool of edge and ordinary values, bool, text
//! from a pool, lists of i64 as byte-valued lists of a few lengths (most
//! list code in the libraries is byte code), tuples, options and structs
//! field by field, a generic parameter at i64. A vector holds the
//! arguments in the Form's JSON embedding and either `out` (the result,
//! `interp::form_value`) or `trap` (the trap). An input on which the
//! interpreter runs out of fuel or nesting gives no vector.
use crate::ast::*;
use serde_json::{Value, json};
use std::collections::HashMap;

const INTS: [i64; 16] = [
    0,
    1,
    -1,
    2,
    3,
    7,
    10,
    255,
    256,
    -7,
    65535,
    1000003,
    2147483647,
    4294967295,
    i64::MAX,
    i64::MIN,
];
const TEXTS: [&str; 6] = ["", "a", "abc", "Hello, World", "h\u{e9}llo", "0123456789"];
const LENS: [usize; 5] = [0, 1, 3, 8, 32];
/// Fuel per vector. Lowered loops rebuild their environment each step,
/// so the IR evaluator's work grows faster than the interpreter's; inputs
/// that need more than this many interpreter steps give no vector.
const VEC_FUEL: u64 = 200_000;
/// Stands for a generated map until `vectors` embeds the evaluated value.
const MAP_HOLE: &str = "\u{0}$map";
/// Fuel for the one retry of an input that ran out of VEC_FUEL.
const VEC_FUEL_MAX: u64 = 20_000_000;

struct Gen<'a> {
    structs: HashMap<String, (Vec<String>, &'a Vec<Field>)>,
    kinds: HashMap<String, (Vec<String>, &'a Vec<(String, Vec<(String, Ty)>)>)>,
    generics: Vec<String>,
}

impl Gen<'_> {
    /// Value k of type t, as (embedding, source expression).
    fn val(&self, t: &Ty, k: usize, depth: usize) -> Option<(Value, Expr)> {
        if depth > 6 {
            return None;
        }
        // a qualified type `r.x` names the linked item `x` (link rewrites
        // qualified references to bare names)
        if let Ty::Path(p, a) = t
            && p.len() > 1
        {
            return self.val(&Ty::Path(vec![p[p.len() - 1].clone()], a.clone()), k, depth);
        }
        Some(match t {
            Ty::Path(p, a)
                if p.len() == 1
                    && a.is_empty()
                    && (p[0] == "i64" || self.generics.contains(&p[0])) =>
            {
                let v = INTS[k % INTS.len()];
                (json!(v), Expr::Int(v as i128, 0))
            }
            // the other widths: the pool's values in range, then the
            // width's own edges (u64 above i64::MAX included)
            Ty::Path(p, a)
                if p.len() == 1
                    && a.is_empty()
                    && ["i8", "i16", "i32", "u8", "u16", "u32", "u64", "usize"].contains(&p[0].as_str()) =>
            {
                let (lo, hi) = crate::types::int_range(&p[0]);
                let mut pool: Vec<i128> = INTS
                    .iter()
                    .map(|x| *x as i128)
                    .filter(|x| *x >= lo && *x <= hi)
                    .collect();
                pool.extend([hi, lo, hi - 1, hi / 2 + 1]);
                let v = pool[k % pool.len()];
                let j = match i64::try_from(v) {
                    Ok(i) => json!(i),
                    Err(_) => json!(v as u64),
                };
                (j, Expr::Int(v, 0))
            }
            Ty::Path(p, a) if p.len() == 1 && a.is_empty() && p[0] == "bool" => {
                (json!(k % 2 == 0), Expr::Bool(k % 2 == 0, 0))
            }
            Ty::Path(p, a) if p.len() == 1 && a.is_empty() && p[0] == "text" => {
                let s = TEXTS[k % TEXTS.len()];
                (json!(s), Expr::Str(s.to_string(), 0))
            }
            // a shared borrow is the borrowed value (its embedding is the same)
            Ty::Ref(false, x) => {
                let (v, e) = self.val(x, k, depth)?;
                (v, Expr::Borrow(false, Box::new(e), 0))
            }
            // a slice argument is a list
            Ty::Slice(x) => {
                return self.val(
                    &Ty::Path(vec!["List".into()], vec![(**x).clone()]),
                    k,
                    depth,
                );
            }
            Ty::Path(p, a) if p.len() == 1 && p[0] == "List" && a.len() == 1 => {
                let n = LENS[(k / 2) % LENS.len()];
                let mut vs = vec![];
                let mut es = vec![];
                for j in 0..n {
                    let (v, e) = if matches!(&a[0], Ty::Path(q, b) if q.len() == 1 && b.is_empty() && (q[0] == "i64" || self.generics.contains(&q[0])))
                    {
                        let x = ((k * 31 + j * 17 + 5) % 256) as i64;
                        (json!(x), Expr::Int(x as i128, 0))
                    } else {
                        self.val(&a[0], k + j, depth + 1)?
                    };
                    vs.push(v);
                    es.push(e);
                }
                (Value::Array(vs), Expr::List(es, 0))
            }
            // a map of up to two entries at distinct keys; its embedding
            // (pairs in canonical key order) is filled in by `vectors`
            // from the evaluated value, so it is the interpreter's own
            Ty::Path(p, a) if p.len() == 1 && p[0] == "Map" && a.len() == 2 => {
                let n = k % 3;
                let mut ps = vec![];
                for j in 0..n {
                    let (_, ke) = self.val(&a[0], k + j, depth + 1)?;
                    let (_, ve) = self.val(&a[1], k + 2 * j + 1, depth + 1)?;
                    ps.push((ke, ve));
                }
                (json!(MAP_HOLE), Expr::Map(ps, 0))
            }
            Ty::Tuple(ts) => {
                let mut vs = vec![];
                let mut es = vec![];
                for (j, t) in ts.iter().enumerate() {
                    let (v, e) = self.val(t, k + j, depth + 1)?;
                    vs.push(v);
                    es.push(e);
                }
                (Value::Array(vs), Expr::Tuple(es, 0))
            }
            Ty::Opt(x) => {
                if k % 3 == 0 {
                    (Value::Null, Expr::None(0))
                } else {
                    self.val(x, k, depth + 1)?
                }
            }
            // structs field by field, generic ones at their arguments
            Ty::Path(p, a)
                if p.len() == 1
                    && self.structs.get(&p[0]).is_some_and(|(g, _)| g.len() == a.len()) =>
            {
                let (gs, fs) = &self.structs[&p[0]];
                let env: HashMap<&str, &Ty> = gs.iter().map(|x| x.as_str()).zip(a.iter()).collect();
                let mut m = serde_json::Map::new();
                let mut args = vec![];
                for (j, f) in fs.iter().enumerate() {
                    let (v, e) = self.val(&sub(&f.ty, &env), k + j, depth + 1)?;
                    m.insert(f.name.clone(), v);
                    args.push(Arg {
                        label: Some(f.name.clone()),
                        e,
                    });
                }
                (
                    Value::Object(m),
                    Expr::Call(Box::new(Expr::Name(p[0].clone(), 0)), args, 0, false),
                )
            }
            // a kind: variant k (a fieldless one once nested deep, so
            // recursive kinds stay finite), fields like a struct's, the
            // embedding a record with the variant's name under "$v"
            Ty::Path(p, a)
                if p.len() == 1
                    && self.kinds.get(&p[0]).is_some_and(|(g, vs)| g.len() == a.len() && !vs.is_empty()) =>
            {
                let (gs, vs) = &self.kinds[&p[0]];
                let env: HashMap<&str, &Ty> = gs.iter().map(|x| x.as_str()).zip(a.iter()).collect();
                let leaf = vs.iter().position(|(_, f)| f.is_empty());
                let i = match leaf {
                    Some(l) if depth >= 3 => l,
                    _ => k % vs.len(),
                };
                let (vn, fs) = &vs[i];
                let mut m = serde_json::Map::new();
                let mut args = vec![];
                for (j, (fname, fty)) in fs.iter().enumerate() {
                    let (v, e) = self.val(&sub(fty, &env), k / vs.len() + j, depth + 1)?;
                    m.insert(fname.clone(), v);
                    args.push(Arg {
                        label: Some(fname.clone()),
                        e,
                    });
                }
                m.insert("$v".into(), json!(vn));
                let e = if fs.is_empty() {
                    Expr::Name(vn.clone(), 0)
                } else {
                    Expr::Call(Box::new(Expr::Name(vn.clone(), 0)), args, 0, false)
                };
                (Value::Object(m), e)
            }
            _ => return None,
        })
    }
}

/// Up to `n` vectors of `d` in the linked module `m`.
pub fn vectors(m: &Module, d: &FnDecl, n: usize) -> Vec<Value> {
    let mut structs = HashMap::new();
    let mut kinds = HashMap::new();
    for it in &m.items {
        match it {
            Item::Struct {
                name,
                fields,
                generics,
                ..
            } => {
                structs.insert(name.clone(), (generics.iter().map(|g| g.name.clone()).collect(), fields));
            }
            Item::Kind {
                name,
                generics,
                variants,
                ..
            } => {
                kinds.insert(name.clone(), (generics.iter().map(|g| g.name.clone()).collect(), variants));
            }
            _ => {}
        }
    }
    let g = Gen {
        structs,
        kinds,
        generics: d.sig.generics.iter().map(|x| x.name.clone()).collect(),
    };
    let tys: Vec<&Ty> = d
        .sig
        .params
        .iter()
        .filter_map(|p| {
            if let Param::Named { ty, .. } = p {
                Some(ty)
            } else {
                None
            }
        })
        .collect();
    let mut out = vec![];
    for k in 0..n {
        let mut vs = vec![];
        let mut args = vec![];
        let mut ok = true;
        // &mut parameters: the value is bound to a variable, the call
        // borrows it, and the variable's final value is part of the output
        let mut muts: Vec<(String, Expr)> = vec![];
        let mut reembed: Vec<(usize, Expr)> = vec![];
        for (p, t) in tys.iter().enumerate() {
            let (t, is_mut) = match t {
                Ty::Ref(true, x) => (&**x, true),
                t => (*t, false),
            };
            match g.val(t, k * 7 + p * 3, 0) {
                Some((v, e)) => {
                    if has_hole(&v) {
                        reembed.push((vs.len(), e.clone()));
                    }
                    vs.push(v);
                    if is_mut {
                        let x = format!("$a{p}");
                        muts.push((x.clone(), e));
                        args.push(Arg {
                            label: None,
                            e: Expr::Borrow(true, Box::new(Expr::Name(x, 0)), 0),
                        });
                    } else {
                        args.push(Arg { label: None, e });
                    }
                }
                None => ok = false,
            }
        }
        if !ok {
            return vec![];
        }
        let mut it = crate::interp::Interp::new(m);
        it.fuel = VEC_FUEL;
        it.scopes = vec![HashMap::new()];
        let mut bound = true;
        for (i, e) in &reembed {
            match it.eval(e).map(|v| serde_json::from_str::<Value>(&crate::interp::form_value(&v, m).json())) {
                Ok(Ok(j)) => vs[*i] = j,
                _ => bound = false,
            }
        }
        for (x, e) in &muts {
            match it.eval(e) {
                Ok(v) => {
                    it.scopes[0].insert(x.clone(), v);
                }
                Err(_) => bound = false,
            }
        }
        if !bound {
            continue;
        }
        let call = Expr::Call(Box::new(Expr::Name(d.sig.name.clone(), 0)), args, 0, false);
        let saved = it.scopes.clone();
        let mut r = it.eval(&call);
        // an input over the fuel gets one retry at VEC_FUEL_MAX (heavy
        // library code: hashes, KEMs, bignum arithmetic)
        if let Err(f) = &r
            && crate::interp::why(f.clone()).contains("fuel")
        {
            it = crate::interp::Interp::new(m);
            it.fuel = VEC_FUEL_MAX;
            it.scopes = saved;
            r = it.eval(&call);
        }
        match r {
            Ok(v) => {
                let v = if muts.is_empty() {
                    v
                } else {
                    let mut t = vec![v];
                    for (x, _) in &muts {
                        t.push(
                            it.scopes[0]
                                .get(x)
                                .cloned()
                                .unwrap_or(crate::interp::V::Unit),
                        );
                    }
                    crate::interp::V::Tuple(t.into())
                };
                match serde_json::from_str::<Value>(&crate::interp::form_value(&v, m).json()) {
                    Ok(j) => out.push(json!({"args": vs, "out": j})),
                    Err(_) => continue,
                }
            }
            Err(f) => {
                let w = crate::interp::why(f);
                // resource limits, not results: the IR has its own bounds
                // (form::ir REC_DEPTH, REC_CALLS) and agreement is claimed
                // on results only
                if w.contains("fuel") || w.contains("evaluation depth") {
                    continue;
                }
                out.push(json!({"args": vs, "trap": w}));
            }
        }
    }
    out
}

/// t with the type parameters in env replaced.
fn sub(t: &Ty, env: &HashMap<&str, &Ty>) -> Ty {
    match t {
        Ty::Path(p, a) if p.len() == 1 && a.is_empty() && env.contains_key(p[0].as_str()) => {
            env[p[0].as_str()].clone()
        }
        Ty::Path(p, a) => Ty::Path(p.clone(), a.iter().map(|x| sub(x, env)).collect()),
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|x| sub(x, env)).collect()),
        Ty::Slice(x) => Ty::Slice(Box::new(sub(x, env))),
        Ty::Ref(m, x) => Ty::Ref(*m, Box::new(sub(x, env))),
        Ty::Opt(x) => Ty::Opt(Box::new(sub(x, env))),
        t => t.clone(),
    }
}

/// v holds a map placeholder (its embedding is taken from the evaluated value).
fn has_hole(v: &Value) -> bool {
    match v {
        Value::String(x) => x == MAP_HOLE,
        Value::Array(xs) => xs.iter().any(has_hole),
        Value::Object(m) => m.values().any(has_hole),
        _ => false,
    }
}
