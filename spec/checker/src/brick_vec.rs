//! Test vectors of a brick: generated inputs and the reference
//! interpreter's results on them (`rce index --vectors N`).
//!
//! Inputs are built from the parameter types by a fixed rule (no
//! randomness): i64 from a pool of edge and ordinary values, bool, text
//! from a pool, lists of i64 as byte-valued lists of a few lengths (most
//! list code in the libraries is byte code), tuples, options and structs
//! field by field, a generic parameter at i64. A vector holds the
//! arguments in the Form's JSON embedding and either `out` (the result,
//! `interp::form_value`) or `trap` (the trap). An input on which the
//! interpreter runs out of fuel gives no vector.
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

struct Gen<'a> {
    structs: HashMap<String, &'a Vec<Field>>,
    generics: Vec<String>,
}

impl Gen<'_> {
    /// Value k of type t, as (embedding, source expression).
    fn val(&self, t: &Ty, k: usize, depth: usize) -> Option<(Value, Expr)> {
        if depth > 4 {
            return None;
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
            Ty::Path(p, a) if p.len() == 1 && a.is_empty() && p[0] == "usize" => {
                let v = INTS[k % INTS.len()].max(0);
                (json!(v), Expr::Int(v as i128, 0))
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
            Ty::Path(p, a) if p.len() == 1 && a.is_empty() && self.structs.contains_key(&p[0]) => {
                let fs = self.structs[&p[0]];
                let mut m = serde_json::Map::new();
                let mut args = vec![];
                for (j, f) in fs.iter().enumerate() {
                    let (v, e) = self.val(&f.ty, k + j, depth + 1)?;
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
            _ => return None,
        })
    }
}

/// Up to `n` vectors of `d` in the linked module `m`.
pub fn vectors(m: &Module, d: &FnDecl, n: usize) -> Vec<Value> {
    let mut structs = HashMap::new();
    for it in &m.items {
        if let Item::Struct {
            name,
            fields,
            generics,
            ..
        } = it
            && generics.is_empty()
        {
            structs.insert(name.clone(), fields);
        }
    }
    let g = Gen {
        structs,
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
        for (p, t) in tys.iter().enumerate() {
            let (t, is_mut) = match t {
                Ty::Ref(true, x) => (&**x, true),
                t => (*t, false),
            };
            match g.val(t, k * 7 + p * 3, 0) {
                Some((v, e)) => {
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
        match it.eval(&call) {
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
                    crate::interp::V::Tuple(t)
                };
                match serde_json::from_str::<Value>(&crate::interp::form_value(&v, m).json()) {
                    Ok(j) => out.push(json!({"args": vs, "out": j})),
                    Err(_) => continue,
                }
            }
            Err(f) => {
                let w = crate::interp::why(f);
                if w.contains("fuel") {
                    continue;
                }
                out.push(json!({"args": vs, "trap": w}));
            }
        }
    }
    out
}
