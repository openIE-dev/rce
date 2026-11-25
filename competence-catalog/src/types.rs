//! A minimal type layer for typed monomorphization: generic bodies are CHECKED per
//! instantiation instead of erasing types. I64 / I128 / Bool plus polymorphic integer
//! literals and type variables. Execution stays on the i128 core; the type layer decides
//! which instantiations are admitted and range-checks arguments and returns.

use std::collections::BTreeMap;

use rce_spec_check::ast::Expr;

use crate::lut_emit::Stmt;
use crate::program::{eval_program, Env, FnTable};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    I64,
    I128,
    Bool,
    /// An unbounded integer literal; coerces to the other numeric operand.
    IntLit,
    /// A generic type variable, resolved at monomorphization.
    Var(String),
}

/// A generic function with typed parameters and return.
pub struct TypedFn {
    pub name: String,
    pub tparams: Vec<String>,
    pub params: Vec<(String, Ty)>,
    pub ret: Ty,
    pub body: Stmt,
}

fn numeric(t: &Ty) -> bool {
    matches!(t, Ty::I64 | Ty::I128 | Ty::IntLit)
}

fn resolve(a: &Ty, b: &Ty) -> Option<Ty> {
    match (a, b) {
        (Ty::IntLit, x) => Some(x.clone()),
        (x, Ty::IntLit) => Some(x.clone()),
        (x, y) if x == y => Some(x.clone()),
        _ => Option::None,
    }
}

fn subst(t: &Ty, tp: &[String], args: &[Ty]) -> Ty {
    match t {
        Ty::Var(n) => tp.iter().position(|p| p == n).map(|i| args[i].clone()).unwrap_or_else(|| t.clone()),
        other => other.clone(),
    }
}

/// Infer an expression's type under a typed environment.
pub fn infer(e: &Expr, env: &BTreeMap<String, Ty>) -> Result<Ty, String> {
    use Expr::*;
    match e {
        Int(_, _) => Ok(Ty::IntLit),
        Name(n, _) => env.get(n).cloned().ok_or_else(|| format!("unbound {n}")),
        Neg(a, _) => {
            let t = infer(a, env)?;
            if numeric(&t) { Ok(t) } else { Err("neg on non-numeric".into()) }
        }
        Bin(op, a, b, _) => {
            let x = infer(a, env)?;
            let y = infer(b, env)?;
            match op.as_str() {
                "+" | "-" | "*" | "/" | "%" => {
                    if numeric(&x) && numeric(&y) {
                        resolve(&x, &y).ok_or_else(|| "mixed numeric types".into())
                    } else {
                        Err("arithmetic on non-numeric".into())
                    }
                }
                "==" | "!=" | "<" | ">" | "<=" | ">=" => {
                    if numeric(&x) && numeric(&y) {
                        resolve(&x, &y).ok_or_else(|| "mixed numeric types".to_string())?;
                        Ok(Ty::Bool)
                    } else {
                        Err("comparison on non-numeric".into())
                    }
                }
                "&&" | "||" => {
                    if x == Ty::Bool && y == Ty::Bool { Ok(Ty::Bool) } else { Err("logic on non-bool".into()) }
                }
                _ => Err(format!("unsupported op {op}")),
            }
        }
        _ => Err("unsupported expression in typed body".into()),
    }
}

fn in_range(t: &Ty, v: i128) -> bool {
    match t {
        Ty::I64 => v >= i64::MIN as i128 && v <= i64::MAX as i128,
        Ty::Bool => v == 0 || v == 1,
        _ => true,
    }
}

fn check_stmt(s: &Stmt, env: &BTreeMap<String, Ty>, ret: &Ty) -> Result<(), String> {
    match s {
        Stmt::Return(e) => {
            let t = infer(e, env)?;
            resolve(&t, ret).ok_or_else(|| "return type mismatch".to_string())?;
            Ok(())
        }
        Stmt::Let(n, e) => {
            let t = infer(e, env)?;
            let t = if t == Ty::IntLit { Ty::I128 } else { t };
            let mut e2 = env.clone();
            e2.insert(n.clone(), t);
            let _ = e2;
            Ok(())
        }
        Stmt::Assign(n, e) => {
            let t = infer(e, env)?;
            let cur = env.get(n).ok_or_else(|| format!("assign to undeclared {n}"))?;
            resolve(&t, cur).ok_or_else(|| "assign type mismatch".to_string())?;
            Ok(())
        }
        Stmt::Seq(ss) => {
            let mut e2 = env.clone();
            for s in ss {
                if let Stmt::Let(n, e) = s {
                    let t = infer(e, &e2)?;
                    let t = if t == Ty::IntLit { Ty::I128 } else { t };
                    e2.insert(n.clone(), t);
                }
                check_stmt(s, &e2, ret)?;
            }
            Ok(())
        }
        Stmt::If(c, t, f) => {
            let ct = infer(c, env)?;
            if ct != Ty::Bool && ct != Ty::IntLit {
                return Err("condition not bool".into());
            }
            check_stmt(t, env, ret)?;
            check_stmt(f, env, ret)
        }
        _ => Err("unsupported statement in typed body".into()),
    }
}

/// Check a generic body at a concrete instantiation; returns the instance key.
pub fn monomorphize(tf: &TypedFn, args: &[Ty]) -> Result<String, String> {
    if args.len() != tf.tparams.len() {
        return Err("type argument arity mismatch".into());
    }
    let mut env = BTreeMap::new();
    for (n, t) in &tf.params {
        env.insert(n.clone(), subst(t, &tf.tparams, args));
    }
    let ret = subst(&tf.ret, &tf.tparams, args);
    check_stmt(&tf.body, &env, &ret)?;
    Ok(format!(
        "{}<{}>",
        tf.name,
        args.iter().map(|t| format!("{t:?}")).collect::<Vec<_>>().join(",")
    ))
}

/// Execute a checked instantiation, range-checking arguments and return.
pub fn exec_typed(tf: &TypedFn, args: &[Ty], vals: &[i128]) -> Result<i128, String> {
    monomorphize(tf, args)?;
    if vals.len() != tf.params.len() {
        return Err("value arity mismatch".into());
    }
    let mut env = Env::new();
    for ((n, t), v) in tf.params.iter().zip(vals) {
        let st = subst(t, &tf.tparams, args);
        if !in_range(&st, *v) {
            return Err(format!("arg {n} out of range for {st:?}"));
        }
        env.insert(n.clone(), *v);
    }
    let r = eval_program(&tf.body, &mut env, &mut FnTable::new())?
        .ok_or_else(|| "no return value".to_string())?;
    let rt = subst(&tf.ret, &tf.tparams, args);
    if !in_range(&rt, r) {
        return Err("return out of range".into());
    }
    Ok(r)
}
