//! Lower a core function's blocks, loops and residual breaks to wasm `br`.
//!
//! The depth law lives in [`rce_emit_wasm::control`]: the residual count is
//! the source depth, an `if` does not decrement it, and `break 0` is the
//! local exit. This module walks the core and hands that backend one
//! function whose integers are the wasm i64 carrier. Wrapping `int.wadd`,
//! `int.wsub` and `int.wmul` at `int[-2^63, 2^63 - 1]` are wasm `i64` wrap.
//! A comparison is signed.

use crate::one::ir::{self, Atom, Lit, Place, Stmt, Ty};
use rce_emit_wasm::control::{self, Expr, Func};

pub use rce_emit_wasm::control::{Label, LabelDepth, Residual, br_label};

/// The text module of `f`, exporting `run`.
pub fn emit(f: &ir::Fn) -> Result<String, String> {
    if !fits_i64(&f.ret) {
        return Err(format!(
            "the structured control lowering returns an i64, and this function returns {:?}",
            f.ret
        ));
    }
    for t in &f.params {
        if !fits_i64(t) {
            return Err(format!(
                "a parameter of type {t:?} does not fit the wasm i64 carrier"
            ));
        }
    }
    let mut next = f.params.len() as u32;
    let body = lower(&f.body, &mut next)?;
    control::emit_func(&Func {
        params: f.params.len() as u32,
        locals: next,
        body,
    })
}

/// Assemble `wat` and accept it only when wasmparser and wasmi both accept
/// the module.
pub fn validate(wat: &str) -> Result<Vec<u8>, String> {
    let bin =
        wat::parse_str(wat).map_err(|e| format!("the program module does not assemble: {e}"))?;
    wasmparser::validate(&bin).map_err(|e| format!("the program module is not valid: {e}"))?;
    let engine = engine();
    wasmi::Module::new(&engine, &bin)
        .map_err(|e| format!("the program module is not valid: {e}"))?;
    Ok(bin)
}

/// Run the exported `run` of a validated module. Fuel stops a loop that
/// never reaches its break.
pub fn run(wat: &str) -> Result<i64, String> {
    let bin = validate(wat)?;
    let engine = engine();
    let module = wasmi::Module::new(&engine, &bin).map_err(|e| e.to_string())?;
    let mut store = wasmi::Store::new(&engine, ());
    store.set_fuel(200_000).map_err(|e| e.to_string())?;
    let inst = wasmi::Linker::<()>::new(&engine)
        .instantiate_and_start(&mut store, &module)
        .map_err(|e| e.to_string())?;
    let f = inst
        .get_typed_func::<(), i64>(&store, "run")
        .map_err(|e| e.to_string())?;
    f.call(&mut store, ()).map_err(|e| e.to_string())
}

fn engine() -> wasmi::Engine {
    let mut c = wasmi::Config::default();
    c.consume_fuel(true);
    wasmi::Engine::new(&c)
}

fn lower(b: &[Stmt], next: &mut u32) -> Result<Vec<control::Stmt>, String> {
    let mut out = Vec::with_capacity(b.len());
    for s in b {
        let at = *next;
        *next += ir::binds(s) as u32;
        out.push(match s {
            Stmt::Let(a) | Stmt::Var(_, a) => control::Stmt::Set(at, atom(a)?),
            Stmt::Prim(p, ts, args) => {
                let n = ir::prim_results(p, ts);
                if n != 1 {
                    return Err(format!("the primitive {p} binds {n} results"));
                }
                control::Stmt::Set(at, prim(p, ts, args)?)
            }
            Stmt::Set(Place::V(i), a) => control::Stmt::Set(*i, atom(a)?),
            Stmt::Set(Place::G(_), _) => {
                return Err("a global is outside the structured control lowering".into());
            }
            Stmt::If(c, t, e) => control::Stmt::If(atom(c)?, lower(t, next)?, lower(e, next)?),
            Stmt::Block(x) => control::Stmt::Block(lower(x, next)?),
            Stmt::Loop(x) => control::Stmt::Loop(lower(x, next)?),
            Stmt::Break(d) => control::Stmt::Break(Residual::at_break(*d)),
            Stmt::Ret(a) => control::Stmt::Return(atom(a)?),
            Stmt::Call(..) => {
                return Err("a call is outside the structured control lowering".into());
            }
            Stmt::Catch(..) => {
                return Err("a catch is outside the structured control lowering".into());
            }
            Stmt::Halt(_) => return Err("a halt is outside the structured control lowering".into()),
            Stmt::Out(_) => return Err("an out is outside the structured control lowering".into()),
            Stmt::Graph(..) => {
                return Err("a graph term is outside the structured control lowering".into());
            }
        });
    }
    Ok(out)
}

fn atom(a: &Atom) -> Result<Expr, String> {
    match a {
        Atom::V(i) => Ok(Expr::Local(*i)),
        Atom::G(_) => Err("a global is outside the structured control lowering".into()),
        Atom::K(_, Lit::Bool(b)) => Ok(Expr::Bool(*b)),
        Atom::K(t, Lit::Int(v)) => {
            if !fits_i64(t) {
                return Err(format!("the integer {v} does not fit the wasm i64 carrier"));
            }
            let n = i64::try_from(*v)
                .map_err(|_| format!("the integer {v} does not fit the wasm i64 carrier"))?;
            Ok(Expr::I64(n))
        }
        Atom::K(_, Lit::F64(_) | Lit::F32(_)) => {
            Err("a float literal is outside the structured control lowering".into())
        }
        Atom::K(_, Lit::Bytes(_)) => {
            Err("a byte literal is outside the structured control lowering".into())
        }
    }
}

fn prim(p: &str, ts: &[Ty], args: &[Atom]) -> Result<Expr, String> {
    let bin = |mk: fn(Box<Expr>, Box<Expr>) -> Expr| -> Result<Expr, String> {
        let [a, b] = args else {
            return Err(format!("the primitive {p} takes two operands"));
        };
        Ok(mk(Box::new(atom(a)?), Box::new(atom(b)?)))
    };
    match p {
        "int.wadd" => {
            i64_carrier(ts)?;
            bin(Expr::Add)
        }
        "int.wsub" => {
            i64_carrier(ts)?;
            bin(Expr::Sub)
        }
        "int.wmul" => {
            i64_carrier(ts)?;
            bin(Expr::Mul)
        }
        "int.eq" => {
            cmp_carrier(ts)?;
            bin(Expr::Eq)
        }
        "int.lt" => {
            cmp_carrier(ts)?;
            bin(Expr::Lt)
        }
        "int.le" => {
            cmp_carrier(ts)?;
            bin(Expr::Le)
        }
        "bool.not" => {
            let [a] = args else {
                return Err("bool.not takes one operand".into());
            };
            Ok(Expr::Not(Box::new(atom(a)?)))
        }
        _ => Err(format!(
            "the primitive {p} is outside the structured control lowering"
        )),
    }
}

fn fits_i64(t: &Ty) -> bool {
    match t {
        Ty::Int(lo, hi) => *lo >= i64::MIN as i128 && *hi <= i64::MAX as i128,
        _ => false,
    }
}

fn is_i64(t: &Ty) -> bool {
    match t {
        Ty::Int(lo, hi) => *lo == i64::MIN as i128 && *hi == i64::MAX as i128,
        _ => false,
    }
}

/// Wrapping wasm i64 is the carrier only for the full i64 range.
fn i64_carrier(ts: &[Ty]) -> Result<(), String> {
    match ts {
        [t] if is_i64(t) => Ok(()),
        [Ty::Int(lo, hi)] => Err(format!("int[{lo}, {hi}] is not the wasm i64 carrier")),
        _ => Err(format!(
            "a type argument {ts:?} is not the wasm i64 carrier"
        )),
    }
}

fn cmp_carrier(ts: &[Ty]) -> Result<(), String> {
    match ts {
        [t] if fits_i64(t) => Ok(()),
        _ => Err(format!(
            "a comparison type {ts:?} does not fit the wasm i64 carrier"
        )),
    }
}
