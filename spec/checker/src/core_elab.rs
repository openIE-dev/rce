//! S6: elaboration of an admitted module's `fn main` subset into the Lean
//! core (`spec/lean/RCE/Core/Lang.lean`) as the JSON read by the
//! `rce-lean-core` executable. The subset keeps the reference's tick count
//! exact (book 12 Fuel): integers, bools and text, `+ - * / % & | ^ << >> < > ==`,
//! `if`, `let`, and calls of one-parameter functions with arrow bodies.
//! Anything else is `Err(reason)`, which the adapter reports as
//! `unsupported`, never as a pass.
use crate::ast::{Body, Expr, Item, Module, Param, Pat, Stmt, Ty};
use crate::types::{INTS, int_range};
use serde_json::{Value, json};

#[derive(Clone, PartialEq, Debug)]
enum CT {
    Int(i128, i128),
    Bool,
    Text,
    F64,
    F32,
}

fn cty(t: &Ty) -> Result<CT, String> {
    match t {
        Ty::Path(p, g) if p.len() == 1 && g.is_empty() => match p[0].as_str() {
            "bool" => Ok(CT::Bool),
            "text" | "str" => Ok(CT::Text),
            "f64" => Ok(CT::F64),
            "f32" => Ok(CT::F32),
            n if INTS.contains(&n) => {
                let (lo, hi) = int_range(n);
                Ok(CT::Int(lo, hi))
            }
            n => Err(format!("type `{n}` is outside the core subset")),
        },
        _ => Err("type outside the core subset".into()),
    }
}

fn tj(t: &CT) -> Value {
    match t {
        CT::Int(lo, hi) => json!(["int", num(*lo), num(*hi)]),
        CT::Bool => json!("bool"),
        CT::Text => json!("text"),
        CT::F64 => json!("f64"),
        CT::F32 => json!("f32"),
    }
}

fn num(n: i128) -> Value {
    if let Ok(v) = i64::try_from(n) {
        json!(v)
    } else if let Ok(v) = u64::try_from(n) {
        json!(v)
    } else {
        Value::Null
    }
}

const I64: CT = CT::Int(i64::MIN as i128, i64::MAX as i128);

struct Fun {
    name: String,
    arg: CT,
    ret: CT,
}

struct Cx<'a> {
    funs: &'a [Fun],
    env: Vec<(String, CT)>,
    /// declared errors; error k is raised as user code k + 1
    errs: Vec<String>,
}

/// Bounds wide enough for any checked 64-bit add, sub or mul (as strings,
/// beyond JSON's 64-bit numbers)
const WIDE: i128 = 1 << 100;

impl Cx<'_> {
    /// type of an expression without elaborating (for literal defaults)
    fn peek(&self, e: &Expr) -> Option<CT> {
        match e {
            Expr::Name(n, _) => self.env.iter().rev().find(|(m, _)| m == n).map(|x| x.1.clone()),
            Expr::Bool(..) => Some(CT::Bool),
            Expr::Str(..) => Some(CT::Text),
            Expr::Float(..) => Some(CT::F64),
            Expr::Neg(x, _) => self.peek(x),
            Expr::Cast(_, t, _) => cty(t).ok(),
            Expr::Bin(op, a, b, _) => match op.as_str() {
                "<" | ">" | "==" => Some(CT::Bool),
                _ => self.peek(a).or_else(|| self.peek(b)),
            },
            Expr::If(_, a, b, _) => self.peek(a).or_else(|| self.peek(b)),
            Expr::Call(f, _, _, _) => match &**f {
                Expr::Name(n, _) => self.funs.iter().find(|g| &g.name == n).map(|g| g.ret.clone()),
                Expr::Member(o, m, _) if method_op(m).is_some() || unary_method(m) => {
                    Some(self.peek(o).unwrap_or(I64))
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn atom(e: &Expr) -> bool {
        matches!(e, Expr::Name(..) | Expr::Int(..) | Expr::Bool(..) | Expr::Str(..) | Expr::Float(..))
    }

    fn expr(&self, e: &Expr, want: Option<&CT>) -> Result<(Value, CT), String> {
        match e {
            Expr::Name(n, _) => {
                let i = self.env.iter().rev().position(|(m, _)| m == n).ok_or(format!("name `{n}`"))?;
                Ok((json!(["var", i]), self.env[self.env.len() - 1 - i].1.clone()))
            }
            Expr::Int(n, _) => {
                let t = match want {
                    Some(t @ CT::Int(..)) => t.clone(),
                    _ => I64,
                };
                let v = num(*n);
                if v.is_null() {
                    return Err("literal outside i64/u64".into());
                }
                Ok((json!(["int", v]), t))
            }
            Expr::Bool(b, _) => Ok((json!(["bool", b]), CT::Bool)),
            // S3: a float literal is its binary64 pattern
            Expr::Float(f, _) if want == Some(&CT::F32) => {
                let r = *f as f32;
                if !r.is_finite() {
                    return Err("f32 literal out of range".into());
                }
                Ok((json!(["f32", (r as f64).to_bits()]), CT::F32))
            }
            Expr::Float(f, _) => Ok((json!(["f64", f.to_bits()]), CT::F64)),
            Expr::Cast(x, ty, _) => {
                let to = cty(ty)?;
                let src = self.peek(x).ok_or("cast of an untyped operand")?;
                let (xj, st) = self.expr(x, Some(&src))?;
                match (&st, &to) {
                    (CT::F32, CT::Int(lo, hi)) => {
                        let (l, h) = (num(*lo), num(*hi));
                        Ok((json!(["un", ["trunc32", l, h], "f32", xj]), to.clone()))
                    }
                    (CT::Int(..) | CT::F64, CT::Int(lo, hi)) => {
                        let (l, h) = (num(*lo), num(*hi));
                        Ok((json!(["cast", l, h, xj]), to.clone()))
                    }
                    (CT::Int(..), CT::F64) => Ok((json!(["tof", xj]), CT::F64)),
                    (CT::Int(..) | CT::F64, CT::F32) => Ok((json!(["un", "tof32", tj(&st), xj]), CT::F32)),
                    (CT::F32, CT::F64) => Ok((json!(["un", "widen", "f32", xj]), CT::F64)),
                    _ => Err("cast outside the core subset".into()),
                }
            }
            Expr::Str(s, _) => Ok((json!(["text", s]), CT::Text)),
            // `-x`: operand, then negation checked into the result type (interp.rs Neg)
            Expr::Neg(x, _) => {
                let t = want.cloned().or_else(|| self.peek(x)).unwrap_or(I64);
                match (&t, &**x) {
                    (CT::Int(lo, hi), Expr::Int(n, _)) => {
                        // a literal keeps its own range, so `-9223372036854775808` is well typed
                        let v = num(*n);
                        if v.is_null() {
                            return Err("literal outside i64/u64".into());
                        }
                        Ok((json!(["un", ["ineg", num(*lo), num(*hi)], ["int", 0, v], ["int", v]]), t.clone()))
                    }
                    (CT::Int(lo, hi), _) => {
                        let (xj, xt) = self.expr(x, Some(&t))?;
                        Ok((json!(["un", ["ineg", num(*lo), num(*hi)], tj(&xt), xj]), t.clone()))
                    }
                    (CT::F64 | CT::F32, _) => {
                        let (xj, _) = self.expr(x, Some(&t))?;
                        Ok((json!(["un", "fneg", tj(&t), xj]), t.clone()))
                    }
                    _ => Err("negation of a non-number".into()),
                }
            }
            Expr::If(c, a, b, _) => {
                let (cj, _) = self.expr(c, Some(&CT::Bool))?;
                let w = want.cloned().or_else(|| self.peek(a)).or_else(|| self.peek(b));
                let (aj, t) = self.expr(a, w.as_ref())?;
                let (bj, _) = self.expr(b, Some(&t))?;
                Ok((json!(["ite", cj, aj, bj]), t))
            }
            Expr::Bin(op, a, b, _) => {
                let (core, swap, cmp) = match op.as_str() {
                    "+" => ("add", false, false),
                    "-" => ("sub", false, false),
                    "*" => ("mul", false, false),
                    "/" => ("div", false, false),
                    "%" => ("rem", false, false),
                    "&" => ("band", false, false),
                    "|" => ("bor", false, false),
                    "^" => ("bxor", false, false),
                    "<<" => ("shl", false, false),
                    ">>" => ("shr", false, false),
                    "<" => ("lt", false, true),
                    ">" => ("lt", true, true),
                    "==" => ("eq", false, true),
                    o => return Err(format!("operator `{o}`")),
                };
                // `a > b` is `b < a`; exact only when operand order cannot be observed
                if swap && !(Self::atom(a) && Self::atom(b)) {
                    return Err("`>` with compound operands".into());
                }
                let w = if cmp { None } else { want.cloned() };
                let ot = self.peek(a).or_else(|| self.peek(b)).or(w).unwrap_or(I64);
                if !cmp && ot == CT::Text && core == "add" {
                    let (aj, _) = self.expr(a, Some(&ot))?;
                    let (bj, _) = self.expr(b, Some(&ot))?;
                    return Ok((json!(["bin", "cat", "text", aj, bj]), CT::Text));
                }
                let fop = matches!(ot, CT::F64 | CT::F32) && matches!(core, "add" | "sub" | "mul" | "div" | "lt");
                if !matches!(ot, CT::Int(..)) && core != "eq" && !fop {
                    return Err(format!("`{op}` on a non-integer"));
                }
                let (aj, _) = self.expr(a, Some(&ot))?;
                let (bj, _) = self.expr(b, Some(&ot))?;
                let (x, y) = if swap { (bj, aj) } else { (aj, bj) };
                let res = if cmp { CT::Bool } else { ot.clone() };
                Ok((json!(["bin", core, tj(&ot), x, y]), res))
            }
            Expr::Call(f, args, _, false)
                if matches!(&**f, Expr::Member(_, m, _) if unary_method(m)) && args.is_empty() =>
            {
                let Expr::Member(o, m, _) = &**f else { unreachable!() };
                let rt = self.peek(o).or(want.cloned()).unwrap_or(I64);
                let op = match (&rt, m.as_str()) {
                    (CT::Int(..), "abs") => "iabs",
                    (CT::F64 | CT::F32, "abs") => "fabs",
                    (CT::F64, "sqrt" | "floor" | "ceil" | "round") => m.as_str(),
                    _ => return Err(format!("method `{m}` on this type")),
                };
                let (oj, _) = self.expr(o, Some(&rt))?;
                Ok((json!(["un", op, tj(&rt), oj]), rt))
            }
            // `try x.checked_op(y) else E` (6.3 N3): the exact result in a wide
            // range, `inr` when it fits the receiver's type, else error E
            Expr::Try(x, Some(eb), _) => {
                let Expr::Name(en, _) = &**eb else { return Err("try else of a non-error".into()) };
                let code = self.errs.iter().position(|e| e == en).ok_or(format!("try else `{en}`"))? + 1;
                let Expr::Call(f, args, _, false) = &**x else { return Err("try of a non-call".into()) };
                let Expr::Member(o, m, _) = &**f else { return Err("try of a non-method".into()) };
                let op = match m.as_str() {
                    "checked_add" => "add",
                    "checked_sub" => "sub",
                    "checked_mul" => "mul",
                    _ => return Err(format!("try of method `{m}`")),
                };
                if args.len() != 1 || args[0].label.is_some() {
                    return Err(format!("method `{m}` arity"));
                }
                let rt = self.peek(o).or(want.cloned()).unwrap_or(I64);
                let CT::Int(lo, hi) = rt else { return Err(format!("method `{m}` on this type")) };
                let (oj, _) = self.expr(o, Some(&rt))?;
                let (aj, _) = self.expr(&args[0].e, Some(&rt))?;
                let (wl, wh) = ((-WIDE).to_string(), WIDE.to_string());
                let w = json!(["int", wl, wh]);
                let v = json!(["var", 0]);
                let none = json!(["inl", ["unit"]]);
                let sum = json!(["let", ["bin", op, w, ["cast", wl, wh, oj], ["cast", wl, wh, aj]],
                    ["ite", ["bin", "lt", w, v, ["int", num(lo)]], none,
                        ["ite", ["bin", "lt", w, ["int", num(hi)], v], none,
                            ["inr", ["cast", num(lo), num(hi), v]]]]]);
                Ok((json!(["try", sum, code]), rt))
            }
            Expr::Call(f, args, _, false) if matches!(&**f, Expr::Member(..)) => {
                let Expr::Member(o, m, _) = &**f else { unreachable!() };
                let op = method_op(m).ok_or(format!("method `{m}` outside the core subset"))?;
                if args.len() != 1 || args[0].label.is_some() {
                    return Err(format!("method `{m}` arity"));
                }
                // receiver then argument, one tick for the call (interp.rs Member)
                let rt = self.peek(o).or(want.cloned()).unwrap_or(I64);
                let int_only = !matches!(op, "min" | "max");
                match &rt {
                    CT::Int(..) => {}
                    CT::F64 if !int_only => {}
                    _ => return Err(format!("method `{m}` on this type")),
                }
                let (oj, _) = self.expr(o, Some(&rt))?;
                let (aj, _) = self.expr(&args[0].e, Some(&rt))?;
                Ok((json!(["bin", op, tj(&rt), oj, aj]), rt))
            }
            Expr::Call(f, args, _, false) => {
                let Expr::Name(n, _) = &**f else { return Err("call of a non-name".into()) };
                let (i, g) = self.funs.iter().enumerate().find(|(_, g)| &g.name == n)
                    .ok_or(format!("call of `{n}`"))?;
                if args.len() != 1 || args[0].label.is_some() {
                    return Err("call arity".into());
                }
                let (aj, _) = self.expr(&args[0].e, Some(&g.arg))?;
                Ok((json!(["call", i, aj]), g.ret.clone()))
            }
            _ => Err(format!("expression outside the core subset: {}", head(e))),
        }
    }
}

impl Cx<'_> {
    /// A statement list in its own scope (interp.rs `exec_block_tail`).
    fn block(&mut self, ss: &[Stmt]) -> Result<Vec<Value>, String> {
        let n = self.env.len();
        let r = ss.iter().map(|s| self.stmt(s)).collect::<Result<Vec<_>, _>>();
        self.env.truncate(n);
        r
    }

    fn stmt(&mut self, s: &Stmt) -> Result<Value, String> {
        match s {
            Stmt::Let(Pat::Path(p), ann, e, _) if p.len() == 1 => {
                let want = ann.as_ref().map(cty).transpose()?;
                let (j, t) = self.expr(e, want.as_ref())?;
                let tyj = tj(&t);
                self.env.push((p[0].clone(), t));
                Ok(json!(["let", j, tyj]))
            }
            // `var x: T = e`: an integer variable needs its type, as the
            // reference checks assignments against the declared type only
            Stmt::Var(n, ann, e, _) => {
                let want = ann.as_ref().map(cty).transpose()?;
                let (j, t) = self.expr(e, want.as_ref())?;
                if matches!(t, CT::Int(..)) && want.is_none() {
                    return Err("untyped integer var".into());
                }
                let tyj = tj(&t);
                self.env.push((n.clone(), t));
                Ok(json!(["let", j, tyj]))
            }
            Stmt::Emit(e) => {
                let (j, t) = self.expr(e, None)?;
                Ok(json!(["emit", j, tj(&t)]))
            }
            Stmt::Assign(op, Expr::Name(x, _), rhs) => {
                let i = self.env.iter().rev().position(|(m, _)| m == x).ok_or(format!("name `{x}`"))?;
                let t = self.env[self.env.len() - 1 - i].1.clone();
                if op == "=" {
                    let (j, _) = self.expr(rhs, Some(&t))?;
                    let rng = match &t {
                        CT::Int(lo, hi) => json!([num(*lo), num(*hi)]),
                        _ => Value::Null,
                    };
                    return Ok(json!(["set", i, rng, j]));
                }
                let core = match op.as_str() {
                    "+=" => "add",
                    "-=" => "sub",
                    "*=" => "mul",
                    "/=" => "div",
                    "%=" => "rem",
                    o => return Err(format!("assignment `{o}`")),
                };
                let fop = matches!(t, CT::F64 | CT::F32) && core != "rem";
                if !matches!(t, CT::Int(..)) && !fop {
                    return Err(format!("`{op}` on this type"));
                }
                let (j, _) = self.expr(rhs, Some(&t))?;
                Ok(json!(["setop", i, core, tj(&t), j]))
            }
            Stmt::If(arms, els) => {
                let mut aj = vec![];
                for (c, b) in arms {
                    let (cj, _) = self.expr(c, Some(&CT::Bool))?;
                    aj.push(json!([cj, self.block(b)?]));
                }
                let ej = match els {
                    Some(b) => self.block(b)?,
                    None => vec![],
                };
                Ok(json!(["if", aj, ej]))
            }
            Stmt::While(c, None, body) => {
                let (cj, _) = self.expr(c, Some(&CT::Bool))?;
                Ok(json!(["while", cj, crate::interp::LOOP_LIMIT as u64, self.block(body)?]))
            }
            Stmt::Arena(m, None, body, _) if m == "$forpar" => self.forpar(body),
            Stmt::Fail(Expr::Name(en, _)) if self.errs.contains(en) => {
                let code = self.errs.iter().position(|e| e == en).unwrap() + 1;
                Ok(json!(["let", ["raise", code], "unit"]))
            }
            // `fail E(f: e, ..)`: the payload is evaluated (it may trap), then
            // the error keeps the state; only its name is observable (T8)
            Stmt::Fail(Expr::Call(f, args, _, false))
                if matches!(&**f, Expr::Name(en, _) if self.errs.contains(en)) =>
            {
                let Expr::Name(en, _) = &**f else { unreachable!() };
                let code = self.errs.iter().position(|e| e == en).unwrap() + 1;
                let mut p = json!(["unit"]);
                for a in args.iter().rev() {
                    let (aj, _) = self.expr(&a.e, None)?;
                    p = json!(["pair", aj, p]);
                }
                Ok(json!(["let", ["let", p, ["raise", code]], "unit"]))
            }
            Stmt::Break(_) => Ok(json!(["break"])),
            Stmt::Continue(_) => Ok(json!(["continue"])),
            _ => Err(format!("statement outside the core subset: {}", head(s))),
        }
    }
}

impl Cx<'_> {
    /// `for x in a..b par` with one `+=`/`*=` reduction target that is an
    /// outer integer or f64 variable (conc.rs `forpar_exec`, C.8 tree).
    fn forpar(&mut self, body: &[Stmt]) -> Result<Value, String> {
        let [Stmt::For(Pat::Path(p), Expr::Bin(rop, a, b, _), fb, _)] = body else {
            return Err("for par shape".into());
        };
        if p.len() != 1 || rop != ".." {
            return Err("for par shape".into());
        }
        let mut targets = vec![];
        crate::conc::assign_ops(fb, &mut targets);
        let local = crate::conc::declared_in(fb);
        targets.retain(|(n, _)| !local.contains(n));
        let [(x, op)] = &targets[..] else { return Err("for par with other than one target".into()) };
        let i = self.env.iter().rev().position(|(m, _)| m == x).ok_or(format!("name `{x}`"))?;
        let t = self.env[self.env.len() - 1 - i].1.clone();
        let (core, idv) = match (op.as_str(), &t) {
            ("+=", CT::Int(..)) => ("add", json!(["int", 0])),
            ("*=", CT::Int(..)) => ("mul", json!(["int", 1])),
            ("+=", CT::F64) => ("add", json!(["f64", 0x8000_0000_0000_0000u64])),
            ("*=", CT::F64) => ("mul", json!(["f64", 1.0f64.to_bits()])),
            _ => return Err(format!("for par `{op}` on this type")),
        };
        let (aj, _) = self.expr(a, Some(&I64))?;
        let (bj, _) = self.expr(b, Some(&I64))?;
        self.env.push((p[0].clone(), I64));
        let r = self.block(fb);
        self.env.pop();
        Ok(json!(["forpar", aj, bj, tj(&I64), crate::interp::LOOP_LIMIT as u64, i, core, tj(&t), idv, r?]))
    }
}

fn unary_method(m: &str) -> bool {
    matches!(m, "abs" | "sqrt" | "floor" | "ceil" | "round")
}

/// The variant name of a syntax node (for the unsupported reason).
fn head<T: std::fmt::Debug>(x: &T) -> String {
    format!("{x:?}").split(|c: char| !c.is_alphanumeric()).next().unwrap_or("").to_string()
}

/// Binary methods the core models (book 6.3 N2, N5).
fn method_op(m: &str) -> Option<&'static str> {
    Some(match m {
        "min" => "min",
        "max" => "max",
        "saturating_add" => "sadd",
        "saturating_sub" => "ssub",
        "saturating_mul" => "smul",
        "wrapping_add" => "wadd",
        "wrapping_sub" => "wsub",
        "wrapping_mul" => "wmul",
        _ => return None,
    })
}

/// S4: elaborate an event run of `m` over `events` (the syntax of an
/// `example` line) into regions, steps and event tags for Events.lean.
/// Returns the JSON and, for display, the region names with their field
/// names, and the event names in order.
pub fn elaborate_events(m: &Module, src: &str, events: &str, fuel: u64)
    -> Result<(Value, Vec<(String, Vec<String>)>, Vec<String>, Vec<String>), String> {
    // the third component is each event of the run as the reference shows it
    let mut sigs = vec![];
    let mut decls = vec![];
    let (mut states, mut evs, mut errs, mut steps) = (vec![], vec![], vec![], vec![]);
    let mut epay: Vec<Vec<(String, CT)>> = vec![];
    for it in &m.items {
        match it {
            Item::Fn(d) if d.sig.name == "main" => {}
            Item::Fn(d) => {
                if d.sig.comptime || !d.sig.generics.is_empty() || d.sig.bound.is_some() {
                    return Err(format!("fn `{}` modifiers", d.sig.name));
                }
                let [Param::Named { comptime: false, name, ty, .. }] = &d.sig.params[..] else {
                    return Err(format!("fn `{}` has other than one parameter", d.sig.name));
                };
                let ret = cty(d.sig.ret.as_ref().ok_or("unit fn")?)?;
                sigs.push(Fun { name: d.sig.name.clone(), arg: cty(ty)?, ret });
                decls.push((name.clone(), d));
            }
            Item::State { name, fields, replicated: false, .. } => states.push((name, fields)),
            Item::Event { name, payload, interrupt: None, .. } => {
                let p = payload.iter().map(|(n, t)| Ok((n.clone(), cty(t)?))).collect::<Result<Vec<_>, String>>()?;
                evs.push(name.clone());
                epay.push(p);
            }
            Item::Error { name, .. } => errs.push(name.clone()),
            Item::Step { region, pat, guard, body, deadline: None, .. } => steps.push((region, pat, guard, body)),
            Item::Example(_) | Item::Check(_) | Item::Intent(..) => {}
            _ => return Err(format!("item outside the core subset: {}", head(it))),
        }
    }
    let mut fns = vec![];
    for ((pname, d), g) in decls.iter().zip(&sigs) {
        let Body::Arrow(e) = &d.body else { return Err(format!("fn `{}` block body", g.name)) };
        let cx = Cx { funs: &sigs, env: vec![(pname.clone(), g.arg.clone())], errs: errs.clone() };
        let (bj, _) = cx.expr(e, Some(&g.ret))?;
        fns.push(json!({"arg": tj(&g.arg), "ret": tj(&g.ret), "body": bj}));
    }
    let probe = crate::parse::parse(&format!("{src}\nexample {events} -> 0\n")).map_err(|e| format!("events: {e}"))?;
    let Some(Item::Example(x)) = probe.items.last() else { return Err("events: not an event list".into()) };
    let mut tags = vec![];
    let mut shown = vec![];
    let mut it = crate::interp::Interp::new(m);
    for (n, args) in &x.events {
        it.scopes = vec![std::collections::HashMap::new()];
        match it.construct(n, args) {
            Ok(Some(v)) => shown.push(v.show()),
            _ => return Err(format!("event `{n}`")),
        }
        let k = evs.iter().position(|e| e == n).ok_or(format!("event `{n}`"))?;
        if args.len() != epay[k].len() {
            return Err("event arguments".into());
        }
        let mut pj = vec![];
        for (a, (pn, pt)) in args.iter().zip(&epay[k]) {
            if a.label.as_ref().is_some_and(|l| l != pn) {
                return Err("event argument order".into());
            }
            pj.push(Cx { funs: &sigs, env: vec![], errs: vec![] }.expr(&a.e, Some(pt))?.0);
        }
        tags.push(json!([k, pj]));
    }
    let mut regions = vec![];
    let mut names = vec![];
    for (rname, fields) in &states {
        let mut cx = Cx { funs: &sigs, env: vec![], errs: errs.clone() };
        let mut init = vec![];
        for f in fields.iter() {
            let t = cty(&f.ty)?;
            let e = f.init.as_ref().ok_or("field without initializer")?;
            let (j, _) = Cx { funs: &sigs, env: vec![], errs: vec![] }.expr(e, Some(&t))?;
            init.push(j);
            cx.env.push((f.name.clone(), t));
        }
        init.reverse(); // environment order: the last field is index 0
        let mut sj = vec![];
        for (r, pat, guard, body) in &steps {
            if r != rname {
                continue;
            }
            // the payload is bound above the fields: a named sub-pattern binds
            // its field, others get a name no source can spell
            let (p, subs) = match pat {
                Pat::Path(p) => (p, &vec![]),
                Pat::Ctor(p, subs) => (p, subs),
                _ => return Err("step pattern outside the core subset".into()),
            };
            let [en] = &p[..] else { return Err("step pattern outside the core subset".into()) };
            let tag = evs.iter().position(|e| e == en).ok_or(format!("step on `{en}`"))?;
            let nf = cx.env.len();
            for (j, (pn, pt)) in epay[tag].iter().enumerate() {
                let sub = subs.iter().find(|(l, _)| l.as_deref() == Some(pn.as_str()))
                    .or_else(|| if subs.iter().all(|(l, _)| l.is_none()) { subs.get(j) } else { None });
                let bind = match sub.map(|x| &x.1) {
                    None | Some(Pat::Wild) => format!("${pn}"),
                    Some(Pat::Path(q)) if q.len() == 1 => q[0].clone(),
                    _ => return Err("payload pattern outside the core subset".into()),
                };
                cx.env.push((bind, pt.clone()));
            }
            let gj = match guard {
                Some(g) => cx.expr(g, Some(&CT::Bool))?.0,
                None => Value::Null,
            };
            let bj = match body {
                Body::Arrow(e) => {
                    let f0 = Expr::Name(fields[0].name.clone(), 0);
                    vec![cx.stmt(&Stmt::Assign("=".into(), f0, e.clone()))?]
                }
                Body::Block(b) => cx.block(b)?,
            };
            cx.env.truncate(nf);
            sj.push(json!([tag, gj, bj]));
        }
        regions.push(json!({"init": init, "steps": sj}));
        names.push((rname.to_string(), fields.iter().map(|f| f.name.clone()).collect()));
    }
    Ok((json!({"fns": fns, "regions": regions, "events": tags, "fuel": fuel}), names, shown, errs))
}

/// Elaborate `m` with run budget `fuel`.
pub fn elaborate(m: &Module, fuel: u64) -> Result<Value, String> {
    if let Some(e) = crate::v09::unemittable_for(m, true) {
        return Err(e);
    }
    let mut sigs = vec![];
    let mut decls = vec![];
    let mut main = None;
    for it in &m.items {
        match it {
            Item::Fn(d) if d.sig.name == "main" => main = Some(d),
            Item::Fn(d) => {
                if d.sig.comptime || !d.sig.generics.is_empty() || d.sig.bound.is_some() {
                    return Err(format!("fn `{}` modifiers", d.sig.name));
                }
                let [Param::Named { comptime: false, name, ty, .. }] = &d.sig.params[..] else {
                    return Err(format!("fn `{}` has other than one parameter", d.sig.name));
                };
                let ret = cty(d.sig.ret.as_ref().ok_or("unit fn")?)?;
                sigs.push(Fun { name: d.sig.name.clone(), arg: cty(ty)?, ret });
                decls.push((name.clone(), d));
            }
            Item::Example(_) | Item::Check(_) | Item::Intent(..) => {}
            _ => return Err("item outside the core subset".into()),
        }
    }
    let main = main.ok_or("no fn main")?;
    if !main.sig.params.is_empty() || main.sig.ret.is_some() {
        return Err("fn main signature".into());
    }
    let mut fns = vec![];
    for ((pname, d), g) in decls.iter().zip(&sigs) {
        let Body::Arrow(e) = &d.body else { return Err(format!("fn `{}` block body", g.name)) };
        let cx = Cx { funs: &sigs, env: vec![(pname.clone(), g.arg.clone())], errs: vec![] };
        let (bj, _) = cx.expr(e, Some(&g.ret))?;
        fns.push(json!({"arg": tj(&g.arg), "ret": tj(&g.ret), "body": bj}));
    }
    let Body::Block(ss) = &main.body else { return Err("fn main arrow body".into()) };
    let mut cx = Cx { funs: &sigs, env: vec![], errs: vec![] };
    let out = cx.block(ss)?;
    Ok(json!({"fns": fns, "main": out, "fuel": fuel}))
}
