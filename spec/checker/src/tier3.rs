//! Version 0.9 Tier 3 (Appendix F.6): contracts `requires` and `ensures`
//! (P2) and `ghost` (P3). `check` runs on the module as written and reports
//! W-CON-1, W-CON-2 and W-GHOST-1; `lower` rewrites the module for the
//! evaluator: a clause becomes an ordinary guard that traps with the fixed
//! record of F.6.2, so a contract is *always enforced*. A ghost value is
//! kept only while a clause (transitively) names it, because an enforced
//! clause has to be evaluated; no other ghost code reaches the run.
use crate::ast::*;
use std::collections::{BTreeSet, HashMap};

fn err(l: usize, code: &str, m: String) -> String {
    format!("line {l}: {code} {m}")
}

pub(crate) fn clause(s: &Stmt) -> Option<(&'static str, &Expr)> {
    if let Stmt::Check(Expr::Call(c, a, _, _)) = s
        && let Expr::Name(n, _) = &**c
        && a.len() == 1
    {
        match n.as_str() {
            "@requires" => return Some(("requires", &a[0].e)),
            "@ensures" => return Some(("ensures", &a[0].e)),
            _ => {}
        }
    }
    None
}

pub(crate) fn ghost_let(s: &Stmt) -> Option<(&Pat, &Expr)> {
    if let Stmt::Let(p, _, Expr::Call(c, a, _, _), _) = s
        && matches!(&**c, Expr::Name(n, _) if n == "@ghost")
        && a.len() == 1
    {
        return Some((p, &a[0].e));
    }
    None
}

pub(crate) fn measure(s: &Stmt) -> Option<Vec<&Expr>> {
    if let Stmt::Check(Expr::Call(c, a, _, _)) = s
        && matches!(&**c, Expr::Name(n, _) if n == "@decreases")
        && !a.is_empty()
    {
        return Some(a.iter().map(|x| &x.e).collect());
    }
    None
}

fn prelude_len(b: &[Stmt]) -> usize {
    b.iter().take_while(|s| clause(s).is_some() || ghost_let(s).is_some() || measure(s).is_some()).count()
}

fn names_in(e: &Expr, out: &mut BTreeSet<String>) {
    crate::v09::each_expr(e, false, &mut |x, _| {
        if let Expr::Name(n, _) = x {
            out.insert(n.clone());
        }
    });
}

fn all_stmts<'a>(ss: &'a [Stmt], f: &mut dyn FnMut(&'a Stmt)) {
    for s in ss {
        f(s);
        for b in crate::v09::stmt_blocks(s) {
            all_stmts(b, f);
        }
    }
}

/// Integer bounds of the atoms of a conjunction, or `Some(false)` when an
/// atom is false by itself.
fn atoms(e: &Expr, out: &mut Vec<(String, &'static str, i128)>) -> bool {
    fn lit(e: &Expr) -> Option<i128> {
        match e {
            Expr::Int(v, _) => Some(*v),
            Expr::Neg(x, _) => lit(x).map(|v| -v),
            _ => None,
        }
    }
    fn cmp(op: &str, a: i128, b: i128) -> Option<bool> {
        Some(match op {
            "<" => a < b,
            "<=" => a <= b,
            ">" => a > b,
            ">=" => a >= b,
            "==" => a == b,
            "!=" => a != b,
            _ => return None,
        })
    }
    match e {
        Expr::Bool(false, _) => false,
        Expr::Not(x, _) if matches!(&**x, Expr::Bool(true, _)) => false,
        Expr::Bin(op, a, b, _) if op == "and" => atoms(a, out) && atoms(b, out),
        Expr::Bin(op, a, b, _) => {
            if let (Some(x), Some(y)) = (lit(a), lit(b)) {
                return cmp(op, x, y).unwrap_or(true);
            }
            let flip = |o: &str| match o {
                "<" => ">",
                "<=" => ">=",
                ">" => "<",
                ">=" => "<=",
                x => if x == "==" { "==" } else { "!=" },
            };
            let o: Option<&'static str> = ["<", "<=", ">", ">=", "=="].into_iter().find(|k| *k == op.as_str());
            if let Some(o) = o {
                if let (Expr::Name(n, _), Some(v)) = (&**a, lit(b)) {
                    out.push((n.clone(), o, v));
                } else if let (Some(v), Expr::Name(n, _)) = (lit(a), &**b) {
                    let f = flip(o);
                    let f: &'static str = ["<", "<=", ">", ">=", "=="].into_iter().find(|k| *k == f).unwrap_or("==");
                    out.push((n.clone(), f, v));
                }
            }
            true
        }
        _ => true,
    }
}

fn contradictory(reqs: &[&Expr]) -> bool {
    let mut at = vec![];
    for e in reqs {
        if !atoms(e, &mut at) {
            return true;
        }
    }
    let mut by: HashMap<&str, (Option<i128>, Option<i128>)> = HashMap::new();
    for (n, op, v) in &at {
        let e = by.entry(n.as_str()).or_default();
        let lo = |e: &mut (Option<i128>, Option<i128>), x: i128| e.0 = Some(e.0.map_or(x, |y| y.max(x)));
        let hi = |e: &mut (Option<i128>, Option<i128>), x: i128| e.1 = Some(e.1.map_or(x, |y| y.min(x)));
        match *op {
            ">" => lo(e, v + 1),
            ">=" => lo(e, *v),
            "<" => hi(e, v - 1),
            "<=" => hi(e, *v),
            _ => {
                lo(e, *v);
                hi(e, *v);
            }
        }
    }
    by.values().any(|(l, h)| matches!((l, h), (Some(l), Some(h)) if l > h))
}

fn binds_result(f: &FnDecl) -> bool {
    if f.sig.params.iter().any(|p| matches!(p, Param::Named { name, .. } if name == "result")) {
        return true;
    }
    let mut found = false;
    if let Body::Block(b) = &f.body {
        all_stmts(b, &mut |s| {
            let mut n = BTreeSet::new();
            match s {
                Stmt::Let(p, ..) => crate::check::pat_names_pub(p, &mut n),
                Stmt::For(p, ..) => crate::check::pat_names_pub(p, &mut n),
                Stmt::Var(x, ..) => {
                    n.insert(x.clone());
                }
                _ => {}
            }
            if n.contains("result") {
                found = true;
            }
        });
    }
    found
}

/// W-CON-1, W-CON-2 (F.6.2) and W-GHOST-1 (F.6.3).
pub fn check(m: &Module) -> Vec<String> {
    let mut errs = vec![];
    let ghost_fns: BTreeSet<String> = crate::v09::fn_decls(m)
        .iter()
        .filter(|(f, _)| f.sig.x09 & 2 != 0)
        .map(|(f, _)| f.sig.name.clone())
        .collect();
    for (f, _) in crate::v09::fn_decls(m) {
        let Body::Block(b) = &f.body else { continue };
        let lead = prelude_len(b);
        // a loop measure is the first statement of the body of a `while` (F.6.4)
        let mut allowed: BTreeSet<*const Stmt> = BTreeSet::new();
        all_stmts(b, &mut |s| {
            if let Stmt::While(_, _, body) = s
                && body.first().is_some_and(|x| measure(x).is_some())
            {
                allowed.insert(&body[0] as *const Stmt);
            }
        });
        // a clause is written only at the start of a fn body
        for s in &b[lead..] {
            let mut stray = |s: &Stmt| {
                if measure(s).is_some() && !allowed.contains(&(s as *const Stmt)) {
                    errs.push(err(
                        stmt_line(s),
                        "W-RES-1",
                        "`decreases` is written only before the first statement of a fn body or as the first statement of the body of a `while` in this checker; a measure on another loop (`for`, `loop`) or elsewhere is specified (F.6.4) but not implemented, so it is reserved, not implemented".into(),
                    ));
                }
                if clause(s).is_some() {
                    errs.push(err(
                        stmt_line(s),
                        "W-CON-2",
                        "a `requires` or `ensures` clause is written only before the first statement of a fn body".into(),
                    ));
                }
            };
            stray(s);
            for blk in crate::v09::stmt_blocks(s) {
                all_stmts(blk, &mut |x| stray(x));
            }
        }
        let mut reqs: Vec<&Expr> = vec![];
        let mut has_ens = false;
        for s in &b[..lead] {
            if let Some((k, e)) = clause(s) {
                let mut bad = None;
                crate::v09::each_expr(e, false, &mut |x, _| match x {
                    Expr::Recv(..) => bad = Some("a channel receive"),
                    Expr::ChanNew(..) => bad = Some("a channel"),
                    Expr::Name(n, _) if n == "result" && k == "requires" => bad = Some("`result`, which exists only in `ensures`"),
                    _ => {}
                });
                if let Some(w) = bad {
                    errs.push(err(e.line(), "W-CON-2", format!("a contract clause is a pure total bool expression and may not use {w}")));
                }
                if k == "requires" {
                    reqs.push(e);
                } else {
                    has_ens = true;
                }
            }
        }
        if !reqs.is_empty() && contradictory(&reqs) {
            errs.push(err(f.sig.line, "W-CON-1", format!("the `requires` clauses of `{}` are false for every argument", f.sig.name)));
        }
        if has_ens && binds_result(f) {
            errs.push(err(f.sig.line, "W-CON-2", format!("`result` is reserved in `ensures` and `{}` binds it", f.sig.name)));
        }
        // W-GHOST-1
        if f.sig.x09 & 2 != 0 {
            continue;
        }
        let mut g: BTreeSet<String> = ghost_fns.clone();
        all_stmts(b, &mut |s| {
            if let Some((p, _)) = ghost_let(s) {
                crate::check::pat_names_pub(p, &mut g);
            }
        });
        if g.is_empty() {
            continue;
        }
        let mut scan = |ss: &[Stmt]| {
            all_stmts(ss, &mut |s| {
                if clause(s).is_some() || ghost_let(s).is_some() {
                    return;
                }
                for e in crate::v09::stmt_exprs(s) {
                    crate::v09::each_expr(e, false, &mut |x, _| {
                        if let Expr::Name(n, l) = x
                            && g.contains(n)
                        {
                            errs.push(err(*l, "W-GHOST-1", format!("ghost value `{n}` is used in code that runs; a ghost value may only be named by contracts and other ghost code")));
                        }
                    });
                }
            });
        };
        scan(b);
    }
    errs.extend(decreases(m));
    errs.extend(loop_measures(m));
    // W-IFLET-1: every `and` operand of an `if let` is a comparison
    for (f, _) in crate::v09::fn_decls(m) {
        let Body::Block(b) = &f.body else { continue };
        all_stmts(b, &mut |s| {
            if let Stmt::Match(Expr::Call(c, _, _, _), arms, _) = s
                && matches!(&**c, Expr::Name(n, _) if n == "@iflet")
                && let Some(g) = arms.first().and_then(|a| a.guard.as_ref())
            {
                fn atoms<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
                    match e {
                        Expr::Bin(op, a, b, _) if op == "and" => {
                            atoms(a, out);
                            atoms(b, out);
                        }
                        x => out.push(x),
                    }
                }
                let mut v = vec![];
                atoms(g, &mut v);
                for e in v {
                    if !matches!(e, Expr::Bin(op, ..) if ["==", "!=", "<", "<=", ">", ">="].contains(&op.as_str())) {
                        errs.push(err(e.line(), "W-IFLET-1", "an operand of an `if let` chain after the pattern is a comparison".into()));
                    }
                }
            }
        });
    }
    errs
}

type Lin = (std::collections::BTreeMap<String, i128>, i128);

fn lin(e: &Expr, env: &HashMap<String, Lin>) -> Option<Lin> {
    match e {
        Expr::Int(v, _) => Some((Default::default(), *v)),
        Expr::Name(n, _) => env.get(n).cloned(),
        Expr::Neg(x, _) => lin(x, env).map(|(c, k)| (c.into_iter().map(|(n, v)| (n, -v)).collect(), -k)),
        Expr::Bin(op, a, b, _) if op == "+" || op == "-" => {
            let (mut ca, ka) = lin(a, env)?;
            let (cb, kb) = lin(b, env)?;
            let s = if op == "+" { 1 } else { -1 };
            for (n, v) in cb {
                *ca.entry(n).or_insert(0) += s * v;
            }
            ca.retain(|_, v| *v != 0);
            Some((ca, ka + s * kb))
        }
        Expr::Bin(op, a, b, _) if op == "*" => {
            let (ca, ka) = lin(a, env)?;
            let (cb, kb) = lin(b, env)?;
            if ca.is_empty() {
                Some((cb.into_iter().map(|(n, v)| (n, v * ka)).collect(), kb * ka))
            } else if cb.is_empty() {
                Some((ca.into_iter().map(|(n, v)| (n, v * kb)).collect(), ka * kb))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn unsigned(t: &Ty) -> bool {
    matches!(t, Ty::Path(p, a) if a.is_empty() && p.len() == 1 && matches!(p[0].as_str(), "u8" | "u16" | "u32" | "u64" | "u128"))
}

/// W-DEC-1 (F.6.4), the fragment this checker decides: a measure is a tuple
/// of expressions linear in the unsigned integer parameters of the fn, and
/// every call of a fn of the same recursive group passes a tuple that is
/// strictly smaller in the lexicographic order, shown by a constant
/// difference at the first position where the two differ.
fn decreases(m: &Module) -> Vec<String> {
    let mut errs = vec![];
    let fns: Vec<&FnDecl> = m.items.iter().filter_map(|i| if let Item::Fn(f) = i { Some(f) } else { None }).collect();
    let idx: HashMap<&str, usize> = fns.iter().enumerate().map(|(i, f)| (f.sig.name.as_str(), i)).collect();
    // calls between top-level fns (with the line of each call)
    let mut calls: Vec<Vec<(usize, Vec<Arg>, usize)>> = vec![vec![]; fns.len()];
    for (i, f) in fns.iter().enumerate() {
        let mut visit = |e: &Expr| {
            crate::v09::each_expr(e, false, &mut |x, _| {
                if let Expr::Call(c, a, l, _) = x
                    && let Expr::Name(n, _) = &**c
                    && let Some(&j) = idx.get(n.as_str())
                {
                    calls[i].push((j, a.clone(), *l));
                }
            });
        };
        match &f.body {
            Body::Arrow(e) => visit(e),
            Body::Block(b) => crate::v09::each_stmt_expr(b, &mut |e| visit(e)),
        }
    }
    let reach = |from: usize, to: usize| -> bool {
        let mut seen = vec![false; fns.len()];
        let mut st = vec![from];
        while let Some(x) = st.pop() {
            for (j, _, _) in &calls[x] {
                if *j == to {
                    return true;
                }
                if !seen[*j] {
                    seen[*j] = true;
                    st.push(*j);
                }
            }
        }
        false
    };
    let meas: Vec<Option<Vec<&Expr>>> = fns
        .iter()
        .map(|f| if let Body::Block(b) = &f.body { b.iter().take(prelude_len(b)).find_map(measure) } else { None })
        .collect();
    let env_of = |f: &FnDecl| -> HashMap<String, Lin> {
        f.sig
            .params
            .iter()
            .filter_map(|p| match p {
                Param::Named { name, ty, .. } if unsigned(ty) => Some((name.clone(), ([(name.clone(), 1)].into_iter().collect(), 0))),
                _ => None,
            })
            .collect()
    };
    for (i, f) in fns.iter().enumerate() {
        let Some(mi) = &meas[i] else { continue };
        let env = env_of(f);
        let mut mine: Vec<Lin> = vec![];
        for e in mi {
            match lin(e, &env) {
                Some(l) => mine.push(l),
                None => errs.push(err(e.line(), "W-DEC-1", "a measure is a natural number: an expression linear in the unsigned integer parameters of the fn (other measures are not implemented)".into())),
            }
        }
        if mine.len() != mi.len() {
            continue;
        }
        for (j, args, l) in &calls[i] {
            if !(reach(*j, i) || *j == i) {
                continue;
            }
            let g = fns[*j];
            let Some(mj) = &meas[*j] else {
                errs.push(err(*l, "W-DEC-1", format!("`{}` is in the recursive group of `{}` and has no `decreases`; a group has one measure tuple (F.6.4)", g.sig.name, f.sig.name)));
                continue;
            };
            if mj.len() != mine.len() {
                errs.push(err(*l, "W-DEC-1", format!("the measure of `{}` has {} parts and that of `{}` has {}", g.sig.name, mj.len(), f.sig.name, mine.len())));
                continue;
            }
            // g's parameters bound to the linear forms of the arguments
            let named: Vec<&Param> = g.sig.params.iter().filter(|p| matches!(p, Param::Named { .. })).collect();
            let mut genv: HashMap<String, Lin> = HashMap::new();
            for (p, a) in named.iter().zip(args.iter()) {
                if let Param::Named { name, .. } = p
                    && let Some(v) = lin(&a.e, &env)
                {
                    genv.insert(name.clone(), v);
                }
            }
            let genv_g = env_of(g);
            let mut ok = false;
            let mut decided = true;
            for (k, e) in mj.iter().enumerate() {
                // the measure of g, in g's own names, then with the arguments
                let Some(_) = lin(e, &genv_g) else {
                    decided = false;
                    break;
                };
                let Some((cn, ck)) = lin(e, &genv) else {
                    decided = false;
                    break;
                };
                let (cm, km) = &mine[k];
                let mut d: std::collections::BTreeMap<String, i128> = cm.clone();
                for (n, v) in cn {
                    *d.entry(n).or_insert(0) -= v;
                }
                d.retain(|_, v| *v != 0);
                let dk = km - ck;
                if d.is_empty() && dk == 0 {
                    continue;
                }
                if d.is_empty() && dk > 0 {
                    ok = true;
                }
                break;
            }
            if !decided {
                errs.push(err(*l, "W-DEC-1", format!("the call of `{}` passes an argument outside the fragment (linear in unsigned parameters), so its measure cannot be compared", g.sig.name)));
            } else if !ok {
                errs.push(err(*l, "W-DEC-1", format!("the call of `{}` does not pass a measure strictly smaller than the entry measure of `{}` (lexicographic, shown by a constant difference)", g.sig.name, f.sig.name)));
            }
        }
    }
    errs
}

fn has_loop_measure(b: &[Stmt]) -> bool {
    let mut any = false;
    all_stmts(b, &mut |s| {
        if let Stmt::While(_, _, body) = s
            && body.first().is_some_and(|x| measure(x).is_some())
        {
            any = true;
        }
    });
    any
}

/// A loop measure has no run-time meaning (F.6.4): it is erased with the
/// other ghost code before evaluation and emission.
fn strip_loop_measures(b: &mut Vec<Stmt>) {
    for s in b.iter_mut() {
        if let Stmt::While(_, _, body) = s
            && body.first().is_some_and(|x| measure(x).is_some())
        {
            body.remove(0);
            if body.is_empty() {
                body.push(Stmt::Pass);
            }
        }
        for blk in blocks_mut(s) {
            strip_loop_measures(blk);
        }
    }
}

/// W-DEC-1 for a `while` (F.6.4), the fragment this checker decides. The
/// loop body starts with `decreases m1, m2, ...`, each `mi` linear in the
/// unsigned integer names (parameters, and `var` with a written unsigned
/// type, each bound once in the fn). The body's own assignments at its top
/// level (`x = e`, `x += e`, `x -= e` with `e` linear) give the measure at the
/// end of the body in terms of the entry values, and that tuple must be
/// strictly smaller in the lexicographic order, shown by a constant difference
/// at the first position where the two differ. A measure name that is assigned
/// anywhere else in the body, borrowed mutably, or a `continue` in the body,
/// is outside the fragment, because the end of the body then is not the
/// straight line this rule computes.
fn loop_measures(m: &Module) -> Vec<String> {
    let mut errs = vec![];
    for (f, _) in crate::v09::fn_decls(m) {
        let Body::Block(b) = &f.body else { continue };
        if !has_loop_measure(b) {
            continue;
        }
        let mut kinds: HashMap<String, (usize, bool)> = HashMap::new();
        fn bind(k: &mut HashMap<String, (usize, bool)>, n: &str, u: bool) {
            let e = k.entry(n.to_string()).or_insert((0, true));
            e.0 += 1;
            e.1 &= u;
        }
        for p in &f.sig.params {
            if let Param::Named { name, ty, .. } = p {
                bind(&mut kinds, name, unsigned(ty));
            }
        }
        let mut pats: Vec<&Pat> = vec![];
        all_stmts(b, &mut |s| match s {
            Stmt::Var(n, t, _, _) => bind(&mut kinds, n, t.as_ref().is_some_and(unsigned)),
            Stmt::Let(p, ..) | Stmt::For(p, ..) => pats.push(p),
            Stmt::Match(_, arms, _) => pats.extend(arms.iter().map(|a| &a.pat)),
            Stmt::Select(arms, _) => {
                for a in arms {
                    if let SelArm::Recv(p, ..) = a {
                        pats.push(p);
                    }
                }
            }
            _ => {}
        });
        for p in pats {
            let mut n = BTreeSet::new();
            crate::check::pat_names_pub(p, &mut n);
            for x in n {
                bind(&mut kinds, &x, false);
            }
        }
        crate::v09::each_stmt_expr(b, &mut |e| {
            crate::v09::each_expr(e, true, &mut |x, _| {
                if let Expr::Lambda(ps, ..) = x {
                    for p in ps {
                        if let Param::Named { name, .. } = p {
                            bind(&mut kinds, name, false);
                        }
                    }
                }
            });
        });
        let env: HashMap<String, Lin> = kinds
            .iter()
            .filter(|(_, (c, u))| *c == 1 && *u)
            .map(|(n, _)| (n.clone(), ([(n.clone(), 1)].into_iter().collect(), 0)))
            .collect();
        let mut loops: Vec<&Stmt> = vec![];
        all_stmts(b, &mut |s| {
            if let Stmt::While(_, _, body) = s
                && body.first().is_some_and(|x| measure(x).is_some())
            {
                loops.push(s);
            }
        });
        for s in loops {
            let Stmt::While(_, _, body) = s else { continue };
            let ms = measure(&body[0]).unwrap();
            let line = ms[0].line();
            let bad = |msg: String| err(line, "W-DEC-1", msg);
            let mut mine: Vec<Lin> = vec![];
            for e in &ms {
                match lin(e, &env) {
                    Some(l) => mine.push(l),
                    None => errs.push(bad("a measure is a natural number: an expression linear in unsigned integer parameters and in `var` names with a written unsigned type, each bound once in the fn".into())),
                }
            }
            if mine.len() != ms.len() {
                continue;
            }
            let mvars: BTreeSet<String> = mine.iter().flat_map(|(c, _)| c.keys().cloned()).collect();
            let rest = &body[1..];
            // anything that could change a measure name other than the top-level straight line
            let mut other = false;
            let mut cont = false;
            for (i, st) in rest.iter().enumerate() {
                let top_assign = matches!(st, Stmt::Assign(_, Expr::Name(x, _), _) if mvars.contains(x));
                let mut check_one = |t: &Stmt| {
                    if let Stmt::Assign(_, Expr::Name(x, _), _) = t
                        && mvars.contains(x)
                        && !(top_assign && std::ptr::eq(t, st))
                    {
                        other = true;
                    }
                    if matches!(t, Stmt::Continue(_)) {
                        cont = true;
                    }
                    for e in crate::v09::stmt_exprs(t) {
                        crate::v09::each_expr(e, true, &mut |x, _| {
                            if let Expr::Borrow(true, inner, _) = x {
                                let mut n = BTreeSet::new();
                                names_in(inner, &mut n);
                                if n.iter().any(|y| mvars.contains(y)) {
                                    other = true;
                                }
                            }
                        });
                    }
                };
                check_one(st);
                all_stmts(crate::v09::stmt_blocks(st).concat().as_slice(), &mut |t| check_one(t));
                let _ = i;
            }
            if other || cont {
                errs.push(bad("the body assigns a measure name other than in a top-level `x = e`, `x += e` or `x -= e`, borrows one mutably, or has a `continue`; this checker decides only the straight line (F.6.4)".into()));
                continue;
            }
            // the measure names after the straight line, in terms of the entry values
            let mut cur = env.clone();
            for st in rest {
                if let Stmt::Assign(op, Expr::Name(x, _), rhs) = st
                    && mvars.contains(x)
                {
                    let r = lin(rhs, &cur);
                    let old = cur.get(x).cloned();
                    let new = match (op.as_str(), r, old) {
                        ("=", Some(r), _) => Some(r),
                        ("+=", Some((cr, kr)), Some((mut co, ko))) => {
                            for (n, v) in cr {
                                *co.entry(n).or_insert(0) += v;
                            }
                            co.retain(|_, v| *v != 0);
                            Some((co, ko + kr))
                        }
                        ("-=", Some((cr, kr)), Some((mut co, ko))) => {
                            for (n, v) in cr {
                                *co.entry(n).or_insert(0) -= v;
                            }
                            co.retain(|_, v| *v != 0);
                            Some((co, ko - kr))
                        }
                        _ => None,
                    };
                    match new {
                        Some(n) => {
                            cur.insert(x.clone(), n);
                        }
                        None => {
                            cur.remove(x);
                        }
                    }
                }
            }
            let mut ok = false;
            let mut decided = true;
            for (k, e) in ms.iter().enumerate() {
                let Some((cn, ck)) = lin(e, &cur) else {
                    decided = false;
                    break;
                };
                let (cm, km) = &mine[k];
                let mut d: std::collections::BTreeMap<String, i128> = cm.clone();
                for (n, v) in cn {
                    *d.entry(n).or_insert(0) -= v;
                }
                d.retain(|_, v| *v != 0);
                let dk = km - ck;
                if d.is_empty() && dk == 0 {
                    continue;
                }
                if d.is_empty() && dk > 0 {
                    ok = true;
                }
                break;
            }
            if !decided {
                errs.push(bad("an assignment of the body is outside the fragment (linear in the unsigned names), so the measure at the end of the body cannot be computed".into()));
            } else if !ok {
                errs.push(bad("the measure at the end of the loop body is not strictly smaller than at its start (lexicographic, shown by a constant difference)".into()));
            }
        }
    }
    errs
}

fn stmt_line(s: &Stmt) -> usize {
    crate::v09::stmt_exprs(s).first().map(|e| e.line()).unwrap_or(0)
}

fn blocks_mut(s: &mut Stmt) -> Vec<&mut Vec<Stmt>> {
    match s {
        Stmt::If(arms, els) => {
            let mut v: Vec<&mut Vec<Stmt>> = arms.iter_mut().map(|(_, b)| b).collect();
            if let Some(b) = els {
                v.push(b);
            }
            v
        }
        Stmt::While(_, _, b) | Stmt::Loop(_, b) | Stmt::For(_, _, b, _) | Stmt::Comptime(b) => vec![b],
        Stmt::Arena(_, _, b, _) | Stmt::Device(_, _, b, _) => vec![b],
        Stmt::Match(_, arms, _) => arms.iter_mut().map(|a| &mut a.body).collect(),
        Stmt::Select(arms, _) => arms
            .iter_mut()
            .map(|a| match a {
                SelArm::Recv(_, _, b) | SelArm::Send(_, _, b) | SelArm::After(_, b) | SelArm::Default(b) => b,
            })
            .collect(),
        _ => vec![],
    }
}

fn result_name(l: usize) -> Expr {
    Expr::Name("result".into(), l)
}

fn rewrite_returns(ss: &mut Vec<Stmt>, ens: &[Stmt]) {
    let old = std::mem::take(ss);
    for mut s in old {
        if let Stmt::Return(e, l) = &mut s {
            let l = *l;
            match e.take() {
                Some(v) => {
                    ss.push(Stmt::Let(Pat::Path(vec!["result".into()]), None, v, l));
                    ss.extend(ens.iter().cloned());
                    ss.push(Stmt::Return(Some(result_name(l)), l));
                }
                None => {
                    ss.push(Stmt::Let(Pat::Path(vec!["result".into()]), None, Expr::Unit(l), l));
                    ss.extend(ens.iter().cloned());
                    ss.push(Stmt::Return(None, l));
                }
            }
            continue;
        }
        for b in blocks_mut(&mut s) {
            rewrite_returns(b, ens);
        }
        ss.push(s);
    }
}

fn rewrite_tail(ss: &mut Vec<Stmt>, ens: &[Stmt]) {
    match ss.last_mut() {
        Some(Stmt::Expr(_)) => {
            let Some(Stmt::Expr(e)) = ss.pop() else { return };
            let l = e.line();
            ss.push(Stmt::Let(Pat::Path(vec!["result".into()]), None, e, l));
            ss.extend(ens.iter().cloned());
            ss.push(Stmt::Expr(result_name(l)));
        }
        Some(Stmt::If(..)) | Some(Stmt::Match(..)) => {
            if let Some(s) = ss.last_mut() {
                for b in blocks_mut(s) {
                    rewrite_tail(b, ens);
                }
            }
        }
        _ => {}
    }
}

fn fns_mut(m: &mut Module) -> Vec<&mut FnDecl> {
    let mut v = vec![];
    for it in m.items.iter_mut() {
        match it {
            Item::Fn(f) => v.push(f),
            Item::Impl { members, .. } => members.iter_mut().for_each(|f| v.push(f)),
            Item::Actor { fns, .. } => fns.iter_mut().for_each(|f| v.push(f)),
            _ => {}
        }
    }
    v
}

/// Rewrite contracts to guards and erase the ghost code no clause needs.
pub fn lower(m: Module) -> Result<Module, String> {
    let any = {
        let mut a = false;
        for (f, _) in crate::v09::fn_decls(&m) {
            if f.sig.x09 & 2 != 0 {
                a = true;
            }
            if let Body::Block(b) = &f.body
                && (prelude_len(b) > 0 || has_loop_measure(b))
            {
                a = true;
            }
        }
        a
    };
    if !any {
        return Ok(m);
    }
    let defs: HashMap<String, String> = crate::apidiff::defs(&m).into_iter().map(|d| (d.name, d.hash)).collect();
    let hashes: Vec<String> = crate::v09::fn_decls(&m)
        .iter()
        .map(|(f, member)| {
            if *member {
                crate::apidiff::member_hash(&m, f)
            } else {
                defs.get(&f.sig.name).cloned().unwrap_or_else(|| crate::apidiff::member_hash(&m, f))
            }
        })
        .collect();
    // names that contracts and kept ghost lets mention
    let mut mentioned: BTreeSet<String> = BTreeSet::new();
    for (f, _) in crate::v09::fn_decls(&m) {
        if let Body::Block(b) = &f.body {
            for s in &b[..prelude_len(b)] {
                if let Some((_, e)) = clause(s) {
                    names_in(e, &mut mentioned);
                }
            }
        }
    }
    // fixpoint over ghost lets (per name) and ghost fn bodies
    loop {
        let before = mentioned.len();
        for (f, _) in crate::v09::fn_decls(&m) {
            if f.sig.x09 & 2 != 0 && mentioned.contains(&f.sig.name) {
                match &f.body {
                    Body::Arrow(e) => names_in(e, &mut mentioned),
                    Body::Block(b) => crate::v09::each_stmt_expr(b, &mut |e| names_in(e, &mut mentioned)),
                }
            }
            if let Body::Block(b) = &f.body {
                for s in &b[..prelude_len(b)] {
                    if let Some((p, e)) = ghost_let(s) {
                        let mut n = BTreeSet::new();
                        crate::check::pat_names_pub(p, &mut n);
                        if n.iter().any(|x| mentioned.contains(x)) {
                            names_in(e, &mut mentioned);
                        }
                    }
                }
            }
        }
        if mentioned.len() == before {
            break;
        }
    }
    let mut m = m;
    for (f, h) in fns_mut(&mut m).into_iter().zip(hashes.iter()) {
        lower_fn(f, h, &mentioned);
    }
    m.items.retain(|it| !matches!(it, Item::Fn(f) if f.sig.x09 & 2 != 0 && !mentioned.contains(&f.sig.name)));
    for it in m.items.iter_mut() {
        if let Item::Fn(f) = it {
            f.sig.x09 &= !2;
        }
    }
    let j = serde_json::to_string(&m).map_err(|e| e.to_string())?;
    for w in ["\"@requires\"", "\"@ensures\"", "\"@decreases\"", "\"@ghost\""] {
        if j.contains(w) {
            return Err(format!("W-CON-2 a `requires`, `ensures` or `ghost let` is not in a position where it has a meaning (a clause is written only before the first statement of a fn body, F.6.2)"));
        }
    }
    Ok(m)
}

fn guard(kind: &str, hash: &str, index: usize, cond: Expr) -> Stmt {
    let l = cond.line();
    let rec = format!(r#"{{"clause":"{kind}","function":"{hash}","index":{index},"kind":"contract"}}"#);
    Stmt::Check(Expr::Call(
        Box::new(Expr::Name("@contract".into(), l)),
        vec![Arg { label: None, e: Expr::Str(rec, l) }, Arg { label: None, e: cond }],
        l,
        false,
    ))
}

fn lower_fn(f: &mut FnDecl, hash: &str, mentioned: &BTreeSet<String>) {
    let Body::Block(b) = &mut f.body else { return };
    strip_loop_measures(b);
    let lead = prelude_len(b);
    if lead == 0 {
        return;
    }
    let mut rest = b.split_off(lead);
    let pre = std::mem::take(b);
    let (mut reqs, mut enss, mut ghosts) = (vec![], vec![], vec![]);
    let mut idx = 0;
    for s in pre {
        if let Some((k, e)) = clause(&s) {
            let g = guard(k, hash, idx, e.clone());
            idx += 1;
            if k == "requires" { reqs.push(g) } else { enss.push(g) }
        } else if measure(&s).is_some() {
            // a measure is checked statically and hashed; it has no run-time meaning
        } else if let Stmt::Let(p, t, Expr::Call(_, mut a, _, _), l) = s {
            let mut n = BTreeSet::new();
            crate::check::pat_names_pub(&p, &mut n);
            if n.iter().any(|x| mentioned.contains(x)) {
                ghosts.push(Stmt::Let(p, t, a.remove(0).e, l));
            }
        }
    }
    if !enss.is_empty() {
        rewrite_returns(&mut rest, &enss);
        if f.sig.ret.is_some() {
            rewrite_tail(&mut rest, &enss);
        } else if !matches!(rest.last(), Some(Stmt::Return(..) | Stmt::Fail(_))) {
            rest.push(Stmt::Let(Pat::Path(vec!["result".into()]), None, Expr::Unit(f.sig.line), f.sig.line));
            rest.extend(enss.iter().cloned());
        }
    }
    let mut out = reqs;
    out.extend(ghosts);
    out.extend(rest);
    if out.is_empty() {
        out.push(Stmt::Pass);
    }
    f.body = Body::Block(out);
}
