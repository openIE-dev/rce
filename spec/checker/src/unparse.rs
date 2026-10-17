//! The source text of a module (W2b): the inverse of `parse` on the AST.
//! `parse(unparse(m))` gives `m` again up to source line numbers
//! (tests/form_roundtrip.rs checks this on every example). Draft 0.6
//! constructs that the parser encodes as marker regions (`$par`, `$forpar`,
//! `$deadline`, ...) are printed back as their surface syntax. Every
//! compound subexpression is parenthesized, so no precedence is assumed.
use crate::ast::*;

const IND: &str = "    ";

pub fn unparse(m: &Module) -> String {
    let mut o = String::new();
    if m.v09 {
        o.push_str("# rce-version: 0.9\n");
    } else if m.v08 {
        o.push_str("# rce-version: 0.8\n");
    } else if m.v07 {
        o.push_str("# rce-version: 0.7\n");
    } else if m.draft06 {
        o.push_str("# rce-version: 0.6\n");
    }
    if let Some(p) = &m.path {
        o.push_str(&format!("module {}\n", p.join(".")));
    }
    for it in &m.items {
        let pubd = crate::link::item_name(it).is_some_and(|n| m.pubs.contains(&n))
            && matches!(it, Item::Const { .. } | Item::TypeAlias { .. } | Item::Struct { .. } | Item::Kind { .. } | Item::Trait { .. } | Item::Fn(_) | Item::Actor { .. } | Item::Protocol { .. });
        o.push('\n');
        if pubd {
            o.push_str("pub ");
        }
        item(&mut o, it);
    }
    o
}

pub fn text(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\0' => o.push_str("\\0"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => o.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn float(f: f64) -> String {
    let s = format!("{f:?}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") { s } else { format!("{s}.0") }
}

fn gparams(g: &[GParam]) -> String {
    if g.is_empty() {
        return String::new();
    }
    let v: Vec<String> = g
        .iter()
        .map(|p| {
            let mut s = String::new();
            if p.comptime {
                s.push_str("comptime ");
            }
            s.push_str(&p.name);
            if !p.bounds.is_empty() {
                s.push_str(": ");
                s.push_str(&p.bounds.iter().map(ty).collect::<Vec<_>>().join(" + "));
            }
            s
        })
        .collect();
    format!("[{}]", v.join(", "))
}

pub fn ty(t: &Ty) -> String {
    match t {
        Ty::Path(p, a) if p.len() == 1 && p[0] == "session" && a.len() == 1 => format!("session {}", ty(&a[0])),
        Ty::Path(p, a) if p.len() == 1 && p[0] == "lvar" && a.len() == 1 => format!("lvar {}", base_ty(&a[0])),
        Ty::Path(p, a) => {
            if a.is_empty() {
                p.join(".")
            } else {
                format!("{}[{}]", p.join("."), a.iter().map(ty).collect::<Vec<_>>().join(", "))
            }
        }
        Ty::Unit => "()".into(),
        Ty::Tuple(v) if v.len() == 1 => format!("({},)", ty(&v[0])),
        Ty::Tuple(v) => format!("({})", v.iter().map(ty).collect::<Vec<_>>().join(", ")),
        Ty::Slice(e) => format!("[{}]", ty(e)),
        Ty::Array(e, n) => format!("[{}; {}]", ty(e), expr(n)),
        Ty::Chan(e) => format!("chan {}", base_ty(e)),
        Ty::Iter(e) => format!("iter {}", base_ty(e)),
        Ty::Fn(a, r) => format!("fn({}) -> {}", a.iter().map(ty).collect::<Vec<_>>().join(", "), base_ty(r)),
        Ty::Ref(m, e) => format!("&{}{}", if *m { "mut " } else { "" }, borrow_ty(e)),
        Ty::Opt(e) => format!("{}?", borrow_ty(e)),
        Ty::Union(a, b) => format!("{} ! {}", opt_ty(a), opt_ty(b)),
        Ty::Region(b, r) => format!("{} @{r}", base_ty(b)),
    }
}
fn base_ty(t: &Ty) -> String {
    match t {
        Ty::Ref(..) | Ty::Opt(_) | Ty::Union(..) | Ty::Region(..) => format!("({})", ty(t)),
        Ty::Path(p, a) if p.len() == 1 && (p[0] == "session" || p[0] == "lvar") && a.len() == 1 => format!("({})", ty(t)),
        _ => ty(t),
    }
}
fn borrow_ty(t: &Ty) -> String {
    match t {
        Ty::Opt(_) | Ty::Union(..) => format!("({})", ty(t)),
        _ => ty(t),
    }
}
fn opt_ty(t: &Ty) -> String {
    match t {
        Ty::Union(..) => format!("({})", ty(t)),
        _ => ty(t),
    }
}

fn params(ps: &[Param]) -> String {
    ps.iter()
        .map(|p| match p {
            Param::SelfVal => "self".to_string(),
            Param::SelfRef(m) => format!("&{}self", if *m { "mut " } else { "" }),
            Param::Named { comptime, label, name, ty: t } => {
                format!("{}{}{name}: {}", if *comptime { "comptime " } else { "" }, label.as_ref().map(|l| format!("{l} ")).unwrap_or_default(), ty(t))
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn sig(s: &FnSig) -> String {
    let mut o = format!("{}{}fn {}{}({})", if s.x09 & 1 != 0 { "ct " } else if s.x09 & 2 != 0 { "ghost " } else { "" }, if s.comptime { "comptime " } else { "" }, s.name, gparams(&s.generics), params(&s.params));
    if let Some(r) = &s.ret {
        o.push_str(&format!(" -> {}", ty(r)));
    }
    if let Some(b) = s.bound {
        o.push_str(&format!(" bound {b}"));
    }
    o
}

fn payload(p: &[(String, Ty)]) -> String {
    if p.is_empty() {
        String::new()
    } else {
        format!("({})", p.iter().map(|(n, t)| format!("{n}: {}", ty(t))).collect::<Vec<_>>().join(", "))
    }
}

fn args(a: &[Arg]) -> String {
    a.iter()
        .map(|x| match &x.label {
            Some(l) => format!("{l}: {}", expr(&x.e)),
            None => expr(&x.e),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn is_atom(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Name(..) | Expr::SelfRef(_) | Expr::Str(..) | Expr::Bool(..) | Expr::None(_) | Expr::Unit(_) | Expr::Tuple(..) | Expr::List(..)
            | Expr::ListRepeat(..) | Expr::ListComp(..) | Expr::Map(..) | Expr::ChanNew(..) | Expr::Call(..) | Expr::Index(..)
            | Expr::Slice(..) | Expr::Member(..) | Expr::TupleMember(..)
    )
}

/// An expression usable as an operand or postfix receiver.
fn atom(e: &Expr) -> String {
    if is_atom(e) { expr(e) } else { format!("({})", expr(e)) }
}

pub fn expr(e: &Expr) -> String {
    use Expr as E;
    match e {
        E::Name(n, _) => n.clone(),
        E::SelfRef(_) => "self".into(),
        E::Int(i, _) => i.to_string(),
        E::Float(f, _) => float(*f),
        E::Str(s, _) => text(s),
        E::Bool(b, _) => b.to_string(),
        E::None(_) => "none".into(),
        E::Unit(_) => "()".into(),
        E::Tuple(v, _) if v.len() == 1 => format!("({},)", expr(&v[0])),
        E::Tuple(v, _) => format!("({})", v.iter().map(expr).collect::<Vec<_>>().join(", ")),
        E::List(v, _) => format!("[{}]", v.iter().map(expr).collect::<Vec<_>>().join(", ")),
        E::ListRepeat(a, n, _) => format!("[{}; {}]", expr(a), expr(n)),
        E::ListComp(a, p, it, c, _) => {
            let mut s = format!("[{} for {} in {}", expr(a), pat(p), atom(it));
            if let Some(c) = c {
                s.push_str(&format!(" if {}", atom(c)));
            }
            s.push(']');
            s
        }
        E::Map(v, _) => format!("{{{}}}", v.iter().map(|(k, x)| format!("{}: {}", expr(k), expr(x))).collect::<Vec<_>>().join(", ")),
        E::ChanNew(t, c, _) => format!("chan[{}]({})", ty(t), c.as_ref().map(|c| expr(c)).unwrap_or_default()),
        E::Call(f, a, _, _) => format!("{}({})", atom(f), args(a)),
        E::Index(x, ix, _) => format!("{}[{}]", atom(x), ix.iter().map(expr).collect::<Vec<_>>().join(", ")),
        E::Slice(x, lo, hi, _) => format!(
            "{}[{}:{}]",
            atom(x),
            lo.as_ref().map(|l| expr(l)).unwrap_or_default(),
            hi.as_ref().map(|h| expr(h)).unwrap_or_default()
        ),
        E::Member(x, n, _) => format!("{}.{n}", atom(x)),
        E::TupleMember(x, i, _) => format!("{}.{i}", atom(x)),
        E::Neg(x, _) => format!("-{}", atom(x)),
        E::BitNot(x, _) => format!("~{}", atom(x)),
        E::Borrow(m, x, _) => format!("&{}{}", if *m { "mut " } else { "" }, atom(x)),
        E::Recv(x, _) => format!("<- {}", atom(x)),
        E::Comptime(x, _) => format!("comptime {}", atom(x)),
        E::Cast(x, t, _) => format!("{} as {}", atom(x), base_ty(t)),
        E::Bin(op, a, b, _) => format!("{} {op} {}", atom(a), atom(b)),
        E::Not(x, _) => format!("not {}", atom(x)),
        E::Catch(a, bind, h, _) => {
            format!("{} catch {}{}", atom(a), bind.as_ref().map(|b| format!("|{b}| ")).unwrap_or_default(), atom(h))
        }
        E::Try(a, el, _) => {
            let mut s = format!("try {}", atom(a));
            if let Some(x) = el {
                s.push_str(&format!(" else {}", atom(x)));
            }
            s
        }
        E::If(c, a, b, _) => format!("if {} then {} else {}", atom(c), atom(a), atom(b)),
        E::Lambda(ps, b, _) => format!("fn({}) => {}", params(ps), atom(b)),
    }
}

fn lit_pat(e: &Expr) -> String {
    match e {
        Expr::Int(i, _) => i.to_string(),
        Expr::Float(f, _) => float(*f),
        _ => expr(e),
    }
}

pub fn pat(p: &Pat) -> String {
    match p {
        Pat::Wild => "_".into(),
        Pat::Path(v) => v.join("."),
        Pat::Ctor(v, a) => format!(
            "{}({})",
            v.join("."),
            a.iter()
                .map(|(l, p)| match l {
                    Some(l) => format!("{l}: {}", pat(p)),
                    None => pat(p),
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pat::Lit(e) => lit_pat(e),
        Pat::Range(a, incl, b) => format!("{}{}{}", lit_pat(a), if *incl { "..=" } else { ".." }, lit_pat(b)),
        Pat::Tuple(v) if v.len() == 1 => format!("({},)", pat(&v[0])),
        Pat::Tuple(v) => format!("({})", v.iter().map(pat).collect::<Vec<_>>().join(", ")),
        Pat::Alt(v) => v.iter().map(pat).collect::<Vec<_>>().join(" | "),
    }
}

fn line(o: &mut String, d: usize, s: &str) {
    for _ in 0..d {
        o.push_str(IND);
    }
    o.push_str(s);
    o.push('\n');
}

fn block(o: &mut String, d: usize, b: &[Stmt]) {
    if b.is_empty() {
        line(o, d, "pass");
    }
    for s in b {
        stmt(o, d, s);
    }
}

/// An expression in statement position: one that would start a statement
/// keyword (`if`, `try`, `fn`, `comptime`) is parenthesized.
fn stmt_expr(e: &Expr) -> String {
    match e {
        Expr::If(..) | Expr::Try(..) | Expr::Lambda(..) | Expr::Comptime(..) => format!("({})", expr(e)),
        _ => expr(e),
    }
}

fn spawn_text(s: &Stmt) -> Option<String> {
    match s {
        Stmt::Spawn(e) => Some(format!("spawn {}", expr(e))),
        Stmt::Arena(n, None, b, _) if n == "$priority" && b.len() == 2 => {
            let (Stmt::Expr(p), Stmt::Spawn(e)) = (&b[0], &b[1]) else { return None };
            Some(format!("spawn {} priority {}", expr(e), expr(p)))
        }
        _ => None,
    }
}

fn stmt(o: &mut String, d: usize, s: &Stmt) {
    match s {
        Stmt::Expr(e) => line(o, d, &stmt_expr(e)),
        Stmt::Assign(op, l, r) => line(o, d, &format!("{} {op} {}", stmt_expr(l), expr(r))),
        Stmt::Send(c, v) => line(o, d, &format!("{} <- {}", stmt_expr(c), expr(v))),
        Stmt::Let(p, None, Expr::Call(c, a, _, _), _) if matches!(&**c, Expr::Name(n, _) if n == "@ghost") && a.len() == 1 => line(o, d, &format!("ghost let {} = {}", pat(p), expr(&a[0].e))),
        Stmt::Let(p, t, Expr::Call(c, a, _, _), _) if matches!(&**c, Expr::Name(n, _) if n == "@ghost") && a.len() == 1 => line(o, d, &format!("ghost let {}: {} = {}", pat(p), ty(t.as_ref().unwrap()), expr(&a[0].e))),
        Stmt::Check(Expr::Call(c, a, _, _)) if matches!(&**c, Expr::Name(n, _) if n == "@decreases") && !a.is_empty() => {
            line(o, d, &format!("decreases {}", a.iter().map(|x| expr(&x.e)).collect::<Vec<_>>().join(", ")))
        }
        Stmt::Check(Expr::Call(c, a, _, _)) if matches!(&**c, Expr::Name(n, _) if n == "@requires" || n == "@ensures") && a.len() == 1 => {
            let k = if let Expr::Name(n, _) = &**c { &n[1..] } else { "" };
            line(o, d, &format!("{k} {}", expr(&a[0].e)))
        }
        Stmt::Let(p, t, e, _) => line(o, d, &format!("let {}{} = {}", pat(p), t.as_ref().map(|t| format!(": {}", ty(t))).unwrap_or_default(), expr(e))),
        Stmt::Var(n, t, e, _) => line(o, d, &format!("var {n}{} = {}", t.as_ref().map(|t| format!(": {}", ty(t))).unwrap_or_default(), expr(e))),
        Stmt::If(arms, els) => {
            for (k, (c, b)) in arms.iter().enumerate() {
                line(o, d, &format!("{} {}", if k == 0 { "if" } else { "elif" }, expr(c)));
                block(o, d + 1, b);
            }
            if let Some(b) = els {
                line(o, d, "else");
                block(o, d + 1, b);
            }
        }
        Stmt::While(c, b, body) => {
            line(o, d, &format!("while {}{}", expr(c), b.as_ref().map(|b| format!(" bound {}", expr(b))).unwrap_or_default()));
            block(o, d + 1, body);
        }
        Stmt::Loop(b, body) => {
            line(o, d, &format!("loop{}", b.as_ref().map(|b| format!(" bound {}", expr(b))).unwrap_or_default()));
            block(o, d + 1, body);
        }
        Stmt::For(p, e, body, _) => {
            line(o, d, &format!("for {} in {}", pat(p), expr(e)));
            block(o, d + 1, body);
        }
        Stmt::Match(Expr::Call(c, sa, _, _), arms, _)
            if matches!(&**c, Expr::Name(n, _) if n == "@iflet") && sa.len() == 1 && arms.len() == 2 && matches!(arms[1].pat, Pat::Wild) =>
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
            let mut conds = vec![];
            if let Some(g) = &arms[0].guard {
                atoms(g, &mut conds);
            }
            let tail: String = conds.iter().map(|c| format!(" and {}", expr(c))).collect();
            line(o, d, &format!("if let {} = {}{tail}", pat(&arms[0].pat), expr(&sa[0].e)));
            block(o, d + 1, &arms[0].body);
            if arms[1].body != [Stmt::Pass] {
                line(o, d, "else");
                block(o, d + 1, &arms[1].body);
            }
        }
        Stmt::Match(e, arms, _) => {
            line(o, d, &format!("match {}", expr(e)));
            for a in arms {
                line(o, d + 1, &format!("{}{}", pat(&a.pat), a.guard.as_ref().map(|g| format!(" if {}", expr(g))).unwrap_or_default()));
                block(o, d + 2, &a.body);
            }
        }
        Stmt::Return(e, _) => line(o, d, &match e {
            Some(e) => format!("return {}", expr(e)),
            None => "return".into(),
        }),
        Stmt::Break(_) => line(o, d, "break"),
        Stmt::Continue(_) => line(o, d, "continue"),
        Stmt::Pass => line(o, d, "pass"),
        Stmt::Emit(e) => line(o, d, &format!("emit {}", expr(e))),
        Stmt::Fail(e) => line(o, d, &format!("fail {}", expr(e))),
        Stmt::Spawn(e) => line(o, d, &format!("spawn {}", expr(e))),
        Stmt::Yield(e, _) => line(o, d, &format!("yield {}", expr(e))),
        Stmt::Check(e) => line(o, d, &format!("check {}", expr(e))),
        Stmt::Comptime(b) => {
            line(o, d, "comptime");
            block(o, d + 1, b);
        }
        Stmt::Select(arms, _) => select(o, d, arms, false),
        Stmt::Device(n, b, body, _) => {
            line(o, d, &format!("with device {n}{}", b.as_ref().map(|b| format!(" bound {}", expr(b))).unwrap_or_default()));
            block(o, d + 1, body);
        }
        Stmt::Arena(n, b, body, _) => arena(o, d, n, b.as_ref(), body),
    }
}

fn select(o: &mut String, d: usize, arms: &[SelArm], nondet: bool) {
    line(o, d, if nondet { "select nondet" } else { "select" });
    for a in arms {
        let (head, body) = match a {
            SelArm::Recv(p, c, b) => (format!("recv {} from {}", pat(p), expr(c)), b),
            SelArm::Send(v, c, b) => (format!("send {} to {}", expr(v), expr(c)), b),
            SelArm::After(n, b) => (format!("after {}", expr(n)), b),
            SelArm::Default(b) => ("default".to_string(), b),
        };
        line(o, d + 1, &head);
        block(o, d + 2, body);
    }
}

fn arena(o: &mut String, d: usize, n: &str, b: Option<&Expr>, body: &[Stmt]) {
    let first_expr = || match body.first() {
        Some(Stmt::Expr(e)) => Some(e),
        _ => None,
    };
    match n {
        "$forpar" if body.len() == 1 => {
            if let Stmt::For(p, e, fb, _) = &body[0] {
                line(o, d, &format!("for {} in {} par", pat(p), expr(e)));
                block(o, d + 1, fb);
                return;
            }
        }
        "$priority" => {
            if let Some(t) = spawn_text(&Stmt::Arena(n.into(), None, body.to_vec(), 0)) {
                line(o, d, &t);
                return;
            }
        }
        "$barrier" if body.is_empty() => return line(o, d, "barrier"),
        "$scope" | "$atomic" => {
            line(o, d, &n[1..]);
            return block(o, d + 1, body);
        }
        "$nondet" if body.len() == 1 => {
            if let Stmt::Select(arms, _) = &body[0] {
                return select(o, d, arms, true);
            }
        }
        "$par" => {
            line(o, d, "par");
            for br in body {
                if let Stmt::Arena(bn, None, bb, _) = br
                    && let Some(name) = bn.strip_prefix("$branch:")
                {
                    line(o, d + 1, &format!("branch {name}"));
                    block(o, d + 2, bb);
                }
            }
            return;
        }
        "$budget" => {
            if let Some(e) = first_expr() {
                line(o, d, &format!("with budget {}", expr(e)));
                let (rest, else_b) = match body.last() {
                    Some(Stmt::If(arms, _)) if arms.len() == 1
                        && matches!(&arms[0].0, Expr::Bool(false, _)) =>
                        (&body[1..body.len() - 1], Some(&arms[0].1[..])),
                    _ => (&body[1..], None),
                };
                block(o, d + 1, rest);
                if let Some(eb) = else_b {
                    match eb.first() {
                        Some(Stmt::Let(p, None, Expr::Call(c, a, _, _), _))
                            if matches!(&**c, Expr::Name(n, _) if n == "Exhausted")
                                && a.iter().any(|x| matches!(&x.e, Expr::Name(n, _) if n == "\u{1}__bu")) =>
                        {
                            line(o, d, &format!("else as {}", pat(p)));
                            block(o, d + 1, &eb[1..]);
                        }
                        _ => {
                            line(o, d, "else");
                            block(o, d + 1, eb);
                        }
                    }
                }
                return;
            }
        }
        "$deadline" => {
            if let (Some(e), Some(Stmt::Arena(dl, None, db, _))) = (first_expr(), body.get(1))
                && dl == "$dlbody"
            {
                line(o, d, &format!("with deadline {}", expr(e)));
                block(o, d + 1, db);
                if let Some(Stmt::If(arms, None)) = body.get(2) {
                    line(o, d, "else");
                    block(o, d + 1, &arms[0].1);
                }
                return;
            }
        }
        _ => {}
    }
    if let Some(h) = n.strip_prefix("$spawnas:")
        && body.len() == 1
        && let Some(t) = spawn_text(&body[0])
    {
        return line(o, d, &format!("{t} -> {h}"));
    }
    if let Some(g) = n.strip_prefix("$group:")
        && let Some(e) = first_expr()
    {
        line(o, d, &format!("with group {g} lanes {}", expr(e)));
        return block(o, d + 1, &body[1..]);
    }
    line(o, d, &format!("with arena {n}{}", b.map(|b| format!(" bound {}", expr(b))).unwrap_or_default()));
    block(o, d + 1, body);
}

fn fn_decl(o: &mut String, d: usize, f: &FnDecl) {
    match &f.body {
        Body::Arrow(e) => line(o, d, &format!("{} = {}", sig(&f.sig), expr(e))),
        Body::Block(b) => {
            line(o, d, &sig(&f.sig));
            block(o, d + 1, b);
        }
    }
}

fn step_body(o: &mut String, d: usize, head: String, b: &Body) {
    match b {
        Body::Arrow(e) => line(o, d, &format!("{head} -> {}", expr(e))),
        Body::Block(s) => {
            line(o, d, &head);
            block(o, d + 1, s);
        }
    }
}

fn field(o: &mut String, d: usize, f: &Field) {
    line(o, d, &format!("{}: {}{}", f.name, ty(&f.ty), f.init.as_ref().map(|e| format!(" = {}", expr(e))).unwrap_or_default()));
}

fn example(x: &Example) -> String {
    let evs: Vec<String> = x.events.iter().map(|(n, a)| if a.is_empty() { n.clone() } else { format!("{n}({})", args(a)) }).collect();
    let rhs = match (&x.refused, &x.value) {
        (Some(r), _) => match r.strip_prefix("trap(").and_then(|k| k.strip_suffix(')')) {
            Some(k) => format!("refused: trap({})", serde_json::Value::String(k.into())),
            None => format!("refused: {r}"),
        },
        (None, Some(v)) => expr(v),
        (None, None) => "()".into(),
    };
    if evs.is_empty() { format!("example -> {rhs}") } else { format!("example {} -> {rhs}", evs.join(", ")) }
}

fn proto(o: &mut String, d: usize, steps: &[ProtoStep]) {
    for s in steps {
        match s {
            ProtoStep::Send(t) => line(o, d, &format!("send {}", ty(t))),
            ProtoStep::Recv(t) => line(o, d, &format!("recv {}", ty(t))),
            ProtoStep::Offer(arms) | ProtoStep::Choose(arms) => {
                line(o, d, if matches!(s, ProtoStep::Offer(_)) { "offer" } else { "choose" });
                for (l, b) in arms {
                    line(o, d + 1, l);
                    proto(o, d + 2, b);
                }
            }
            ProtoStep::Loop(b) => {
                line(o, d, "loop");
                proto(o, d + 1, b);
            }
            ProtoStep::Break => line(o, d, "break"),
        }
    }
}

fn item(o: &mut String, it: &Item) {
    match it {
        Item::Use(p, a, _) => line(o, 0, &format!("use {}{}", p.join("."), a.as_ref().map(|a| format!(" as {a}")).unwrap_or_default())),
        Item::Intent(s, ex, ck) => {
            line(o, 0, &format!("intent {}", text(s)));
            for x in ex {
                line(o, 1, &example(x));
            }
            for c in ck {
                line(o, 1, &format!("check {}", expr(c)));
            }
        }
        Item::Example(x) => line(o, 0, &example(x)),
        Item::Check(e) => line(o, 0, &format!("check {}", expr(e))),
        Item::Const { name, ty: t, e, .. } => line(o, 0, &format!("const {name}{} = {}", t.as_ref().map(|t| format!(": {}", ty(t))).unwrap_or_default(), expr(e))),
        Item::TypeAlias { name, generics, ty: t, pred, .. } => line(
            o,
            0,
            &format!("type {name}{} = {}{}", gparams(generics), ty(t), pred.as_ref().map(|p| format!(" where {}", expr(p))).unwrap_or_default()),
        ),
        Item::Struct { name, generics, fields, linear, .. } => {
            line(o, 0, &format!("{}struct {name}{}", if *linear { "linear " } else { "" }, gparams(generics)));
            for f in fields {
                field(o, 1, f);
            }
        }
        Item::State { name, fields, replicated, .. } => {
            line(o, 0, &format!("{}state {name}", if *replicated { "replicated " } else { "" }));
            for f in fields {
                field(o, 1, f);
            }
        }
        Item::Event { name, payload: p, interrupt, .. } => match interrupt {
            Some(pr) => line(o, 0, &format!("interrupt event {name}{} priority {}", payload(p), expr(pr))),
            None => line(o, 0, &format!("event {name}{}", payload(p))),
        },
        Item::Error { name, payload: p, .. } => line(o, 0, &format!("error {name}{}", payload(p))),
        Item::Kind { name, generics, variants, .. } => {
            line(o, 0, &format!("kind {name}{}", gparams(generics)));
            for (v, p) in variants {
                line(o, 1, &format!("{v}{}", payload(p)));
            }
        }
        Item::Trait { name, generics, members, .. } => {
            line(o, 0, &format!("trait {name}{}", gparams(generics)));
            for m in members {
                match m {
                    TraitMember::Type(n) => line(o, 1, &format!("type {n}")),
                    TraitMember::Fn(s) => line(o, 1, &sig(s)),
                }
            }
        }
        Item::Impl { generics, ty: t, trait_, members, .. } => {
            let g = if generics.is_empty() { String::new() } else { format!("forall {} ", gparams(generics)) };
            match trait_ {
                Some(tr) => line(o, 0, &format!("impl {g}{} for {}", ty(tr), ty(t))),
                None => line(o, 0, &format!("impl {g}{}", ty(t))),
            }
            for f in members {
                fn_decl(o, 1, f);
            }
        }
        Item::Fn(f) => fn_decl(o, 0, f),
        Item::Extern { abi, sig: s, .. } => line(o, 0, &format!("extern {} {}", text(abi), sig(s))),
        Item::Step { region, pat: p, guard, body, deadline, .. } => {
            let head = format!(
                "step {region} on {}{}{}",
                pat(p),
                guard.as_ref().map(|g| format!(" if {}", expr(g))).unwrap_or_default(),
                deadline.as_ref().map(|g| format!(" deadline {}", expr(g))).unwrap_or_default()
            );
            step_body(o, 0, head, body);
        }
        Item::Actor { name, fields, handlers, fns, .. } => {
            line(o, 0, &format!("actor {name}"));
            for f in fields {
                field(o, 1, f);
            }
            for h in handlers {
                let head = format!("on {}{}", pat(&h.pat), h.guard.as_ref().map(|g| format!(" if {}", expr(g))).unwrap_or_default());
                step_body(o, 1, head, &h.body);
            }
            for f in fns {
                fn_decl(o, 1, f);
            }
        }
        Item::Protocol { name, steps, .. } => {
            line(o, 0, &format!("protocol {name}"));
            proto(o, 1, steps);
        }
        Item::Node { name, actors, .. } => line(o, 0, &format!("node {name} {{{}}}", actors.join(", "))),
        Item::Supervisor { name, strategy, children, .. } => {
            line(o, 0, &format!("supervisor {name} strategy {strategy}"));
            for c in children {
                line(o, 1, &format!("child {}: {} restart bound {}", c.name, c.ty, c.restart));
            }
        }
    }
}
