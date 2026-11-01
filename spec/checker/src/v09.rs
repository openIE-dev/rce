//! Version 0.9 (Appendix F, `book/17-draft09.md`): the implemented subset.
//!
//! Status is kept in `IMPLEMENTED` and `UNIMPLEMENTED` at the end of this
//! file and is printed by `rce api-diff --status`. A construct of Appendix F
//! that this checker does not implement is *reserved, not implemented* (W-RES-1); it
//! is never read as an ordinary name and never silently ignored.
//!
//! Nothing here proves a theorem. T42 to T56 stay stated and unproved.
use crate::ast::*;
use std::collections::{BTreeSet, HashMap, HashSet};

/// The words of Appendix F ("Reserved words") whose construct is not
/// implemented. A word leaves this list when the construct it introduces is
/// implemented (`law` left it for the laws of 6.11).
pub const RESERVED_GATED: &[&str] = &["declassify"];

/// Reserved words whose construct is implemented: a use that the parser did
/// not consume as that construct requires change as a name (W-NAME-2).
pub const RESERVED_LIVE: &[&str] = &["label", "release", "endorse", "ct", "requires", "ensures", "ghost", "decreases", "law"];

/// W-NAME-2: a live reserved word used as a name.
pub fn stray_words(toks: &[(crate::parse::Tok, usize)], used: &[usize]) -> Result<(), String> {
    use crate::parse::Tok;
    let mut errs = vec![];
    for (i, (t, line)) in toks.iter().enumerate() {
        if let Tok::Id(w) = t
            && RESERVED_LIVE.contains(&w.as_str())
            && !used.contains(&i)
            && !(i > 0 && matches!(toks[i - 1].0, Tok::P(".")))
        {
            errs.push(format!("line {line}: W-NAME-2 `{w}` is reserved and is not used as its construct here"));
        }
    }
    if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
}

/// The reserved type names (F, "Reserved words"), gated the same way.
pub const TYPES_GATED: &[&str] = &["Template", "mathint"];

fn gate_msg(line: usize, w: &str) -> String {
    if w == "declassify" {
        return format!("line {line}: W-RES-1 `declassify` is reserved and is never admitted: a labeled value leaves its label only by `release` with a held capability (F.4.2)");
    }
    format!(
        "line {line}: W-RES-1 `{w}` is reserved and its construct (Appendix F) is specified but not implemented by this checker, so it is reserved, not implemented"
    )
}

/// W-RES-1: reserved words and type names whose construct is not implemented.
pub fn gate(toks: &[(crate::parse::Tok, usize)]) -> Result<(), String> {
    use crate::parse::Tok;
    let mut errs = vec![];
    for (i, (t, line)) in toks.iter().enumerate() {
        if let Tok::Id(w) = t {
            let after_dot = i > 0 && matches!(toks[i - 1].0, Tok::P("."));
            if after_dot {
                continue;
            }
            if RESERVED_GATED.contains(&w.as_str()) || TYPES_GATED.contains(&w.as_str()) {
                errs.push(gate_msg(*line, w));
            }
            // F.5.1 W-CAP-4/5: `host.revoke(id)` is a host log entry, which this
            // reference checker does not have (the caretaker model is `caretaker.rs`)
            if w == "host"
                && matches!(toks.get(i + 1), Some((Tok::P("."), _)))
                && matches!(toks.get(i + 2), Some((Tok::Id(r), _)) if r == "revoke")
            {
                errs.push(format!("line {line}: W-RES-1 `host.revoke` (F.5.1, W-CAP-4) needs the host log, which the reference checker does not implement, so it is reserved, not implemented"));
            }
        }
    }
    if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
}

// ---------------------------------------------------------------- S8 lexical

/// Characters that W-LEX-4 does not admit inside a string or comment: bidirectional
/// controls, zero-width and invisible format characters, C0 and C1 controls
/// (other than the line feed that ends a line), Unicode noncharacters and the
/// tag block. `\u{...}` in a string is the one way to write them.
pub fn w_lex_4_class(c: char) -> Option<&'static str> {
    let u = c as u32;
    Some(match u {
        0x202A..=0x202E | 0x2066..=0x2069 | 0x061C | 0x200E | 0x200F => "bidirectional control",
        0x200B..=0x200D | 0x2060..=0x2064 | 0xFEFF | 0x00AD | 0x034F | 0x180E | 0x115F | 0x1160
        | 0x17B4 | 0x17B5 | 0x3164 | 0xFFA0 | 0xFFF9..=0xFFFB => "invisible format character",
        0xE0000..=0xE007F => "tag character",
        0x0000..=0x0008 | 0x000B..=0x000C | 0x000E..=0x001F | 0x007F..=0x009F => "control character",
        0xFDD0..=0xFDEF => "noncharacter",
        _ if (u & 0xFFFE) == 0xFFFE => "noncharacter",
        0x40000..=0xDFFFF => "unassigned scalar value",
        _ => return None,
    })
}

/// W-LEX-4 over the whole source text: a character it does not admit
/// anywhere in a string or a comment. (Outside both, the lexer already
/// reports every non-ASCII character, 2.3.) A leading U+FEFF is already W-LEX-1.
pub fn lexical(src: &str) -> Result<(), String> {
    let mut errs = vec![];
    for (n, ln) in src.split('\n').enumerate() {
        for c in ln.chars() {
            if let Some(k) = w_lex_4_class(c) {
                // the tab and BOM have their own W-LEX-1 reports in the lexer
                if c == '\t' || (c == '\u{feff}' && n == 0) || c == '\r' {
                    continue;
                }
                errs.push(format!(
                    "line {}: W-LEX-4 {k} U+{:04X} in source text; write it with \\u{{...}} inside a string",
                    n + 1,
                    c as u32
                ));
                break;
            }
        }
    }
    if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
}

// ------------------------------------------------------------------ hygiene

#[derive(Clone)]
struct B {
    name: String,
    line: usize,
    read: bool,
    /// a `&mut` parameter: a write through it is an observable effect
    mut_ref: bool,
    /// not subject to the unused rule (arena names, fields)
    exempt: bool,
    /// the lines on which the binding is read or written
    uses: Vec<usize>,
    /// a name beginning with `_` that is read (W-UNUSED-1)
    read_marked: bool,
}

/// A rename that `rce fmt` writes for a W-UNUSED-1 report (the method
/// *altered*, 7.1): an unread binding `x` is named `_x`, and a binding `_x`
/// that is read is named `x`. Names carry no meaning (T7), so the rename
/// keeps the program's meaning. `to` is the name the rule gives; the
/// formatter takes the first of `to`, `to1`, `to2`, ... that the text does
/// not already use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rename {
    pub from: String,
    pub to: String,
    /// the line of the binding
    pub line: usize,
    /// the line of the binding and every line that reads or writes it
    pub lines: Vec<usize>,
}

struct W<'a> {
    scopes: Vec<Vec<B>>,
    errs: Vec<String>,
    renames: Vec<Rename>,
    variants: &'a HashSet<String>,
    modnames: &'a HashMap<String, usize>,
    fields: HashSet<String>,
    free: HashSet<String>,
}

fn err(l: usize, code: &str, m: String) -> String {
    format!("line {l}: {code} {m}")
}

fn const_bool(e: &Expr) -> bool {
    match e {
        Expr::Bool(..) => true,
        Expr::Not(x, _) => const_bool(x),
        _ => false,
    }
}

impl<'a> W<'a> {
    fn push(&mut self) {
        self.scopes.push(vec![]);
    }
    fn pop(&mut self) {
        let s = self.scopes.pop().unwrap();
        for b in s {
            let mut lines = vec![b.line];
            lines.extend(b.uses.iter().copied());
            lines.sort();
            lines.dedup();
            if b.read_marked {
                let base = b.name.trim_start_matches('_');
                let to = if base.is_empty() || base.starts_with(|c: char| c.is_ascii_digit()) { format!("v{base}") } else { base.to_string() };
                self.renames.push(Rename { from: b.name.clone(), to, line: b.line, lines: lines.clone() });
            }
            if !b.read && !b.exempt && b.name != "_" && !b.name.starts_with('_') {
                self.renames.push(Rename { from: b.name.clone(), to: format!("_{}", b.name), line: b.line, lines });
                self.errs.push(err(
                    b.line,
                    "W-UNUSED-1",
                    format!("`{}` is never read; remove it or name it `_{}`", b.name, b.name),
                ));
            }
        }
    }
    fn declare(&mut self, name: &str, line: usize, mut_ref: bool, exempt: bool) {
        if name == "_" {
            return;
        }
        let mut hit: Option<usize> = None;
        for s in &self.scopes {
            if let Some(b) = s.iter().find(|b| b.name == name) {
                hit = Some(b.line);
            }
        }
        if let Some(l0) = hit {
            self.errs.push(err(line, "W-NAME-3", format!("`{name}` shadows a name declared at line {l0} in an enclosing scope")));
        } else if self.fields.contains(name) {
            self.errs.push(err(line, "W-NAME-3", format!("`{name}` shadows a field of the state region")));
        } else if let Some(l0) = self.modnames.get(name) {
            self.errs.push(err(line, "W-NAME-3", format!("`{name}` shadows the module-level name declared at line {l0}")));
        }
        self.scopes.last_mut().unwrap().push(B { name: name.into(), line, read: false, mut_ref, exempt, uses: vec![], read_marked: false });
    }
    fn read(&mut self, n: &str, line: usize) {
        for s in self.scopes.iter_mut().rev() {
            if let Some(b) = s.iter_mut().rev().find(|b| b.name == n) {
                b.read = true;
                b.uses.push(line);
                if n.starts_with('_') && !b.exempt {
                    b.read_marked = true;
                    self.errs.push(err(line, "W-UNUSED-1", format!("`{n}` begins with `_` (deliberately unused) and is read")));
                }
                return;
            }
        }
        self.free.insert(n.to_string());
    }
    fn write(&mut self, n: &str, line: usize) {
        for s in self.scopes.iter_mut().rev() {
            if let Some(b) = s.iter_mut().rev().find(|b| b.name == n) {
                b.uses.push(line);
                if b.mut_ref {
                    b.read = true;
                }
                return;
            }
        }
    }
    fn binds(&self, p: &Pat, out: &mut Vec<String>) {
        match p {
            Pat::Wild | Pat::Lit(_) | Pat::Range(..) => {}
            Pat::Path(q) => {
                if q.len() == 1 && !self.variants.contains(&q[0]) {
                    out.push(q[0].clone());
                }
            }
            Pat::Ctor(_, fs) => {
                for (_, p) in fs {
                    self.binds(p, out);
                }
            }
            Pat::Tuple(ps) => {
                for p in ps {
                    self.binds(p, out);
                }
            }
            Pat::Alt(ps) => {
                if let Some(p) = ps.first() {
                    self.binds(p, out);
                }
            }
        }
    }
    fn pat_heads(&mut self, p: &Pat) {
        match p {
            Pat::Path(q) if q.len() > 1 || self.variants.contains(&q[0]) => {
                self.free.insert(q[0].clone());
            }
            Pat::Ctor(q, fs) => {
                self.free.insert(q[0].clone());
                for (_, p) in fs {
                    self.pat_heads(p);
                }
            }
            Pat::Tuple(ps) | Pat::Alt(ps) => {
                for p in ps {
                    self.pat_heads(p);
                }
            }
            Pat::Lit(e) => self.expr(e),
            Pat::Range(a, _, b) => {
                self.expr(a);
                self.expr(b);
            }
            _ => {}
        }
    }
    fn bind_pat(&mut self, p: &Pat, line: usize) {
        self.pat_heads(p);
        let mut v = vec![];
        self.binds(p, &mut v);
        for n in v {
            self.declare(&n, line, false, false);
        }
    }
    fn ty(&mut self, t: &Ty) {
        match t {
            Ty::Path(p, a) => {
                self.free.insert(p[0].clone());
                for x in a {
                    self.ty(x);
                }
            }
            Ty::Unit => {}
            Ty::Tuple(v) => v.iter().for_each(|x| self.ty(x)),
            Ty::Slice(x) | Ty::Chan(x) | Ty::Iter(x) | Ty::Ref(_, x) | Ty::Opt(x) | Ty::Region(x, _) => self.ty(x),
            Ty::Array(x, e) => {
                self.ty(x);
                self.expr(e);
            }
            Ty::Fn(a, r) => {
                a.iter().for_each(|x| self.ty(x));
                self.ty(r);
            }
            Ty::Union(a, b) => {
                self.ty(a);
                self.ty(b);
            }
        }
    }
    fn lhs(&mut self, e: &Expr, root: bool) {
        match e {
            Expr::Name(n, l) => {
                if root {
                    self.write(n, *l)
                } else {
                    // a write through a field or element is an effect on the
                    // value, so the root counts as used
                    self.read_quiet(n, *l)
                }
            }
            Expr::Member(b, _, _) | Expr::TupleMember(b, _, _) => self.lhs(b, false),
            Expr::Index(b, ix, _) => {
                self.lhs(b, false);
                ix.iter().for_each(|x| self.expr(x));
            }
            Expr::Slice(b, lo, hi, _) => {
                self.lhs(b, false);
                lo.iter().for_each(|x| self.expr(x));
                hi.iter().for_each(|x| self.expr(x));
            }
            other => self.expr(other),
        }
    }
    fn read_quiet(&mut self, n: &str, line: usize) {
        for s in self.scopes.iter_mut().rev() {
            if let Some(b) = s.iter_mut().rev().find(|b| b.name == n) {
                b.read = true;
                b.uses.push(line);
                return;
            }
        }
        self.free.insert(n.to_string());
    }
    fn expr(&mut self, e: &Expr) {
        use Expr::*;
        match e {
            Name(n, l) => self.read(n, *l),
            SelfRef(_) | Int(..) | Float(..) | Str(..) | Bool(..) | None(_) | Unit(_) => {}
            Tuple(v, _) | List(v, _) => v.iter().for_each(|x| self.expr(x)),
            ListRepeat(a, b, _) => {
                self.expr(a);
                self.expr(b);
            }
            ListComp(el, pat, it, cond, l) => {
                self.expr(it);
                self.push();
                self.bind_pat(pat, *l);
                if let Some(c) = cond {
                    self.expr(c);
                }
                self.expr(el);
                self.pop();
            }
            Map(kv, _) => kv.iter().for_each(|(k, v)| {
                self.expr(k);
                self.expr(v);
            }),
            ChanNew(t, n, _) => {
                self.ty(t);
                n.iter().for_each(|x| self.expr(x));
            }
            Call(f, a, _, _) => {
                self.expr(f);
                a.iter().for_each(|x| self.expr(&x.e));
            }
            Index(b, ix, _) => {
                self.expr(b);
                ix.iter().for_each(|x| self.expr(x));
            }
            Slice(b, lo, hi, _) => {
                self.expr(b);
                lo.iter().for_each(|x| self.expr(x));
                hi.iter().for_each(|x| self.expr(x));
            }
            Member(b, _, _) | TupleMember(b, _, _) | Neg(b, _) | BitNot(b, _) | Borrow(_, b, _) | Recv(b, _) | Comptime(b, _) | Not(b, _) => self.expr(b),
            Cast(b, t, _) => {
                self.expr(b);
                self.ty(t);
            }
            Bin(_, a, b, _) => {
                self.expr(a);
                self.expr(b);
            }
            Catch(a, bind, h, l) => {
                self.expr(a);
                self.push();
                if let Some(n) = bind {
                    self.declare(n, *l, false, false);
                }
                self.expr(h);
                self.pop();
            }
            Try(a, el, _) => {
                self.expr(a);
                el.iter().for_each(|x| self.expr(x));
            }
            If(c, a, b, l) => {
                if const_bool(c) {
                    self.errs.push(err(*l, "W-UNUSED-2", "a condition that is a constant".into()));
                }
                self.expr(c);
                self.expr(a);
                self.expr(b);
            }
            Lambda(ps, b, l) => {
                self.push();
                self.params(ps, *l);
                self.expr(b);
                self.pop();
            }
        }
    }
    fn params(&mut self, ps: &[Param], line: usize) {
        for p in ps {
            if let Param::Named { name, ty, .. } = p {
                self.ty(ty);
                self.declare(name, line, matches!(ty, Ty::Ref(true, _)), false);
            }
        }
    }
    fn cond(&mut self, c: &Expr) {
        // `with budget ... else` and `with deadline ... else` encode their else
        // block as `if false` on the line of the `with` (parse.rs)
        let encoded = matches!(c, Expr::Bool(false, l) if ENC_LINES.with(|s| s.borrow().contains(l)));
        if const_bool(c) && !encoded {
            self.errs.push(err(c.line(), "W-UNUSED-2", "a condition that is a constant".into()));
        }
        self.expr(c);
    }
    fn body(&mut self, b: &Body) {
        match b {
            Body::Arrow(e) => self.expr(e),
            Body::Block(s) => self.block(s),
        }
    }
    fn block(&mut self, ss: &[Stmt]) {
        self.push();
        for (i, s) in ss.iter().enumerate() {
            self.stmt(s);
            if let Stmt::Assign(op, Expr::Name(x, l), _) = s
                && op == "="
            {
                for later in &ss[i + 1..] {
                    if mentions(later, x) {
                        if let Stmt::Assign(op2, Expr::Name(x2, _), r2) = later
                            && op2 == "="
                            && x2 == x
                            && !expr_mentions(r2, x)
                        {
                            self.errs.push(err(*l, "W-ASSIGN-1", format!("the value stored in `{x}` is overwritten before any read")));
                        }
                        break;
                    }
                }
            }
        }
        self.pop();
    }
    fn stmt(&mut self, s: &Stmt) {
        use Stmt::*;
        match s {
            Expr(e) | Emit(e) | Fail(e) | Spawn(e) | Check(e) => self.expr(e),
            Yield(e, _) => self.expr(e),
            Assign(_, l, r) => {
                self.expr(r);
                self.lhs(l, true);
            }
            Send(a, b) => {
                self.expr(a);
                self.expr(b);
            }
            Let(p, t, e, l) => {
                if let Some(t) = t {
                    self.ty(t);
                }
                self.expr(e);
                self.bind_pat(p, *l);
            }
            Var(n, t, e, l) => {
                if let Some(t) = t {
                    self.ty(t);
                }
                self.expr(e);
                self.declare(n, *l, false, false);
            }
            If(bs, el) => {
                for (c, b) in bs {
                    self.cond(c);
                    self.block(b);
                }
                if let Some(b) = el {
                    self.block(b);
                }
            }
            While(c, b, body) => {
                self.cond(c);
                b.iter().for_each(|x| self.expr(x));
                self.block(body);
            }
            Loop(b, body) => {
                b.iter().for_each(|x| self.expr(x));
                self.block(body);
            }
            For(p, it, body, l) => {
                self.expr(it);
                self.push();
                self.bind_pat(p, *l);
                self.block(body);
                self.pop();
            }
            Match(e, arms, _) => {
                self.expr(e);
                for a in arms {
                    self.push();
                    self.bind_pat(&a.pat, a.line);
                    if let Some(g) = &a.guard {
                        self.cond(g);
                    }
                    self.block(&a.body);
                    self.pop();
                }
            }
            Return(e, _) => e.iter().for_each(|x| self.expr(x)),
            Break(_) | Continue(_) | Pass => {}
            Select(arms, l) => {
                for a in arms {
                    match a {
                        SelArm::Recv(p, c, b) => {
                            self.expr(c);
                            self.push();
                            self.bind_pat(p, *l);
                            self.block(b);
                            self.pop();
                        }
                        SelArm::Send(c, v, b) => {
                            self.expr(c);
                            self.expr(v);
                            self.block(b);
                        }
                        SelArm::After(e, b) => {
                            self.expr(e);
                            self.block(b);
                        }
                        SelArm::Default(b) => self.block(b),
                    }
                }
            }
            Arena(n, e, b, l) | Device(n, e, b, l) => {
                e.iter().for_each(|x| self.expr(x));
                self.push();
                if !n.starts_with('$') {
                    self.declare(n, *l, false, true);
                }
                self.block(b);
                self.pop();
            }
            Comptime(b) => self.block(b),
        }
    }
    fn func(&mut self, sig: &FnSig, body: &Body) {
        self.push();
        for g in &sig.generics {
            g.bounds.iter().for_each(|t| self.ty(t));
        }
        self.params(&sig.params, sig.line);
        if let Some(r) = &sig.ret {
            self.ty(r);
        }
        self.body(body);
        self.pop();
    }
}

fn expr_mentions(e: &Expr, x: &str) -> bool {
    let mut w = Mention { x, hit: false };
    w.expr(e);
    w.hit
}

fn mentions(s: &Stmt, x: &str) -> bool {
    let mut w = Mention { x, hit: false };
    w.stmt(s);
    w.hit
}

/// Whether a statement or expression names `x` anywhere (a conservative
/// over-approximation of "reads x": it makes W-ASSIGN-1 report less).
struct Mention<'a> {
    x: &'a str,
    hit: bool,
}

impl Mention<'_> {
    fn expr(&mut self, e: &Expr) {
        use Expr::*;
        match e {
            Name(n, _) => self.hit |= n == self.x,
            SelfRef(_) | Int(..) | Float(..) | Str(..) | Bool(..) | None(_) | Unit(_) => {}
            Tuple(v, _) | List(v, _) => v.iter().for_each(|x| self.expr(x)),
            ListRepeat(a, b, _) | Bin(_, a, b, _) => {
                self.expr(a);
                self.expr(b);
            }
            ListComp(a, _, b, c, _) => {
                self.expr(a);
                self.expr(b);
                c.iter().for_each(|x| self.expr(x));
            }
            Map(kv, _) => kv.iter().for_each(|(k, v)| {
                self.expr(k);
                self.expr(v);
            }),
            ChanNew(_, n, _) => n.iter().for_each(|x| self.expr(x)),
            Call(f, a, _, _) => {
                self.expr(f);
                a.iter().for_each(|x| self.expr(&x.e));
            }
            Index(b, ix, _) => {
                self.expr(b);
                ix.iter().for_each(|x| self.expr(x));
            }
            Slice(b, lo, hi, _) => {
                self.expr(b);
                lo.iter().for_each(|x| self.expr(x));
                hi.iter().for_each(|x| self.expr(x));
            }
            Member(b, _, _) | TupleMember(b, _, _) | Neg(b, _) | BitNot(b, _) | Borrow(_, b, _) | Recv(b, _) | Comptime(b, _) | Not(b, _) | Cast(b, _, _) => self.expr(b),
            Catch(a, _, b, _) => {
                self.expr(a);
                self.expr(b);
            }
            Try(a, el, _) => {
                self.expr(a);
                el.iter().for_each(|x| self.expr(x));
            }
            If(c, a, b, _) => {
                self.expr(c);
                self.expr(a);
                self.expr(b);
            }
            Lambda(_, b, _) => self.expr(b),
        }
    }
    fn block(&mut self, ss: &[Stmt]) {
        ss.iter().for_each(|s| self.stmt(s));
    }
    fn stmt(&mut self, s: &Stmt) {
        use Stmt::*;
        match s {
            Expr(e) | Emit(e) | Fail(e) | Spawn(e) | Check(e) | Yield(e, _) => self.expr(e),
            Assign(_, a, b) | Send(a, b) => {
                self.expr(a);
                self.expr(b);
            }
            Let(_, _, e, _) | Var(_, _, e, _) => self.expr(e),
            If(bs, el) => {
                for (c, b) in bs {
                    self.expr(c);
                    self.block(b);
                }
                el.iter().for_each(|b| self.block(b));
            }
            While(c, b, body) => {
                self.expr(c);
                b.iter().for_each(|x| self.expr(x));
                self.block(body);
            }
            Loop(b, body) => {
                b.iter().for_each(|x| self.expr(x));
                self.block(body);
            }
            For(_, it, body, _) => {
                self.expr(it);
                self.block(body);
            }
            Match(e, arms, _) => {
                self.expr(e);
                for a in arms {
                    if let Some(g) = &a.guard {
                        self.expr(g);
                    }
                    self.block(&a.body);
                }
            }
            Return(e, _) => e.iter().for_each(|x| self.expr(x)),
            Break(_) | Continue(_) | Pass => {}
            Select(arms, _) => {
                for a in arms {
                    match a {
                        SelArm::Recv(_, c, b) => {
                            self.expr(c);
                            self.block(b);
                        }
                        SelArm::Send(c, v, b) => {
                            self.expr(c);
                            self.expr(v);
                            self.block(b);
                        }
                        SelArm::After(e, b) => {
                            self.expr(e);
                            self.block(b);
                        }
                        SelArm::Default(b) => self.block(b),
                    }
                }
            }
            Arena(_, e, b, _) | Device(_, e, b, _) => {
                e.iter().for_each(|x| self.expr(x));
                self.block(b);
            }
            Comptime(b) => self.block(b),
        }
    }
}

thread_local! {
    static ENC_LINES: std::cell::RefCell<HashSet<usize>> = std::cell::RefCell::new(HashSet::new());
}

/// F.3.2 and F.3.3 over one parsed module: W-UNUSED-1, W-UNUSED-2,
/// W-ASSIGN-1 (straight-line stores only) and W-NAME-3. W-DROP-1 needs types
/// and is in `check.rs`.
pub fn hygiene(m: &Module) -> Vec<String> {
    walk(m).0
}

/// The renames `rce fmt` writes for the W-UNUSED-1 reports on bindings
/// (an unused `use` is altered by the formatter from its line).
pub fn unused_renames(m: &Module) -> Vec<Rename> {
    walk(m).1
}

fn walk(m: &Module) -> (Vec<String>, Vec<Rename>) {
    ENC_LINES.with(|s| {
        let mut set = s.borrow_mut();
        set.clear();
        fn scan(ss: &[Stmt], set: &mut HashSet<usize>) {
            for s in ss {
                if let Stmt::Arena(n, _, _, l) = s
                    && (n == "$budget" || n == "$deadline")
                {
                    set.insert(*l);
                }
                for b in stmt_blocks(s) {
                    scan(b, set);
                }
            }
        }
        for (f, _) in fn_decls(m) {
            if let Body::Block(b) = &f.body {
                scan(b, &mut set);
            }
        }
        for it in &m.items {
            if let Item::Step { body: Body::Block(b), .. } = it {
                scan(b, &mut set);
            }
        }
    });
    let mut variants: HashSet<String> = HashSet::new();
    let mut modnames: HashMap<String, usize> = HashMap::new();
    let mut state_fields: HashMap<String, Vec<String>> = HashMap::new();
    for it in &m.items {
        match it {
            Item::Kind { variants: vs, .. } => {
                for (v, _) in vs {
                    variants.insert(v.clone());
                }
            }
            Item::Event { name, .. } | Item::Error { name, .. } => {
                variants.insert(name.clone());
            }
            Item::Protocol { steps, .. } => {
                // the branch labels of `offer` and `choose` are constants of the session, not bindings
                fn labels(ps: &[ProtoStep], out: &mut HashSet<String>) {
                    for p in ps {
                        match p {
                            ProtoStep::Offer(bs) | ProtoStep::Choose(bs) => {
                                for (n, inner) in bs {
                                    out.insert(n.clone());
                                    labels(inner, out);
                                }
                            }
                            ProtoStep::Loop(inner) => labels(inner, out),
                            _ => {}
                        }
                    }
                }
                labels(steps, &mut variants);
            }
            Item::Fn(f) => {
                modnames.insert(f.sig.name.clone(), f.sig.line);
            }
            Item::Extern { sig, .. } => {
                modnames.insert(sig.name.clone(), sig.line);
            }
            Item::Const { name, line, .. } => {
                modnames.insert(name.clone(), *line);
            }
            Item::Use(p, alias, line) => {
                let n = alias.clone().unwrap_or_else(|| p.last().cloned().unwrap_or_default());
                modnames.insert(n, *line);
            }
            Item::State { name, fields, .. } => {
                state_fields.insert(name.clone(), fields.iter().map(|f| f.name.clone()).collect());
            }
            _ => {}
        }
    }
    let mut w = W {
        scopes: vec![],
        errs: vec![],
        renames: vec![],
        variants: &variants,
        modnames: &modnames,
        fields: HashSet::new(),
        free: HashSet::new(),
    };
    for it in &m.items {
        match it {
            Item::Fn(f) => w.func(&f.sig, &f.body),
            Item::Impl { ty, trait_, members, .. } => {
                w.ty(ty);
                trait_.iter().for_each(|t| w.ty(t));
                for f in members {
                    w.func(&f.sig, &f.body);
                }
            }
            Item::Extern { sig, .. } => {
                for p in &sig.params {
                    if let Param::Named { ty, .. } = p {
                        w.ty(ty);
                    }
                }
                sig.ret.iter().for_each(|t| w.ty(t));
            }
            Item::Step { region, pat, guard, body, line, deadline } => {
                w.fields = state_fields.get(region).cloned().unwrap_or_default().into_iter().collect();
                w.push();
                w.bind_pat(pat, *line);
                guard.iter().for_each(|g| w.cond(g));
                deadline.iter().for_each(|g| w.expr(g));
                w.body(body);
                w.pop();
                w.fields.clear();
            }
            Item::Actor { fields, handlers, fns, .. } => {
                for f in fields {
                    w.ty(&f.ty);
                }
                w.fields = fields.iter().map(|f| f.name.clone()).collect();
                for h in handlers {
                    w.push();
                    w.bind_pat(&h.pat, h.line);
                    h.guard.iter().for_each(|g| w.cond(g));
                    w.body(&h.body);
                    w.pop();
                }
                for f in fns {
                    w.func(&f.sig, &f.body);
                }
                w.fields.clear();
            }
            Item::Const { ty, e, .. } => {
                ty.iter().for_each(|t| w.ty(t));
                w.expr(e);
            }
            Item::TypeAlias { ty, pred, .. } => {
                w.ty(ty);
                pred.iter().for_each(|e| w.expr(e));
            }
            Item::Struct { fields, .. } | Item::State { fields, .. } => fields.iter().for_each(|f| {
                w.ty(&f.ty);
                f.init.iter().for_each(|e| w.expr(e));
            }),
            Item::Event { payload, .. } | Item::Error { payload, .. } => payload.iter().for_each(|(_, t)| w.ty(t)),
            Item::Kind { variants, .. } => variants.iter().for_each(|(_, fs)| fs.iter().for_each(|(_, t)| w.ty(t))),
            Item::Check(e) => w.expr(e),
            Item::Example(ex) => example(&mut w, ex),
            Item::Intent(_, exs, es, ..) => {
                exs.iter().for_each(|ex| example(&mut w, ex));
                es.iter().for_each(|e| w.expr(e));
            }
            Item::Trait { members, .. } => {
                for tm in members {
                    if let TraitMember::Fn(s) = tm {
                        for p in &s.params {
                            if let Param::Named { ty, .. } = p {
                                w.ty(ty);
                            }
                        }
                        s.ret.iter().for_each(|t| w.ty(t));
                    }
                }
            }
            _ => {}
        }
    }
    let mut errs = std::mem::take(&mut w.errs);
    for it in &m.items {
        if let Item::Use(p, alias, line) = it {
            let n = alias.clone().unwrap_or_else(|| p.last().cloned().unwrap_or_default());
            if !w.free.contains(&n) && !n.starts_with('_') {
                errs.push(err(*line, "W-UNUSED-1", format!("`use {}` is never read", p.join("."))));
            }
        }
    }
    errs.sort();
    errs.dedup();
    (errs, std::mem::take(&mut w.renames))
}

fn example(w: &mut W, ex: &Example) {
    for (_, args) in &ex.events {
        args.iter().for_each(|a| w.expr(&a.e));
    }
    ex.value.iter().for_each(|e| w.expr(e));
}

// ------------------------------------------------------------------- World

/// The type `World` as written: exactly `&World`.
fn is_world_ref(t: &Ty) -> bool {
    matches!(t, Ty::Ref(false, x) if matches!(&**x, Ty::Path(p, a) if p.len() == 1 && p[0] == "World" && a.is_empty()))
}

fn mentions_world(t: &Ty) -> bool {
    match t {
        Ty::Path(p, a) => (p.len() == 1 && p[0] == "World") || a.iter().any(mentions_world),
        Ty::Unit => false,
        Ty::Tuple(v) => v.iter().any(mentions_world),
        Ty::Slice(x) | Ty::Chan(x) | Ty::Iter(x) | Ty::Ref(_, x) | Ty::Opt(x) | Ty::Region(x, _) | Ty::Array(x, _) => mentions_world(x),
        Ty::Fn(a, r) => a.iter().any(mentions_world) || mentions_world(r),
        Ty::Union(a, b) => mentions_world(a) || mentions_world(b),
    }
}

fn mentions_chan(t: &Ty) -> bool {
    match t {
        Ty::Chan(_) => true,
        Ty::Path(p, a) => (p.len() == 1 && p[0] == "session") || a.iter().any(mentions_chan),
        Ty::Unit => false,
        Ty::Tuple(v) => v.iter().any(mentions_chan),
        Ty::Slice(x) | Ty::Iter(x) | Ty::Ref(_, x) | Ty::Opt(x) | Ty::Region(x, _) | Ty::Array(x, _) => mentions_chan(x),
        Ty::Fn(a, r) => a.iter().any(mentions_chan) || mentions_chan(r),
        Ty::Union(a, b) => mentions_chan(a) || mentions_chan(b),
    }
}

pub(crate) fn children(e: &Expr) -> Vec<&Expr> {
    use Expr::*;
    match e {
        Name(..) | SelfRef(_) | Int(..) | Float(..) | Str(..) | Bool(..) | None(_) | Unit(_) => vec![],
        Tuple(v, _) | List(v, _) => v.iter().collect(),
        ListRepeat(a, b, _) | Bin(_, a, b, _) => vec![a, b],
        ListComp(a, _, b, c, _) => {
            let mut v: Vec<&Expr> = vec![a, b];
            v.extend(c.iter().map(|x| &**x));
            v
        }
        Map(kv, _) => kv.iter().flat_map(|(k, v)| [k, v]).collect(),
        ChanNew(_, n, _) => n.iter().map(|x| &**x).collect(),
        Call(f, a, _, _) => {
            let mut v: Vec<&Expr> = vec![f];
            v.extend(a.iter().map(|x| &x.e));
            v
        }
        Index(b, ix, _) => {
            let mut v: Vec<&Expr> = vec![b];
            v.extend(ix.iter());
            v
        }
        Slice(b, lo, hi, _) => {
            let mut v: Vec<&Expr> = vec![b];
            v.extend(lo.iter().map(|x| &**x));
            v.extend(hi.iter().map(|x| &**x));
            v
        }
        Member(b, _, _) | TupleMember(b, _, _) | Neg(b, _) | BitNot(b, _) | Borrow(_, b, _) | Recv(b, _) | Comptime(b, _) | Not(b, _) | Cast(b, _, _) => vec![b],
        Catch(a, _, b, _) => vec![a, b],
        Try(a, el, _) => {
            let mut v: Vec<&Expr> = vec![a];
            v.extend(el.iter().map(|x| &**x));
            v
        }
        If(c, a, b, _) => vec![c, a, b],
        Lambda(_, b, _) => vec![b],
    }
}

/// The expressions directly in a statement and its nested statement lists.
pub(crate) fn stmt_parts(s: &Stmt) -> (Vec<&Expr>, Vec<&[Stmt]>) {
    use Stmt::*;
    match s {
        Expr(e) | Emit(e) | Fail(e) | Spawn(e) | Check(e) | Yield(e, _) => (vec![e], vec![]),
        Assign(_, a, b) | Send(a, b) => (vec![a, b], vec![]),
        Let(_, _, e, _) | Var(_, _, e, _) => (vec![e], vec![]),
        If(bs, el) => {
            let mut es: Vec<&crate::ast::Expr> = bs.iter().map(|(c, _)| c).collect();
            let mut ss: Vec<&[Stmt]> = bs.iter().map(|(_, b)| b.as_slice()).collect();
            if let Some(b) = el {
                ss.push(b);
            }
            (std::mem::take(&mut es), ss)
        }
        While(c, b, body) => {
            let mut es: Vec<&crate::ast::Expr> = vec![c];
            es.extend(b.iter());
            (es, vec![body.as_slice()])
        }
        Loop(b, body) => (b.iter().collect(), vec![body.as_slice()]),
        For(_, it, body, _) => (vec![it], vec![body.as_slice()]),
        Match(e, arms, _) => {
            let mut es = vec![e];
            es.extend(arms.iter().filter_map(|a| a.guard.as_ref()));
            (es, arms.iter().map(|a| a.body.as_slice()).collect())
        }
        Return(e, _) => (e.iter().collect(), vec![]),
        Break(_) | Continue(_) | Pass => (vec![], vec![]),
        Select(arms, _) => {
            let mut es = vec![];
            let mut ss: Vec<&[Stmt]> = vec![];
            for a in arms {
                match a {
                    SelArm::Recv(_, c, b) => {
                        es.push(c);
                        ss.push(b);
                    }
                    SelArm::Send(c, v, b) => {
                        es.push(c);
                        es.push(v);
                        ss.push(b);
                    }
                    SelArm::After(e, b) => {
                        es.push(e);
                        ss.push(b);
                    }
                    SelArm::Default(b) => ss.push(b),
                }
            }
            (es, ss)
        }
        Arena(_, e, b, _) | Device(_, e, b, _) => (e.iter().collect(), vec![b.as_slice()]),
        Comptime(b) => (vec![], vec![b.as_slice()]),
    }
}

/// Every expression in a body, with whether it sits inside a lambda.
pub(crate) fn each_expr<'a>(e: &'a Expr, in_lambda: bool, f: &mut dyn FnMut(&'a Expr, bool)) {
    f(e, in_lambda);
    let lam = in_lambda || matches!(e, Expr::Lambda(..));
    for c in children(e) {
        each_expr(c, lam, f);
    }
}

pub(crate) fn stmt_exprs(s: &Stmt) -> Vec<&Expr> {
    stmt_parts(s).0
}

pub(crate) fn stmt_blocks(s: &Stmt) -> Vec<&[Stmt]> {
    stmt_parts(s).1
}

/// Every top-level expression of every statement of a body, at any depth.
pub(crate) fn each_stmt_expr<'a>(ss: &'a [Stmt], f: &mut dyn FnMut(&'a Expr)) {
    for s in ss {
        let (es, nested) = stmt_parts(s);
        for e in es {
            f(e);
        }
        for n in nested {
            each_stmt_expr(n, f);
        }
    }
}

fn each_body_expr<'a>(ss: &'a [Stmt], f: &mut dyn FnMut(&'a Expr, bool)) {
    for s in ss {
        let (es, nested) = stmt_parts(s);
        for e in es {
            each_expr(e, false, f);
        }
        for n in nested {
            each_body_expr(n, f);
        }
    }
}

/// Positions of the `&World` parameters of a signature.
fn world_positions(sig: &FnSig) -> Vec<usize> {
    sig.params
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p, Param::Named { ty, .. } if is_world_ref(ty)))
        .map(|(i, _)| i)
        .collect()
}

fn param_name(p: &Param) -> Option<&str> {
    if let Param::Named { name, .. } = p { Some(name) } else { None }
}

/// P7 (F.4.1): W-WORLD-1 over the module as written. Returns the findings
/// and the free functions and externs that take a `&World`, with the
/// position of that parameter (used to erase it, `erase_world`).
/// The `Cap[DirRead|DirWrite]` parameters of foreign functions (F.5.1): attested by
/// the host and logged with the call, erased like the world (a capability is
/// `()` at run time).
pub fn dir_cap_positions(m: &Module) -> HashMap<String, Vec<usize>> {
    let mut r = HashMap::new();
    for it in &m.items {
        if let Item::Extern { sig, .. } = it {
            let pos: Vec<usize> = sig
                .params
                .iter()
                .enumerate()
                .filter(|(_, p)| matches!(p, Param::Named { ty, .. } if matches!(ty, Ty::Path(p, a) if p.len() == 1 && p[0] == "Cap" && a.len() == 1
                    && matches!(&a[0], Ty::Path(q, b) if q.len() == 1 && b.is_empty() && (q[0] == "DirRead" || q[0] == "DirWrite")))))
                .map(|(i, _)| i)
                .collect();
            if !pos.is_empty() {
                r.insert(sig.name.clone(), pos);
            }
        }
    }
    r
}

pub fn world(m: &Module) -> (Vec<String>, HashMap<String, Vec<usize>>) {
    let mut errs: Vec<String> = vec![];
    let mut wf: HashMap<String, Vec<usize>> = HashMap::new();
    // 1. the type name is reserved (F, "Reserved words"; W-NAME-2)
    for it in &m.items {
        let (n, l) = match it {
            Item::Struct { name, line, .. } | Item::Kind { name, line, .. } | Item::TypeAlias { name, line, .. }
            | Item::State { name, line, .. } | Item::Event { name, line, .. } | Item::Error { name, line, .. }
            | Item::Actor { name, line, .. } | Item::Trait { name, line, .. } => (name, *line),
            _ => continue,
        };
        if n == "World" {
            errs.push(err(l, "W-NAME-2", "`World` is a version 0.9 type (F.4.1) and cannot be declared".into()));
        }
    }
    // 2. holders: main and task functions; externs take it first
    for it in &m.items {
        match it {
            Item::Fn(f) => {
                let pos = world_positions(&f.sig);
                if pos.is_empty() {
                    continue;
                }
                let task = f.sig.params.iter().any(|p| matches!(p, Param::Named { ty, .. } if mentions_chan(ty)));
                if f.sig.name != "main" && !task {
                    errs.push(err(f.sig.line, "W-WORLD-1", format!("`{}` takes a &World, but only `fn main` and task functions (those with a channel parameter) may hold one", f.sig.name)));
                }
                if pos.len() > 1 {
                    errs.push(err(f.sig.line, "W-WORLD-1", "a function takes at most one &World".into()));
                }
                wf.insert(f.sig.name.clone(), pos);
            }
            Item::Extern { sig, line, .. } => {
                let pos = world_positions(sig);
                if pos != [0] {
                    errs.push(err(*line, "W-WORLD-1", format!("the foreign function `{}` must take `&World` as its first parameter, and only there", sig.name)));
                }
                wf.insert(sig.name.clone(), pos);
            }
            _ => {}
        }
    }
    // 3. World is written only as `&World` in a parameter position of the above
    let mut other: Vec<(usize, &Ty)> = vec![];
    for it in &m.items {
        match it {
            Item::Struct { fields, line, .. } | Item::State { fields, line, .. } => fields.iter().for_each(|f| other.push((*line, &f.ty))),
            Item::Event { payload, line, .. } | Item::Error { payload, line, .. } => payload.iter().for_each(|(_, t)| other.push((*line, t))),
            Item::Kind { variants, line, .. } => variants.iter().flat_map(|(_, p)| p.iter()).for_each(|(_, t)| other.push((*line, t))),
            Item::TypeAlias { ty, line, .. } => other.push((*line, ty)),
            Item::Const { ty: Some(t), line, .. } => other.push((*line, t)),
            Item::Fn(f) => {
                if let Some(r) = &f.sig.ret {
                    other.push((f.sig.line, r));
                }
                for (i, p) in f.sig.params.iter().enumerate() {
                    if let Param::Named { ty, .. } = p
                        && !(is_world_ref(ty) && wf.get(&f.sig.name).is_some_and(|v| v.contains(&i)))
                    {
                        other.push((f.sig.line, ty));
                    }
                }
                collect_body_tys(&f.body, &mut other);
            }
            Item::Extern { sig, line, .. } => {
                sig.ret.iter().for_each(|r| other.push((*line, r)));
                for (i, p) in sig.params.iter().enumerate() {
                    if let Param::Named { ty, .. } = p
                        && !(i == 0 && is_world_ref(ty))
                    {
                        other.push((*line, ty));
                    }
                }
            }
            Item::Impl { members, .. } => {
                for f in members {
                    for p in &f.sig.params {
                        if let Param::Named { ty, .. } = p {
                            other.push((f.sig.line, ty));
                        }
                    }
                    f.sig.ret.iter().for_each(|r| other.push((f.sig.line, r)));
                    collect_body_tys(&f.body, &mut other);
                }
            }
            Item::Actor { fields, fns, line, .. } => {
                fields.iter().for_each(|f| other.push((*line, &f.ty)));
                for f in fns {
                    for p in &f.sig.params {
                        if let Param::Named { ty, .. } = p {
                            other.push((f.sig.line, ty));
                        }
                    }
                    collect_body_tys(&f.body, &mut other);
                }
            }
            _ => {}
        }
    }
    for (l, t) in other {
        if mentions_world(t) {
            errs.push(err(l, "W-WORLD-1", "`World` appears only as the type `&World` of a parameter of `fn main`, a task function or a foreign function; it is never stored, returned, sent or nested".into()));
        }
    }
    // 4. uses: a World parameter is only passed, as the whole argument, to a
    //    function that takes one; never captured by a lambda
    for it in &m.items {
        let Item::Fn(f) = it else { continue };
        let held: Vec<String> = wf
            .get(&f.sig.name)
            .map(|pos| pos.iter().filter_map(|i| param_name(&f.sig.params[*i]).map(String::from)).collect())
            .unwrap_or_default();
        let body: Vec<Stmt> = match &f.body {
            Body::Block(s) => s.clone(),
            Body::Arrow(e) => vec![Stmt::Expr(e.clone())],
        };
        let mut total: HashMap<String, usize> = HashMap::new();
        let mut passed: HashMap<String, usize> = HashMap::new();
        let mut lines: Vec<(usize, String)> = vec![];
        each_body_expr(&body, &mut |e, in_lambda| match e {
            Expr::Name(n, l) => {
                if held.contains(n) {
                    *total.entry(n.clone()).or_default() += 1;
                    if in_lambda {
                        lines.push((*l, format!("`{n}` is a &World and a closure cannot capture it (authority held by a closure must be visible)")));
                    }
                }
            }
            Expr::Call(callee, args, l, _) => {
                if let Expr::Name(cn, _) = &**callee
                    && let Some(pos) = wf.get(cn)
                {
                    let ext = matches!(m.items.iter().find(|i| matches!(i, Item::Extern { sig, .. } if &sig.name == cn)), Some(_));
                    let params: &Vec<Param> = m
                        .items
                        .iter()
                        .find_map(|i| match i {
                            Item::Fn(g) if &g.sig.name == cn => Some(&g.sig.params),
                            Item::Extern { sig, .. } if &sig.name == cn => Some(&sig.params),
                            _ => None,
                        })
                        .unwrap();
                    if args.len() != params.len() {
                        if !pos.is_empty() {
                            lines.push((*l, format!("a call to `{cn}` passes no `&World` (it takes {} argument(s); a call that has authority passes it explicitly)", params.len())));
                        }
                        return;
                    }
                    for i in pos {
                        match &args[*i].e {
                            Expr::Name(a, _) if held.contains(a) => {
                                *passed.entry(a.clone()).or_default() += 1;
                                if in_lambda {
                                    // reported through `total` above
                                }
                            }
                            _ => lines.push((*l, format!("the &World argument of `{cn}` must be the caller's own &World parameter by name{}", if ext { " (a foreign call is authority, F.4.1)" } else { "" }))),
                        }
                    }
                }
            }
            _ => {}
        });
        for n in &held {
            let t = total.get(n).copied().unwrap_or(0);
            let p = passed.get(n).copied().unwrap_or(0);
            if t > p {
                lines.push((f.sig.line, format!("`{n}` is a &World: it may only be passed whole to a function that takes one, never stored, returned, sent or used otherwise")));
            }
        }
        // a call that needs a World from a context that holds none
        if held.is_empty() {
            each_body_expr(&body, &mut |e, _| {
                if let Expr::Call(callee, _, l, _) = e
                    && let Expr::Name(cn, _) = &**callee
                    && wf.get(cn).is_some_and(|p| !p.is_empty())
                {
                    lines.push((*l, format!("`{}` needs a &World and `{}` holds none: only `fn main` and task functions hold one", cn, f.sig.name)));
                }
            });
        }
        for (l, msg) in lines {
            errs.push(err(l, "W-WORLD-1", msg));
        }
    }
    errs.sort();
    errs.dedup();
    wf.retain(|_, v| !v.is_empty());
    (errs, wf)
}

pub(crate) fn collect_body_tys<'a>(b: &'a Body, out: &mut Vec<(usize, &'a Ty)>) {
    fn go<'a>(ss: &'a [Stmt], out: &mut Vec<(usize, &'a Ty)>) {
        for s in ss {
            if let Stmt::Let(_, Some(t), _, l) | Stmt::Var(_, Some(t), _, l) = s {
                out.push((*l, t));
            }
            for n in stmt_parts(s).1 {
                go(n, out);
            }
        }
    }
    if let Body::Block(s) = b {
        go(s, out);
    }
}

/// Erase `&World` (F.4.1: "erased at run time"): remove the parameter from
/// every free function and extern that has one, and the argument from every
/// call to it. After this the module is a 0.8 module as far as the
/// evaluator, the emitters and the Form are concerned.
pub fn erase_world(m: Module, wf: &HashMap<String, Vec<usize>>) -> Result<Module, String> {
    let mut v = serde_json::to_value(&m).map_err(|e| e.to_string())?;
    fn go(v: &mut serde_json::Value, wf: &HashMap<String, Vec<usize>>) {
        // label sets and `release`/`endorse` (F.4.2) are checked on the surface
        // module and carry no run-time meaning: erase them
        if let serde_json::Value::Object(o) = v {
            if let Some(serde_json::Value::Array(a)) = o.get("Region")
                && a.len() == 2
                && a[1].as_str().is_some_and(|s| s.starts_with("@{"))
            {
                *v = a[0].clone();
                return go(v, wf);
            }
            if let Some(serde_json::Value::Array(a)) = o.get("Call")
                && a.len() == 4
                && let Some(n) = a[0].get("Name").and_then(|n| n.get(0)).and_then(|s| s.as_str())
                && (n == "@release" || n == "@endorse" || n == "@iflet")
                && let Some(arg) = a[1].get(0).and_then(|x| x.get("e"))
            {
                *v = arg.clone();
                return go(v, wf);
            }
        }
        match v {
            serde_json::Value::Object(o) => {
                if let Some(serde_json::Value::Array(a)) = o.get_mut("Call")
                    && a.len() == 4
                {
                    let name = a[0].get("Name").and_then(|n| n.get(0)).and_then(|s| s.as_str()).map(String::from);
                    if let Some(n) = name
                        && let Some(pos) = wf.get(&n)
                        && let serde_json::Value::Array(args) = &mut a[1]
                    {
                        for i in pos.iter().rev() {
                            if *i < args.len() {
                                args.remove(*i);
                            }
                        }
                    }
                }
                for x in o.values_mut() {
                    go(x, wf);
                }
            }
            serde_json::Value::Array(a) => a.iter_mut().for_each(|x| go(x, wf)),
            _ => {}
        }
    }
    go(&mut v, wf);
    let mut m2: Module = serde_json::from_value(v).map_err(|e| e.to_string())?;
    m2.items.retain(|i| !matches!(i, Item::Struct { name, .. } if name.starts_with("@label:")));
    for it in &mut m2.items {
        match it {
            Item::Fn(f) => {
                if let Some(pos) = wf.get(&f.sig.name) {
                    for i in pos.iter().rev() {
                        f.sig.params.remove(*i);
                    }
                }
            }
            Item::Extern { sig, .. } => {
                if let Some(pos) = wf.get(&sig.name) {
                    for i in pos.iter().rev() {
                        sig.params.remove(*i);
                    }
                }
            }
            _ => {}
        }
    }
    m2.v09_hygiene = m.v09_hygiene;
    m2.x09 = m.x09;
    Ok(m2)
}

/// The inverse of `erase_world` for foreign functions: the module as the Form
/// carries it (F.4.1: `&World` erased) is written back as source that P7
/// admits. Every `extern` gets `w: &World` as its first parameter, every
/// function that calls one holds `w: &World` as its first parameter, and
/// every such call passes `w`. (Only `fn main` and task functions may call a
/// foreign function, so no other function is touched.)
pub fn restore_world(m: Module) -> Result<Module, String> {
    use serde_json::{Value, json};
    let mut v = serde_json::to_value(&m).map_err(|e| e.to_string())?;
    let mut ext: HashSet<String> = HashSet::new();
    if let Some(Value::Array(items)) = v.get("items") {
        for it in items {
            if let Some(n) = it.get("Extern").and_then(|e| e.get("sig")).and_then(|s| s.get("name")).and_then(|n| n.as_str()) {
                ext.insert(n.to_string());
            }
        }
    }
    if ext.is_empty() {
        return Ok(m);
    }
    let world_param = json!({"Named": {"comptime": false, "label": null, "name": "w", "ty": {"Ref": [false, {"Path": [["World"], []]}]}}});
    fn calls(v: &mut Value, ext: &HashSet<String>, found: &mut bool) {
        match v {
            Value::Object(o) => {
                if let Some(Value::Array(a)) = o.get_mut("Call")
                    && a.len() == 4
                    && let Some(n) = a[0].get("Name").and_then(|n| n.get(0)).and_then(|s| s.as_str())
                    && ext.contains(n)
                {
                    let line = a[0]["Name"][1].clone();
                    if let Value::Array(args) = &mut a[1] {
                        args.insert(0, json!({"label": null, "e": {"Name": ["w", line]}}));
                    }
                    *found = true;
                }
                for x in o.values_mut() {
                    calls(x, ext, found);
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|x| calls(x, ext, found)),
            _ => {}
        }
    }
    if let Some(Value::Array(items)) = v.get_mut("items") {
        for it in items {
            if let Some(e) = it.get_mut("Extern") {
                if let Some(Value::Array(ps)) = e.get_mut("sig").and_then(|s| s.get_mut("params")) {
                    ps.insert(0, world_param.clone());
                }
            } else if let Some(f) = it.get_mut("Fn") {
                let mut found = false;
                if let Some(b) = f.get_mut("body") {
                    calls(b, &ext, &mut found);
                }
                if found && let Some(Value::Array(ps)) = f.get_mut("sig").and_then(|s| s.get_mut("params")) {
                    ps.insert(0, world_param.clone());
                }
            }
        }
    }
    serde_json::from_value(v).map_err(|e| e.to_string())
}

// ------------------------------------------------------ P1 transactional calls

/// Whether an expression, in evaluation order, performs a non-undoable
/// effect itself (a receive or a foreign call) or calls a function that does.
struct Tx<'a> {
    externs: &'a HashSet<String>,
    effectful: &'a HashSet<String>,
    errs: Vec<String>,
    fname: String,
    /// compute mode: only look for effects, report nothing
    quiet: bool,
    found: bool,
}

impl Tx<'_> {
    fn effect_name(&self, e: &Expr) -> bool {
        match e {
            Expr::Recv(..) => true,
            Expr::Call(f, _, _, _) => match &**f {
                Expr::Name(n, _) => self.externs.contains(n) || self.effectful.contains(n),
                Expr::Member(_, m, _) => self.effectful.contains(m),
                _ => false,
            },
            _ => false,
        }
    }
    /// Walk `e` in evaluation order (arguments before the call), with the
    /// taint flag in and out.
    fn expr(&mut self, e: &Expr, t: &mut bool) {
        // a `try e else h` handles the failure of e: no fail point
        let handled = matches!(e, Expr::Try(_, Some(_), _));
        if matches!(e, Expr::Lambda(..)) {
            return; // a closure body runs when called; it is a call to a function value
        }
        for c in children(e) {
            self.expr(c, t);
        }
        if self.effect_name(e) {
            self.found = true;
            *t = true;
        }
        if let Expr::Try(_, None, l) = e
            && !handled
            && *t
            && !self.quiet
        {
            self.errs.push(err(*l, "W-TX-1", format!("`{}`: `try` can fail here after a channel, foreign or other non-undoable effect; return a Result value instead (F.4.4)", self.fname)));
        }
    }
    /// Returns (taint after, falls through).
    fn block(&mut self, ss: &[Stmt], mut t: bool) -> (bool, bool) {
        for s in ss {
            let (t2, ft) = self.stmt(s, t);
            t = t2;
            if !ft {
                return (t, false);
            }
        }
        (t, true)
    }
    fn stmt(&mut self, s: &Stmt, mut t: bool) -> (bool, bool) {
        use Stmt::*;
        match s {
            Fail(e) => {
                self.expr(e, &mut t);
                if t && !self.quiet {
                    self.errs.push(err(0, "W-TX-1", format!("`{}`: `fail` after a channel, foreign or other non-undoable effect; return a Result value instead (F.4.4)", self.fname)));
                }
                (t, false)
            }
            Return(e, _) => {
                e.iter().for_each(|x| self.expr(x, &mut t));
                (t, false)
            }
            Send(a, b) => {
                self.expr(a, &mut t);
                self.expr(b, &mut t);
                self.found = true;
                (true, true)
            }
            Spawn(e) => {
                self.expr(e, &mut t);
                self.found = true;
                (true, true)
            }
            Select(arms, _) => {
                self.found = true;
                let mut out = true;
                for a in arms {
                    let b = match a {
                        SelArm::Recv(_, _, b) | SelArm::Send(_, _, b) | SelArm::After(_, b) | SelArm::Default(b) => b,
                    };
                    let (t2, _) = self.block(b, true);
                    out |= t2;
                }
                (out, true)
            }
            If(bs, el) => {
                let mut out = false;
                let mut any_ft = false;
                for (c, b) in bs {
                    self.expr(c, &mut t);
                    let (t2, ft) = self.block(b, t);
                    if ft {
                        out |= t2;
                        any_ft = true;
                    }
                }
                match el {
                    Some(b) => {
                        let (t2, ft) = self.block(b, t);
                        if ft {
                            out |= t2;
                            any_ft = true;
                        }
                    }
                    None => {
                        out |= t;
                        any_ft = true;
                    }
                }
                (out || (!any_ft && t), any_ft)
            }
            While(c, b, body) => {
                self.expr(c, &mut t);
                b.iter().for_each(|x| self.expr(x, &mut t));
                let (t1, _) = self.block(body, t);
                let t_in = t | t1;
                let (t2, _) = self.block(body, t_in);
                (t_in | t2, true)
            }
            Loop(b, body) => {
                b.iter().for_each(|x| self.expr(x, &mut t));
                let (t1, _) = self.block(body, t);
                let t_in = t | t1;
                let (t2, _) = self.block(body, t_in);
                (t_in | t2, true)
            }
            For(_, it, body, _) => {
                self.expr(it, &mut t);
                let (t1, _) = self.block(body, t);
                let t_in = t | t1;
                let (t2, _) = self.block(body, t_in);
                (t_in | t2, true)
            }
            Match(e, arms, _) => {
                self.expr(e, &mut t);
                let mut out = false;
                let mut any_ft = false;
                for a in arms {
                    let mut ta = t;
                    if let Some(g) = &a.guard {
                        self.expr(g, &mut ta);
                    }
                    let (t2, ft) = self.block(&a.body, ta);
                    if ft {
                        out |= t2;
                        any_ft = true;
                    }
                }
                (out, any_ft || arms.is_empty())
            }
            Arena(_, e, b, _) | Device(_, e, b, _) => {
                e.iter().for_each(|x| self.expr(x, &mut t));
                self.block(b, t)
            }
            Comptime(b) => self.block(b, t),
            Expr(e) | Emit(e) | Check(e) | Yield(e, _) => {
                self.expr(e, &mut t);
                (t, true)
            }
            Assign(_, a, b) => {
                self.expr(b, &mut t);
                self.expr(a, &mut t);
                (t, true)
            }
            Let(_, _, e, _) | Var(_, _, e, _) => {
                self.expr(e, &mut t);
                (t, true)
            }
            Break(_) | Continue(_) | Pass => (t, true),
        }
    }
}

pub(crate) fn fn_decls(m: &Module) -> Vec<(&FnDecl, bool)> {
    let mut v = vec![];
    for it in &m.items {
        match it {
            Item::Fn(f) => v.push((f, false)),
            Item::Impl { members, .. } => members.iter().for_each(|f| v.push((f, true))),
            Item::Actor { fns, .. } => fns.iter().for_each(|f| v.push((f, true))),
            _ => {}
        }
    }
    v
}

/// A version 0.9 construct that the emitters and the Lean elaboration do not
/// translate: the reference evaluator runs it, every other target reports it
/// as not implemented (never a different meaning). `None` when the module has none.
pub fn unemittable(m: &Module) -> Option<String> {
    unemittable_for(m, false)
}

/// As `unemittable`; with `lean`, also the contexts the Lean elaboration does
/// not translate (`Prompt`, `Header`; the emitters translate them, F.12).
pub fn unemittable_for(m: &Module, lean: bool) -> Option<String> {
    if !m.x09.ct.is_empty() {
        return Some(format!("W-CTFN-2 ct fn is not emitted for this target (`{}`): 10.5 states no constant-time guarantee for any target", m.x09.ct.iter().next().unwrap()));
    }
    let mut found: Option<String> = None;
    let mut scan = |e: &Expr| {
        crate::v09::each_expr(e, false, &mut |x, _| {
            let what = match x {
                Expr::Call(f, _, _, _) => match &**f {
                    Expr::Member(o, c, _) if c == "cmd" && matches!(&**o, Expr::Name(n, _) if n == "Shell") => Some("Shell.cmd"),
                    Expr::Member(_, c, _) if c == "narrow" => Some("Cap.narrow"),
                    Expr::Name(n, _) if n == "ct_choose" => Some("ct_choose"),
                    Expr::Name(n, _) if n.starts_with("@contract") => Some("a contract clause"),
                    _ => None,
                },
                Expr::Member(o, c, _) if (c == "encode" || c == "lit") && matches!(&**o, Expr::Name(n, _) if n == "ShellArg") => Some("ShellArg text"),
                Expr::Member(o, c, _) if lean && (c == "encode" || c == "lit") && matches!(&**o, Expr::Name(n, _) if n == "Prompt" || n == "Header") => Some("Prompt/Header text"),
                _ => None,
            };
            if let Some(w) = what && found.is_none() {
                found = Some(w.to_string());
            }
        });
    };
    for (f, _) in fn_decls(m) {
        match &f.body {
            Body::Arrow(e) => scan(e),
            Body::Block(b) => each_stmt_expr(b, &mut |e| scan(e)),
        }
    }
    for it in &m.items {
        if let Item::Step { body, .. } = it {
            match body {
                Body::Arrow(e) => scan(e),
                Body::Block(b) => each_stmt_expr(b, &mut |e| scan(e)),
            }
        }
    }
    found.map(|w| format!("W-RES-1 `{w}` (Appendix F) is run by the reference evaluator only; this target does not translate it, so it is reserved, not implemented"))
}

/// Whether a module has a function that can fail after taking a `&mut`
/// argument, so that F.4.4's restore changes its meaning (the emitters do not
/// implement the restore, so they do not yet print such a module).
pub fn needs_restore(m: &Module) -> bool {
    fn fails(ss: &[Stmt]) -> bool {
        ss.iter().any(|s| {
            matches!(s, Stmt::Fail(_))
                || { let (es, ns) = stmt_parts(s); es.iter().any(|e| has_try(e)) || ns.iter().any(|n| fails(n)) }
        })
    }
    fn has_try(e: &Expr) -> bool {
        let mut hit = false;
        each_expr(e, false, &mut |x, _| hit |= matches!(x, Expr::Try(..)));
        hit
    }
    fn_decls(m).iter().any(|(f, _)| {
        let mut_arg = f.sig.params.iter().any(|p| matches!(p, Param::SelfRef(true)) || matches!(p, Param::Named { ty: Ty::Ref(true, _), .. }));
        let may_fail = match &f.body {
            Body::Block(b) => fails(b),
            Body::Arrow(e) => has_try(e),
        };
        mut_arg && may_fail
    })
}

/// F.4.4, W-TX-1 over the module as written.
pub fn transactional(m: &Module) -> Vec<String> {
    let externs: HashSet<String> = m.items.iter().filter_map(|i| if let Item::Extern { sig, .. } = i { Some(sig.name.clone()) } else { None }).collect();
    let decls = fn_decls(m);
    // functions that transitively perform a non-undoable effect
    let mut effectful: HashSet<String> = HashSet::new();
    loop {
        let before = effectful.len();
        for (f, _) in &decls {
            if effectful.contains(&f.sig.name) {
                continue;
            }
            let ext = externs.clone();
            let eff = effectful.clone();
            let mut tx = Tx { externs: &ext, effectful: &eff, errs: vec![], fname: f.sig.name.clone(), quiet: true, found: false };
            match &f.body {
                Body::Block(b) => {
                    tx.block(b, false);
                }
                Body::Arrow(e) => {
                    let mut t = false;
                    tx.expr(e, &mut t);
                }
            }
            if tx.found {
                effectful.insert(f.sig.name.clone());
            }
        }
        if effectful.len() == before {
            break;
        }
    }
    let mut errs = vec![];
    for (f, _) in &decls {
        let mut tx = Tx { externs: &externs, effectful: &effectful, errs: vec![], fname: f.sig.name.clone(), quiet: false, found: false };
        match &f.body {
            Body::Block(b) => {
                tx.block(b, false);
            }
            Body::Arrow(e) => {
                let mut t = false;
                tx.expr(e, &mut t);
            }
        }
        for mut e in tx.errs {
            // a `fail` statement carries no line of its own: use the function's
            if e.starts_with("line 0:") {
                e = format!("line {}:{}", f.sig.line, &e["line 0:".len()..]);
            }
            errs.push(e);
        }
    }
    errs
}

/// F.1 status. `IMPLEMENTED` rules are enforced by this checker; every
/// `UNIMPLEMENTED` rule is reserved, not implemented when a program uses its
/// construct, or is not enforced at all (stated in the second field).
pub const IMPLEMENTED: &[(&str, &str)] = &[
    ("W-RES-1", "reserved words and type names of unimplemented constructs are reserved, not implemented"),
    ("W-LEX-4", "bidirectional, invisible, control, noncharacter, tag and unassigned-plane characters in strings and comments"),
    ("W-UNUSED-1", "unused let, var, parameter, pattern binding and use; `_` prefix is the one spelling"),
    ("W-UNUSED-2", "a constant boolean condition (literal or `not` of a literal)"),
    ("W-ASSIGN-1", "straight-line overwritten store within one block"),
    ("W-WORLD-1", "&World: reserved type, held only by main and task functions, first parameter of every extern, passed whole by name, never stored or captured; erased before evaluation and emission (T53 unproved)"),
    ("W-TX-1", "no `fail` or unhandled `try` after a channel, spawn, receive, select, foreign call or call that does one; a failed call restores its &mut arguments (T47 unproved). Linear consumption is not tracked."),
    ("W-IFC-1", "labeled value reaching emit, send, a public store or a call parameter, incl. implicit flows through the control label; provenance in the message"),
    ("W-IFC-2", "a labeled value deciding a loop count or a for range"),
    ("W-IFC-4", "an operation that can trap, or a fail, with a labeled deciding operand or control context; checked per call instantiation"),
    ("W-IFC-5", "release without a held Cap[AUTH] or decided by untrusted data; endorse without `label untrusted by AUTH` and its capability (T49 to T51 unproved)"),
    ("W-IFC-6", "a label used but not declared"),
    ("W-NAME-2", "label, release, endorse, ct, requires, ensures, ghost and decreases used as ordinary names in a 0.9 module"),
    ("W-CAP-3", "path-taking foreign functions need Cap[DirRead|DirWrite]; DirRead and DirWrite are structs with `root: text`; narrow only extends a relative path"),
    ("W-CAP-1 (closures)", "a closure capturing &World or a capability, by declared type (labeled data is W-IFC-1)"),
    ("W-CAP-4 and W-CAP-5 (model)", "caretaker log fold: record of a kept use, idempotent revoke, no un-revoke (caretaker.rs); `host.revoke` itself is reserved, not implemented"),
    ("W-LEN-1", "`len` on text in a 0.9 module; `len_bytes` and `len_chars` name the unit"),
    ("S3 contexts", "ShellArg, Prompt, Header (CR and LF trap, F.5.7) and Shell.cmd as typed output"),
    ("W-BUD-1 (model)", "a call to an `extern \"model\"` function outside `with budget`"),
    ("W-CTX-2", "a text parameter of an `extern \"model\"` function that is not `&Text[Prompt]`; an untrusted argument of a model call outside that parameter (label analysis, F.5.6)"),
    ("W-IFC-7", "a field of an `extern \"model\" event`, or a model function result, declared without `untrusted`; a `Cap` or `Guarded` field on such an event; `extern` events of another ABI (F.5.6)"),
    ("W-CTX-3", "a model call control dependent on untrusted data (branch, loop, early exit) or with a Text[Prompt] argument steered outside Prompt.encode (label analysis, F.5.6)"),
    ("W-TRI-1", "by W-IFC-1 and W-IFC-5, with no separate check (F.7.1)"),
    ("W-CTFN-1", "a secret-labeled value deciding control flow, an index, a shift amount, a divisor or a non-ct call argument inside a ct fn"),
    ("W-CTFN-2", "emitters and the Lean elaboration do not yet translate a module with a ct fn"),
    ("W-CON-1", "requires clauses false in the literal and integer-bound fragment"),
    ("W-CON-2", "contract clause position, `result`, purity of a clause"),
    ("W-DEC-1", "decreases: linear measures over unsigned parameters, lexicographic decrease at every call of the recursive group"),
    ("W-IFLET-1", "an if let operand that is not a comparison (an irrefutable pattern is W-MATCH-2)"),
    ("W-TPL-1", "template renderer is neither text nor a context"),
    ("W-TPL-2", "malformed template field or brace"),
    ("W-GHOST-1", "a ghost name used in code that runs; unneeded ghost code is erased"),
    ("W-PIN-1", "rce manifest-check: canonical manifest, module and read hashes, pinned dependencies and imports, no glob or relative path, no cycle; signed advisories (Ed25519) and the unsafe list"),
    ("W-NAME-3", "shadowing of any enclosing local, parameter, state field or module-level name"),
];

pub const UNIMPLEMENTED: &[(&str, &str)] = &[
    ("F.3.3 qualified variants", "variants are still written unqualified (W-PAT-1 still applies); not enforced"),
    ("F.3.3 W-LEX-4 unassigned scalars", "only noncharacters, the tag block and planes 4 to 13 require change; no Unicode general-category table is vendored"),
    ("F.3.3 tokenizer check (C-F2)", "the longest-token rule is not checked against the reference lexer"),
    ("F.3.4 arithmetic additions", "distinct bounded integer types, div_floor/div_trunc/div_euclid/div_exact and named int/float conversions are not implemented (a call of an unknown method requires change, naming the method); check C-F1 not done"),
    ("F.3.2 W-UNUSED-2 beyond literals", "only literal boolean conditions require change, not named comptime constants"),
    ("F.3.2 W-ASSIGN-1 beyond one block", "only an overwrite later in the same block with no mention between requires change"),
    ("F.4.1 World in event payloads", "a &World is a parameter of main, task functions and externs only"),
    ("F.4.2 declassify and label polymorphism (S1)", "`declassify` is reserved, not implemented; labels are not generic parameters; Secret[T] keeps its 0.8 rules (W-SEC-1); labels are not yet part of the definition hash"),
    ("F.4.3 contracts, measures, ghost and labels in the hash", "those constructs do not exist yet; module hash and manifest (F.5.3) are not implemented"),
    ("F.4.3 api-diff rows for requires/ensures and escaping closures", "no contracts or escaping-closure marker exist yet"),
    ("F.4.4 linear consumption", "W-TX-1 does not track a consumed linear value (the value is consumed on failure either way)"),
    ("F.4.4 emitters and Lean elaboration", "do not yet translate a 0.9 module with a function that can fail after taking a &mut argument (W-RES-1)"),
    ("F.5.1 host.revoke, closures capturing labeled or capability values found by inference", "host.revoke is reserved, not implemented; capture is checked by declared type"),
    ("F.5.2 strict parsing, migrations, R.check", "not implemented"),
    ("F.5.5 manifest-declared `model`, decreases-based counter elision, allocation bound", "`extern \"model\"` stands for the manifest declaration; no counter elision; Appendix F states no allocation bound"),
    ("F.5.3 provenance, cache rule, advisory key distribution", "no SLSA provenance record, no artifact cache and no way to publish advisory keys; the manifest is checked by `rce manifest-check`, not by `rce check`"),
    ("F.5.4 ct fn beyond declared labels", "W-CTFN-1 is a label analysis: a value that is secret but not labeled `secret` is not seen; no processor-level claim (T45 unproved)"),
    ("F.6.1 Template values", "`Template[t]` and a template that is not rendered at once are reserved, not implemented; `text.t\"..\"` and `C.t\"..\"` are implemented"),
    ("F.6.2 contracts (P2)", "R.check, opaque invariant types, static use of callee contracts and contracts on function values are not implemented; emitters do not yet print a contract (W-RES-1)"),
    ("F.6.3 ghost (P3)", "`ghost` parameters, `mathint` and quantifiers are reserved, not implemented"),
    ("F.6.4 decreases (P4)", "measures on `for` and `loop`, structural and mathint measures are not implemented (a `decreases` there is reserved, not implemented); a `while` measure is decided over a straight-line body"),
    ("F.6.5 if let chains", "`and let P = e` is reserved, not implemented; the single `if let P = e [and c]... else` is implemented"),
    ("F.6.6 trait laws (P9)", "`law` inside a trait is reserved, not implemented; `law` items over reachable states (6.11) are implemented"),
    ("F.7 broker contract (S7)", "tooling contract only, nothing in the checker"),
    ("T42 to T56", "stated, unproved; nothing here proves any of them"),
];

#[allow(dead_code)]
fn _unused(_: &BTreeSet<String>) {}
