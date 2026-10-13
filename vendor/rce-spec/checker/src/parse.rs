//! Lexer, layout pass (spec 2.6) and recursive-descent parser for the
//! grammar of spec section 3. The first error refuses the text.
use crate::ast::*;

pub const KEYWORDS: &[&str] = &[
    "actor", "after", "and", "arena", "as", "bound", "break", "catch", "chan", "check", "comptime",
    "const", "continue", "default", "device", "elif", "else", "emit", "error", "event", "example",
    "extern", "fail", "false", "fn", "for", "forall", "from", "if", "impl", "in", "intent", "iter",
    "kind", "let", "linear", "loop", "match", "module", "mut", "none", "not", "on", "or", "pass",
    "pub", "recv", "refused", "return", "select", "self", "send", "spawn", "state", "step",
    "struct", "then", "to", "trait", "true", "try", "type", "use", "var", "while", "with", "yield",
    // draft 0.6 (spec/CONCURRENCY-0.6.md)
    "par", "branch", "nondet", "budget", "deadline", "priority",
];

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Id(String),
    Int(i128),
    Float(f64),
    Str(String),
    P(&'static str),
    Nl,
    Indent,
    Dedent,
    Eof,
}

const PUNCT: &[&str] = &[
    "<<=", ">>=", "..=", "==", "!=", "<=", ">=", "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=",
    "|=", "^=", "<-", "->", "=>", "..", "+", "-", "*", "/", "%", "&", "|", "^", "~", "<", ">", "=",
    "!", "?", "@", ":", ";", ",", ".", "(", ")", "[", "]", "{", "}",
];

pub fn lex(src: &str) -> Result<Vec<(Tok, usize)>, String> {
    if src.starts_with('\u{feff}') {
        return Err("line 1: W-LEX-1 byte-order mark".into());
    }
    if src.contains('\t') {
        return Err("W-LEX-1 tab character".into());
    }
    let s: Vec<char> = src.chars().collect();
    let mut out: Vec<(Tok, usize)> = vec![];
    let mut i = 0;
    let mut line = 1;
    let mut depth = 0i32;
    let mut stack = vec![0usize];
    let mut at_line_start = true;
    while i < s.len() {
        if at_line_start && depth == 0 {
            // measure indentation; skip blank / comment-only lines
            let mut j = i;
            let mut w = 0;
            while j < s.len() && s[j] == ' ' {
                j += 1;
                w += 1;
            }
            if j < s.len() && s[j] == '\r' && j + 1 < s.len() && s[j + 1] == '\n' {
                j += 1;
            }
            if j >= s.len() {
                i = j;
                break;
            }
            if s[j] == '\n' {
                i = j + 1;
                line += 1;
                continue;
            }
            if s[j] == '#' {
                while j < s.len() && s[j] != '\n' {
                    j += 1;
                }
                i = j;
                continue;
            }
            if s[j] == '\r' {
                return Err(format!("line {line}: W-LEX-1 bare CR"));
            }
            let top = *stack.last().unwrap();
            if w > top {
                stack.push(w);
                out.push((Tok::Indent, line));
            } else if w < top {
                while w < *stack.last().unwrap() {
                    stack.pop();
                    out.push((Tok::Dedent, line));
                }
                if w != *stack.last().unwrap() {
                    return Err(format!("line {line}: W-LEX-2 inconsistent dedent"));
                }
            }
            i = j;
            at_line_start = false;
            continue;
        }
        let c = s[i];
        if c == ' ' {
            i += 1;
            continue;
        }
        if c == '#' {
            while i < s.len() && s[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '\r' {
            if i + 1 < s.len() && s[i + 1] == '\n' {
                i += 1;
                continue;
            }
            return Err(format!("line {line}: W-LEX-1 bare CR"));
        }
        if c == '\n' {
            if depth == 0 && !matches!(out.last(), Some((Tok::Nl, _)) | None) {
                out.push((Tok::Nl, line));
            }
            line += 1;
            i += 1;
            at_line_start = depth == 0;
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < s.len() && (s[i].is_ascii_alphanumeric() || s[i] == '_') {
                i += 1;
            }
            out.push((Tok::Id(s[st..i].iter().collect()), line));
            continue;
        }
        if c.is_ascii_digit() {
            let st = i;
            let radix = if c == '0' && i + 1 < s.len() && matches!(s[i + 1], 'x' | 'o' | 'b') {
                Some(s[i + 1])
            } else {
                None
            };
            if let Some(r) = radix {
                i += 2;
                let ds = i;
                let ok = |ch: char| match r {
                    'x' => ch.is_ascii_hexdigit(),
                    'o' => ('0'..='7').contains(&ch),
                    _ => ch == '0' || ch == '1',
                };
                while i < s.len() && (ok(s[i]) || s[i] == '_') {
                    i += 1;
                }
                let digits: String = s[ds..i].iter().filter(|c| **c != '_').collect();
                if digits.is_empty() || s[ds] == '_' || s[i - 1] == '_' {
                    return Err(format!("line {line}: W-LEX-3 bad literal"));
                }
                let base = match r {
                    'x' => 16,
                    'o' => 8,
                    _ => 2,
                };
                let v = i128::from_str_radix(&digits, base)
                    .map_err(|_| format!("line {line}: W-LEX-3 literal out of range"))?;
                out.push((Tok::Int(v), line));
                continue;
            }
            while i < s.len()
                && (s[i].is_ascii_digit()
                    || (s[i] == '_' && i + 1 < s.len() && s[i + 1].is_ascii_digit()))
            {
                i += 1;
            }
            let mut is_float = false;
            if i + 1 < s.len() && s[i] == '.' && s[i + 1].is_ascii_digit() {
                is_float = true;
                i += 1;
                while i < s.len()
                    && (s[i].is_ascii_digit()
                        || (s[i] == '_' && i + 1 < s.len() && s[i + 1].is_ascii_digit()))
                {
                    i += 1;
                }
            }
            if i < s.len() && (s[i] == 'e' || s[i] == 'E') {
                let mut j = i + 1;
                if j < s.len() && (s[j] == '+' || s[j] == '-') {
                    j += 1;
                }
                if j < s.len() && s[j].is_ascii_digit() {
                    is_float = true;
                    i = j;
                    while i < s.len() && s[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let txt: String = s[st..i].iter().filter(|c| **c != '_').collect();
            if is_float {
                let v: f64 = txt
                    .parse()
                    .map_err(|_| format!("line {line}: W-LEX-3 bad float"))?;
                if !v.is_finite() {
                    return Err(format!("line {line}: W-LEX-3 float literal out of range"));
                }
                out.push((Tok::Float(v), line));
            } else {
                let v: i128 = txt
                    .parse()
                    .map_err(|_| format!("line {line}: W-LEX-3 literal out of range"))?;
                if v > u64::MAX as i128 {
                    return Err(format!("line {line}: W-LEX-3 literal out of range"));
                }
                out.push((Tok::Int(v), line));
            }
            continue;
        }
        if c == '"' {
            i += 1;
            let mut v = String::new();
            loop {
                if i >= s.len() || s[i] == '\n' {
                    return Err(format!("line {line}: W-SYN-1 unterminated string"));
                }
                let ch = s[i];
                if ch == '"' {
                    i += 1;
                    break;
                }
                if ch == '\\' {
                    let e = *s.get(i + 1).ok_or(format!("line {line}: bad escape"))?;
                    i += 2;
                    match e {
                        '"' => v.push('"'),
                        '\\' => v.push('\\'),
                        'n' => v.push('\n'),
                        'r' => v.push('\r'),
                        't' => v.push('\t'),
                        '0' => v.push('\0'),
                        'u' => {
                            if s.get(i) != Some(&'{') {
                                return Err(format!("line {line}: W-LEX-3 bad \\u escape"));
                            }
                            let st = i + 1;
                            let mut j = st;
                            while j < s.len() && s[j].is_ascii_hexdigit() {
                                j += 1;
                            }
                            if s.get(j) != Some(&'}') || j == st || j - st > 6 {
                                return Err(format!("line {line}: W-LEX-3 bad \\u escape"));
                            }
                            let code =
                                u32::from_str_radix(&s[st..j].iter().collect::<String>(), 16)
                                    .unwrap();
                            v.push(char::from_u32(code).ok_or(format!(
                                "line {line}: W-LEX-3 \\u escape is not a scalar value"
                            ))?);
                            i = j + 1;
                        }
                        _ => return Err(format!("line {line}: W-SYN-1 bad escape")),
                    }
                    continue;
                }
                v.push(ch);
                i += 1;
            }
            out.push((Tok::Str(v), line));
            continue;
        }
        let mut matched = None;
        for p in PUNCT {
            let pc: Vec<char> = p.chars().collect();
            if s[i..].starts_with(&pc) {
                matched = Some(*p);
                break;
            }
        }
        let p = matched.ok_or(format!("line {line}: W-SYN-1 unexpected character {c:?}"))?;
        match p {
            "(" | "[" | "{" => depth += 1,
            ")" | "]" | "}" => depth -= 1,
            _ => {}
        }
        if depth < 0 {
            return Err(format!("line {line}: W-SYN-1 unbalanced bracket"));
        }
        out.push((Tok::P(p), line));
        i += p.len();
    }
    if depth != 0 {
        return Err(format!("line {line}: W-SYN-1 unclosed bracket"));
    }
    if !matches!(out.last(), Some((Tok::Nl, _)) | None) {
        out.push((Tok::Nl, line));
    }
    while stack.len() > 1 {
        stack.pop();
        out.push((Tok::Dedent, line));
    }
    out.push((Tok::Eof, line));
    Ok(out)
}

pub struct Parser {
    t: Vec<(Tok, usize)>,
    p: usize,
}

type R<T> = Result<T, String>;

fn is_kw(s: &str) -> bool {
    KEYWORDS.contains(&s)
}

impl Parser {
    pub fn new(t: Vec<(Tok, usize)>) -> Self {
        Parser { t, p: 0 }
    }
    fn peek(&self) -> &Tok {
        &self.t[self.p].0
    }
    fn peek2(&self) -> &Tok {
        &self.t[(self.p + 1).min(self.t.len() - 1)].0
    }
    fn line(&self) -> usize {
        self.t[self.p].1
    }
    fn err<T>(&self, m: &str) -> R<T> {
        Err(format!(
            "line {}: W-SYN-1 {m} (found {:?})",
            self.line(),
            self.peek()
        ))
    }
    fn bump(&mut self) -> Tok {
        let t = self.t[self.p].0.clone();
        if self.p < self.t.len() - 1 {
            self.p += 1;
        }
        t
    }
    fn is_p(&self, p: &str) -> bool {
        matches!(self.peek(), Tok::P(q) if *q == p)
    }
    fn is_kw(&self, k: &str) -> bool {
        matches!(self.peek(), Tok::Id(q) if q == k)
    }
    fn eat_p(&mut self, p: &str) -> bool {
        if self.is_p(p) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn eat_kw(&mut self, k: &str) -> bool {
        if self.is_kw(k) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn want_p(&mut self, p: &str) -> R<()> {
        if self.eat_p(p) {
            Ok(())
        } else {
            self.err(&format!("expected `{p}`"))
        }
    }
    fn want_kw(&mut self, k: &str) -> R<()> {
        if self.eat_kw(k) {
            Ok(())
        } else {
            self.err(&format!("expected `{k}`"))
        }
    }
    fn nl(&mut self) -> R<()> {
        if matches!(self.peek(), Tok::Nl) {
            self.bump();
            Ok(())
        } else {
            self.err("expected end of line")
        }
    }
    fn indent(&mut self) -> R<()> {
        if matches!(self.peek(), Tok::Indent) {
            self.bump();
            Ok(())
        } else {
            self.err("expected indented block")
        }
    }
    fn is_dedent(&self) -> bool {
        matches!(self.peek(), Tok::Dedent)
    }
    fn name(&mut self) -> R<String> {
        match self.peek().clone() {
            Tok::Id(s) if !is_kw(&s) && s != "_" => {
                self.bump();
                Ok(s)
            }
            _ => self.err("expected a name"),
        }
    }
    fn is_name(&self) -> bool {
        matches!(self.peek(), Tok::Id(s) if !is_kw(s) && s != "_")
    }
    fn path(&mut self) -> R<Vec<String>> {
        let mut v = vec![self.name()?];
        while self.is_p(".") && matches!(self.peek2(), Tok::Id(_)) {
            self.bump();
            v.push(self.name()?);
        }
        Ok(v)
    }

    pub fn module(&mut self) -> R<Module> {
        while matches!(self.peek(), Tok::Nl) {
            self.bump();
        }
        let mut m = Module {
            path: None,
            items: vec![],
        };
        if self.eat_kw("module") {
            m.path = Some(self.path()?);
            self.nl()?;
        }
        while !matches!(self.peek(), Tok::Eof) {
            m.items.push(self.item()?);
        }
        Ok(m)
    }

    fn item(&mut self) -> R<Item> {
        let line = self.line();
        let kw = match self.peek() {
            Tok::Id(s) => s.clone(),
            _ => return self.err("expected a declaration"),
        };
        let pubd = kw == "pub";
        if pubd {
            self.bump();
            let k = match self.peek() {
                Tok::Id(s) => s.clone(),
                _ => return self.err("expected a declaration after pub"),
            };
            if ![
                "const", "type", "struct", "linear", "kind", "trait", "fn", "comptime", "actor",
            ]
            .contains(&k.as_str())
            {
                return self.err("this declaration cannot be pub");
            }
        }
        let kw = match self.peek() {
            Tok::Id(s) => s.clone(),
            _ => unreachable!(),
        };
        match kw.as_str() {
            "use" => {
                self.bump();
                let p = self.path()?;
                let a = if self.eat_kw("as") {
                    Some(self.name()?)
                } else {
                    None
                };
                self.nl()?;
                Ok(Item::Use(p, a))
            }
            "intent" => {
                self.bump();
                let s = match self.bump() {
                    Tok::Str(s) => s,
                    _ => return self.err("intent needs a string"),
                };
                self.nl()?;
                let (mut ex, mut ck) = (vec![], vec![]);
                if matches!(self.peek(), Tok::Indent) {
                    self.bump();
                    while !self.is_dedent() {
                        if self.is_kw("example") {
                            ex.push(self.example()?);
                        } else if self.eat_kw("check") {
                            ck.push(self.expr()?);
                            self.nl()?;
                        } else if let Tok::Str(_) = self.peek() {
                            self.bump();
                            self.nl()?;
                        } else {
                            return self.err("expected example, check or text in intent");
                        }
                    }
                    self.bump();
                }
                Ok(Item::Intent(s, ex, ck))
            }
            "example" => Ok(Item::Example(self.example()?)),
            "check" => {
                self.bump();
                let e = self.expr()?;
                self.nl()?;
                Ok(Item::Check(e))
            }
            "const" => {
                self.bump();
                let name = self.name()?;
                let ty = if self.eat_p(":") {
                    Some(self.ty()?)
                } else {
                    None
                };
                self.want_p("=")?;
                let e = self.expr()?;
                self.nl()?;
                Ok(Item::Const { name, ty, e, line })
            }
            "type" => {
                self.bump();
                let name = self.name()?;
                let generics = self.generics_opt()?;
                self.want_p("=")?;
                let ty = self.ty()?;
                self.nl()?;
                Ok(Item::TypeAlias {
                    name,
                    generics,
                    ty,
                    line,
                })
            }
            "struct" | "linear" => {
                let linear = kw == "linear";
                if linear {
                    self.bump();
                    if !self.is_kw("struct") {
                        return self.err("expected `struct` after `linear`");
                    }
                }
                self.bump();
                let name = self.name()?;
                let generics = self.generics_opt()?;
                self.nl()?;
                let mut fields = vec![];
                if matches!(self.peek(), Tok::Indent) {
                    self.bump();
                    while !self.is_dedent() {
                        fields.push(self.field()?);
                    }
                    self.bump();
                }
                Ok(Item::Struct {
                    name,
                    generics,
                    fields,
                    line,
                    linear,
                })
            }
            "state" => {
                self.bump();
                let name = self.name()?;
                self.nl()?;
                self.indent()?;
                let mut fields = vec![];
                while !self.is_dedent() {
                    fields.push(self.field()?);
                }
                self.bump();
                Ok(Item::State { name, fields, line })
            }
            "event" | "error" => {
                self.bump();
                let name = self.name()?;
                let payload = if self.is_p("(") {
                    self.payload()?
                } else {
                    vec![]
                };
                self.nl()?;
                Ok(if kw == "event" {
                    Item::Event {
                        name,
                        payload,
                        line,
                    }
                } else {
                    Item::Error {
                        name,
                        payload,
                        line,
                    }
                })
            }
            "kind" => {
                self.bump();
                let name = self.name()?;
                let generics = self.generics_opt()?;
                self.nl()?;
                self.indent()?;
                let mut variants = vec![];
                while !self.is_dedent() {
                    let v = self.name()?;
                    let p = if self.is_p("(") {
                        self.payload()?
                    } else {
                        vec![]
                    };
                    self.nl()?;
                    variants.push((v, p));
                }
                self.bump();
                Ok(Item::Kind {
                    name,
                    generics,
                    variants,
                    line,
                })
            }
            "trait" => {
                self.bump();
                let name = self.name()?;
                let generics = self.generics_opt()?;
                if self.eat_p(":") {
                    self.bounds()?;
                }
                self.nl()?;
                self.indent()?;
                let mut members = vec![];
                while !self.is_dedent() {
                    if self.eat_kw("type") {
                        let n = self.name()?;
                        if self.eat_p(":") {
                            self.bounds()?;
                        }
                        self.nl()?;
                        members.push(TraitMember::Type(n));
                    } else {
                        let s = self.fn_sig()?;
                        self.nl()?;
                        members.push(TraitMember::Fn(s));
                    }
                }
                self.bump();
                Ok(Item::Trait {
                    name,
                    generics,
                    members,
                    line,
                })
            }
            "impl" => {
                self.bump();
                let generics = if self.eat_kw("forall") {
                    self.generics()?
                } else {
                    vec![]
                };
                let t1 = self.ty()?;
                let (ty, trait_) = if self.eat_kw("for") {
                    (self.ty()?, Some(t1))
                } else {
                    (t1, None)
                };
                self.nl()?;
                self.indent()?;
                let mut members = vec![];
                while !self.is_dedent() {
                    if self.is_kw("type") {
                        self.bump();
                        self.name()?;
                        self.generics_opt()?;
                        self.want_p("=")?;
                        self.ty()?;
                        self.nl()?;
                    } else {
                        members.push(self.fn_decl()?);
                    }
                }
                self.bump();
                Ok(Item::Impl {
                    generics,
                    ty,
                    trait_,
                    members,
                    line,
                })
            }
            "fn" | "comptime" => Ok(Item::Fn(self.fn_decl()?)),
            "extern" => {
                self.bump();
                let abi = match self.bump() {
                    Tok::Str(s) => s,
                    _ => return self.err("extern needs an ABI string"),
                };
                let sig = self.fn_sig()?;
                self.nl()?;
                Ok(Item::Extern { abi, sig, line })
            }
            "step" => {
                self.bump();
                let region = self.name()?;
                self.want_kw("on")?;
                let pat = self.pattern()?;
                let guard = if self.eat_kw("if") {
                    Some(self.expr()?)
                } else {
                    None
                };
                let body = self.step_body()?;
                Ok(Item::Step {
                    region,
                    pat,
                    guard,
                    body,
                    line,
                })
            }
            "actor" => {
                self.bump();
                let name = self.name()?;
                self.nl()?;
                self.indent()?;
                let (mut fields, mut handlers, mut fns) = (vec![], vec![], vec![]);
                while !self.is_dedent() {
                    let l = self.line();
                    if self.eat_kw("on") {
                        let pat = self.pattern()?;
                        let guard = if self.eat_kw("if") {
                            Some(self.expr()?)
                        } else {
                            None
                        };
                        let body = self.step_body()?;
                        handlers.push(Handler {
                            pat,
                            guard,
                            body,
                            line: l,
                        });
                    } else if self.is_kw("fn") || self.is_kw("comptime") {
                        fns.push(self.fn_decl()?);
                    } else {
                        fields.push(self.field()?);
                    }
                }
                self.bump();
                Ok(Item::Actor {
                    name,
                    fields,
                    handlers,
                    fns,
                    line,
                })
            }
            _ => self.err("expected a declaration"),
        }
    }

    fn step_body(&mut self) -> R<Body> {
        if self.eat_p("->") {
            let e = self.expr()?;
            self.nl()?;
            Ok(Body::Arrow(e))
        } else {
            Ok(Body::Block(self.block()?))
        }
    }

    fn example(&mut self) -> R<Example> {
        let line = self.line();
        self.want_kw("example")?;
        let mut events = vec![];
        if !self.is_p("->") {
            loop {
                let n = self.name()?;
                let args = if self.eat_p("(") {
                    let a = if self.is_p(")") { vec![] } else { self.args()? };
                    self.want_p(")")?;
                    a
                } else {
                    vec![]
                };
                events.push((n, args));
                if !self.eat_p(",") {
                    break;
                }
            }
        }
        self.want_p("->")?;
        let (mut refused, mut value) = (None, None);
        if self.eat_kw("refused") {
            self.want_p(":")?;
            refused = Some(self.name()?);
        } else {
            value = Some(self.expr()?);
        }
        self.nl()?;
        Ok(Example {
            events,
            refused,
            value,
            line,
        })
    }

    fn field(&mut self) -> R<Field> {
        let line = self.line();
        let name = self.name()?;
        self.want_p(":")?;
        let ty = self.ty()?;
        let init = if self.eat_p("=") {
            Some(self.expr()?)
        } else {
            None
        };
        self.nl()?;
        Ok(Field {
            name,
            ty,
            init,
            line,
        })
    }

    fn payload(&mut self) -> R<Vec<(String, Ty)>> {
        self.want_p("(")?;
        let mut v = vec![];
        while !self.is_p(")") {
            let n = self.name()?;
            self.want_p(":")?;
            let t = self.ty()?;
            v.push((n, t));
            if !self.eat_p(",") {
                break;
            }
        }
        self.want_p(")")?;
        Ok(v)
    }

    fn generics_opt(&mut self) -> R<Vec<GParam>> {
        if self.is_p("[") {
            self.generics()
        } else {
            Ok(vec![])
        }
    }
    fn generics(&mut self) -> R<Vec<GParam>> {
        self.want_p("[")?;
        let mut v = vec![];
        loop {
            let comptime = self.eat_kw("comptime");
            let name = self.name()?;
            let bounds = if self.eat_p(":") {
                self.bounds()?
            } else {
                vec![]
            };
            v.push(GParam {
                comptime,
                name,
                bounds,
            });
            if !self.eat_p(",") {
                break;
            }
        }
        self.want_p("]")?;
        Ok(v)
    }
    fn bounds(&mut self) -> R<Vec<Ty>> {
        let mut v = vec![self.type_path()?];
        while self.eat_p("+") {
            v.push(self.type_path()?);
        }
        Ok(v)
    }

    fn fn_sig(&mut self) -> R<FnSig> {
        let line = self.line();
        let comptime = self.eat_kw("comptime");
        self.want_kw("fn")?;
        let name = self.name()?;
        let generics = self.generics_opt()?;
        self.want_p("(")?;
        let params = if self.is_p(")") {
            vec![]
        } else {
            self.params()?
        };
        self.want_p(")")?;
        let ret = if self.eat_p("->") {
            Some(self.ty()?)
        } else {
            None
        };
        Ok(FnSig {
            comptime,
            name,
            generics,
            params,
            ret,
            line,
        })
    }
    fn fn_decl(&mut self) -> R<FnDecl> {
        let sig = self.fn_sig()?;
        let body = if self.eat_p("=") {
            let e = self.expr()?;
            self.nl()?;
            Body::Arrow(e)
        } else {
            Body::Block(self.block()?)
        };
        Ok(FnDecl { sig, body })
    }
    fn params(&mut self) -> R<Vec<Param>> {
        let mut v = vec![];
        loop {
            if self.eat_kw("self") {
                v.push(Param::SelfVal);
            } else if self.is_p("&")
                && (matches!(self.peek2(), Tok::Id(s) if s == "self" || s == "mut"))
            {
                self.bump();
                let m = self.eat_kw("mut");
                self.want_kw("self")?;
                v.push(Param::SelfRef(m));
            } else {
                let comptime = self.eat_kw("comptime");
                let a = self.name()?;
                let (label, name) = if self.is_name() {
                    (Some(a), self.name()?)
                } else {
                    (None, a)
                };
                self.want_p(":")?;
                let ty = self.ty()?;
                v.push(Param::Named {
                    comptime,
                    label,
                    name,
                    ty,
                });
            }
            if !self.eat_p(",") || self.is_p(")") {
                break;
            }
        }
        Ok(v)
    }

    // ---- types ----
    pub fn ty(&mut self) -> R<Ty> {
        let a = self.opt_ty()?;
        if self.eat_p("!") {
            let b = self.opt_ty()?;
            return Ok(Ty::Union(Box::new(a), Box::new(b)));
        }
        Ok(a)
    }
    fn opt_ty(&mut self) -> R<Ty> {
        let a = self.borrow_ty()?;
        if self.eat_p("?") {
            Ok(Ty::Opt(Box::new(a)))
        } else {
            Ok(a)
        }
    }
    fn borrow_ty(&mut self) -> R<Ty> {
        if self.eat_p("&") {
            let m = self.eat_kw("mut");
            return Ok(Ty::Ref(m, Box::new(self.borrow_ty()?)));
        }
        let b = self.base_ty()?;
        if self.eat_p("@") {
            return Ok(Ty::Region(Box::new(b), self.name()?));
        }
        Ok(b)
    }
    fn type_path(&mut self) -> R<Ty> {
        let p = self.path()?;
        let mut args = vec![];
        if self.eat_p("[") {
            loop {
                args.push(self.ty()?);
                if !self.eat_p(",") || self.is_p("]") {
                    break;
                }
            }
            self.want_p("]")?;
        }
        Ok(Ty::Path(p, args))
    }
    pub fn base_ty(&mut self) -> R<Ty> {
        if self.is_name() {
            return self.type_path();
        }
        if self.eat_p("(") {
            if self.eat_p(")") {
                return Ok(Ty::Unit);
            }
            let a = self.ty()?;
            if self.eat_p(",") {
                let mut v = vec![a];
                while !self.is_p(")") {
                    v.push(self.ty()?);
                    if !self.eat_p(",") {
                        break;
                    }
                }
                self.want_p(")")?;
                return Ok(Ty::Tuple(v));
            }
            self.want_p(")")?;
            return Ok(a);
        }
        if self.eat_p("[") {
            let a = self.ty()?;
            if self.eat_p(";") {
                let n = self.expr()?;
                self.want_p("]")?;
                return Ok(Ty::Array(Box::new(a), Box::new(n)));
            }
            self.want_p("]")?;
            return Ok(Ty::Slice(Box::new(a)));
        }
        if self.eat_kw("chan") {
            return Ok(Ty::Chan(Box::new(self.base_ty()?)));
        }
        if self.eat_kw("iter") {
            return Ok(Ty::Iter(Box::new(self.base_ty()?)));
        }
        if self.eat_kw("fn") {
            self.want_p("(")?;
            let mut v = vec![];
            while !self.is_p(")") {
                v.push(self.ty()?);
                if !self.eat_p(",") {
                    break;
                }
            }
            self.want_p(")")?;
            self.want_p("->")?;
            let r = self.base_ty()?;
            return Ok(Ty::Fn(v, Box::new(r)));
        }
        self.err("expected a type")
    }

    // ---- statements ----
    pub fn block(&mut self) -> R<Vec<Stmt>> {
        self.nl()?;
        self.indent()?;
        let mut v = vec![];
        while !self.is_dedent() {
            v.push(self.stmt()?);
        }
        self.bump();
        Ok(v)
    }
    fn arm_body(&mut self) -> R<Vec<Stmt>> {
        if self.eat_p("->") {
            Ok(vec![self.inline_stmt()?])
        } else {
            self.block()
        }
    }
    fn inline_stmt(&mut self) -> R<Stmt> {
        for k in [
            "return", "break", "continue", "pass", "emit", "fail", "yield",
        ] {
            if self.is_kw(k) {
                return self.stmt();
            }
        }
        self.simple_stmt()
    }
    fn simple_stmt(&mut self) -> R<Stmt> {
        let e = self.expr()?;
        const AOPS: &[&str] = &[
            "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=",
        ];
        if let Tok::P(p) = self.peek().clone() {
            if AOPS.contains(&p) {
                self.bump();
                let r = self.expr()?;
                self.nl()?;
                return Ok(Stmt::Assign(p.to_string(), e, r));
            }
            if p == "<-" {
                self.bump();
                let r = self.expr()?;
                self.nl()?;
                return Ok(Stmt::Send(e, r));
            }
        }
        self.nl()?;
        Ok(Stmt::Expr(e))
    }
    fn stmt(&mut self) -> R<Stmt> {
        let line = self.line();
        let kw = match self.peek() {
            Tok::Id(s) => s.clone(),
            _ => String::new(),
        };
        match kw.as_str() {
            "let" => {
                self.bump();
                let p = self.pattern()?;
                let t = if self.eat_p(":") {
                    Some(self.ty()?)
                } else {
                    None
                };
                self.want_p("=")?;
                let e = self.expr()?;
                self.nl()?;
                Ok(Stmt::Let(p, t, e, line))
            }
            "var" => {
                self.bump();
                let n = self.name()?;
                let t = if self.eat_p(":") {
                    Some(self.ty()?)
                } else {
                    None
                };
                self.want_p("=")?;
                let e = self.expr()?;
                self.nl()?;
                Ok(Stmt::Var(n, t, e, line))
            }
            "if" => {
                self.bump();
                let c = self.expr()?;
                let b = self.block()?;
                let mut arms = vec![(c, b)];
                let mut els = None;
                loop {
                    if self.eat_kw("elif") {
                        let c = self.expr()?;
                        let b = self.block()?;
                        arms.push((c, b));
                    } else if self.eat_kw("else") {
                        els = Some(self.block()?);
                        break;
                    } else {
                        break;
                    }
                }
                Ok(Stmt::If(arms, els))
            }
            "while" => {
                self.bump();
                let c = self.expr()?;
                let b = if self.eat_kw("bound") {
                    Some(self.expr()?)
                } else {
                    None
                };
                Ok(Stmt::While(c, b, self.block()?))
            }
            "loop" => {
                self.bump();
                let b = if self.eat_kw("bound") {
                    Some(self.expr()?)
                } else {
                    None
                };
                Ok(Stmt::Loop(b, self.block()?))
            }
            "for" => {
                self.bump();
                let p = self.pattern()?;
                self.want_kw("in")?;
                let e = self.expr()?;
                // draft 0.6 `for ... par`: encoded as the marker region `$forpar`
                // (the AST has no new variant, see spec/CONCURRENCY-0.6.md C.8).
                if self.eat_kw("par") {
                    let f = Stmt::For(p, e, self.block()?, line);
                    return Ok(Stmt::Arena("$forpar".into(), None, vec![f], line));
                }
                Ok(Stmt::For(p, e, self.block()?, line))
            }
            "match" => {
                self.bump();
                let e = self.expr()?;
                self.nl()?;
                self.indent()?;
                let mut arms = vec![];
                while !self.is_dedent() {
                    let l = self.line();
                    let pat = self.pattern()?;
                    let guard = if self.eat_kw("if") {
                        Some(self.expr()?)
                    } else {
                        None
                    };
                    let body = self.arm_body()?;
                    arms.push(Arm {
                        pat,
                        guard,
                        body,
                        line: l,
                    });
                }
                self.bump();
                Ok(Stmt::Match(e, arms, line))
            }
            "return" => {
                self.bump();
                let e = if matches!(self.peek(), Tok::Nl) {
                    None
                } else {
                    Some(self.expr()?)
                };
                self.nl()?;
                Ok(Stmt::Return(e, line))
            }
            "break" => {
                self.bump();
                self.nl()?;
                Ok(Stmt::Break(line))
            }
            "continue" => {
                self.bump();
                self.nl()?;
                Ok(Stmt::Continue(line))
            }
            "pass" => {
                self.bump();
                self.nl()?;
                Ok(Stmt::Pass)
            }
            "emit" => {
                self.bump();
                let e = self.expr()?;
                self.nl()?;
                Ok(Stmt::Emit(e))
            }
            "fail" => {
                self.bump();
                let e = self.expr()?;
                self.nl()?;
                Ok(Stmt::Fail(e))
            }
            "yield" => {
                self.bump();
                let e = self.expr()?;
                self.nl()?;
                Ok(Stmt::Yield(e, line))
            }
            "spawn" => {
                self.bump();
                let e = self.expr()?;
                if self.eat_kw("priority") {
                    let pr = self.expr()?;
                    self.nl()?;
                    return Ok(Stmt::Arena(
                        "$priority".into(),
                        None,
                        vec![Stmt::Expr(pr), Stmt::Spawn(e)],
                        line,
                    ));
                }
                self.nl()?;
                Ok(Stmt::Spawn(e))
            }
            "check" => {
                self.bump();
                let e = self.expr()?;
                self.nl()?;
                Ok(Stmt::Check(e))
            }
            "select" => {
                self.bump();
                let nondet = self.eat_kw("nondet");
                self.nl()?;
                self.indent()?;
                let mut arms = vec![];
                while !self.is_dedent() {
                    if self.eat_kw("recv") {
                        let p = self.pattern()?;
                        self.want_kw("from")?;
                        let c = self.expr()?;
                        arms.push(SelArm::Recv(p, c, self.arm_body()?));
                    } else if self.eat_kw("send") {
                        let v = self.expr()?;
                        self.want_kw("to")?;
                        let c = self.expr()?;
                        arms.push(SelArm::Send(v, c, self.arm_body()?));
                    } else if self.eat_kw("after") {
                        let n = self.expr()?;
                        arms.push(SelArm::After(n, self.arm_body()?));
                    } else if self.eat_kw("default") {
                        arms.push(SelArm::Default(self.arm_body()?));
                    } else {
                        return self.err("expected a select arm");
                    }
                }
                self.bump();
                if nondet {
                    return Ok(Stmt::Arena(
                        "$nondet".into(),
                        None,
                        vec![Stmt::Select(arms, line)],
                        line,
                    ));
                }
                Ok(Stmt::Select(arms, line))
            }
            "par" => {
                self.bump();
                self.nl()?;
                self.indent()?;
                let mut branches = vec![];
                while !self.is_dedent() {
                    let bl = self.line();
                    self.want_kw("branch")?;
                    let n = self.name()?;
                    let body = self.block()?;
                    branches.push(Stmt::Arena(format!("$branch:{n}"), None, body, bl));
                }
                self.bump();
                if branches.is_empty() {
                    return self.err("par needs at least one branch");
                }
                Ok(Stmt::Arena("$par".into(), None, branches, line))
            }
            "with" if matches!(self.peek2(), Tok::Id(k) if k == "budget" || k == "deadline") => {
                self.bump();
                let budget = self.eat_kw("budget");
                if !budget {
                    self.want_kw("deadline")?;
                }
                let n = self.expr()?;
                let body = self.block()?;
                if budget {
                    let mut v = vec![Stmt::Expr(n)];
                    v.extend(body);
                    return Ok(Stmt::Arena("$budget".into(), None, v, line));
                }
                let mut v = vec![
                    Stmt::Expr(n),
                    Stmt::Arena("$dlbody".into(), None, body, line),
                ];
                if self.eat_kw("else") {
                    let eb = self.block()?;
                    // The else block runs only on cancellation, which never happens
                    // in a run without real concurrency: encoded as `if false`.
                    v.push(Stmt::If(vec![(Expr::Bool(false, line), eb)], None));
                }
                Ok(Stmt::Arena("$deadline".into(), None, v, line))
            }
            "with" => {
                self.bump();
                let dev = self.eat_kw("device");
                if !dev {
                    self.want_kw("arena")?;
                }
                let n = self.name()?;
                let b = if self.eat_kw("bound") {
                    Some(self.expr()?)
                } else {
                    None
                };
                let body = self.block()?;
                Ok(if dev {
                    Stmt::Device(n, b, body, line)
                } else {
                    Stmt::Arena(n, b, body, line)
                })
            }
            "comptime" if matches!(self.peek2(), Tok::Nl) => {
                self.bump();
                Ok(Stmt::Comptime(self.block()?))
            }
            _ => self.simple_stmt(),
        }
    }

    // ---- expressions ----
    pub fn expr(&mut self) -> R<Expr> {
        let l = self.line();
        if self.eat_kw("try") {
            let e = self.catch_expr()?;
            let el = if self.eat_kw("else") {
                Some(Box::new(self.catch_expr()?))
            } else {
                None
            };
            return Ok(Expr::Try(Box::new(e), el, l));
        }
        if self.eat_kw("if") {
            let c = self.expr()?;
            self.want_kw("then")?;
            let a = self.catch_expr()?;
            self.want_kw("else")?;
            let b = self.catch_expr()?;
            return Ok(Expr::If(Box::new(c), Box::new(a), Box::new(b), l));
        }
        if self.is_kw("fn") {
            self.bump();
            self.want_p("(")?;
            let ps = if self.is_p(")") {
                vec![]
            } else {
                self.params()?
            };
            self.want_p(")")?;
            self.want_p("=>")?;
            let b = self.catch_expr()?;
            return Ok(Expr::Lambda(ps, Box::new(b), l));
        }
        self.catch_expr()
    }
    fn catch_expr(&mut self) -> R<Expr> {
        let l = self.line();
        let a = self.or_expr()?;
        if self.eat_kw("catch") {
            let bind = if self.eat_p("|") {
                let n = self.name()?;
                self.want_p("|")?;
                Some(n)
            } else {
                None
            };
            let h = self.or_expr()?;
            return Ok(Expr::Catch(Box::new(a), bind, Box::new(h), l));
        }
        Ok(a)
    }
    fn or_expr(&mut self) -> R<Expr> {
        let mut a = self.and_expr()?;
        while self.is_kw("or") {
            let l = self.line();
            self.bump();
            let b = self.and_expr()?;
            a = Expr::Bin("or".into(), Box::new(a), Box::new(b), l);
        }
        Ok(a)
    }
    fn and_expr(&mut self) -> R<Expr> {
        let mut a = self.not_expr()?;
        while self.is_kw("and") {
            let l = self.line();
            self.bump();
            let b = self.not_expr()?;
            a = Expr::Bin("and".into(), Box::new(a), Box::new(b), l);
        }
        Ok(a)
    }
    fn not_expr(&mut self) -> R<Expr> {
        let l = self.line();
        if self.eat_kw("not") {
            return Ok(Expr::Not(Box::new(self.not_expr()?), l));
        }
        self.cmp_expr()
    }
    fn cmp_expr(&mut self) -> R<Expr> {
        let a = self.range_expr()?;
        let l = self.line();
        let op = match self.peek() {
            Tok::P(p) if ["==", "!=", "<", "<=", ">", ">="].contains(p) => p.to_string(),
            Tok::Id(s) if s == "in" => "in".into(),
            Tok::Id(s) if s == "not" && matches!(self.peek2(), Tok::Id(t) if t == "in") => {
                self.bump();
                "not in".into()
            }
            _ => return Ok(a),
        };
        self.bump();
        let b = self.range_expr()?;
        Ok(Expr::Bin(op, Box::new(a), Box::new(b), l))
    }
    fn range_expr(&mut self) -> R<Expr> {
        let a = self.bin_level(0)?;
        let l = self.line();
        if self.is_p("..") || self.is_p("..=") {
            let op = if let Tok::P(p) = self.bump() {
                p
            } else {
                unreachable!()
            };
            let b = self.bin_level(0)?;
            return Ok(Expr::Bin(op.into(), Box::new(a), Box::new(b), l));
        }
        Ok(a)
    }
    fn bin_level(&mut self, lvl: usize) -> R<Expr> {
        const LV: &[&[&str]] = &[
            &["|"],
            &["^"],
            &["&"],
            &["<<", ">>"],
            &["+", "-"],
            &["*", "/", "%"],
        ];
        if lvl == LV.len() {
            return self.cast_expr();
        }
        let mut a = self.bin_level(lvl + 1)?;
        loop {
            let l = self.line();
            match self.peek() {
                Tok::P(p) if LV[lvl].contains(p) => {
                    let op = p.to_string();
                    self.bump();
                    let b = self.bin_level(lvl + 1)?;
                    a = Expr::Bin(op, Box::new(a), Box::new(b), l);
                }
                _ => return Ok(a),
            }
        }
    }
    fn cast_expr(&mut self) -> R<Expr> {
        let a = self.unary()?;
        let l = self.line();
        if self.eat_kw("as") {
            let t = self.base_ty()?;
            return Ok(Expr::Cast(Box::new(a), t, l));
        }
        Ok(a)
    }
    fn unary(&mut self) -> R<Expr> {
        let l = self.line();
        if self.eat_p("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?), l));
        }
        if self.eat_p("~") {
            return Ok(Expr::BitNot(Box::new(self.unary()?), l));
        }
        if self.eat_p("&") {
            let m = self.eat_kw("mut");
            return Ok(Expr::Borrow(m, Box::new(self.unary()?), l));
        }
        if self.eat_p("<-") {
            return Ok(Expr::Recv(Box::new(self.unary()?), l));
        }
        if self.eat_kw("comptime") {
            return Ok(Expr::Comptime(Box::new(self.unary()?), l));
        }
        self.postfix()
    }
    fn args(&mut self) -> R<Vec<Arg>> {
        let mut v = vec![];
        loop {
            let label = if self.is_name() && matches!(self.peek2(), Tok::P(":")) {
                let n = self.name()?;
                self.bump();
                Some(n)
            } else {
                None
            };
            v.push(Arg {
                label,
                e: self.expr()?,
            });
            if !self.eat_p(",") || self.is_p(")") {
                break;
            }
        }
        Ok(v)
    }
    fn postfix(&mut self) -> R<Expr> {
        let mut e = self.primary()?;
        loop {
            let l = self.line();
            if self.eat_p("(") {
                let a = if self.is_p(")") { vec![] } else { self.args()? };
                self.want_p(")")?;
                e = Expr::Call(Box::new(e), a, l);
            } else if self.eat_p("[") {
                if self.eat_p(":") {
                    let hi = if self.is_p("]") {
                        None
                    } else {
                        Some(Box::new(self.expr()?))
                    };
                    self.want_p("]")?;
                    e = Expr::Slice(Box::new(e), None, hi, l);
                    continue;
                }
                let first = self.expr()?;
                if self.eat_p(":") {
                    let hi = if self.is_p("]") {
                        None
                    } else {
                        Some(Box::new(self.expr()?))
                    };
                    self.want_p("]")?;
                    e = Expr::Slice(Box::new(e), Some(Box::new(first)), hi, l);
                    continue;
                }
                let mut ix = vec![first];
                while self.eat_p(",") {
                    ix.push(self.expr()?);
                }
                self.want_p("]")?;
                e = Expr::Index(Box::new(e), ix, l);
            } else if self.eat_p(".") {
                match self.bump() {
                    Tok::Id(s) if s != "_" => e = Expr::Member(Box::new(e), s, l), // keywords allowed as member names (spec 2.4)
                    Tok::Int(n) => e = Expr::TupleMember(Box::new(e), n as usize, l),
                    _ => {
                        self.p -= 1;
                        return self.err("expected a field name");
                    }
                }
            } else {
                return Ok(e);
            }
        }
    }
    fn primary(&mut self) -> R<Expr> {
        let l = self.line();
        match self.peek().clone() {
            Tok::Int(n) => {
                self.bump();
                Ok(Expr::Int(n, l))
            }
            Tok::Float(f) => {
                self.bump();
                Ok(Expr::Float(f, l))
            }
            Tok::Str(s) => {
                self.bump();
                Ok(Expr::Str(s, l))
            }
            Tok::Id(s) if s == "true" || s == "false" => {
                self.bump();
                Ok(Expr::Bool(s == "true", l))
            }
            Tok::Id(s) if s == "none" => {
                self.bump();
                Ok(Expr::None(l))
            }
            Tok::Id(s) if s == "self" => {
                self.bump();
                Ok(Expr::SelfRef(l))
            }
            Tok::Id(s) if s == "chan" => {
                self.bump();
                self.want_p("[")?;
                let t = self.ty()?;
                self.want_p("]")?;
                self.want_p("(")?;
                let c = if self.is_p(")") {
                    None
                } else {
                    Some(Box::new(self.expr()?))
                };
                self.want_p(")")?;
                Ok(Expr::ChanNew(t, c, l))
            }
            Tok::Id(_) if self.is_name() => Ok(Expr::Name(self.name()?, l)),
            Tok::P("(") => {
                self.bump();
                if self.eat_p(")") {
                    return Ok(Expr::Unit(l));
                }
                let a = self.expr()?;
                if self.eat_p(",") {
                    let mut v = vec![a];
                    while !self.is_p(")") {
                        v.push(self.expr()?);
                        if !self.eat_p(",") {
                            break;
                        }
                    }
                    self.want_p(")")?;
                    return Ok(Expr::Tuple(v, l));
                }
                self.want_p(")")?;
                Ok(a)
            }
            Tok::P("[") => {
                self.bump();
                if self.eat_p("]") {
                    return Ok(Expr::List(vec![], l));
                }
                let a = self.expr()?;
                if self.eat_p(";") {
                    let n = self.expr()?;
                    self.want_p("]")?;
                    return Ok(Expr::ListRepeat(Box::new(a), Box::new(n), l));
                }
                if self.eat_kw("for") {
                    let p = self.pattern()?;
                    self.want_kw("in")?;
                    let it = self.or_expr()?;
                    let c = if self.eat_kw("if") {
                        Some(Box::new(self.or_expr()?))
                    } else {
                        None
                    };
                    self.want_p("]")?;
                    return Ok(Expr::ListComp(Box::new(a), Box::new(p), Box::new(it), c, l));
                }
                let mut v = vec![a];
                while self.eat_p(",") {
                    if self.is_p("]") {
                        break;
                    }
                    v.push(self.expr()?);
                }
                self.want_p("]")?;
                Ok(Expr::List(v, l))
            }
            Tok::P("{") => {
                self.bump();
                let mut v = vec![];
                while !self.is_p("}") {
                    let k = self.expr()?;
                    self.want_p(":")?;
                    let x = self.expr()?;
                    v.push((k, x));
                    if !self.eat_p(",") {
                        break;
                    }
                }
                self.want_p("}")?;
                Ok(Expr::Map(v, l))
            }
            _ => self.err("expected an expression"),
        }
    }

    // ---- patterns ----
    pub fn pattern(&mut self) -> R<Pat> {
        let a = self.atom_pat()?;
        if self.is_p("|") {
            let mut v = vec![a];
            while self.eat_p("|") {
                v.push(self.atom_pat()?);
            }
            return Ok(Pat::Alt(v));
        }
        Ok(a)
    }
    fn lit_pat(&mut self) -> R<Option<Expr>> {
        let l = self.line();
        Ok(Some(match self.peek().clone() {
            Tok::Int(n) => {
                self.bump();
                Expr::Int(n, l)
            }
            Tok::P("-") => {
                self.bump();
                match self.bump() {
                    Tok::Int(n) => Expr::Int(-n, l),
                    Tok::Float(f) => Expr::Float(-f, l),
                    _ => {
                        self.p -= 1;
                        return self.err("expected a number");
                    }
                }
            }
            Tok::Float(f) => {
                self.bump();
                Expr::Float(f, l)
            }
            Tok::Str(s) => {
                self.bump();
                Expr::Str(s, l)
            }
            Tok::Id(s) if s == "true" || s == "false" => {
                self.bump();
                Expr::Bool(s == "true", l)
            }
            Tok::Id(s) if s == "none" => {
                self.bump();
                Expr::None(l)
            }
            _ => return Ok(None),
        }))
    }
    fn atom_pat(&mut self) -> R<Pat> {
        if matches!(self.peek(), Tok::Id(s) if s == "_") {
            self.bump();
            return Ok(Pat::Wild);
        }
        if let Some(lit) = self.lit_pat()? {
            if self.is_p("..") || self.is_p("..=") {
                let incl = self.is_p("..=");
                self.bump();
                let hi = self
                    .lit_pat()?
                    .ok_or(format!("line {}: W-SYN-1 expected a literal", self.line()))?;
                return Ok(Pat::Range(lit, incl, hi));
            }
            return Ok(Pat::Lit(lit));
        }
        if self.eat_p("(") {
            let a = self.pattern()?;
            if self.eat_p(",") {
                let mut v = vec![a];
                while !self.is_p(")") {
                    v.push(self.pattern()?);
                    if !self.eat_p(",") {
                        break;
                    }
                }
                self.want_p(")")?;
                return Ok(Pat::Tuple(v));
            }
            self.want_p(")")?;
            return Ok(a);
        }
        // `some` is an ordinary name in the grammar
        let p = self.path()?;
        if self.eat_p("(") {
            let mut v = vec![];
            while !self.is_p(")") {
                let label = if self.is_name() && matches!(self.peek2(), Tok::P(":")) {
                    let n = self.name()?;
                    self.bump();
                    Some(n)
                } else {
                    None
                };
                v.push((label, self.pattern()?));
                if !self.eat_p(",") {
                    break;
                }
            }
            self.want_p(")")?;
            return Ok(Pat::Ctor(p, v));
        }
        Ok(Pat::Path(p))
    }
}

pub fn parse(src: &str) -> Result<Module, String> {
    let toks = lex(src)?;
    Parser::new(toks).module()
}
