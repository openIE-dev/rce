//! Lexer, layout pass (spec 2.6) and recursive-descent parser for the
//! grammar of spec section 3. The first error refuses the text.
use crate::ast::*;

/// Reserved words of the language (spec 2.4). The words of Appendix C are in
/// `DRAFT06_KEYWORDS`; both lists are reserved in every module (2.7: there is
/// one language, and the release label selects nothing).
pub const KEYWORDS: &[&str] = &[
    "actor", "after", "and", "arena", "as", "bound", "break", "catch", "chan", "check", "comptime",
    "const", "continue", "default", "device", "elif", "else", "emit", "error", "event", "example",
    "extern", "fail", "false", "fn", "for", "forall", "from", "if", "impl", "in", "intent", "iter",
    "kind", "let", "linear", "loop", "match", "module", "mut", "none", "not", "on", "or", "pass",
    "pub", "recv", "return", "select", "self", "send", "spawn", "state", "step",
    "struct", "then", "to", "trait", "true", "try", "type", "use", "var", "while", "with", "yield",
];

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Id(String),
    Int(i128),
    Float(f64),
    Str(String),
    /// `t"..."` (F.6.1): the text of the literal after escapes
    Tpl(String),
    P(&'static str),
    Nl,
    Indent,
    Dedent,
    Eof,
}

impl Tok {
    /// How a diagnostic names this token (no internal representation).
    pub fn describe(&self) -> String {
        match self {
            Tok::Id(s) => format!("`{s}`"),
            Tok::Int(v) => format!("integer `{v}`"),
            Tok::Float(v) => format!("float `{v}`"),
            Tok::Str(_) => "a text literal".into(),
            Tok::Tpl(_) => "a template literal".into(),
            Tok::P(p) => format!("`{p}`"),
            Tok::Nl => "end of line".into(),
            Tok::Indent => "an indented block".into(),
            Tok::Dedent => "end of block".into(),
            Tok::Eof => "end of file".into(),
        }
    }
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
    if let Some(k) = src.find('\t') {
        let line = src[..k].matches('\n').count() + 1;
        return Err(format!("line {line}: W-LEX-1 tab character"));
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
                    if s[j] == '\r' && s.get(j + 1) != Some(&'\n') {
                        return Err(format!("line {line}: W-LEX-1 bare CR"));
                    }
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
                if s[i] == '\r' && s.get(i + 1) != Some(&'\n') {
                    return Err(format!("line {line}: W-LEX-1 bare CR"));
                }
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
            // 2.5 and maximal munch (2.3): "0x", "0o" and "0b" belong to the literal
            // only when a digit of the base follows; otherwise the token is `0` and
            // the letter starts the next token. An "_" belongs to it only before a digit.
            let radix = radix.filter(|r| {
                s.get(i + 2).is_some_and(|ch| match r {
                    'x' => ch.is_ascii_hexdigit(),
                    'o' => ('0'..='7').contains(ch),
                    _ => *ch == '0' || *ch == '1',
                })
            });
            if let Some(r) = radix {
                i += 2;
                let ds = i;
                let ok = |ch: char| match r {
                    'x' => ch.is_ascii_hexdigit(),
                    'o' => ('0'..='7').contains(&ch),
                    _ => ch == '0' || ch == '1',
                };
                while i < s.len() && (ok(s[i]) || (s[i] == '_' && i + 1 < s.len() && ok(s[i + 1]))) {
                    i += 1;
                }
                let digits: String = s[ds..i].iter().filter(|c| **c != '_').collect();
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
            let quote_at = i;
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
            // F.6.1: `t` written directly before the quote is the template prefix
            let tpl = quote_at >= 1
                && s[quote_at - 1] == 't'
                && (quote_at < 2 || !(s[quote_at - 2].is_alphanumeric() || s[quote_at - 2] == '_'))
                && matches!(out.last(), Some((Tok::Id(x), _)) if x == "t");
            if tpl {
                out.pop();
                out.push((Tok::Tpl(v), line));
            } else {
                out.push((Tok::Str(v), line));
            }
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
    pub t: Vec<(Tok, usize)>,
    p: usize,
    /// a `with budget ... else as e` was parsed: the builtin
    /// `struct Exhausted { limit: u64, used: u64 }` is added to the module
    pub need_exhausted: bool,
    /// token indices consumed as 0.9 keywords (the rest are refused as names)
    pub kw_used: Vec<usize>,
    /// refusals found while expanding template renderers (W-TPL-1, W-TPL-2)
    pub tpl_errs: Vec<String>,
    /// names of the `ct fn` functions parsed so far (F.5.4)
    pub ct_names: Vec<String>,
}

type R<T> = Result<T, String>;

fn is_kw(s: &str) -> bool {
    KEYWORDS.contains(&s)
}

/// Whether `s` is reserved in a module of the given version (2.4, 2.7).
pub fn reserved(s: &str) -> bool {
    KEYWORDS.contains(&s) || DRAFT06_KEYWORDS.contains(&s)
}

impl Parser {
    pub fn new(t: Vec<(Tok, usize)>) -> Self {
        Parser { t, p: 0, need_exhausted: false, kw_used: vec![], tpl_errs: vec![], ct_names: vec![] }
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
            "line {}: W-SYN-1 {m} (found {})",
            self.line(),
            self.peek().describe()
        ))
    }
    fn bump(&mut self) -> Tok {
        let t = self.t[self.p].0.clone();
        if self.p < self.t.len() - 1 {
            self.p += 1;
        }
        t
    }
    /// The head word used to dispatch an item or statement.
    fn head(&self, kw: String) -> String {
        kw
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
            Tok::Id(s) if !reserved(&s) && s != "_" => {
                self.bump();
                Ok(s)
            }
            _ => self.err("expected a name"),
        }
    }
    fn is_name(&self) -> bool {
        matches!(self.peek(), Tok::Id(s) if !reserved(s) && s != "_")
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
            pubs: Default::default(),
            v09_hygiene: vec![],
            x09: Default::default(),
        };
        if self.eat_kw("module") {
            m.path = Some(self.path()?);
            self.nl()?;
        }
        while !matches!(self.peek(), Tok::Eof) {
            let pubd = matches!(self.peek(), Tok::Id(s) if s == "pub");
            let it = self.item()?;
            if pubd && let Some(n) = crate::link::item_name(&it) {
                m.pubs.insert(n);
            }
            m.items.push(it);
        }
        m.x09.ct = self.ct_names.iter().cloned().collect();
        if self.need_exhausted
            && !m.items.iter().any(|it| matches!(it, Item::Struct { name, .. } if name == "Exhausted"))
        {
            let f = |n: &str| Field { name: n.into(), ty: Ty::Path(vec!["u64".into()], vec![]), init: None, line: 0 };
            m.items.push(Item::Struct {
                name: "Exhausted".into(),
                generics: vec![],
                fields: vec![f("limit"), f("used")],
                line: 0,
                linear: false,
            });
        }
        Ok(m)
    }

    fn item(&mut self) -> R<Item> {
        let line = self.line();
        let kw = match self.peek() {
            Tok::Id(s) => s.clone(),
            _ => return self.err("expected a declaration"),
        };
        if kw == "label" && matches!(self.peek2(), Tok::Id(_)) {
            // F.4.2 `label NAME [by AUTH]`; carried as a struct named `@label:NAME`
            self.kw_used.push(self.p);
            self.bump();
            let name = self.name()?;
            let mut fields = vec![];
            if matches!(self.peek(), Tok::Id(b) if b == "by") {
                self.bump();
                let a = self.name()?;
                fields.push(Field { name: "by".into(), ty: Ty::Path(vec![a], vec![]), init: None, line });
            }
            self.nl()?;
            return Ok(Item::Struct { name: format!("@label:{name}"), generics: vec![], fields, line, linear: false });
        }
        let pubd = kw == "pub";
        if pubd {
            self.bump();
            let k = match self.peek() {
                Tok::Id(s) => s.clone(),
                _ => return self.err("expected a declaration after pub"),
            };
            if ![
                "const", "type", "struct", "linear", "kind", "trait", "fn", "comptime", "actor", "ct", "ghost",
                "protocol",
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
        let kw = self.head(kw);
        match kw.as_str() {
            "use" => {
                let line = self.line();
                self.bump();
                let p = self.path()?;
                let a = if self.eat_kw("as") {
                    Some(self.name()?)
                } else {
                    None
                };
                self.nl()?;
                Ok(Item::Use(p, a, line))
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
                // `where` is contextual: no type is followed by a name, so
                // it reserves no identifier (5.1 refined types, C.15 extension).
                let pred = if matches!(self.peek(), Tok::Id(w) if w == "where") {
                    self.bump();
                    Some(self.expr()?)
                } else {
                    None
                };
                self.nl()?;
                Ok(Item::TypeAlias {
                    name,
                    generics,
                    ty,
                    pred,
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
            "state" | "replicated" => {
                let replicated = self.eat_kw("replicated");
                self.want_kw("state")?;
                let name = self.name()?;
                self.nl()?;
                self.indent()?;
                let mut fields = vec![];
                while !self.is_dedent() {
                    fields.push(self.field()?);
                }
                self.bump();
                Ok(Item::State {
                    name,
                    fields,
                    line,
                    replicated,
                })
            }
            "event" | "error" | "interrupt" => {
                let intr = self.eat_kw("interrupt");
                if intr && !self.is_kw("event") {
                    return self.err("expected `event` after `interrupt`");
                }
                let kw = match self.bump() {
                    Tok::Id(k) => k,
                    _ => unreachable!(),
                };
                let name = self.name()?;
                let payload = if self.is_p("(") {
                    self.payload()?
                } else {
                    vec![]
                };
                let prio = if kw == "event" && self.eat_kw("priority") {
                    Some(self.expr()?)
                } else {
                    None
                };
                if intr && prio.is_none() {
                    return self.err("an interrupt event needs `priority p`");
                }
                let interrupt = if intr {
                    prio
                } else if prio.is_some() {
                    return self.err("`priority` is only for interrupt events");
                } else {
                    None
                };
                self.nl()?;
                Ok(if kw == "event" {
                    Item::Event {
                        name,
                        payload,
                        line,
                        interrupt,
                        foreign: None,
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
            "ct" | "ghost" if matches!(self.peek2(), Tok::Id(k) if k == "fn") => Ok(Item::Fn(self.fn_decl()?)),
            "extern" => {
                self.bump();
                let abi = match self.bump() {
                    Tok::Str(s) => s,
                    _ => return self.err("extern needs an ABI string"),
                };
                // F.5.6: `extern "model" event Name(fields)` declares the event
                // on which the host delivers that runtime's answers
                if self.eat_kw("event") {
                    let name = self.name()?;
                    let payload = if self.is_p("(") { self.payload()? } else { vec![] };
                    self.nl()?;
                    return Ok(Item::Event {
                        name,
                        payload,
                        line,
                        interrupt: None,
                        foreign: Some(abi),
                    });
                }
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
                let deadline = if self.eat_kw("deadline") {
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
                    deadline,
                })
            }
            "protocol" => {
                self.bump();
                let name = self.name()?;
                self.nl()?;
                let steps = self.proto_block()?;
                Ok(Item::Protocol { name, steps, line })
            }
            "node" => {
                self.bump();
                let name = self.name()?;
                self.want_p("{")?;
                let mut actors = vec![self.name()?];
                while self.eat_p(",") {
                    actors.push(self.name()?);
                }
                self.want_p("}")?;
                self.nl()?;
                Ok(Item::Node { name, actors, line })
            }
            "supervisor" => {
                self.bump();
                let name = self.name()?;
                self.want_kw("strategy")?;
                let strategy = self.name()?;
                self.nl()?;
                self.indent()?;
                let mut children = vec![];
                while !self.is_dedent() {
                    let l = self.line();
                    self.want_kw("child")?;
                    let n = self.name()?;
                    self.want_p(":")?;
                    let ty = self.path()?.join(".");
                    let restart = if self.eat_kw("restart") {
                        self.want_kw("bound")?;
                        match self.bump() {
                            Tok::Int(k) if (0..=u32::MAX as i128).contains(&k) => k as u32,
                            _ => return self.err("restart bound needs an integer"),
                        }
                    } else {
                        3
                    };
                    self.nl()?;
                    children.push(SupChild {
                        name: n,
                        ty,
                        restart,
                        line: l,
                    });
                }
                self.bump();
                Ok(Item::Supervisor {
                    name,
                    strategy,
                    children,
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

    /// INDENT ProtoStep { ProtoStep } DEDENT (C.11.3)
    fn proto_block(&mut self) -> R<Vec<ProtoStep>> {
        self.indent()?;
        let mut v = vec![];
        while !self.is_dedent() {
            v.push(self.proto_step()?);
        }
        self.bump();
        Ok(v)
    }
    fn proto_step(&mut self) -> R<ProtoStep> {
        if self.eat_kw("send") {
            let t = self.ty()?;
            self.nl()?;
            Ok(ProtoStep::Send(t))
        } else if self.eat_kw("recv") {
            let t = self.ty()?;
            self.nl()?;
            Ok(ProtoStep::Recv(t))
        } else if self.is_kw("offer") || self.is_kw("choose") {
            let offer = self.is_kw("offer");
            self.bump();
            self.nl()?;
            self.indent()?;
            let mut arms = vec![];
            while !self.is_dedent() {
                let l = self.name()?;
                self.nl()?;
                let body = if matches!(self.peek(), Tok::Indent) {
                    self.proto_block()?
                } else {
                    vec![]
                };
                arms.push((l, body));
            }
            self.bump();
            Ok(if offer {
                ProtoStep::Offer(arms)
            } else {
                ProtoStep::Choose(arms)
            })
        } else if self.eat_kw("loop") {
            self.nl()?;
            Ok(ProtoStep::Loop(self.proto_block()?))
        } else if self.eat_kw("break") {
            self.nl()?;
            Ok(ProtoStep::Break)
        } else {
            self.err("expected send, recv, offer, choose, loop or break")
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
        let (mut kept, mut value) = (None, None);
        // `kept` is contextual (COMP-2): the outcome keyword only where it
        // is followed by `:`, so it reserves no name
        if self.is_kw("kept") && matches!(self.peek2(), Tok::P(":")) {
            self.bump();
            self.want_p(":")?;
            let n = self.name()?;
            // a step ends in a value, a named error or a trap (T2), so an
            // example can state each: `kept: trap("division by zero")`
            if n == "trap" && self.eat_p("(") {
                let k = match self.bump() {
                    Tok::Str(s) => s,
                    _ => return self.err("`kept: trap(…)` needs the trap text"),
                };
                self.want_p(")")?;
                kept = Some(format!("trap({k})"));
            } else {
                kept = Some(n);
            }
        } else {
            value = Some(self.expr()?);
        }
        self.nl()?;
        Ok(Example {
            events,
            kept,
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
        let bound = if self.eat_kw("bound") {
            match self.bump() {
                Tok::Int(k) if (1..=u64::MAX as i128).contains(&k) => Some(k as u64),
                _ => return self.err("a recursion bound is a positive integer literal"),
            }
        } else {
            None
        };
        Ok(FnSig {
            comptime,
            name,
            generics,
            params,
            ret,
            bound,
            line,
            x09: 0,
        })
    }
    fn fn_decl(&mut self) -> R<FnDecl> {
        // F.5.4 `ct fn` (FnDecl = [ "ct" ] FnSig ...)
        let ct = self.is_kw("ct") && matches!(self.peek2(), Tok::Id(k) if k == "fn");
        if ct {
            self.kw_used.push(self.p);
            self.bump();
        }
        let ghost = !ct && self.is_kw("ghost") && matches!(self.peek2(), Tok::Id(k) if k == "fn");
        if ghost {
            self.kw_used.push(self.p);
            self.bump();
        }
        let mut sig = self.fn_sig()?;
        if ct {
            self.ct_names.push(sig.name.clone());
            sig.x09 |= 1;
        }
        if ghost {
            sig.x09 |= 2;
        }
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
                if self.is_kw("ghost") {
                    return self.err("W-RES-1 `ghost` parameters (F.6.3) are specified but not implemented by this checker, so they are reserved, not implemented");
                }
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
            if self.is_p("{") {
                // F.4.2 label set `T @ {A, B}`; carried as a region named `@{A,B}`
                // until `v09::erase` removes it (surface reading only)
                self.bump();
                let mut names = vec![];
                while !self.is_p("}") {
                    names.push(self.name()?);
                    if !self.eat_p(",") {
                        break;
                    }
                }
                self.want_p("}")?;
                return Ok(Ty::Region(Box::new(b), format!("@{{{}}}", names.join(","))));
            }
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
        // draft 0.6b (C.11.3, C.11.8)
        if self.eat_kw("session") {
            let p = self.type_path()?;
            return Ok(Ty::Path(vec!["session".into()], vec![p]));
        }
        if self.eat_kw("lvar") {
            let t = self.base_ty()?;
            return Ok(Ty::Path(vec!["lvar".into()], vec![t]));
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
        let kw = self.head(kw);
        if kw == "requires" || kw == "ensures" {
            // F.6.2: a clause is carried as `check @requires(e)` until `tier3::lower`
            self.kw_used.push(self.p);
            self.bump();
            let e = self.expr()?;
            self.nl()?;
            return Ok(Stmt::Check(Expr::Call(Box::new(Expr::Name(format!("@{kw}"), line)), vec![Arg { label: None, e }], line, false)));
        }
        if kw == "decreases" {
            // F.6.4: carried as `check @decreases(m1, m2, ...)` until `tier3::lower`
            self.kw_used.push(self.p);
            self.bump();
            let args = self.args()?;
            self.nl()?;
            return Ok(Stmt::Check(Expr::Call(Box::new(Expr::Name("@decreases".into(), line)), args, line, false)));
        }
        if kw == "ghost" && matches!(self.peek2(), Tok::Id(k) if k == "let") {
            self.kw_used.push(self.p);
            self.bump();
            return match self.stmt()? {
                Stmt::Let(p, t, e, l) => Ok(Stmt::Let(p, t, Expr::Call(Box::new(Expr::Name("@ghost".into(), l)), vec![Arg { label: None, e }], l, false), l)),
                _ => self.err("ghost is followed by let"),
            };
        }
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
            "if" if matches!(self.peek2(), Tok::Id(k) if k == "let") => {
                // F.6.5, block form: `if let P = e [and c]... Block [else Block]` is carried as
                // a `match` on `@iflet(e)` with the guard `c and ...` and a wildcard arm
                self.bump();
                self.bump();
                let pat = self.pattern()?;
                self.want_p("=")?;
                let scrut = self.cmp_expr()?;
                let mut guard: Option<Expr> = None;
                while self.eat_kw("and") {
                    if self.is_kw("let") {
                        return self.err("W-RES-1 a chain `and let P = e` (F.6.5) is specified but not implemented by this checker, so it is reserved, not implemented");
                    }
                    let l = self.line();
                    let c = self.cmp_expr()?;
                    guard = Some(match guard {
                        None => c,
                        Some(g) => Expr::Bin("and".into(), Box::new(g), Box::new(c), l),
                    });
                }
                let body = self.block()?;
                let els = if self.eat_kw("else") { self.block()? } else { vec![Stmt::Pass] };
                let sc = Expr::Call(Box::new(Expr::Name("@iflet".into(), line)), vec![Arg { label: None, e: scrut }], line, false);
                Ok(Stmt::Match(
                    sc,
                    vec![
                        Arm { pat, guard, body, line },
                        Arm { pat: Pat::Wild, guard: None, body: els, line },
                    ],
                    line,
                ))
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
                // (the AST has no new variant, see Appendix C, book/14-draft06-concurrency.md C.8).
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
                let mut st = Stmt::Spawn(e);
                if self.eat_kw("priority") {
                    let pr = self.expr()?;
                    st = Stmt::Arena("$priority".into(), None, vec![Stmt::Expr(pr), st], line);
                }
                // draft 0.6b `-> h` join handle (C.11.9)
                if self.is_p("->") {
                    self.bump();
                    let h = self.name()?;
                    st = Stmt::Arena(format!("$spawnas:{h}"), None, vec![st], line);
                }
                self.nl()?;
                Ok(st)
            }
            "barrier" => {
                self.bump();
                self.nl()?;
                Ok(Stmt::Arena("$barrier".into(), None, vec![], line))
            }
            "scope" | "atomic" => {
                self.bump();
                let body = self.block()?;
                Ok(Stmt::Arena(format!("${kw}"), None, body, line))
            }
            "with" if matches!(self.peek2(), Tok::Id(k) if k == "group") => {
                self.bump();
                self.bump();
                let g = self.name()?;
                self.want_kw("lanes")?;
                let n = self.expr()?;
                let body = self.block()?;
                let mut v = vec![Stmt::Expr(n)];
                v.extend(body);
                Ok(Stmt::Arena(format!("$group:{g}"), None, v, line))
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
            "with" if matches!(self.peek2(), Tok::Id(k) if k == "budget" || k == "deadline") =>
            {
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
                    if self.eat_kw("else") {
                        // `else as e` binds the typed outcome
                        // Exhausted{limit, used} (C.5); the runtimes supply
                        // the two values through hidden names
                        let bind = if self.eat_kw("as") { Some(self.pattern()?) } else { None };
                        let mut eb = self.block()?;
                        if let Some(pat) = bind {
                            self.need_exhausted = true;
                            let arg = |l: &str, n: &str| Arg {
                                label: Some(l.into()),
                                e: Expr::Name(format!("\u{1}{n}"), line),
                            };
                            eb.insert(0, Stmt::Let(
                                pat,
                                None,
                                Expr::Call(
                                    Box::new(Expr::Name("Exhausted".into(), line)),
                                    vec![arg("limit", "__bn"), arg("used", "__bu")],
                                    line,
                                    false,
                                ),
                                line,
                            ));
                        }
                        // the else runs only when the budget traps: encoded as
                        // `if false` so a run that never charges past n never
                        // evaluates it (comptime, emitters' observation model)
                        v.push(Stmt::If(vec![(Expr::Bool(false, line), eb)], None));
                    }
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
        if let Tok::Id(w) = self.peek().clone()
            && (w == "release" || w == "endorse")
        {
            // F.4.2: `release e` and `endorse e` are prefix forms; carried as a
            // call of `@release` / `@endorse` until `v09::erase` removes them
            self.kw_used.push(self.p);
            self.bump();
            let a = self.unary()?;
            return Ok(Expr::Call(Box::new(Expr::Name(format!("@{w}"), l)), vec![Arg { label: None, e: a }], l, false));
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
                e = Expr::Call(Box::new(e), a, l, true);
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
                if let Tok::Tpl(raw) = self.peek().clone() {
                    self.bump();
                    e = self.template(e, &raw, l);
                    continue;
                }
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
    /// F.6.1: `C.t"..."` and `text.t"..."` are expanded here to the
    /// concatenation they mean, so every later pass sees the fields as reads.
    fn template(&mut self, base: Expr, raw: &str, l: usize) -> Expr {
        let ctx = match &base {
            Expr::Name(n, _) if n == "text" || ["Html", "Url", "Sql", "Json", "ShellArg", "Prompt"].contains(&n.as_str()) => n.clone(),
            _ => {
                self.tpl_errs.push(format!("line {l}: W-TPL-1 a template is rendered by `text.t\"...\"` or by a context (Html, Url, Sql, Json, ShellArg, Prompt), never used as a value"));
                return Expr::Str(String::new(), l);
            }
        };
        let cs: Vec<char> = raw.chars().collect();
        let mut pieces: Vec<Result<String, Expr>> = vec![];
        let mut lit = String::new();
        let mut i = 0;
        while i < cs.len() {
            match cs[i] {
                '{' if cs.get(i + 1) == Some(&'{') => {
                    lit.push('{');
                    i += 2;
                }
                '}' if cs.get(i + 1) == Some(&'}') => {
                    lit.push('}');
                    i += 2;
                }
                '{' => {
                    let Some(end) = cs[i..].iter().position(|c| *c == '}') else {
                        self.tpl_errs.push(format!("line {l}: W-TPL-2 a `{{` in a template starts a field that is never closed (write `{{{{` for a brace)"));
                        return Expr::Str(String::new(), l);
                    };
                    let field: String = cs[i + 1..i + end].iter().collect();
                    let ok = !field.is_empty()
                        && field.split('.').all(|p| {
                            let mut ch = p.chars();
                            matches!(ch.next(), Some(c) if c.is_ascii_alphabetic() || c == '_') && ch.all(|c| c.is_ascii_alphanumeric() || c == '_')
                        });
                    if !ok {
                        self.tpl_errs.push(format!("line {l}: W-TPL-2 a template field is an identifier or a field path `{{a.b}}`, never an expression or empty: `{{{field}}}`"));
                        return Expr::Str(String::new(), l);
                    }
                    let mut parts = field.split('.');
                    let mut fe = Expr::Name(parts.next().unwrap().to_string(), l);
                    for p in parts {
                        fe = Expr::Member(Box::new(fe), p.to_string(), l);
                    }
                    if !lit.is_empty() {
                        pieces.push(Ok(std::mem::take(&mut lit)));
                    }
                    pieces.push(Err(fe));
                    i += end + 1;
                }
                '}' => {
                    self.tpl_errs.push(format!("line {l}: W-TPL-2 a `}}` in a template closes no field (write `}}}}` for a brace)"));
                    return Expr::Str(String::new(), l);
                }
                c => {
                    lit.push(c);
                    i += 1;
                }
            }
        }
        if !lit.is_empty() || pieces.is_empty() {
            pieces.push(Ok(lit));
        }
        let call = |obj: Expr, m: &str, a: Expr| Expr::Call(Box::new(Expr::Member(Box::new(obj), m.into(), l)), vec![Arg { label: None, e: a }], l, true);
        let mut out: Option<Expr> = None;
        for p in pieces {
            let piece = match p {
                Ok(s) if ctx == "text" => Expr::Str(s, l),
                Ok(s) => call(Expr::Name(ctx.clone(), l), "lit", Expr::Str(s, l)),
                Err(f) if ctx == "text" => Expr::Call(Box::new(Expr::Member(Box::new(f), "tostr".into(), l)), vec![], l, true),
                Err(f) => call(Expr::Name(ctx.clone(), l), "encode", f),
            };
            out = Some(match out {
                None => piece,
                Some(a) => Expr::Bin("+".into(), Box::new(a), Box::new(piece), l),
            });
        }
        out.unwrap()
    }
    fn primary(&mut self) -> R<Expr> {
        let l = self.line();
        if matches!(self.peek(), Tok::Tpl(_)) {
            return self.err("W-RES-1 a template literal that is not rendered at once (`text.t\"...\"` or `C.t\"...\"`) is a `Template` value, which is specified (F.6.1) but not implemented by this checker, so it is reserved, not implemented");
        }
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

/// Keywords of Appendix C (concurrency). Reserved in every module, like
/// `KEYWORDS`; the two lists are separate only because the LALR(1) check
/// builds the section 3 and Appendix C grammars in turn.
pub const DRAFT06_KEYWORDS: &[&str] = &[
    "par", "branch", "nondet", "budget", "deadline", "priority", "replicated", "interrupt",
    "session", "lvar", "group", "lanes", "barrier", "scope", "atomic", "protocol", "offer",
    "choose", "node", "supervisor", "strategy", "child", "restart",
];

/// The release label (spec 2.7): the first line may be the comment
/// `# rce-version: V`. It is a label only: tolerated, ignored, not hashed, and
/// it gates nothing. There is one language.
/// The release of the one language this checker implements.
pub const LANGUAGE_RELEASE: &str = "0.9";

pub fn parse(src: &str) -> Result<Module, String> {
    parse_as(src, true)
}

/// The module as written, with `&World` parameters still in place (spec F.4.1
/// erases them for evaluation; the formatter and tools that show source use
/// this reading).
pub fn parse_surface(src: &str) -> Result<Module, String> {
    parse_as(src, false)
}

fn parse_as(src: &str, erase: bool) -> Result<Module, String> {
    crate::v09::lexical(src)?;
    let toks = lex(src)?;
    crate::v09::gate(&toks)?;
    let mut p = Parser::new(toks);
    let mut m = p.module()?;
    let tpl_errs = std::mem::take(&mut p.tpl_errs);
    crate::v09::stray_words(&p.t, &p.kw_used)?;
    let mut hy = crate::v09::hygiene(&m);
    let (we, wf) = crate::v09::world(&m);
    hy.extend(we);
    hy.extend(crate::v09::transactional(&m));
    hy.extend(crate::ifc::check(&m));
    hy.extend(crate::tier2::s2(&m));
    hy.extend(crate::tier2::s6(&m));
    hy.extend(crate::tier2::s8(&m));
    hy.extend(crate::tier3::check(&m));
    hy.extend(tpl_errs);
    hy.sort();
    hy.dedup();
    m.v09_hygiene = hy;
    if erase {
        let mut wf = wf;
        for (k, v) in crate::v09::dir_cap_positions(&m) {
            let e = wf.entry(k).or_default();
            e.extend(v);
            e.sort();
        }
        m = crate::tier3::lower(m)?;
        m = crate::v09::erase_world(m, &wf)?;
    }
    Ok(m)
}
