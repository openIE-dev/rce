//! Language features of `rce lsp` (10.7): hover, go-to-definition,
//! completion and document symbols. All four read one index built from the
//! parsed module and the checker's declarations. The index uses no network
//! and no model; a document that does not parse gives an empty index.
use crate::ast::*;
use crate::check::Checker;
use crate::parse::{KEYWORDS, parse};
use serde_json::{Value, json};
use std::collections::HashSet;

/// Builtin methods of the reference evaluator, offered after `.`.
pub const BUILTIN_METHODS: &[&str] = &[
    "abs",
    "append",
    "ceil",
    "chars",
    "clear",
    "collect",
    "contains",
    "endswith",
    "filter",
    "floor",
    "fold",
    "get",
    "insert",
    "isalpha",
    "isdigit",
    "isempty",
    "isnone",
    "issome",
    "isspace",
    "join",
    "keys",
    "len",
    "lines",
    "lower",
    "map",
    "max",
    "min",
    "or",
    "pop",
    "push",
    "remove",
    "reverse",
    "round",
    "sort",
    "split",
    "sqrt",
    "startswith",
    "strip",
    "sum",
    "toint",
    "tostr",
    "upper",
    "usize",
    "values",
    "with",
    "words",
];

/// A named thing in a document, 1-based lines.
#[derive(Clone, Debug)]
pub struct Sym {
    pub name: String,
    /// LSP SymbolKind.
    pub kind: u32,
    pub line: usize,
    pub end: usize,
    pub detail: String,
    pub children: Vec<Sym>,
}

pub struct Index {
    pub lines: Vec<String>,
    pub items: Vec<Sym>,
    /// Local bindings: (name, line, type as text).
    pub locals: Vec<(String, usize, String)>,
}

fn item_line(i: &Item) -> Option<usize> {
    Some(match i {
        Item::Use(..) | Item::Intent(..) => return None,
        Item::Example(x) => x.line,
        Item::Check(e) => e.line(),
        Item::Const { line, .. }
        | Item::TypeAlias { line, .. }
        | Item::Struct { line, .. }
        | Item::State { line, .. }
        | Item::Event { line, .. }
        | Item::Error { line, .. }
        | Item::Kind { line, .. }
        | Item::Trait { line, .. }
        | Item::Impl { line, .. }
        | Item::Extern { line, .. }
        | Item::Step { line, .. }
        | Item::Actor { line, .. } => *line,
        Item::Fn(f) => f.sig.line,
    })
}

impl Index {
    pub fn new(text: &str) -> Index {
        let lines: Vec<String> = text
            .split('\n')
            .map(|l| l.trim_end_matches('\r').to_string())
            .collect();
        let mut ix = Index {
            lines,
            items: vec![],
            locals: vec![],
        };
        let Ok(m) = parse(text) else { return ix };
        let mut c = Checker::new();
        c.check_module(&m, &HashSet::new());
        ix.locals = c
            .decls
            .iter()
            .map(|(n, l, t)| (n.clone(), *l, c.sub.resolve(t).to_string()))
            .collect();
        let mut starts: Vec<usize> = m.items.iter().filter_map(item_line).collect();
        starts.sort();
        starts.dedup();
        let last = ix.lines.len();
        let end_of = |l: usize, lines: &[String]| -> usize {
            let mut e = starts
                .iter()
                .find(|&&s| s > l)
                .map(|s| s - 1)
                .unwrap_or(last);
            while e > l
                && lines
                    .get(e - 1)
                    .is_some_and(|x| x.trim().is_empty() || x.trim_start().starts_with('#'))
            {
                e -= 1;
            }
            e
        };
        for i in &m.items {
            let Some(line) = item_line(i) else { continue };
            let end = end_of(line, &ix.lines);
            let head = ix
                .lines
                .get(line - 1)
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let block = ix.lines[line - 1..end.min(last)].join("\n");
            let mk = |name: &str, kind: u32, detail: String, children: Vec<Sym>| Sym {
                name: name.to_string(),
                kind,
                line,
                end,
                detail,
                children,
            };
            let fields = |fs: &[Field], kind: u32| -> Vec<Sym> {
                fs.iter()
                    .map(|f| Sym {
                        name: f.name.clone(),
                        kind,
                        line: f.line,
                        end: f.line,
                        detail: ix
                            .lines
                            .get(f.line - 1)
                            .map(|s| s.trim().to_string())
                            .unwrap_or_default(),
                        children: vec![],
                    })
                    .collect()
            };
            let sub_lines = |names: Vec<String>, kind: u32| -> Vec<Sym> {
                names
                    .into_iter()
                    .filter_map(|n| {
                        let l = (line + 1..=end).find(|&l| {
                            let t = ix.lines[l - 1].trim_start();
                            t.starts_with(&n)
                                && !t[n.len()..]
                                    .starts_with(|c: char| c.is_alphanumeric() || c == '_')
                        })?;
                        Some(Sym {
                            name: n,
                            kind,
                            line: l,
                            end: l,
                            detail: ix.lines[l - 1].trim().to_string(),
                            children: vec![],
                        })
                    })
                    .collect()
            };
            let s = match i {
                Item::Fn(f) => mk(&f.sig.name, 12, head, vec![]),
                Item::Extern { sig, .. } => mk(&sig.name, 12, head, vec![]),
                Item::Const { name, .. } => mk(name, 14, head, vec![]),
                Item::TypeAlias { name, .. } => mk(name, 26, head, vec![]),
                Item::Struct {
                    name, fields: fs, ..
                } => mk(name, 23, block, fields(fs, 8)),
                Item::State {
                    name, fields: fs, ..
                } => mk(name, 23, block, fields(fs, 8)),
                Item::Event { name, .. } => mk(name, 24, head, vec![]),
                Item::Error { name, .. } => mk(name, 23, head, vec![]),
                Item::Kind { name, variants, .. } => mk(
                    name,
                    10,
                    block,
                    sub_lines(variants.iter().map(|v| v.0.clone()).collect(), 22),
                ),
                Item::Trait { name, .. } => mk(name, 11, block, vec![]),
                Item::Actor {
                    name, fields: fs, ..
                } => mk(name, 5, head, fields(fs, 8)),
                Item::Impl { ty, members, .. } => {
                    let h = crate::check::ty_head(ty);
                    let ms = members
                        .iter()
                        .map(|f| Sym {
                            name: f.sig.name.clone(),
                            kind: 6,
                            line: f.sig.line,
                            end: f.sig.line,
                            detail: ix
                                .lines
                                .get(f.sig.line - 1)
                                .map(|s| s.trim().to_string())
                                .unwrap_or_default(),
                            children: vec![],
                        })
                        .collect();
                    mk(&format!("impl {h}"), 5, head, ms)
                }
                Item::Step { region, .. } => mk(&format!("step {region}"), 24, head, vec![]),
                Item::Example(_) => mk("example", 24, head, vec![]),
                Item::Check(_) => mk("check", 24, head, vec![]),
                Item::Use(..) | Item::Intent(..) => continue,
            };
            ix.items.push(s);
        }
        ix
    }

    /// The identifier at a 0-based position, with its start column and
    /// whether a `.` precedes it.
    pub fn word_at(&self, line: usize, col: usize) -> Option<(String, usize, bool)> {
        let chars: Vec<char> = self.lines.get(line)?.chars().collect();
        let id = |c: char| c.is_alphanumeric() || c == '_';
        let mut s = col.min(chars.len());
        while s > 0 && id(chars[s - 1]) {
            s -= 1;
        }
        let mut e = col.min(chars.len());
        while e < chars.len() && id(chars[e]) {
            e += 1;
        }
        let dot = s > 0 && chars[s - 1] == '.';
        Some((chars[s..e].iter().collect(), s, dot))
    }

    /// The top-level item whose lines contain 1-based line `l`.
    fn enclosing(&self, l: usize) -> Option<&Sym> {
        self.items
            .iter()
            .filter(|s| s.line <= l && l <= s.end)
            .max_by_key(|s| s.line)
    }

    /// The binding of local `n` visible at 1-based line `l`: the latest one
    /// declared in the same item at or before `l`.
    fn local(&self, n: &str, l: usize) -> Option<&(String, usize, String)> {
        let it = self.enclosing(l)?;
        self.locals
            .iter()
            .filter(|(x, d, _)| x == n && *d >= it.line && *d <= l)
            .max_by_key(|x| x.1)
    }

    fn all(&self) -> impl Iterator<Item = &Sym> {
        self.items
            .iter()
            .chain(self.items.iter().flat_map(|s| s.children.iter()))
    }

    /// What `n` at 1-based line `l` refers to: (definition line, hover text).
    pub fn resolve(&self, n: &str, l: usize, after_dot: bool) -> Option<(usize, String)> {
        if n.is_empty() || KEYWORDS.contains(&n) {
            return None;
        }
        if after_dot {
            if let Some(m) = self
                .items
                .iter()
                .flat_map(|s| s.children.iter())
                .find(|c| c.name == n && c.kind == 6)
            {
                return Some((m.line, m.detail.clone()));
            }
            if let Some(f) = self
                .items
                .iter()
                .flat_map(|s| s.children.iter())
                .find(|c| c.name == n && c.kind == 8)
            {
                return Some((f.line, f.detail.clone()));
            }
            return None;
        }
        if let Some((_, d, t)) = self.local(n, l) {
            return Some((*d, format!("{n}: {t}")));
        }
        if let Some(s) = self.items.iter().find(|s| s.name == n) {
            return Some((s.line, s.detail.clone()));
        }
        // variants, fields (a state's fields are names inside its steps), methods
        self.all()
            .find(|s| s.name == n)
            .map(|s| (s.line, s.detail.clone()))
    }

    pub fn hover(&self, line: usize, col: usize) -> Value {
        let Some((w, _, dot)) = self.word_at(line, col) else {
            return Value::Null;
        };
        match self.resolve(&w, line + 1, dot) {
            Some((_, text)) => {
                json!({"contents": {"kind": "markdown", "value": format!("```rce\n{text}\n```")}})
            }
            None => Value::Null,
        }
    }

    pub fn definition(&self, uri: &str, line: usize, col: usize) -> Value {
        let Some((w, _, dot)) = self.word_at(line, col) else {
            return Value::Null;
        };
        let Some((d, _)) = self.resolve(&w, line + 1, dot) else {
            return Value::Null;
        };
        let text = self.lines.get(d - 1).cloned().unwrap_or_default();
        let c = find_word(&text, &w).unwrap_or(0);
        json!({"uri": uri, "range": {"start": {"line": d - 1, "character": c},
                                     "end": {"line": d - 1, "character": c + w.chars().count()}}})
    }

    pub fn completion(&self, line: usize, col: usize) -> Value {
        let (prefix, _, dot) = self.word_at(line, col).unwrap_or_default();
        let prefix: String = {
            // only the part before the cursor
            let chars: Vec<char> = self
                .lines
                .get(line)
                .map(|l| l.chars().collect())
                .unwrap_or_default();
            let mut s = col.min(chars.len());
            while s > 0 && (chars[s - 1].is_alphanumeric() || chars[s - 1] == '_') {
                s -= 1;
            }
            let p: String = chars[s..col.min(chars.len())].iter().collect();
            if prefix.starts_with(&p) {
                p
            } else {
                String::new()
            }
        };
        let mut out: Vec<(String, u32, String)> = vec![];
        let mut add = |n: &str, k: u32, d: &str| {
            if n.starts_with(&prefix) && !out.iter().any(|x| x.0 == n) {
                out.push((n.to_string(), k, d.to_string()));
            }
        };
        if dot {
            for s in self.items.iter().flat_map(|s| s.children.iter()) {
                match s.kind {
                    6 => add(&s.name, 2, &s.detail),
                    8 => add(&s.name, 5, &s.detail),
                    _ => {}
                }
            }
            for m in BUILTIN_METHODS {
                add(m, 2, "builtin method");
            }
        } else {
            let l = line + 1;
            if let Some(it) = self.enclosing(l) {
                let mut ls: Vec<&(String, usize, String)> = self
                    .locals
                    .iter()
                    .filter(|(_, d, _)| *d >= it.line && *d <= l)
                    .collect();
                ls.sort_by_key(|x| std::cmp::Reverse(x.1));
                for (n, _, t) in ls {
                    add(n, 6, t);
                }
            }
            for s in &self.items {
                let k = match s.kind {
                    12 => 3,
                    14 => 21,
                    10 => 13,
                    23 => 22,
                    24 => 23,
                    11 => 8,
                    26 => 22,
                    _ => continue,
                };
                if !s.name.contains(' ') && s.name != "example" && s.name != "check" {
                    add(&s.name, k, &s.detail);
                }
                for c in &s.children {
                    match c.kind {
                        22 => add(&c.name, 20, &c.detail),
                        8 => add(&c.name, 5, &c.detail),
                        _ => {}
                    }
                }
            }
            for k in KEYWORDS {
                add(k, 14, "keyword");
            }
        }
        Value::Array(
            out.into_iter()
                .map(|(n, k, d)| json!({"label": n, "kind": k, "detail": d}))
                .collect(),
        )
    }

    pub fn symbols(&self) -> Value {
        fn one(s: &Sym, lines: &[String]) -> Value {
            let text = lines.get(s.line - 1).cloned().unwrap_or_default();
            let c = find_word(&text, s.name.rsplit(' ').next().unwrap_or(&s.name)).unwrap_or(0);
            let end_len = lines.get(s.end - 1).map(|l| l.chars().count()).unwrap_or(0);
            json!({
                "name": s.name, "kind": s.kind,
                "detail": s.detail.lines().next().unwrap_or(""),
                "range": {"start": {"line": s.line - 1, "character": 0}, "end": {"line": s.end - 1, "character": end_len}},
                "selectionRange": {"start": {"line": s.line - 1, "character": c},
                                   "end": {"line": s.line - 1, "character": c + s.name.rsplit(' ').next().unwrap_or("").chars().count()}},
                "children": s.children.iter().map(|x| one(x, lines)).collect::<Vec<_>>(),
            })
        }
        Value::Array(self.items.iter().map(|s| one(s, &self.lines)).collect())
    }
}

/// The column of `w` as a whole word in `text`.
fn find_word(text: &str, w: &str) -> Option<usize> {
    let chars: Vec<char> = text.chars().collect();
    let wc: Vec<char> = w.chars().collect();
    let id = |c: char| c.is_alphanumeric() || c == '_';
    if wc.is_empty() || wc.len() > chars.len() {
        return None;
    }
    (0..=chars.len() - wc.len()).find(|&i| {
        chars[i..].starts_with(&wc)
            && (i == 0 || !id(chars[i - 1]))
            && chars.get(i + wc.len()).is_none_or(|c| !id(*c))
    })
}
