//! LALR(1) check of the RCE grammar, read from its single source: the EBNF
//! of spec section 3 (`book/03-syntax.md`). Replaces the former Lark check
//! (the retired `rce.lark` built with `parser="lalr", strict=True` by
//! the retired check.py).
//!
//! 1. [`book_grammar`] reads every rule `Name = ... .` from the section's
//!    code blocks. Prose beside a rule is skipped: anything after a rule's
//!    closing `.` on its line (unless another rule starts there), and, inside
//!    a rule, a lower-case word and the rest of its line (the annotation
//!    column, e.g. `unit`, `error union  T ! E`).
//! 2. EBNF becomes BNF the way Lark expands it for LALR: `[ X ]` and `{ X }`
//!    are inlined as the alternatives with and without, `( A | B )` as the
//!    product of alternatives, and `{ X }` uses one left-recursive helper
//!    rule `H = X | H X` per distinct X.
//! 3. [`Lalr::build`] makes the LR(0) automaton and propagates LALR(1)
//!    lookaheads to a fixpoint, then lists every shift/reduce and
//!    reduce/reduce conflict. The grammar must have none.
//! 4. [`Lalr::accepts`] runs the table on the token stream of the reference
//!    lexer (layout of 2.6 included), so the LALR parser can be compared
//!    with tree-sitter and the Rust parser on the examples and fuzz programs.
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Sym {
    T(usize),
    N(usize),
}

#[derive(Debug, Default)]
pub struct Grammar {
    pub terms: Vec<String>,
    pub nts: Vec<String>,
    /// (lhs, rhs)
    pub prods: Vec<(usize, Vec<Sym>)>,
    /// rules as written in the source (before EBNF expansion)
    pub source_rules: usize,
}

#[derive(Clone, Debug)]
enum E {
    T(String),
    /// a bare all-capitals name: a terminal unless a rule defines it
    Up(String),
    N(String),
    Seq(Vec<E>),
    Alt(Vec<E>),
    Opt(Box<E>),
    Rep(Box<E>),
}

fn lex_ebnf(block: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in block.lines() {
        let mut toks: Vec<String> = Vec::new();
        let c: Vec<char> = line.chars().collect();
        let mut i = 0;
        let mut skip_rest = false;
        while i < c.len() && !skip_rest {
            let ch = c[i];
            if ch.is_whitespace() {
                i += 1;
            } else if ch == '"' {
                let mut j = i + 1;
                while j < c.len() && c[j] != '"' {
                    j += 1;
                }
                toks.push(c[i..=j.min(c.len() - 1)].iter().collect());
                i = j + 1;
            } else if ch.is_ascii_alphabetic() {
                let mut j = i;
                while j < c.len() && (c[j].is_ascii_alphanumeric() || c[j] == '_') {
                    j += 1;
                }
                let w: String = c[i..j].iter().collect();
                if ch.is_ascii_lowercase() {
                    skip_rest = true; // annotation column
                } else {
                    toks.push(w);
                }
                i = j;
            } else if "=|[]{}().".contains(ch) {
                toks.push(ch.to_string());
                i += 1;
            } else {
                skip_rest = true; // prose punctuation (e.g. `&T` in the annotation column)
            }
        }
        // After a rule's closing "." the rest of the line is prose, unless a rule starts.
        let mut k = 0;
        let mut kept = Vec::new();
        let mut depth = 0i32;
        while k < toks.len() {
            let t = &toks[k];
            match t.as_str() {
                "[" | "{" | "(" => depth += 1,
                "]" | "}" | ")" => depth -= 1,
                _ => {}
            }
            kept.push(t.clone());
            if t == "." && depth == 0 {
                let starts_rule = k + 2 < toks.len()
                    && toks[k + 2] == "="
                    && toks[k + 1].starts_with(|c: char| c.is_ascii_uppercase());
                if !starts_rule {
                    break;
                }
            }
            k += 1;
        }
        out.extend(kept);
    }
    out
}

struct P {
    t: Vec<String>,
    i: usize,
}

impl P {
    fn peek(&self) -> &str {
        self.t.get(self.i).map(String::as_str).unwrap_or("")
    }
    fn alt(&mut self) -> Result<E, String> {
        let mut alts = vec![self.seq()?];
        while self.peek() == "|" {
            self.i += 1;
            alts.push(self.seq()?);
        }
        Ok(if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            E::Alt(alts)
        })
    }
    fn seq(&mut self) -> Result<E, String> {
        let mut v = Vec::new();
        loop {
            let t = self.peek().to_string();
            match t.as_str() {
                "" | "|" | "]" | "}" | ")" | "." => break,
                "[" | "{" | "(" => {
                    self.i += 1;
                    let inner = self.alt()?;
                    let close = match t.as_str() {
                        "[" => "]",
                        "{" => "}",
                        _ => ")",
                    };
                    if self.peek() != close {
                        return Err(format!(
                            "expected {close} at token {} ({:?})",
                            self.i,
                            self.peek()
                        ));
                    }
                    self.i += 1;
                    v.push(match t.as_str() {
                        "[" => E::Opt(Box::new(inner)),
                        "{" => E::Rep(Box::new(inner)),
                        _ => inner,
                    });
                }
                _ if t.starts_with('"') => {
                    v.push(E::T(t.trim_matches('"').to_string()));
                    self.i += 1;
                }
                _ if t.chars().all(|c| c.is_ascii_uppercase() || c == '_') => {
                    v.push(E::Up(t));
                    self.i += 1;
                }
                _ => {
                    // A rule name: stop if it starts the next rule (`Name =`).
                    if self.t.get(self.i + 1).map(String::as_str) == Some("=") {
                        return Err(format!("rule before {t} lacks its closing '.'"));
                    }
                    v.push(E::N(t));
                    self.i += 1;
                }
            }
        }
        Ok(E::Seq(v))
    }
}

struct Expander {
    g: Grammar,
    nt_ix: HashMap<String, usize>,
    t_ix: HashMap<String, usize>,
    helpers: HashMap<String, usize>,
}

impl Expander {
    fn nt(&mut self, n: &str) -> usize {
        if let Some(&i) = self.nt_ix.get(n) {
            return i;
        }
        self.g.nts.push(n.to_string());
        self.nt_ix.insert(n.to_string(), self.g.nts.len() - 1);
        self.g.nts.len() - 1
    }
    fn t(&mut self, n: &str) -> usize {
        if let Some(&i) = self.t_ix.get(n) {
            return i;
        }
        self.g.terms.push(n.to_string());
        self.t_ix.insert(n.to_string(), self.g.terms.len() - 1);
        self.g.terms.len() - 1
    }
    /// All symbol sequences an EBNF expression stands for.
    fn expand(&mut self, e: &E) -> Vec<Vec<Sym>> {
        match e {
            E::T(s) | E::Up(s) => vec![vec![Sym::T(self.t(s))]],
            E::N(s) => vec![vec![Sym::N(self.nt(s))]],
            E::Seq(v) => {
                let mut acc = vec![vec![]];
                for x in v {
                    let xs = self.expand(x);
                    let mut next = Vec::new();
                    for a in &acc {
                        for b in &xs {
                            let mut c = a.clone();
                            c.extend(b.iter().cloned());
                            next.push(c);
                        }
                    }
                    acc = next;
                }
                acc
            }
            E::Alt(v) => v.iter().flat_map(|x| self.expand(x)).collect(),
            E::Opt(x) => {
                let mut r = vec![vec![]];
                r.extend(self.expand(x));
                r
            }
            E::Rep(x) => {
                let key = format!("{x:?}");
                let h = if let Some(&h) = self.helpers.get(&key) {
                    h
                } else {
                    let name = format!("__rep{}", self.helpers.len());
                    let h = self.nt(&name);
                    self.helpers.insert(key, h);
                    for alt in self.expand(x) {
                        self.g.prods.push((h, alt.clone()));
                        let mut rec = vec![Sym::N(h)];
                        rec.extend(alt);
                        self.g.prods.push((h, rec));
                    }
                    h
                };
                vec![vec![], vec![Sym::N(h)]]
            }
        }
    }
}

/// The section-3 grammar from the text of `book/03-syntax.md`.
pub fn book_grammar(md: &str) -> Result<Grammar, String> {
    let mut blocks = String::new();
    let mut inside = false;
    for line in md.lines() {
        if line.trim_start().starts_with("```") {
            inside = !inside;
            continue;
        }
        if inside {
            blocks.push_str(line);
            blocks.push('\n');
        }
    }
    let toks = lex_ebnf(&blocks);
    let mut p = P { t: toks, i: 0 };
    let mut rules: Vec<(String, E)> = Vec::new();
    while p.i < p.t.len() {
        let name = p.t[p.i].clone();
        if p.t.get(p.i + 1).map(String::as_str) != Some("=") {
            return Err(format!("expected `Name =` at {name:?}"));
        }
        p.i += 2;
        let e = p.alt()?;
        if p.peek() != "." {
            return Err(format!(
                "rule {name}: expected '.' but found {:?}",
                p.peek()
            ));
        }
        p.i += 1;
        if rules.iter().any(|r| r.0 == name) {
            return Err(format!("rule {name} defined twice"));
        }
        rules.push((name, e));
    }
    // An all-capitals name is a terminal (NAME, NEWLINE, ...) unless a rule defines it.
    fn fix(e: &mut E, defined: &BTreeSet<String>) {
        match e {
            E::Up(t) if defined.contains(t.as_str()) => *e = E::N(t.clone()),
            E::Up(t) => *e = E::T(t.clone()),
            E::Seq(v) | E::Alt(v) => v.iter_mut().for_each(|x| fix(x, defined)),
            E::Opt(x) | E::Rep(x) => fix(x, defined),
            _ => {}
        }
    }
    let defined: BTreeSet<String> = rules.iter().map(|r| r.0.clone()).collect();
    for r in &mut rules {
        fix(&mut r.1, &defined);
    }
    let mut x = Expander {
        g: Grammar::default(),
        nt_ix: HashMap::new(),
        t_ix: HashMap::new(),
        helpers: HashMap::new(),
    };
    x.t("$end");
    x.nt("$accept");
    for (n, _) in &rules {
        x.nt(n);
    }
    let start = x.nt(&rules[0].0);
    x.g.prods.push((0, vec![Sym::N(start)]));
    for (n, e) in &rules {
        let lhs = x.nt_ix[n];
        let mut seen = BTreeSet::new();
        for alt in x.expand(e) {
            if seen.insert(alt.clone()) {
                x.g.prods.push((lhs, alt));
            }
        }
    }
    let defined: BTreeSet<usize> = x.g.prods.iter().map(|p| p.0).collect();
    for (i, n) in x.g.nts.iter().enumerate() {
        if !defined.contains(&i) {
            return Err(format!("nonterminal {n} is used but not defined"));
        }
    }
    x.g.source_rules = rules.len();
    Ok(x.g)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Shift(usize),
    Reduce(usize),
    Accept,
}

pub struct Lalr {
    pub g: Grammar,
    pub states: usize,
    action: Vec<HashMap<usize, Act>>,
    goto: Vec<HashMap<usize, usize>>,
    pub conflicts: Vec<String>,
}

type Item = (usize, usize); // (production, dot)

impl Lalr {
    pub fn build(g: Grammar) -> Lalr {
        let nn = g.nts.len();
        let nt_prods: Vec<Vec<usize>> = (0..nn)
            .map(|n| (0..g.prods.len()).filter(|&p| g.prods[p].0 == n).collect())
            .collect();
        // nullable and FIRST
        let mut nullable = vec![false; nn];
        let mut first: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nn];
        loop {
            let mut ch = false;
            for (lhs, rhs) in &g.prods {
                let mut all_null = true;
                for s in rhs {
                    match s {
                        Sym::T(t) => {
                            ch |= first[*lhs].insert(*t);
                            all_null = false;
                        }
                        Sym::N(n) => {
                            let f: Vec<usize> = first[*n].iter().copied().collect();
                            for t in f {
                                ch |= first[*lhs].insert(t);
                            }
                            if !nullable[*n] {
                                all_null = false;
                            }
                        }
                    }
                    if !all_null {
                        break;
                    }
                }
                if all_null && !nullable[*lhs] {
                    nullable[*lhs] = true;
                    ch = true;
                }
            }
            if !ch {
                break;
            }
        }
        let closure0 = |kernel: &BTreeSet<Item>| -> BTreeSet<Item> {
            let mut set = kernel.clone();
            let mut work: Vec<Item> = kernel.iter().copied().collect();
            while let Some((p, d)) = work.pop() {
                if let Some(Sym::N(n)) = g.prods[p].1.get(d) {
                    for &q in &nt_prods[*n] {
                        if set.insert((q, 0)) {
                            work.push((q, 0));
                        }
                    }
                }
            }
            set
        };
        // LR(0) automaton over kernels
        let mut kernels: Vec<BTreeSet<Item>> = vec![[(0usize, 0usize)].into_iter().collect()];
        let mut index: HashMap<BTreeSet<Item>, usize> = HashMap::new();
        index.insert(kernels[0].clone(), 0);
        let mut trans: Vec<BTreeMap<Sym, usize>> = Vec::new();
        let mut s = 0;
        while s < kernels.len() {
            let cl = closure0(&kernels[s]);
            let mut next: BTreeMap<Sym, BTreeSet<Item>> = BTreeMap::new();
            for &(p, d) in &cl {
                if let Some(x) = g.prods[p].1.get(d) {
                    next.entry(x.clone()).or_default().insert((p, d + 1));
                }
            }
            let mut tr = BTreeMap::new();
            for (x, k) in next {
                let id = *index.entry(k.clone()).or_insert_with(|| {
                    kernels.push(k);
                    kernels.len() - 1
                });
                tr.insert(x, id);
            }
            trans.push(tr);
            s += 1;
        }
        let ns = kernels.len();
        // LALR(1) lookaheads by propagation to a fixpoint
        let mut la: Vec<HashMap<Item, BTreeSet<usize>>> = kernels
            .iter()
            .map(|k| k.iter().map(|&i| (i, BTreeSet::new())).collect())
            .collect();
        la[0].get_mut(&(0, 0)).unwrap().insert(0); // $end
        let first_seq = |rest: &[Sym]| -> (BTreeSet<usize>, bool) {
            let mut f = BTreeSet::new();
            for s in rest {
                match s {
                    Sym::T(t) => {
                        f.insert(*t);
                        return (f, false);
                    }
                    Sym::N(n) => {
                        f.extend(first[*n].iter().copied());
                        if !nullable[*n] {
                            return (f, false);
                        }
                    }
                }
            }
            (f, true)
        };
        // Full LR(1)-style closure of a state with lookahead sets.
        let closure1 = |kla: &HashMap<Item, BTreeSet<usize>>| -> HashMap<Item, BTreeSet<usize>> {
            let mut m: HashMap<Item, BTreeSet<usize>> = kla.clone();
            let mut work: Vec<Item> = m.keys().copied().collect();
            while let Some((p, d)) = work.pop() {
                if let Some(Sym::N(n)) = g.prods[p].1.get(d) {
                    let (mut f, null) = first_seq(&g.prods[p].1[d + 1..]);
                    if null {
                        f.extend(m[&(p, d)].iter().copied());
                    }
                    for &q in &nt_prods[*n] {
                        let e = m.entry((q, 0)).or_default();
                        let before = e.len();
                        e.extend(f.iter().copied());
                        if e.len() != before || before == 0 && !work.contains(&(q, 0)) {
                            work.push((q, 0));
                        }
                    }
                }
            }
            m
        };
        let mut closures: Vec<HashMap<Item, BTreeSet<usize>>> = vec![HashMap::new(); ns];
        let mut dirty = vec![true; ns];
        loop {
            let mut any = false;
            for s in 0..ns {
                if !dirty[s] {
                    continue;
                }
                dirty[s] = false;
                let cl = closure1(&la[s]);
                for (&(p, d), l) in &cl {
                    if let Some(x) = g.prods[p].1.get(d) {
                        let t = trans[s][x];
                        let e = la[t].get_mut(&(p, d + 1)).unwrap();
                        let before = e.len();
                        e.extend(l.iter().copied());
                        if e.len() != before {
                            dirty[t] = true;
                            any = true;
                        }
                    }
                }
                closures[s] = cl;
            }
            if !any && dirty.iter().all(|d| !d) {
                break;
            }
        }
        // tables and conflicts
        let mut action: Vec<HashMap<usize, Act>> = vec![HashMap::new(); ns];
        let mut goto: Vec<HashMap<usize, usize>> = vec![HashMap::new(); ns];
        let mut conflicts = Vec::new();
        let show = |p: usize| {
            let (l, r) = &g.prods[p];
            let rhs: Vec<String> = r
                .iter()
                .map(|s| match s {
                    Sym::T(t) => format!("{:?}", g.terms[*t]),
                    Sym::N(n) => g.nts[*n].clone(),
                })
                .collect();
            format!("{} = {}", g.nts[*l], rhs.join(" "))
        };
        for s in 0..ns {
            for (x, &t) in &trans[s] {
                match x {
                    Sym::T(a) => {
                        action[s].insert(*a, Act::Shift(t));
                    }
                    Sym::N(n) => {
                        goto[s].insert(*n, t);
                    }
                }
            }
            let mut items: Vec<(&Item, &BTreeSet<usize>)> = closures[s].iter().collect();
            items.sort();
            for (&(p, d), l) in items {
                if d != g.prods[p].1.len() {
                    continue;
                }
                for &a in l {
                    let new = if p == 0 { Act::Accept } else { Act::Reduce(p) };
                    match action[s].get(&a) {
                        None => {
                            action[s].insert(a, new);
                        }
                        Some(Act::Shift(_)) => conflicts.push(format!(
                            "state {s}: shift/reduce on {:?}: reduce {}",
                            g.terms[a],
                            show(p)
                        )),
                        Some(Act::Reduce(q)) => conflicts.push(format!(
                            "state {s}: reduce/reduce on {:?}: {} | {}",
                            g.terms[a],
                            show(*q),
                            show(p)
                        )),
                        Some(Act::Accept) => {
                            conflicts.push(format!("state {s}: accept/reduce on {:?}", g.terms[a]))
                        }
                    }
                }
            }
        }
        Lalr {
            g,
            states: ns,
            action,
            goto,
            conflicts,
        }
    }

    pub fn term(&self, name: &str) -> Option<usize> {
        self.g.terms.iter().position(|t| t == name)
    }

    /// Run the table on terminal ids (without the final `$end`).
    pub fn accepts_ids(&self, toks: &[usize]) -> bool {
        let mut stack = vec![0usize];
        let mut i = 0;
        loop {
            let a = toks.get(i).copied().unwrap_or(0);
            match self.action[*stack.last().unwrap()].get(&a) {
                Some(Act::Shift(t)) => {
                    stack.push(*t);
                    i += 1;
                }
                Some(Act::Reduce(p)) => {
                    let (lhs, rhs) = &self.g.prods[*p];
                    stack.truncate(stack.len() - rhs.len());
                    match self.goto[*stack.last().unwrap()].get(lhs) {
                        Some(&t) => stack.push(t),
                        None => return false,
                    }
                }
                Some(Act::Accept) => return i >= toks.len(),
                None => return false,
            }
        }
    }

    /// Tokens of the reference lexer (2.1 to 2.6) as grammar terminals.
    /// A keyword is its own terminal except after `.` (a member name, 2.4).
    pub fn tokens(&self, src: &str) -> Option<Vec<usize>> {
        use crate::parse::{KEYWORDS, Tok, lex};
        let toks = lex(src).ok()?;
        let mut out = Vec::new();
        let mut after_dot = false;
        for (t, _) in toks {
            let name: String = match &t {
                Tok::Id(s) if s == "_" => "_".into(),
                Tok::Id(s) if KEYWORDS.contains(&s.as_str()) && !after_dot => s.clone(),
                Tok::Id(_) => "NAME".into(),
                Tok::Int(_) => "INT".into(),
                Tok::Float(_) => "FLOAT".into(),
                Tok::Str(_) => "STRING".into(),
                Tok::P(p) => p.to_string(),
                Tok::Nl => "NEWLINE".into(),
                Tok::Indent => "INDENT".into(),
                Tok::Dedent => "DEDENT".into(),
                Tok::Eof => break,
            };
            after_dot = matches!(&t, Tok::P("."));
            out.push(self.term(&name)?);
        }
        Some(out)
    }

    pub fn accepts(&self, src: &str) -> bool {
        self.tokens(src).is_some_and(|t| self.accepts_ids(&t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ambiguous_grammar_has_conflicts() {
        let md = "```\nS = E NEWLINE .\nE = E \"+\" E | NAME .\n```\n";
        let l = Lalr::build(book_grammar(md).unwrap());
        assert!(!l.conflicts.is_empty());
    }
    #[test]
    fn layered_grammar_is_lalr_and_parses() {
        let md = "```\nS = { E NEWLINE } .\nE = T { \"+\" T } .      sum\nT = NAME | \"(\" E \")\" .   atom\n```\n"; // single capitals are rules here
        let l = Lalr::build(book_grammar(md).unwrap());
        assert!(l.conflicts.is_empty(), "{:?}", l.conflicts);
        let id = |s: &str| l.term(s).unwrap();
        assert!(l.accepts_ids(&[
            id("NAME"),
            id("+"),
            id("("),
            id("NAME"),
            id(")"),
            id("NEWLINE")
        ]));
        assert!(!l.accepts_ids(&[id("NAME"), id("+"), id("NEWLINE")]));
    }
}
