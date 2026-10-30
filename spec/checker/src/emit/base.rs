//! Shared state and syntax helpers for the three emitters, plus types and
//! patterns as runtime descriptors.
use super::Lang;
use crate::ast::*;
use std::collections::{HashMap, HashSet};

/// Builtin methods that may write back their receiver (interp `builtin`).
pub(super) const MUTATING: &[&str] =
    &["append", "push", "pop", "clear", "sort", "insert", "remove"];

pub(super) const RT_RS: &str = include_str!("rt.rs");
pub(super) const RT_PY: &str = include_str!("rt.py");
pub(super) const RT_JS: &str = include_str!("rt.js");
pub(super) const RT_SCALA: &str = include_str!("rt.scala");

/// 6.3 N7: the display escape set of `spec/data/display-escape.json`
/// (`interp::display_escaped`) as a declaration of the runtime of `l`, the
/// flat list `lo0, hi0, lo1, hi1, ...` of its ranges, ascending. Every
/// runtime's `rdebug` reads it, so no emitted display reads the Unicode
/// tables of the toolchain that runs the program.
pub(super) fn display_escape_decl(l: Lang) -> String {
    let rs = crate::interp::display_escaped();
    let xs = rs.iter().flat_map(|(a, b)| [format!("{a:#x}"), format!("{b:#x}")]).collect::<Vec<_>>().join(", ");
    match l {
        Lang::Rust => format!("const DISPLAY_ESCAPE: [u32; {}] = [{xs}];", 2 * rs.len()),
        Lang::Python => format!("DISPLAY_ESCAPE = ({xs})"),
        Lang::Js => format!("const DISPLAY_ESCAPE = [{xs}];"),
        Lang::Scala => format!("val DISPLAY_ESCAPE: Array[Int] = Array({xs})"),
    }
}

thread_local! {
    /// set while the Scala emitter runs: string literals use `\uXXXX`
    pub(super) static SCALA_Q: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[derive(Clone, PartialEq)]
pub(super) enum Ctx {
    /// A function body: `return` leaves with (value, copy-out, self).
    Fn {
        muts: Vec<Option<String>>,
        self_mut: bool,
    },
    /// An iterator body, a constant or the driver: `return` ends the iterator.
    Gen,
    /// A step body: `return` keeps the state, as in the reference evaluator.
    Step,
}

#[derive(Clone)]
pub(super) struct G<'a> {
    pub m: &'a Module,
    pub lang: Lang,
    pub fns: HashMap<String, &'a FnDecl>,
    pub methods: HashMap<(String, String), (String, &'a FnDecl)>,
    pub types: HashMap<String, &'a Item>,
    pub variants: HashMap<String, (String, Vec<(String, Ty)>)>,
    pub consts: HashSet<String>,
    pub mut_method_names: HashSet<String>,
    pub scopes: Vec<Vec<String>>,
    pub ctx: Ctx,
    pub out: Vec<String>,
    pub ind: usize,
    pub tmp: usize,
    /// inside a `for .. par` being emitted (enables the reset/collect hooks)
    pub par_hooks: bool,
    /// the next statement is the loop of a `for .. par`, whose tick the
    /// enclosing `for .. par` statement has already taken (one tick, as the
    /// reference's forpar_exec)
    pub forpar_loop: bool,
    pub atomic_depth: usize,
    /// loops opened inside each enclosing task closure (scope / deadline bodies)
    pub cl: Vec<usize>,
    pub loops: usize,
    /// JavaScript: top-level functions for par branches run on workers
    pub hoist: Vec<String>,
    pub hoist_names: Vec<String>,
    /// Above 0 while emitting code the reference evaluator does not evaluate
    /// (a place read only to keep a value's tag): no fuel ticks there.
    pub no_fuel: usize,
}

pub(super) type E = Result<String, String>;

pub(super) fn unsupported<T>(what: &str) -> Result<T, String> {
    Err(format!("not supported by the emitter: {what}"))
}

pub(super) fn has_mut_params(f: &FnDecl) -> bool {
    f.sig.params.iter().any(|p| {
        matches!(
            p,
            Param::Named {
                ty: Ty::Ref(true, _),
                ..
            }
        )
    })
}

pub(super) fn named_params(f: &FnDecl) -> Vec<(String, Ty)> {
    f.sig
        .params
        .iter()
        .filter_map(|p| match p {
            Param::Named { name, ty, .. } => Some((name.clone(), ty.clone())),
            _ => None,
        })
        .collect()
}

pub(super) fn ident(n: &str) -> String {
    n.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn expr_names(e: &Expr, out: &mut HashSet<String>) {
    crate::check::walk_expr(e, &mut |x| match x {
        Expr::Name(n, _) => {
            out.insert(n.clone());
        }
        Expr::SelfRef(_) => {
            out.insert("self".into());
        }
        _ => {}
    });
}

pub(super) fn is_place(e: &Expr) -> bool {
    match e {
        Expr::Name(..) | Expr::SelfRef(_) => true,
        Expr::Borrow(_, x, _)
        | Expr::Member(x, _, _)
        | Expr::TupleMember(x, _, _)
        | Expr::Index(x, _, _) => is_place(x),
        _ => false,
    }
}

/// A string literal that is valid in Rust, Python and JavaScript alike.
pub(super) fn q(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if ((c as u32) < 0x20 || c as u32 == 0x7f) && SCALA_Q.with(|s| s.get()) => {
                o.push_str(&format!("\\u{:04x}", c as u32))
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                o.push_str(&format!("\\x{:02x}", c as u32))
            }
            // Other characters as themselves: the emitted file is UTF-8.
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

impl<'a> G<'a> {
    /// Scala output (rt.scala): JavaScript-shaped control flow, `Vector` for arrays
    pub fn sc(&self) -> bool {
        self.lang == Lang::Scala
    }
    pub fn new(m: &'a Module, lang: Lang) -> Self {
        SCALA_Q.with(|s| s.set(lang == Lang::Scala));
        let mut g = G {
            m,
            lang,
            fns: HashMap::new(),
            methods: HashMap::new(),
            types: HashMap::new(),
            variants: HashMap::new(),
            consts: HashSet::new(),
            mut_method_names: HashSet::new(),
            scopes: vec![],
            ctx: Ctx::Gen,
            out: vec![],
            ind: 0,
            tmp: 0,
            par_hooks: false,
            forpar_loop: false,
            atomic_depth: 0,
            cl: vec![],
            loops: 0,
            hoist: vec![],
            hoist_names: vec![],
            no_fuel: 0,
        };
        for i in &m.items {
            match i {
                Item::Fn(f) => {
                    g.fns.insert(f.sig.name.clone(), f);
                }
                Item::Impl { ty, members, .. } => {
                    let h = crate::check::ty_head(ty);
                    for f in members {
                        let id = format!("m_{}__{}", ident(&h), ident(&f.sig.name));
                        g.methods.insert((h.clone(), f.sig.name.clone()), (id, f));
                        if has_mut_params(f)
                            || matches!(f.sig.params.first(), Some(Param::SelfRef(true)))
                        {
                            g.mut_method_names.insert(f.sig.name.clone());
                        }
                    }
                }
                Item::Struct { name, .. }
                | Item::State { name, .. }
                | Item::Event { name, .. }
                | Item::Error { name, .. }
                | Item::Actor { name, .. }
                | Item::TypeAlias { name, .. } => {
                    g.types.insert(name.clone(), i);
                }
                Item::Kind { name, variants, .. } => {
                    g.types.insert(name.clone(), i);
                    for (v, p) in variants {
                        g.variants.insert(v.clone(), (name.clone(), p.clone()));
                    }
                }
                Item::Const { name, .. } => {
                    g.consts.insert(name.clone());
                }
                _ => {}
            }
        }
        g
    }

    pub fn is_tag(&self, n: &str) -> bool {
        self.variants.contains_key(n)
            || matches!(
                self.types.get(n),
                Some(Item::Event { .. } | Item::Error { .. })
            )
    }
    pub fn local(&self, n: &str) -> bool {
        self.scopes.iter().any(|s| s.iter().any(|x| x == n))
    }
    pub fn declare(&mut self, n: &str) {
        if let Some(s) = self.scopes.last_mut()
            && !s.iter().any(|x| x == n)
        {
            s.push(n.to_string());
        }
    }
    pub fn all_locals(&self) -> Vec<String> {
        let mut v: Vec<String> = vec![];
        for s in &self.scopes {
            for n in s {
                if !v.contains(n) {
                    v.push(n.clone());
                }
            }
        }
        v
    }
    pub fn pat_names(&self, p: &Pat, out: &mut Vec<String>) {
        match p {
            Pat::Path(path) if path.len() == 1 && !self.is_tag(&path[0]) => {
                if !out.contains(&path[0]) {
                    out.push(path[0].clone())
                }
            }
            Pat::Ctor(_, args) => {
                for (_, x) in args {
                    self.pat_names(x, out)
                }
            }
            Pat::Tuple(ps) | Pat::Alt(ps) => {
                for x in ps {
                    self.pat_names(x, out)
                }
            }
            _ => {}
        }
    }
    pub fn fresh(&mut self, p: &str) -> String {
        self.tmp += 1;
        format!("__{p}{}", self.tmp)
    }
    pub fn line(&mut self, s: impl AsRef<str>) {
        let pad = if self.lang == Lang::Python {
            "    "
        } else {
            "  "
        };
        for l in s.as_ref().split('\n') {
            self.out.push(format!("{}{}", pad.repeat(self.ind), l));
        }
    }

    // ----- target syntax -----
    /// A runtime helper's name in the target (camelCase in JavaScript).
    pub fn rt(&self, name: &str) -> String {
        match self.lang {
            Lang::Js => {
                if name.ends_with('_') {
                    return name.to_string();
                }
                let mut o = String::new();
                let mut up = false;
                for c in name.chars() {
                    if c == '_' {
                        up = true;
                    } else if up {
                        o.push(c.to_ascii_uppercase());
                        up = false;
                    } else {
                        o.push(c);
                    }
                }
                o
            }
            Lang::Python if name == "slice" => "slice_".into(),
            _ => name.to_string(),
        }
    }
    /// A call to a fallible runtime helper, as a value expression.
    pub fn h(&self, name: &str, args: &[String]) -> String {
        let a = args.join(", ");
        match self.lang {
            Lang::Rust => format!("{}({a})?", self.rt(name)),
            _ => format!("{}({a})", self.rt(name)),
        }
    }
    /// `fit(v, t)`, by reference to the type in Rust.
    pub fn fit(&self, v: &str, t: &str) -> String {
        match self.lang {
            Lang::Rust => format!("fit({v}, &{t})?"),
            _ => format!("fit({v}, {t})"),
        }
    }
    pub fn vec(&self, xs: &[String]) -> String {
        match self.lang {
            Lang::Rust => format!("vec![{}]", xs.join(", ")),
            Lang::Scala => format!("V({})", xs.join(", ")),
            _ => format!("[{}]", xs.join(", ")),
        }
    }
    pub fn pair(&self, a: &str, b: &str) -> String {
        match self.lang {
            Lang::Js => format!("[{a}, {b}]"),
            Lang::Scala => format!("V({a}, {b})"),
            _ => format!("({a}, {b})"),
        }
    }
    pub fn new_(&self, class: &str, args: &str) -> String {
        match self.lang {
            Lang::Js | Lang::Scala => format!("new {class}({args})"),
            _ => format!("{class}({args})"),
        }
    }
    pub fn unit(&self) -> &'static str {
        if self.lang == Lang::Rust {
            "V::Unit"
        } else {
            "UNIT"
        }
    }
    pub fn none(&self) -> &'static str {
        if matches!(self.lang, Lang::Js | Lang::Scala) {
            "null"
        } else {
            "None"
        }
    }
    pub fn opt_label(&self, l: &Option<String>) -> String {
        match (l, self.lang) {
            (Some(x), Lang::Rust) => format!("Some({})", q(x)),
            (Some(x), _) => q(x),
            (None, _) => self.none().into(),
        }
    }
    pub fn tru(&self) -> &'static str {
        if self.lang == Lang::Python {
            "True"
        } else {
            "true"
        }
    }
    pub fn not(&self, x: &str) -> String {
        if self.lang == Lang::Python {
            format!("not {x}")
        } else {
            format!("!{x}")
        }
    }
    pub fn text(&self, x: &str) -> String {
        match self.lang {
            Lang::Rust => format!("V::Text({}.to_string())", q(x)),
            _ => q(x),
        }
    }
    pub fn rec0(&self, n: &str) -> String {
        match self.lang {
            Lang::Rust => format!("V::Rec({}.to_string(), vec![])", q(n)),
            Lang::Scala => format!("new Rec({}, Vector())", q(n)),
            _ => self.new_("Rec", &format!("{}, []", q(n))),
        }
    }
    pub fn trap_e(&self, m: &str) -> String {
        match self.lang {
            Lang::Rust => format!("trap::<V>({})?", q(m)),
            _ => format!("trap({})", q(m)),
        }
    }
    pub fn trap_s(&self, m: &str) -> String {
        match self.lang {
            Lang::Rust => format!("return Err(Flow::Trap({}.to_string()));", q(m)),
            Lang::Python => format!("trap({})", q(m)),
            Lang::Js => format!("trap({});", q(m)),
            Lang::Scala => format!("trap({})", q(m)),
        }
    }
    pub fn var(&self, n: &str) -> String {
        match self.lang {
            Lang::Rust => format!("l_{}.clone()", ident(n)),
            _ => format!("l_{}", ident(n)),
        }
    }
    pub fn let_(&self, name: &str, e: &str) -> String {
        match self.lang {
            Lang::Rust => format!("let mut {name} = {e};"),
            Lang::Python => format!("{name} = {e}"),
            Lang::Js => format!("let {name} = {e};"),
            Lang::Scala => format!("var {name} : Any = {e}"),
        }
    }
    pub fn set_(&self, name: &str, e: &str) -> String {
        match self.lang {
            Lang::Python | Lang::Scala => format!("{name} = {e}"),
            _ => format!("{name} = {e};"),
        }
    }
    pub fn discard(&self, e: &str) -> String {
        match self.lang {
            Lang::Rust => format!("let _ = {e};"),
            Lang::Python => e.to_string(),
            Lang::Js => format!("{e};"),
            Lang::Scala => format!("discard({e})"),
        }
    }
    pub fn stmt_call(&self, f: &str, a: &str) -> String {
        match self.lang {
            Lang::Rust => format!("{f}({a})?;"),
            Lang::Python => format!("{f}({a})"),
            Lang::Js => format!("{}({a});", self.rt(f)),
            Lang::Scala => format!("{f}({a})"),
        }
    }
    pub fn truthy(&self, e: &str) -> String {
        match self.lang {
            Lang::Rust => format!("truthy(&{e})?"),
            _ => format!("truthy({e})"),
        }
    }
    pub fn open(&mut self, head: &str) {
        match self.lang {
            Lang::Python => self.line(format!("{head}:")),
            _ => self.line(format!("{head} {{")),
        }
        self.ind += 1;
    }
    pub fn close(&mut self) {
        self.pad_py();
        self.ind -= 1;
        if self.lang != Lang::Python {
            self.line("}");
        }
    }
    /// Python needs a statement in an otherwise empty block.
    pub fn pad_py(&mut self) {
        if self.lang == Lang::Python {
            let last = self
                .out
                .last()
                .map(|l| l.trim_end().to_string())
                .unwrap_or_default();
            if last.ends_with(':') {
                self.line("pass");
            }
        }
    }
    pub fn if_head(&self, cond: &str) -> String {
        match self.lang {
            Lang::Js | Lang::Scala => format!("if ({cond})"),
            _ => format!("if {cond}"),
        }
    }
    /// Open `if <value is not none>` on a runtime result `x` (Option in Rust).
    pub fn open_some(&mut self, bind: &str, x: &str) {
        match self.lang {
            Lang::Rust => self.open(&format!("if let Some({bind}) = {x}")),
            Lang::Python => {
                self.line(format!("{bind} = {x}"));
                self.open(&format!("if {bind} is not None"));
            }
            Lang::Js => {
                self.line(format!("const {bind} = {x};"));
                self.open(&format!("if ({bind} !== null && {bind} !== undefined)"));
            }
            Lang::Scala => {
                self.line(format!("val {bind} = {x}"));
                self.open(&format!("if ({bind} != null)"));
            }
        }
    }
    /// A zero-argument closure returning `e`, capturing by value.
    pub fn thunk(&self, e: &str, names: &HashSet<String>) -> String {
        match self.lang {
            Lang::Rust => format!(
                "{{ {}Rc::new(move || -> R<V> {{ Ok({e}) }}) as Th }}",
                self.captures(names)
            ),
            Lang::Python => format!("(lambda: {e})"),
            Lang::Js | Lang::Scala => format!("(() => {e})"),
        }
    }
    pub fn captures(&self, names: &HashSet<String>) -> String {
        let mut s = String::new();
        for n in self.all_locals() {
            if names.contains(&n) {
                let i = ident(&n);
                s.push_str(&format!("let l_{i} = l_{i}.clone(); "));
            }
        }
        s
    }

    // ----- types -----
    fn tdesc(&self, k: &str, a: Vec<String>) -> String {
        match self.lang {
            Lang::Rust => match k {
                "int" => format!("ti({})", a[0]),
                "f64" => "T::F64".into(),
                "f32" => "T::F32".into(),
                "opt" => format!("topt({})", a[0]),
                "list" | "arr" => format!("tlist({})", a[0]),
                "map" => format!("tmap({}, {})", a[0], a[1]),
                "tup" => format!("T::Tup(vec![{}])", a.join(", ")),
                "union" => format!("tunion({})", a[0]),
                "ref" => format!("tref({})", a[0]),
                _ => "T::Any".into(),
            },
            Lang::Scala => {
                if k == "tup" {
                    return format!("V(\"tup\", V({}))", a.join(", "));
                }
                let mut p = vec![format!("\"{k}\"")];
                p.extend(a);
                format!("V({})", p.join(", "))
            }
            Lang::Python | Lang::Js => {
                let (o, c) = if self.lang == Lang::Python {
                    ("(", ",)")
                } else {
                    ("[", "]")
                };
                if k == "tup" {
                    return format!(
                        "{o}\"tup\", [{}]{}",
                        a.join(", "),
                        if self.lang == Lang::Python { ")" } else { "]" }
                    );
                }
                let mut p = vec![format!("\"{k}\"")];
                p.extend(a);
                format!("{o}{}{c}", p.join(", "))
            }
        }
    }
    /// A type as a runtime descriptor, with aliases resolved.
    pub fn ty(&self, t: &Ty) -> String {
        self.ty_d(t, 0)
    }
    fn ty_d(&self, t: &Ty, depth: usize) -> String {
        if depth > 32 {
            return self.tdesc("any", vec![]);
        }
        let d = depth + 1;
        match t {
            Ty::Path(p, a) if p.len() == 1 => {
                let n = p[0].as_str();
                if a.is_empty() && crate::types::INTS.contains(&n) {
                    return self.tdesc("int", vec![q(n)]);
                }
                if n == "f64" {
                    return self.tdesc("f64", vec![]);
                }
                if n == "f32" {
                    return self.tdesc("f32", vec![]);
                }
                if n == "List" && a.len() == 1 {
                    return self.tdesc("list", vec![self.ty_d(&a[0], d)]);
                }
                if n == "Map" && a.len() == 2 {
                    return self.tdesc("map", vec![self.ty_d(&a[0], d), self.ty_d(&a[1], d)]);
                }
                if !self.types.contains_key(n) {
                    // 5.16 indexed types are erased
                    match (n, a.as_slice()) {
                        ("Guarded", [x, _]) | ("Secret", [x]) => return self.ty_d(x, d),
                        _ => {}
                    }
                }
                if let Some(Item::TypeAlias { ty, pred: Some(e), .. }) = self.types.get(n)
                    && let Some(bk) = crate::refine::base_of(ty)
                    && let Ok(pr) = crate::refine::compile(e, bk)
                {
                    let inner = self.ty_d(ty, d);
                    let pd = if self.sc() {
                        crate::refine::desc(&pr, false, true).replace("(\"", "V(\"").replace(",)", ")")
                    } else {
                        crate::refine::desc(&pr, self.lang == Lang::Rust, self.lang == Lang::Python)
                    };
                    return match self.lang {
                        Lang::Scala => format!("V(\"refn\", {inner}, \"{n}\", {pd})"),
                        Lang::Rust => format!("trefn({inner}, \"{n}\", {pd})"),
                        Lang::Python => format!("(\"refn\", {inner}, \"{n}\", {pd})"),
                        Lang::Js => format!("[\"refn\", {inner}, \"{n}\", {pd}]"),
                    };
                }
                if let Some(Item::TypeAlias { ty, .. }) = self.types.get(n) {
                    return self.ty_d(ty, d);
                }
                self.tdesc("any", vec![])
            }
            Ty::Opt(x) => self.tdesc("opt", vec![self.ty_d(x, d)]),
            Ty::Array(x, _) => self.tdesc("arr", vec![self.ty_d(x, d)]),
            Ty::Tuple(ts) => self.tdesc("tup", ts.iter().map(|t| self.ty_d(t, d)).collect()),
            Ty::Union(x, _) => self.tdesc("union", vec![self.ty_d(x, d)]),
            Ty::Ref(_, x) | Ty::Region(x, _) => self.tdesc("ref", vec![self.ty_d(x, d)]),
            _ => self.tdesc("any", vec![]),
        }
    }

    // ----- patterns -----
    fn pdesc(&self, rust: String, k: &str, rest: &[String]) -> String {
        match self.lang {
            Lang::Rust => rust,
            Lang::Python => {
                let mut p = vec![format!("\"{k}\"")];
                p.extend(rest.iter().cloned());
                format!("({},)", p.join(", "))
            }
            Lang::Js => {
                let mut p = vec![format!("\"{k}\"")];
                p.extend(rest.iter().cloned());
                format!("[{}]", p.join(", "))
            }
            Lang::Scala => {
                let mut p = vec![format!("\"{k}\"")];
                p.extend(rest.iter().cloned());
                format!("V({})", p.join(", "))
            }
        }
    }
    /// A pattern as a runtime descriptor (literals and ranges as thunks).
    pub fn pat(&self, p: &Pat) -> E {
        let list = |xs: &[String]| match self.lang {
            Lang::Rust => format!("vec![{}]", xs.join(", ")),
            Lang::Scala => format!("V({})", xs.join(", ")),
            _ => format!("[{}]", xs.join(", ")),
        };

        Ok(match p {
            Pat::Wild => self.pdesc("P::Wild".into(), "wild", &[]),
            Pat::Path(path) => {
                let n = &path[path.len() - 1];
                if path.len() == 1 && !self.is_tag(n) {
                    self.pdesc(format!("P::Bind({})", q(n)), "bind", &[q(n)])
                } else {
                    self.pdesc(format!("P::Tag({})", q(n)), "tag", &[q(n)])
                }
            }
            Pat::Ctor(path, args) => {
                let n = &path[path.len() - 1];
                if n == "some" && path.len() == 1 && !args.is_empty() {
                    let inner = self.pat(&args[0].1)?;
                    return Ok(self.pdesc(format!("P::Some(Box::new({inner}))"), "some", &[inner]));
                }
                let mut parts = vec![];
                for (lb, x) in args {
                    parts.push(self.pair(&self.opt_label(lb), &self.pat(x)?));
                }
                let l = list(&parts);
                self.pdesc(
                    format!("P::Ctor({}, {l})", q(n)),
                    "ctor",
                    &[q(n), l.clone()],
                )
            }
            Pat::Lit(e) => {
                let mut ns = HashSet::new();
                expr_names(e, &mut ns);
                let th = self.thunk(&self.clone().ex(e)?, &ns);
                self.pdesc(format!("P::Lit({th})"), "lit", std::slice::from_ref(&th))
            }
            Pat::Range(a, incl, b) => {
                let mut ns = HashSet::new();
                expr_names(a, &mut ns);
                expr_names(b, &mut ns);
                let ta = self.thunk(&self.clone().ex(a)?, &ns);
                let tb = self.thunk(&self.clone().ex(b)?, &ns);
                let inc = if *incl {
                    self.tru()
                } else if self.lang == Lang::Python {
                    "False"
                } else {
                    "false"
                };
                self.pdesc(
                    format!("P::Range({ta}, {incl}, {tb})"),
                    "range",
                    &[ta.clone(), inc.into(), tb.clone()],
                )
            }
            Pat::Tuple(ps) => {
                let xs = ps
                    .iter()
                    .map(|x| self.pat(x))
                    .collect::<Result<Vec<_>, _>>()?;
                let l = list(&xs);
                self.pdesc(format!("P::Tup({l})"), "tup", std::slice::from_ref(&l))
            }
            Pat::Alt(ps) => {
                let xs = ps
                    .iter()
                    .map(|x| self.pat(x))
                    .collect::<Result<Vec<_>, _>>()?;
                let l = list(&xs);
                self.pdesc(format!("P::Alt({l})"), "alt", std::slice::from_ref(&l))
            }
        })
    }
}
