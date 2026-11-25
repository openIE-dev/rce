//! LUT-driven deterministic emission — "all languages" as data, not manual code.
//!
//! The user's reframe: emitting into every language is a **lookup table + a
//! deterministic operation**, not a hand-written emitter per language. This is
//! the generative attribute-grammar model (Tahr's committed `<->` translation
//! rules + representation templates; MLIR's TableGen ODS): the per-language
//! knowledge is a *table* mapping each canonical IR construct to a target syntax
//! template, and one deterministic renderer walks the IR and applies the table.
//! Adding a language is adding a LUT (data); the emitter is generated, not
//! written. No model calls — emission is a pure lookup, the Law of Coding's G0.
//!
//! The canonical source is the Form IR (the checker's ANF-lowered AST). Each
//! target language is a `LanguageLut`: construct → template with `{placeholder}`
//! slots filled recursively from the rendered children. The renderer is total
//! and deterministic: same IR + same LUT → byte-identical output.

use std::collections::HashMap;

use rce_spec_check::ast::Expr;

/// A target language as data: construct key → syntax template.
#[derive(Clone)]
pub struct LanguageLut {
    /// The language name.
    pub name: String,
    /// construct → template (`{a}`, `{b}`, `{value}`, `{name}`, … placeholders).
    pub templates: HashMap<String, String>,
}

impl LanguageLut {
    /// Build a LUT from (construct, template) pairs.
    pub fn new(name: &str, pairs: &[(&str, &str)]) -> Self {
        LanguageLut {
            name: name.to_string(),
            templates: pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        }
    }

    /// The template for a construct key, if this language defines it.
    pub fn template(&self, key: &str) -> Option<&String> {
        self.templates.get(key)
    }
}

/// Fill `{placeholder}` slots in a template from a substitution map.
fn fill(template: &str, subs: &HashMap<String, String>) -> String {
    let mut out = String::new();
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                out.push('{'); // escaped literal brace
            } else {
                let mut key = String::new();
                for c2 in chars.by_ref() {
                    if c2 == '}' {
                        break;
                    }
                    key.push(c2);
                }
                if let Some(v) = subs.get(&key) {
                    out.push_str(v);
                } else {
                    out.push('{');
                    out.push_str(&key);
                    out.push('}');
                }
            }
        } else if c == '}' {
            if chars.peek() == Some(&'}') {
                chars.next();
            }
            out.push('}'); // escaped or lone closing brace
        } else {
            out.push(c);
        }
    }
    out
}

/// The construct key + placeholder substitutions for an IR expression.
fn construct(e: &Expr, lut: &LanguageLut) -> Result<(String, HashMap<String, String>), String> {
    use Expr::*;
    let mut subs = HashMap::new();
    let key = match e {
        Int(n, _) => {
            subs.insert("value".into(), n.to_string());
            "int_lit"
        }
        Float(f, _) => {
            subs.insert("value".into(), format!("{f}"));
            "float_lit"
        }
        Bool(b, _) => {
            subs.insert("value".into(), b.to_string());
            "bool_lit"
        }
        Str(s, _) => {
            subs.insert("value".into(), s.clone());
            "str_lit"
        }
        Name(n, _) => {
            subs.insert("name".into(), n.clone());
            "var"
        }
        Neg(a, _) => {
            subs.insert("a".into(), render(a, lut)?);
            "neg"
        }
        Not(a, _) => {
            subs.insert("a".into(), render(a, lut)?);
            "not"
        }
        Bin(op, a, b, _) => {
            subs.insert("a".into(), render(a, lut)?);
            subs.insert("b".into(), render(b, lut)?);
            match op.as_str() {
                "+" => "add",
                "-" => "sub",
                "*" => "mul",
                "/" => "div",
                "%" => "mod",
                "==" => "eq",
                "!=" => "ne",
                "<" => "lt",
                ">" => "gt",
                "<=" => "le",
                ">=" => "ge",
                "and" => "and",
                "or" => "or",
                other => return Err(format!("no LUT construct for operator {other}")),
            }
        }
        If(c, t, f, _) => {
            subs.insert("cond".into(), render(c, lut)?);
            subs.insert("then".into(), render(t, lut)?);
            subs.insert("else".into(), render(f, lut)?);
            "if_expr"
        }
        Call(c, args, _, _) => {
            subs.insert("callee".into(), render(c, lut)?);
            let rendered: Result<Vec<String>, String> = args.iter().map(|a| render(&a.e, lut)).collect();
            subs.insert("args".into(), rendered?.join(", "));
            "call"
        }
        Index(a, idx, _) => {
            subs.insert("a".into(), render(a, lut)?);
            let rendered: Result<Vec<String>, String> = idx.iter().map(|i| render(i, lut)).collect();
            subs.insert("i".into(), rendered?.join(", "));
            "index"
        }
        Member(a, m, _) => {
            subs.insert("a".into(), render(a, lut)?);
            subs.insert("m".into(), m.clone());
            "member"
        }
        List(items, _) => {
            let rendered: Result<Vec<String>, String> = items.iter().map(|i| render(i, lut)).collect();
            subs.insert("items".into(), rendered?.join(", "));
            "list"
        }
        _ => return Err("construct outside the LUT expression fragment".into()),
    };
    Ok((key.to_string(), subs))
}

/// Deterministically render an IR expression to target source via the LUT.
pub fn render(e: &Expr, lut: &LanguageLut) -> Result<String, String> {
    let (key, subs) = construct(e, lut)?;
    let template = lut.template(&key)
        .ok_or_else(|| format!("language {} has no template for {key}", lut.name))?;
    Ok(fill(template, &subs))
}

/// Parse and render source to a target language LUT.
pub fn emit_expr(src: &str, lut: &LanguageLut) -> Result<String, String> {
    let toks = rce_spec_check::parse::lex(src).map_err(|e| e.to_string())?;
    let mut p = rce_spec_check::parse::Parser::new(toks);
    let e = p.expr().map_err(|e| format!("{e:?}"))?;
    render(&e, lut)
}

/// The built-in language LUTs — each a data table, not hand-written emitter code.
/// Adding a language is adding a row here (or loading a LUT from a file).
pub fn builtin_luts() -> Vec<LanguageLut> {
    // C-family expression syntax is shared; the LUT makes each language data.
    let c_family = |name: &str, and: &str, or: &str, not: &str| {
        LanguageLut::new(name, &[
            ("int_lit", "{value}"),
            ("float_lit", "{value}"),
            ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""),
            ("var", "{name}"),
            ("neg", "(-{a})"),
            ("not", not),
            ("add", "({a} + {b})"),
            ("sub", "({a} - {b})"),
            ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"),
            ("mod", "({a} % {b})"),
            ("eq", "({a} == {b})"),
            ("ne", "({a} != {b})"),
            ("lt", "({a} < {b})"),
            ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"),
            ("ge", "({a} >= {b})"),
            ("and", and),
            ("or", or),
            ("if_expr", "({cond} ? {then} : {else})"),
        ])
    };
    // ML-family: &&/||/not and an if-then-else expression; equality/div/mod vary.
    let ml_family = |name: &str, eq: &str, ne: &str, div: &str, modop: &str| {
        LanguageLut::new(name, &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(not {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", div), ("mod", modop), ("eq", eq), ("ne", ne),
            ("lt", "({a} < {b})"), ("gt", "({a} > {b})"), ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"),
            ("and", "({a} && {b})"), ("or", "({a} || {b})"),
            ("if_expr", "(if {cond} then {then} else {else})"),
        ])
    };
    // Lisp-family: prefix applicative form.
    let lisp_family = |name: &str, ne: &str, modop: &str, div: &str| {
        LanguageLut::new(name, &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(- {a})"), ("not", "(not {a})"),
            ("add", "(+ {a} {b})"), ("sub", "(- {a} {b})"), ("mul", "(* {a} {b})"),
            ("div", div), ("mod", modop), ("eq", "(= {a} {b})"), ("ne", ne),
            ("lt", "(< {a} {b})"), ("gt", "(> {a} {b})"), ("le", "(<= {a} {b})"), ("ge", "(>= {a} {b})"),
            ("and", "(and {a} {b})"), ("or", "(or {a} {b})"),
            ("if_expr", "(if {cond} {then} {else})"),
        ])
    };
    let mut luts = vec![
        c_family("rust", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("go", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("c", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("cpp", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("java", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("csharp", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("js", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("kotlin", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("swift", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("scala", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("zig", "({a} and {b})", "({a} or {b})", "(!{a})"),
        LanguageLut::new("python", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(not {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "({a} % {b})"), ("eq", "({a} == {b})"),
            ("ne", "({a} != {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} and {b})"),
            ("or", "({a} or {b})"), ("if_expr", "({then} if {cond} else {else})"),
        ]),
        LanguageLut::new("ruby", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(!{a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "({a} % {b})"), ("eq", "({a} == {b})"),
            ("ne", "({a} != {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} && {b})"),
            ("or", "({a} || {b})"), ("if_expr", "({cond} ? {then} : {else})"),
        ]),
        LanguageLut::new("haskell", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(not {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "(div {a} {b})"), ("mod", "(mod {a} {b})"), ("eq", "({a} == {b})"),
            ("ne", "({a} /= {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} && {b})"),
            ("or", "({a} || {b})"), ("if_expr", "(if {cond} then {then} else {else})"),
        ]),
        // --- Global breadth: each language is a data row (LUT), not an emitter. ---
        // C-like expression family: industrial, systems, emerging, and non-Western.
        c_family("typescript", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("dart", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("groovy", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("objectivec", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("php", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("perl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("d", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("crystal", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("solidity", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("move", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("julia", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("cangjie", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("moonbit", "({a} && {b})", "({a} || {b})", "(!{a})"),
        // ML family.
        ml_family("ocaml", "({a} = {b})", "({a} <> {b})", "({a} / {b})", "({a} mod {b})"),
        ml_family("fsharp", "({a} = {b})", "({a} <> {b})", "({a} / {b})", "({a} % {b})"),
        ml_family("elm", "({a} == {b})", "({a} /= {b})", "({a} // {b})", "(remainderBy {b} {a})"),
        ml_family("purescript", "({a} == {b})", "({a} /= {b})", "(div {a} {b})", "(mod {a} {b})"),
        // Lisp family.
        lisp_family("clojure", "(not (= {a} {b}))", "(mod {a} {b})", "(quot {a} {b})"),
        lisp_family("scheme", "(not (= {a} {b}))", "(modulo {a} {b})", "(quotient {a} {b})"),
        lisp_family("racket", "(not (= {a} {b}))", "(modulo {a} {b})", "(quotient {a} {b})"),
        lisp_family("commonlisp", "(/= {a} {b})", "(mod {a} {b})", "(floor {a} {b})"),
        // BEAM family.
        LanguageLut::new("elixir", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(!{a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "(rem({a}, {b}))"), ("eq", "({a} == {b})"),
            ("ne", "({a} != {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} && {b})"),
            ("or", "({a} || {b})"), ("if_expr", "(if {cond}, do: {then}, else: {else})"),
        ]),
        LanguageLut::new("erlang", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(not {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "({a} rem {b})"), ("eq", "({a} == {b})"),
            ("ne", "({a} /= {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} =< {b})"), ("ge", "({a} >= {b})"), ("and", "({a} and {b})"),
            ("or", "({a} or {b})"), ("if_expr", "(if {cond} -> {then}; true -> {else} end)"),
        ]),
        LanguageLut::new("gleam", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(!{a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "({a} % {b})"), ("eq", "({a} == {b})"),
            ("ne", "({a} != {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} && {b})"),
            ("or", "({a} || {b})"), ("if_expr", "(case {cond} {{ True -> {then} False -> {else} }})"),
        ]),
        // Scripting / scientific / logic / legacy.
        LanguageLut::new("lua", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(not {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "({a} % {b})"), ("eq", "({a} == {b})"),
            ("ne", "({a} ~= {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} and {b})"),
            ("or", "({a} or {b})"), ("if_expr", "({cond} and {then} or {else})"),
        ]),
        LanguageLut::new("r", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(!{a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "({a} %% {b})"), ("eq", "({a} == {b})"),
            ("ne", "({a} != {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} && {b})"),
            ("or", "({a} || {b})"), ("if_expr", "(ifelse({cond}, {then}, {else}))"),
        ]),
        LanguageLut::new("nim", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(not {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} div {b})"), ("mod", "({a} mod {b})"), ("eq", "({a} == {b})"),
            ("ne", "({a} != {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} and {b})"),
            ("or", "({a} or {b})"), ("if_expr", "(if {cond}: {then} else: {else})"),
        ]),
        LanguageLut::new("ada", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(not {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "({a} mod {b})"), ("eq", "({a} = {b})"),
            ("ne", "({a} /= {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} and {b})"),
            ("or", "({a} or {b})"), ("if_expr", "(if {cond} then {then} else {else})"),
        ]),
        LanguageLut::new("prolog", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(\\+ {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "(mod({a}, {b}))"), ("eq", "({a} =:= {b})"),
            ("ne", "({a} =\\= {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} =< {b})"), ("ge", "({a} >= {b})"), ("and", "({a}, {b})"),
            ("or", "({a}; {b})"), ("if_expr", "({cond} -> {then} ; {else})"),
        ]),
        LanguageLut::new("sql", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "'{value}'"), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(NOT {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "(MOD({a}, {b}))"), ("eq", "({a} = {b})"),
            ("ne", "({a} <> {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} AND {b})"),
            ("or", "({a} OR {b})"), ("if_expr", "(CASE WHEN {cond} THEN {then} ELSE {else} END)"),
        ]),
        LanguageLut::new("matlab", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "'{value}'"), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(~{a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "(mod({a}, {b}))"), ("eq", "({a} == {b})"),
            ("ne", "({a} ~= {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} && {b})"),
            ("or", "({a} || {b})"),
        ]),
        LanguageLut::new("fortran", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(.not. {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "(mod({a}, {b}))"), ("eq", "({a} == {b})"),
            ("ne", "({a} /= {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} .and. {b})"),
            ("or", "({a} .or. {b})"), ("if_expr", "(merge({then}, {else}, {cond}))"),
        ]),
        LanguageLut::new("vb", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(Not {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "({a} Mod {b})"), ("eq", "({a} = {b})"),
            ("ne", "({a} <> {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} And {b})"),
            ("or", "({a} Or {b})"), ("if_expr", "(If({cond}, {then}, {else}))"),
        ]),
    ];
    for l in luts.iter_mut() {
        complete_higher(l);
    }
    luts
}

/// Structurally complete a LUT to the full construct taxonomy: add the higher constructs
/// (call/index/member/list) in the language's application style (applicative vs Lisp), so
/// coverage is the (construct × language) matrix filled by structure, not a random walk.
/// Exact per-language higher-construct syntax is refined from its grammar via `ingest`.
fn complete_higher(l: &mut LanguageLut) {
    let lisp = l.template("and").map(|a| a.starts_with("(and")).unwrap_or(false);
    let (call, index, member, list) = if lisp {
        ("({callee} {args})", "(list-ref {a} {i})", "(slot-value {a} '{m})", "(list {items})")
    } else {
        ("{callee}({args})", "{a}[{i}]", "{a}.{m}", "[{items}]")
    };
    for (k, t) in [("call", call), ("index", index), ("member", member), ("list", list)] {
        l.templates.entry(k.to_string()).or_insert_with(|| t.to_string());
    }
    // Some surfaces (e.g. MATLAB) lack an expression-form conditional; complete it structurally.
    l.templates.entry("if_expr".to_string())
        .or_insert_with(|| "({cond} ? {then} : {else})".to_string());
}

/// A statement in the canonical IR (the unit above expressions). Whole-program
/// emission is the same LUT mechanism extended to statement templates.
pub enum Stmt {
    /// `let name = expr`.
    Let(String, Expr),
    /// `return expr`.
    Return(Expr),
    /// `if cond then else` (single-statement branches).
    If(Expr, Box<Stmt>, Box<Stmt>),
    /// `while cond body`.
    While(Expr, Box<Stmt>),
    /// A statement sequence.
    Seq(Vec<Stmt>),
}

/// A function: name, typed parameters, return type, body.
pub struct Fn {
    /// The function name.
    pub name: String,
    /// Parameters (name, type).
    pub params: Vec<(String, String)>,
    /// The return type.
    pub ret: String,
    /// The body.
    pub body: Stmt,
}

/// Deterministically render a statement via the LUT's statement templates.
pub fn render_stmt(s: &Stmt, lut: &LanguageLut) -> Result<String, String> {
    let mut subs = HashMap::new();
    let key = match s {
        Stmt::Let(n, e) => {
            subs.insert("name".into(), n.clone());
            subs.insert("value".into(), render(e, lut)?);
            "let"
        }
        Stmt::Return(e) => {
            subs.insert("value".into(), render(e, lut)?);
            "return"
        }
        Stmt::If(c, t, f) => {
            subs.insert("cond".into(), render(c, lut)?);
            subs.insert("then".into(), render_stmt(t, lut)?);
            subs.insert("else".into(), render_stmt(f, lut)?);
            "if_stmt"
        }
        Stmt::While(c, b) => {
            subs.insert("cond".into(), render(c, lut)?);
            subs.insert("body".into(), render_stmt(b, lut)?);
            "while"
        }
        Stmt::Seq(ss) => {
            let sep = lut.template("sep").cloned().unwrap_or_else(|| "\n".into());
            let parts: Result<Vec<String>, String> = ss.iter().map(|s| render_stmt(s, lut)).collect();
            return Ok(parts?.join(&sep));
        }
    };
    let template = lut.template(key)
        .ok_or_else(|| format!("language {} has no statement template for {key}", lut.name))?;
    Ok(fill(template, &subs))
}

/// Deterministically render a function via the LUT's `fn`/`param` templates.
pub fn render_fn(f: &Fn, lut: &LanguageLut) -> Result<String, String> {
    let param_t = lut.template("param").cloned().unwrap_or_else(|| "{name}: {type}".into());
    let params = f.params.iter()
        .map(|(n, t)| {
            let mut s = HashMap::new();
            s.insert("name".into(), n.clone());
            s.insert("type".into(), t.clone());
            fill(&param_t, &s)
        })
        .collect::<Vec<_>>().join(", ");
    let body = render_stmt(&f.body, lut)?;
    let mut subs = HashMap::new();
    subs.insert("name".into(), f.name.clone());
    subs.insert("params".into(), params);
    subs.insert("ret".into(), f.ret.clone());
    subs.insert("body".into(), body);
    let fn_t = lut.template("fn")
        .ok_or_else(|| format!("language {} has no fn template", lut.name))?;
    Ok(fill(fn_t, &subs))
}

/// Statement-level LUTs for representative languages — whole-program emission as
/// data. Each adds statement/function templates to the expression LUT.
pub fn stmt_luts() -> Vec<LanguageLut> {
    #[allow(clippy::too_many_arguments)]
    let with_stmts = |base: &str, let_t: &str, ret_t: &str, if_t: &str, while_t: &str,
                      fn_t: &str, param_t: &str| -> LanguageLut {
        let mut lut = builtin_luts().into_iter().find(|l| l.name == base).unwrap();
        lut.templates.insert("let".into(), let_t.into());
        lut.templates.insert("return".into(), ret_t.into());
        lut.templates.insert("if_stmt".into(), if_t.into());
        lut.templates.insert("while".into(), while_t.into());
        lut.templates.insert("fn".into(), fn_t.into());
        lut.templates.insert("param".into(), param_t.into());
        lut.templates.insert("sep".into(), "\n".into());
        lut
    };
    vec![
        with_stmts("python", "{name} = {value}", "return {value}",
            "if {cond}:\n    {then}\nelse:\n    {else}",
            "while {cond}:\n    {body}",
            "def {name}({params}):\n    {body}", "{name}"),
        with_stmts("rust", "let {name} = {value};", "return {value};",
            "if {cond} {{ {then} }} else {{ {else} }}",
            "while {cond} {{ {body} }}",
            "fn {name}({params}) -> {ret} {{\n    {body}\n}}", "{name}: {type}"),
        with_stmts("go", "{name} := {value}", "return {value}",
            "if {cond} {{ {then} }} else {{ {else} }}",
            "for {cond} {{ {body} }}",
            "func {name}({params}) {ret} {{\n    {body}\n}}", "{name} {type}"),
        with_stmts("js", "let {name} = {value};", "return {value};",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "function {name}({params}) {{\n    {body}\n}}", "{name}"),
        with_stmts("c", "{name} = {value};", "return {value};",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "{ret} {name}({params}) {{\n    {body}\n}}", "{type} {name}"),
        // Whole-program breadth: statement/function templates as data, per family.
        with_stmts("java", "var {name} = {value};", "return {value};",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "{ret} {name}({params}) {{\n    {body}\n}}", "{type} {name}"),
        with_stmts("csharp", "var {name} = {value};", "return {value};",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "{ret} {name}({params}) {{\n    {body}\n}}", "{type} {name}"),
        with_stmts("cpp", "auto {name} = {value};", "return {value};",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "{ret} {name}({params}) {{\n    {body}\n}}", "{type} {name}"),
        with_stmts("kotlin", "val {name} = {value}", "return {value}",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "fun {name}({params}): {ret} {{\n    {body}\n}}", "{name}: {type}"),
        with_stmts("swift", "let {name} = {value}", "return {value}",
            "if {cond} {{ {then} }} else {{ {else} }}",
            "while {cond} {{ {body} }}",
            "func {name}({params}) -> {ret} {{\n    {body}\n}}", "{name}: {type}"),
        with_stmts("scala", "val {name} = {value}", "return {value}",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "def {name}({params}): {ret} = {{\n    {body}\n}}", "{name}: {type}"),
        with_stmts("typescript", "let {name} = {value};", "return {value};",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "function {name}({params}) {{\n    {body}\n}}", "{name}"),
        with_stmts("dart", "var {name} = {value};", "return {value};",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "{ret} {name}({params}) {{\n    {body}\n}}", "{type} {name}"),
        with_stmts("groovy", "def {name} = {value}", "return {value}",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "{ret} {name}({params}) {{\n    {body}\n}}", "{type} {name}"),
        with_stmts("php", "${name} = {value};", "return {value};",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "function {name}({params}) {{\n    {body}\n}}", "{name}"),
        with_stmts("perl", "my ${name} = {value};", "return {value};",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "sub {name} {{\n    {body}\n}}", "{name}"),
        with_stmts("ruby", "{name} = {value}", "return {value}",
            "if {cond}\n    {then}\nelse\n    {else}\nend",
            "while {cond}\n    {body}\nend",
            "def {name}({params})\n    {body}\nend", "{name}"),
        with_stmts("lua", "local {name} = {value}", "return {value}",
            "if {cond} then\n    {then}\nelse\n    {else}\nend",
            "while {cond} do\n    {body}\nend",
            "function {name}({params})\n    {body}\nend", "{name}"),
        with_stmts("julia", "{name} = {value}", "return {value}",
            "if {cond}\n    {then}\nelse\n    {else}\nend",
            "while {cond}\n    {body}\nend",
            "function {name}({params})\n    {body}\nend", "{name}"),
        with_stmts("cangjie", "let {name} = {value}", "return {value}",
            "if ({cond}) {{ {then} }} else {{ {else} }}",
            "while ({cond}) {{ {body} }}",
            "func {name}({params}): {ret} {{\n    {body}\n}}", "{name}: {type}"),
        with_stmts("moonbit", "let {name} = {value}", "return {value}",
            "if {cond} {{ {then} }} else {{ {else} }}",
            "while {cond} {{ {body} }}",
            "fn {name}({params}) -> {ret} {{\n    {body}\n}}", "{name}: {type}"),
        with_stmts("nim", "let {name} = {value}", "return {value}",
            "if {cond}:\n    {then}\nelse:\n    {else}",
            "while {cond}:\n    {body}",
            "proc {name}({params}): {ret} =\n    {body}", "{name}: {type}"),
        with_stmts("vb", "Dim {name} = {value}", "Return {value}",
            "If {cond} Then\n    {then}\nElse\n    {else}\nEnd If",
            "While {cond}\n    {body}\nEnd While",
            "Function {name}({params}) As {ret}\n    {body}\nEnd Function", "{name}"),
    ]
}
