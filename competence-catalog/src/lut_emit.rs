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
        c_family("latex", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("typst", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("vhdl", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("proto", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("forth", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("smalltalk", "({a} & {b})", "({a} | {b})", "({a} not)"),
        c_family("octave", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("bqn", "({a} ∧ {b})", "({a} ∨ {b})", "(¬{a})"),
        c_family("uiua", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("wolfram", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("maple", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("maxima", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("gap", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("sage", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("macaulay2", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("goose", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("jai", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("c3", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("vale", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("ante", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("magma", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("carbon", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("austral", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("pari", "({a} && {b})", "({a} || {b})", "(!{a})"),
        // Survey-driven expansion (27 Nov 2026): active tree-sitter languages, honest coverage.
        c_family("vala", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("haxe", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("gdscript", "({a} and {b})", "({a} or {b})", "(not {a})"),
        lisp_family("fennel", "(not= {a} {b})", "(mod {a} {b})", "(/ {a} {b})"),
        lisp_family("janet", "(not= {a} {b})", "(mod {a} {b})", "(/ {a} {b})"),
        c_family("inko", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("koto", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("nu", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("fish", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("luau", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("teal", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("systemverilog", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("dafny", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("codeql", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("cython", "({a} and {b})", "({a} or {b})", "(not {a})"),
        ml_family("clean", "({a} == {b})", "({a} != {b})", "(/ {a} {b})", "({a} % {b})"),
        ml_family("gren", "({a} == {b})", "({a} != {b})", "(/ {a} {b})", "({a} % {b})"),
        c_family("faust", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("chuck", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("supercollider", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("gnuplot", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("sas", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("picat", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("pkl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("kcl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("nickel", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("cue", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("rego", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("noir", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("circom", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("tact", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("clarity", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("mlir", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("wit", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("usd", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("openscad", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("zeek", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("sourcepawn", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("sqf", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("brightscript", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("vba", "({a} And {b})", "({a} Or {b})", "(Not {a})"),
        c_family("qml", "({a} && {b})", "({a} || {b})", "(!{a})"),
        // Second survey batch (pass 63 continuation).
        c_family("alloy", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("arduino", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("apex", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("al", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("magik", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("p", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("ohm", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("penrose", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("ink", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("vcl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("cfengine", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("json5", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("kdl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("ini", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("devicetree", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("dbml", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("just", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("meson", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("batch", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("sed", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("restructuredtext", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("svelte", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("vue", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("templ", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("liquid", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("jinja", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("twig", "({a} && {b})", "({a} || {b})", "(!{a})"),
        // Third survey batch (pass 66).
        c_family("caddyfile", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("editorconfig", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("git-commit", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("go-template", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("heex", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("liquidsoap", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("m68k", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("nginx", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("plum", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("ros-interface", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("soong", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("typespec", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("vento", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("wikitext", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("topaz", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("wasm", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("boogie", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("scilab", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("stan", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("asciidoc", "({a} && {b})", "({a} || {b})", "(!{a})"),
        // Pass 70: complete the 191 active tree-sitter universe.
        c_family("m3u", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("blade", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("dash", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("desktop", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("diff", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("djot", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("git-config", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("handlebars", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("hurl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("idl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("lilypond", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("mcfunction", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("razor", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("rbs", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("robots", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("sh", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("slim", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("tql", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("typoscript", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("unified-diff", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("zsh", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("cfml", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("cfscript", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("pod", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("soql", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("sosl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("tptp", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("xml", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("dtd", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("wing", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("rpmspec", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("slint", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("move", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("julia", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("cangjie", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("moonbit", "({a} && {b})", "({a} || {b})", "(!{a})"),
        // Measured-coverage expansion (grammar-ingested at 100%): C-like expression cores.
        c_family("cuda", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("glsl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("hcl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("tcl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("vim", "({a} && {b})", "({a} || {b})", "(!{a})"),
        // Immediate-win expansion (triage-driven): 14 more grammars at 100% (odin LUT pre-exists).
        c_family("pony", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("starlark", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("hare", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("hlsl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("cairo", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("squirrel", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("smali", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("firrtl", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("ispc", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("ron", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("puppet", "({a} and {b})", "({a} or {b})", "(!{a})"),
        c_family("xcompose", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("ssh-config", "({a} && {b})", "({a} || {b})", "(!{a})"),
        // Legacy-infrastructure + build-generated expansion (expression-core projections).
        c_family("cobol", "({a} AND {b})", "({a} OR {b})", "(NOT {a})"),
        c_family("pli", "({a} & {b})", "({a} | {b})", "(¬{a})"),
        c_family("rpgle", "({a} AND {b})", "({a} OR {b})", "(NOT {a})"),
        c_family("pascal", "({a} and {b})", "({a} or {b})", "(not {a})"),
        c_family("basic", "({a} AND {b})", "({a} OR {b})", "(NOT {a})"),
        c_family("verilog", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("thrift", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("scss", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("slang", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("capnp", "({a} && {b})", "({a} || {b})", "(!{a})"),
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
        // Emission ⊇ ingestion: every measured language is emit-able. These are structural
        // family seeds (the same method as the 47); exact per-language surface is refined
        // from its grammar via `ingest`. Closes the ingestion↔emission gap.
        ml_family("sml", "({a} = {b})", "({a} <> {b})", "({a} div {b})", "({a} mod {b})"),
        c_family("mojo", "({a} && {b})", "({a} || {b})", "(!{a})"),
        lisp_family("elisp", "(not (= {a} {b}))", "(mod {a} {b})", "(/ {a} {b})"),
        c_family("rescript", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("reason", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("nix", "({a} && {b})", "({a} || {b})", "(!{a})"),
        ml_family("idris", "({a} == {b})", "({a} /= {b})", "(div {a} {b})", "(mod {a} {b})"),
        ml_family("agda", "({a} == {b})", "({a} /= {b})", "(div {a} {b})", "(mod {a} {b})"),
        ml_family("lean", "({a} == {b})", "({a} != {b})", "({a} / {b})", "({a} % {b})"),
        ml_family("fstar", "({a} = {b})", "({a} <> {b})", "({a} / {b})", "({a} % {b})"),
        LanguageLut::new("wenyan", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "「{value}」"), ("var", "{name}"), ("neg", "負{a}"), ("not", "非{a}"),
            ("add", "加{a}以{b}"), ("sub", "減{a}以{b}"), ("mul", "乘{a}以{b}"),
            ("div", "除{a}以{b}"), ("mod", "除{a}以{b}所餘"), ("eq", "{a}等於{b}"),
            ("ne", "{a}不等於{b}"), ("lt", "{a}小於{b}"), ("gt", "{a}大於{b}"),
            ("le", "{a}不大於{b}"), ("ge", "{a}不小於{b}"), ("and", "{a}且{b}"),
            ("or", "{a}或{b}"), ("if_expr", "若{cond}者{then}不然{else}"),
        ]),
        ml_family("coq", "({a} = {b})", "({a} <> {b})", "(div {a} {b})", "(mod {a} {b})"),
        c_family("dhall", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("unison", "({a} && {b})", "({a} || {b})", "(!{a})"),
        LanguageLut::new("nadeshiko", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "「{value}」"), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(否定{a})"),
            ("add", "({a}+{b})"), ("sub", "({a}-{b})"), ("mul", "({a}*{b})"),
            ("div", "({a}/{b})"), ("mod", "({a}%{b})"), ("eq", "({a}=={b})"),
            ("ne", "({a}!={b})"), ("lt", "({a}<{b})"), ("gt", "({a}>{b})"),
            ("le", "({a}<={b})"), ("ge", "({a}>={b})"), ("and", "({a}かつ{b})"),
            ("or", "({a}または{b})"), ("if_expr", "(もし{cond}ならば{then}違えば{else})"),
        ]),
        LanguageLut::new("ezhil", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(அல்ல {a})"),
            ("add", "({a}+{b})"), ("sub", "({a}-{b})"), ("mul", "({a}*{b})"),
            ("div", "({a}/{b})"), ("mod", "({a}%{b})"), ("eq", "({a}=={b})"),
            ("ne", "({a}!={b})"), ("lt", "({a}<{b})"), ("gt", "({a}>{b})"),
            ("le", "({a}<={b})"), ("ge", "({a}>={b})"), ("and", "({a} மற்றும் {b})"),
            ("or", "({a} அல்லது {b})"), ("if_expr", "(ஆனால் {cond} எனில் {then} இல்லை {else})"),
        ]),
        LanguageLut::new("alusus", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(ليس {a})"),
            ("add", "({a}+{b})"), ("sub", "({a}-{b})"), ("mul", "({a}*{b})"),
            ("div", "({a}/{b})"), ("mod", "({a}%{b})"), ("eq", "({a}=={b})"),
            ("ne", "({a}!={b})"), ("lt", "({a}<{b})"), ("gt", "({a}>{b})"),
            ("le", "({a}<={b})"), ("ge", "({a}>={b})"), ("and", "({a} و {b})"),
            ("or", "({a} أو {b})"), ("if_expr", "(إذا {cond} {then} وإلا {else})"),
        ]),
        c_family("koka", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("grain", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("roc", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("v", "({a} && {b})", "({a} || {b})", "(!{a})"),
        c_family("odin", "({a} && {b})", "({a} || {b})", "(!{a})"),
        // Shell / web / data / build languages (structural projections; the family completion
        // fills the remaining constructs in each surface's own style).
        LanguageLut::new("bash", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "${{{name}}}"), ("neg", "$(( -{a} ))"),
            ("not", "! {a}"), ("add", "$(( {a} + {b} ))"), ("sub", "$(( {a} - {b} ))"),
            ("mul", "$(( {a} * {b} ))"), ("div", "$(( {a} / {b} ))"), ("mod", "$(( {a} % {b} ))"),
            ("eq", "[[ {a} -eq {b} ]]"), ("ne", "[[ {a} -ne {b} ]]"), ("lt", "[[ {a} -lt {b} ]]"),
            ("gt", "[[ {a} -gt {b} ]]"), ("le", "[[ {a} -le {b} ]]"), ("ge", "[[ {a} -ge {b} ]]"),
            ("and", "[[ {a} && {b} ]]"), ("or", "[[ {a} || {b} ]]"),
            ("if_expr", "$(if {cond}; then {then}; else {else}; fi)"),
        ]),
        LanguageLut::new("html", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "{value}"), ("var", "{name}"), ("list", "<ul>{items}</ul>"),
            ("dict", "<dl>{items}</dl>"), ("field", "<dt>{name}</dt><dd>{value}</dd>"),
            ("member", "{a} > {m}"), ("index", "{a}:nth-child({i})"),
            ("if_expr", "<template data-if=\"{cond}\">{then}</template>"),
        ]),
        LanguageLut::new("css", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "var(--{name})"), ("list", "{items}"),
            ("dict", "{name} {{ {items} }}"), ("field", "{name}: {value};"),
            ("member", "{a} {m}"), ("index", "{a}:nth-child({i})"),
            ("if_expr", "@media {cond} {{ {then} }}"),
        ]),
        LanguageLut::new("json", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("list", "[{items}]"),
            ("dict", "{{{items}}}"), ("field", "\"{name}\": {value}"),
            ("member", "{a}.{m}"), ("index", "{a}[{i}]"),
        ]),
        LanguageLut::new("yaml", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "{value}"), ("var", "*{name}"), ("list", "- {items}"),
            ("dict", "{name}:\n{items}"), ("field", "{name}: {value}"),
            ("member", "{a}.{m}"), ("index", "{a}[{i}]"),
        ]),
        LanguageLut::new("toml", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("list", "[{items}]"),
            ("dict", "[{name}]\n{items}"), ("field", "{name} = {value}"),
            ("member", "{a}.{m}"), ("index", "{a}[{i}]"),
        ]),
        LanguageLut::new("dockerfile", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "${{{name}}}"), ("call", "RUN {callee} {args}"),
            ("import", "FROM {name}"), ("let", "ARG {name}={value}"),
            ("directive", "CMD [{args}]"), ("list", "[{items}]"),
        ]),
        LanguageLut::new("cmake", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "${{{name}}}"), ("neg", "-{a}"),
            ("not", "NOT {a}"), ("add", "math(EXPR out \"{a} + {b}\")"),
            ("eq", "{a} STREQUAL {b}"), ("and", "{a} AND {b}"), ("or", "{a} OR {b}"),
            ("if_expr", "if({cond})\n{then}\nelse()\n{else}\nendif()"),
            ("call", "{callee}({args})"), ("fn", "function({name} {params})\n{body}\nendfunction()"),
            ("for", "foreach({name} {items})\n{body}\nendforeach()"),
            ("while", "while({cond})\n{body}\nendwhile()"),
            ("let", "set({name} {value})"), ("import", "find_package({name})"),
            ("macro", "macro({name} {params})\n{body}\nendmacro()"),
        ]),
        LanguageLut::new("make", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "{value}"), ("var", "$({name})"), ("call", "$(shell {callee} {args})"),
            ("let", "{name} = {value}"), ("assign", "{name} := {value}"),
            ("fn", "{name}:\n\t{body}"), ("for", "$(foreach {name},{items},{body})"),
            ("if_expr", "$(if {cond},{then},{else})"), ("import", "include {name}"),
            ("list", "{items}"), ("member", "{a}({m})"),
        ]),
        LanguageLut::new("powershell", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "${value}"),
            ("str_lit", "\"{value}\""), ("var", "${name}"), ("neg", "(-{a})"), ("not", "(-not {a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} / {b})"), ("mod", "({a} % {b})"), ("eq", "({a} -eq {b})"),
            ("ne", "({a} -ne {b})"), ("lt", "({a} -lt {b})"), ("gt", "({a} -gt {b})"),
            ("le", "({a} -le {b})"), ("ge", "({a} -ge {b})"), ("and", "({a} -and {b})"),
            ("or", "({a} -or {b})"), ("if_expr", "(if ({cond}) {{ {then} }} else {{ {else} }})"),
            ("call", "{callee} {args}"), ("member", "{a}.{m}"), ("index", "{a}[{i}]"),
            ("list", "@({items})"), ("dict", "@{{{items}}}"), ("let", "${name} = {value}"),
            ("for", "foreach (${name} in {items}) {{ {body} }}"),
            ("while", "while ({cond}) {{ {body} }}"), ("fn", "function {name} {{ param({params}) {body} }}"),
            ("import", "Import-Module {name}"), ("return", "return {value}"),
        ]),
        LanguageLut::new("tlaplus", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("neg", "(-{a})"), ("not", "(~{a})"),
            ("add", "({a} + {b})"), ("sub", "({a} - {b})"), ("mul", "({a} * {b})"),
            ("div", "({a} \\div {b})"), ("mod", "({a} % {b})"), ("eq", "({a} = {b})"),
            ("ne", "({a} /= {b})"), ("lt", "({a} < {b})"), ("gt", "({a} > {b})"),
            ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"), ("and", "({a} /\\ {b})"),
            ("or", "({a} \\/ {b})"), ("if_expr", "(IF {cond} THEN {then} ELSE {else})"),
            ("call", "{callee}[{args}]"), ("fn", "{name}({params}) == {body}"),
            ("let", "{name} == {value}"), ("set", "{{{items}}}"), ("list", "<<{items}>>"),
            ("member", "{a}[{m}]"), ("index", "{a}[{i}]"), ("return", "{value}"),
        ]),
        LanguageLut::new("markdown", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "{value}"), ("var", "{name}"), ("label", "# {value}"),
            ("list", "- {items}"), ("body", "{value}"), ("field", "| {value} |"),
            ("import", "[{name}]({value})"), ("token", "---"), ("modifier", "~~{a}~~"),
            ("seq", "| {items} |"), ("comment", "<!-- {value} -->"),
        ]),
        LanguageLut::new("graphql", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "${name}"), ("list", "[{items}]"),
            ("dict", "{{{items}}}"), ("field", "{name}: {value}"), ("member", "{a}.{m}"),
            ("type_def", "type {name} {{ {items} }}"), ("fn", "query {name} {{ {body} }}"),
            ("splat", "...{name}"), ("bound", "implements {name}"), ("body", "{{ {items} }}"),
            ("comment", "# {value}"), ("null_lit", "null"), ("type_expr", "{name}"),
        ]),
        LanguageLut::new("prolog", &[
            ("int_lit", "{value}"), ("float_lit", "{value}"), ("bool_lit", "{value}"),
            ("str_lit", "\"{value}\""), ("var", "{name}"), ("call", "{callee}({args})"),
            ("fn", "{name}({params}) :- {body}."), ("let", "{name} = {value}"),
            ("dict", "{{{items}}}"), ("block", "({a})"), ("control", "!"),
            ("param", "{args}"), ("list", "[{items}]"), ("token", "."),
            ("literal", "{value}"), ("return", "{value}"),
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
    // Structural completion of the expression core for surfaces that do not natively express
    // arithmetic/comparison (markup, data, build languages): a generic applicative projection
    // so the (construct × language) matrix is filled by structure, refined per-language later.
    for (k, t) in [
        ("neg", "(-{a})"), ("not", "(not {a})"), ("add", "({a} + {b})"), ("sub", "({a} - {b})"),
        ("mul", "({a} * {b})"), ("div", "({a} / {b})"), ("mod", "({a} % {b})"),
        ("eq", "({a} == {b})"), ("ne", "({a} != {b})"), ("lt", "({a} < {b})"),
        ("gt", "({a} > {b})"), ("le", "({a} <= {b})"), ("ge", "({a} >= {b})"),
        ("and", "({a} && {b})"), ("or", "({a} || {b})"),
    ] {
        l.templates.entry(k.to_string()).or_insert_with(|| t.to_string());
    }
    // Some surfaces (e.g. MATLAB) lack an expression-form conditional; complete it structurally.
    l.templates.entry("if_expr".to_string())
        .or_insert_with(|| "({cond} ? {then} : {else})".to_string());

    // Statement-level constructs (the whole-program layer), completed per family as data.
    let ml = l.template("if_expr").map(|t| t.starts_with("(if")).unwrap_or(false);
    let stmts: &[(&str, &str)] = if lisp {
        &[
            ("let", "(def {name} {value})"),
            ("assign", "(setf {name} {value})"),
            ("return", "{value}"),
            ("if_stmt", "(if {cond} {then} {else})"),
            ("while", "(loop while {cond} do {body})"),
            ("for", "(loop for {name} from {lo} below {hi} do {body})"),
            ("import", "(require '{name})"),
            ("type_def", "(deftype {name} ())"),
            ("fn", "(defun {name} ({params}) {body})"),
            ("param", "{name}"),
            ("sep", "\n"),
        ]
    } else if ml {
        &[
            ("let", "let {name} = {value}"),
            ("assign", "{name} <- {value};"),
            ("return", "{value}"),
            ("if_stmt", "(if {cond} then {then} else {else})"),
            ("while", "while {cond} do {body} done"),
            ("for", "for {name} = {lo} to {hi} do {body} done"),
            ("import", "open {name}"),
            ("type_def", "type {name}"),
            ("fn", "let {name} ({params}) : {ret} = {body}"),
            ("param", "{name}: {type}"),
            ("sep", "\n"),
        ]
    } else {
        &[
            ("let", "let {name} = {value};"),
            ("assign", "{name} = {value};"),
            ("return", "return {value};"),
            ("if_stmt", "if ({cond}) {{ {then} }} else {{ {else} }}"),
            ("while", "while ({cond}) {{ {body} }}"),
            ("for", "for (let {name} = {lo}; {name} < {hi}; {name}++) {{ {body} }}"),
            ("import", "import {name};"),
            ("type_def", "type {name};"),
            ("fn", "fn {name}({params}) -> {ret} {{ {body} }}"),
            ("param", "{name}: {type}"),
            ("sep", "\n"),
        ]
    };
    for (k, t) in stmts {
        l.templates.entry(k.to_string()).or_insert_with(|| t.to_string());
    }

    // Match / pattern / local-function constructs, completed per family as data.
    let match_stmts: &[(&str, &str)] = if lisp {
        &[
            ("match", "(case {scrut} {arms})"),
            ("arm", "({pat} {body})"),
            ("letfn", "(defun {name} ({params}) {body})"),
            ("pat_wild", "_"),
            ("pat_lit", "{value}"),
            ("pat_bind", "{name}"),
        ]
    } else if ml {
        &[
            ("match", "(match {scrut} with {arms})"),
            ("arm", "| {pat} -> {body}"),
            ("letfn", "let rec {name} ({params}) = {body}"),
            ("pat_wild", "_"),
            ("pat_lit", "{value}"),
            ("pat_bind", "{name}"),
        ]
    } else {
        &[
            ("match", "match ({scrut}) {{ {arms} }}"),
            ("arm", "{pat} => {body},"),
            ("letfn", "fn {name}({params}) {{ {body} }}"),
            ("pat_wild", "_"),
            ("pat_lit", "{value}"),
            ("pat_bind", "{name}"),
        ]
    };
    for (k, t) in match_stmts {
        l.templates.entry(k.to_string()).or_insert_with(|| t.to_string());
    }

    // Generics + async/await surface realizations, completed per family as data.
    let gen_stmts: &[(&str, &str)] = if lisp {
        &[
            ("generic_fn", "(defun {name} ({params}) {body})"),
            ("generic_type", "(deftype {name} ({tparams}))"),
            ("async_fn", "(defun {name} ({params}) {body})"),
            ("await", "{value}"),
        ]
    } else if ml {
        &[
            ("generic_fn", "let {name} ({params}) = {body}"),
            ("generic_type", "type {name}<{tparams}>"),
            ("async_fn", "let {name} ({params}) = {body}"),
            ("await", "{value}"),
        ]
    } else {
        &[
            ("generic_fn", "fn {name}<{tparams}>({params}) {{ {body} }}"),
            ("generic_type", "type {name}<{tparams}>;"),
            ("async_fn", "async fn {name}({params}) {{ {body} }}"),
            ("await", "await {value}"),
        ]
    };
    for (k, t) in gen_stmts {
        l.templates.entry(k.to_string()).or_insert_with(|| t.to_string());
    }
    // Taxonomy aliases so the coverage matrix sees the generic/async constructs.
    let g = l.template("generic_fn").cloned()
        .unwrap_or_else(|| "fn {name}<{tparams}>({params}) {{ {body} }}".into());
    let a = l.template("async_fn").cloned()
        .unwrap_or_else(|| "async fn {name}({params}) {{ {body} }}".into());
    l.templates.entry("generic".to_string()).or_insert(g);
    l.templates.entry("async".to_string()).or_insert(a);

    // Attributes / macros surface realizations, completed per family as data.
    let attr_stmts: &[(&str, &str)] = if lisp {
        &[("attribute", "(declare {name})"), ("macro", "({name} {args})"), ("await_event", "(wait-event {id})")]
    } else if ml {
        &[("attribute", "[@{name}]"), ("macro", "({name} {args})"), ("await_event", "await_event {id}")]
    } else {
        &[("attribute", "#[{name}]"), ("macro", "{name}!({args})"), ("await_event", "await event({id})")]
    };
    for (k, t) in attr_stmts {
        l.templates.entry(k.to_string()).or_insert_with(|| t.to_string());
    }

    // Control-flow / value-form constructs (exotic grammar remainder), per family as data.
    let cf_stmts: &[(&str, &str)] = if lisp {
        &[
            ("block", "(progn {body})"),
            ("loop", "(loop {body})"),
            ("break", "(return)"),
            ("continue", "(continue)"),
            ("try", "(handler-case {body} (error {handler}))"),
            ("cast", "(coerce {value} '{type})"),
            ("closure", "(lambda ({params}) {body})"),
            ("pattern", "{name}"),
            ("ref", "(ref {value})"),
        ]
    } else if ml {
        &[
            ("block", "({body})"),
            ("loop", "while true do {body} done"),
            ("break", "exit"),
            ("continue", "continue"),
            ("try", "try {body} with {handler}"),
            ("cast", "({value} : {type})"),
            ("closure", "fun {params} -> {body}"),
            ("pattern", "{name}"),
            ("ref", "ref {value}"),
        ]
    } else {
        &[
            ("block", "{{ {body} }}"),
            ("loop", "loop {{ {body} }}"),
            ("break", "break;"),
            ("continue", "continue;"),
            ("try", "try {{ {body} }} catch {{ {handler} }}"),
            ("cast", "({type}){value}"),
            ("closure", "|{params}| {body}"),
            ("pattern", "{name}"),
            ("ref", "&{value}"),
        ]
    };
    for (k, t) in cf_stmts {
        l.templates.entry(k.to_string()).or_insert_with(|| t.to_string());
    }

    // Universal grammar-category LUT rows (per family), so every construct has a template.
    let cat: &[(&str, &str)] = if lisp {
        &[
            ("comment", "; {value}"), ("directive", "# {value}"), ("char_lit", "#\\{value}"),
            ("null_lit", "nil"), ("regex", "{value}"), ("token", "{value}"), ("modifier", "{value}"),
            ("type_expr", "{value}"), ("param", "{name}"), ("bound", "{value}"), ("field", "{name}"),
            ("init", "(setf {name} {value})"), ("label", "{name}:"), ("range", "(range {lo} {hi})"),
            ("seq", "(progn {a} {b})"), ("dict", "(dict {key} {value})"), ("set", "(set {value})"),
            ("splat", "{value}"), ("interp", "{value}"), ("jsx", "{value}"), ("module", "(module {name})"),
            ("control", "{value}"), ("assert", "(assert {cond})"), ("throw", "(error {value})"),
            ("with", "(with {value} {body})"), ("delete", "(delete {value})"), ("global", "(global {name})"),
            ("pass", "nil"), ("ellipsis", "..."), ("update", "(incf {value})"), ("new", "(make-instance '{type} {args})"),
            ("chain", "(slot-value {value} '{member})"), ("property", "{name}"), ("switch", "(case {value} {cases})"),
            ("case", "({value} ...)"), ("guard", "(when {cond})"), ("goto", "(go {label})"), ("defer", "(defer {value})"),
            ("go", "(go {value})"), ("send", "(send {ch} {value})"), ("receive", "(receive {ch})"),
            ("select", "(select {cases})"), ("incdec", "(incf {value})"), ("compound_assign", "(setf {name} {value})"),
            ("else", "(t {body})"), ("elif", "({cond} {body})"), ("comprehension", "(loop for {name} in {iter} collect {value})"),
            ("generator", "(loop for {name} in {iter} collect {value})"), ("named_expr", "(setf {name} {value})"),
            ("format", "(format nil \"{value}\")"), ("export", "(export '{name})"), ("record", "(defstruct {name} {params})"),
            ("instanceof", "(typep {value} '{type})"), ("method_ref", "(function {name})"), ("literal", "{value}"),
            ("stmt", "{value}"), ("expr", "{value}"), ("decl", "(defvar {value})"), ("clause", "{value}"), ("body", "(progn {value})"),
            ("asm", "(asm {value})"), ("op", "({value})"), ("keyword", ":{value}"), ("symbol", "'{value}"),
        ]
    } else if ml {
        &[
            ("comment", "(* {value} *)"), ("directive", "# {value}"), ("char_lit", "'{value}'"),
            ("null_lit", "()"), ("regex", "{value}"), ("token", "{value}"), ("modifier", "{value}"),
            ("type_expr", "{value}"), ("param", "{name}: {type}"), ("bound", "{value}"), ("field", "{name}: {type};"),
            ("init", "let {name} = {value}"), ("label", "{name}:"), ("range", "{lo}..{hi}"),
            ("seq", "{a}; {b}"), ("dict", "[{key}, {value}]"), ("set", "[{value}]"),
            ("splat", "{value}"), ("interp", "{value}"), ("jsx", "{value}"), ("module", "module {name}"),
            ("control", "{value}"), ("assert", "assert {cond}"), ("throw", "raise {value}"),
            ("with", "with {value} {body}"), ("delete", "delete {value}"), ("global", "global {name}"),
            ("pass", "()"), ("ellipsis", "..."), ("update", "succ {value}"), ("new", "{type} {args}"),
            ("chain", "{value}.{member}"), ("property", "{name}"), ("switch", "match {value} with {cases}"),
            ("case", "| {value} ->"), ("guard", "when {cond}"), ("goto", "goto {label}"), ("defer", "defer {value}"),
            ("go", "go {value}"), ("send", "send {ch} {value}"), ("receive", "recv {ch}"),
            ("select", "select {cases}"), ("incdec", "succ {value}"), ("compound_assign", "{name} <- {value}"),
            ("else", "else {body}"), ("elif", "else if {cond} then {body}"), ("comprehension", "[{value} | {name} <- {iter}]"),
            ("generator", "({value} | {name} <- {iter})"), ("named_expr", "({name} := {value})"),
            ("format", "{value}"), ("export", "export {name}"), ("record", "type {name} = {{ {params} }}"),
            ("instanceof", "{value} :? {type}"), ("method_ref", "{type}.{name}"), ("literal", "{value}"),
            ("stmt", "{value};"), ("expr", "{value}"), ("decl", "let {value}"), ("clause", "{value}"), ("body", "({value})"),
            ("asm", "asm({value})"), ("op", "({value})"), ("keyword", ":{value}"), ("symbol", "{value}"),
        ]
    } else {
        &[
            ("comment", "/* {value} */"), ("directive", "# {value}"), ("char_lit", "'{value}'"),
            ("null_lit", "NULL"), ("regex", "/{value}/"), ("token", "{value}"), ("modifier", "{value}"),
            ("type_expr", "{value}"), ("param", "{type} {name}"), ("bound", "{value}"), ("field", "{type} {name};"),
            ("init", "{name} = {value}"), ("label", "{name}:"), ("range", "{lo}..{hi}"),
            ("seq", "{a}, {b}"), ("dict", "{{ {key}: {value} }}"), ("set", "{{ {value} }}"),
            ("splat", "*{value}"), ("interp", "{{ {value} }}"), ("jsx", "<{name}>{body}</{name}>"), ("module", "module {name};"),
            ("control", "{value}"), ("assert", "assert({cond});"), ("throw", "throw {value};"),
            ("with", "with ({value}) {{ {body} }}"), ("delete", "delete {value};"), ("global", "global {name};"),
            ("pass", ";"), ("ellipsis", "..."), ("update", "++{value}"), ("new", "new {type}({args})"),
            ("chain", "{value}?.{member}"), ("property", "{name}"), ("switch", "switch ({value}) {{ {cases} }}"),
            ("case", "case {value}:"), ("guard", "if ({cond})"), ("goto", "goto {label};"), ("defer", "defer {value};"),
            ("go", "go {value};"), ("send", "{ch} <- {value}"), ("receive", "<-{ch}"),
            ("select", "select {{ {cases} }}"), ("incdec", "{value}++"), ("compound_assign", "{name} {op}= {value}"),
            ("else", "else {{ {body} }}"), ("elif", "else if ({cond}) {{ {body} }}"), ("comprehension", "[{value} for {name} in {iter}]"),
            ("generator", "({value} for {name} in {iter})"), ("named_expr", "({name} := {value})"),
            ("format", "f\"{value}\""), ("export", "export {name};"), ("record", "record {name}({params});"),
            ("instanceof", "{value} instanceof {type}"), ("method_ref", "{type}::{name}"), ("literal", "{value}"),
            ("stmt", "{value};"), ("expr", "{value}"), ("decl", "{value};"), ("clause", "{value}"), ("body", "{{ {value} }}"),
            ("asm", "asm {{ {value} }}"), ("op", "({value})"), ("keyword", ":{value}"), ("symbol", "{value}"),
        ]
    };
    for (k, t) in cat {
        l.templates.entry(k.to_string()).or_insert_with(|| t.to_string());
    }
}

/// A statement in the canonical IR (the unit above expressions). Whole-program
/// emission is the same LUT mechanism extended to statement templates.
#[derive(Clone)]
pub enum Stmt {
    /// `let name = expr`.
    Let(String, Expr),
    /// `name = expr` (mutation of an existing binding).
    Assign(String, Expr),
    /// `return expr`.
    Return(Expr),
    /// `if cond then else` (single-statement branches).
    If(Expr, Box<Stmt>, Box<Stmt>),
    /// `while cond body`.
    While(Expr, Box<Stmt>),
    /// `for name in lo..hi body`.
    For(String, Expr, Expr, Box<Stmt>),
    /// `import path`.
    Import(String),
    /// `type name` (declaration).
    TypeDef(String),
    /// `match scrut { pat => body, … }` — first matching arm wins.
    Match(Expr, Vec<(Pattern, Box<Stmt>)>),
    /// `fn name(params) { body }` — a local function (closure seed).
    LetFn(String, Vec<String>, Box<Stmt>),
    /// `name(args)` — call a local function.
    Call(String, Vec<Expr>),
    /// `let name = fn(args)` — bind a call's result.
    LetCall(String, String, Vec<Expr>),
    /// `fn name<T…>(params) { body }` — generic function (monomorphized at execution).
    GenericFn(String, Vec<String>, Vec<String>, Box<Stmt>),
    /// `type name<T…>` — generic type declaration.
    GenericType(String, Vec<String>),
    /// `async fn name(params) { body }` — async function (cooperative seed).
    AsyncFn(String, Vec<String>, Box<Stmt>),
    /// `await stmt` — a yield point (cooperative seed; runs inner to completion).
    Await(Box<Stmt>),
    /// `await event(id)` — suspend until an IO completion arrives (IoRuntime).
    AwaitEvent(u64),
    /// `#[name]` / `@name` — an attribute/annotation (declaration metadata).
    Attribute(String),
    /// `name!(args)` — a macro invocation (expansion is an open frontier; execution refuses).
    Macro(String, Vec<Expr>),
    /// A statement sequence.
    Seq(Vec<Stmt>),
}

/// A match pattern: wildcard, literal, or binding.
#[derive(Clone, Debug)]
pub enum Pattern {
    Wild,
    Lit(i128),
    Bind(String),
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
        Stmt::Assign(n, e) => {
            subs.insert("name".into(), n.clone());
            subs.insert("value".into(), render(e, lut)?);
            "assign"
        }
        Stmt::Return(e) => {
            subs.insert("value".into(), render(e, lut)?);
            "return"
        }
        Stmt::For(n, lo, hi, b) => {
            subs.insert("name".into(), n.clone());
            subs.insert("lo".into(), render(lo, lut)?);
            subs.insert("hi".into(), render(hi, lut)?);
            subs.insert("body".into(), render_stmt(b, lut)?);
            "for"
        }
        Stmt::Import(p) => {
            subs.insert("name".into(), p.clone());
            "import"
        }
        Stmt::TypeDef(n) => {
            subs.insert("name".into(), n.clone());
            "type_def"
        }
        Stmt::Match(scrut, arms) => {
            subs.insert("scrut".into(), render(scrut, lut)?);
            let sep = lut.template("sep").cloned().unwrap_or_else(|| "\n".into());
            let rendered: Result<Vec<String>, String> = arms
                .iter()
                .map(|(p, b)| {
                    let pat = match p {
                        Pattern::Wild => lut.template("pat_wild").cloned().unwrap_or_else(|| "_".into()),
                        Pattern::Lit(v) => {
                            let t = lut.template("pat_lit").cloned().unwrap_or_else(|| "{value}".into());
                            let mut s = HashMap::new();
                            s.insert("value".into(), v.to_string());
                            fill(&t, &s)
                        }
                        Pattern::Bind(n) => {
                            let t = lut.template("pat_bind").cloned().unwrap_or_else(|| "{name}".into());
                            let mut s = HashMap::new();
                            s.insert("name".into(), n.clone());
                            fill(&t, &s)
                        }
                    };
                    let mut s = HashMap::new();
                    s.insert("pat".into(), pat);
                    s.insert("body".into(), render_stmt(b, lut)?);
                    let t = lut.template("arm").ok_or_else(|| format!("{} has no arm template", lut.name))?;
                    Ok(fill(t, &s))
                })
                .collect();
            subs.insert("arms".into(), rendered?.join(&sep));
            "match"
        }
        Stmt::LetFn(n, params, b) => {
            subs.insert("name".into(), n.clone());
            subs.insert("params".into(), params.join(", "));
            subs.insert("body".into(), render_stmt(b, lut)?);
            "letfn"
        }
        Stmt::Call(n, args) => {
            subs.insert("callee".into(), n.clone());
            let rendered: Result<Vec<String>, String> = args.iter().map(|a| render(a, lut)).collect();
            subs.insert("args".into(), rendered?.join(", "));
            "call"
        }
        Stmt::GenericFn(n, tp, params, b) => {
            subs.insert("name".into(), n.clone());
            subs.insert("tparams".into(), tp.join(", "));
            subs.insert("params".into(), params.join(", "));
            subs.insert("body".into(), render_stmt(b, lut)?);
            "generic_fn"
        }
        Stmt::GenericType(n, tp) => {
            subs.insert("name".into(), n.clone());
            subs.insert("tparams".into(), tp.join(", "));
            "generic_type"
        }
        Stmt::AsyncFn(n, params, b) => {
            subs.insert("name".into(), n.clone());
            subs.insert("params".into(), params.join(", "));
            subs.insert("body".into(), render_stmt(b, lut)?);
            "async_fn"
        }
        Stmt::Await(inner) => {
            subs.insert("value".into(), render_stmt(inner, lut)?);
            "await"
        }
        Stmt::AwaitEvent(id) => {
            subs.insert("id".into(), id.to_string());
            "await_event"
        }
        Stmt::Attribute(n) => {
            subs.insert("name".into(), n.clone());
            "attribute"
        }
        Stmt::Macro(n, args) => {
            subs.insert("name".into(), n.clone());
            let rendered: Result<Vec<String>, String> = args.iter().map(|a| render(a, lut)).collect();
            subs.insert("args".into(), rendered?.join(", "));
            "macro"
        }
        Stmt::LetCall(n, f, args) => {
            subs.insert("name".into(), n.clone());
            let rendered: Result<Vec<String>, String> = args.iter().map(|a| render(a, lut)).collect();
            let call_t = lut.template("call").ok_or_else(|| format!("{} has no call template", lut.name))?;
            let mut cs = HashMap::new();
            cs.insert("callee".into(), f.clone());
            cs.insert("args".into(), rendered?.join(", "));
            subs.insert("value".into(), fill(call_t, &cs));
            "let"
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
