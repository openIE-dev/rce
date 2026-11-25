//! Structure-first coverage — the universal construct taxonomy × language matrix.
//!
//! Not a random walk: programming's structure is a closed taxonomy of constructs (the Form IR's
//! expression kinds). Every language is a surface realization of that same structure. Coverage is
//! the (construct × language) matrix; it is filled systematically — each LUT is completed to the
//! full taxonomy in its application style, and per-language syntax is ingested from its grammar
//! (tree-sitter / AST / manual), which is why VS Code already supports hundreds of languages.
//! This module measures that matrix so "100% coverage" is a structural fact, not a claim.

use crate::lut_emit::{builtin_luts, LanguageLut};

/// The universal construct taxonomy (the columns of the coverage matrix).
pub const CONSTRUCTS: &[&str] = &[
    "int_lit", "float_lit", "bool_lit", "str_lit", "var", "neg", "not",
    "add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "gt", "le", "ge",
    "and", "or", "if_expr", "call", "index", "member", "list",
    // Whole-program (statement) layer — the IR-kind expansion beyond expressions.
    "let", "assign", "return", "if_stmt", "while", "for", "import", "type_def", "fn",
];

/// Constructs a LUT covers (present templates ∩ taxonomy).
pub fn covered(lut: &LanguageLut) -> Vec<&'static str> {
    CONSTRUCTS.iter().copied().filter(|c| lut.template(c).is_some()).collect()
}

/// Missing constructs for a LUT (the gaps to fill structurally).
pub fn gaps(lut: &LanguageLut) -> Vec<&'static str> {
    CONSTRUCTS.iter().copied().filter(|c| lut.template(c).is_none()).collect()
}

/// Per-language coverage fraction (0..1).
pub fn coverage(lut: &LanguageLut) -> f64 {
    covered(lut).len() as f64 / CONSTRUCTS.len() as f64
}

/// The coverage matrix: (language, fraction) for every LUT.
pub fn matrix() -> Vec<(String, f64)> {
    builtin_luts().iter().map(|l| (l.name.clone(), coverage(l))).collect()
}

/// Overall (mean) coverage across all languages — the structural completeness figure.
pub fn overall() -> f64 {
    let m = matrix();
    if m.is_empty() { 0.0 } else { m.iter().map(|(_, f)| f).sum::<f64>() / m.len() as f64 }
}