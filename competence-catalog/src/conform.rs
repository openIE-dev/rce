//! RCE-Conform / RCE-Convert — upkeep and conversion of old code to modernity.
//!
//! The largest practical AI-coding demand today is maintaining and converting legacy code.
//! RCE does it deterministically, not by risky LLM rewrite: **lift** the old source (any surface
//! in the fragment) into the name-free Form, **conform/repair** it to the modern acceptance spec
//! if defective, and **re-emit** to any modern language via the LUT. The gate verifies the
//! conversion preserves (or meets) the behaviour, so a port is checked, not hoped. Because the
//! Form is the universal IR, conversion is lift→conform→emit — the same competence reused across
//! every language, old or new.

use crate::canonical::{emit, normalize};
use crate::lift::lift_expr;
use crate::lut_emit::{builtin_luts, emit_expr};
use crate::repair::{eval_expr, repair_to_value};

/// A verified legacy→modern conversion.
pub struct Conversion {
    /// The original (legacy) source.
    pub original: String,
    /// The canonical Form the legacy code lifts to.
    pub form: String,
    /// Name-free content key (the competence identity, language-independent).
    pub key: String,
    /// The modern target language.
    pub target_lang: String,
    /// The emitted modern source.
    pub modern: String,
    /// Whether a repair was applied to meet the modern acceptance spec.
    pub repaired: bool,
    /// The gate: the converted form reproduces the required behaviour.
    pub verified: bool,
}

/// Convert legacy source to a modern language. If `accept` is given, conform/repair the form to
/// that modern acceptance value; otherwise preserve the original behaviour. The gate verifies.
pub fn convert(old_source: &str, target_lang: &str, accept: Option<i128>) -> Result<Conversion, String> {
    let orig = lift_expr(old_source)?;
    let orig_val = eval_expr(&orig);

    let (form_expr, repaired) = match accept {
        Some(t) if orig_val != Some(t) => match repair_to_value(old_source, t) {
            Some(r) => (lift_expr(&r.repaired)?, true),
            None => (orig.clone(), false),
        },
        _ => (orig.clone(), false),
    };

    let required = accept.or(orig_val);
    let verified = eval_expr(&form_expr) == required && required.is_some();

    let form = emit(&normalize(form_expr));
    let key = rcevc_core::sha256(form.as_bytes()).hex();

    let lut = builtin_luts().into_iter().find(|l| l.name == target_lang)
        .ok_or_else(|| format!("no LUT for {target_lang}"))?;
    let modern = emit_expr(&form, &lut)?;

    Ok(Conversion {
        original: old_source.to_string(),
        form,
        key,
        target_lang: target_lang.to_string(),
        modern,
        repaired,
        verified,
    })
}