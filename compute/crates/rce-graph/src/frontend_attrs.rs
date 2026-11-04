//! Extract per-function effect-allow declarations from frontend
//! source via a comment-marker convention.
//!
//! The substrate's compile-time effect gate (`effect_check`) needs
//! per-function allow lists. The architecturally correct way to get
//! them would be a Lux/JMax language attribute like
//! `#[allows("regulatory:financial")]`. Because Lux's parser doesn't
//! yet surface attributes onto `FuncDef`, this module ships the
//! same capability as a *convention*: a comment immediately above a
//! `fn` line whose body matches:
//!
//! ```text
//! # @allows: regulatory:financial, pii:read
//! fn pay(amount: int) -> int:
//!     ...
//! ```
//!
//! Both `#` (Lux/Python-style) and `//` (JMax/Rust-style) line
//! comments are recognised. Multiple `@allows:` lines stack. The
//! comment must be on the line immediately before the `fn`
//! declaration with at most one blank line between (so doc comments
//! and the marker can coexist).
//!
//! When RCE's grammar carries these attributes, this module
//! gets replaced by a parser-side pass that reads the AST instead
//! of re-scanning source. The contract — a `BTreeMap<fn_name,
//! Vec<effect>>` — stays the same so consumers don't churn.

use std::collections::BTreeMap;

use regex::Regex;

/// Scan `src` and return a per-function allow-list extracted from
/// `@allows:` comment markers immediately preceding `fn` lines.
pub fn extract_allows_from_source(src: &str) -> BTreeMap<String, Vec<String>> {
    let fn_re = Regex::new(r"(?m)^\s*(?:pub\s+)?fn\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(").unwrap();
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for m in fn_re.captures_iter(src) {
        let fn_name = m.get(1).unwrap().as_str().to_string();
        let fn_start = m.get(0).unwrap().start();
        let collected = collect_preceding_allows(src, fn_start);
        if !collected.is_empty() {
            out.entry(fn_name).or_default().extend(collected);
        }
    }
    // dedup per function (preserve order)
    for v in out.values_mut() {
        let mut seen = std::collections::HashSet::new();
        v.retain(|e| seen.insert(e.clone()));
    }
    out
}

/// Walk lines backwards from the line containing `fn`, collecting
/// every `@allows:` payload found in `#`/`//` comments. Stops at the
/// first non-blank, non-comment line so the marker is unambiguously
/// attached to the function it's above.
fn collect_preceding_allows(src: &str, fn_start: usize) -> Vec<String> {
    let bytes = src.as_bytes();
    // Walk backwards to the start of the line that contains fn_start.
    let mut line_start = fn_start;
    while line_start > 0 && bytes[line_start - 1] != b'\n' {
        line_start -= 1;
    }
    let mut out: Vec<String> = Vec::new();
    let mut cursor = line_start;
    // Walk lines backwards. cursor is the start of the next line
    // we're about to inspect; subtract 1 to land on its trailing \n.
    while cursor > 0 {
        // Find start of the previous line.
        let mut prev_end = cursor - 1; // points at '\n' that ends prev line
        if prev_end == 0 {
            // edge case: first line of file
            break;
        }
        let mut prev_start = prev_end;
        while prev_start > 0 && bytes[prev_start - 1] != b'\n' {
            prev_start -= 1;
        }
        let line = &src[prev_start..prev_end];
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // blank line — keep walking; markers can have a blank
            // between them and the fn, but only one.
            cursor = prev_start;
            // Bail if the line *before* this blank line isn't an
            // allows marker — single blank line is the limit.
            if let Some(more) = peek_allows(src, prev_start) {
                if !more {
                    break;
                }
            }
            continue;
        }
        if let Some(payload) = parse_allows_marker(trimmed) {
            // Push in source order: prepend so earlier markers come first.
            let mut prepend = payload;
            prepend.extend(out);
            out = prepend;
            cursor = prev_start;
        } else if is_doc_or_plain_comment(trimmed) {
            // Doc comment or unrelated comment — skip past, may
            // have an @allows above it.
            cursor = prev_start;
        } else {
            // Real code (e.g., previous fn's `}` or a let). Stop.
            break;
        }
    }
    out
}

fn peek_allows(src: &str, before: usize) -> Option<bool> {
    if before == 0 {
        return None;
    }
    // line ending at before-1
    let mut prev_end = before - 1;
    if prev_end == 0 {
        return None;
    }
    let bytes = src.as_bytes();
    let mut prev_start = prev_end;
    while prev_start > 0 && bytes[prev_start - 1] != b'\n' {
        prev_start -= 1;
    }
    let line = src[prev_start..prev_end].trim();
    Some(parse_allows_marker(line).is_some())
}

fn is_doc_or_plain_comment(line: &str) -> bool {
    line.starts_with('#') || line.starts_with("//")
}

/// Parse a single line as an `@allows:` marker, returning the list
/// of comma-separated effect strings if it matches.
fn parse_allows_marker(line: &str) -> Option<Vec<String>> {
    let stripped = if let Some(rest) = line.strip_prefix("//") {
        rest
    } else if let Some(rest) = line.strip_prefix('#') {
        rest
    } else {
        return None;
    };
    let stripped = stripped.trim_start();
    let payload = stripped.strip_prefix("@allows:")?.trim();
    if payload.is_empty() {
        return Some(vec![]);
    }
    Some(
        payload
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    // Tests live in tests/frontend_attrs.rs (integration) so unrelated
    // pre-existing test-binary errors don't block them.
}
