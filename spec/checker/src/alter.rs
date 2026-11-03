//! The method *altered* (spec 7.1): the rewrites `rce fmt` writes for the
//! rules whose method is `altered` in the table of 7.1.
//!
//! * W-UNUSED-1: an unread binding `x` is named `_x`; a binding `_x` that is
//!   read is named `x`; an unread `use p.q` is written `use p.q as _q`, which
//!   links the same module (an import's impls stay linked). Names carry no
//!   meaning (T7). The new name is the first of `n`, `n1`, `n2`, ... that is
//!   not already a name in the text.
//! * W-DROP-1: an expression statement whose value is not `()` is written
//!   `let _ = e`, which evaluates `e` and drops its value, as the statement
//!   did.
//!
//! Each rewrite is applied alone and kept only when the static reports of
//! the new text are those of the old text less the report it answers: no
//! report is added and none changes. A rewrite that would add one is not
//! written, and its report stays. No line is added or removed, so the line
//! numbers of every other report hold, except where an arm written on one
//! line becomes a block (W-DROP-1 in `P -> e`, where `let` cannot stand). The rewrite is deterministic, and a
//! second run finds nothing to rewrite (`tests/three_methods.rs`).
//!
//! Each altered rule has a mode in the modes table of 7.1 (`admission::Mode`):
//! the rewrites of the rules in `auto` are written; then the rewrites of the
//! rules in `observe` are worked out on that text as if they were written,
//! each recorded as a canonical diff (`repair::diff`) with its rule, and
//! the text is not changed; a rule that is `off` is not rewritten. Applying
//! the recorded diffs in order to the text gives the text the rewrites of
//! those rules would write after the `auto` ones (`tests/rule_modes.rs`).
use std::collections::HashSet;

/// The rules whose reports `alter` rewrites. `tests/three_methods.rs`
/// checks that this is the set of rules whose method is `altered` in 7.1.
pub const RULES: &[&str] = &["W-DROP-1", "W-UNUSED-1"];

/// One rewrite written: the line (of the text as it was when the rewrite
/// was written), the rule and what was written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Alteration {
    pub line: usize,
    pub rule: &'static str,
    pub what: String,
}

/// One rewrite recorded in observe mode and not written: what `Alteration`
/// says, and the canonical diff (`repair::diff`) from the text as the
/// earlier rewrites left it to the text with this rewrite written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observed {
    pub line: usize,
    pub rule: &'static str,
    pub what: String,
    pub diff: Vec<String>,
}

/// What `alter_modes` did: the text with the `auto` rewrites written, those
/// rewrites, and the `observe` rewrites recorded on that text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Altered {
    pub text: String,
    pub applied: Vec<Alteration>,
    pub observed: Vec<Observed>,
}

/// The static reports of a text (parse, link, check; admission, which
/// evaluates examples, is not run). None when the text does not parse.
fn reports(src: &str) -> Option<Vec<String>> {
    let m = crate::parse::parse(src).ok()?;
    let lk = crate::link::link(&m, &[]);
    let mut c = crate::check::Checker::new();
    c.imported_fns = lk.imported_fns;
    c.imported_types = lk.imported_types;
    for e in lk.errs {
        c.e(e);
    }
    c.check_module(&lk.module, &HashSet::new());
    let mut v = c.errs;
    v.sort();
    v.dedup();
    Some(v)
}

/// Whether `new` answers `target` and adds or changes no other report.
fn answers(old: &[String], new: &[String], target: &str) -> bool {
    new.len() < old.len() && !new.iter().any(|r| r == target) && new.iter().all(|r| old.contains(r))
}

/// An identifier of the text outside strings and comments.
struct Ident {
    start: usize,
    end: usize,
    line: usize,
    /// preceded by `.` that is not part of `..` (a field or method)
    after_dot: bool,
    /// a label: followed by `:` inside `(` or `{` that is not a parameter list
    label: bool,
}

fn is_id_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}
fn is_id(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The identifiers of `src`, with what places them.
fn idents(src: &str) -> Vec<Ident> {
    let b = src.as_bytes();
    let mut out = vec![];
    // open brackets: (char, is a parameter list)
    let mut stack: Vec<(u8, bool)> = vec![];
    // the last two words or punctuation marks, for parameter lists
    let mut last: Vec<String> = vec![];
    let (mut i, mut line) = (0usize, 1usize);
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            line += 1;
            i += 1;
        } else if c == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'"' {
            i += 1;
            while i < b.len() && b[i] != b'"' && b[i] != b'\n' {
                i += if b[i] == b'\\' { 2 } else { 1 };
            }
            i += 1;
            last.push("\"".into());
        } else if is_id_start(c) {
            let s = i;
            while i < b.len() && is_id(b[i]) {
                i += 1;
            }
            let after_dot = s >= 1 && b[s - 1] == b'.' && !(s >= 2 && b[s - 2] == b'.');
            let mut j = i;
            while j < b.len() && b[j] == b' ' {
                j += 1;
            }
            let colon = j < b.len() && b[j] == b':' && b.get(j + 1) != Some(&b':');
            let label = colon && matches!(stack.last(), Some((b'(' | b'{', false)));
            out.push(Ident { start: s, end: i, line, after_dot, label });
            last.push(src[s..i].to_string());
        } else {
            let n = last.len();
            let fn_name = n >= 2 && last[n - 2] == "fn" && last[n - 1].bytes().all(is_id);
            let mut mark = (c as char).to_string();
            match c {
                // `fn (`, `fn name (` and `fn name[T] (` open a parameter list
                b'(' => stack.push((c, last.last().is_some_and(|w| w == "fn" || w == "]fn") || fn_name)),
                // the flag of `[` records the generics of a function
                b'[' => stack.push((c, fn_name)),
                b'{' => stack.push((c, false)),
                b')' | b'}' => {
                    stack.pop();
                }
                b']' if stack.pop().is_some_and(|(o, g)| o == b'[' && g) => mark = "]fn".into(),
                _ => {}
            }
            if !c.is_ascii_whitespace() {
                last.push(mark);
            }
            i += 1;
        }
        if last.len() > 64 {
            last.drain(..32);
        }
    }
    out
}

/// The first of `to`, `to1`, `to2`, ... that is not in `taken`.
fn fresh(to: &str, taken: &HashSet<String>) -> String {
    if !taken.contains(to) {
        return to.to_string();
    }
    (1..).map(|k| format!("{to}{k}")).find(|n| !taken.contains(n)).unwrap()
}

fn line_start(src: &str, line: usize) -> Option<usize> {
    if line == 1 {
        return Some(0);
    }
    src.match_indices('\n').nth(line - 2).map(|(k, _)| k + 1)
}

fn line_text(src: &str, line: usize) -> Option<(usize, &str)> {
    let s = line_start(src, line)?;
    let e = src[s..].find('\n').map(|k| s + k).unwrap_or(src.len());
    Some((s, &src[s..e]))
}

/// The text with `from` named `to` on the given lines, at every place that
/// names a value (not after `.`, not a label).
fn rename(src: &str, from: &str, to: &str, lines: &[usize]) -> String {
    let mut out = String::with_capacity(src.len() + 8);
    let mut at = 0;
    for id in idents(src) {
        if lines.contains(&id.line) && !id.after_dot && !id.label && &src[id.start..id.end] == from {
            out.push_str(&src[at..id.start]);
            out.push_str(to);
            at = id.end;
        }
    }
    out.push_str(&src[at..]);
    out
}

/// The places on a line where an expression statement can start: the first
/// character of the line, and the first character after each `->` (a match
/// arm written on one line).
fn statement_starts(text: &str) -> Vec<usize> {
    let b = text.as_bytes();
    let mut v = vec![];
    let first = text.len() - text.trim_start().len();
    v.push(first);
    let mut in_str = false;
    let mut k = 0;
    while k + 1 < b.len() {
        match b[k] {
            b'\\' if in_str => k += 1,
            b'"' => in_str = !in_str,
            b'#' if !in_str => break,
            b'-' if !in_str && b[k + 1] == b'>' => {
                let mut j = k + 2;
                while j < b.len() && b[j] == b' ' {
                    j += 1;
                }
                if j < b.len() {
                    v.push(j);
                }
            }
            _ => {}
        }
        k += 1;
    }
    v
}

use crate::admission::{Mode, RuleSet, rule_of};

/// A report as it reads after a line is added after line `at`.
fn shift(report: &str, at: usize) -> String {
    let l = crate::cli::line_of(report);
    match report.strip_prefix(&format!("line {l}:")) {
        Some(rest) if l > at => format!("line {}:{rest}", l + 1),
        _ => report.to_string(),
    }
}

/// One round: every report of an altered rule in `src`, each rewrite tried
/// on the text as the earlier ones left it.
fn round(src: &str, rules: &[&str], taken: &HashSet<String>, done: &mut Vec<Alteration>, steps: &mut Vec<(String, String)>) -> String {
    let Some(mut old) = reports(src) else { return src.to_string() };
    let mut text = src.to_string();
    let renames = match crate::parse::parse(src) {
        Ok(m) => crate::v09::unused_renames(&m),
        Err(_) => vec![],
    };
    let targets: Vec<String> = old.iter().filter(|r| rule_of(r).is_some_and(|c| rules.contains(&c))).cloned().collect();
    for target in targets {
        if !old.contains(&target) {
            continue; // answered by an earlier rewrite (a `_x` read on several lines)
        }
        let line = crate::cli::line_of(&target);
        // (new text, what was written, rule, the line after which a line was added)
        let mut tries: Vec<(String, String, &'static str, Option<usize>)> = vec![];
        match rule_of(&target) {
            Some("W-DROP-1") => {
                if let Some((s, t)) = line_text(&text, line) {
                    for k in statement_starts(t) {
                        let mut n = text.clone();
                        n.insert_str(s + k, "let _ = ");
                        tries.push((n, "the expression statement is written `let _ = …`".into(), "W-DROP-1", None));
                    }
                    // a match arm on one line takes an expression, not `let`:
                    // the arm's body becomes a block of one statement
                    let lead = t.len() - t.trim_start().len();
                    for k in statement_starts(t).into_iter().skip(1) {
                        let head = t[..k].trim_end();
                        if let Some(pat) = head.strip_suffix("->") {
                            let mut n = text.clone();
                            let body = format!("{}\n{}let _ = {}", pat.trim_end(), " ".repeat(lead + 4), &t[k..]);
                            n.replace_range(s..s + t.len(), &body);
                            tries.push((n, "the arm's expression statement is written `let _ = …` in a block".into(), "W-DROP-1", Some(line)));
                        }
                    }
                }
            }
            Some("W-UNUSED-1") if target.contains(": W-UNUSED-1 `use ") => {
                if let Some((s, t)) = line_text(&text, line) {
                    let body = t.trim_end();
                    let words: Vec<&str> = body.split_whitespace().collect();
                    let (head, alias) = match words.as_slice() {
                        ["use", p] => (body.to_string(), p.rsplit('.').next().unwrap_or(p).to_string()),
                        ["use", p, "as", a] => (format!("use {p}"), a.to_string()),
                        _ => (String::new(), String::new()),
                    };
                    if !head.is_empty() {
                        let a = fresh(&format!("_{alias}"), taken);
                        let lead = &t[..t.len() - t.trim_start().len()];
                        let mut n = text.clone();
                        n.replace_range(s..s + t.len(), &format!("{lead}{} as {a}", head.trim_start()));
                        tries.push((n, format!("`use` is written with the name `{a}`"), "W-UNUSED-1", None));
                    }
                }
            }
            Some("W-UNUSED-1") => {
                let named = target.split('`').nth(1).unwrap_or("");
                for r in renames.iter().filter(|r| r.from == named && r.lines.contains(&line)) {
                    let to = fresh(&r.to, taken);
                    let n = rename(&text, &r.from, &to, &r.lines);
                    tries.push((n, format!("`{}` is named `{to}`", r.from), "W-UNUSED-1", None));
                }
            }
            _ => {}
        }
        for (n, what, rule, added) in tries {
            let base: Vec<String> = match added {
                Some(at) => old.iter().map(|r| shift(r, at)).collect(),
                None => old.clone(),
            };
            let target = match added {
                Some(at) => shift(&target, at),
                None => target.clone(),
            };
            if let Some(new) = reports(&n)
                && answers(&base, &new, &target)
            {
                done.push(Alteration { line, rule, what });
                steps.push((text.clone(), n.clone()));
                if added.is_some() {
                    // the line numbers of the other reports moved: the next
                    // round reads them again
                    return n;
                }
                text = n;
                old = new;
                break;
            }
        }
    }
    text
}

/// The altered rules in mode `m` under `rules`, in the order of `RULES`.
fn in_mode(rules: &RuleSet, m: Mode) -> Vec<&'static str> {
    RULES.iter().copied().filter(|r| rules.mode(r).unwrap_or(Mode::Observe) == m).collect()
}

/// The reports of a rule in `auto` mode that `alter` left in `src`: those
/// whose rewrite would add or change another report. They require change.
pub fn unaltered(src: &str) -> Vec<String> {
    unaltered_under(RuleSet::in_force(), src)
}

/// `unaltered` under the rule set `rules`.
pub fn unaltered_under(rules: &RuleSet, src: &str) -> Vec<String> {
    let auto = in_mode(rules, Mode::Auto);
    reports(src).unwrap_or_default().into_iter().filter(|r| rule_of(r).is_some_and(|c| auto.contains(&c))).collect()
}

/// The text with every rewrite of an altered rule in `auto` mode written
/// (7.1), and what was written, in order. Layout is `fmt_text`'s; this
/// changes only the places the rewrites name.
pub fn alter(src: &str) -> (String, Vec<Alteration>) {
    let a = alter_modes(RuleSet::in_force(), src);
    (a.text, a.applied)
}

/// The rewrites of the altered rules under the modes of `rules` (7.1): the
/// `auto` ones written, then the `observe` ones recorded on that text.
pub fn alter_modes(rules: &RuleSet, src: &str) -> Altered {
    let (text, applied, _) = rewrite(src, &in_mode(rules, Mode::Auto));
    let (_, seen, steps) = rewrite(&text, &in_mode(rules, Mode::Observe));
    let observed = seen.into_iter().zip(steps).map(|(a, (before, after))| Observed { line: a.line, rule: a.rule, what: a.what, diff: crate::repair::diff(&before, &after) }).collect();
    Altered { text, applied, observed }
}

/// Every rewrite of the rules `rules` written, in rounds: the text, what was
/// written, and the text before and after each rewrite.
fn rewrite(src: &str, rules: &[&str]) -> (String, Vec<Alteration>, Vec<(String, String)>) {
    let (mut done, mut steps) = (vec![], vec![]);
    if rules.is_empty() {
        return (src.to_string(), done, steps);
    }
    let taken: HashSet<String> = idents(src).iter().map(|i| src[i.start..i.end].to_string()).collect();
    let mut text = src.to_string();
    // each round writes at least one rewrite or ends the loop; a round that
    // adds a line ends early, so the bound is the number of reports
    let bound = reports(src).map(|r| r.len()).unwrap_or(0) + 1;
    for _ in 0..bound {
        let n = round(&text, rules, &taken, &mut done, &mut steps);
        if n == text {
            break;
        }
        text = n;
    }
    (text, done, steps)
}

/// The text with a canonical diff of `repair::diff` applied: the lines it
/// removes (`-N text`) replaced by the lines it adds (`+N text`). None when
/// the removed lines are not the text's lines at N.
pub fn apply_diff(src: &str, diff: &[String]) -> Option<String> {
    let mut lines: Vec<&str> = src.lines().collect();
    let parse = |l: &str| -> Option<(char, usize, String)> {
        let sign = l.chars().next()?;
        let rest = &l[1..];
        let (num, text) = rest.trim_start().split_once(' ').map(|(a, b)| (a, b.to_string())).unwrap_or((rest.trim(), String::new()));
        Some((sign, num.parse().ok()?, text))
    };
    let hunk: Vec<(char, usize, String)> = diff.iter().map(|l| parse(l)).collect::<Option<_>>()?;
    let removed: Vec<&(char, usize, String)> = hunk.iter().filter(|h| h.0 == '-').collect();
    let added: Vec<String> = hunk.iter().filter(|h| h.0 == '+').map(|h| h.2.clone()).collect();
    let at = hunk.first()?.1.checked_sub(1)?;
    for (k, r) in removed.iter().enumerate() {
        if r.1 != at + k + 1 || lines.get(at + k).copied() != Some(r.2.as_str()) {
            return None;
        }
    }
    let tail = src.ends_with('\n');
    let added_refs: Vec<&str> = added.iter().map(String::as_str).collect();
    lines.splice(at..at + removed.len(), added_refs);
    let mut out = lines.join("\n");
    if tail {
        out.push('\n');
    }
    Some(out)
}
