//! Structured Form edits: whole top-level items, addressed by name.
//!
//! An RCE source file is a sequence of top-level items. An item starts on a
//! line with no indentation (outside any open bracket) and runs through the
//! indented lines below it. Its name is the header line up to the first of
//! `(`, ` ->`, ` =`, `:` or ` {`, with spaces collapsed: `state Counter`,
//! `step Counter on Inc`, `fn fib`, `impl Describe for Point`.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub name: String,
    /// Line range, 0-based, end exclusive (trailing blank lines excluded).
    pub start: usize,
    pub end: usize,
}

fn depth_delta(line: &str) -> i64 {
    let (mut d, mut in_str, mut esc) = (0i64, false, false);
    for c in line.chars() {
        if in_str {
            if esc {
                esc = false
            } else if c == '\\' {
                esc = true
            } else if c == '"' {
                in_str = false
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '#' => break,
            '(' | '[' | '{' => d += 1,
            ')' | ']' | '}' => d -= 1,
            _ => {}
        }
    }
    d
}

pub fn item_name(header: &str) -> String {
    let mut cut = header.len();
    for pat in ["(", " ->", " =", ":", " {"] {
        // Do not cut inside a string literal (intent "...").
        if let Some(i) = header.find(pat) {
            let quotes = header[..i].matches('"').count();
            if quotes.is_multiple_of(2) {
                cut = cut.min(i);
            }
        }
    }
    header[..cut]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split a source into its top-level items.
pub fn items(src: &str) -> Vec<Item> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out: Vec<Item> = vec![];
    let mut depth = 0i64;
    for (i, l) in lines.iter().enumerate() {
        let t = l.trim();
        let top = depth == 0 && !t.is_empty() && !t.starts_with('#') && !l.starts_with([' ', '\t']);
        if top {
            out.push(Item {
                name: item_name(t),
                start: i,
                end: i + 1,
            });
        } else if let Some(last) = out.last_mut()
            && !t.is_empty()
        {
            last.end = i + 1;
        }
        depth += depth_delta(l);
    }
    out
}

/// One structured change to a file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Edit {
    /// Replace the item called `name` with `text` (one or more items).
    ReplaceItem { name: String, text: String },
    /// Insert `text` after the item called `after`, or at the end.
    InsertItem {
        #[serde(default)]
        after: Option<String>,
        text: String,
    },
    /// Remove the item called `name`.
    DeleteItem { name: String },
    /// Set the whole file (create a new file, or rewrite one).
    SetFile { text: String },
}

fn find(src: &str, name: &str) -> Result<Item, String> {
    let want = item_name(name);
    let hits: Vec<Item> = items(src).into_iter().filter(|i| i.name == want).collect();
    match hits.len() {
        1 => Ok(hits.into_iter().next().expect("one")),
        0 => Err(format!("no top-level item named `{want}`")),
        n => Err(format!("`{want}` names {n} items; it must name one")),
    }
}

fn splice(src: &str, start: usize, end: usize, text: Option<&str>) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let mut out: Vec<String> = lines[..start].iter().map(|s| s.to_string()).collect();
    if let Some(t) = text {
        out.extend(t.trim_end_matches('\n').lines().map(str::to_string));
    }
    out.extend(lines[end..].iter().map(|s| s.to_string()));
    // Collapse a run of blank lines left by a delete.
    let mut tidy: Vec<String> = vec![];
    for l in out {
        if l.trim().is_empty() && tidy.last().is_some_and(|p| p.trim().is_empty()) {
            continue;
        }
        tidy.push(l);
    }
    while tidy.last().is_some_and(|l| l.trim().is_empty()) {
        tidy.pop();
    }
    let mut s = tidy.join("\n");
    s.push('\n');
    s
}

/// Apply edits in order to `src` (empty for a new file).
pub fn apply(src: &str, edits: &[Edit]) -> Result<String, String> {
    let mut s = src.to_string();
    for (k, e) in edits.iter().enumerate() {
        let at = |r: Result<Item, String>| r.map_err(|m| format!("edit {k}: {m}"));
        s = match e {
            Edit::SetFile { text } => {
                let mut t = text.trim_end_matches('\n').to_string();
                t.push('\n');
                t
            }
            Edit::ReplaceItem { name, text } => {
                let it = at(find(&s, name))?;
                splice(&s, it.start, it.end, Some(text))
            }
            Edit::DeleteItem { name } => {
                let it = at(find(&s, name))?;
                splice(&s, it.start, it.end, None)
            }
            Edit::InsertItem { after, text } => {
                let n = s.lines().count();
                let pos = match after {
                    Some(a) => at(find(&s, a))?.end,
                    None => n,
                };
                let block = format!("\n{}\n", text.trim_matches('\n'));
                splice(&s, pos, pos, Some(&block))
            }
        };
    }
    Ok(s)
}
