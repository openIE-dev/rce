//! DB-as-competence (JetBrains database-in-chat parity): deterministic understanding of SQL as
//! structured data — schema (tables/columns/types) and query (projection/source/filter) — derived
//! by a grammar-table lift, not an LLM guess. The schema/query become nodes in the state graph so
//! database code is admitted, cross-referenced and reusable like any other competence.

use serde_json::{json, Value};

fn upper(s: &str) -> String { s.to_uppercase() }

/// Lift CREATE TABLE statements into {tables:[{name, columns:[{name,ty}]}]}.
pub fn schema(src: &str) -> Value {
    let u = upper(src);
    let mut tables = Vec::new();
    let mut rest = u.as_str();
    let orig = src;
    let mut off = 0usize;
    while let Some(pos) = rest.find("CREATE TABLE") {
        let after = &rest[pos + "CREATE TABLE".len()..];
        let name: String = after.trim_start().chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        // find the column list between the first '(' and its matching ')'
        if let Some(open) = after.find('(') {
            let bytes = after.as_bytes();
            let mut depth = 0; let mut close = None;
            for (i, b) in bytes.iter().enumerate().skip(open) {
                if *b == b'(' { depth += 1; }
                if *b == b')' { depth -= 1; if depth == 0 { close = Some(i); break; } }
            }
            if let Some(c) = close {
                let cols_src = &orig[off + pos + "CREATE TABLE".len() + open + 1 .. off + pos + "CREATE TABLE".len() + c];
                let cols: Vec<Value> = cols_src.split(',').filter_map(|c| {
                    let t = c.trim();
                    if t.is_empty() { return None; }
                    let mut it = t.split_whitespace();
                    let cn = it.next()?.to_string();
                    let ty = it.next().unwrap_or("").to_string();
                    Some(json!({ "name": cn, "ty": ty }))
                }).collect();
                tables.push(json!({ "name": name, "columns": cols }));
            }
        }
        off += pos + "CREATE TABLE".len();
        rest = &rest[pos + "CREATE TABLE".len()..];
    }
    json!({ "tables": tables })
}

/// Lift a SELECT into {projection, source, filter}.
pub fn query(src: &str) -> Value {
    let u = upper(src);
    let proj = between(&u, "SELECT", "FROM").map(|s| s.trim().to_string()).unwrap_or_default();
    let source = between(&u, "FROM", "WHERE").or_else(|| between(&u, "FROM", ";")).map(|s| s.trim().to_string()).unwrap_or_default();
    let filter = between(&u, "WHERE", ";").map(|s| s.trim().to_string()).unwrap_or_default();
    let projection: Vec<String> = proj.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    json!({ "projection": projection, "source": source, "filter": filter })
}

fn between<'a>(s: &'a str, a: &str, b: &str) -> Option<&'a str> {
    let i = s.find(a)? + a.len();
    let j = s[i..].find(b).map(|k| i + k).unwrap_or(s.len());
    Some(&s[i..j])
}
