//! A small built-in BM25 (Okapi, k1 = 1.2, b = 0.75). Pure Rust, no
//! dependencies, rebuilt per query from the rows it is given.
use std::collections::HashMap;

/// Lowercase alphanumeric tokens; `snake_case` and `camelCase` are also
/// split into parts so `checked_sub` matches `checked` and `sub`.
pub fn tokenize(s: &str) -> Vec<String> {
    let mut out = vec![];
    for w in s.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if w.is_empty() {
            continue;
        }
        let lw = w.to_lowercase();
        let mut parts = vec![];
        let mut cur = String::new();
        let mut prev_lower = false;
        for c in w.chars() {
            if c == '_' {
                if !cur.is_empty() {
                    parts.push(std::mem::take(&mut cur));
                }
                prev_lower = false;
                continue;
            }
            if c.is_uppercase() && prev_lower && !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
            prev_lower = c.is_lowercase() || c.is_numeric();
            cur.extend(c.to_lowercase());
        }
        if !cur.is_empty() {
            parts.push(cur);
        }
        let joined = lw.replace('_', "");
        if parts.len() > 1 {
            out.push(lw.clone());
            if joined != lw {
                out.push(joined);
            }
            out.extend(parts);
        } else {
            out.push(joined);
        }
    }
    out
}

pub struct Bm25 {
    docs: Vec<(String, HashMap<String, u32>, usize)>,
    df: HashMap<String, u32>,
    avgdl: f64,
}

impl Bm25 {
    pub fn new(docs: impl IntoIterator<Item = (String, String)>) -> Self {
        let mut v = vec![];
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut total = 0usize;
        for (k, text) in docs {
            let toks = tokenize(&text);
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in &toks {
                *tf.entry(t.clone()).or_default() += 1;
            }
            for t in tf.keys() {
                *df.entry(t.clone()).or_default() += 1;
            }
            total += toks.len();
            v.push((k, tf, toks.len()));
        }
        let avgdl = if v.is_empty() {
            0.0
        } else {
            total as f64 / v.len() as f64
        };
        Bm25 { docs: v, df, avgdl }
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Top `k` (key, score) with score > 0, best first, ties by key.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f64)> {
        let q = tokenize(query);
        let n = self.docs.len() as f64;
        let (k1, b) = (1.2, 0.75);
        let mut out: Vec<(String, f64)> = vec![];
        for (key, tf, dl) in &self.docs {
            let mut s = 0.0;
            for t in &q {
                let Some(&f) = tf.get(t) else { continue };
                let df = *self.df.get(t).unwrap_or(&0) as f64;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                let f = f as f64;
                s += idf * f * (k1 + 1.0)
                    / (f + k1 * (1.0 - b + b * *dl as f64 / self.avgdl.max(1e-9)));
            }
            if s > 0.0 {
                out.push((key.clone(), s));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }
}
