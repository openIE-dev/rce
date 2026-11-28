//! External-corpus holdout (the thesis-deciding edge, scaled to code we did not build).
//!
//! The corpus below is VERBATIM third-party Python from Rosetta Code (free/public), fetched
//! 28 Nov 2026. It is not hand-crafted to match the substrate. Two measurements are made:
//!   1. lift coverage: how many real snippets lift to Form and replay over the generative battery;
//!   2. external holdout: admit a canonical library, then issue the real snippets as held-out
//!      requests — same-behavior spellings should hit via the name-free behavior key, novel
//!      behavior should miss and fall to the proposer.
//!
//! The normalizer is a deterministic source-to-source rewrite (no model) of the pure-integer
//! iterative patterns into the brace statement language (`program::lift_program`). Recursive
//! variants with expression-level calls / ternaries remain the documented lift frontier.

use crate::holdout;
use crate::program::{generative_probes, lift_program};

/// (task, variant, verbatim Python source). Source: rosettacode.org, fetched 28 Nov 2026.
pub const CORPUS: &[(&str, &str, &str)] = &[
    (
        "gcd",
        "iterative-euclid",
        "def gcd_iter(u, v):\n    while v:\n        u, v = v, u % v\n    return abs(u)\n",
    ),
    (
        "gcd",
        "recursive-euclid",
        "def gcd(u, v):\n    return gcd(v, u % v) if v else abs(u)\n",
    ),
    (
        "factorial",
        "iterative",
        "def factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result\n",
    ),
    (
        "factorial",
        "recursive",
        "def factorial(n):\n    z=1\n    if n>1:\n        z=n*factorial(n-1)\n    return z\n",
    ),
    (
        "fibonacci",
        "iterative",
        "def fib_iter(n):\n    if n < 2:\n        return n\n    fib_prev = 1\n    fib = 1\n    for _ in range(2, n):\n        fib_prev, fib = fib, fib + fib_prev\n    return fib\n",
    ),
    (
        "fibonacci",
        "recursive",
        "def fib_rec(n):\n    if n < 2:\n        return n\n    return fib_rec(n-1) + fib_rec(n-2)\n",
    ),
];

/// Split on a top-level comma (paren-aware).
fn split_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => { depth += 1; cur.push(c); }
            ')' => { depth -= 1; cur.push(c); }
            ',' if depth == 0 => { out.push(cur.trim().to_string()); cur = String::new(); }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() { out.push(cur.trim().to_string()); }
    out
}

fn is_ident(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Deterministic Python-subset -> brace statement language rewrite (targets `lift_program`).
/// Handles: def/if/while/else, truthy conditions, range-for, tuple assignment (simultaneous,
/// temp-hoisted), augmented assignment, plain assignment, return, and `return abs(x)` hoisted
/// to a let-call. Expression-level calls / ternaries are left as-is (they fail lift = frontier).
pub fn normalize(py: &str) -> String {
    let mut out = String::new();
    let mut stack: Vec<usize> = vec![0];
    for line in py.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') { continue; }
        let indent = line.len() - line.trim_start().len();
        while indent < *stack.last().unwrap() { out.push_str("}\n"); stack.pop(); }

        if let Some(rest) = t.strip_prefix("def ") {
            let head = rest.trim_end_matches(':');
            out.push_str(&format!("fn {head} {{\n"));
            stack.push(indent + 1);
        } else if t == "else:" {
            out.push_str("else {\n");
            stack.push(indent + 1);
        } else if let Some(rest) = t.strip_prefix("if ") {
            let mut cond = rest.trim_end_matches(':').trim().to_string();
            if is_ident(&cond) { cond = format!("{cond} != 0"); }
            out.push_str(&format!("if {cond} {{\n"));
            stack.push(indent + 1);
        } else if let Some(rest) = t.strip_prefix("while ") {
            let mut cond = rest.trim_end_matches(':').trim().to_string();
            if is_ident(&cond) { cond = format!("{cond} != 0"); }
            out.push_str(&format!("while {cond} {{\n"));
            stack.push(indent + 1);
        } else if let Some(rest) = t.strip_prefix("for ") {
            // for X in range(A, B):  /  for X in range(B):
            let body = rest.trim_end_matches(':');
            if let Some((lhs, rhs)) = body.split_once(" in ") {
                let x = lhs.trim();
                let inner = rhs.trim().strip_prefix("range(").and_then(|s| s.strip_suffix(')'));
                if let Some(inner) = inner {
                    let args = split_commas(inner);
                    let (lo, hi) = if args.len() == 2 { (args[0].clone(), args[1].clone()) } else { ("0".to_string(), args[0].clone()) };
                    out.push_str(&format!("for {x} in {lo}..{hi} {{\n"));
                    stack.push(indent + 1);
                    continue;
                }
            }
            out.push_str(&format!("{t};\n"));
        } else if let Some(rest) = t.strip_prefix("return ") {
            let e = rest.trim();
            if let Some(arg) = e.strip_prefix("abs(").and_then(|s| s.strip_suffix(')')) {
                out.push_str(&format!("let __r = abs({arg}); return __r;\n"));
            } else {
                out.push_str(&format!("return {e};\n"));
            }
        } else if let Some((idx, op)) = ["+=", "-=", "*=", "/=", "%="]
            .iter()
            .find_map(|op| t.find(op).map(|i| (i, *op)))
        {
            // augmented assignment: x op= e  ->  let x = x op e
            let n = t[..idx].trim();
            let rhs = t[idx + 2..].trim();
            let o = &op[..1];
            out.push_str(&format!("let {n} = {n} {o} {rhs};\n"));
        } else if let Some((lhs, rhs)) = t.split_once('=') {
            let lhs = lhs.trim();
            let rhs = rhs.trim();
            if lhs.contains(',') && split_commas(lhs).iter().all(|s| is_ident(s)) {
                // simultaneous tuple assignment: hoist RHS into temps first
                let ls = split_commas(lhs);
                let rs = split_commas(rhs);
                for (i, r) in rs.iter().enumerate() { out.push_str(&format!("let __t{i} = {r};\n")); }
                for (i, l) in ls.iter().enumerate() { out.push_str(&format!("let {l} = __t{i};\n")); }
            } else if is_ident(lhs) {
                out.push_str(&format!("let {lhs} = {rhs};\n"));
            } else {
                out.push_str(&format!("{t};\n"));
            }
        } else {
            out.push_str(&format!("{t};\n"));
        }
    }
    while stack.len() > 1 { out.push_str("}\n"); stack.pop(); }
    out
}

/// The function name and arity declared by a Python def.
pub fn def_signature(py: &str) -> Option<(String, usize)> {
    for line in py.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("def ") {
            let head = rest.trim_end_matches(':');
            if let Some((name, params)) = head.split_once('(') {
                let params = params.trim_end_matches(')');
                let arity = if params.trim().is_empty() { 0 } else { split_commas(params).len() };
                return Some((name.trim().to_string(), arity));
            }
        }
    }
    None
}

/// Wrap a normalized snippet with the `abs` builtin and a uniform call harness over free
/// inputs (a, b), so the generative battery exercises the third-party behavior.
pub fn request(py: &str) -> Option<String> {
    let (name, arity) = def_signature(py)?;
    let args = match arity { 0 => "", 1 => "a", _ => "a, b" };
    Some(format!(
        "fn abs(x) {{ if x < 0 {{ return 0 - x; }} return x; }}\n{}\nlet r = {name}({args}); return r;",
        normalize(py)
    ))
}

/// Does a real snippet lift to Form AND replay over a non-empty generative battery?
pub fn lift_ok(py: &str) -> bool {
    let Some(w) = request(py) else { return false };
    match lift_program(&w) {
        Ok(s) => !generative_probes(&s).is_empty(),
        Err(_) => false,
    }
}

/// (lifted, total) over the real corpus — the measured lift frontier.
pub fn lift_coverage() -> (usize, usize) {
    let lifted = CORPUS.iter().filter(|(_, _, src)| lift_ok(src)).count();
    (lifted, CORPUS.len())
}

/// Canonical admitted library (brace Form, uniform harness). Differently spelled from the
/// real corpus but behavior-equal, so a real same-behavior snippet is a genuine paraphrase hit.
pub fn canonical_gcd() -> String {
    "fn abs(x) { if x < 0 { return 0 - x; } return x; }
fn g(u, v) { while v != 0 { let t = u % v; let u = v; let v = t; } let r = abs(u); return r; }
let r = g(a, b); return r;"
        .to_string()
}

pub fn canonical_factorial() -> String {
    "fn f(n) { let r = 1; let i = 2; while i <= n { let r = r * i; let i = i + 1; } return r; }
let r = f(a); return r;"
        .to_string()
}

/// External holdout: admit the canonical library, issue real Rosetta snippets as held-out
/// requests. Returns (hits, total) by name-free behavior key. Novel behavior misses.
pub fn external_holdout() -> (usize, usize) {
    let admitted = [canonical_gcd(), canonical_factorial()];
    let admitted_refs: Vec<&str> = admitted.iter().map(|s| s.as_str()).collect();
    let reqs: Vec<String> = CORPUS
        .iter()
        .filter(|(task, _, _)| *task == "gcd" || *task == "factorial" || *task == "fibonacci")
        .filter(|(_, variant, _)| *variant == "iterative-euclid" || *variant == "iterative")
        .filter_map(|(_, _, src)| request(src))
        .collect();
    let req_refs: Vec<&str> = reqs.iter().map(|s| s.as_str()).collect();
    holdout::hit_rate(&admitted_refs, &req_refs)
}