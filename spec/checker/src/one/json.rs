//! The JSON form of the core (one-emitter.md, section 3.4).
//!
//! ```text
//! module = {"format":"rce-core/1","globals":[[τ,LIT|null]…],"defs":{KEY:{"fns":[fn…]}},
//!           "entry":[KEY,i],"names": the concept record, never hashed}
//! fn     = {"params":[τ…],"ret":τ,"body":[stmt…]}
//! τ      = ["bool"]|["int","lo","hi"]|["f64"]|["f32"]|["bytes"]|["arr",τ]|["list",τ]|["tup",τ…]|["sum",τ…]
//! atom   = ["v",i]|["g",i]|["k",τ,LIT]
//! stmt   = ["let",atom] | ["prim",NAME,[ι…],atom…] | ["call",F,atom…] | ["catch",F,atom…]
//!        | ["var",τ,atom] | ["set",["v",i]|["g",i],atom] | ["if",atom,[stmt…],[stmt…]]
//!        | ["block",[stmt…]] | ["loop",[stmt…]] | ["break",d] | ["ret",atom] | ["halt",atom] | ["out",atom]
//! F      = [KEY,i] | ["rec",i]
//! ```
//!
//! A literal is `true` or `false`, an integer as decimal text, an f64 or f32
//! as the 16 hex digits of the binary64 bits, and bytes as hex. The text is
//! canonical JSON (6.1.1): object keys in sorted order, no white space, and
//! text escaped as the log writes it. A group's key is the SHA-256 of the
//! canonical JSON of `{"fns":[fn…]}` (W5).
use super::ir::*;
use serde_json::{Value as J, json};
use std::collections::BTreeMap;

/// Text escaped as 6.1.1 writes it: `"`, `\`, LF, CR and TAB by their short
/// escapes, every other code point below U+0020 as `\u00xx`.
pub fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Canonical JSON of a value whose numbers are all integers.
pub fn canon(v: &J) -> String {
    let mut s = String::new();
    write(v, &mut s);
    s
}

fn write(v: &J, o: &mut String) {
    match v {
        J::Null => o.push_str("null"),
        J::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
        J::Number(n) => o.push_str(&n.to_string()),
        J::String(t) => o.push_str(&jstr(t)),
        J::Array(xs) => {
            o.push('[');
            for (i, x) in xs.iter().enumerate() {
                if i > 0 {
                    o.push(',');
                }
                write(x, o);
            }
            o.push(']');
        }
        J::Object(m) => {
            // serde_json's map is ordered by key (no `preserve_order`)
            let mut ks: Vec<(&String, &J)> = m.iter().collect();
            ks.sort_by(|a, b| a.0.cmp(b.0));
            o.push('{');
            for (i, (k, x)) in ks.into_iter().enumerate() {
                if i > 0 {
                    o.push(',');
                }
                o.push_str(&jstr(k));
                o.push(':');
                write(x, o);
            }
            o.push('}');
        }
    }
}

pub fn ty_json(t: &Ty) -> J {
    match t {
        Ty::Bool => json!(["bool"]),
        Ty::Int(lo, hi) => json!(["int", lo.to_string(), hi.to_string()]),
        Ty::F64 => json!(["f64"]),
        Ty::F32 => json!(["f32"]),
        Ty::Bytes => json!(["bytes"]),
        Ty::Arr(x) => json!(["arr", ty_json(x)]),
        Ty::List(x) => json!(["list", ty_json(x)]),
        Ty::Tup(xs) | Ty::Sum(xs) => {
            let mut v = vec![J::from(if matches!(t, Ty::Tup(_)) { "tup" } else { "sum" })];
            v.extend(xs.iter().map(ty_json));
            J::Array(v)
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn lit_json(l: &Lit) -> J {
    match l {
        Lit::Bool(b) => J::Bool(*b),
        Lit::Int(v) => J::String(v.to_string()),
        Lit::F64(b) | Lit::F32(b) => J::String(format!("{b:016x}")),
        Lit::Bytes(b) => J::String(hex(b)),
    }
}

pub fn atom_json(a: &Atom) -> J {
    match a {
        Atom::V(i) => json!(["v", i]),
        Atom::G(i) => json!(["g", i]),
        Atom::K(t, l) => json!(["k", ty_json(t), lit_json(l)]),
    }
}

fn fref_json(f: &FnRef) -> J {
    match f {
        FnRef::Key(k, i) => json!([k, i]),
        FnRef::Rec(i) => json!(["rec", i]),
    }
}

fn block_json(b: &[Stmt]) -> J {
    J::Array(b.iter().map(stmt_json).collect())
}

pub fn stmt_json(s: &Stmt) -> J {
    let tagged = |tag: &str, rest: Vec<J>| {
        let mut v = vec![J::from(tag)];
        v.extend(rest);
        J::Array(v)
    };
    match s {
        Stmt::Let(a) => tagged("let", vec![atom_json(a)]),
        Stmt::Prim(p, ts, args) => {
            let mut v = vec![J::from(p.as_str()), J::Array(ts.iter().map(ty_json).collect())];
            v.extend(args.iter().map(atom_json));
            tagged("prim", v)
        }
        Stmt::Call(f, args) | Stmt::Catch(f, args) => {
            let mut v = vec![fref_json(f)];
            v.extend(args.iter().map(atom_json));
            tagged(if matches!(s, Stmt::Call(..)) { "call" } else { "catch" }, v)
        }
        Stmt::Var(t, a) => tagged("var", vec![ty_json(t), atom_json(a)]),
        Stmt::Set(p, a) => tagged(
            "set",
            vec![
                match p {
                    Place::V(i) => json!(["v", i]),
                    Place::G(i) => json!(["g", i]),
                },
                atom_json(a),
            ],
        ),
        Stmt::If(c, t, e) => tagged("if", vec![atom_json(c), block_json(t), block_json(e)]),
        Stmt::Block(b) => tagged("block", vec![block_json(b)]),
        Stmt::Loop(b) => tagged("loop", vec![block_json(b)]),
        Stmt::Break(d) => tagged("break", vec![J::from(*d)]),
        Stmt::Ret(a) => tagged("ret", vec![atom_json(a)]),
        Stmt::Halt(a) => tagged("halt", vec![atom_json(a)]),
        Stmt::Out(a) => tagged("out", vec![atom_json(a)]),
    }
}

pub fn fn_json(f: &Fn) -> J {
    json!({
        "params": f.params.iter().map(ty_json).collect::<Vec<_>>(),
        "ret": ty_json(&f.ret),
        "body": block_json(&f.body),
    })
}

pub fn group_json(g: &Group) -> J {
    json!({"fns": g.fns.iter().map(fn_json).collect::<Vec<_>>()})
}

/// The key of a group: the SHA-256 of its canonical JSON (W5).
pub fn group_key(g: &Group) -> String {
    use sha2::{Digest, Sha256};
    hex(&Sha256::digest(canon(&group_json(g)).as_bytes()))
}

pub fn module_json(m: &Module) -> J {
    let defs: serde_json::Map<String, J> = m.defs.iter().map(|(k, g)| (k.clone(), group_json(g))).collect();
    json!({
        "format": FORMAT,
        "globals": m.globals.iter().map(|(t, l)| json!([ty_json(t), l.as_ref().map(lit_json).unwrap_or(J::Null)])).collect::<Vec<_>>(),
        "defs": J::Object(defs),
        "entry": [m.entry.0, m.entry.1],
        "names": m.names,
    })
}

/// The canonical JSON text of a module, as `rce emit core` prints it.
pub fn module_text(m: &Module) -> String {
    canon(&module_json(m))
}

// ---- reading

type R<T> = Result<T, String>;

fn arr(v: &J, what: &str) -> R<Vec<J>> {
    v.as_array().cloned().ok_or_else(|| format!("{what}: expected an array, found {v}"))
}
fn st(v: &J, what: &str) -> R<String> {
    v.as_str().map(|s| s.to_string()).ok_or_else(|| format!("{what}: expected text, found {v}"))
}
fn idx(v: &J, what: &str) -> R<u32> {
    v.as_u64().filter(|x| *x <= u32::MAX as u64).map(|x| x as u32).ok_or_else(|| format!("{what}: expected an index, found {v}"))
}
fn int_text(v: &J, what: &str) -> R<i128> {
    let s = st(v, what)?;
    let ok = !s.is_empty()
        && (s == "0" || (!s.starts_with('0') && !s.starts_with("-0")))
        && s.trim_start_matches('-').bytes().all(|b| b.is_ascii_digit())
        && s.matches('-').count() <= 1;
    if !ok {
        return Err(format!("{what}: `{s}` is not a canonical decimal integer"));
    }
    s.parse::<i128>().map_err(|_| format!("{what}: `{s}` is out of range"))
}

pub fn ty_from(v: &J) -> R<Ty> {
    let a = arr(v, "type")?;
    let tag = st(a.first().ok_or("type: empty")?, "type tag")?;
    let n = a.len();
    let want = |k: usize| if n == k { Ok(()) } else { Err(format!("type `{tag}` takes {} fields", k - 1)) };
    Ok(match tag.as_str() {
        "bool" => { want(1)?; Ty::Bool }
        "f64" => { want(1)?; Ty::F64 }
        "f32" => { want(1)?; Ty::F32 }
        "bytes" => { want(1)?; Ty::Bytes }
        "int" => {
            want(3)?;
            let (lo, hi) = (int_text(&a[1], "int lo")?, int_text(&a[2], "int hi")?);
            if !(INT_MIN <= lo && lo <= hi && hi <= INT_MAX) {
                return Err(format!("int[{lo}, {hi}] is outside -2^63 <= lo <= hi <= 2^64 - 1"));
            }
            Ty::Int(lo, hi)
        }
        "arr" => { want(2)?; Ty::Arr(Box::new(ty_from(&a[1])?)) }
        "list" => { want(2)?; Ty::List(Box::new(ty_from(&a[1])?)) }
        "tup" => Ty::Tup(a[1..].iter().map(ty_from).collect::<R<_>>()?),
        "sum" => {
            if n < 2 {
                return Err("sum needs at least one variant".into());
            }
            Ty::Sum(a[1..].iter().map(ty_from).collect::<R<_>>()?)
        }
        t => return Err(format!("unknown type `{t}`")),
    })
}

fn unhex(s: &str) -> R<Vec<u8>> {
    if s.len() % 2 != 0 || !s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(format!("`{s}` is not lower-case hex"));
    }
    Ok((0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect())
}

pub fn lit_from(t: &Ty, v: &J) -> R<Lit> {
    Ok(match t {
        Ty::Bool => Lit::Bool(v.as_bool().ok_or_else(|| format!("bool literal: {v}"))?),
        Ty::Int(..) => Lit::Int(int_text(v, "int literal")?),
        Ty::F64 | Ty::F32 => {
            let s = st(v, "float literal")?;
            if s.len() != 16 {
                return Err(format!("float literal `{s}` is not 16 hex digits"));
            }
            let b = u64::from_str_radix(&s, 16).map_err(|_| format!("float literal `{s}`"))?;
            if format!("{b:016x}") != s {
                return Err(format!("float literal `{s}` is not lower-case hex"));
            }
            if matches!(t, Ty::F64) { Lit::F64(b) } else { Lit::F32(b) }
        }
        Ty::Bytes => Lit::Bytes(unhex(&st(v, "bytes literal")?)?),
        o => return Err(format!("a literal of the owned type {o:?}")),
    })
}

pub fn atom_from(v: &J) -> R<Atom> {
    let a = arr(v, "atom")?;
    match a.first().and_then(|x| x.as_str()) {
        Some("v") if a.len() == 2 => Ok(Atom::V(idx(&a[1], "local")?)),
        Some("g") if a.len() == 2 => Ok(Atom::G(idx(&a[1], "global")?)),
        Some("k") if a.len() == 3 => {
            let t = ty_from(&a[1])?;
            let l = lit_from(&t, &a[2])?;
            Ok(Atom::K(t, l))
        }
        _ => Err(format!("not an atom: {v}")),
    }
}

fn fref_from(v: &J) -> R<FnRef> {
    let a = arr(v, "callee")?;
    if a.len() != 2 {
        return Err(format!("not a callee: {v}"));
    }
    let i = idx(&a[1], "function index")?;
    let k = st(&a[0], "callee")?;
    if k == "rec" { Ok(FnRef::Rec(i)) } else { Ok(FnRef::Key(k, i)) }
}

fn block_from(v: &J) -> R<Vec<Stmt>> {
    arr(v, "statements")?.iter().map(stmt_from).collect()
}

pub fn stmt_from(v: &J) -> R<Stmt> {
    let a = arr(v, "statement")?;
    let tag = st(a.first().ok_or("statement: empty")?, "statement tag")?;
    let n = a.len();
    let want = |k: usize| if n == k { Ok(()) } else { Err(format!("`{tag}` takes {} fields, found {}", k - 1, n - 1)) };
    let atoms = |from: usize| a[from..].iter().map(atom_from).collect::<R<Vec<_>>>();
    Ok(match tag.as_str() {
        "let" => { want(2)?; Stmt::Let(atom_from(&a[1])?) }
        "prim" => {
            if n < 3 {
                return Err("`prim` takes a name and type arguments".into());
            }
            let p = st(&a[1], "primitive")?;
            if !PRIMS.contains(&p.as_str()) {
                return Err(format!("unknown primitive `{p}`"));
            }
            let ts = arr(&a[2], "type arguments")?.iter().map(ty_from).collect::<R<_>>()?;
            Stmt::Prim(p, ts, atoms(3)?)
        }
        "call" | "catch" => {
            if n < 2 {
                return Err(format!("`{tag}` takes a callee"));
            }
            let f = fref_from(&a[1])?;
            if tag == "call" { Stmt::Call(f, atoms(2)?) } else { Stmt::Catch(f, atoms(2)?) }
        }
        "var" => { want(3)?; Stmt::Var(ty_from(&a[1])?, atom_from(&a[2])?) }
        "set" => {
            want(3)?;
            let p = match atom_from(&a[1])? {
                Atom::V(i) => Place::V(i),
                Atom::G(i) => Place::G(i),
                _ => return Err("`set` names a local or a global".into()),
            };
            Stmt::Set(p, atom_from(&a[2])?)
        }
        "if" => { want(4)?; Stmt::If(atom_from(&a[1])?, block_from(&a[2])?, block_from(&a[3])?) }
        "block" => { want(2)?; Stmt::Block(block_from(&a[1])?) }
        "loop" => { want(2)?; Stmt::Loop(block_from(&a[1])?) }
        "break" => { want(2)?; Stmt::Break(idx(&a[1], "break depth")?) }
        "ret" => { want(2)?; Stmt::Ret(atom_from(&a[1])?) }
        "halt" => { want(2)?; Stmt::Halt(atom_from(&a[1])?) }
        "out" => { want(2)?; Stmt::Out(atom_from(&a[1])?) }
        t => return Err(format!("unknown statement `{t}`")),
    })
}

fn obj<'a>(v: &'a J, what: &str, keys: &[&str]) -> R<&'a serde_json::Map<String, J>> {
    let o = v.as_object().ok_or_else(|| format!("{what}: expected an object"))?;
    for k in o.keys() {
        if !keys.contains(&k.as_str()) {
            return Err(format!("{what}: unknown field `{k}`"));
        }
    }
    for k in keys {
        if !o.contains_key(*k) {
            return Err(format!("{what}: missing field `{k}`"));
        }
    }
    Ok(o)
}

pub fn fn_from(v: &J) -> R<Fn> {
    let o = obj(v, "fn", &["params", "ret", "body"])?;
    Ok(Fn {
        params: arr(&o["params"], "params")?.iter().map(ty_from).collect::<R<_>>()?,
        ret: ty_from(&o["ret"])?,
        body: block_from(&o["body"])?,
    })
}

pub fn group_from(v: &J) -> R<Group> {
    let o = obj(v, "group", &["fns"])?;
    Ok(Group { fns: arr(&o["fns"], "fns")?.iter().map(fn_from).collect::<R<_>>()? })
}

/// Read a core module from its JSON text. The text need not be canonical;
/// the well-formedness check (`wf::check`) recomputes every key.
pub fn module_from_text(s: &str) -> R<Module> {
    let v: J = serde_json::from_str(s).map_err(|e| format!("core JSON: {e}"))?;
    module_from(&v)
}

pub fn module_from(v: &J) -> R<Module> {
    let o = obj(v, "module", &["format", "globals", "defs", "entry", "names"])?;
    if o["format"] != J::from(FORMAT) {
        return Err(format!("format is {}, not {FORMAT}", o["format"]));
    }
    let mut globals = vec![];
    for g in arr(&o["globals"], "globals")? {
        let a = arr(&g, "global")?;
        if a.len() != 2 {
            return Err("a global is [τ, LIT|null]".into());
        }
        let t = ty_from(&a[0])?;
        let l = if a[1].is_null() { None } else { Some(lit_from(&t, &a[1])?) };
        globals.push((t, l));
    }
    let mut defs = BTreeMap::new();
    for (k, g) in o["defs"].as_object().ok_or("defs: expected an object")? {
        defs.insert(k.clone(), group_from(g)?);
    }
    let e = arr(&o["entry"], "entry")?;
    if e.len() != 2 {
        return Err("entry is [KEY, i]".into());
    }
    Ok(Module { globals, defs, entry: (st(&e[0], "entry key")?, idx(&e[1], "entry index")?), names: o["names"].clone() })
}

/// A readable text of a core module, one statement per line, for reading
/// and for diagnostics (not a format any tool reads back).
pub fn text(m: &Module) -> String {
    fn ty(t: &Ty) -> String {
        match t {
            Ty::Bool => "bool".into(),
            Ty::Int(lo, hi) => format!("int[{lo},{hi}]"),
            Ty::F64 => "f64".into(),
            Ty::F32 => "f32".into(),
            Ty::Bytes => "bytes".into(),
            Ty::Arr(x) => format!("arr {}", ty(x)),
            Ty::List(x) => format!("list {}", ty(x)),
            Ty::Tup(xs) => format!("tup({})", xs.iter().map(ty).collect::<Vec<_>>().join(", ")),
            Ty::Sum(xs) => format!("sum({})", xs.iter().map(ty).collect::<Vec<_>>().join(", ")),
        }
    }
    fn atom(a: &Atom) -> String {
        match a {
            Atom::V(i) => format!("v{i}"),
            Atom::G(i) => format!("g{i}"),
            Atom::K(_, Lit::Bool(b)) => b.to_string(),
            Atom::K(_, Lit::Int(v)) => v.to_string(),
            Atom::K(_, Lit::F64(b) | Lit::F32(b)) => format!("{}", f64::from_bits(*b)),
            Atom::K(_, Lit::Bytes(b)) => format!("{:?}", String::from_utf8_lossy(b)),
        }
    }
    fn fref(f: &FnRef, names: &serde_json::Value) -> String {
        match f {
            FnRef::Key(k, i) => names["fns"][k][*i as usize].as_str().map(|s| s.to_string()).unwrap_or(format!("{}/{i}", &k[..8.min(k.len())])),
            FnRef::Rec(i) => format!("rec {i}"),
        }
    }
    fn block(b: &[Stmt], d: usize, next: &mut u32, o: &mut String, names: &serde_json::Value) {
        let pad = "  ".repeat(d);
        for s in b {
            let at = *next;
            *next += binds(s) as u32;
            let lhs = |n: usize| (0..n).map(|k| format!("v{}", at + k as u32)).collect::<Vec<_>>().join(", ");
            let args = |a: &[Atom]| a.iter().map(atom).collect::<Vec<_>>().join(", ");
            match s {
                Stmt::Let(a) => o.push_str(&format!("{pad}{} = {}\n", lhs(1), atom(a))),
                Stmt::Prim(p, t, a) => {
                    let n = binds(s);
                    let l = if n == 0 { String::new() } else { format!("{} = ", lhs(n)) };
                    o.push_str(&format!("{pad}{l}{p}[{}]({})\n", t.iter().map(ty).collect::<Vec<_>>().join(", "), args(a)));
                }
                Stmt::Call(f, a) => o.push_str(&format!("{pad}{} = call {}({})\n", lhs(1), fref(f, names), args(a))),
                Stmt::Catch(f, a) => o.push_str(&format!("{pad}{} = catch {}({})\n", lhs(1), fref(f, names), args(a))),
                Stmt::Var(t, a) => o.push_str(&format!("{pad}var {}: {} = {}\n", lhs(1), ty(t), atom(a))),
                Stmt::Set(Place::V(i), a) => o.push_str(&format!("{pad}set v{i} = {}\n", atom(a))),
                Stmt::Set(Place::G(i), a) => o.push_str(&format!("{pad}set g{i} = {}\n", atom(a))),
                Stmt::If(c, t, e) => {
                    o.push_str(&format!("{pad}if {} {{\n", atom(c)));
                    block(t, d + 1, next, o, names);
                    o.push_str(&format!("{pad}}} else {{\n"));
                    block(e, d + 1, next, o, names);
                    o.push_str(&format!("{pad}}}\n"));
                }
                Stmt::Block(x) | Stmt::Loop(x) => {
                    o.push_str(&format!("{pad}{} {{\n", if matches!(s, Stmt::Block(_)) { "block" } else { "loop" }));
                    block(x, d + 1, next, o, names);
                    o.push_str(&format!("{pad}}}\n"));
                }
                Stmt::Break(n) => o.push_str(&format!("{pad}break {n}\n")),
                Stmt::Ret(a) => o.push_str(&format!("{pad}ret {}\n", atom(a))),
                Stmt::Halt(a) => o.push_str(&format!("{pad}halt {}\n", atom(a))),
                Stmt::Out(a) => o.push_str(&format!("{pad}out {}\n", atom(a))),
            }
        }
    }
    let mut o = String::new();
    for (i, (t, l)) in m.globals.iter().enumerate() {
        o.push_str(&format!("global g{i}: {} = {:?}  ; {}\n", ty(t), l, m.names["globals"][i].as_str().unwrap_or("")));
    }
    for (k, g) in &m.defs {
        for (i, f) in g.fns.iter().enumerate() {
            let name = m.names["fns"][k][i].as_str().unwrap_or("?");
            o.push_str(&format!("\n; {name}\nfn {}/{i}({}) -> {}\n", &k[..12], f.params.iter().enumerate().map(|(j, t)| format!("v{j}: {}", ty(t))).collect::<Vec<_>>().join(", "), ty(&f.ret)));
            let mut next = f.params.len() as u32;
            block(&f.body, 1, &mut next, &mut o, &m.names);
        }
    }
    o
}
