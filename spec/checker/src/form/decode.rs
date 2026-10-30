//! Strict decoding of an `rce-form/1` module (C1, gate rule FG1): every field
//! is present and no unknown field is; integers are canonical decimal text
//! and floats 16 lower-case hexadecimal digits; a module with an unknown
//! kind, entry, type or catalogue key is kept. Nothing defaults.
use super::RequiresChange;
use super::catalogue;
use super::ir::*;
use serde_json::Value as J;
use std::collections::BTreeMap;

type R<T> = Result<T, RequiresChange>;

/// Where the decoder is: a group and function, if inside one, and a path.
struct Cx {
    group: Option<String>,
    func: Option<u32>,
    path: Vec<String>,
    ntypes: usize,
    nerrors: usize,
    nregions: usize,
}

impl Cx {
    fn fail<T>(&self, reason: impl Into<String>) -> R<T> {
        Err(RequiresChange {
            rule: "FG1",
            group: self.group.clone(),
            func: self.func,
            path: self.path.join(" "),
            reason: format!("C1: {}", reason.into()),
        })
    }
    fn at<T>(&mut self, seg: impl Into<String>, f: impl FnOnce(&mut Cx) -> R<T>) -> R<T> {
        self.path.push(seg.into());
        let r = f(self);
        self.path.pop();
        r
    }
    fn arr<'a>(&self, v: &'a J, what: &str) -> R<&'a Vec<J>> {
        match v.as_array() {
            Some(a) => Ok(a),
            None => self.fail(format!("{what}: expected an array")),
        }
    }
    fn arr_n<'a>(&self, v: &'a J, n: usize, what: &str) -> R<&'a Vec<J>> {
        let a = self.arr(v, what)?;
        if a.len() != n {
            return self.fail(format!("{what}: expected {n} elements, found {}", a.len()));
        }
        Ok(a)
    }
    fn st<'a>(&self, v: &'a J, what: &str) -> R<&'a str> {
        match v.as_str() {
            Some(s) => Ok(s),
            None => self.fail(format!("{what}: expected text")),
        }
    }
    fn idx(&self, v: &J, what: &str) -> R<u32> {
        match v.as_u64() {
            Some(x) if x <= u32::MAX as u64 && !v.is_f64() => Ok(x as u32),
            _ => self.fail(format!("{what}: expected an index, found {v}")),
        }
    }
    /// An object with exactly the named fields (C1: every field present, no unknown field).
    fn obj<'a>(&self, v: &'a J, fields: &[&str], what: &str) -> R<&'a serde_json::Map<String, J>> {
        let Some(o) = v.as_object() else { return self.fail(format!("{what}: expected an object")) };
        for f in fields {
            if !o.contains_key(*f) {
                return self.fail(format!("{what}: the field `{f}` is missing"));
            }
        }
        if let Some(k) = o.keys().find(|k| !fields.contains(&k.as_str())) {
            return self.fail(format!(
                "{what}: the field `{k}` is unknown; no field may hold a name, a comment, a span or layout (C10)"
            ));
        }
        Ok(o)
    }
}

/// Canonical decimal text of an integer.
fn dec(s: &str) -> Option<i128> {
    let ok = !s.is_empty()
        && (s == "0" || (!s.starts_with('0') && !s.starts_with("-0")))
        && s.trim_start_matches('-').bytes().all(|b| b.is_ascii_digit())
        && !s.trim_start_matches('-').is_empty()
        && s.matches('-').count() <= 1
        && (s.find('-').is_none_or(|i| i == 0));
    if ok { s.parse::<i128>().ok() } else { None }
}

pub fn is_key(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

const INT_MIN: i128 = -(1i128 << 63);
const INT_MAX: i128 = (1i128 << 64) - 1;

fn int_range(cx: &Cx, a: &[J], what: &str) -> R<(i128, i128)> {
    let lo = dec(cx.st(&a[1], what)?);
    let hi = dec(cx.st(&a[2], what)?);
    match (lo, hi) {
        (Some(lo), Some(hi)) if INT_MIN <= lo && lo <= hi && hi <= INT_MAX => Ok((lo, hi)),
        _ => cx.fail(format!("{what}: `{}` is not an integer type with -2^63 <= lo <= hi <= 2^64 - 1", J::Array(a.to_vec()))),
    }
}

fn bty(cx: &Cx, v: &J) -> R<BTy> {
    let a = cx.arr(v, "type")?;
    let Some(tag) = a.first().and_then(|t| t.as_str()) else { return cx.fail(format!("unknown type {v}")) };
    let n = a.len();
    let want = |k: usize| if n == k { Ok(()) } else { cx.fail(format!("type `{tag}` takes {} fields", k - 1)) };
    Ok(match tag {
        "bool" => { want(1)?; BTy::Bool }
        "f64" => { want(1)?; BTy::F64 }
        "f32" => { want(1)?; BTy::F32 }
        "bytes" => { want(1)?; BTy::Bytes }
        "order" => { want(1)?; BTy::Order }
        "int" => {
            want(3)?;
            let (lo, hi) = int_range(cx, a, "int type")?;
            BTy::Int(lo, hi)
        }
        "arr" => { want(2)?; BTy::Arr(Box::new(bty(cx, &a[1])?)) }
        "list" => { want(2)?; BTy::List(Box::new(bty(cx, &a[1])?)) }
        "tup" => BTy::Tup(a[1..].iter().map(|x| bty(cx, x)).collect::<R<_>>()?),
        "sum" => {
            if n < 2 {
                return cx.fail("a sum type needs at least one variant");
            }
            BTy::Sum(a[1..].iter().map(|x| bty(cx, x)).collect::<R<_>>()?)
        }
        "decl" => {
            want(2)?;
            let i = cx.idx(&a[1], "decl index")?;
            if i as usize >= cx.ntypes {
                return cx.fail(format!("`[\"decl\",{i}]` names no declared type"));
            }
            BTy::Decl(i)
        }
        t => return cx.fail(format!("unknown body type `{t}`")),
    })
}

fn fty(cx: &Cx, v: &J) -> R<FTy> {
    let a = cx.arr(v, "frame type")?;
    let Some(tag) = a.first().and_then(|t| t.as_str()) else { return cx.fail(format!("unknown frame type {v}")) };
    let n = a.len();
    let want = |k: usize| if n == k { Ok(()) } else { cx.fail(format!("frame type `{tag}` takes {} fields", k - 1)) };
    let one = |i: usize| fty(cx, &a[i]).map(Box::new);
    Ok(match tag {
        "bool" => { want(1)?; FTy::Bool }
        "f64" => { want(1)?; FTy::F64 }
        "f32" => { want(1)?; FTy::F32 }
        "text" => { want(1)?; FTy::Text }
        "unit" => { want(1)?; FTy::Unit }
        "int" => {
            want(3)?;
            let (lo, hi) = int_range(cx, a, "int type")?;
            FTy::Int(lo, hi)
        }
        "tup" => {
            if n < 3 {
                return cx.fail("a frame tuple type has at least two components");
            }
            FTy::Tup(a[1..].iter().map(|x| fty(cx, x)).collect::<R<_>>()?)
        }
        "arr" => {
            want(3)?;
            let n = match dec(cx.st(&a[2], "array length")?) {
                Some(n) if (0..=u64::MAX as i128).contains(&n) => n as u64,
                _ => return cx.fail("array length: not a canonical decimal length"),
            };
            FTy::Arr(one(1)?, n)
        }
        "list" => { want(2)?; FTy::List(one(1)?) }
        "set" => { want(2)?; FTy::Set(one(1)?) }
        "map" => { want(3)?; FTy::Map(one(1)?, one(2)?) }
        "opt" => { want(2)?; FTy::Opt(one(1)?) }
        "chan" => { want(2)?; FTy::Chan(one(1)?) }
        "res" => {
            want(3)?;
            let mut xs = vec![];
            for x in cx.arr(&a[2], "result errors")? {
                let x = cx.idx(x, "error index")?;
                if x as usize >= cx.nerrors {
                    return cx.fail(format!("error index {x} names no declared error"));
                }
                xs.push(x);
            }
            FTy::Res(one(1)?, xs)
        }
        "decl" => {
            want(2)?;
            let i = cx.idx(&a[1], "decl index")?;
            if i as usize >= cx.ntypes {
                return cx.fail(format!("`[\"decl\",{i}]` names no declared type"));
            }
            FTy::Decl(i)
        }
        "actor" => {
            want(2)?;
            let r = cx.idx(&a[1], "actor region")?;
            if r as usize >= cx.nregions {
                return cx.fail(format!("`[\"actor\",{r}]` names no region"));
            }
            FTy::Actor(r)
        }
        t => return cx.fail(format!("unknown frame type `{t}`")),
    })
}

fn key(cx: &Cx, v: &J, what: &str) -> R<String> {
    let s = cx.st(v, what)?;
    if !is_key(s) {
        return cx.fail(format!("{what}: `{s}` is not 64 lower-case hexadecimal digits"));
    }
    Ok(s.to_string())
}

fn rf(cx: &Cx, v: &J) -> R<Ref> {
    let a = cx.arr_n(v, 2, "reference")?;
    if a[0] == "rec" {
        return Ok(Ref::Rec(cx.idx(&a[1], "rec index")?));
    }
    Ok(Ref::Key(key(cx, &a[0], "reference key")?, cx.idx(&a[1], "function index")?))
}

fn cell(cx: &Cx, v: &J) -> R<Cell> {
    let a = cx.arr_n(v, 2, "cell")?;
    match a[0].as_str() {
        Some("memo") => Ok(Cell::Memo(rf(cx, &a[1])?)),
        Some("buffer") => Ok(Cell::Buffer(rf(cx, &a[1])?)),
        _ => cx.fail(format!("unknown cell {v}")),
    }
}

fn src(cx: &Cx, v: &J) -> R<Src> {
    let a = cx.arr_n(v, 2, "source")?;
    if a[0].as_i64() == Some(-1) && !a[0].is_f64() {
        return Ok(Src::In(cx.idx(&a[1], "input index")?));
    }
    Ok(Src::Out(cx.idx(&a[0], "node index")?, cx.idx(&a[1], "output index")?))
}

fn tylist(cx: &Cx, v: &J) -> R<Vec<BTy>> {
    cx.arr(v, "types")?.iter().map(|t| bty(cx, t)).collect()
}

fn lit(cx: &Cx, t: &BTy, v: &J) -> R<Lit> {
    use crate::one::ir::Ty;
    let ct = match t {
        BTy::Bool => Ty::Bool,
        BTy::Int(lo, hi) => Ty::Int(*lo, *hi),
        BTy::F64 => Ty::F64,
        BTy::F32 => Ty::F32,
        BTy::Bytes => Ty::Bytes,
        _ => return cx.fail("a literal of a type that is not plain"),
    };
    let l = match crate::one::json::lit_from(&ct, v) {
        Ok(l) => l,
        Err(e) => return cx.fail(format!("literal: {e}")),
    };
    if let (Lit::Int(x), BTy::Int(lo, hi)) = (&l, t)
        && !(lo <= x && x <= hi)
    {
        return cx.fail(format!("the literal {x} is not a value of int[{lo}, {hi}]"));
    }
    Ok(l)
}

fn attrs(cx: &Cx, k: u64, v: &J) -> R<Attr> {
    let a = cx.arr(v, "attrs")?;
    let head = a.first().and_then(|x| x.as_str());
    Ok(match (k, head) {
        (0, Some("lit")) => {
            cx.arr_n(v, 3, "Bind attrs")?;
            let t = bty(cx, &a[1])?;
            Attr::Lit(t.clone(), lit(cx, &t, &a[2])?)
        }
        (1, Some("prim")) => {
            if a.len() != 3 && a.len() != 4 {
                return cx.fail("Apply attrs `prim` take a name, type arguments and, for a cell entry, a cell");
            }
            let name = cx.st(&a[1], "entry name")?;
            let Some(e) = catalogue::entry(name) else {
                return cx.fail(format!("`{name}` is not an entry of the catalogue"));
            };
            let mut args = vec![];
            for (i, t) in cx.arr(&a[2], "type arguments")?.iter().enumerate() {
                args.push(if e.params.get(i).is_some_and(|p| p == "frame") { TyArg::Frame(fty(cx, t)?) } else { TyArg::Body(bty(cx, t)?) });
            }
            let c = match (e.cell, a.len()) {
                (true, 4) => Some(cell(cx, &a[3])?),
                (false, 3) => None,
                (true, _) => return cx.fail(format!("`{name}` names a memo cell, and its attrs name none")),
                (false, _) => return cx.fail(format!("`{name}` names no memo cell, and its attrs name one")),
            };
            Attr::Prim(name.to_string(), args, c)
        }
        (1, Some("call")) => Attr::Call(rf(cx, &cx.arr_n(v, 2, "call attrs")?[1])?),
        (1, Some("catch")) => Attr::Catch(rf(cx, &cx.arr_n(v, 2, "catch attrs")?[1])?),
        (1, Some("cap")) => Attr::Cap(key(cx, &cx.arr_n(v, 2, "cap attrs")?[1], "capability key")?),
        (3, _) => {
            cx.arr_n(v, 2, "Branch attrs")?;
            let n = cx.idx(&a[0], "arms")?;
            if n < 2 {
                return cx.fail("a Branch has n >= 2 arms");
            }
            Attr::Branch(n, tylist(cx, &a[1])?)
        }
        (4, Some("count")) => Attr::Count(tylist(cx, &cx.arr_n(v, 2, "Iterate attrs")?[1])?),
        (4, Some("measure")) => {
            cx.arr_n(v, 3, "Iterate attrs")?;
            let ij = cx.arr_n(&a[1], 2, "measure positions")?;
            Attr::Measure(cx.idx(&ij[0], "measure i")?, cx.idx(&ij[1], "measure j")?, tylist(cx, &a[2])?)
        }
        (13, _) => Attr::Halt(tylist(cx, &cx.arr_n(v, 1, "Halt attrs")?[0])?),
        (0 | 1 | 4, _) => return cx.fail(format!("attrs {v} are not attrs of a node of kind {k}")),
        _ => return cx.fail(format!("unknown node kind {k}")),
    })
}

fn body(cx: &mut Cx, v: &J) -> R<Body> {
    let o = cx.obj(v, &["nodes", "out"], "region")?;
    let mut nodes = vec![];
    for (i, n) in cx.arr(&o["nodes"], "nodes")?.iter().enumerate() {
        nodes.push(cx.at(format!("node {i}"), |cx| node(cx, n))?);
    }
    let out = cx.arr(&o["out"], "out")?.iter().map(|s| src(cx, s)).collect::<R<_>>()?;
    Ok(Body { nodes, out })
}

fn node(cx: &mut Cx, v: &J) -> R<Node> {
    let a = cx.arr_n(v, 4, "node")?;
    let k = match a[0].as_u64() {
        Some(k) if !a[0].is_f64() => k,
        _ => return cx.fail(format!("unknown node kind {}", a[0])),
    };
    let attrs = attrs(cx, k, &a[1])?;
    let ins = cx.arr(&a[2], "ins")?.iter().map(|s| src(cx, s)).collect::<R<_>>()?;
    let mut regions = vec![];
    for (i, r) in cx.arr(&a[3], "regions")?.iter().enumerate() {
        regions.push(cx.at(format!("region {i}"), |cx| body(cx, r))?);
    }
    Ok(Node { attrs, ins, regions })
}

fn func(cx: &mut Cx, v: &J) -> R<Func> {
    let o = cx.obj(v, &["sig", "bound", "body"], "function")?;
    let sig = cx.arr_n(&o["sig"], 3, "sig")?;
    let ins = tylist(cx, &sig[0])?;
    let outs = tylist(cx, &sig[1])?;
    let eff = match sig[2].as_u64() {
        Some(e) if e <= 3 && !sig[2].is_f64() => e as u8,
        _ => return cx.fail(format!("effect class {} is not 0, 1, 2 or 3", sig[2])),
    };
    let bound = match &o["bound"] {
        J::Null => Bound::None,
        J::String(s) => match dec(s) {
            Some(n) if n > 0 => Bound::N(n as u128),
            _ => return cx.fail(format!("bound `{s}` is not a positive canonical decimal")),
        },
        J::Array(a) if a.len() == 2 && a[0] == "layers" => {
            let mut layers = vec![];
            for l in cx.arr(&a[1], "layers")? {
                let mut per = vec![];
                for f in cx.arr(l, "layer")? {
                    per.push(cx.arr(f, "positions")?.iter().map(|i| cx.idx(i, "position")).collect::<R<Vec<u32>>>()?);
                }
                layers.push(per);
            }
            Bound::Layers(layers)
        }
        b => return cx.fail(format!("bound {b} is not null, a decimal or a layered-measure certificate")),
    };
    let body = cx.at("body", |cx| body(cx, &o["body"]))?;
    Ok(Func { ins, outs, eff, bound, body })
}

fn group(cx: &mut Cx, v: &J) -> R<Group> {
    let o = cx.obj(v, &["fns"], "group")?;
    let mut fns = vec![];
    for (i, f) in cx.arr(&o["fns"], "fns")?.iter().enumerate() {
        cx.func = Some(i as u32);
        fns.push(func(cx, f)?);
    }
    cx.func = None;
    if fns.is_empty() {
        return cx.fail("a group holds at least one function");
    }
    Ok(Group { fns })
}

/// Decode a module from its bytes, or keep it under FG1.
pub fn decode(bytes: &[u8]) -> R<Module> {
    let mut cx = Cx { group: None, func: None, path: vec![], ntypes: 0, nerrors: 0, nregions: 0 };
    let v: J = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(e) => return cx.fail(format!("the bytes are not JSON: {e}")),
    };
    let o = cx.obj(&v, &["format", "catalogue", "types", "regions", "events", "errors", "steps", "main", "caps", "defs"], "module")?;
    if o["format"] != FORMAT {
        return cx.fail(format!("format {} is not `{FORMAT}`", o["format"]));
    }
    let catalogue = key(&cx, &o["catalogue"], "catalogue")?;
    if catalogue != catalogue::key() {
        return cx.fail(format!("catalogue key {catalogue} is unknown: this checker carries {}", catalogue::key()));
    }
    cx.ntypes = cx.arr(&o["types"], "types")?.len();
    cx.nerrors = cx.arr(&o["errors"], "errors")?.len();
    cx.nregions = cx.arr(&o["regions"], "regions")?.len();
    let mut types = vec![];
    for (i, t) in o["types"].as_array().unwrap().iter().enumerate() {
        types.push(cx.at(format!("types[{i}]"), |cx| {
            let a = cx.arr(t, "type declaration")?;
            match a.first().and_then(|x| x.as_str()) {
                Some("record") if a.len() == 2 => Ok(TypeDecl::Record(cx.arr(&a[1], "fields")?.iter().map(|x| fty(cx, x)).collect::<R<_>>()?)),
                Some("kind") if a.len() == 2 => {
                    let mut vs = vec![];
                    for x in cx.arr(&a[1], "variants")? {
                        vs.push(cx.arr(x, "variant fields")?.iter().map(|y| fty(cx, y)).collect::<R<_>>()?);
                    }
                    Ok(TypeDecl::Kind(vs))
                }
                Some("refined") if a.len() == 3 => Ok(TypeDecl::Refined(fty(cx, &a[1])?, rf(cx, &a[2])?)),
                _ => cx.fail(format!("unknown type declaration {t}")),
            }
        })?);
    }
    let mut regions = vec![];
    for (i, r) in o["regions"].as_array().unwrap().iter().enumerate() {
        regions.push(cx.at(format!("regions[{i}]"), |cx| {
            let a = cx.arr_n(r, 2, "region")?;
            let actor = match a[0].as_str() {
                Some("state") => false,
                Some("actor") => true,
                _ => return cx.fail(format!("region kind {} is not `state` or `actor`", a[0])),
            };
            let mut fields = vec![];
            for f in cx.arr(&a[1], "fields")? {
                let f = cx.arr_n(f, 2, "field")?;
                fields.push((fty(cx, &f[0])?, f[1].clone()));
            }
            Ok(RegionDecl { actor, fields })
        })?);
    }
    let table = |cx: &mut Cx, name: &str| -> R<Vec<Vec<FTy>>> {
        let mut out = vec![];
        for (i, e) in cx.arr(&o[name], name)?.iter().enumerate() {
            out.push(cx.at(format!("{name}[{i}]"), |cx| cx.arr(e, "fields")?.iter().map(|t| fty(cx, t)).collect::<R<Vec<_>>>())?);
        }
        Ok(out)
    };
    let events = table(&mut cx, "events")?;
    let errors = table(&mut cx, "errors")?;
    let mut steps = vec![];
    for (i, s) in cx.arr(&o["steps"], "steps")?.iter().enumerate() {
        steps.push(cx.at(format!("steps[{i}]"), |cx| {
            let a = cx.arr_n(s, 5, "step")?;
            let ev = match &a[1] {
                J::String(n) => Ev::Name(n.clone()),
                x => Ev::Idx(cx.idx(x, "event")?),
            };
            let guard = if a[2].is_null() { None } else { Some(rf(cx, &a[2])?) };
            let deadline = match &a[4] {
                J::Null => None,
                J::String(d) => match dec(d) {
                    Some(n) if n >= 0 => Some(n as u128),
                    _ => return cx.fail(format!("deadline `{d}` is not a canonical decimal")),
                },
                d => return cx.fail(format!("deadline {d} is not a decimal or null")),
            };
            Ok(Step { region: cx.idx(&a[0], "region")?, ev, guard, body: rf(cx, &a[3])?, deadline })
        })?);
    }
    let main = if o["main"].is_null() { None } else { Some(cx.at("main", |cx| rf(cx, &o["main"]))?) };
    let caps = cx.arr(&o["caps"], "caps")?.iter().map(|c| key(&cx, c, "capability key")).collect::<R<Vec<_>>>()?;
    let Some(dv) = o["defs"].as_object() else { return cx.fail("defs: expected an object") };
    let mut defs = BTreeMap::new();
    for (k, g) in dv {
        if !is_key(k) {
            return cx.fail(format!("defs: `{k}` is not a group key"));
        }
        cx.group = Some(k.clone());
        defs.insert(k.clone(), group(&mut cx, g)?);
        cx.group = None;
    }
    Ok(Module { catalogue, types, regions, events, errors, steps, main, caps, defs })
}
