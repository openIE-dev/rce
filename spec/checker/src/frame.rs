//! Frames (Appendix E, HC-29): the `ui.view` tree a program emits as
//! `Frame(view)` after a step, its digests, and the reference library calls
//! the host makes on frames.
//!
//! A frame is derived, never an input: it is an output of δ for the entry
//! at its seq, so the frames of a session are a function of D and replay
//! re-derives every one with every adapter disconnected. The encoding is
//! the one `ui.view` defines (stdlib/lib/ui/view.rce): each node is the
//! deterministic CBOR array `[tag, key, {attrs}, text, [children]]`,
//! `view_digest` is its BLAKE3, and the frame record
//! `["rce.ui.frame/1", seq, view_digest]` has `frame_digest` its BLAKE3.
//! `tests/host_frames.rs` checks both digests against `ui.view` itself.
//!
//! The diff operations of a sink line and the `ui.rules` report are not
//! computed here: they are the reference functions `view.diff`,
//! `rules.findings` and `rules.report`, run by the interpreter on a module
//! that uses only `ui.view` and `ui.rules`.
use crate::ast::{Arg, Expr, Module, Ty};
use crate::interp::{Interp, V};
use serde_json::{Value, json};
use std::collections::HashMap;

// ---------------- BLAKE3 (unkeyed hash, 32-byte output) ----------------

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
const PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;
const CHUNK: usize = 1024;

fn g(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(x);
    s[d] = (s[d] ^ s[a]).rotate_right(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(12);
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(y);
    s[d] = (s[d] ^ s[a]).rotate_right(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(7);
}

fn compress(cv: &[u32; 8], block: &[u32; 16], counter: u64, len: u32, flags: u32) -> [u32; 16] {
    let mut s = [cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7], IV[0], IV[1], IV[2], IV[3], counter as u32, (counter >> 32) as u32, len, flags];
    let mut m = *block;
    for r in 0..7 {
        g(&mut s, 0, 4, 8, 12, m[0], m[1]);
        g(&mut s, 1, 5, 9, 13, m[2], m[3]);
        g(&mut s, 2, 6, 10, 14, m[4], m[5]);
        g(&mut s, 3, 7, 11, 15, m[6], m[7]);
        g(&mut s, 0, 5, 10, 15, m[8], m[9]);
        g(&mut s, 1, 6, 11, 12, m[10], m[11]);
        g(&mut s, 2, 7, 8, 13, m[12], m[13]);
        g(&mut s, 3, 4, 9, 14, m[14], m[15]);
        if r < 6 {
            let mut p = [0u32; 16];
            for (i, &j) in PERM.iter().enumerate() {
                p[i] = m[j];
            }
            m = p;
        }
    }
    for i in 0..8 {
        s[i] ^= s[i + 8];
        s[i + 8] ^= cv[i];
    }
    s
}

fn words(b: &[u8]) -> [u32; 16] {
    let mut w = [0u32; 16];
    for (i, x) in b.iter().enumerate() {
        w[i / 4] |= (*x as u32) << (8 * (i % 4));
    }
    w
}

/// The last compression of a node, with its root flag still to be decided.
struct Node {
    cv: [u32; 8],
    block: [u32; 16],
    counter: u64,
    len: u32,
    flags: u32,
}

impl Node {
    fn chaining(&self) -> [u32; 8] {
        let s = compress(&self.cv, &self.block, self.counter, self.len, self.flags);
        let mut o = [0u32; 8];
        o.copy_from_slice(&s[..8]);
        o
    }
}

fn chunk(data: &[u8], index: u64) -> Node {
    let mut cv = IV;
    let blocks: Vec<&[u8]> = if data.is_empty() { vec![&[][..]] } else { data.chunks(64).collect() };
    let last = blocks.len() - 1;
    for (i, b) in blocks.iter().enumerate() {
        let flags = if i == 0 { CHUNK_START } else { 0 } | if i == last { CHUNK_END } else { 0 };
        let n = Node { cv, block: words(b), counter: index, len: b.len() as u32, flags };
        if i == last {
            return n;
        }
        cv = n.chaining();
    }
    unreachable!()
}

fn subtree(data: &[u8], first: u64) -> Node {
    if data.len() <= CHUNK {
        return chunk(data, first);
    }
    // the left subtree takes the largest power of two of whole chunks that
    // leaves at least one byte for the right
    let full = (data.len() - 1) / CHUNK;
    let left = 1usize << (usize::BITS - 1 - full.leading_zeros());
    let (l, r) = data.split_at(left * CHUNK);
    let (a, b) = (subtree(l, first).chaining(), subtree(r, first + left as u64).chaining());
    let mut block = [0u32; 16];
    block[..8].copy_from_slice(&a);
    block[8..].copy_from_slice(&b);
    Node { cv: IV, block, counter: 0, len: 64, flags: PARENT }
}

/// BLAKE3 of `data`, 32 bytes.
pub fn blake3(data: &[u8]) -> [u8; 32] {
    let n = subtree(data, 0);
    let s = compress(&n.cv, &n.block, n.counter, n.len, n.flags | ROOT);
    let mut o = [0u8; 32];
    for (i, w) in s[..8].iter().enumerate() {
        o[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    o
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ---------------- the view tree ----------------

/// A `ui.view.View`: a flat arena of nodes, node 0 the root.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewTree {
    pub tag: Vec<String>,
    pub key: Vec<String>,
    pub text: Vec<String>,
    pub names: Vec<Vec<String>>,
    pub values: Vec<Vec<String>>,
    pub parent: Vec<i64>,
    pub kids: Vec<Vec<i64>>,
}

fn field<'v>(fs: &'v [(String, V)], k: &str) -> Result<&'v V, String> {
    fs.iter().find(|(n, _)| n == k).map(|(_, v)| v).ok_or_else(|| format!("no field `{k}`"))
}

fn list(v: &V) -> Result<&[V], String> {
    match v {
        V::List(xs) => Ok(xs),
        _ => Err("not a list".into()),
    }
}

fn texts(v: &V) -> Result<Vec<String>, String> {
    list(v)?.iter().map(|x| if let V::Text(t) = x { Ok(t.to_string()) } else { Err("not text".to_string()) }).collect()
}

fn ints(v: &V) -> Result<Vec<i64>, String> {
    list(v)?.iter().map(|x| if let V::Int(i, _) = x { i64::try_from(*i).map_err(|e| e.to_string()) } else { Err("not an integer".to_string()) }).collect()
}

impl ViewTree {
    /// The view a value of `ui.view.View` holds, when it is a tree: every
    /// node is reached from node 0 exactly once, and parent links agree.
    pub fn from_value(v: &V) -> Result<ViewTree, String> {
        let V::Rec(_, fs) = v else { return Err("the view is not a record".into()) };
        let t = |k: &str| field(fs, k).and_then(texts).map_err(|e| format!("view.{k}: {e}"));
        let tl = |k: &str| -> Result<Vec<Vec<String>>, String> { list(field(fs, k)?)?.iter().map(texts).collect::<Result<_, _>>().map_err(|e| format!("view.{k}: {e}")) };
        let kids: Vec<Vec<i64>> = list(field(fs, "kids")?)?.iter().map(ints).collect::<Result<_, _>>().map_err(|e| format!("view.kids: {e}"))?;
        let v = ViewTree { tag: t("tag")?, key: t("key")?, text: t("text")?, names: tl("names")?, values: tl("values")?, parent: ints(field(fs, "parent")?).map_err(|e| format!("view.parent: {e}"))?, kids };
        v.check()?;
        Ok(v)
    }

    fn check(&self) -> Result<(), String> {
        let n = self.tag.len();
        if n == 0 {
            return Err("the view has no root".into());
        }
        if [self.key.len(), self.text.len(), self.names.len(), self.values.len(), self.parent.len(), self.kids.len()].iter().any(|l| *l != n) {
            return Err("the view's lists differ in length".into());
        }
        let mut seen = vec![false; n];
        let mut st = vec![0usize];
        seen[0] = true;
        while let Some(u) = st.pop() {
            if self.names[u].len() != self.values[u].len() {
                return Err(format!("node {u}: names and values differ in length"));
            }
            for &c in &self.kids[u] {
                let c = usize::try_from(c).ok().filter(|c| *c < n).ok_or_else(|| format!("node {u}: child {c} is not a node"))?;
                if seen[c] || self.parent[c] != u as i64 {
                    return Err(format!("node {c} is not a tree child of node {u}"));
                }
                seen[c] = true;
                st.push(c);
            }
        }
        Ok(())
    }

    /// The deterministic CBOR of the tree under node 0, as `view.view_cbor`.
    pub fn cbor(&self) -> Vec<u8> {
        let mut o = vec![];
        let mut st = vec![0usize];
        while let Some(u) = st.pop() {
            head(&mut o, 4, 5);
            text(&mut o, &self.tag[u]);
            text(&mut o, &self.key[u]);
            head(&mut o, 5, self.names[u].len() as u64);
            for (k, v) in self.names[u].iter().zip(&self.values[u]) {
                text(&mut o, k);
                text(&mut o, v);
            }
            text(&mut o, &self.text[u]);
            head(&mut o, 4, self.kids[u].len() as u64);
            for c in self.kids[u].iter().rev() {
                st.push(*c as usize);
            }
        }
        o
    }

    pub fn digest(&self) -> [u8; 32] {
        blake3(&self.cbor())
    }

    /// The Form JSON of the view: its seven lists, keys sorted.
    pub fn json(&self) -> Value {
        json!({"kids": self.kids, "key": self.key, "names": self.names, "parent": self.parent, "tag": self.tag, "text": self.text, "values": self.values})
    }

    pub fn from_json(v: &Value) -> Result<ViewTree, String> {
        let s = |k: &str| -> Result<Vec<String>, String> { serde_json::from_value(v[k].clone()).map_err(|e| format!("view.{k}: {e}")) };
        let ss = |k: &str| -> Result<Vec<Vec<String>>, String> { serde_json::from_value(v[k].clone()).map_err(|e| format!("view.{k}: {e}")) };
        let t = ViewTree {
            tag: s("tag")?,
            key: s("key")?,
            text: s("text")?,
            names: ss("names")?,
            values: ss("values")?,
            parent: serde_json::from_value(v["parent"].clone()).map_err(|e| format!("view.parent: {e}"))?,
            kids: serde_json::from_value(v["kids"].clone()).map_err(|e| format!("view.kids: {e}"))?,
        };
        t.check()?;
        Ok(t)
    }
}

fn head(o: &mut Vec<u8>, m: u8, n: u64) {
    let mt = m << 5;
    if n < 24 {
        o.push(mt | n as u8);
    } else if n < 256 {
        o.extend([mt | 24, n as u8]);
    } else if n < 65536 {
        o.push(mt | 25);
        o.extend((n as u16).to_be_bytes());
    } else if n < 1 << 32 {
        o.push(mt | 26);
        o.extend((n as u32).to_be_bytes());
    } else {
        o.push(mt | 27);
        o.extend(n.to_be_bytes());
    }
}

fn text(o: &mut Vec<u8>, t: &str) {
    head(o, 3, t.len() as u64);
    o.extend(t.as_bytes());
}

/// `["rce.ui.frame/1", seq, view_digest]` as deterministic CBOR (`view.frame_record`).
pub fn frame_record(seq: u64, view_digest: &[u8; 32]) -> Vec<u8> {
    let mut o = vec![];
    head(&mut o, 4, 3);
    text(&mut o, "rce.ui.frame/1");
    head(&mut o, 0, seq);
    head(&mut o, 2, 32);
    o.extend(view_digest);
    o
}

// ---------------- frames ----------------

/// One frame: the view the step of the entry at `seq` emitted, with the
/// `ui.rules` step word and time the host derives for it (HC-29).
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub seq: u64,
    /// the event type of the entry at seq
    pub input: String,
    /// the `ui.rules` event word of the step
    pub ev: String,
    /// milliseconds after the previous frame
    pub ms: i64,
    pub view: ViewTree,
    pub view_digest: String,
    pub frame_digest: String,
}

/// The output struct a program emits a frame with.
pub const FRAME: &str = "Frame";

/// What the host derives for frames from the entries before one (HC-29).
#[derive(Clone, Debug, Default)]
pub struct Clock {
    /// frames derived so far
    pub count: u64,
    /// `Tick` entries since the previous frame
    pub ticks: u64,
}

impl Clock {
    /// Count the program entry `name` (call once per program entry).
    pub fn entry(&mut self, name: &str) {
        if name == "Tick" {
            self.ticks += 1;
        }
    }

    /// The frames among the outputs of the step at `seq`, in emit order:
    /// each `Frame` whose `view` is a tree, or why it is not one.
    pub fn frames(&mut self, seq: u64, input: &str, outs: &[V]) -> Vec<Result<Frame, String>> {
        let mut r = vec![];
        for o in outs {
            let V::Rec(n, fs) = o else { continue };
            if n != FRAME {
                continue;
            }
            r.push(self.one(seq, input, fs).map_err(|e| format!("seq {seq}: Frame is not delivered: {e}")));
        }
        r
    }

    fn one(&mut self, seq: u64, input: &str, fs: &[(String, V)]) -> Result<Frame, String> {
        let view = ViewTree::from_value(field(fs, "view")?)?;
        let ev = match fs.iter().find(|(k, _)| k == "ev") {
            Some((_, V::Text(t))) => t.to_string(),
            Some(_) => return Err("ev is not text".into()),
            None if self.count == 0 => "start".into(),
            None if input == "Tick" => "tick".into(),
            None => input.to_string(),
        };
        let ms = match fs.iter().find(|(k, _)| k == "ms") {
            Some((_, V::Int(i, _))) => i64::try_from(*i).map_err(|e| format!("ms: {e}"))?,
            Some(_) => return Err("ms is not an integer".into()),
            None if self.count == 0 => 0,
            None => (crate::cli::FRAME_MS * self.ticks) as i64,
        };
        let vd = view.digest();
        let fd = blake3(&frame_record(seq, &vd));
        self.count += 1;
        self.ticks = 0;
        Ok(Frame { seq, input: input.to_string(), ev, ms, view_digest: hex(&vd), frame_digest: hex(&fd), view })
    }
}

/// The sink lines of a sequence of frames (HC-29): one canonical JSON object
/// per frame, keys sorted, with the `ui.view.diff` operations from the
/// previous frame's view (`null` for the first).
#[derive(Default)]
pub struct Lines {
    prev: Option<ViewTree>,
}

impl Lines {
    /// The lines of the frames after one whose view is `prev`.
    pub fn after(prev: Option<&ViewTree>) -> Lines {
        Lines { prev: prev.cloned() }
    }

    pub fn line(&mut self, f: &Frame) -> Result<String, String> {
        let ops = match &self.prev {
            None => Value::Null,
            Some(a) => Value::Array(diff(a, &f.view)?),
        };
        self.prev = Some(f.view.clone());
        let v = json!({"ev": f.ev, "frame_digest": f.frame_digest, "input": f.input, "ms": f.ms, "ops": ops, "seq": f.seq, "view": f.view.json(), "view_digest": f.view_digest});
        Ok(serde_json::to_string(&v).map_err(|e| e.to_string())? + "\n")
    }
}

/// A frame read back from a sink line.
pub fn from_line(l: &str) -> Result<Frame, String> {
    let v: Value = serde_json::from_str(l).map_err(|e| e.to_string())?;
    let s = |k: &str| v[k].as_str().map(String::from).ok_or_else(|| format!("{k} is not text"));
    Ok(Frame {
        seq: v["seq"].as_u64().ok_or("seq")?,
        input: s("input")?,
        ev: s("ev")?,
        ms: v["ms"].as_i64().ok_or("ms")?,
        view: ViewTree::from_json(&v["view"])?,
        view_digest: s("view_digest")?,
        frame_digest: s("frame_digest")?,
    })
}

// ---------------- the reference library ----------------

const LIB: &str = "module rce_frames\nuse ui.view\nuse ui.rules\n";

fn lib() -> Result<&'static Module, String> {
    static M: std::sync::OnceLock<Result<Module, String>> = std::sync::OnceLock::new();
    M.get_or_init(|| crate::program(LIB)).as_ref().map_err(Clone::clone)
}

fn path(n: &str) -> Ty {
    Ty::Path(vec![n.to_string()], vec![])
}

fn call(f: &str, args: Vec<(Value, Ty)>) -> Result<V, String> {
    let m = lib()?;
    let mut it = Interp::new(m);
    it.scopes = vec![HashMap::new()];
    it.refuel();
    let mut a = vec![];
    for (v, t) in &args {
        a.push(Arg { label: None, e: crate::host::json_expr(v, Some(t), m)? });
    }
    let e = Expr::Call(Box::new(Expr::Name(f.to_string(), 0)), a, 0, false);
    it.eval(&e).map_err(|x| format!("{f}: {}", crate::interp::why(x)))
}

fn rec_fields(v: &V) -> Result<HashMap<String, Value>, String> {
    let V::Rec(_, fs) = v else { return Err("not a record".into()) };
    Ok(fs
        .iter()
        .map(|(k, x)| {
            let j = match x {
                V::Int(i, _) => json!(*i as i64),
                V::Text(t) => json!(t.to_string()),
                _ => Value::Null,
            };
            (k.clone(), j)
        })
        .collect())
}

/// `view.diff(a, b)`: the operations that turn a into b, in the order the
/// reference emits them, as {op, nd, parent, before, name, value}.
pub fn diff(a: &ViewTree, b: &ViewTree) -> Result<Vec<Value>, String> {
    let r = call("diff", vec![(a.json(), path("View")), (b.json(), path("View"))])?;
    const OPS: [&str; 7] = ["", "remove", "create", "move", "set", "unset", "text"];
    let mut out = vec![];
    for o in list(&r)? {
        let f = rec_fields(o)?;
        let k = f["op"].as_i64().and_then(|k| OPS.get(k as usize)).ok_or("op")?;
        out.push(json!({"before": f["before"], "name": f["name"], "nd": f["nd"], "op": k, "parent": f["parent"], "value": f["value"]}));
    }
    Ok(out)
}

fn trace(frames: &[Frame]) -> (Value, Ty) {
    let steps: Vec<Value> = frames.iter().map(|f| json!({"ev": f.ev, "ms": f.ms, "v": f.view.json()})).collect();
    (Value::Array(steps), Ty::Path(vec!["List".into()], vec![path("Step")]))
}

/// The `ui.rules` report of a session's frames as one trace
/// (`rules.findings` and `rules.report`): each finding as
/// (index of its frame, text), then each verdict's text.
pub struct Report {
    pub findings: Vec<(usize, String)>,
    pub verdicts: Vec<String>,
}

pub fn rules_report(frames: &[Frame]) -> Result<Report, String> {
    if frames.is_empty() {
        return Ok(Report { findings: vec![], verdicts: vec![] });
    }
    let fs = call("findings", vec![trace(frames)])?;
    let mut findings = vec![];
    for x in list(&fs)? {
        let f = rec_fields(x)?;
        let s = |k: &str| f[k].as_str().unwrap_or("").to_string();
        // a finding of the trace as a whole has frame -1
        let i = f["frame"].as_i64().and_then(|i| usize::try_from(i).ok()).unwrap_or(usize::MAX);
        let why = s("reason");
        findings.push((i, format!("{} {} at node {} ({}){}", s("sc"), s("outcome"), f["at"], s("rule"), if why.is_empty() { String::new() } else { format!(": {why}") })));
    }
    let vs = call("report", vec![trace(frames)])?;
    let mut verdicts = vec![];
    for x in list(&vs)? {
        let f = rec_fields(x)?;
        let s = |k: &str| f[k].as_str().unwrap_or("").to_string();
        let why = s("reason");
        verdicts.push(format!("{} {}{}", s("sc"), s("outcome"), if why.is_empty() { String::new() } else { format!(": {why}") }));
    }
    Ok(Report { findings, verdicts })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blake3_official_vectors() {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../stdlib/vectors/hash/blake3/test_vectors.json");
        let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let mut n = 0;
        for c in v["cases"].as_array().unwrap() {
            let len = c["input_len"].as_u64().unwrap() as usize;
            let input: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            assert_eq!(hex(&blake3(&input)), c["hash"].as_str().unwrap()[..64], "input_len {len}");
            n += 1;
        }
        assert!(n >= 30);
    }

    fn as_bytes(v: &V) -> Vec<u8> {
        list(v).unwrap().iter().map(|x| if let V::Int(i, _) = x { *i as u8 } else { panic!("{x:?}") }).collect()
    }

    /// The host's digests are the reference's: `view.view_digest` and
    /// `view.frame_digest` of `ui.view`, on a view with long texts, a
    /// non-ASCII text, many attributes and more than 23 children, so every
    /// CBOR head length occurs.
    #[test]
    fn digests_are_ui_views() {
        let n = 30;
        let mut v = ViewTree { tag: vec!["div".into()], key: vec!["".into()], text: vec!["".into()], names: vec![vec![]], values: vec![vec![]], parent: vec![-1], kids: vec![vec![]] };
        for i in 0..n {
            v.tag.push(if i % 3 == 0 { String::new() } else { "span".into() });
            v.key.push(if i % 3 == 0 { String::new() } else { format!("k{i}") });
            v.text.push(if i % 3 == 0 { "é漢".repeat(i * 10) } else { String::new() });
            let names: Vec<String> = if i % 3 == 0 { vec![] } else { (0..i).map(|j| format!("a{j:02}")).collect() };
            v.values.push(names.iter().map(|x| x.repeat(i)).collect());
            v.names.push(names);
            v.parent.push(0);
            v.kids.push(vec![]);
            v.kids[0].push(i as i64 + 1);
        }
        let r = call("view_digest", vec![(v.json(), path("View"))]).unwrap();
        assert_eq!(as_bytes(&r), v.digest());
        for seq in [0u64, 23, 24, 255, 256, 70000, 1 << 33] {
            let r = call("frame_digest", vec![(json!(seq), path("i64")), (v.json(), path("View"))]).unwrap();
            assert_eq!(as_bytes(&r), blake3(&frame_record(seq, &v.digest())), "seq {seq}");
        }
    }
}
