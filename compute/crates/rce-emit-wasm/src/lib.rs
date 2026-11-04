//! `rce emit wasm` (spec 10.18): a graph term as WebAssembly.
//!
//! Two modules realise a term. The kernel library ([`KERNELS`], built from
//! `rce-wasm-kernels` by `scripts/build-wasm-kernels.sh` and pinned by its
//! BLAKE3 digest, [`KERNELS_DIGEST`]) is the reference node step of
//! `rce-vm-backend-cpu` compiled to WebAssembly. The program module this
//! crate prints holds, for each node of each term, a descriptor (a passive
//! data segment: the node, its operand slots, its level) and one call of
//! the library's `rce_node` on it, in the order of the reference
//! evaluation (EW-3). It imports the library under the module name
//! `rce-kernels/DIGEST`, so it links only to the library it was emitted
//! for (EW-2).
//!
//! Slots: parameter `i` is put in slot `i`; the value of node `n` is in
//! slot [`NODE_SLOT`]` + n`; the bytes of the blob at node `n` in slot
//! [`BLOB_SLOT`]` + n`; the result in slot [`OUT_SLOT`]. A term's export
//! `term_KEY` returns 0, or `(n + 1) << 8 | status` for the first node `n`
//! that did not complete (statuses of `rce-wasm-kernels`).

use rce_graph::graph::{FunctionGraph, GraphNode, LiteralValue, OpKind};
use rce_graph::term::{Level, TermTypes};
use std::fmt::Write;

/// The kernel library, `wasm32-unknown-unknown`.
pub const KERNELS: &[u8] = include_bytes!("../kernels/rce-kernels.wasm");
/// The BLAKE3 digest of [`KERNELS`], as `kernels/rce-kernels.wasm.blake3` records it.
pub const KERNELS_DIGEST: &str = include_str!("../kernels/rce-kernels.wasm.blake3");

pub const NODE_SLOT: u32 = 1_000_000;
pub const BLOB_SLOT: u32 = 2_000_000;
pub const OUT_SLOT: u32 = 3_000_000;
pub const LIT_SLOT: u32 = 4_000_000;

/// The module name the program imports the library under.
pub fn kernels_module() -> String {
    format!("rce-kernels/{}", KERNELS_DIGEST.trim())
}

/// One graph term to emit: its key (the content hash of its `.fg`), its
/// graph and its typing.
pub struct TermIn<'a> {
    pub key: &'a str,
    pub graph: &'a FunctionGraph,
    pub types: &'a TermTypes,
}

/// What a host must provide before it calls a term's export: the blobs,
/// each by the slot it goes in and its digest.
#[derive(Debug, Clone, PartialEq)]
pub struct Needs {
    pub blobs: Vec<(u32, String)>,
}

/// The blob node `i` of `g`, by its digest: an `Apply` of `blob` fed by a
/// text literal (6.13, GT-9).
fn blob_digest(g: &FunctionGraph, i: usize) -> Option<String> {
    if !matches!(&g.nodes[i], GraphNode::Apply { op: OpKind::Custom(c), .. } if c == "blob") {
        return None;
    }
    let w = g.wires.iter().find(|w| w.to == i)?;
    match &g.nodes[w.from] {
        GraphNode::Literal { value: LiteralValue::String(d), .. } => Some(d.clone()),
        _ => None,
    }
}

/// The blobs term `g` reads, as [`Needs`].
pub fn needs(g: &FunctionGraph) -> Needs {
    Needs { blobs: (0..g.nodes.len()).filter_map(|i| blob_digest(g, i).map(|d| (BLOB_SLOT + i as u32, d))).collect() }
}

fn level_of(types: &TermTypes, n: usize) -> Option<Level> {
    types.levels.iter().find(|l| l.node as usize == n).map(|l| l.level)
}

/// The tensor encoding of a tensor literal's data (`rce-wasm-kernels`).
fn literal_encoding(dtype: &str, shape: &[usize], data: &[u8]) -> Result<Vec<u8>, String> {
    let code: u32 = match dtype {
        "f32" => 0,
        "f64" => 1,
        "i32" => 2,
        d => return Err(format!("a tensor literal of {d}")),
    };
    let mut out = vec![];
    out.extend(code.to_le_bytes());
    out.extend((shape.len() as u32).to_le_bytes());
    for d in shape {
        out.extend((*d as u32).to_le_bytes());
    }
    out.extend(data);
    Ok(out)
}

fn wat_string(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() + 2);
    s.push('"');
    for &c in b {
        if (0x20..0x7f).contains(&c) && c != b'"' && c != b'\\' {
            s.push(c as char);
        } else {
            let _ = write!(s, "\\{c:02x}");
        }
    }
    s.push('"');
    s
}

/// The node's operand slots, in port order.
fn operand_slots(g: &FunctionGraph, i: usize) -> Vec<u32> {
    let mut ws: Vec<_> = g.wires.iter().filter(|w| w.to == i).collect();
    ws.sort_by_key(|w| w.port_in_idx);
    ws.iter().map(|w| NODE_SLOT + w.from as u32).collect()
}

/// The WAT text of the program module of `terms` (EW-1): one export
/// `term_KEY` per term.
pub fn emit_program(terms: &[TermIn]) -> Result<String, String> {
    let k = kernels_module();
    let mut data = String::new();
    let mut funcs = String::new();
    let mut levels = serde_json::Map::new();
    let mut seg = 0usize;
    for t in terms {
        let g = t.graph;
        // the reference evaluates the graph with each blob node replaced by
        // its tensor (no operand), so its order is that graph's
        let mut og = g.clone();
        og.wires.retain(|w| blob_digest(g, w.to).is_none());
        let order = rce_vm_backend_cpu::reference_order(&og).map_err(|e| format!("term {}: {e}", t.key))?;
        let mut body = String::new();
        let mut lv = vec![];
        for n in order {
            let node = &g.nodes[n];
            let level = level_of(t.types, n);
            if let Some(l) = level {
                lv.push(serde_json::json!([n, l.text()]));
            }
            let mut desc = serde_json::json!({"node": n, "out": NODE_SLOT + n as u32, "level": level.map(|l| l.text())});
            let what;
            match node {
                GraphNode::Param(p) => {
                    let i: u32 = p.name.strip_prefix('p').and_then(|x| x.parse().ok()).ok_or_else(|| format!("term {}: a parameter named `{}`", t.key, p.name))?;
                    desc["args"] = serde_json::json!([i]);
                    desc["run"] = serde_json::json!("value");
                    what = format!("parameter {i}");
                }
                GraphNode::TensorLiteral(td) => {
                    let enc = literal_encoding(&td.dtype, &td.shape, &td.data)?;
                    let _ = writeln!(data, "  (data $d{seg} {})", wat_string(&enc));
                    let _ = writeln!(
                        body,
                        "    (local.set $p (call $alloc (i32.const {len})))\n    (memory.init $d{seg} (local.get $p) (i32.const 0) (i32.const {len}))\n    (local.set $s (call $put (i32.const {slot}) (local.get $p) (i32.const {len})))\n    (if (local.get $s) (then (return (i32.or (i32.const {fail}) (local.get $s)))))",
                        len = enc.len(),
                        slot = LIT_SLOT + n as u32,
                        fail = (n + 1) << 8
                    );
                    seg += 1;
                    desc["args"] = serde_json::json!([LIT_SLOT + n as u32]);
                    desc["run"] = serde_json::json!("value");
                    what = format!("tensor literal {}", td.dtype);
                }
                GraphNode::Apply { output_type, .. } if blob_digest(g, n).is_some() => {
                    let d = blob_digest(g, n).unwrap_or_default();
                    let (elem, dims) = output_type.split_once('[').ok_or_else(|| format!("term {}: a blob of type `{output_type}`", t.key))?;
                    let shape: Vec<usize> = dims.trim_end_matches(']').split(',').filter(|x| !x.is_empty()).map(|x| x.parse().map_err(|_| format!("a blob of type `{output_type}`"))).collect::<Result<_, _>>()?;
                    desc["args"] = serde_json::json!([BLOB_SLOT + n as u32]);
                    desc["run"] = serde_json::json!({"blob": {"digest": d, "elem": elem, "shape": shape}});
                    what = format!("blob {d}");
                }
                GraphNode::Apply { op, .. } => {
                    desc["args"] = serde_json::json!(operand_slots(g, n));
                    desc["run"] = serde_json::json!({"node": node});
                    what = match op {
                        OpKind::Custom(c) => c.clone(),
                        o => format!("{o:?}").to_lowercase(),
                    };
                }
                GraphNode::Literal { .. } => {
                    desc["run"] = serde_json::json!({"node": node});
                    what = "literal".into();
                }
                other => return Err(format!("term {}: node {n} is not a node of a graph term ({other:?})", t.key)),
            }
            let bytes = serde_json::to_vec(&desc).map_err(|e| e.to_string())?;
            let _ = writeln!(data, "  (data $d{seg} {})", wat_string(&bytes));
            let lvl = level.map(|l| format!(", {}", l.text())).unwrap_or_default();
            let _ = writeln!(
                body,
                "    ;; node {n}: {what}{lvl}\n    (local.set $p (call $alloc (i32.const {len})))\n    (memory.init $d{seg} (local.get $p) (i32.const 0) (i32.const {len}))\n    (local.set $s (call $node (local.get $p) (i32.const {len})))\n    (if (local.get $s) (then (return (i32.or (i32.const {fail}) (local.get $s)))))",
                len = bytes.len(),
                fail = (n + 1) << 8
            );
            seg += 1;
        }
        levels.insert(t.key.to_string(), serde_json::Value::Array(lv));
        let _ = writeln!(
            funcs,
            "  (func (export \"term_{key}\") (result i32) (local $p i32) (local $s i32)\n{body}    (call $copy (i32.const {from}) (i32.const {OUT_SLOT})))",
            key = t.key,
            from = NODE_SLOT + g.output_node as u32
        );
    }
    let lv = serde_json::to_vec(&serde_json::Value::Object(levels)).map_err(|e| e.to_string())?;
    let mut out = String::new();
    let _ = writeln!(out, ";; rce emit wasm: one call of the kernel library per node, in the reference order (spec 10.18)");
    let _ = writeln!(out, "(module");
    let _ = writeln!(out, "  (import \"{k}\" \"memory\" (memory 0))");
    let _ = writeln!(out, "  (import \"{k}\" \"rce_alloc\" (func $alloc (param i32) (result i32)))");
    let _ = writeln!(out, "  (import \"{k}\" \"rce_put\" (func $put (param i32 i32 i32) (result i32)))");
    let _ = writeln!(out, "  (import \"{k}\" \"rce_node\" (func $node (param i32 i32) (result i32)))");
    let _ = writeln!(out, "  (import \"{k}\" \"rce_copy\" (func $copy (param i32 i32) (result i32)))");
    out.push_str(&funcs);
    out.push_str(&data);
    let _ = writeln!(out, "  (@custom \"rce.levels\" {})", wat_string(&lv));
    out.push_str(")\n");
    Ok(out)
}
