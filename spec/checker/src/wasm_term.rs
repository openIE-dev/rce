//! `rce emit wasm` and `rce graph run wasm` (spec 10.18, EW-1 to EW-6):
//! a graph term as a WebAssembly program module that calls the kernel
//! library once per node, run under wasmi.

use crate::ast::Module;
use crate::graph::{TTy, TRAP_NON_FINITE, of_host, terms_of, to_host};
use crate::interp::V;
use rce_emit_wasm::{KERNELS, KERNELS_DIGEST, NODE_SLOT, OUT_SLOT, TermIn, emit_program, kernels_module, needs};
use rce_graph::graph::FunctionGraph;
use rce_graph::term::TermTypes;
use rce_vm_backend_cpu::HostTensor;

/// The WAT text of the program module of every graph term of the admitted
/// module `m` (EW-1). A module without graph terms requires change.
pub fn emit(m: &Module) -> Result<String, String> {
    let ts = terms_of(m);
    if ts.is_empty() {
        return Err("requires change: `wasm` prints the graph terms of a program, and this program has none (6.13)".into());
    }
    let ins: Vec<TermIn> = ts.iter().map(|t| TermIn { key: &t.key, graph: &t.graph, types: &t.types }).collect();
    emit_program(&ins).map_err(|e| format!("internal: {e}"))
}

/// A run of one term under the program module: its result (Err the trap
/// text, as [`crate::graph::evaluate`]), and the value of every node that
/// completed, by node index (EW-4).
pub struct WasmRun {
    pub result: Result<V, String>,
    pub nodes: Vec<(usize, HostTensor)>,
}

fn engine() -> Result<wasmi::Engine, String> {
    let mut c = wasmi::Config::default();
    c.floats(true);
    let e = wasmi::Engine::new(&c);
    // HC-13: floats run under this engine only when the probe agrees with
    // the reference bit for bit on this host
    let (ok, exact) = crate::wasm_float::probe(&e);
    if !ok {
        return Err(format!("graph term: the WebAssembly engine's floats differ from the reference on this host (HC-13): {exact}"));
    }
    Ok(e)
}

/// Run the term `key` (`graph`, `types`) on `vals` through its program
/// module and the kernel library (EW-2 to EW-5).
pub fn run(key: &str, graph: &FunctionGraph, types: &TermTypes, vals: &[V]) -> Result<WasmRun, String> {
    run_with(key, graph, types, vals, &|d| crate::graph::blob_bytes(d))
}

/// [`run`], with the bytes of each blob from `blob` (by digest). The
/// program module checks them against the digest before the node uses
/// them (EW-5), whatever `blob` returns.
pub fn run_with(key: &str, graph: &FunctionGraph, types: &TermTypes, vals: &[V], blob: &dyn Fn(&str) -> Result<std::sync::Arc<[u8]>, String>) -> Result<WasmRun, String> {
    use wasmi::{Linker, Memory, Module as WModule, Store, TypedFunc};
    if blake3_hex(KERNELS) != KERNELS_DIGEST.trim() {
        return Err("internal: the kernel library does not hash to its pin (EW-2)".into());
    }
    if vals.len() != types.params.len() {
        return Err(format!("graph term: {} operands for {} parameters", vals.len(), types.params.len()));
    }
    let wat = emit_program(&[TermIn { key, graph, types }]).map_err(|e| format!("graph term: {e}"))?;
    let prog = wat::parse_str(&wat).map_err(|e| format!("internal: the program module does not assemble: {e}"))?;
    let e = engine()?;
    let km = WModule::new(&e, KERNELS).map_err(|x| format!("internal: kernel library: {x}"))?;
    let pm = WModule::new(&e, &prog[..]).map_err(|x| format!("internal: program module: {x}"))?;
    let mut st = Store::new(&e, ());
    let mut lk = Linker::<()>::new(&e);
    let ki = lk.instantiate_and_start(&mut st, &km).map_err(|x| format!("internal: kernel library: {x}"))?;
    lk.instance(&mut st, &kernels_module(), ki).map_err(|x| format!("internal: {x}"))?;
    let pi = lk.instantiate_and_start(&mut st, &pm).map_err(|x| format!("internal: program module: {x}"))?;
    let mem: Memory = ki.get_memory(&st, "memory").ok_or("internal: the kernel library exports no memory")?;
    let alloc: TypedFunc<i32, i32> = ki.get_typed_func(&st, "rce_alloc").map_err(|x| format!("internal: {x}"))?;
    let put: TypedFunc<(i32, i32, i32), i32> = ki.get_typed_func(&st, "rce_put").map_err(|x| format!("internal: {x}"))?;
    let put_bytes: TypedFunc<(i32, i32, i32), i32> = ki.get_typed_func(&st, "rce_put_bytes").map_err(|x| format!("internal: {x}"))?;
    let get: TypedFunc<i32, i32> = ki.get_typed_func(&st, "rce_get").map_err(|x| format!("internal: {x}"))?;
    let get_len: TypedFunc<(), i32> = ki.get_typed_func(&st, "rce_get_len").map_err(|x| format!("internal: {x}"))?;
    let msg: TypedFunc<(), i32> = ki.get_typed_func(&st, "rce_msg").map_err(|x| format!("internal: {x}"))?;
    let msg_len: TypedFunc<(), i32> = ki.get_typed_func(&st, "rce_msg_len").map_err(|x| format!("internal: {x}"))?;
    let tr = |x: wasmi::Error| format!("internal: a trap of the kernel library: {x}");
    let mut send = |st: &mut Store<()>, slot: u32, b: &[u8], raw: bool| -> Result<(), String> {
        let p = alloc.call(&mut *st, b.len() as i32).map_err(tr)?;
        mem.write(&mut *st, p as usize, b).map_err(|x| format!("internal: {x}"))?;
        let s = if raw { put_bytes.call(&mut *st, (slot as i32, p, b.len() as i32)) } else { put.call(&mut *st, (slot as i32, p, b.len() as i32)) }.map_err(tr)?;
        if s != 0 { Err(format!("internal: slot {slot} not taken ({s})")) } else { Ok(()) }
    };
    for (i, (v, t)) in vals.iter().zip(&types.params).enumerate() {
        let tt = TTy::of_spec(t).ok_or("graph term: a parameter of an unknown element type")?;
        let h = to_host(v, &tt)?;
        send(&mut st, i as u32, &rce_wasm_kernels::encode(&h).ok_or("graph term: a parameter of an unknown element type")?, false)?;
    }
    for (slot, d) in needs(graph).blobs {
        let b = blob(&d)?;
        send(&mut st, slot, &b, true)?;
    }
    let f: TypedFunc<(), i32> = pi.get_typed_func(&st, &format!("term_{key}")).map_err(|x| format!("internal: {x}"))?;
    let status = f.call(&mut st, ()).map_err(tr)?;
    let mut read = |st: &mut Store<()>, slot: u32| -> Result<Option<HostTensor>, String> {
        let p = get.call(&mut *st, slot as i32).map_err(tr)? as usize;
        let n = get_len.call(&mut *st, ()).map_err(tr)? as usize;
        if n == 0 {
            return Ok(None);
        }
        let mut b = vec![0u8; n];
        mem.read(&*st, p, &mut b).map_err(|x| format!("internal: {x}"))?;
        rce_wasm_kernels::decode(&b).map(Some)
    };
    let mut nodes = vec![];
    for n in 0..graph.nodes.len() {
        if let Some(h) = read(&mut st, NODE_SLOT + n as u32)? {
            nodes.push((n, h));
        }
    }
    let result = if status == 0 {
        let rt = TTy::of_spec(&types.result).ok_or("graph term: a result of an unknown element type")?;
        let h = read(&mut st, OUT_SLOT)?.ok_or("graph term: a scalar result")?;
        of_host(&h, &rt)
    } else {
        let p = msg.call(&mut st, ()).map_err(tr)? as usize;
        let n = msg_len.call(&mut st, ()).map_err(tr)? as usize;
        let mut b = vec![0u8; n];
        mem.read(&st, p, &mut b).map_err(|x| format!("internal: {x}"))?;
        let text = String::from_utf8_lossy(&b).into_owned();
        match status & 0xff {
            1 => Err(TRAP_NON_FINITE.to_string()),
            2 => Err(text),
            4 => Err(format!("graph term: {text}")),
            _ => Err(format!("graph term: node {}: {text}", (status >> 8) - 1)),
        }
    };
    Ok(WasmRun { result, nodes })
}

fn blake3_hex(b: &[u8]) -> String {
    rce_fgm::hex(&rce_fgm::digest(b))
}

fn ulps64(a: f64, b: f64) -> u64 {
    let k = |x: f64| {
        let b = x.to_bits() as i64;
        if b < 0 { i64::MIN - b } else { b }
    };
    k(a).abs_diff(k(b))
}

fn ulps32(a: f32, b: f32) -> u64 {
    let k = |x: f32| {
        let b = x.to_bits() as i32 as i64;
        if b < 0 { i32::MIN as i64 - b } else { b }
    };
    k(a).abs_diff(k(b))
}

/// Whether `got` is `want` at `level` (EW-4): exact, bit for bit; within
/// N, at most N units in the last place; lossy, of the same type.
pub fn at_level(got: &HostTensor, want: &HostTensor, level: rce_graph::term::Level) -> bool {
    use rce_graph::term::Level;
    use rce_vm_backend_cpu::DType;
    if (got.dtype, &got.shape) != (want.dtype, &want.shape) {
        return false;
    }
    match level {
        Level::Exact => got == want,
        Level::Lossy => true,
        Level::Within { ulps } => match got.dtype {
            DType::F64 => match (got.as_f64(), want.as_f64()) {
                (Ok(a), Ok(b)) => a.iter().zip(b.iter()).all(|(x, y)| ulps64(*x, *y) <= ulps as u64),
                _ => false,
            },
            DType::F32 => match (got.as_f32(), want.as_f32()) {
                (Ok(a), Ok(b)) => a.iter().zip(b.iter()).all(|(x, y)| ulps32(*x, *y) <= ulps as u64),
                _ => false,
            },
            _ => got == want,
        },
    }
}

/// HC-39: the probe of a graph term on the wasm fabric (HC-13). The
/// reference digest is the BLAKE3 of the reference evaluation's completed
/// nodes and its outcome; the probe digest is the same text built from the
/// wasm run, listing a node only when its value is the reference's at the
/// node's level. The two are equal exactly when the same nodes completed,
/// every one at its level, and the outcome (value or trap text) agrees.
pub fn probe(t: &crate::graph::Term, vals: &[V]) -> Result<(String, String), String> {
    let (rr, rn) = crate::graph::reference_nodes(&t.graph, &t.types, vals)?;
    let w = run(&t.key, &t.graph, &t.types, vals)?;
    let level = |n: usize| t.types.levels.iter().find(|l| l.node as usize == n).map(|l| l.level).unwrap_or(rce_graph::term::Level::Exact);
    let want: std::collections::BTreeMap<usize, &HostTensor> = rn.iter().map(|(n, h)| (*n, h)).collect();
    let outcome = |r: &Result<V, String>| match r {
        Ok(_) => "value".to_string(),
        Err(m) => format!("trap: {m}"),
    };
    let text = |nodes: Vec<usize>, o: String| serde_json::json!({"nodes": nodes, "outcome": o}).to_string();
    let reference = text(want.keys().copied().collect(), outcome(&rr));
    let mut ok: Vec<usize> = w.nodes.iter().filter(|(n, h)| want.get(n).is_some_and(|r| at_level(h, r, level(*n)))).map(|x| x.0).collect();
    ok.sort();
    let got = text(ok, outcome(&w.result));
    Ok((blake3_hex(got.as_bytes()), blake3_hex(reference.as_bytes())))
}

/// HC-39: the wasm fabric of graph-term placement.
pub fn fabric() -> crate::interp::GraphFabric {
    crate::interp::GraphFabric { run: |t, v| run(&t.key, &t.graph, &t.types, v).map(|w| w.result), probe }
}
