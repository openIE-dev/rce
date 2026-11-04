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
