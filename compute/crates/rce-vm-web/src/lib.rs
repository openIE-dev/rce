//! rce Graph VM — browser bindings.
//!
//! Compiles to `wasm32-unknown-unknown` and exposes a wasm-bindgen API that
//! ships as the `@rce/runtime` npm package. Same Rust code paths as the
//! native CLI: the CPU interpreter (`rce_vm_backend_cpu`) plus the wgpu
//! backend (`rce_vm_backend_wgpu`), which targets WebGPU in the browser
//! through wgpu's native web target.
//!
//! ## JS API (sketch)
//!
//! ```js
//! import init, { Vm, Module } from "@rce/runtime";
//!
//! await init();
//! const vm = new Vm();
//! await vm.initWgpu();  // optional; CPU-only otherwise
//!
//! const fgBytes = new Uint8Array(await (await fetch("/model.fg")).arrayBuffer());
//! const stBytes = new Uint8Array(await (await fetch("/model.safetensors")).arrayBuffer());
//! const module = Module.load(fgBytes, stBytes);
//!
//! const out = vm.run(module, JSON.stringify({ x: [[1, 2, 3, 4]] }));
//! console.log(JSON.parse(out));
//! ```
//!
//! Inputs and outputs cross the JS/Rust boundary as JSON strings — this
//! mirrors the CLI's `--input` flag and avoids serde-wasm-bindgen
//! polymorphism issues.

use rce_format::{graph_section, FgFile, SectionKind};
use rce_vm_backend_cpu::{
    interpret_with_config, DType, HostTensor, Inputs, InterpretConfig, ScalarValue, Value,
};
use rce_vm_backend_wgpu::WgpuExecutor;
use rce_weight_loader::SafetensorsFile;
use rce_graph::graph::{FunctionGraph, GraphNode};
use std::collections::HashMap;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn _start() {
    #[cfg(feature = "panic_hook")]
    console_error_panic_hook::set_once();
}

// ── Module ───────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct Module {
    graph: FunctionGraph,
}

#[wasm_bindgen]
impl Module {
    /// Parse a `.fg` byte buffer and (optionally) reattach weights from a
    /// safetensors byte buffer. Throws on malformed input.
    pub fn load(fg_bytes: &[u8], safetensors: Option<Vec<u8>>) -> Result<Module, JsValue> {
        let cursor = Cursor::new(fg_bytes);
        let file = FgFile::read(cursor).map_err(js_err)?;
        let section = file
            .section(SectionKind::Graph)
            .ok_or_else(|| js_err("module has no Graph section"))?;
        let mut graph: FunctionGraph =
            graph_section::graph_from_section(section).map_err(js_err)?;

        if let Some(bytes) = safetensors {
            let st = SafetensorsFile::from_bytes(&bytes).map_err(|e| js_err(format!("{e}")))?;
            reattach_weights(&mut graph, &st)?;
        }
        Ok(Module { graph })
    }

    /// Number of nodes in the graph.
    #[wasm_bindgen(getter)]
    pub fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Module's name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.graph.name.clone()
    }

    /// Comma-separated list of param names in declaration order.
    #[wasm_bindgen(getter)]
    pub fn params(&self) -> String {
        let parts: Vec<String> = self
            .graph
            .nodes
            .iter()
            .filter_map(|n| match n {
                GraphNode::Param(p) => Some(format!("{}: {}", p.name, p.data_type)),
                _ => None,
            })
            .collect();
        parts.join(", ")
    }
}

// ── Vm ───────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct Vm {
    wgpu: Option<WgpuExecutor>,
    cast_f32: bool,
}

#[wasm_bindgen]
impl Vm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Vm {
        Vm {
            wgpu: None,
            cast_f32: true, // default-on for browser; WebGPU is f32-only
        }
    }

    /// Initialise the WebGPU backend. Returns a Promise; resolves when the
    /// adapter + device are ready, rejects with a message if WebGPU is
    /// unavailable in this browser/host.
    ///
    /// CPU-only mode is the default if you skip this call.
    #[wasm_bindgen(js_name = initWgpu)]
    pub async fn init_wgpu(&mut self) -> Result<JsValue, JsValue> {
        let exec = WgpuExecutor::new_async()
            .await
            .map_err(|e| js_err(format!("wgpu init: {e}")))?;
        let info = exec.adapter_info.clone();
        self.wgpu = Some(exec);
        Ok(JsValue::from_str(&info))
    }

    /// Whether wgpu has been initialised.
    #[wasm_bindgen(js_name = wgpuReady)]
    pub fn wgpu_ready(&self) -> bool {
        self.wgpu.is_some()
    }

    /// Toggle automatic f64→f32 casting of input tensors. Default: true.
    /// (Required for wgpu, harmless on CPU.)
    #[wasm_bindgen(js_name = setCastF32)]
    pub fn set_cast_f32(&mut self, on: bool) {
        self.cast_f32 = on;
    }

    /// Run a module. Inputs are a JSON object; output is a JSON-encoded
    /// scalar/array. Errors propagate as exceptions.
    pub fn run(&self, module: &Module, input_json: &str) -> Result<String, JsValue> {
        let raw: HashMap<String, serde_json::Value> =
            serde_json::from_str(input_json).map_err(|e| js_err(format!("input JSON: {e}")))?;
        let mut inputs = Inputs::new();
        for (k, v) in raw {
            let mut value = json_to_value(&v).map_err(js_err)?;
            if self.cast_f32 {
                value = cast_value_to_f32(value).map_err(js_err)?;
            }
            inputs.insert(k, value);
        }

        let ops_storage: Vec<&dyn rce_vm_backend_cpu::OpDispatch> = match self.wgpu.as_ref() {
            Some(e) => vec![e],
            None => Vec::new(),
        };
        let cfg = InterpretConfig {
            leaves: &[],
            ops: &ops_storage,
        };
        let out = interpret_with_config(&module.graph, &inputs, cfg)
            .map_err(|e| js_err(format!("interpret: {e}")))?;
        Ok(value_to_json_string(&out))
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn js_err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&format!("{e}"))
}

fn reattach_weights(
    graph: &mut FunctionGraph,
    st: &SafetensorsFile,
) -> Result<usize, JsValue> {
    let mut count = 0usize;
    for (idx, node) in graph.nodes.iter_mut().enumerate() {
        if let GraphNode::TensorLiteral(td) = node {
            if !td.data.is_empty() {
                continue;
            }
            let loaded = st.get(&td.name).ok_or_else(|| {
                js_err(format!(
                    "weight `{}` (TensorLiteral at node {idx}) not present in sidecar",
                    td.name
                ))
            })?;
            if loaded.dtype != td.dtype {
                return Err(js_err(format!(
                    "weight `{}`: dtype mismatch (graph wants `{}`, sidecar has `{}`)",
                    td.name, td.dtype, loaded.dtype
                )));
            }
            if loaded.shape != td.shape {
                return Err(js_err(format!(
                    "weight `{}`: shape mismatch (graph wants {:?}, sidecar has {:?})",
                    td.name, td.shape, loaded.shape
                )));
            }
            td.data = loaded.data.to_arc();
            count += 1;
        }
    }
    Ok(count)
}

fn cast_value_to_f32(v: Value) -> Result<Value, String> {
    match v {
        Value::Tensor(t) => {
            let v: Vec<f32> = match t.dtype {
                DType::F64 => t
                    .as_f64()
                    .map_err(|e| format!("{e}"))?
                    .into_iter()
                    .map(|x| x as f32)
                    .collect(),
                DType::F32 => t.as_f32().map_err(|e| format!("{e}"))?,
                DType::I64 => t
                    .as_i64()
                    .map_err(|e| format!("{e}"))?
                    .into_iter()
                    .map(|x| x as f32)
                    .collect(),
                DType::I32 => t
                    .as_i32()
                    .map_err(|e| format!("{e}"))?
                    .into_iter()
                    .map(|x| x as f32)
                    .collect(),
                DType::Bool => return Err("--cast-f32 cannot cast bool tensor".into()),
                DType::F16 | DType::BF16 => {
                    return Err(format!("--cast-f32 from {:?} not yet implemented", t.dtype));
                }
            };
            Ok(Value::Tensor(
                HostTensor::from_f32(t.shape, &v).map_err(|e| format!("{e}"))?,
            ))
        }
        Value::Scalar(ScalarValue::F64(x)) => Ok(Value::Scalar(ScalarValue::F32(x as f32))),
        Value::Scalar(ScalarValue::I64(x)) => Ok(Value::Scalar(ScalarValue::F32(x as f32))),
        Value::Scalar(ScalarValue::I32(x)) => Ok(Value::Scalar(ScalarValue::F32(x as f32))),
        other => Ok(other),
    }
}

fn json_to_value(v: &serde_json::Value) -> Result<Value, String> {
    use serde_json::Value::*;
    match v {
        Bool(b) => Ok(Value::Scalar(ScalarValue::Bool(*b))),
        Number(n) => Ok(Value::Scalar(json_number_to_scalar(n)?)),
        String(s) => Ok(Value::Scalar(ScalarValue::String(s.clone()))),
        Array(arr) => Ok(Value::Tensor(json_array_to_tensor(arr)?)),
        Null => Err("unsupported JSON value: null".into()),
        Object(_) => Err(
            "JSON objects are not yet supported as input values; \
             pass scalars or arrays at the top level"
                .into(),
        ),
    }
}

fn json_number_to_scalar(n: &serde_json::Number) -> Result<ScalarValue, String> {
    if let Some(i) = n.as_i64() {
        if i32::try_from(i).is_ok() {
            return Ok(ScalarValue::I32(i as i32));
        }
        return Ok(ScalarValue::I64(i));
    }
    if let Some(f) = n.as_f64() {
        return Ok(ScalarValue::F64(f));
    }
    Err(format!("unsupported JSON number: {n}"))
}

fn json_array_to_tensor(arr: &[serde_json::Value]) -> Result<HostTensor, String> {
    let shape = infer_shape(arr)?;
    let leaves = collect_leaves(arr);
    if leaves.is_empty() {
        return Err("empty tensor input not supported".into());
    }
    let first = &leaves[0];
    let dtype = match first {
        serde_json::Value::Bool(_) => DType::Bool,
        serde_json::Value::Number(n) => {
            if n.is_f64() {
                DType::F64
            } else if let Some(i) = n.as_i64() {
                if i32::try_from(i).is_ok() {
                    DType::I32
                } else {
                    DType::I64
                }
            } else {
                return Err(format!("unsupported JSON number in tensor: {n}"));
            }
        }
        other => return Err(format!("tensor leaves must be numeric or bool, got {other}")),
    };
    match dtype {
        DType::F64 => {
            let v: Vec<f64> = leaves
                .iter()
                .map(|l| l.as_f64().ok_or_else(|| "non-f64 leaf".to_string()))
                .collect::<Result<_, _>>()?;
            HostTensor::from_f64(shape, &v).map_err(|e| format!("{e}"))
        }
        DType::I64 => {
            let v: Vec<i64> = leaves
                .iter()
                .map(|l| l.as_i64().ok_or_else(|| "non-i64 leaf".to_string()))
                .collect::<Result<_, _>>()?;
            HostTensor::from_i64(shape, &v).map_err(|e| format!("{e}"))
        }
        DType::I32 => {
            let v: Vec<i32> = leaves
                .iter()
                .map(|l| {
                    let i = l.as_i64().ok_or_else(|| "non-int leaf".to_string())?;
                    i32::try_from(i).map_err(|_| "i32 overflow".to_string())
                })
                .collect::<Result<_, _>>()?;
            HostTensor::from_i32(shape, &v).map_err(|e| format!("{e}"))
        }
        DType::Bool => {
            let bytes: Vec<u8> = leaves
                .iter()
                .map(|l| l.as_bool().map(|b| b as u8).ok_or_else(|| "non-bool leaf".to_string()))
                .collect::<Result<_, _>>()?;
            HostTensor::new(shape, DType::Bool, bytes).map_err(|e| format!("{e}"))
        }
        DType::F32 => unreachable!("dtype inference never produces F32 from JSON"),
        DType::F16 | DType::BF16 => unreachable!("dtype inference never produces F16/BF16 from JSON"),
    }
}

fn infer_shape(arr: &[serde_json::Value]) -> Result<Vec<usize>, String> {
    if arr.is_empty() {
        return Ok(vec![0]);
    }
    if !matches!(arr[0], serde_json::Value::Array(_)) {
        return Ok(vec![arr.len()]);
    }
    let serde_json::Value::Array(first_child) = &arr[0] else {
        unreachable!()
    };
    let inner = infer_shape(first_child)?;
    for child in &arr[1..] {
        let serde_json::Value::Array(c) = child else {
            return Err("ragged JSON tensor: expected nested array".into());
        };
        let s = infer_shape(c)?;
        if s != inner {
            return Err("ragged JSON tensor: inner shapes differ".into());
        }
    }
    let mut shape = vec![arr.len()];
    shape.extend(inner);
    Ok(shape)
}

fn collect_leaves<'a>(arr: &'a [serde_json::Value]) -> Vec<&'a serde_json::Value> {
    let mut out = Vec::new();
    for v in arr {
        match v {
            serde_json::Value::Array(child) => out.extend(collect_leaves(child)),
            other => out.push(other),
        }
    }
    out
}

fn value_to_json_string(v: &Value) -> String {
    match v {
        Value::Scalar(s) => match s {
            ScalarValue::Bool(b) => b.to_string(),
            ScalarValue::I32(x) => x.to_string(),
            ScalarValue::I64(x) => x.to_string(),
            ScalarValue::F32(x) => format!("{x}"),
            ScalarValue::F64(x) => format!("{x}"),
            ScalarValue::String(s) => format!("{s:?}"),
        },
        Value::Tensor(t) => tensor_to_json(t),

        // DeviceTensor lives in accelerator memory (GPU/NPU/WebGPU); its
        // bytes are not host-readable without an async device→host transfer.
        // This helper is sync, and the interpreter currently has no
        // `DeviceTensor::to_host()` materialiser, so we return a JSON
        // descriptor instead of attempting to serialise contents. The
        // frontend can still render shape/dtype/residency/size and the
        // user can request a readback explicitly if needed. When a
        // `to_host()` lands on `DeviceTensor`, change this arm to call it
        // and fall through to `tensor_to_json`.
        Value::DeviceTensor(d) => {
            let shape_str: Vec<String> = d.shape.iter().map(|n| n.to_string()).collect();
            format!(
                "{{\"residency\":\"device\",\"device\":\"{}\",\"dtype\":\"{}\",\"shape\":[{}],\"size_bytes\":{},\"ptr\":{}}}",
                d.device.name(),
                d.dtype.name(),
                shape_str.join(","),
                d.mem.size_bytes(),
                d.mem.ptr(),
            )
        }
    }
}

fn tensor_to_json(t: &HostTensor) -> String {
    fn nest<T: std::fmt::Display>(shape: &[usize], data: &[T]) -> String {
        if shape.len() <= 1 {
            let inner: Vec<String> = data.iter().map(|x| format!("{x}")).collect();
            return format!("[{}]", inner.join(", "));
        }
        let chunk: usize = shape[1..].iter().product();
        let n = shape[0];
        let parts: Vec<String> = (0..n)
            .map(|i| nest(&shape[1..], &data[i * chunk..(i + 1) * chunk]))
            .collect();
        format!("[{}]", parts.join(", "))
    }
    match t.dtype {
        DType::F64 => nest(&t.shape, &t.as_f64().unwrap_or_default()),
        DType::F32 => nest(&t.shape, &t.as_f32().unwrap_or_default()),
        DType::I64 => nest(&t.shape, &t.as_i64().unwrap_or_default()),
        DType::I32 => nest(&t.shape, &t.as_i32().unwrap_or_default()),
        DType::Bool => nest(&t.shape, &t.as_bool().unwrap_or_default()),
        DType::F16 | DType::BF16 => format!("[{:?} display not implemented]", t.dtype),
    }
}
