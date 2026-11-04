use crate::error::{Result, WasmError};
use crate::tensor_abi::{dtype_code, dtype_from_code, DESC_HEADER_BYTES};
use rce_vm_backend_cpu::tensor::{DType, HostTensor};
use rce_vm_backend_cpu::{InterpretError, LeafDispatch, ScalarValue, Value};
use std::collections::HashMap;
use std::sync::Mutex;
use wasmtime::{Caller, Config, Engine, Instance, Memory, Module, Store, Val, ValType};

/// v0.1 entry point: scalar-only.
pub const ENTRY_EXPORT: &str = "rce_dispatch";

/// v0.2 entry point: tensors via linear memory. See
/// `docs/spec/fg-wasm-leaf-abi.md`.
pub const ENTRY_EXPORT_TENSOR: &str = "rce_dispatch_tensor";

/// v0.2 allocator + deallocator.
pub const ALLOC_EXPORT: &str = "rce_alloc";
pub const FREE_EXPORT: &str = "rce_free";

/// Embeds Wasmtime and serves as a `LeafDispatch` for `language: "wasm"` (WAT
/// source) and `language: "wasm-hex"` (hex-encoded raw wasm bytes).
pub struct WasmExecutor {
    engine: Engine,
    /// Compiled module cache, keyed by content hash of the source bytes
    /// (post-WAT-expansion for the WAT path).
    cache: Mutex<HashMap<u64, Module>>,
}

impl WasmExecutor {
    pub fn new() -> Result<Self> {
        let mut cfg = Config::new();
        // Conservative defaults — strict, deterministic, no host imports.
        cfg.consume_fuel(false);
        cfg.wasm_threads(false);
        cfg.wasm_simd(true);
        cfg.wasm_bulk_memory(true);
        cfg.wasm_reference_types(true);
        let engine = Engine::new(&cfg).map_err(|e| WasmError::Engine(e.to_string()))?;
        Ok(Self {
            engine,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Compile + cache a module from raw wasm bytes.
    fn compile(&self, bytes: &[u8]) -> Result<Module> {
        let key = stable_hash(bytes);
        let mut cache = self.cache.lock().unwrap();
        if let Some(m) = cache.get(&key) {
            return Ok(m.clone());
        }
        let m = Module::new(&self.engine, bytes).map_err(|e| WasmError::Compile(e.to_string()))?;
        cache.insert(key, m.clone());
        Ok(m)
    }

    /// Invoke `rce_dispatch` from a WAT source with positional scalar args.
    pub fn invoke_wat(&self, wat_src: &str, args: &[ScalarValue]) -> Result<ScalarValue> {
        let wasm = wat::parse_str(wat_src).map_err(|e| WasmError::Wat(e.to_string()))?;
        self.invoke_bytes(&wasm, args)
    }

    /// v0.2 tensor protocol: invoke `rce_dispatch_tensor` over linear
    /// memory. Inputs and output cross the boundary as descriptors per
    /// `docs/spec/fg-wasm-leaf-abi.md`. The host-owned output tensor is
    /// returned as a fresh [`HostTensor`] (data is copied out of the
    /// module's memory).
    pub fn invoke_tensor_wat(
        &self,
        wat_src: &str,
        args: &[HostTensor],
    ) -> Result<HostTensor> {
        let wasm = wat::parse_str(wat_src).map_err(|e| WasmError::Wat(e.to_string()))?;
        self.invoke_tensor_bytes(&wasm, args)
    }

    pub fn invoke_tensor_bytes(
        &self,
        wasm: &[u8],
        args: &[HostTensor],
    ) -> Result<HostTensor> {
        let module = self.compile(wasm)?;
        let mut store: Store<()> = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &module, &[])
            .map_err(|e| WasmError::Instantiate(e.to_string()))?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| WasmError::MissingExport("memory".into()))?;
        let alloc = instance
            .get_typed_func::<i32, i32>(&mut store, ALLOC_EXPORT)
            .map_err(|_| WasmError::MissingExport(ALLOC_EXPORT.into()))?;
        let dispatch = instance
            .get_typed_func::<(i32, i32), i32>(&mut store, ENTRY_EXPORT_TENSOR)
            .map_err(|_| WasmError::MissingExport(ENTRY_EXPORT_TENSOR.into()))?;

        // Step 1: allocate + write each input descriptor + data buffer.
        let mut input_desc_ptrs: Vec<i32> = Vec::with_capacity(args.len());
        for t in args {
            let desc_ptr = self.write_tensor_to_memory(&mut store, &memory, &alloc, t)?;
            input_desc_ptrs.push(desc_ptr);
        }

        // Step 2: allocate the inputs array (count × u32 LE).
        let arr_size = (args.len() * 4) as i32;
        let arr_ptr = alloc
            .call(&mut store, arr_size)
            .map_err(|e| WasmError::Call(e.to_string()))?;
        if arr_ptr == 0 {
            return Err(WasmError::Call("rce_alloc returned 0 (OOM)".into()));
        }
        // Write the descriptor pointers.
        let mut arr_bytes = Vec::with_capacity(args.len() * 4);
        for p in &input_desc_ptrs {
            arr_bytes.extend_from_slice(&(*p as u32).to_le_bytes());
        }
        memory
            .write(&mut store, arr_ptr as usize, &arr_bytes)
            .map_err(|e| WasmError::Call(format!("write inputs array: {e}")))?;

        // Step 3: invoke.
        let out_ptr = dispatch
            .call(&mut store, (args.len() as i32, arr_ptr))
            .map_err(|e| WasmError::Call(e.to_string()))?;
        if out_ptr == 0 {
            return Err(WasmError::Call(
                "rce_dispatch_tensor returned 0 (module declined inputs)".into(),
            ));
        }

        // Step 4: read output descriptor + data.
        let out_tensor = self.read_tensor_from_memory(&store, &memory, out_ptr)?;
        Ok(out_tensor)
    }

    /// Write a HostTensor into the module's linear memory and return the
    /// descriptor pointer. Allocates two buffers via `rce_alloc`: the
    /// data buffer and the descriptor (header + shape).
    fn write_tensor_to_memory(
        &self,
        store: &mut Store<()>,
        memory: &Memory,
        alloc: &wasmtime::TypedFunc<i32, i32>,
        t: &HostTensor,
    ) -> Result<i32> {
        let elem = t.dtype.byte_size();
        let numel = t.shape.iter().product::<usize>().max(1);
        let data_size = (numel * elem) as i32;
        let desc_size = (DESC_HEADER_BYTES + 4 * t.shape.len()) as i32;

        let data_ptr = alloc
            .call(&mut *store, data_size.max(1))
            .map_err(|e| WasmError::Call(e.to_string()))?;
        if data_ptr == 0 {
            return Err(WasmError::Call(
                "rce_alloc returned 0 (data buffer)".into(),
            ));
        }
        let desc_ptr = alloc
            .call(&mut *store, desc_size)
            .map_err(|e| WasmError::Call(e.to_string()))?;
        if desc_ptr == 0 {
            return Err(WasmError::Call(
                "rce_alloc returned 0 (descriptor)".into(),
            ));
        }

        // Write the data buffer (handle 0-size case for 0-D tensors with
        // numel=1 — the data buffer is still 1 elem).
        if !t.data.is_empty() {
            memory
                .write(&mut *store, data_ptr as usize, &t.data)
                .map_err(|e| WasmError::Call(format!("write data: {e}")))?;
        }

        // Build and write the descriptor.
        let mut desc = Vec::with_capacity(desc_size as usize);
        desc.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        desc.extend_from_slice(&dtype_code(t.dtype).to_le_bytes());
        desc.extend_from_slice(&(data_ptr as u32).to_le_bytes());
        desc.extend_from_slice(&0u32.to_le_bytes()); // reserved
        for &dim in &t.shape {
            desc.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        memory
            .write(&mut *store, desc_ptr as usize, &desc)
            .map_err(|e| WasmError::Call(format!("write descriptor: {e}")))?;
        Ok(desc_ptr)
    }

    /// Read an output tensor from the module's linear memory. The host
    /// owns the returned bytes (copied out of the module).
    fn read_tensor_from_memory(
        &self,
        store: &Store<()>,
        memory: &Memory,
        desc_ptr: i32,
    ) -> Result<HostTensor> {
        let mem = memory.data(store);
        let base = desc_ptr as usize;
        if base + DESC_HEADER_BYTES > mem.len() {
            return Err(WasmError::Call(format!(
                "output descriptor at {desc_ptr} extends past linear memory ({} bytes)",
                mem.len()
            )));
        }
        let rank = u32::from_le_bytes(mem[base..base + 4].try_into().unwrap()) as usize;
        let dtype_c = u32::from_le_bytes(mem[base + 4..base + 8].try_into().unwrap());
        let data_ptr =
            u32::from_le_bytes(mem[base + 8..base + 12].try_into().unwrap()) as usize;

        let dtype = dtype_from_code(dtype_c)?;
        let shape_start = base + DESC_HEADER_BYTES;
        let shape_end = shape_start + 4 * rank;
        if shape_end > mem.len() {
            return Err(WasmError::Call(format!(
                "output shape (rank {rank}) extends past linear memory"
            )));
        }
        let mut shape: Vec<usize> = Vec::with_capacity(rank);
        for i in 0..rank {
            let off = shape_start + 4 * i;
            shape.push(u32::from_le_bytes(mem[off..off + 4].try_into().unwrap()) as usize);
        }
        let elem = dtype.byte_size();
        let numel: usize = shape.iter().product::<usize>().max(1);
        let data_bytes = numel * elem;
        if data_ptr + data_bytes > mem.len() {
            return Err(WasmError::Call(format!(
                "output data (data_ptr={data_ptr}, size={data_bytes}) extends past linear memory"
            )));
        }
        let data = mem[data_ptr..data_ptr + data_bytes].to_vec();
        HostTensor::new(shape, dtype, data).map_err(|e| {
            WasmError::Call(format!("read tensor host-side: {e}"))
        })
    }

    /// Invoke `rce_dispatch` from raw wasm bytes.
    pub fn invoke_bytes(&self, wasm: &[u8], args: &[ScalarValue]) -> Result<ScalarValue> {
        let module = self.compile(wasm)?;
        let mut store: Store<()> = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &module, &[])
            .map_err(|e| WasmError::Instantiate(e.to_string()))?;
        let func = instance
            .get_func(&mut store, ENTRY_EXPORT)
            .ok_or_else(|| WasmError::MissingExport(ENTRY_EXPORT.into()))?;
        let ty = func.ty(&store);
        let params: Vec<ValType> = ty.params().collect();
        if args.len() != params.len() {
            return Err(WasmError::ArityMismatch {
                expected: params.len(),
                got: args.len(),
            });
        }
        let mut wasm_args: Vec<Val> = Vec::with_capacity(args.len());
        for (a, pt) in args.iter().zip(&params) {
            wasm_args.push(scalar_to_val(a, pt)?);
        }
        let result_types: Vec<ValType> = ty.results().collect();
        let mut results: Vec<Val> = vec![Val::I32(0); result_types.len()];
        func.call(&mut store, &wasm_args, &mut results)
            .map_err(|e| WasmError::Call(e.to_string()))?;
        if results.len() != 1 {
            return Err(WasmError::ResultArity { got: results.len() });
        }
        val_to_scalar(&results[0])
    }
}

impl LeafDispatch for WasmExecutor {
    fn languages(&self) -> &[&'static str] {
        &["wasm", "wat", "wasm-hex"]
    }

    fn invoke(
        &self,
        _node_id: usize,
        language: &str,
        code_text: &str,
        _output_type: &str,
        args: &[Value],
    ) -> rce_vm_backend_cpu::Result<Value> {
        // Pick the protocol by argument type. Mixed scalar+tensor inputs
        // route through v0.2 (tensor); pure-scalar through v0.1 (scalar).
        // A DeviceTensor arg also routes through the tensor path so it
        // can hit the explicit Unsupported in the materialise step
        // rather than falling into the scalar path's unreachable!.
        let any_tensor = args
            .iter()
            .any(|v| matches!(v, Value::Tensor(_) | Value::DeviceTensor(_)));
        if any_tensor {
            // Promote any scalars to 0-D tensors so the v0.2 ABI sees a
            // homogeneous list of HostTensors.
            let tensors: Vec<HostTensor> = args
                .iter()
                .map(|v| match v {
                    Value::Tensor(t) => Ok(t.clone()),
                    Value::Scalar(s) => HostTensor::from_scalar(s).map_err(wasm_to_interp_owned),
                    Value::DeviceTensor(d) => Err(InterpretError::Unsupported(format!(
                        "wasm executor cannot consume DeviceTensor (device={}); \
                         materialise to host first.",
                        d.device.name()
                    ))),
                })
                .collect::<rce_vm_backend_cpu::Result<Vec<_>>>()?;
            let out = match language {
                "wasm" | "wat" => self
                    .invoke_tensor_wat(code_text, &tensors)
                    .map_err(wasm_to_interp)?,
                "wasm-hex" => {
                    let bytes = hex::decode(code_text.trim())
                        .map_err(|e| InterpretError::Internal(format!("wasm-hex decode: {e}")))?;
                    self.invoke_tensor_bytes(&bytes, &tensors)
                        .map_err(wasm_to_interp)?
                }
                other => {
                    return Err(InterpretError::Unsupported(format!(
                        "wasm executor: unknown language tag `{other}`"
                    )));
                }
            };
            return Ok(Value::Tensor(out));
        }
        // Scalar-only fast path (v0.1 protocol).
        let scalars: Vec<ScalarValue> = args
            .iter()
            .map(|v| match v {
                Value::Scalar(s) => s.clone(),
                Value::Tensor(_) | Value::DeviceTensor(_) => unreachable!(
                    "scalar fast path reached with non-scalar arg \
                     (the any_tensor branch should have handled this)"
                ),
            })
            .collect();
        let scalar = match language {
            "wasm" | "wat" => self.invoke_wat(code_text, &scalars).map_err(wasm_to_interp)?,
            "wasm-hex" => {
                let bytes = hex::decode(code_text.trim())
                    .map_err(|e| InterpretError::Internal(format!("wasm-hex decode: {e}")))?;
                self.invoke_bytes(&bytes, &scalars).map_err(wasm_to_interp)?
            }
            other => {
                return Err(InterpretError::Unsupported(format!(
                    "wasm executor: unknown language tag `{other}`"
                )));
            }
        };
        Ok(Value::Scalar(scalar))
    }
}

fn wasm_to_interp_owned(e: rce_vm_backend_cpu::InterpretError) -> InterpretError {
    e
}

fn scalar_to_val(s: &ScalarValue, ty: &ValType) -> Result<Val> {
    match (s, ty) {
        // Direct matches.
        (ScalarValue::I32(x), ValType::I32) => Ok(Val::I32(*x)),
        (ScalarValue::I64(x), ValType::I64) => Ok(Val::I64(*x)),
        (ScalarValue::F32(x), ValType::F32) => Ok(Val::F32(x.to_bits())),
        (ScalarValue::F64(x), ValType::F64) => Ok(Val::F64(x.to_bits())),
        (ScalarValue::Bool(b), ValType::I32) => Ok(Val::I32(*b as i32)),
        // Lossless / standard host-to-wasm coercions.
        (ScalarValue::I32(x), ValType::I64) => Ok(Val::I64(*x as i64)),
        (ScalarValue::F32(x), ValType::F64) => Ok(Val::F64((*x as f64).to_bits())),
        // Lossy but pragmatic — JSON literals like `1.5` parse as f64 but wasm
        // function may want f32. Convert via the natural f64→f32 cast.
        (ScalarValue::F64(x), ValType::F32) => Ok(Val::F32((*x as f32).to_bits())),
        (ScalarValue::I64(x), ValType::I32) => {
            let v = i32::try_from(*x).map_err(|_| {
                WasmError::UnsupportedValType(format!("i64 value {x} overflows i32 wasm param"))
            })?;
            Ok(Val::I32(v))
        }
        (s, t) => Err(WasmError::UnsupportedValType(format!(
            "cannot pass {} to wasm parameter {:?}",
            s.type_name(),
            t
        ))),
    }
}

fn val_to_scalar(v: &Val) -> Result<ScalarValue> {
    Ok(match v {
        Val::I32(x) => ScalarValue::I32(*x),
        Val::I64(x) => ScalarValue::I64(*x),
        Val::F32(bits) => ScalarValue::F32(f32::from_bits(*bits)),
        Val::F64(bits) => ScalarValue::F64(f64::from_bits(*bits)),
        other => {
            return Err(WasmError::UnsupportedValType(format!(
                "wasm return value not yet supported: {other:?}"
            )));
        }
    })
}

fn wasm_to_interp(e: WasmError) -> InterpretError {
    InterpretError::Internal(format!("wasm leaf: {e}"))
}

fn stable_hash(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

// hex import for the wasm-hex path.
extern crate hex as _hex;
use _hex as hex;

#[cfg(test)]
mod tests {
    use super::*;

    const ADD_WAT: &str = r#"
        (module
          (func $add (export "rce_dispatch") (param $a i32) (param $b i32) (result i32)
            local.get $a
            local.get $b
            i32.add))
    "#;

    const ADD_F32_WAT: &str = r#"
        (module
          (func $add (export "rce_dispatch") (param $a f32) (param $b f32) (result f32)
            local.get $a
            local.get $b
            f32.add))
    "#;

    #[test]
    fn invokes_i32_add() {
        let exec = WasmExecutor::new().unwrap();
        let out = exec
            .invoke_wat(ADD_WAT, &[ScalarValue::I32(3), ScalarValue::I32(4)])
            .unwrap();
        assert_eq!(out, ScalarValue::I32(7));
    }

    #[test]
    fn invokes_f32_add() {
        let exec = WasmExecutor::new().unwrap();
        let out = exec
            .invoke_wat(ADD_F32_WAT, &[ScalarValue::F32(1.5), ScalarValue::F32(2.25)])
            .unwrap();
        assert_eq!(out, ScalarValue::F32(3.75));
    }

    #[test]
    fn cache_hits_on_second_call() {
        let exec = WasmExecutor::new().unwrap();
        let _ = exec
            .invoke_wat(ADD_WAT, &[ScalarValue::I32(1), ScalarValue::I32(2)])
            .unwrap();
        let n_before = exec.cache.lock().unwrap().len();
        let _ = exec
            .invoke_wat(ADD_WAT, &[ScalarValue::I32(10), ScalarValue::I32(20)])
            .unwrap();
        let n_after = exec.cache.lock().unwrap().len();
        assert_eq!(n_before, n_after, "second call should hit cache");
    }

    #[test]
    fn arity_mismatch_errors() {
        let exec = WasmExecutor::new().unwrap();
        let err = exec
            .invoke_wat(ADD_WAT, &[ScalarValue::I32(1)])
            .unwrap_err();
        assert!(matches!(err, WasmError::ArityMismatch { expected: 2, got: 1 }));
    }

    // ── v0.2 tensor ABI ─────────────────────────────────────────────

    const TENSOR_ADD_WAT: &str = include_str!("../examples/elementwise_add_f32.wat");

    #[test]
    fn tensor_add_via_v02_abi() {
        let exec = WasmExecutor::new().unwrap();
        let a = HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let b = HostTensor::from_f32(vec![3], &[10.0, 20.0, 30.0]).unwrap();
        let out = exec.invoke_tensor_wat(TENSOR_ADD_WAT, &[a, b]).unwrap();
        assert_eq!(out.shape, vec![3]);
        assert_eq!(out.as_f32().unwrap(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn tensor_add_2d_via_v02_abi() {
        let exec = WasmExecutor::new().unwrap();
        let a = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = HostTensor::from_f32(
            vec![2, 3],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        .unwrap();
        let out = exec.invoke_tensor_wat(TENSOR_ADD_WAT, &[a, b]).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(
            out.as_f32().unwrap(),
            vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        );
    }

    #[test]
    fn tensor_add_module_declines_on_dtype_mismatch() {
        // The WAT module returns 0 (decline) when inputs are not f32.
        // The host turns 0 into a typed error.
        let exec = WasmExecutor::new().unwrap();
        let a = HostTensor::from_i32(vec![3], &[1, 2, 3]).unwrap();
        let b = HostTensor::from_i32(vec![3], &[10, 20, 30]).unwrap();
        let err = exec.invoke_tensor_wat(TENSOR_ADD_WAT, &[a, b]).unwrap_err();
        assert!(matches!(err, WasmError::Call(_)));
    }

    #[test]
    fn tensor_add_dispatched_via_leaf_trait() {
        // Exercise the LeafDispatch path (what the interpreter calls).
        let exec = WasmExecutor::new().unwrap();
        let a = HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let b = HostTensor::from_f32(vec![3], &[10.0, 20.0, 30.0]).unwrap();
        let args = [Value::Tensor(a), Value::Tensor(b)];
        let out = exec
            .invoke(0, "wat", TENSOR_ADD_WAT, "f32", &args)
            .unwrap();
        match out {
            Value::Tensor(t) => assert_eq!(t.as_f32().unwrap(), vec![11.0, 22.0, 33.0]),
            other => panic!("expected tensor output, got {other:?}"),
        }
    }
}
