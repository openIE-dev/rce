//! rce Wasm leaf backend.
//!
//! Implements the [`rce_vm_backend_cpu::LeafDispatch`] trait by embedding
//! Wasmtime. When the interpreter reaches a
//! `GraphNode::CodeBlock { language: "wasm" | "wat", code_text }` node,
//! this backend:
//!
//! 1. Compiles the WAT source (or wasm bytes via the `language: "wasm-hex"`
//!    convention — pass hex-encoded `.wasm` bytes) once and caches the
//!    compiled `Module`.
//! 2. Instantiates the module against an empty store.
//! 3. Looks up the export named `rce_dispatch` (the v0.1 entry-point
//!    convention) and invokes it with the gathered scalar inputs.
//! 4. Translates the single Wasm return value back to a `ScalarValue`.
//!
//! Tensor inputs/outputs are not supported in v0.1; the interpreter
//! errors before we get here.
//!
//! This is the concrete answer to "subsume Wasm" — rce orchestrates
//! the graph, Wasm is one of many possible leaves.

pub mod error;
pub mod executor;
pub mod tensor_abi;

pub use error::{Result, WasmError};
pub use executor::WasmExecutor;
pub use tensor_abi::{dtype_code, dtype_from_code, DESC_HEADER_BYTES};
