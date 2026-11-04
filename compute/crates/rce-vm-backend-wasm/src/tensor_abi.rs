//! Wasm leaf tensor ABI — v0.2.
//!
//! Spec: `docs/spec/fg-wasm-leaf-abi.md`. This module owns the
//! host-side encoding/decoding of tensor descriptors in the Wasm
//! module's linear memory.
//!
//! The descriptor is 16 bytes header + `4 × rank` bytes of shape:
//!
//! ```text
//!   offset 0: rank     u32 LE
//!   offset 4: dtype    u32 LE
//!   offset 8: data_ptr u32 LE
//!   offset 12: _reserved u32 (zero)
//!   offset 16: shape   u32 × rank, LE
//! ```

use crate::error::{Result, WasmError};
use rce_vm_backend_cpu::tensor::DType;

pub const DESC_HEADER_BYTES: usize = 16;

/// Map a rce [`DType`] to its v0.2 ABI code.
pub fn dtype_code(dt: DType) -> u32 {
    match dt {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::I32 => 2,
        DType::I64 => 3,
        DType::Bool => 4,
        DType::F16 => 5,
        DType::BF16 => 6,
    }
}

/// Inverse of [`dtype_code`]. Returns `Err` for reserved or unknown codes.
pub fn dtype_from_code(code: u32) -> Result<DType> {
    Ok(match code {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::I32,
        3 => DType::I64,
        4 => DType::Bool,
        5 => DType::F16,
        6 => DType::BF16,
        other => {
            return Err(WasmError::UnsupportedValType(format!(
                "tensor ABI dtype code {other} is reserved or unknown"
            )));
        }
    })
}
