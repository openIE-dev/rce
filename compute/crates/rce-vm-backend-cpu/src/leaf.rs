//! Leaf-dispatch trait.
//!
//! When the interpreter encounters a `GraphNode::CodeBlock { language, .. }`,
//! it delegates to a `LeafDispatch` implementation that knows how to evaluate
//! that language. This is the seam where Wasm, ONNX subgraphs, embedded
//! GPU kernels, or any other "opaque leaf" plug into the Graph VM without
//! the CPU interpreter taking a dependency on them.
//!
//! ## Wire-protocol contract
//!
//! - **inputs**: a slice of [`Value`]s, in port order. Each can be a
//!   scalar or a tensor. Backends choose how to handle each.
//! - **returns**: a single [`Value`].
//! - Tensor-bearing leaves use a per-language ABI; for Wasm, see
//!   `docs/spec/fg-wasm-leaf-abi.md` (v0.2 = tensors via linear memory).
//!
//! ## Falling through
//!
//! If a leaf cannot handle a specific call (e.g. tensor arg into a v0.1
//! scalar-only Wasm module), it returns `InterpretError::Unsupported(_)`
//! and the interpreter tries the next registered leaf for the same
//! language. There is no implicit conversion.

use crate::error::Result;
use crate::value::Value;

/// A leaf backend invoked when `GraphNode::CodeBlock { language: <kind>, code_text }`
/// is reached. Implementations match on the `language` string and compile/run
/// the embedded code.
pub trait LeafDispatch {
    /// The languages this dispatcher accepts (e.g. `["wasm", "wat"]`).
    fn languages(&self) -> &[&'static str];

    /// Evaluate a code block with positional inputs (scalar or tensor).
    fn invoke(
        &self,
        node_id: usize,
        language: &str,
        code_text: &str,
        output_type: &str,
        args: &[Value],
    ) -> Result<Value>;
}
