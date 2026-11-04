//! Op-level dispatch — preempts the built-in `ml::dispatch_custom` for
//! `Apply { op: Custom(name) }` nodes whose `name` is claimed by an
//! external backend (wgpu, CUDA, dedicated silicon, …).
//!
//! Contract:
//! - `handles()` returns the set of `Custom(name)` strings this backend
//!   can compute. Names are global-scope (no namespace prefix) — the
//!   first dispatcher whose `handles()` claims a name wins.
//! - `execute()` is called with the gathered tensor/scalar args. If the
//!   dispatcher returns `InterpretError::Unsupported(_)` (e.g. wrong
//!   shape/dtype for the kernel it actually has), the interpreter falls
//!   through to the next dispatcher and finally to the CPU `ml::` path.
//!
//! Returning `Unsupported` is the correct way to say "I claim this op
//! name but cannot run *these specific* inputs." Other errors propagate.

use crate::error::{InterpretError, Result};
use crate::value::Value;
use rce_graph::graph::OpKind;

pub trait OpDispatch {
    /// `Custom(name)` op names this backend claims.
    fn handles(&self) -> &[&'static str];

    /// True iff this backend claims `name`. Default implementation checks
    /// `handles()` for a string equality match — backends that claim a
    /// **fixed list** of names need do nothing else.
    ///
    /// Override when claiming a **name family** that can't be enumerated
    /// up front. Today's case: the Metal leaf's codegen-driven fused-
    /// chain dispatcher claims any name shaped `fused_<...>_f32` whose
    /// body parses through `rce_fusion::codegen::parse_function_name`
    /// — that's an unbounded set as new fragments register, so we can't
    /// pre-list it in `handles()`. The override calls the parser and
    /// reports whether the name is one the backend can resolve.
    fn claims(&self, name: &str) -> bool {
        self.handles().iter().any(|h| *h == name)
    }

    /// Execute a claimed op. Return `InterpretError::Unsupported(_)` to
    /// fall through to the next backend / CPU path; any other error
    /// propagates.
    fn execute(&self, node_id: usize, name: &str, args: &[Value]) -> Result<Value>;

    /// Optionally handle a primitive `OpKind` (Add/Sub/Mul/Div/Neg/etc.)
    /// on tensor args. Default: `Unsupported` so the interpreter falls
    /// through to the built-in CPU kernel.
    ///
    /// Backends should match on `op` and return `Unsupported(_)` if the
    /// shape/dtype combination is outside their kernel coverage. The
    /// interpreter will then try the next dispatcher and finally the
    /// CPU path.
    fn execute_primitive(
        &self,
        _node_id: usize,
        _op: &OpKind,
        _args: &[Value],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "primitive op not handled by this backend".into(),
        ))
    }
}
