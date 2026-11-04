//! Placement-driven dispatch for the scheduler — Stage 4 wiring inside
//! rce-vm-scheduler (docs/rce-unified-ir-and-energy-routing.md §3, §4).
//!
//! When `Backends.placement` is non-empty, every `Apply { op: Custom(name) }`
//! node whose `name` is `handles()`-claimed by at least one registered
//! `PricedBackend` is routed through [`pick_priced`] (min-joule selection) and
//! executed by the chosen backend — replacing the cost-blind first-claim-wins
//! `OpDispatch` chain for the ops it covers. Nodes whose op is *not* covered
//! by any priced backend fall through to the existing chain unchanged, so
//! placement is incrementally adoptable per op.
//!
//! `PricedBackend` is a deliberately small, concrete trait (typed `HostTensor`
//! inputs, scalar `(picojoules, measured)` cost) rather than the
//! `rce-vm-core::Backend` trait's `TensorHandle` + `Plan` indirection, which
//! needs a value store the scheduler doesn't have today. The two coexist: the
//! `Backend` trait remains the architectural contract for `compile` / `cost`
//! reasoning; `PricedBackend` is the pragmatic execution interface the
//! scheduler calls during dispatch.

use std::sync::Arc;

use rce_vm_backend_cpu::Value;

/// A backend that prices an op (cost) and executes it on `Value` inputs.
///
/// Implementors are expected to be cheap to share (`Arc`) and fully thread-safe
/// — the scheduler may call `cost` and `execute` from any worker thread.
///
/// `Value` carries any residency the runtime supports — host (`Value::Tensor`)
/// or device (`Value::DeviceTensor`). Backends that can consume device-resident
/// inputs directly do so (no host roundtrip); backends that can't are expected
/// to demote internally. Likewise, backends are free to *return* device-resident
/// outputs when that's cheaper than copying — the scheduler forwards the
/// `Value` to downstream consumers without forcing it to host. This is the
/// DeviceTensor-preservation path: a chain of consecutive same-device ops
/// stays on-device end-to-end.
pub trait PricedBackend: Send + Sync {
    /// Stable backend identifier (e.g. `"cpu:op-blas"`, `"gpu:op-mps"`).
    fn id(&self) -> &str;

    /// Op names this backend will accept in `cost` / `execute`.
    fn handles(&self) -> &[&'static str];

    /// Energy estimate (picojoules) for executing `op` with the given input
    /// shapes, plus a flag indicating whether the estimate is measurement-
    /// derived (`true`) or analytical (`false`) — the §4.5 provenance seam.
    fn cost(&self, op: &str, input_shapes: &[&[usize]]) -> (f64, bool);

    /// Execute the op on the given `Value` inputs; return the result `Value`.
    /// Inputs may arrive as `Tensor` or `DeviceTensor`; output residency is
    /// the backend's choice (device-resident outputs let downstream GPU ops
    /// avoid host roundtrips).
    fn execute(&self, op: &str, inputs: &[Value]) -> Result<Value, String>;
}

/// Map a `rce_graph::graph::OpKind` to the canonical op name a priced
/// backend would list in `handles()`. Returns:
///
/// - `Some(custom_name)` for `OpKind::Custom(name)` — the existing
///   extension point (matmul, qmm, silu, rms_norm, …).
/// - `Some(lowercase_str)` for the built-in arithmetic / comparison /
///   logic / concat ops (`"add"`, `"mul"`, `"eq"`, `"and"`, …).
///
/// A priced backend listing one of these names in its `handles()` will
/// intercept the corresponding `Apply` node; everything else falls
/// through to the legacy `OpDispatch` chain.
pub fn opkind_name(op: &rce_graph::graph::OpKind) -> Option<&str> {
    use rce_graph::graph::OpKind::*;
    Some(match op {
        Add => "add", Sub => "sub", Mul => "mul", Div => "div",
        Rem => "rem", Neg => "neg",
        Eq  => "eq",  Ne  => "ne",  Lt  => "lt",  Le  => "le",
        Gt  => "gt",  Ge  => "ge",
        And => "and", Or  => "or",  Not => "not",
        Concat => "concat",
        Custom(name) => name.as_str(),
    })
}

/// Inspect the shape of a `Value` if it carries one (host or device tensor).
/// Returns `None` for scalars and other non-tensor values. The shape is what
/// `pick_priced` reasons about; the data residency is what `execute` consumes.
pub fn value_shape(v: &Value) -> Option<&[usize]> {
    match v {
        Value::Tensor(t) => Some(&t.shape),
        Value::DeviceTensor(d) => Some(&d.shape),
        Value::Scalar(_) => None,
    }
}

/// Min-joule selection across a set of priced backends.
///
/// Returns the cheapest backend that `handles()` claims the op, along with its
/// estimate. Backends that don't claim the op are silently skipped. `None`
/// means no registered backend handles `op` — the caller should fall through
/// to the legacy first-claim-wins path.
pub fn pick_priced<'a>(
    backends: &'a [Arc<dyn PricedBackend>],
    op: &str,
    input_shapes: &[&[usize]],
) -> Option<(&'a Arc<dyn PricedBackend>, f64, bool)> {
    let mut best: Option<(&'a Arc<dyn PricedBackend>, f64, bool)> = None;
    for be in backends {
        if !be.handles().iter().any(|h| *h == op) {
            continue;
        }
        let (pj, measured) = be.cost(op, input_shapes);
        let cheaper = best.as_ref().map_or(true, |(_, prev_pj, _)| pj < *prev_pj);
        if cheaper {
            best = Some((be, pj, measured));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_vm_backend_cpu::{DType, HostTensor, Value};

    /// Tiny test fixture: a `PricedBackend` that handles "matmul" with a
    /// caller-provided `cost_pj` and an identity execute (returns the first input).
    struct Fixed {
        id_: String,
        handles_: Vec<&'static str>,
        cost_pj: f64,
        measured: bool,
    }
    impl PricedBackend for Fixed {
        fn id(&self) -> &str { &self.id_ }
        fn handles(&self) -> &[&'static str] { &self.handles_ }
        fn cost(&self, _op: &str, _shapes: &[&[usize]]) -> (f64, bool) {
            (self.cost_pj, self.measured)
        }
        fn execute(&self, _op: &str, inputs: &[Value]) -> Result<Value, String> {
            // Echo back the first input as a clone (correctness isn't the
            // point of this fixture; pick_priced is).
            Ok(inputs.first().ok_or("no inputs")?.clone())
        }
    }

    fn fixed(id: &str, cost_pj: f64, handles: Vec<&'static str>) -> Arc<dyn PricedBackend> {
        Arc::new(Fixed { id_: id.into(), handles_: handles, cost_pj, measured: true })
    }

    #[test]
    fn pick_priced_chooses_lowest_joules() {
        let backends = vec![
            fixed("cpu", 100.0, vec!["matmul"]),
            fixed("gpu", 50.0, vec!["matmul"]),
            fixed("silicon", 5.0, vec!["matmul"]),
        ];
        let (be, pj, _) = pick_priced(&backends, "matmul", &[&[64, 64], &[64, 64]]).unwrap();
        assert_eq!(be.id(), "silicon");
        assert_eq!(pj, 5.0);
    }

    #[test]
    fn pick_priced_skips_backends_that_dont_handle_op() {
        let backends = vec![
            fixed("cpu_blas", 100.0, vec!["matmul"]),         // claims matmul
            fixed("gpu_attn", 1.0, vec!["attention"]),         // claims attention only
        ];
        // GPU is cheaper but doesn't handle matmul, so CPU wins.
        let (be, _, _) = pick_priced(&backends, "matmul", &[&[64, 64]]).unwrap();
        assert_eq!(be.id(), "cpu_blas");
    }

    #[test]
    fn pick_priced_returns_none_when_no_backend_handles_op() {
        let backends = vec![fixed("cpu", 100.0, vec!["matmul"])];
        assert!(pick_priced(&backends, "conv2d", &[&[1, 3, 224, 224]]).is_none());
    }

    #[test]
    fn pick_priced_propagates_measured_flag() {
        let backends: Vec<Arc<dyn PricedBackend>> = vec![
            Arc::new(Fixed {
                id_: "estimate".into(),
                handles_: vec!["matmul"],
                cost_pj: 1.0,
                measured: false,
            }),
            Arc::new(Fixed {
                id_: "measured".into(),
                handles_: vec!["matmul"],
                cost_pj: 2.0, // costlier but measured
                measured: true,
            }),
        ];
        let (be, pj, measured) = pick_priced(&backends, "matmul", &[&[1]]).unwrap();
        // Cheapest still wins (the policy is min-joules); the `measured` bit
        // is reported alongside so callers can downweight estimates if desired.
        assert_eq!(be.id(), "estimate");
        assert_eq!(pj, 1.0);
        assert!(!measured);
    }
}
