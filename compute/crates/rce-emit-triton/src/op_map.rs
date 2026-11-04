//! graph `OpKind` -> Triton expression mapping.
//!
//! Triton uses Python-style infix operators for arithmetic/comparison and
//! `tl.*` intrinsics for math builtins. Bitwise `&`/`|`/`~` serve as
//! mask-level logic (Triton tensors are implicitly mask-typed booleans).
//!
//! Ops that require block-tiling patterns (matmul) or reductions (softmax)
//! are marked `Unmapped` — they need dedicated codegen in a later phase.

use rce_graph::graph::OpKind;

/// A Triton expression emission instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TritonOp {
    /// Binary infix: `a {op} b` (e.g. `a + b`)
    BinaryInfix(&'static str),
    /// Unary prefix: `{op}a` (e.g. `~a`, `-a`)
    UnaryPrefix(&'static str),
    /// Unary `tl.*` call: `tl.{name}(a)` (e.g. `tl.exp(a)`)
    UnaryCall(&'static str),
    /// Binary `tl.*` call: `tl.{name}(a, b)` (e.g. `tl.maximum(a, b)`)
    BinaryCall(&'static str),
    /// Ternary `tl.*` call: `tl.{name}(a, b, c)` (e.g. `tl.where(cond, a, b)`)
    TernaryCall(&'static str),
    /// Cast: `a.to({triton_type})` — requires type info from context
    Cast,
    /// No Triton equivalent (needs block-tile or reduction pattern).
    Unmapped,
}

/// Map a graph OpKind to a Triton emission instruction.
pub fn triton_op_for(op: &OpKind) -> TritonOp {
    match op {
        // -- Arithmetic -------------------------------------------------
        OpKind::Add => TritonOp::BinaryInfix("+"),
        OpKind::Sub => TritonOp::BinaryInfix("-"),
        OpKind::Mul => TritonOp::BinaryInfix("*"),
        OpKind::Div => TritonOp::BinaryInfix("/"),
        OpKind::Rem => TritonOp::BinaryInfix("%"),
        OpKind::Neg => TritonOp::UnaryPrefix("-"),
        // -- Comparison -------------------------------------------------
        OpKind::Eq => TritonOp::BinaryInfix("=="),
        OpKind::Ne => TritonOp::BinaryInfix("!="),
        OpKind::Lt => TritonOp::BinaryInfix("<"),
        OpKind::Le => TritonOp::BinaryInfix("<="),
        OpKind::Gt => TritonOp::BinaryInfix(">"),
        OpKind::Ge => TritonOp::BinaryInfix(">="),
        // -- Logic (bitwise for mask ops) --------------------------------
        OpKind::And => TritonOp::BinaryInfix("&"),
        OpKind::Or => TritonOp::BinaryInfix("|"),
        OpKind::Not => TritonOp::UnaryPrefix("~"),
        // -- String concat: no GPU equivalent ----------------------------
        OpKind::Concat => TritonOp::Unmapped,
        // -- Custom ops --------------------------------------------------
        OpKind::Custom(name) => custom_triton_op(name),
    }
}

fn custom_triton_op(name: &str) -> TritonOp {
    match name {
        // -- Element-wise math ------------------------------------------
        "exp" | "Exp" => TritonOp::UnaryCall("tl.exp"),
        "log" | "Log" => TritonOp::UnaryCall("tl.log"),
        "sqrt" | "Sqrt" => TritonOp::UnaryCall("tl.sqrt"),
        "abs" | "Abs" => TritonOp::UnaryCall("tl.abs"),
        "sin" | "Sin" => TritonOp::UnaryCall("tl.sin"),
        "cos" | "Cos" => TritonOp::UnaryCall("tl.cos"),
        "sigmoid" | "Sigmoid" => TritonOp::UnaryCall("tl.sigmoid"),
        // -- Activations ------------------------------------------------
        "relu" | "Relu" => TritonOp::BinaryCall("tl.maximum"),
        // -- Min / max --------------------------------------------------
        "max" | "Max" | "maximum" => TritonOp::BinaryCall("tl.maximum"),
        "min" | "Min" | "minimum" => TritonOp::BinaryCall("tl.minimum"),
        // -- Conditional ------------------------------------------------
        "where" | "Where" | "select" => TritonOp::TernaryCall("tl.where"),
        // -- Cast -------------------------------------------------------
        "cast" | "Cast" | "convert" => TritonOp::Cast,
        // -- Unmapped: need block-tile or reduction patterns -------------
        "matmul" | "MatMul" => TritonOp::Unmapped,
        "softmax" | "Softmax" => TritonOp::Unmapped,
        "layer_norm" | "LayerNorm" | "LayerNormalization" => TritonOp::Unmapped,
        "batch_norm" | "BatchNorm" | "BatchNormalization" => TritonOp::Unmapped,
        "conv" | "Conv" => TritonOp::Unmapped,
        "reduce_sum" | "ReduceSum" => TritonOp::Unmapped,
        "reduce_mean" | "ReduceMean" => TritonOp::Unmapped,
        "reduce_max" | "ReduceMax" => TritonOp::Unmapped,
        // -- Fallback ---------------------------------------------------
        _ => TritonOp::Unmapped,
    }
}

/// Count how many ops have a non-Unmapped mapping.
pub fn mapped_triton_count() -> usize {
    CORE_MAPPED + CUSTOM_MAPPED
}

const CORE_MAPPED: usize = 15; // all core OpKind variants except Concat
const CUSTOM_MAPPED: usize = 14; // exp, log, sqrt, abs, sin, cos, sigmoid, relu, max, min, where, cast + max/min aliases counted once each

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_maps_to_plus() {
        assert_eq!(triton_op_for(&OpKind::Add), TritonOp::BinaryInfix("+"));
    }

    #[test]
    fn sub_maps_to_minus() {
        assert_eq!(triton_op_for(&OpKind::Sub), TritonOp::BinaryInfix("-"));
    }

    #[test]
    fn eq_maps_to_double_eq() {
        assert_eq!(triton_op_for(&OpKind::Eq), TritonOp::BinaryInfix("=="));
    }

    #[test]
    fn and_maps_to_bitwise() {
        assert_eq!(triton_op_for(&OpKind::And), TritonOp::BinaryInfix("&"));
    }

    #[test]
    fn not_maps_to_tilde() {
        assert_eq!(triton_op_for(&OpKind::Not), TritonOp::UnaryPrefix("~"));
    }

    #[test]
    fn exp_maps_to_tl_exp() {
        assert_eq!(
            triton_op_for(&OpKind::Custom("exp".into())),
            TritonOp::UnaryCall("tl.exp"),
        );
    }

    #[test]
    fn sigmoid_maps_to_tl_sigmoid() {
        assert_eq!(
            triton_op_for(&OpKind::Custom("sigmoid".into())),
            TritonOp::UnaryCall("tl.sigmoid"),
        );
    }

    #[test]
    fn relu_maps_to_tl_maximum() {
        assert_eq!(
            triton_op_for(&OpKind::Custom("relu".into())),
            TritonOp::BinaryCall("tl.maximum"),
        );
    }

    #[test]
    fn matmul_is_unmapped() {
        assert_eq!(
            triton_op_for(&OpKind::Custom("matmul".into())),
            TritonOp::Unmapped,
        );
    }

    #[test]
    fn softmax_is_unmapped() {
        assert_eq!(
            triton_op_for(&OpKind::Custom("softmax".into())),
            TritonOp::Unmapped,
        );
    }

    #[test]
    fn concat_is_unmapped() {
        assert_eq!(triton_op_for(&OpKind::Concat), TritonOp::Unmapped);
    }

    #[test]
    fn where_maps_to_ternary() {
        assert_eq!(
            triton_op_for(&OpKind::Custom("where".into())),
            TritonOp::TernaryCall("tl.where"),
        );
    }

    #[test]
    fn cast_maps() {
        assert_eq!(
            triton_op_for(&OpKind::Custom("cast".into())),
            TritonOp::Cast,
        );
    }

    #[test]
    fn maximum_maps_to_tl_maximum() {
        assert_eq!(
            triton_op_for(&OpKind::Custom("maximum".into())),
            TritonOp::BinaryCall("tl.maximum"),
        );
    }

    #[test]
    fn minimum_maps_to_tl_minimum() {
        assert_eq!(
            triton_op_for(&OpKind::Custom("minimum".into())),
            TritonOp::BinaryCall("tl.minimum"),
        );
    }

    #[test]
    fn mapped_count() {
        assert_eq!(mapped_triton_count(), 29);
    }
}
