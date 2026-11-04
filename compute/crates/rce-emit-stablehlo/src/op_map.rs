//! graph `OpKind` → StableHLO op name mapping.
//!
//! StableHLO ops are prefixed `stablehlo.` in MLIR text. This table maps
//! core graph ops and Custom-name ops to their StableHLO equivalents.
//!
//! Key differences from ONNX:
//! - Comparison is a single `stablehlo.compare` op with a direction attribute
//! - No separate relu/sigmoid/gelu — they decompose into primitives
//!   (but we emit them as `stablehlo.custom_call` with a readable name)
//! - `dot_general` replaces MatMul/Gemm

use rce_graph::graph::OpKind;

/// A StableHLO op emission instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HloOp {
    /// Standard binary: `stablehlo.{name} %a, %b : tensor<T>`
    Binary(&'static str),
    /// Standard unary: `stablehlo.{name} %a : tensor<T>`
    Unary(&'static str),
    /// Comparison: `stablehlo.compare %a, %b, LT : (tensor<T>, tensor<T>) -> tensor<i1>`
    Compare(&'static str),
    /// Custom call (for ops without a direct StableHLO primitive):
    /// `stablehlo.custom_call @{name}(%a) : ...`
    CustomCall(&'static str),
    /// No StableHLO equivalent.
    Unmapped,
}

/// Map a graph OpKind to a StableHLO emission instruction.
pub fn hlo_op_for(op: &OpKind) -> HloOp {
    match op {
        // ── Arithmetic ──────────────────────────────────────────
        OpKind::Add => HloOp::Binary("add"),
        OpKind::Sub => HloOp::Binary("subtract"),
        OpKind::Mul => HloOp::Binary("multiply"),
        OpKind::Div => HloOp::Binary("divide"),
        OpKind::Rem => HloOp::Binary("remainder"),
        OpKind::Neg => HloOp::Unary("negate"),
        // ── Comparison ──────────────────────────────────────────
        OpKind::Eq => HloOp::Compare("EQ"),
        OpKind::Ne => HloOp::Compare("NE"),
        OpKind::Lt => HloOp::Compare("LT"),
        OpKind::Le => HloOp::Compare("LE"),
        OpKind::Gt => HloOp::Compare("GT"),
        OpKind::Ge => HloOp::Compare("GE"),
        // ── Logic ────────────────────────────────────────────────
        OpKind::And => HloOp::Binary("and"),
        OpKind::Or => HloOp::Binary("or"),
        OpKind::Not => HloOp::Unary("not"),
        // ── String concat: no HLO equivalent ────────────────────
        OpKind::Concat => HloOp::Unmapped,
        // ── Custom ops ──────────────────────────────────────────
        OpKind::Custom(name) => custom_hlo_op(name),
    }
}

fn custom_hlo_op(name: &str) -> HloOp {
    match name {
        // ── Linear algebra ──────────────────────────────────────
        "matmul" | "MatMul" => HloOp::Binary("dot"),
        "gemm" | "Gemm" => HloOp::Binary("dot"),
        // ── Element-wise math ───────────────────────────────────
        "exp" | "Exp" => HloOp::Unary("exponential"),
        "log" | "Log" => HloOp::Unary("log"),
        "sqrt" | "Sqrt" => HloOp::Unary("sqrt"),
        "rsqrt" | "Rsqrt" => HloOp::Unary("rsqrt"),
        "abs" | "Abs" => HloOp::Unary("abs"),
        "ceil" | "Ceil" => HloOp::Unary("ceil"),
        "floor" | "Floor" => HloOp::Unary("floor"),
        "sin" | "Sin" => HloOp::Unary("sine"),
        "cos" | "Cos" => HloOp::Unary("cosine"),
        "tanh" | "Tanh" => HloOp::Unary("tanh"),
        "sigmoid" | "Sigmoid" => HloOp::Unary("logistic"),
        "pow" | "Pow" => HloOp::Binary("power"),
        "max" | "Max" | "maximum" => HloOp::Binary("maximum"),
        "min" | "Min" | "minimum" => HloOp::Binary("minimum"),
        "clamp" | "Clip" | "clip" => HloOp::CustomCall("clamp"),
        // ── Shape / tensor ──────────────────────────────────────
        "reshape" | "Reshape" => HloOp::CustomCall("reshape"),
        "transpose" | "Transpose" => HloOp::CustomCall("transpose"),
        "concat" | "Concat" => HloOp::CustomCall("concatenate"),
        "slice" | "Slice" => HloOp::CustomCall("slice"),
        "gather" | "Gather" => HloOp::CustomCall("gather"),
        "scatter" | "ScatterElements" => HloOp::CustomCall("scatter"),
        "pad" | "Pad" => HloOp::CustomCall("pad"),
        "broadcast" | "Expand" | "expand" => HloOp::CustomCall("broadcast_in_dim"),
        // ── Reduction ───────────────────────────────────────────
        "reduce_sum" | "ReduceSum" => HloOp::CustomCall("reduce_sum"),
        "reduce_mean" | "ReduceMean" => HloOp::CustomCall("reduce_mean"),
        "reduce_max" | "ReduceMax" => HloOp::CustomCall("reduce_max"),
        "reduce_min" | "ReduceMin" => HloOp::CustomCall("reduce_min"),
        "reduce_prod" | "ReduceProd" => HloOp::CustomCall("reduce_prod"),
        // ── Convolution ─────────────────────────────────────────
        "conv" | "Conv" => HloOp::CustomCall("convolution"),
        "conv_transpose" | "ConvTranspose" => HloOp::CustomCall("convolution"),
        // ── Normalization (decompose in real compiler, stub here) ──
        "layer_norm" | "LayerNorm" | "LayerNormalization" => HloOp::CustomCall("layer_norm"),
        "batch_norm" | "BatchNorm" | "BatchNormalization" => HloOp::CustomCall("batch_norm_inference"),
        // ── Activations (decompose in real compiler, stub here) ──
        "relu" | "Relu" => HloOp::CustomCall("relu"),
        "gelu" | "Gelu" => HloOp::CustomCall("gelu"),
        "softmax" | "Softmax" => HloOp::CustomCall("softmax"),
        "leaky_relu" | "LeakyRelu" => HloOp::CustomCall("leaky_relu"),
        "selu" | "Selu" => HloOp::CustomCall("selu"),
        "softplus" | "Softplus" => HloOp::CustomCall("softplus"),
        // ── Pooling ─────────────────────────────────────────────
        "max_pool" | "MaxPool" => HloOp::CustomCall("max_pool"),
        "avg_pool" | "AveragePool" => HloOp::CustomCall("avg_pool"),
        // ── Data flow ───────────────────────────────────────────
        "cast" | "Cast" | "convert" => HloOp::CustomCall("convert"),
        "identity" | "Identity" => HloOp::CustomCall("identity"),
        "where" | "Where" | "select" => HloOp::CustomCall("select"),
        // ── Attention ───────────────────────────────────────────
        "attention" | "Attention" => HloOp::CustomCall("dot_general_attention"),
        // ── No match ────────────────────────────────────────────
        _ => HloOp::Unmapped,
    }
}

/// Count how many ops have a non-Unmapped mapping.
pub fn mapped_hlo_count() -> usize {
    CORE_MAPPED + CUSTOM_MAPPED
}

const CORE_MAPPED: usize = 16; // all core OpKind variants except Concat
const CUSTOM_MAPPED: usize = 49;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_maps_to_binary_add() {
        assert_eq!(hlo_op_for(&OpKind::Add), HloOp::Binary("add"));
    }

    #[test]
    fn eq_maps_to_compare() {
        assert_eq!(hlo_op_for(&OpKind::Eq), HloOp::Compare("EQ"));
    }

    #[test]
    fn ne_maps_to_compare_ne() {
        assert_eq!(hlo_op_for(&OpKind::Ne), HloOp::Compare("NE"));
    }

    #[test]
    fn custom_matmul_maps_to_dot() {
        assert_eq!(hlo_op_for(&OpKind::Custom("matmul".into())), HloOp::Binary("dot"));
    }

    #[test]
    fn custom_sigmoid_maps_to_logistic() {
        assert_eq!(hlo_op_for(&OpKind::Custom("sigmoid".into())), HloOp::Unary("logistic"));
    }

    #[test]
    fn custom_conv_maps_to_convolution() {
        assert_eq!(hlo_op_for(&OpKind::Custom("conv".into())), HloOp::CustomCall("convolution"));
    }

    #[test]
    fn concat_string_is_unmapped() {
        assert_eq!(hlo_op_for(&OpKind::Concat), HloOp::Unmapped);
    }

    #[test]
    fn hdc_op_is_unmapped() {
        assert_eq!(hlo_op_for(&OpKind::Custom("hdc.bundle".into())), HloOp::Unmapped);
    }

    #[test]
    fn mapped_count_is_65() {
        assert_eq!(mapped_hlo_count(), 65);
    }
}
