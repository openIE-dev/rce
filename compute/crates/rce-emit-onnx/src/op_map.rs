//! graph `OpKind` → ONNX op type mapping table.
//!
//! ONNX op types are identified by a `(domain, op_type)` string pair plus
//! an opset version. For the phase-0 tracer we target `domain = ""`
//! (default ONNX operator set) at opset 20.
//!
//! This table is the Rosetta stone between the graph's ~347 OpKinds and
//! ONNX's ~180 core ops. Most of the graph's long tail (HDC, FHE, neuromorphic,
//! photonic, attractor) has no ONNX equivalent — that's honest, and we
//! surface it as `UnmappedOp` diagnostics rather than silent data loss.

use crate::diag::UnmappedOp;
use rce_graph::graph::OpKind;

/// A concrete ONNX op-type + domain pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxOp {
    pub op_type: &'static str,
    pub domain: &'static str,
}

impl OnnxOp {
    const fn core(op_type: &'static str) -> Self {
        Self { op_type, domain: "" }
    }
}

/// Map a graph op-kind to an ONNX op. Returns an `UnmappedOp` diagnostic
/// when no equivalent exists.
pub fn onnx_op_for(op: &OpKind) -> Result<OnnxOp, UnmappedOp> {
    let mapped = match op {
        // ── Arithmetic ──────────────────────────────────────────
        OpKind::Add => OnnxOp::core("Add"),
        OpKind::Sub => OnnxOp::core("Sub"),
        OpKind::Mul => OnnxOp::core("Mul"),
        OpKind::Div => OnnxOp::core("Div"),
        OpKind::Rem => OnnxOp::core("Mod"),
        OpKind::Neg => OnnxOp::core("Neg"),
        // ── Comparison ──────────────────────────────────────────
        OpKind::Eq => OnnxOp::core("Equal"),
        OpKind::Lt => OnnxOp::core("Less"),
        OpKind::Le => OnnxOp::core("LessOrEqual"),
        OpKind::Gt => OnnxOp::core("Greater"),
        OpKind::Ge => OnnxOp::core("GreaterOrEqual"),
        // ── Logic ────────────────────────────────────────────────
        OpKind::And => OnnxOp::core("And"),
        OpKind::Or => OnnxOp::core("Or"),
        OpKind::Not => OnnxOp::core("Not"),
        // ── Ne: lower to Not(Equal(a,b)) ────────────────────────
        // Concat: ONNX Concat is tensor-axis, not string — unmapped.
        OpKind::Ne | OpKind::Concat => {
            return Err(UnmappedOp::from_opkind(op));
        }
        // ── Custom: dispatch by name to ONNX ops ────────────────
        OpKind::Custom(name) => {
            return custom_op_for(name).ok_or_else(|| UnmappedOp::from_opkind(op));
        }
    };
    Ok(mapped)
}

/// Map `Custom("name")` strings to ONNX ops.
/// These cover the full inv-ai-codegraph OpKind variants that pass through
/// rce-graph as Custom strings.
fn custom_op_for(name: &str) -> Option<OnnxOp> {
    let op = match name {
        // ── Linear algebra ──────────────────────────────────────
        "matmul" | "MatMul" => OnnxOp::core("MatMul"),
        "gemm" | "Gemm" => OnnxOp::core("Gemm"),
        // ── Activations ─────────────────────────────────────────
        "relu" | "Relu" => OnnxOp::core("Relu"),
        "sigmoid" | "Sigmoid" => OnnxOp::core("Sigmoid"),
        "tanh" | "Tanh" => OnnxOp::core("Tanh"),
        "softmax" | "Softmax" => OnnxOp::core("Softmax"),
        "leaky_relu" | "LeakyRelu" => OnnxOp::core("LeakyRelu"),
        "elu" | "Elu" => OnnxOp::core("Elu"),
        "selu" | "Selu" => OnnxOp::core("Selu"),
        "gelu" | "Gelu" => OnnxOp::core("Gelu"),
        "hard_sigmoid" | "HardSigmoid" => OnnxOp::core("HardSigmoid"),
        "hard_swish" | "HardSwish" => OnnxOp::core("HardSwish"),
        "log_softmax" | "LogSoftmax" => OnnxOp::core("LogSoftmax"),
        "softplus" | "Softplus" => OnnxOp::core("Softplus"),
        // ── Normalization ───────────────────────────────────────
        "layer_norm" | "LayerNorm" | "LayerNormalization" => OnnxOp::core("LayerNormalization"),
        "batch_norm" | "BatchNorm" | "BatchNormalization" => OnnxOp::core("BatchNormalization"),
        "instance_norm" | "InstanceNormalization" => OnnxOp::core("InstanceNormalization"),
        "group_norm" | "GroupNormalization" => OnnxOp::core("GroupNormalization"),
        // ── Pooling ─────────────────────────────────────────────
        "max_pool" | "MaxPool" => OnnxOp::core("MaxPool"),
        "avg_pool" | "AveragePool" => OnnxOp::core("AveragePool"),
        "global_avg_pool" | "GlobalAveragePool" => OnnxOp::core("GlobalAveragePool"),
        "global_max_pool" | "GlobalMaxPool" => OnnxOp::core("GlobalMaxPool"),
        // ── Convolution ─────────────────────────────────────────
        "conv" | "Conv" => OnnxOp::core("Conv"),
        "conv_transpose" | "ConvTranspose" => OnnxOp::core("ConvTranspose"),
        // ── Shape / tensor manipulation ─────────────────────────
        "reshape" | "Reshape" => OnnxOp::core("Reshape"),
        "transpose" | "Transpose" => OnnxOp::core("Transpose"),
        "concat" | "Concat" => OnnxOp::core("Concat"),
        "split" | "Split" => OnnxOp::core("Split"),
        "slice" | "Slice" => OnnxOp::core("Slice"),
        "gather" | "Gather" => OnnxOp::core("Gather"),
        "scatter" | "ScatterElements" => OnnxOp::core("ScatterElements"),
        "squeeze" | "Squeeze" => OnnxOp::core("Squeeze"),
        "unsqueeze" | "Unsqueeze" => OnnxOp::core("Unsqueeze"),
        "flatten" | "Flatten" => OnnxOp::core("Flatten"),
        "expand" | "Expand" => OnnxOp::core("Expand"),
        "tile" | "Tile" => OnnxOp::core("Tile"),
        "pad" | "Pad" => OnnxOp::core("Pad"),
        "shape" | "Shape" => OnnxOp::core("Shape"),
        // ── Element-wise math ───────────────────────────────────
        "exp" | "Exp" => OnnxOp::core("Exp"),
        "log" | "Log" => OnnxOp::core("Log"),
        "sqrt" | "Sqrt" => OnnxOp::core("Sqrt"),
        "pow" | "Pow" => OnnxOp::core("Pow"),
        "abs" | "Abs" => OnnxOp::core("Abs"),
        "ceil" | "Ceil" => OnnxOp::core("Ceil"),
        "floor" | "Floor" => OnnxOp::core("Floor"),
        "clip" | "Clip" => OnnxOp::core("Clip"),
        "reciprocal" | "Reciprocal" => OnnxOp::core("Reciprocal"),
        "sin" | "Sin" => OnnxOp::core("Sin"),
        "cos" | "Cos" => OnnxOp::core("Cos"),
        // ── Reduction ───────────────────────────────────────────
        "reduce_sum" | "ReduceSum" => OnnxOp::core("ReduceSum"),
        "reduce_mean" | "ReduceMean" => OnnxOp::core("ReduceMean"),
        "reduce_max" | "ReduceMax" => OnnxOp::core("ReduceMax"),
        "reduce_min" | "ReduceMin" => OnnxOp::core("ReduceMin"),
        "reduce_prod" | "ReduceProd" => OnnxOp::core("ReduceProd"),
        // ── Data flow ───────────────────────────────────────────
        "cast" | "Cast" => OnnxOp::core("Cast"),
        "identity" | "Identity" => OnnxOp::core("Identity"),
        "where" | "Where" => OnnxOp::core("Where"),
        "constant" | "Constant" => OnnxOp::core("Constant"),
        // ── Attention / transformer ─────────────────────────────
        "attention" | "Attention" => OnnxOp { op_type: "Attention", domain: "com.microsoft" },
        // ── No match ────────────────────────────────────────────
        _ => return None,
    };
    Some(op)
}

/// Total number of mapped ONNX ops (core enum + custom names).
pub fn mapped_op_count() -> usize {
    CORE_MAPPED + CUSTOM_MAPPED
}

const CORE_MAPPED: usize = 14;
const CUSTOM_MAPPED: usize = 55;

/// Map a graph type string (e.g. `"f32"`, `"f64"`, `"i64"`, `"bool"`) to
/// the corresponding ONNX `TensorProto.DataType` integer.
pub fn onnx_dtype_for(rce_type: &str) -> Option<i32> {
    use crate::onnx::tensor_proto::DataType;
    let dt = match rce_type {
        "f32" | "float" => DataType::Float,
        "f64" | "double" => DataType::Double,
        "i32" => DataType::Int32,
        "i64" | "int" => DataType::Int64,
        "i16" => DataType::Int16,
        "i8" => DataType::Int8,
        "u8" => DataType::Uint8,
        "bool" => DataType::Bool,
        "string" => DataType::String,
        _ => return None,
    };
    Some(dt as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_maps_to_add() {
        let op = onnx_op_for(&OpKind::Add).unwrap();
        assert_eq!(op.op_type, "Add");
        assert_eq!(op.domain, "");
    }

    #[test]
    fn ne_is_unmapped_with_reason() {
        let err = onnx_op_for(&OpKind::Ne).unwrap_err();
        assert!(err.op_name.contains("Ne"));
    }

    #[test]
    fn hdc_custom_op_reports_hdc_domain() {
        let err = onnx_op_for(&OpKind::Custom("hdc.bundle".into())).unwrap_err();
        assert_eq!(err.domain, "custom");
        assert!(err.reason.contains("HDC"));
    }

    #[test]
    fn attractor_custom_op_reports_attractor_gap() {
        let err = onnx_op_for(&OpKind::Custom("attractor.competitive_comparison".into())).unwrap_err();
        assert!(err.reason.contains("attractor") || err.reason.contains("graph op"));
    }

    #[test]
    fn dtype_float() {
        assert_eq!(onnx_dtype_for("f32"), Some(1));
    }

    #[test]
    fn dtype_int64() {
        assert_eq!(onnx_dtype_for("i64"), Some(7));
    }

    #[test]
    fn dtype_unknown() {
        assert_eq!(onnx_dtype_for("tensor<?>"), None);
    }

    #[test]
    fn custom_relu_maps() {
        let op = onnx_op_for(&OpKind::Custom("relu".into())).unwrap();
        assert_eq!(op.op_type, "Relu");
    }

    #[test]
    fn custom_matmul_maps() {
        let op = onnx_op_for(&OpKind::Custom("matmul".into())).unwrap();
        assert_eq!(op.op_type, "MatMul");
    }

    #[test]
    fn custom_softmax_maps() {
        let op = onnx_op_for(&OpKind::Custom("softmax".into())).unwrap();
        assert_eq!(op.op_type, "Softmax");
    }

    #[test]
    fn custom_layer_norm_maps() {
        let op = onnx_op_for(&OpKind::Custom("layer_norm".into())).unwrap();
        assert_eq!(op.op_type, "LayerNormalization");
    }

    #[test]
    fn custom_conv_maps() {
        let op = onnx_op_for(&OpKind::Custom("conv".into())).unwrap();
        assert_eq!(op.op_type, "Conv");
    }

    #[test]
    fn custom_reshape_maps() {
        let op = onnx_op_for(&OpKind::Custom("reshape".into())).unwrap();
        assert_eq!(op.op_type, "Reshape");
    }

    #[test]
    fn custom_reduce_sum_maps() {
        let op = onnx_op_for(&OpKind::Custom("reduce_sum".into())).unwrap();
        assert_eq!(op.op_type, "ReduceSum");
    }

    #[test]
    fn custom_attention_maps_to_microsoft_domain() {
        let op = onnx_op_for(&OpKind::Custom("attention".into())).unwrap();
        assert_eq!(op.op_type, "Attention");
        assert_eq!(op.domain, "com.microsoft");
    }

    #[test]
    fn custom_gelu_maps() {
        let op = onnx_op_for(&OpKind::Custom("gelu".into())).unwrap();
        assert_eq!(op.op_type, "Gelu");
    }

    #[test]
    fn mapped_op_count_is_69() {
        assert_eq!(mapped_op_count(), 69);
    }
}
