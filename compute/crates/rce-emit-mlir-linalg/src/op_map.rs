//! graph `OpKind` -> MLIR linalg/arith/math op name mapping.
//!
//! Unlike StableHLO (which uses a single `stablehlo.add` for both int and
//! float), the arith dialect requires choosing between `arith.addi` (integer)
//! and `arith.addf` (floating-point). This module encodes those distinctions.
//!
//! Dialect coverage:
//! - `arith.*`  -- integer and float arithmetic, constants, comparisons, logic
//! - `math.*`   -- transcendental and elementary math (exp, log, sin, ...)
//! - `linalg.*` -- structured computation (matmul, conv, generic)
//! - `tensor.*` -- tensor manipulation (extract, insert, reshape, ...)

use rce_graph::graph::OpKind;

/// An MLIR linalg-stack emission instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinalgOp {
    /// Binary op with int and float variants: `arith.{int_name}` / `arith.{float_name}`.
    /// The emitter picks based on the output type.
    ArithBinary {
        int_op: &'static str,
        float_op: &'static str,
    },
    /// Unary op with int and float variants.
    ArithUnary {
        int_op: Option<&'static str>,
        float_op: &'static str,
    },
    /// Float comparison: `arith.cmpf "{predicate}"`.
    CmpFloat(&'static str),
    /// Integer comparison: `arith.cmpi "{predicate}"`.
    CmpInt(&'static str),
    /// Comparison with both int and float predicates (emitter picks based on input type).
    Cmp {
        int_pred: &'static str,
        float_pred: &'static str,
    },
    /// Logical binary: `arith.{name}` (always integer/i1).
    LogicBinary(&'static str),
    /// Logical NOT: `arith.xori %x, %all_ones`.
    LogicNot,
    /// A `math.*` unary op.
    MathUnary(&'static str),
    /// A `math.*` binary op.
    MathBinary(&'static str),
    /// A `linalg.*` named op (e.g., `linalg.matmul`).
    LinalgNamed(&'static str),
    /// ReLU: `arith.maximumf %x, %zero` (float) or `arith.maxsi %x, %zero` (int).
    Relu,
    /// No mapping in the linalg/arith/math dialect stack.
    Unmapped,
}

/// Map a graph OpKind to an MLIR linalg-stack emission instruction.
pub fn linalg_op_for(op: &OpKind) -> LinalgOp {
    match op {
        // -- Arithmetic ------------------------------------------------
        OpKind::Add => LinalgOp::ArithBinary {
            int_op: "arith.addi",
            float_op: "arith.addf",
        },
        OpKind::Sub => LinalgOp::ArithBinary {
            int_op: "arith.subi",
            float_op: "arith.subf",
        },
        OpKind::Mul => LinalgOp::ArithBinary {
            int_op: "arith.muli",
            float_op: "arith.mulf",
        },
        OpKind::Div => LinalgOp::ArithBinary {
            int_op: "arith.divsi",
            float_op: "arith.divf",
        },
        OpKind::Rem => LinalgOp::ArithBinary {
            int_op: "arith.remsi",
            float_op: "arith.remf",
        },
        OpKind::Neg => LinalgOp::ArithUnary {
            int_op: None, // integer neg: 0 - x (emitter handles)
            float_op: "arith.negf",
        },
        // -- Comparison ------------------------------------------------
        OpKind::Eq => LinalgOp::Cmp {
            int_pred: "eq",
            float_pred: "oeq",
        },
        OpKind::Ne => LinalgOp::Cmp {
            int_pred: "ne",
            float_pred: "one",
        },
        OpKind::Lt => LinalgOp::Cmp {
            int_pred: "slt",
            float_pred: "olt",
        },
        OpKind::Le => LinalgOp::Cmp {
            int_pred: "sle",
            float_pred: "ole",
        },
        OpKind::Gt => LinalgOp::Cmp {
            int_pred: "sgt",
            float_pred: "ogt",
        },
        OpKind::Ge => LinalgOp::Cmp {
            int_pred: "sge",
            float_pred: "oge",
        },
        // -- Logic -----------------------------------------------------
        OpKind::And => LinalgOp::LogicBinary("arith.andi"),
        OpKind::Or => LinalgOp::LogicBinary("arith.ori"),
        OpKind::Not => LinalgOp::LogicNot,
        // -- String concat: no MLIR equivalent -------------------------
        OpKind::Concat => LinalgOp::Unmapped,
        // -- Custom ops ------------------------------------------------
        OpKind::Custom(name) => custom_linalg_op(name),
    }
}

fn custom_linalg_op(name: &str) -> LinalgOp {
    match name {
        // -- Linear algebra (linalg named ops) -------------------------
        "matmul" | "MatMul" => LinalgOp::LinalgNamed("linalg.matmul"),
        "gemm" | "Gemm" => LinalgOp::LinalgNamed("linalg.matmul"),
        "conv" | "Conv" => LinalgOp::LinalgNamed("linalg.conv_2d_nhwc_hwcf"),
        "conv_transpose" | "ConvTranspose" => LinalgOp::LinalgNamed("linalg.conv_2d_nhwc_hwcf"),
        "batch_matmul" | "BatchMatMul" => LinalgOp::LinalgNamed("linalg.batch_matmul"),
        "matvec" | "MatVec" => LinalgOp::LinalgNamed("linalg.matvec"),
        "dot" | "Dot" => LinalgOp::LinalgNamed("linalg.dot"),

        // -- Element-wise math (math dialect) --------------------------
        "exp" | "Exp" => LinalgOp::MathUnary("math.exp"),
        "log" | "Log" => LinalgOp::MathUnary("math.log"),
        "sqrt" | "Sqrt" => LinalgOp::MathUnary("math.sqrt"),
        "rsqrt" | "Rsqrt" => LinalgOp::MathUnary("math.rsqrt"),
        "abs" | "Abs" => LinalgOp::MathUnary("math.absf"),
        "ceil" | "Ceil" => LinalgOp::MathUnary("math.ceil"),
        "floor" | "Floor" => LinalgOp::MathUnary("math.floor"),
        "sin" | "Sin" => LinalgOp::MathUnary("math.sin"),
        "cos" | "Cos" => LinalgOp::MathUnary("math.cos"),
        "tanh" | "Tanh" => LinalgOp::MathUnary("math.tanh"),
        "erf" | "Erf" => LinalgOp::MathUnary("math.erf"),
        "log2" | "Log2" => LinalgOp::MathUnary("math.log2"),
        "log10" | "Log10" => LinalgOp::MathUnary("math.log10"),
        "exp2" | "Exp2" => LinalgOp::MathUnary("math.exp2"),
        "atan" | "Atan" => LinalgOp::MathUnary("math.atan"),
        "atan2" | "Atan2" => LinalgOp::MathBinary("math.atan2"),
        "pow" | "Pow" => LinalgOp::MathBinary("math.powf"),
        "copysign" | "CopySign" => LinalgOp::MathBinary("math.copysign"),

        // -- Activations -----------------------------------------------
        "relu" | "Relu" => LinalgOp::Relu,

        // -- arith max/min ---------------------------------------------
        "max" | "Max" | "maximum" => LinalgOp::ArithBinary {
            int_op: "arith.maxsi",
            float_op: "arith.maximumf",
        },
        "min" | "Min" | "minimum" => LinalgOp::ArithBinary {
            int_op: "arith.minsi",
            float_op: "arith.minimumf",
        },

        // -- No match --------------------------------------------------
        _ => LinalgOp::Unmapped,
    }
}

/// Count how many ops have a non-Unmapped mapping.
pub fn mapped_linalg_count() -> usize {
    CORE_MAPPED + CUSTOM_MAPPED
}

// 16 core OpKind variants: Add,Sub,Mul,Div,Rem,Neg,Eq,Ne,Lt,Le,Gt,Ge,And,Or,Not + Concat(unmapped) = 15 mapped
const CORE_MAPPED: usize = 15;
// Custom ops: matmul,gemm,conv,conv_transpose,batch_matmul,matvec,dot (7)
//   + exp,log,sqrt,rsqrt,abs,ceil,floor,sin,cos,tanh,erf,log2,log10,exp2,atan (15)
//   + atan2,pow,copysign (3)
//   + relu (1) + max,min (2) = 28
const CUSTOM_MAPPED: usize = 28;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_maps_to_arith_binary() {
        assert_eq!(
            linalg_op_for(&OpKind::Add),
            LinalgOp::ArithBinary {
                int_op: "arith.addi",
                float_op: "arith.addf",
            }
        );
    }

    #[test]
    fn neg_maps_to_arith_unary() {
        assert_eq!(
            linalg_op_for(&OpKind::Neg),
            LinalgOp::ArithUnary {
                int_op: None,
                float_op: "arith.negf",
            }
        );
    }

    #[test]
    fn eq_maps_to_cmp() {
        assert_eq!(
            linalg_op_for(&OpKind::Eq),
            LinalgOp::Cmp {
                int_pred: "eq",
                float_pred: "oeq",
            }
        );
    }

    #[test]
    fn gt_maps_to_cmp() {
        assert_eq!(
            linalg_op_for(&OpKind::Gt),
            LinalgOp::Cmp {
                int_pred: "sgt",
                float_pred: "ogt",
            }
        );
    }

    #[test]
    fn and_maps_to_logic() {
        assert_eq!(
            linalg_op_for(&OpKind::And),
            LinalgOp::LogicBinary("arith.andi")
        );
    }

    #[test]
    fn not_maps_to_logic_not() {
        assert_eq!(linalg_op_for(&OpKind::Not), LinalgOp::LogicNot);
    }

    #[test]
    fn custom_matmul_maps_to_linalg() {
        assert_eq!(
            linalg_op_for(&OpKind::Custom("matmul".into())),
            LinalgOp::LinalgNamed("linalg.matmul")
        );
    }

    #[test]
    fn custom_exp_maps_to_math() {
        assert_eq!(
            linalg_op_for(&OpKind::Custom("exp".into())),
            LinalgOp::MathUnary("math.exp")
        );
    }

    #[test]
    fn custom_pow_maps_to_math_binary() {
        assert_eq!(
            linalg_op_for(&OpKind::Custom("pow".into())),
            LinalgOp::MathBinary("math.powf")
        );
    }

    #[test]
    fn custom_relu_maps_to_relu() {
        assert_eq!(
            linalg_op_for(&OpKind::Custom("relu".into())),
            LinalgOp::Relu
        );
    }

    #[test]
    fn custom_conv_maps_to_linalg() {
        assert_eq!(
            linalg_op_for(&OpKind::Custom("conv".into())),
            LinalgOp::LinalgNamed("linalg.conv_2d_nhwc_hwcf")
        );
    }

    #[test]
    fn concat_string_is_unmapped() {
        assert_eq!(linalg_op_for(&OpKind::Concat), LinalgOp::Unmapped);
    }

    #[test]
    fn hdc_op_is_unmapped() {
        assert_eq!(
            linalg_op_for(&OpKind::Custom("hdc.bundle".into())),
            LinalgOp::Unmapped
        );
    }

    #[test]
    fn mapped_count() {
        assert_eq!(mapped_linalg_count(), 43);
    }
}
