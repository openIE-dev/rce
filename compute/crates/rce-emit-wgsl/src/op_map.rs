//! graph `OpKind` -> WGSL operator / builtin mapping.
//!
//! WGSL operators are infix (`a + b`), builtins are function calls (`exp(a)`),
//! and some custom ops expand to inline expressions (relu, sigmoid).

use rce_graph::graph::OpKind;

/// A WGSL op emission instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgslOp {
    /// Infix binary operator: `a {op} b`
    BinaryOp(&'static str),
    /// Unary prefix operator: `{op}a`
    UnaryOp(&'static str),
    /// Built-in unary function: `{name}(a)`
    BuiltinUnary(&'static str),
    /// Built-in binary function: `{name}(a, b)`
    BuiltinBinary(&'static str),
    /// Built-in ternary function: `{name}(a, b, c)`
    BuiltinTernary(&'static str),
    /// Inline expansion for relu: `max(a, 0.0)`
    Relu,
    /// Inline expansion for sigmoid: `1.0 / (1.0 + exp(-a))`
    Sigmoid,
    /// No WGSL equivalent.
    Unmapped,
}

/// Map a graph OpKind to a WGSL emission instruction.
pub fn wgsl_op_for(op: &OpKind) -> WgslOp {
    match op {
        // -- Arithmetic --
        OpKind::Add => WgslOp::BinaryOp("+"),
        OpKind::Sub => WgslOp::BinaryOp("-"),
        OpKind::Mul => WgslOp::BinaryOp("*"),
        OpKind::Div => WgslOp::BinaryOp("/"),
        OpKind::Rem => WgslOp::BinaryOp("%"),
        OpKind::Neg => WgslOp::UnaryOp("-"),
        // -- Comparison --
        OpKind::Eq => WgslOp::BinaryOp("=="),
        OpKind::Ne => WgslOp::BinaryOp("!="),
        OpKind::Lt => WgslOp::BinaryOp("<"),
        OpKind::Le => WgslOp::BinaryOp("<="),
        OpKind::Gt => WgslOp::BinaryOp(">"),
        OpKind::Ge => WgslOp::BinaryOp(">="),
        // -- Logic --
        OpKind::And => WgslOp::BinaryOp("&&"),
        OpKind::Or => WgslOp::BinaryOp("||"),
        OpKind::Not => WgslOp::UnaryOp("!"),
        // -- String concat: no WGSL equivalent --
        OpKind::Concat => WgslOp::Unmapped,
        // -- Custom ops --
        OpKind::Custom(name) => custom_wgsl_op(name),
    }
}

fn custom_wgsl_op(name: &str) -> WgslOp {
    match name {
        // -- Element-wise math builtins (unary) --
        "exp" | "Exp" => WgslOp::BuiltinUnary("exp"),
        "log" | "Log" => WgslOp::BuiltinUnary("log"),
        "sqrt" | "Sqrt" => WgslOp::BuiltinUnary("sqrt"),
        "abs" | "Abs" => WgslOp::BuiltinUnary("abs"),
        "ceil" | "Ceil" => WgslOp::BuiltinUnary("ceil"),
        "floor" | "Floor" => WgslOp::BuiltinUnary("floor"),
        "sin" | "Sin" => WgslOp::BuiltinUnary("sin"),
        "cos" | "Cos" => WgslOp::BuiltinUnary("cos"),
        "tanh" | "Tanh" => WgslOp::BuiltinUnary("tanh"),
        // -- Element-wise math builtins (binary) --
        "pow" | "Pow" => WgslOp::BuiltinBinary("pow"),
        "max" | "Max" | "maximum" => WgslOp::BuiltinBinary("max"),
        "min" | "Min" | "minimum" => WgslOp::BuiltinBinary("min"),
        // -- Element-wise math builtins (ternary) --
        "clamp" | "Clip" | "clip" => WgslOp::BuiltinTernary("clamp"),
        // -- Activation inline expansions --
        "relu" | "Relu" => WgslOp::Relu,
        "sigmoid" | "Sigmoid" => WgslOp::Sigmoid,
        // -- matmul needs tiled kernel (future phase) --
        "matmul" | "MatMul" => WgslOp::Unmapped,
        // -- No match --
        _ => WgslOp::Unmapped,
    }
}

/// Count how many ops have a non-Unmapped mapping.
pub fn mapped_wgsl_count() -> usize {
    CORE_MAPPED + CUSTOM_MAPPED
}

// 15 core ops (all except Concat) + Neg/Not
const CORE_MAPPED: usize = 15;
// exp, log, sqrt, abs, ceil, floor, sin, cos, tanh, pow, max, min, clamp, relu, sigmoid
const CUSTOM_MAPPED: usize = 15;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_maps_to_plus() {
        assert_eq!(wgsl_op_for(&OpKind::Add), WgslOp::BinaryOp("+"));
    }

    #[test]
    fn sub_maps_to_minus() {
        assert_eq!(wgsl_op_for(&OpKind::Sub), WgslOp::BinaryOp("-"));
    }

    #[test]
    fn eq_maps_to_double_equals() {
        assert_eq!(wgsl_op_for(&OpKind::Eq), WgslOp::BinaryOp("=="));
    }

    #[test]
    fn ne_maps_to_bang_equals() {
        assert_eq!(wgsl_op_for(&OpKind::Ne), WgslOp::BinaryOp("!="));
    }

    #[test]
    fn neg_maps_to_unary_minus() {
        assert_eq!(wgsl_op_for(&OpKind::Neg), WgslOp::UnaryOp("-"));
    }

    #[test]
    fn not_maps_to_unary_bang() {
        assert_eq!(wgsl_op_for(&OpKind::Not), WgslOp::UnaryOp("!"));
    }

    #[test]
    fn and_maps_to_double_amp() {
        assert_eq!(wgsl_op_for(&OpKind::And), WgslOp::BinaryOp("&&"));
    }

    #[test]
    fn custom_exp_maps_to_builtin() {
        assert_eq!(wgsl_op_for(&OpKind::Custom("exp".into())), WgslOp::BuiltinUnary("exp"));
    }

    #[test]
    fn custom_pow_maps_to_builtin_binary() {
        assert_eq!(wgsl_op_for(&OpKind::Custom("pow".into())), WgslOp::BuiltinBinary("pow"));
    }

    #[test]
    fn custom_clamp_maps_to_builtin_ternary() {
        assert_eq!(wgsl_op_for(&OpKind::Custom("clamp".into())), WgslOp::BuiltinTernary("clamp"));
    }

    #[test]
    fn custom_relu_maps_to_relu() {
        assert_eq!(wgsl_op_for(&OpKind::Custom("relu".into())), WgslOp::Relu);
    }

    #[test]
    fn custom_sigmoid_maps_to_sigmoid() {
        assert_eq!(wgsl_op_for(&OpKind::Custom("sigmoid".into())), WgslOp::Sigmoid);
    }

    #[test]
    fn custom_matmul_is_unmapped() {
        assert_eq!(wgsl_op_for(&OpKind::Custom("matmul".into())), WgslOp::Unmapped);
    }

    #[test]
    fn concat_string_is_unmapped() {
        assert_eq!(wgsl_op_for(&OpKind::Concat), WgslOp::Unmapped);
    }

    #[test]
    fn hdc_op_is_unmapped() {
        assert_eq!(wgsl_op_for(&OpKind::Custom("hdc.bundle".into())), WgslOp::Unmapped);
    }

    #[test]
    fn mapped_count_is_30() {
        assert_eq!(mapped_wgsl_count(), 30);
    }
}
