//! PyTorch target string to graph OpKind mapping table.

use rce_graph::graph::OpKind;

/// Map a `call_function` target string to a graph `OpKind`.
///
/// Returns `None` for targets that have no direct mapping (caller should
/// fall back to `Custom(target)`).
pub fn map_function_target(target: &str) -> OpKind {
    match target {
        // ── Arithmetic ──────────────────────────────────────────
        "torch.add" | "operator.add" | "torch.ops.aten.add" | "torch.ops.aten.add.Tensor" => {
            OpKind::Add
        }
        "torch.sub" | "operator.sub" | "torch.ops.aten.sub" | "torch.ops.aten.sub.Tensor" => {
            OpKind::Sub
        }
        "torch.mul" | "operator.mul" | "torch.ops.aten.mul" | "torch.ops.aten.mul.Tensor" => {
            OpKind::Mul
        }
        "torch.div"
        | "torch.true_divide"
        | "operator.truediv"
        | "torch.ops.aten.div"
        | "torch.ops.aten.div.Tensor" => OpKind::Div,
        "torch.remainder" | "operator.mod" | "torch.ops.aten.remainder" => OpKind::Rem,
        "torch.neg" | "operator.neg" | "torch.ops.aten.neg" => OpKind::Neg,

        // ── Comparison ──────────────────────────────────────────
        "torch.eq" | "operator.eq" | "torch.ops.aten.eq" | "torch.ops.aten.eq.Tensor" => {
            OpKind::Eq
        }
        "torch.ne" | "operator.ne" | "torch.ops.aten.ne" | "torch.ops.aten.ne.Tensor" => {
            OpKind::Ne
        }
        "torch.lt" | "operator.lt" | "torch.ops.aten.lt" | "torch.ops.aten.lt.Tensor" => {
            OpKind::Lt
        }
        "torch.le" | "operator.le" | "torch.ops.aten.le" | "torch.ops.aten.le.Tensor" => {
            OpKind::Le
        }
        "torch.gt" | "operator.gt" | "torch.ops.aten.gt" | "torch.ops.aten.gt.Tensor" => {
            OpKind::Gt
        }
        "torch.ge" | "operator.ge" | "torch.ops.aten.ge" | "torch.ops.aten.ge.Tensor" => {
            OpKind::Ge
        }

        // ── Logic ───────────────────────────────────────────────
        "torch.logical_and" | "operator.and_" | "torch.ops.aten.logical_and" => OpKind::And,
        "torch.logical_or" | "operator.or_" | "torch.ops.aten.logical_or" => OpKind::Or,
        "torch.logical_not" | "operator.not_" | "torch.ops.aten.logical_not" => OpKind::Not,

        // ── String ──────────────────────────────────────────────
        "torch.cat" | "torch.ops.aten.cat" => OpKind::Concat,

        // ── NN functional ───────────────────────────────────────
        "torch.nn.functional.relu" | "torch.ops.aten.relu" => {
            OpKind::Custom("relu".into())
        }
        "torch.nn.functional.sigmoid" | "torch.sigmoid" | "torch.ops.aten.sigmoid" => {
            OpKind::Custom("sigmoid".into())
        }
        "torch.nn.functional.tanh" | "torch.tanh" | "torch.ops.aten.tanh" => {
            OpKind::Custom("tanh".into())
        }
        "torch.nn.functional.softmax" | "torch.ops.aten.softmax" => {
            OpKind::Custom("softmax".into())
        }
        "torch.nn.functional.log_softmax" | "torch.ops.aten.log_softmax" => {
            OpKind::Custom("log_softmax".into())
        }
        "torch.nn.functional.gelu" | "torch.ops.aten.gelu" => {
            OpKind::Custom("gelu".into())
        }
        "torch.nn.functional.silu" | "torch.ops.aten.silu" => {
            OpKind::Custom("silu".into())
        }
        "torch.nn.functional.leaky_relu" | "torch.ops.aten.leaky_relu" => {
            OpKind::Custom("leaky_relu".into())
        }
        "torch.nn.functional.elu" | "torch.ops.aten.elu" => {
            OpKind::Custom("elu".into())
        }
        "torch.nn.functional.dropout" | "torch.ops.aten.dropout" => {
            OpKind::Custom("dropout".into())
        }
        "torch.nn.functional.linear" | "torch.ops.aten.linear" => {
            OpKind::Custom("linear".into())
        }
        "torch.nn.functional.conv2d" | "torch.ops.aten.conv2d" => {
            OpKind::Custom("conv2d".into())
        }
        "torch.nn.functional.batch_norm" | "torch.ops.aten.batch_norm" => {
            OpKind::Custom("batch_norm".into())
        }
        "torch.nn.functional.layer_norm" | "torch.ops.aten.layer_norm" => {
            OpKind::Custom("layer_norm".into())
        }
        "torch.nn.functional.embedding" | "torch.ops.aten.embedding" => {
            OpKind::Custom("embedding".into())
        }
        "torch.nn.functional.cross_entropy" | "torch.ops.aten.cross_entropy_loss" => {
            OpKind::Custom("cross_entropy".into())
        }

        // ── Matrix / tensor ops ─────────────────────────────────
        "torch.matmul" | "torch.ops.aten.matmul" | "torch.ops.aten.mm" => {
            OpKind::Custom("matmul".into())
        }
        "torch.mm" | "torch.bmm" | "torch.ops.aten.bmm" => {
            OpKind::Custom("matmul".into())
        }
        "torch.transpose" | "torch.ops.aten.transpose" | "torch.ops.aten.t" => {
            OpKind::Custom("transpose".into())
        }

        // ── Elementwise math ────────────────────────────────────
        "torch.exp" | "torch.ops.aten.exp" => OpKind::Custom("exp".into()),
        "torch.log" | "torch.ops.aten.log" => OpKind::Custom("log".into()),
        "torch.sqrt" | "torch.ops.aten.sqrt" => OpKind::Custom("sqrt".into()),
        "torch.abs" | "torch.ops.aten.abs" => OpKind::Custom("abs".into()),
        "torch.pow" | "torch.ops.aten.pow" | "torch.ops.aten.pow.Tensor_Scalar" => {
            OpKind::Custom("pow".into())
        }
        "torch.clamp" | "torch.ops.aten.clamp" => OpKind::Custom("clamp".into()),

        // ── Reduction ───────────────────────────────────────────
        "torch.sum" | "torch.ops.aten.sum" => OpKind::Custom("sum".into()),
        "torch.mean" | "torch.ops.aten.mean" => OpKind::Custom("mean".into()),
        "torch.max" | "torch.ops.aten.max" => OpKind::Custom("max".into()),
        "torch.min" | "torch.ops.aten.min" => OpKind::Custom("min".into()),
        "torch.argmax" | "torch.ops.aten.argmax" => OpKind::Custom("argmax".into()),

        // ── Shape manipulation ──────────────────────────────────
        "torch.reshape" | "torch.ops.aten.reshape" => OpKind::Custom("reshape".into()),
        "torch.flatten" | "torch.ops.aten.flatten" => OpKind::Custom("flatten".into()),
        "torch.unsqueeze" | "torch.ops.aten.unsqueeze" => {
            OpKind::Custom("unsqueeze".into())
        }
        "torch.squeeze" | "torch.ops.aten.squeeze" => OpKind::Custom("squeeze".into()),

        // ── Fallback: keep the full target as Custom ────────────
        other => OpKind::Custom(other.into()),
    }
}

/// Map a `call_method` method name to a graph `OpKind`.
pub fn map_method_name(method: &str) -> OpKind {
    match method {
        "add" | "add_" => OpKind::Add,
        "sub" | "sub_" => OpKind::Sub,
        "mul" | "mul_" => OpKind::Mul,
        "div" | "div_" => OpKind::Div,

        "relu" | "relu_" => OpKind::Custom("relu".into()),
        "sigmoid" | "sigmoid_" => OpKind::Custom("sigmoid".into()),
        "tanh" | "tanh_" => OpKind::Custom("tanh".into()),
        "softmax" => OpKind::Custom("softmax".into()),

        "view" | "reshape" => OpKind::Custom("reshape".into()),
        "permute" => OpKind::Custom("transpose".into()),
        "transpose" | "t" => OpKind::Custom("transpose".into()),
        "contiguous" => OpKind::Custom("contiguous".into()),
        "flatten" => OpKind::Custom("flatten".into()),
        "unsqueeze" => OpKind::Custom("unsqueeze".into()),
        "squeeze" => OpKind::Custom("squeeze".into()),
        "expand" => OpKind::Custom("expand".into()),
        "repeat" => OpKind::Custom("repeat".into()),

        "sum" => OpKind::Custom("sum".into()),
        "mean" => OpKind::Custom("mean".into()),
        "max" => OpKind::Custom("max".into()),
        "min" => OpKind::Custom("min".into()),

        "matmul" | "mm" | "bmm" => OpKind::Custom("matmul".into()),
        "to" | "float" | "half" | "int" | "long" => OpKind::Custom("cast".into()),
        "detach" => OpKind::Custom("detach".into()),
        "clone" => OpKind::Custom("clone".into()),
        "item" => OpKind::Custom("item".into()),
        "size" | "shape" => OpKind::Custom("shape".into()),

        other => OpKind::Custom(other.into()),
    }
}
