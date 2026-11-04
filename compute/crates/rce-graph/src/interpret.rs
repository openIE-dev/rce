//! Interpret a FunctionGraph directly.
//!
//! A tree-walking interpreter over the graph nodes + wires. It runs lowered
//! graphs without the inv-ai-compiler Cranelift backend, and is the live
//! execution engine for flow-ide, jmax-cli, jmax-wasm, and jmax-rce — so
//! the semantics below are user-facing, not merely a test convenience.
//!
//! # Execution contract (lenient runtime)
//!
//! Value evaluation degrades on type/op mismatches but treats division by zero
//! as a real, backend-consistent outcome (no fabricated 0):
//! - **Type mismatches coerce** rather than fail: non-numeric → `0` / `0.0`,
//!   non-bool → `false`, non-string → `""` (see the `Value::as_*` helpers).
//! - **Float division / remainder by zero is total (IEEE-754)**: `x/0.0` = ±inf,
//!   `0.0/0.0` = NaN — identical to the compiled `fdiv` backend.
//! - **Integer division / remainder by zero is a fault**: there is no defined
//!   integer value and the compiled backend traps (sdiv/srem). `interpret_strict`
//!   returns `Err(DivisionByZero)`; lenient `interpret` faults deterministically
//!   (panics) rather than fabricating 0. Route user code through `interpret_strict`
//!   for graceful handling.
//! - **Unknown / unhandled ops → 0** (the zero/empty of the result type).
//! - **`ErrorPropagate` on the non-recoverable path → 0** (it does not halt).
//! - **`WhileLoop` is capped at 100k iterations**, then returns current state.
//! - **Enums** pack as `i64 = (tag << 32) | (payload & 0xFFFF_FFFF)` — a single
//!   32-bit integer payload; richer payloads are not represented.
//!
//! Division is now backend-consistent: float follows IEEE on both paths, and
//! integer div-by-zero faults on both (lenient panics; the compiled backend
//! traps). The remaining intentional divergences from a compiled
//! (Cranelift) backend are enum encoding (string-keyed `Custom` ops) and the
//! 100k loop cap. Treat `interpret` as the reference *runtime*;
//! `interpret_strict` is the checked oracle that surfaces faults as
//! `EvalError` instead of panicking.
//!
//! Structural faults (dangling wires, missing required input ports) are a
//! separate concern: they would panic here, so run untrusted graphs through
//! [`crate::graph::FunctionGraph::validate`] or [`interpret_checked`] first.

use std::collections::HashMap;

use crate::graph::*;

/// Runtime value during interpretation.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    Record(Vec<(String, Value)>),
    Unit,
}

impl Value {
    pub fn as_i64(&self) -> i64 {
        match self {
            Value::I64(v) => *v,
            Value::Bool(b) => if *b { 1 } else { 0 },
            Value::F64(v) => *v as i64,
            _ => 0,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Value::F64(v) => *v,
            Value::I64(v) => *v as f64,
            _ => 0.0,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::I64(v) => *v != 0,
            _ => false,
        }
    }

    pub fn as_str(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::I64(v) => v.to_string(),
            Value::F64(v) => v.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => String::new(),
        }
    }
}

/// Pre-built wire index for O(1) input lookup.
struct WireIndex {
    by_target: HashMap<usize, Vec<(usize, i32)>>,
}

impl WireIndex {
    fn build(wires: &[Wire]) -> Self {
        let mut by_target: HashMap<usize, Vec<(usize, i32)>> = HashMap::new();
        for w in wires {
            by_target.entry(w.to).or_default().push((w.from, w.port_in_idx));
        }
        for entries in by_target.values_mut() {
            entries.sort_by_key(|(_, port)| *port);
        }
        Self { by_target }
    }

    fn input_at_port(&self, node_idx: usize, port: i32) -> Option<usize> {
        self.by_target.get(&node_idx).and_then(|entries| {
            entries.iter().find(|(_, p)| *p == port).map(|(src, _)| *src)
        })
    }

    fn inputs_sorted(&self, node_idx: usize) -> &[(usize, i32)] {
        match self.by_target.get(&node_idx) {
            Some(entries) => entries,
            None => &[],
        }
    }
}

/// Interpret a function graph with the given arguments.
pub fn interpret(
    graph: &FunctionGraph,
    args: &[Value],
    program: Option<&ProgramGraph>,
) -> Value {
    let idx = WireIndex::build(&graph.wires);
    let mut values: HashMap<usize, Value> = HashMap::new();

    // Assign param values
    let mut arg_iter = args.iter();
    for (i, node) in graph.nodes.iter().enumerate() {
        if matches!(node, GraphNode::Param(_)) {
            if let Some(val) = arg_iter.next() {
                values.insert(i, val.clone());
            }
        }
    }

    eval_node(graph.output_node, &graph.nodes, &idx, &mut values, program)
}

/// Validate the graph structurally, then interpret it. Use this at any
/// boundary where the graph comes from untrusted JSON: a structural fault
/// (dangling wire, missing required input port) returns a typed
/// [`GraphError`] instead of panicking inside [`interpret`]. When a
/// `program` is supplied its functions are all validated, since `Call`
/// nodes recurse into them.
pub fn interpret_checked(
    graph: &FunctionGraph,
    args: &[Value],
    program: Option<&ProgramGraph>,
) -> Result<Value, GraphError> {
    match program {
        Some(p) => p.validate()?,
        None => graph.validate()?,
    }
    Ok(interpret(graph, args, program))
}

// ── Strict evaluation mode ──────────────────────────────────────────

/// A value-level fault that [`interpret_strict`] reports instead of silently
/// degrading (the way the default lenient runtime does — see the module-level
/// execution contract).
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    /// Integer or float division / remainder by zero.
    DivisionByZero,
    /// An operator applied in a value context it does not support
    /// (e.g. `And` on floats, `Concat` on ints).
    UnsupportedOp(String),
    /// A non-recoverable `ErrorPropagate` node was reached.
    Unrecoverable(String),
    /// A `WhileLoop` hit the 100k-iteration safety cap without its condition
    /// going false.
    LoopLimitExceeded,
    /// The graph failed structural validation before execution.
    Structural(crate::graph::GraphError),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::DivisionByZero => write!(f, "division by zero"),
            EvalError::UnsupportedOp(s) => write!(f, "unsupported operation: {s}"),
            EvalError::Unrecoverable(k) => write!(f, "unrecoverable error: {k}"),
            EvalError::LoopLimitExceeded => write!(f, "while-loop exceeded 100k iterations"),
            EvalError::Structural(e) => write!(f, "structural fault: {e}"),
        }
    }
}

impl std::error::Error for EvalError {}

thread_local! {
    /// When set, eval records the first divergence into `STRICT_TRAP` instead
    /// of silently degrading. Thread-local so the deep recursive evaluator
    /// needs no extra parameter and lenient `interpret` stays byte-for-byte
    /// unchanged. The interpreter is single-threaded by contract.
    static STRICT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static STRICT_TRAP: std::cell::RefCell<Option<EvalError>> =
        const { std::cell::RefCell::new(None) };
}

/// Record a strict-mode divergence (first one wins). No-op in lenient mode.
fn strict_trap(err: EvalError) {
    if STRICT.with(|s| s.get()) {
        STRICT_TRAP.with(|t| {
            let mut t = t.borrow_mut();
            if t.is_none() {
                *t = Some(err);
            }
        });
    }
}

/// Integer division/remainder by zero is a genuine fault: unlike float
/// (IEEE-total), there is no defined integer value, and the compiled backend
/// traps (sdiv/srem). We never fabricate a 0.
///
/// - **Strict mode**: record `DivisionByZero` and return 0 so the surrounding
///   evaluation finishes and `interpret_strict` can surface `Err(..)`.
/// - **Lenient mode**: fault deterministically (panic), matching the compiled
///   backend's trap. Callers that want graceful handling use `interpret_strict`.
fn int_div_by_zero() -> i64 {
    if STRICT.with(|s| s.get()) {
        strict_trap(EvalError::DivisionByZero);
        0
    } else {
        panic!("integer division by zero");
    }
}

/// Interpret in strict mode: validate structure, then evaluate, returning the
/// first value-level divergence (division by zero, unsupported op,
/// unrecoverable error, loop-limit) as an [`EvalError`] instead of silently
/// returning a degraded value. Use this as a correctness oracle; the lenient
/// [`interpret`] remains the production runtime.
pub fn interpret_strict(
    graph: &FunctionGraph,
    args: &[Value],
    program: Option<&ProgramGraph>,
) -> Result<Value, EvalError> {
    match program {
        Some(p) => p.validate(),
        None => graph.validate(),
    }
    .map_err(EvalError::Structural)?;

    // Restore lenient mode + clear the trap on the way out, even on panic.
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            STRICT.with(|s| s.set(false));
            STRICT_TRAP.with(|t| *t.borrow_mut() = None);
        }
    }
    let _guard = Guard;

    STRICT.with(|s| s.set(true));
    STRICT_TRAP.with(|t| *t.borrow_mut() = None);
    let value = interpret(graph, args, program);
    match STRICT_TRAP.with(|t| t.borrow_mut().take()) {
        Some(e) => Err(e),
        None => Ok(value),
    }
}

fn eval_node(
    node_idx: usize,
    nodes: &[GraphNode],
    idx: &WireIndex,
    values: &mut HashMap<usize, Value>,
    program: Option<&ProgramGraph>,
) -> Value {
    if let Some(v) = values.get(&node_idx) {
        return v.clone();
    }

    let result = match &nodes[node_idx] {
        GraphNode::Param(_) => Value::I64(0), // default if no arg provided

        GraphNode::Literal { value, .. } => match value {
            LiteralValue::I64(v) => Value::I64(*v),
            LiteralValue::I32(v) => Value::I64(*v as i64),
            LiteralValue::F64(v) => Value::F64(*v),
            LiteralValue::F32(v) => Value::F64(*v as f64),
            LiteralValue::Bool(v) => Value::Bool(*v),
            LiteralValue::String(s) => Value::Str(s.clone()),
        },

        GraphNode::Apply { op, output_type } => {
            let inputs = idx.inputs_sorted(node_idx);
            let lhs = if !inputs.is_empty() {
                eval_node(inputs[0].0, nodes, idx, values, program)
            } else {
                Value::I64(0)
            };

            // Unary ops
            if inputs.len() == 1 {
                return match op {
                    OpKind::Neg => match &lhs {
                        Value::I64(v) => Value::I64(-v),
                        Value::F64(v) => Value::F64(-v),
                        _ => Value::I64(0),
                    },
                    OpKind::Not => Value::Bool(!lhs.as_bool()),
                    OpKind::Custom(name) => eval_custom_unary(name, &lhs),
                    _ => lhs,
                };
            }

            let rhs = eval_node(inputs[1].0, nodes, idx, values, program);

            // Float path
            if output_type == "f64" || matches!((&lhs, &rhs), (Value::F64(_), _) | (_, Value::F64(_))) {
                let l = lhs.as_f64();
                let r = rhs.as_f64();
                return match op {
                    OpKind::Add => Value::F64(l + r),
                    OpKind::Sub => Value::F64(l - r),
                    OpKind::Mul => Value::F64(l * r),
                    // Float division is total under IEEE-754: x/0.0 = ±inf,
                    // 0.0/0.0 = NaN — matching the compiled fdiv backend. Not a
                    // fault; compute directly rather than fabricating 0.
                    OpKind::Div => Value::F64(l / r),
                    OpKind::Rem => Value::F64(l % r),
                    OpKind::Eq => Value::Bool(l == r),
                    OpKind::Ne => Value::Bool(l != r),
                    OpKind::Lt => Value::Bool(l < r),
                    OpKind::Le => Value::Bool(l <= r),
                    OpKind::Gt => Value::Bool(l > r),
                    OpKind::Ge => Value::Bool(l >= r),
                    OpKind::Custom(name) => eval_custom_binary(name, &lhs, &rhs),
                    _ => {
                        strict_trap(EvalError::UnsupportedOp(format!("{op:?} on floats")));
                        Value::F64(0.0)
                    }
                };
            }

            // String path
            if matches!((&lhs, &rhs), (Value::Str(_), _) | (_, Value::Str(_))) {
                return match op {
                    OpKind::Add | OpKind::Concat => {
                        Value::Str(format!("{}{}", lhs.as_str(), rhs.as_str()))
                    }
                    OpKind::Eq => Value::Bool(lhs.as_str() == rhs.as_str()),
                    OpKind::Ne => Value::Bool(lhs.as_str() != rhs.as_str()),
                    _ => {
                        strict_trap(EvalError::UnsupportedOp(format!("{op:?} on strings")));
                        Value::Str(String::new())
                    }
                };
            }

            // Integer path
            let l = lhs.as_i64();
            let r = rhs.as_i64();
            match op {
                OpKind::Add => Value::I64(l + r),
                OpKind::Sub => Value::I64(l - r),
                OpKind::Mul => Value::I64(l * r),
                OpKind::Div => Value::I64(if r != 0 { l / r } else { int_div_by_zero() }),
                OpKind::Rem => Value::I64(if r != 0 { l % r } else { int_div_by_zero() }),
                OpKind::Eq => Value::Bool(l == r),
                OpKind::Ne => Value::Bool(l != r),
                OpKind::Lt => Value::Bool(l < r),
                OpKind::Le => Value::Bool(l <= r),
                OpKind::Gt => Value::Bool(l > r),
                OpKind::Ge => Value::Bool(l >= r),
                OpKind::And => Value::Bool(lhs.as_bool() && rhs.as_bool()),
                OpKind::Or => Value::Bool(lhs.as_bool() || rhs.as_bool()),
                OpKind::Custom(name) => eval_custom_binary(name, &lhs, &rhs),
                _ => {
                    strict_trap(EvalError::UnsupportedOp(format!("{op:?} on integers")));
                    Value::I64(0)
                }
            }
        }

        GraphNode::Select { .. } => {
            let cond_src = idx.input_at_port(node_idx, 0).unwrap();
            let cond = eval_node(cond_src, nodes, idx, values, program);
            if cond.as_bool() {
                let then_src = idx.input_at_port(node_idx, 1).unwrap();
                eval_node(then_src, nodes, idx, values, program)
            } else {
                let else_src = idx.input_at_port(node_idx, 2).unwrap();
                eval_node(else_src, nodes, idx, values, program)
            }
        }

        GraphNode::Call { callee, .. } => {
            // Collect args
            let inputs = idx.inputs_sorted(node_idx);
            let call_args: Vec<Value> = inputs
                .iter()
                .map(|(src, _)| eval_node(*src, nodes, idx, values, program))
                .collect();

            // Look up function in program
            if let Some(prog) = program {
                if let Some(func) = prog.functions.iter().find(|f| f.name == *callee) {
                    return interpret(func, &call_args, Some(prog));
                }
            }
            // Built-in: len
            if callee == "len" {
                if let Some(Value::Array(arr)) = call_args.first() {
                    return Value::I64(arr.len() as i64);
                }
            }
            Value::I64(0)
        }

        GraphNode::WhileLoop { loop_vars, condition, body, output_var: _ } => {
            // Collect initial values for loop vars
            let inputs = idx.inputs_sorted(node_idx);
            let mut loop_vals: Vec<Value> = inputs
                .iter()
                .map(|(src, _)| eval_node(*src, nodes, idx, values, program))
                .collect();

            // Pad if needed
            while loop_vals.len() < loop_vars.len() {
                loop_vals.push(Value::I64(0));
            }

            let max_iters = 100_000;
            let mut hit_cap = true;
            for _ in 0..max_iters {
                // Evaluate condition
                let cond_val = eval_subgraph(&condition, &loop_vals, program);
                if !cond_val[0].as_bool() {
                    hit_cap = false;
                    break;
                }
                // Evaluate body
                let new_vals = eval_subgraph(&body, &loop_vals, program);
                loop_vals = new_vals;
            }
            if hit_cap {
                strict_trap(EvalError::LoopLimitExceeded);
            }

            // Return all loop var values as an array.
            // Individual variables are extracted via IndexAccess nodes.
            Value::Array(loop_vals)
        }

        GraphNode::Match { case_values, .. } => {
            let disc_src = idx.input_at_port(node_idx, 0).unwrap();
            let disc = eval_node(disc_src, nodes, idx, values, program).as_i64();

            // Find matching case
            for (i, &cv) in case_values.iter().enumerate() {
                if disc == cv {
                    let arm_src = idx.input_at_port(node_idx, (i + 1) as i32).unwrap();
                    return eval_node(arm_src, nodes, idx, values, program);
                }
            }
            // Default: last arm after case arms
            let default_port = (case_values.len() + 1) as i32;
            if let Some(src) = idx.input_at_port(node_idx, default_port) {
                return eval_node(src, nodes, idx, values, program);
            }
            // Or just the last connected port
            let inputs = idx.inputs_sorted(node_idx);
            if let Some((src, _)) = inputs.last() {
                return eval_node(*src, nodes, idx, values, program);
            }
            Value::I64(0)
        }

        GraphNode::ArrayCreate { length, .. } => {
            let inputs = idx.inputs_sorted(node_idx);
            let mut elems = Vec::with_capacity(*length as usize);
            for (src, _) in inputs {
                elems.push(eval_node(*src, nodes, idx, values, program));
            }
            Value::Array(elems)
        }

        GraphNode::IndexAccess { .. } => {
            let arr_src = idx.input_at_port(node_idx, 0).unwrap();
            let idx_src = idx.input_at_port(node_idx, 1).unwrap();
            let arr = eval_node(arr_src, nodes, idx, values, program);
            let index = eval_node(idx_src, nodes, idx, values, program).as_i64();
            if let Value::Array(elems) = arr {
                if (index as usize) < elems.len() {
                    return elems[index as usize].clone();
                }
            }
            Value::I64(0)
        }

        GraphNode::StructCreate { field_names, .. } => {
            let inputs = idx.inputs_sorted(node_idx);
            let mut fields = Vec::new();
            for (i, (src, _)) in inputs.iter().enumerate() {
                let name = field_names.get(i).cloned().unwrap_or_default();
                fields.push((name, eval_node(*src, nodes, idx, values, program)));
            }
            Value::Record(fields)
        }

        GraphNode::FieldAccess { field_name, .. } => {
            let obj_src = idx.input_at_port(node_idx, 0).unwrap();
            let obj = eval_node(obj_src, nodes, idx, values, program);
            if let Value::Record(fields) = obj {
                for (name, val) in &fields {
                    if name == field_name {
                        return val.clone();
                    }
                }
            }
            Value::I64(0)
        }

        GraphNode::ErrorPropagate { error_kind, .. } => {
            if error_kind == "recoverable" {
                // Port 1 = ok value
                let ok_src = idx.input_at_port(node_idx, 1).unwrap();
                eval_node(ok_src, nodes, idx, values, program)
            } else {
                strict_trap(EvalError::Unrecoverable(error_kind.clone()));
                Value::I64(0) // lenient: degrade to 0; strict: trapped above
            }
        }

        GraphNode::Mutate { .. } => {
            // Port 1 = new value
            if let Some(src) = idx.input_at_port(node_idx, 1) {
                eval_node(src, nodes, idx, values, program)
            } else if let Some(src) = idx.input_at_port(node_idx, 0) {
                eval_node(src, nodes, idx, values, program)
            } else {
                Value::I64(0)
            }
        }

        GraphNode::Sequence { output_idx } => {
            let inputs = idx.inputs_sorted(node_idx);
            // Eval all inputs for side effects
            for (src, _) in inputs.iter() {
                eval_node(*src, nodes, idx, values, program);
            }
            // Return the one at output_idx
            if let Some((src, _)) = inputs.get(*output_idx) {
                eval_node(*src, nodes, idx, values, program)
            } else {
                Value::I64(0)
            }
        }

        // ── Enum support ────────────────────────────────────────────
        // Encoding: i64 = (tag << 32) | (payload & 0xFFFFFFFF)
        GraphNode::EnumCreate { variant_tag, .. } => {
            let payload = if let Some(src) = idx.input_at_port(node_idx, 0) {
                eval_node(src, nodes, idx, values, program).as_i64()
            } else {
                0
            };
            Value::I64((*variant_tag << 32) | (payload & 0xFFFFFFFF))
        }

        GraphNode::EnumTag { .. } => {
            let src = idx.input_at_port(node_idx, 0).unwrap();
            let val = eval_node(src, nodes, idx, values, program).as_i64();
            Value::I64(val >> 32)
        }

        GraphNode::EnumPayload { .. } => {
            let src = idx.input_at_port(node_idx, 0).unwrap();
            let val = eval_node(src, nodes, idx, values, program).as_i64();
            Value::I64(val & 0xFFFFFFFF)
        }

        // CodeBlock: evaluate inputs and return as a structured record
        GraphNode::CodeBlock { language, code_text, .. } => {
            let inputs = idx.inputs_sorted(node_idx);
            let data: Vec<Value> = inputs.iter()
                .map(|(src, _)| eval_node(*src, nodes, idx, values, program))
                .collect();

            // Return a record with metadata + evaluated data
            Value::Record(vec![
                ("__type".to_string(), Value::Str("code_block".to_string())),
                ("language".to_string(), Value::Str(language.clone())),
                ("kind".to_string(), Value::Str(code_text.clone())),
                ("data".to_string(), Value::Array(data)),
            ])
        }

        // Metadata nodes — no runtime value
        GraphNode::TypeDefine { .. } | GraphNode::VTableLookup { .. } => Value::I64(0),

        _ => Value::I64(0),
    };

    values.insert(node_idx, result.clone());
    result
}

/// Evaluate a sub-graph (used for loop condition/body).
fn eval_subgraph(
    sg: &LoopSubGraph,
    loop_var_values: &[Value],
    program: Option<&ProgramGraph>,
) -> Vec<Value> {
    let idx = WireIndex::build(&sg.wires);
    let mut values: HashMap<usize, Value> = HashMap::new();

    // Set param values = loop var values
    let mut arg_idx = 0;
    for (i, node) in sg.nodes.iter().enumerate() {
        if matches!(node, GraphNode::Param(_)) {
            if arg_idx < loop_var_values.len() {
                values.insert(i, loop_var_values[arg_idx].clone());
            }
            arg_idx += 1;
        }
    }

    // Evaluate each output node
    sg.output_nodes
        .iter()
        .map(|&out| eval_node(out, &sg.nodes, &idx, &mut values, program))
        .collect()
}

// ── Value ↔ matrix conversion helpers ──────────────────────────────

fn value_to_mat(v: &Value) -> Option<Vec<Vec<f64>>> {
    if let Value::Array(rows) = v {
        let mat: Vec<Vec<f64>> = rows.iter().map(|r| {
            if let Value::Array(cols) = r { cols.iter().map(|c| c.as_f64()).collect() }
            else { vec![r.as_f64()] }
        }).collect();
        Some(mat)
    } else { None }
}

fn mat_to_value(m: Vec<Vec<f64>>) -> Value {
    Value::Array(m.into_iter().map(|row| Value::Array(row.into_iter().map(Value::F64).collect())).collect())
}

fn value_to_vec(v: &Value) -> Option<Vec<f64>> {
    if let Value::Array(arr) = v { Some(arr.iter().map(|x| x.as_f64()).collect()) }
    else { None }
}

fn vec_to_value(v: Vec<f64>) -> Value {
    Value::Array(v.into_iter().map(Value::F64).collect())
}

// ── Custom OpKind math runtime ─────────────────────────────────────

/// Evaluate a unary Custom operation.
fn eval_custom_unary(name: &str, x: &Value) -> Value {
    use crate::math_runtime;
    let v = x.as_f64();
    match name {
        // Trigonometric
        "sin" => Value::F64(v.sin()),
        "cos" => Value::F64(v.cos()),
        "tan" => Value::F64(v.tan()),
        "asin" => Value::F64(v.asin()),
        "acos" => Value::F64(v.acos()),
        // Exponential / logarithmic
        "exp" => Value::F64(v.exp()),
        "log" | "ln" => Value::F64(v.ln()),
        "log2" => Value::F64(v.log2()),
        "log10" => Value::F64(v.log10()),
        "sqrt" => Value::F64(v.sqrt()),
        "abs" => Value::F64(v.abs()),
        // Rounding
        "floor" => Value::F64(v.floor()),
        "ceil" => Value::F64(v.ceil()),
        "round" => Value::F64(v.round()),
        // Matrix / vector unary — delegated to math_runtime for NxN
        "transpose" => {
            if let Some(m) = value_to_mat(x) { mat_to_value(math_runtime::transpose(&m)) }
            else { x.clone() }
        }
        "norm" => {
            if let Some(v) = value_to_vec(x) { Value::F64(math_runtime::vec_norm(&v)) }
            else { Value::F64(v.abs()) }
        }
        "determinant" | "det" => {
            if let Some(m) = value_to_mat(x) { Value::F64(math_runtime::determinant(&m)) }
            else { Value::F64(0.0) }
        }
        "inverse" | "inv" => {
            if let Some(m) = value_to_mat(x) {
                math_runtime::inverse(&m).map(mat_to_value).unwrap_or(Value::F64(0.0))
            } else { Value::F64(0.0) }
        }
        "eigenvalues" | "eig" | "eigendecompose" => {
            if let Some(m) = value_to_mat(x) { vec_to_value(math_runtime::eigenvalues_symmetric(&m, 200)) }
            else { Value::F64(0.0) }
        }
        "svd" => {
            if let Some(m) = value_to_mat(x) { vec_to_value(math_runtime::singular_values(&m, 200)) }
            else { Value::F64(0.0) }
        }
        "rank" => {
            if let Some(m) = value_to_mat(x) { Value::I64(math_runtime::rank(&m) as i64) }
            else { Value::I64(1) }
        }
        "trace" | "tr" => {
            if let Some(m) = value_to_mat(x) { Value::F64(math_runtime::trace(&m)) }
            else { Value::F64(v) }
        }
        "fft" => {
            if let Some(signal) = value_to_vec(x) {
                let spectrum = math_runtime::fft(&signal);
                let mags = math_runtime::magnitude_spectrum(&spectrum);
                vec_to_value(mags)
            } else { Value::F64(0.0) }
        }
        "gc_content" => {
            if let Value::Str(s) = x { Value::F64(math_runtime::gc_content(s.as_bytes())) }
            else { Value::F64(0.0) }
        }
        "reverse_complement" => {
            if let Value::Str(s) = x {
                let rc = math_runtime::reverse_complement(s.as_bytes());
                Value::Str(String::from_utf8_lossy(&rc).to_string())
            } else { Value::Str(String::new()) }
        }
        "translate" => {
            if let Value::Str(s) = x {
                let protein = math_runtime::translate(s.as_bytes());
                Value::Str(String::from_utf8_lossy(&protein).to_string())
            } else { Value::Str(String::new()) }
        }
        "factorial" => {
            let n = x.as_i64();
            let mut result: i64 = 1;
            for i in 2..=n { result *= i; }
            Value::I64(result)
        }
        // Statistics (unary on arrays)
        "mean" => {
            if let Value::Array(arr) = x {
                if arr.is_empty() { return Value::F64(0.0); }
                let sum: f64 = arr.iter().map(|v| v.as_f64()).sum();
                Value::F64(sum / arr.len() as f64)
            } else { Value::F64(v) }
        }
        "sum" => {
            if let Value::Array(arr) = x {
                Value::F64(arr.iter().map(|v| v.as_f64()).sum())
            } else { Value::F64(v) }
        }
        "prod" => {
            if let Value::Array(arr) = x {
                Value::F64(arr.iter().map(|v| v.as_f64()).product())
            } else { Value::F64(v) }
        }
        "len" | "size" => {
            if let Value::Array(arr) = x {
                Value::I64(arr.len() as i64)
            } else { Value::I64(1) }
        }
        "sort" => {
            if let Value::Array(arr) = x {
                let mut sorted: Vec<f64> = arr.iter().map(|v| v.as_f64()).collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                Value::Array(sorted.into_iter().map(Value::F64).collect())
            } else { x.clone() }
        }
        "unique" => {
            if let Value::Array(arr) = x {
                let mut seen = Vec::new();
                let mut result = Vec::new();
                for v in arr {
                    let f = v.as_f64();
                    if !seen.iter().any(|&s: &f64| (s - f).abs() < 1e-15) {
                        seen.push(f);
                        result.push(Value::F64(f));
                    }
                }
                Value::Array(result)
            } else { x.clone() }
        }
        "cumsum" => {
            if let Value::Array(arr) = x {
                let mut acc = 0.0;
                let result: Vec<Value> = arr.iter().map(|v| { acc += v.as_f64(); Value::F64(acc) }).collect();
                Value::Array(result)
            } else { x.clone() }
        }
        "diff" => {
            if let Value::Array(arr) = x {
                if arr.len() < 2 { return Value::Array(vec![]); }
                let result: Vec<Value> = arr.windows(2).map(|w| Value::F64(w[1].as_f64() - w[0].as_f64())).collect();
                Value::Array(result)
            } else { Value::Array(vec![]) }
        }
        "std" => {
            if let Value::Array(arr) = x {
                let n = arr.len() as f64;
                if n < 2.0 { return Value::F64(0.0); }
                let mean: f64 = arr.iter().map(|v| v.as_f64()).sum::<f64>() / n;
                let var: f64 = arr.iter().map(|v| { let d = v.as_f64() - mean; d * d }).sum::<f64>() / (n - 1.0);
                Value::F64(var.sqrt())
            } else { Value::F64(0.0) }
        }
        "var" => {
            if let Value::Array(arr) = x {
                let n = arr.len() as f64;
                if n < 2.0 { return Value::F64(0.0); }
                let mean: f64 = arr.iter().map(|v| v.as_f64()).sum::<f64>() / n;
                let var: f64 = arr.iter().map(|v| { let d = v.as_f64() - mean; d * d }).sum::<f64>() / (n - 1.0);
                Value::F64(var)
            } else { Value::F64(0.0) }
        }
        "median" => {
            if let Value::Array(arr) = x {
                if arr.is_empty() { return Value::F64(0.0); }
                let mut sorted: Vec<f64> = arr.iter().map(|v| v.as_f64()).collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n % 2 == 0 { Value::F64((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0) }
                else { Value::F64(sorted[n / 2]) }
            } else { Value::F64(v) }
        }
        // ── ML activation functions (unary on arrays) ────────
        "relu" => {
            if let Some(a) = value_to_vec(x) { vec_to_value(a.iter().map(|&v| v.max(0.0)).collect()) }
            else { Value::F64(v.max(0.0)) }
        }
        "leaky_relu" => {
            if let Some(a) = value_to_vec(x) { vec_to_value(a.iter().map(|&v| if v > 0.0 { v } else { 0.01 * v }).collect()) }
            else { Value::F64(if v > 0.0 { v } else { 0.01 * v }) }
        }
        "sigmoid" => {
            if let Some(a) = value_to_vec(x) { vec_to_value(a.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()) }
            else { Value::F64(1.0 / (1.0 + (-v).exp())) }
        }
        "tanh_act" => {
            if let Some(a) = value_to_vec(x) { vec_to_value(a.iter().map(|&v| v.tanh()).collect()) }
            else { Value::F64(v.tanh()) }
        }
        "gelu" => {
            if let Some(a) = value_to_vec(x) {
                vec_to_value(a.iter().map(|&v| 0.5 * v * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())).collect())
            } else {
                Value::F64(0.5 * v * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh()))
            }
        }
        "swish" => {
            if let Some(a) = value_to_vec(x) { vec_to_value(a.iter().map(|&v| v / (1.0 + (-v).exp())).collect()) }
            else { Value::F64(v / (1.0 + (-v).exp())) }
        }
        "softmax" => {
            if let Some(a) = value_to_vec(x) {
                let max_v = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = a.iter().map(|&v| (v - max_v).exp()).collect();
                let sum: f64 = exps.iter().sum();
                vec_to_value(exps.iter().map(|&e| e / sum).collect())
            } else { Value::F64(1.0) }
        }

        // ── ML loss functions (unary: compute loss from predictions) ──
        "mse_loss" | "mae_loss" => {
            if let Some(a) = value_to_vec(x) {
                let m = a.iter().sum::<f64>() / a.len() as f64;
                if name == "mse_loss" {
                    Value::F64(a.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / a.len() as f64)
                } else {
                    Value::F64(a.iter().map(|&v| (v - m).abs()).sum::<f64>() / a.len() as f64)
                }
            } else { Value::F64(0.0) }
        }

        // ── Image processing (unary) ─────────────────────────
        "grayscale" | "sepia" | "invert_img" | "edge_detect" | "blur"
        | "sharpen" | "emboss" | "threshold" | "histogram_eq" => {
            // Pass through — actual image processing requires pixel buffer
            // which the interpreter represents as Array of Arrays
            x.clone()
        }

        // ── DSP windows ──────────────────────────────────────
        "hann" => {
            let n = x.as_i64() as usize;
            vec_to_value(math_runtime::hann_window(n))
        }
        "hamming" => {
            let n = x.as_i64() as usize;
            vec_to_value(math_runtime::hamming_window(n))
        }
        "blackman" => {
            let n = x.as_i64() as usize;
            vec_to_value(math_runtime::blackman_window(n))
        }

        // ── Finance (unary on price series) ──────────────────
        "max_drawdown" => {
            if let Some(prices) = value_to_vec(x) { Value::F64(math_runtime::max_drawdown(&prices)) }
            else { Value::F64(0.0) }
        }

        // ── Stats distributions ──────────────────────────────
        "normal_cdf" => Value::F64(math_runtime::normal_cdf(v)),
        "normal_pdf" => Value::F64(math_runtime::normal_pdf(v, 0.0, 1.0)),

        // Tag extraction for enum pattern matching
        "enum_tag" => Value::I64(x.as_i64() >> 32),
        _ => Value::F64(0.0),
    }
}

/// Evaluate a binary Custom operation.
fn eval_custom_binary(name: &str, lhs: &Value, rhs: &Value) -> Value {
    use crate::math_runtime;
    let l = lhs.as_f64();
    let r = rhs.as_f64();
    match name {
        // Power
        "pow" => Value::F64(l.powf(r)),
        // Trigonometric
        "atan2" => Value::F64(l.atan2(r)),
        // Clamping
        "clamp" => Value::F64(l.clamp(r, r)), // 3-arg clamp handled separately
        "min" => Value::F64(l.min(r)),
        "max" => Value::F64(l.max(r)),
        // Matrix operations — NxN via math_runtime
        "mat_mul" => {
            if let (Some(a), Some(b)) = (value_to_mat(lhs), value_to_mat(rhs)) {
                mat_to_value(math_runtime::mat_mul(&a, &b))
            } else { Value::F64(l * r) }
        }
        "solve" => {
            if let (Some(a), Some(bv)) = (value_to_mat(lhs), value_to_vec(rhs)) {
                math_runtime::solve(&a, &bv).map(vec_to_value).unwrap_or(Value::F64(0.0))
            } else { Value::F64(0.0) }
        }
        "black_scholes_call" => Value::F64(math_runtime::black_scholes_call(l, r, 1.0, 0.05, 0.2)),
        "black_scholes_put" => Value::F64(math_runtime::black_scholes_put(l, r, 1.0, 0.05, 0.2)),
        "correlation" | "corr" | "pearson" => {
            if let (Some(a), Some(b)) = (value_to_vec(lhs), value_to_vec(rhs)) {
                Value::F64(math_runtime::correlation(&a, &b))
            } else { Value::F64(0.0) }
        }
        "covariance" | "cov" => {
            if let (Some(a), Some(b)) = (value_to_vec(lhs), value_to_vec(rhs)) {
                Value::F64(math_runtime::covariance(&a, &b))
            } else { Value::F64(0.0) }
        }
        "convolve" => {
            if let (Some(a), Some(b)) = (value_to_vec(lhs), value_to_vec(rhs)) {
                vec_to_value(math_runtime::convolve(&a, &b))
            } else { Value::F64(0.0) }
        }
        "smith_waterman" => {
            if let (Value::Str(a), Value::Str(b)) = (lhs, rhs) {
                Value::I64(math_runtime::smith_waterman(a.as_bytes(), b.as_bytes(), 2, -1, -1) as i64)
            } else { Value::I64(0) }
        }
        "needleman_wunsch" => {
            if let (Value::Str(a), Value::Str(b)) = (lhs, rhs) {
                Value::I64(math_runtime::needleman_wunsch(a.as_bytes(), b.as_bytes(), 2, -1, -1) as i64)
            } else { Value::I64(0) }
        }
        // Element-wise operations on arrays
        "elem_mul" => array_binop(lhs, rhs, |a, b| a * b),
        "elem_div" => array_binop(lhs, rhs, |a, b| if b != 0.0 { a / b } else { 0.0 }),
        "elem_pow" => array_binop(lhs, rhs, |a, b| a.powf(b)),
        // Concatenation
        "hconcat" => {
            if let (Value::Array(a), Value::Array(b)) = (lhs, rhs) {
                let mut result = a.clone();
                result.extend(b.iter().cloned());
                Value::Array(result)
            } else { lhs.clone() }
        }
        "vconcat" => {
            // Stack rows vertically (for matrices)
            if let (Value::Array(a), Value::Array(b)) = (lhs, rhs) {
                let mut result = a.clone();
                result.extend(b.iter().cloned());
                Value::Array(result)
            } else { lhs.clone() }
        }
        // Dot product
        "dot" | "dot_product" => {
            if let (Value::Array(a), Value::Array(b)) = (lhs, rhs) {
                let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| x.as_f64() * y.as_f64()).sum();
                Value::F64(sum)
            } else { Value::F64(l * r) }
        }
        // Cross product (3D)
        "cross" | "cross_product" => {
            if let (Value::Array(a), Value::Array(b)) = (lhs, rhs) {
                if a.len() == 3 && b.len() == 3 {
                    let (a0, a1, a2) = (a[0].as_f64(), a[1].as_f64(), a[2].as_f64());
                    let (b0, b1, b2) = (b[0].as_f64(), b[1].as_f64(), b[2].as_f64());
                    Value::Array(vec![
                        Value::F64(a1 * b2 - a2 * b1),
                        Value::F64(a2 * b0 - a0 * b2),
                        Value::F64(a0 * b1 - a1 * b0),
                    ])
                } else { Value::Array(vec![]) }
            } else { Value::F64(0.0) }
        }
        // ── ML binary ops ─────────────────────────────────────
        "cross_entropy" => {
            // Cross-entropy: -sum(target * log(pred))
            if let (Some(pred), Some(target)) = (value_to_vec(lhs), value_to_vec(rhs)) {
                let loss: f64 = pred.iter().zip(&target).map(|(&p, &t)| {
                    -t * (p.max(1e-15)).ln()
                }).sum();
                Value::F64(loss / pred.len() as f64)
            } else { Value::F64(0.0) }
        }
        "mse_loss" => {
            if let (Some(pred), Some(target)) = (value_to_vec(lhs), value_to_vec(rhs)) {
                let loss: f64 = pred.iter().zip(&target).map(|(&p, &t)| (p - t).powi(2)).sum();
                Value::F64(loss / pred.len() as f64)
            } else { Value::F64(0.0) }
        }
        "accuracy" => {
            if let (Some(pred), Some(target)) = (value_to_vec(lhs), value_to_vec(rhs)) {
                let correct = pred.iter().zip(&target).filter(|&(&p, &t)| (p - t).abs() < 0.5).count();
                Value::F64(correct as f64 / pred.len().max(1) as f64)
            } else { Value::F64(0.0) }
        }

        // ── Finance binary ops ───────────────────────────────
        "sharpe" => {
            // sharpe(returns, risk_free_rate)
            if let Some(returns) = value_to_vec(lhs) {
                Value::F64(math_runtime::sharpe_ratio(&returns, r))
            } else { Value::F64(0.0) }
        }
        "sma" => {
            if let Some(data) = value_to_vec(lhs) {
                vec_to_value(math_runtime::sma(&data, r as usize))
            } else { Value::F64(0.0) }
        }
        "ema" => {
            if let Some(data) = value_to_vec(lhs) {
                vec_to_value(math_runtime::ema(&data, r as usize))
            } else { Value::F64(0.0) }
        }
        "npv" => {
            if let Some(cf) = value_to_vec(rhs) {
                Value::F64(math_runtime::npv(l, &cf))
            } else { Value::F64(0.0) }
        }

        // ── Statistics binary ops ────────────────────────────
        "t_test" => {
            if let (Some(a), Some(b)) = (value_to_vec(lhs), value_to_vec(rhs)) {
                let (_t, p) = math_runtime::t_test_two_sample(&a, &b);
                Value::F64(p) // return p-value
            } else { Value::F64(1.0) }
        }
        "linear_regression" | "linreg" => {
            if let (Some(x), Some(y)) = (value_to_vec(lhs), value_to_vec(rhs)) {
                let (b0, b1, r2) = math_runtime::linear_regression(&x, &y);
                Value::Array(vec![Value::F64(b0), Value::F64(b1), Value::F64(r2)])
            } else { Value::F64(0.0) }
        }

        // ── DSP binary ops ───────────────────────────────────
        "fir_filter" => {
            if let (Some(signal), Some(coeffs)) = (value_to_vec(lhs), value_to_vec(rhs)) {
                vec_to_value(math_runtime::fir_filter(&signal, &coeffs))
            } else { Value::F64(0.0) }
        }

        _ => Value::F64(0.0),
    }
}

/// Element-wise binary operation on two arrays.
fn array_binop(lhs: &Value, rhs: &Value, f: impl Fn(f64, f64) -> f64) -> Value {
    match (lhs, rhs) {
        (Value::Array(a), Value::Array(b)) => {
            Value::Array(a.iter().zip(b.iter()).map(|(x, y)| Value::F64(f(x.as_f64(), y.as_f64()))).collect())
        }
        (Value::Array(a), _) => {
            let r = rhs.as_f64();
            Value::Array(a.iter().map(|x| Value::F64(f(x.as_f64(), r))).collect())
        }
        (_, Value::Array(b)) => {
            let l = lhs.as_f64();
            Value::Array(b.iter().map(|y| Value::F64(f(l, y.as_f64()))).collect())
        }
        _ => Value::F64(f(lhs.as_f64(), rhs.as_f64())),
    }
}

#[cfg(test)]
mod checked_tests {
    use super::*;

    #[test]
    fn interpret_checked_runs_valid_graph() {
        let fg = FunctionGraph {
            name: "f".into(),
            nodes: vec![
                GraphNode::Literal { value: LiteralValue::I64(2), data_type: "i64".into() },
                GraphNode::Literal { value: LiteralValue::I64(3), data_type: "i64".into() },
                GraphNode::Apply { op: OpKind::Add, output_type: "i64".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 2,
            return_type: "i64".into(),
        };
        assert_eq!(interpret_checked(&fg, &[], None), Ok(Value::I64(5)));
    }

    #[test]
    fn interpret_checked_rejects_malformed_instead_of_panicking() {
        // A Select missing its else port (2) would panic plain interpret() via
        // input_at_port(..).unwrap(); interpret_checked must return Err.
        let fg = FunctionGraph {
            name: "f".into(),
            nodes: vec![
                GraphNode::Literal { value: LiteralValue::Bool(true), data_type: "bool".into() },
                GraphNode::Literal { value: LiteralValue::I64(1), data_type: "i64".into() },
                GraphNode::Select { output_type: "i64".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 2,
            return_type: "i64".into(),
        };
        assert_eq!(
            interpret_checked(&fg, &[], None),
            Err(GraphError::MissingRequiredInput { node: 2, port: 2 })
        );
    }
}

#[cfg(test)]
mod strict_tests {
    use super::*;

    fn lit(v: i64) -> GraphNode {
        GraphNode::Literal { value: LiteralValue::I64(v), data_type: "i64".into() }
    }
    fn w(from: usize, to: usize, port: i32) -> Wire {
        Wire { from, to, port_in_idx: port, wire_kind: WireKind::Move }
    }
    // x OP y as a single-function graph (both operands i64).
    fn binop(op: OpKind, x: i64, y: i64) -> FunctionGraph {
        FunctionGraph {
            name: "f".into(),
            nodes: vec![lit(x), lit(y), GraphNode::Apply { op, output_type: "i64".into() }],
            wires: vec![w(0, 2, 0), w(1, 2, 1)],
            output_node: 2,
            return_type: "i64".into(),
        }
    }

    #[test]
    fn strict_traps_integer_division_by_zero() {
        let g = binop(OpKind::Div, 1, 0);
        // Strict: surfaces the fault as a typed error.
        assert_eq!(interpret_strict(&g, &[], None), Err(EvalError::DivisionByZero));
    }

    #[test]
    fn lenient_integer_division_by_zero_faults_not_silent() {
        // Lenient must NOT fabricate 0; it faults deterministically (matching
        // the compiled sdiv trap). interpret_strict is the graceful path.
        let g = binop(OpKind::Div, 1, 0);
        let r = std::panic::catch_unwind(|| interpret(&g, &[], None));
        assert!(r.is_err(), "lenient integer div-by-zero must fault, not return a value");
    }

    #[test]
    fn float_division_by_zero_is_ieee_total_not_a_fault() {
        // f(a,b) = a / b with float output → IEEE: x/0.0 = inf, not a fault.
        let g = FunctionGraph {
            name: "f".into(),
            nodes: vec![
                GraphNode::Literal { value: LiteralValue::F64(1.0), data_type: "f64".into() },
                GraphNode::Literal { value: LiteralValue::F64(0.0), data_type: "f64".into() },
                GraphNode::Apply { op: OpKind::Div, output_type: "f64".into() },
            ],
            wires: vec![w(0, 2, 0), w(1, 2, 1)],
            output_node: 2,
            return_type: "f64".into(),
        };
        // Lenient yields +inf (IEEE), no panic, no fabricated 0.
        match interpret(&g, &[], None) {
            Value::F64(v) => assert!(v.is_infinite() && v > 0.0, "expected +inf, got {v}"),
            other => panic!("expected F64(inf), got {other:?}"),
        }
        // Strict does NOT flag it — IEEE float div-by-zero is total, not a fault.
        assert!(matches!(interpret_strict(&g, &[], None), Ok(Value::F64(v)) if v.is_infinite()));
    }

    #[test]
    fn strict_traps_unsupported_op() {
        // Concat on integers is not a valid integer-path op.
        let g = binop(OpKind::Concat, 1, 2);
        match interpret_strict(&g, &[], None) {
            Err(EvalError::UnsupportedOp(_)) => {}
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn strict_traps_unrecoverable_error_propagate() {
        let g = FunctionGraph {
            name: "f".into(),
            nodes: vec![GraphNode::ErrorPropagate {
                error_kind: "abort".into(),
                output_type: "i64".into(),
            }],
            wires: vec![],
            output_node: 0,
            return_type: "i64".into(),
        };
        assert_eq!(
            interpret_strict(&g, &[], None),
            Err(EvalError::Unrecoverable("abort".into()))
        );
    }

    #[test]
    fn strict_passes_clean_graph() {
        let g = binop(OpKind::Add, 2, 3);
        assert_eq!(interpret_strict(&g, &[], None), Ok(Value::I64(5)));
    }

    #[test]
    fn strict_surfaces_structural_fault_before_eval() {
        let mut g = binop(OpKind::Add, 2, 3);
        g.output_node = 9; // out of bounds
        match interpret_strict(&g, &[], None) {
            Err(EvalError::Structural(_)) => {}
            other => panic!("expected Structural error, got {other:?}"),
        }
    }

    #[test]
    fn strict_mode_does_not_leak_into_later_lenient_runs() {
        // A strict run that traps must reset STRICT for the next call. If it
        // leaked (stayed strict), lenient int-div-by-zero would return 0 instead
        // of faulting — so a subsequent lenient run still faulting proves no leak.
        let bad = binop(OpKind::Div, 1, 0);
        let _ = interpret_strict(&bad, &[], None);
        let r = std::panic::catch_unwind(|| interpret(&bad, &[], None));
        assert!(r.is_err(), "lenient run after strict must still fault (no STRICT leak)");
    }
}
