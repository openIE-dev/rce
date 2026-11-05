//! The graph term (spec 6.13): the numeric body of a step as a FunctionGraph.
//!
//! `use std.tensor` makes the library module `tensor` visible. Its calls,
//! with RCE's own lane-wise `+ - * /` between the tensors they take, are
//! one graph term: a `rce_graph::graph::FunctionGraph` with its typing
//! (`rce_graph::term::TermTypes`), named by the content hash of its `.fg`
//! file. A tensor is a nested fixed-length array of `f32` or `f64` (5.1);
//! its shape is the lengths from the outside in.
//!
//! - [`build`]: the term rooted at one `tensor` call, from the types of its
//!   operands; the checker (static types), the reference interpreter
//!   (the operands' values) and the lowering all build it with this one
//!   function, so they agree on the term and on its key.
//! - [`evaluate`]: the reference evaluation, through RCE's CPU backend
//!   (`rce_vm_backend_cpu::interpret_reference`).
//! - [`check_term`]: the core's rule W7 on a term read back from JSON: its
//!   key is the hash of its body, its levels are declared and admitted, and
//!   its shapes type.
use crate::ast::{Arg, Expr, Item, Module};
use crate::interp::V;
use crate::types::T;
use rce_graph::graph::{FunctionGraph, GraphNode, LiteralValue, OpKind, Param, Wire, WireKind};
use rce_graph::term::{Level, NodeLevel, TensorType, TermTypes};
use rce_vm_backend_cpu::{DType, HostTensor, InterpretError, Value as BValue};

/// The library module of the graph term.
pub const MODULE: &str = "std.tensor";

/// The name `std.tensor` is visible under in `m` (4.1), when `m` uses it.
pub fn visible_name(m: &Module) -> Option<String> {
    m.items.iter().find_map(|it| match it {
        Item::Use(p, al, _) if p.join(".") == MODULE => Some(al.clone().unwrap_or_else(|| "tensor".into())),
        _ => None,
    })
}

/// The element type of a tensor: a float type, or an integer type whose
/// arithmetic is checked as RCE's is (6.3 N2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Elem {
    F32,
    F64,
    I8,
    I32,
}

impl Elem {
    pub fn name(self) -> &'static str {
        match self {
            Elem::F32 => "f32",
            Elem::F64 => "f64",
            Elem::I8 => "i8",
            Elem::I32 => "i32",
        }
    }
    fn of(s: &str) -> Option<Elem> {
        match s {
            "f32" => Some(Elem::F32),
            "f64" => Some(Elem::F64),
            "i8" => Some(Elem::I8),
            "i32" => Some(Elem::I32),
            _ => None,
        }
    }
    pub fn is_float(self) -> bool {
        matches!(self, Elem::F32 | Elem::F64)
    }
}

/// The static type of a tensor operand or result: an element type and a
/// shape. An empty shape is a scalar of the element type (the result of
/// `tensor.sum`); an operand has rank 1 or more.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TTy {
    pub elem: Elem,
    pub shape: Vec<u64>,
}

impl TTy {
    /// The tensor type of a checker type: `f32`, `f64`, or nested arrays of
    /// fixed length over one of them.
    pub fn of_t(t: &T) -> Option<TTy> {
        match t {
            T::F32 => Some(TTy { elem: Elem::F32, shape: vec![] }),
            T::F64 => Some(TTy { elem: Elem::F64, shape: vec![] }),
            T::Int(n) => Some(TTy { elem: Elem::of(n)?, shape: vec![] }),
            T::Array(e, Some(n)) => {
                let mut inner = TTy::of_t(e)?;
                inner.shape.insert(0, *n);
                Some(inner)
            }
            _ => None,
        }
    }

    /// The checker type of this tensor type.
    pub fn to_t(&self) -> T {
        let mut t = match self.elem {
            Elem::F32 => T::F32,
            Elem::F64 => T::F64,
            Elem::I8 => T::Int("i8"),
            Elem::I32 => T::Int("i32"),
        };
        for d in self.shape.iter().rev() {
            t = T::Array(Box::new(t), Some(*d));
        }
        t
    }

    /// The tensor type of a value of the reference interpreter, or `None`
    /// when it is not a rectangular nested list of one float type.
    pub fn of_value(v: &V) -> Option<TTy> {
        match v {
            V::F(_) => Some(TTy { elem: Elem::F64, shape: vec![] }),
            V::F32(_) => Some(TTy { elem: Elem::F32, shape: vec![] }),
            V::Int(_, n) if Elem::of(n).is_some_and(|e| !e.is_float()) => Some(TTy { elem: Elem::of(n)?, shape: vec![] }),
            V::List(xs) if !xs.is_empty() => {
                let first = TTy::of_value(&xs[0])?;
                if xs[1..].iter().any(|x| TTy::of_value(x).as_ref() != Some(&first)) {
                    return None;
                }
                let mut t = first;
                t.shape.insert(0, xs.len() as u64);
                Some(t)
            }
            _ => None,
        }
    }

    /// RCE's spelling: `[[f64; 3]; 2]`, or `f64` for a scalar.
    pub fn show(&self) -> String {
        self.to_t().to_string()
    }

    /// The name-free text a graph node carries: `f64[2,3]`.
    pub fn code(&self) -> String {
        let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!("{}[{}]", self.elem.name(), dims.join(","))
    }

    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn spec(&self) -> TensorType {
        TensorType { dtype: self.elem.name().into(), shape: self.shape.clone() }
    }

    /// The tensor type of a node's output type, `f32[2,3]`.
    pub fn of_code(s: &str) -> Option<TTy> {
        let (e, rest) = s.split_once('[')?;
        let dims = rest.strip_suffix(']')?;
        let shape = if dims.is_empty() { vec![] } else { dims.split(',').map(|d| d.parse().ok()).collect::<Option<Vec<u64>>>()? };
        Some(TTy { elem: Elem::of(e)?, shape })
    }

    pub fn of_spec(t: &TensorType) -> Option<TTy> {
        Some(TTy { elem: Elem::of(&t.dtype)?, shape: t.shape.clone() })
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }
}

/// One operation of the `tensor` library: its name in source, its
/// operands, the graph operation it applies, the strongest level a
/// realisation of it can be held to (GT-3), and what it computes.
///
/// `args` names each operand in order: `t` a tensor (an operand of the
/// term), `i` an integer literal (an axis), `f` a float literal (an
/// epsilon), `s` a list of integer literals of at least 1 (a shape), `p` a
/// list of integer literals from 0 (a permutation of axes), `h` a text
/// literal of 64 lowercase hexadecimal digits (a blob's digest, GT-9).
/// Literals are part of
/// the term, so its key covers them.
pub struct Op {
    pub name: &'static str,
    pub args: &'static str,
    pub ir: &'static str,
    pub level: Level,
    pub doc: &'static str,
}

const fn op(name: &'static str, args: &'static str, ir: &'static str, level: Level, doc: &'static str) -> Op {
    Op { name, args, ir, level, doc }
}

/// The operations of `std.tensor` (6.13, table GT-4). The graph operation
/// of each is one of RCE's CPU backend (`rce_vm_backend_cpu::ml`), whose
/// order of operations is fixed; `e` is the element type. E, T and R are
/// `math.exp`, `math.tanh` and `math.erf` of the element as binary64,
/// rounded once to e (6.3 F9), and S and C those of `math.sin` and
/// `math.cos`.
pub const OPS: &[Op] = &[
    op("matmul", "tt", "matmul", Level::Exact, "[[e; k]; m] x [[e; n]; k] -> [[e; n]; m]; each element sums its k products in increasing k from +0.0"),
    op("matvec", "tt", "matmul", Level::Exact, "[[e; k]; m] x [e; k] -> [e; m]; matmul with the vector as one column"),
    op("transpose", "t", "transpose", Level::Exact, "[[e; n]; m] -> [[e; m]; n]"),
    op("permute", "tp", "transpose", Level::Exact, "t, [a1, ...] -> the axes of t in the order a1, ... (a permutation of 0 to rank - 1)"),
    op("reshape", "ts", "reshape", Level::Exact, "t, [d1, ...] -> the same elements in row-major order with the shape [d1, ...]"),
    op("concat", "tti", "tensor_concat", Level::Exact, "a, b, AXIS -> a then b along AXIS; every other length equal"),
    op("add", "tt", "add", Level::Exact, "a + b with broadcasting: lengths aligned from the last, each equal or 1"),
    op("mul", "tt", "mul", Level::Exact, "a * b with broadcasting, as add"),
    op("sum", "t", "reduce_sum", Level::Exact, "every element added in row-major order from +0.0 (0 for integers), to one scalar"),
    op("relu", "t", "relu", Level::Exact, "each element x: x when x > 0, else +0.0"),
    op("exp", "t", "exp", Level::Exact, "each element: E(x)"),
    op("tanh", "t", "tanh", Level::Exact, "each element: T(x)"),
    op("sigmoid", "t", "sigmoid", Level::Exact, "each element: 1 / (1 + E(-x)) for x >= 0, E(x) / (1 + E(x)) for x < 0"),
    op("silu", "t", "silu", Level::Exact, "each element: x / (1 + E(-x)) for x >= 0, (x E(x)) / (1 + E(x)) for x < 0"),
    op("gelu", "t", "gelu_erf", Level::Exact, "each element: (x 0.5)(1 + R(x c)), c = 0.7071067811865476"),
    op("gelu_tanh", "t", "gelu", Level::Exact, "each element: (0.5 x)(1 + T(c1 (x + c2 ((x x) x)))), c1 = 0.7978845608028654, c2 = 0.044715"),
    op("softmax", "ti", "softmax", Level::Exact, "t, AXIS: per slice along AXIS, m its maximum, e_i = E(x_i - m), y_i = e_i / (sum of e_i from +0.0)"),
    op("rmsnorm", "ttf", "rms_norm", Level::Exact, "x, w, EPS: per row of the last axis, (x_i (1 / sqrt(ss / n + EPS))) w_i, ss the sum of x_i x_i from +0.0"),
    op("layernorm", "tttf", "layer_norm", Level::Exact, "x, w, b, EPS: per row of the last axis, ((x_i - mean)(1 / sqrt(var + EPS))) w_i + b_i"),
    op("attention", "ttt", "attention", Level::Exact, "q, k, v: softmax(q k^T s) v per head, s = 1/sqrt(d); [s, d] or heads first [h, s, d]"),
    op("attention_causal", "ttt", "attention", Level::Exact, "attention in which query row i attends keys j <= i + sk - sq"),
    op("embed", "tt", "embedding_lookup", Level::Exact, "table [[e; d]; n], ids of i32 -> the rows the ids name; an id outside 0..n traps"),
    op("matmul_q8_0", "ttt", "matmul_q8_0", Level::Exact, "x [[f32; k]; m], q [[i8; k]; n], d [[f32; k/32]; n] -> [[f32; n]; m]: Q8_0 blocks of 32"),
    op("matmul_q4_0", "ttt", "matmul_q4_0", Level::Exact, "as matmul_q8_0, q in -8..7, each block summed in the order 0, 16, 1, 17, ..."),
    op("matmul_q4_k", "tttt", "matmul_q4_k", Level::Exact, "x [[f32; k]; m], q [[i8; k]; n] in 0..15, s and o [[f32; k/32]; n] -> [[f32; n]; m]: each value s q - o; per pair of sub-blocks of 32, y + (((s1 u1 - o1 a1) + s2 u2) - o2 a2), u the sum of q x and a of x; k a multiple of 256"),
    op("matmul_q5_k", "tttt", "matmul_q5_k", Level::Exact, "as matmul_q4_k, q in 0..31"),
    op("matmul_q6_k", "ttt", "matmul_q6_k", Level::Exact, "x [[f32; k]; m], q [[i8; k]; n] in -32..31, s [[f32; k/16]; n] -> [[f32; n]; m]: each value s q; per half of 128 of each super-block of 256, for l in 0..31, positions l, l+32, l+64, l+96: y + (s q) x"),
    op("rope", "ttt", "rope", Level::Exact, "x [s, ..., d], pos [i32; s], freqs [e; d/2]: per index r of the first axis and pair i, a = (pos_r as e) freqs_i, c = C(a), n = S(a); (x_2i, x_2i+1) -> (x_2i c - x_2i+1 n, x_2i n + x_2i+1 c)"),
    op("blob_f32", "hs", "blob", Level::Exact, "DIGEST, [d1, ...] -> the f32 tensor of that shape the store holds as the blob DIGEST (GT-9)"),
    op("blob_f64", "hs", "blob", Level::Exact, "DIGEST, [d1, ...] -> the f64 tensor of that shape the store holds as the blob DIGEST"),
    op("blob_i8", "hs", "blob", Level::Exact, "DIGEST, [d1, ...] -> the i8 tensor of that shape the store holds as the blob DIGEST"),
    op("blob_i32", "hs", "blob", Level::Exact, "DIGEST, [d1, ...] -> the i32 tensor of that shape the store holds as the blob DIGEST"),
    op("round_bf16", "t", "round_bf16", Level::Lossy, "each element rounded once to the nearest bfloat16 value, ties to even; it lowers precision"),
    op("round_q8_0", "t", "round_q8_0", Level::Lossy, "each block of 32 along the last axis replaced by the values its Q8_0 quantization represents"),
    op("round_q4_0", "t", "round_q4_0", Level::Lossy, "each block of 32 along the last axis replaced by the values its Q4_0 quantization represents"),
    op("round_q4_k", "t", "round_q4_k", Level::Lossy, "each super-block of 256 along the last axis replaced by the values its Q4_K quantization (RCE's reference quantizer) represents"),
    op("round_q5_k", "t", "round_q5_k", Level::Lossy, "as round_q4_k, for Q5_K"),
    op("round_q6_k", "t", "round_q6_k", Level::Lossy, "as round_q4_k, for Q6_K"),
];

/// The operation of `std.tensor` named `name` in source.
pub fn op_named(name: &str) -> Option<&'static Op> {
    OPS.iter().find(|o| o.name == name)
}

/// The strongest level of a graph operation (W7): lossy for the operations
/// that lower precision, exact for every other.
pub fn ir_level(ir: &str) -> Level {
    OPS.iter().find(|o| o.ir == ir).map(|o| o.level).unwrap_or(Level::Exact)
}

/// The two declarations of a level (GT-3): `tensor.within(N, t)` and
/// `tensor.lossy(t)` state the level of every operation inside `t`.
pub const DECLS: &[&str] = &["within", "lossy"];

/// A finding of the graph term's rules (W-GRAPH-1 to W-GRAPH-3).
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub line: usize,
    pub rule: &'static str,
    pub msg: String,
}

impl Finding {
    fn new(line: usize, rule: &'static str, msg: impl Into<String>) -> Finding {
        Finding { line, rule, msg: msg.into() }
    }
    /// The report line of `rce check`: `line N: RULE message`.
    pub fn text(&self) -> String {
        format!("line {}: {} {}", self.line, self.rule, self.msg)
    }
}

/// Why a term was not built: a finding of its rules, or what the operand
/// callback stopped with (the reference interpreter's trap).
#[derive(Debug)]
pub enum BuildErr<E> {
    Finding(Finding),
    Leaf(E),
}

impl<E> From<Finding> for BuildErr<E> {
    fn from(f: Finding) -> Self {
        BuildErr::Finding(f)
    }
}

/// A graph term: its graph, its typing, its key and what it costs.
#[derive(Clone, Debug)]
pub struct Term {
    pub graph: FunctionGraph,
    pub types: TermTypes,
    /// the content hash of its `.fg` file, 64 lowercase hexadecimal digits
    pub key: String,
    pub result: TTy,
    /// the scalar operations it performs (6.13, GT-7 charges)
    pub work: u64,
    /// the line of the `tensor` call it is rooted at, and that call's operation
    pub line: usize,
    pub op: String,
}

/// The `tensor` operation `e` calls, when `e`'s callee is `NAME.op` with
/// `NAME` the visible name of `std.tensor`; the caller checks that `NAME`
/// is not a local.
pub fn call_of<'e>(e: &'e Expr, name: &str) -> Option<(&'e str, &'e [Arg], usize)> {
    if let Expr::Call(f, args, l, _) = e
        && let Expr::Member(o, op, _) = &**f
        && let Expr::Name(c, _) = &**o
        && c == name
    {
        return Some((op.as_str(), args.as_slice(), *l));
    }
    None
}

/// The operands of the term rooted at the `tensor` call `op(args)` that
/// are not part of it, in the order [`build`] meets them and the reference
/// interpreter evaluates them (GT-5). A literal argument (an axis, an
/// epsilon, a shape, the bound of `within`) is part of the term, not an
/// operand.
pub fn operands<'e>(op: &str, args: &'e [Arg], name: &str) -> Vec<&'e Expr> {
    fn call<'e>(op: &str, args: &'e [Arg], name: &str, out: &mut Vec<&'e Expr>) {
        let kinds = match op {
            "within" => "it",
            "lossy" => "t",
            _ => op_named(op).map(|o| o.args).unwrap_or(""),
        };
        for (i, a) in args.iter().enumerate() {
            if kinds.as_bytes().get(i).copied().unwrap_or(b't') == b't' {
                expr(&a.e, name, out);
            }
        }
    }
    fn expr<'e>(e: &'e Expr, name: &str, out: &mut Vec<&'e Expr>) {
        if let Some((op, args, _)) = call_of(e, name) {
            return call(op, args, name, out);
        }
        if let Expr::Bin(op, a, b, _) = e
            && lane_op(op).is_some()
        {
            expr(a, name, out);
            expr(b, name, out);
            return;
        }
        out.push(e);
    }
    let mut out = vec![];
    call(op, args, name, &mut out);
    out
}

struct B<'a, E> {
    name: &'a str,
    /// the operation whose operands are being built, for findings
    cur: String,
    nodes: Vec<GraphNode>,
    wires: Vec<Wire>,
    levels: Vec<NodeLevel>,
    params: Vec<TensorType>,
    level: Level,
    work: u64,
    leaf: &'a mut dyn FnMut(&Expr) -> Result<TTy, E>,
}

impl<E> B<'_, E> {
    fn push(&mut self, n: GraphNode) -> usize {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    fn lit(&mut self, s: String) -> usize {
        self.push(GraphNode::Literal { value: LiteralValue::String(s), data_type: "string".into() })
    }

    fn apply(&mut self, op: OpKind, ins: &[usize], out: &TTy, work: u64) -> usize {
        let i = self.push(GraphNode::Apply { op, output_type: out.code() });
        for (p, from) in ins.iter().enumerate() {
            self.wires.push(Wire { from: *from, to: i, port_in_idx: p as i32, wire_kind: WireKind::Move });
        }
        self.levels.push(NodeLevel { node: i as u32, level: self.level });
        self.work = self.work.saturating_add(work);
        i
    }

    fn lit_i32(&mut self, v: i32) -> usize {
        self.push(GraphNode::Literal { value: LiteralValue::I32(v), data_type: "i32".into() })
    }

    /// A float literal of the term, rounded to the element type.
    fn lit_float(&mut self, e: Elem, v: f64) -> usize {
        let value = if e == Elem::F32 { LiteralValue::F32(v as f32) } else { LiteralValue::F64(v) };
        self.push(GraphNode::Literal { value, data_type: e.name().into() })
    }

    fn custom(&mut self, name: &str, ins: &[usize], out: &TTy, work: u64) -> usize {
        self.apply(OpKind::Custom(name.into()), ins, out, work)
    }

    /// An operand that is not part of the term: a parameter, in the order
    /// the operands are met (left to right, depth first).
    fn param(&mut self, e: &Expr) -> Result<(usize, TTy), BuildErr<E>> {
        let t = (self.leaf)(e).map_err(BuildErr::Leaf)?;
        if t.rank() == 0 || t.shape.contains(&0) {
            return Err(Finding::new(
                e.line(),
                "W-GRAPH-2",
                format!("an operand of `{}` is {}; a tensor is an array of f32, f64, i8 or i32 with every length at least 1", self.cur, t.show()),
            )
            .into());
        }
        let i = self.params.len();
        let n = self.push(GraphNode::Param(Param { name: format!("p{i}"), data_type: t.code() }));
        self.params.push(t.spec());
        Ok((n, t))
    }

    fn expr(&mut self, e: &Expr) -> Result<(usize, TTy), BuildErr<E>> {
        if let Some((op, args, line)) = call_of(e, self.name) {
            return self.op(op, args, line);
        }
        if let Expr::Bin(op, a, b, line) = e
            && let Some(k) = lane_op(op)
        {
            let outer = std::mem::replace(&mut self.cur, format!("lane-wise {op}"));
            let (x, tx) = self.expr(a)?;
            self.cur = format!("lane-wise {op}");
            let (y, ty) = self.expr(b)?;
            self.cur = outer;
            same_elem(*line, op, &tx, &ty)?;
            if tx.shape != ty.shape {
                return Err(Finding::new(*line, "W-GRAPH-1", format!("lane-wise `{op}` needs two tensors of one shape, found {} and {}", tx.show(), ty.show())).into());
            }
            let w = tx.numel();
            return Ok((self.apply(k, &[x, y], &tx, w), tx));
        }
        self.param(e)
    }

    fn op(&mut self, op: &str, args: &[Arg], line: usize) -> Result<(usize, TTy), BuildErr<E>> {
        let q = format!("{}.{op}", self.name);
        let kinds = match op {
            "within" => "it",
            "lossy" => "t",
            _ => match op_named(op) {
                Some(o) => o.args,
                None => {
                    let names: Vec<&str> = OPS.iter().map(|o| o.name).chain(DECLS.iter().copied()).collect();
                    return Err(Finding::new(line, "W-GRAPH-2", format!("`{q}` is not an operation of std.tensor ({})", names.join(", "))).into());
                }
            },
        };
        let arity = kinds.len();
        if args.len() != arity {
            return Err(Finding::new(line, "W-GRAPH-2", format!("`{q}` takes {arity} operand{}, found {}", if arity == 1 { "" } else { "s" }, args.len())).into());
        }
        if let Some(a) = args.iter().find(|a| a.label.is_some()) {
            return Err(Finding::new(line, "W-GRAPH-2", format!("`{q}` takes its operands by position; remove the label `{}:`", a.label.clone().unwrap_or_default())).into());
        }
        match op {
            "within" => {
                let ulps = match &args[0].e {
                    Expr::Int(n, _) if (1..=i32::MAX as i128).contains(n) => *n as u32,
                    _ => {
                        return Err(Finding::new(line, "W-GRAPH-2", format!("`{q}(N, t)`: N, the bound in units in the last place, is an integer literal from 1 to {}", i32::MAX)).into());
                    }
                };
                return self.declared(Level::Within { ulps }, &args[1].e);
            }
            "lossy" => return self.declared(Level::Lossy, &args[0].e),
            _ => {}
        }
        let floor = op_named(op).map(|o| o.level).unwrap_or(Level::Exact);
        if !floor.at_least(self.level) {
            let how = match floor {
                Level::Lossy => format!("`{}.lossy(...)`", self.name),
                Level::Within { ulps } => format!("`{}.within({ulps}, ...)` or a larger bound", self.name),
                Level::Exact => String::new(),
            };
            return Err(Finding::new(
                line,
                "W-GRAPH-3",
                format!("`{q}` has level {}, weaker than the level {} in force here; declare it with {how} around the operation", floor.text(), self.level.text()),
            )
            .into());
        }
        // the operands, then the literals, in order
        let mut ins: Vec<(usize, TTy)> = vec![];
        let mut ints: Vec<i64> = vec![];
        let mut floats: Vec<f64> = vec![];
        let mut shape: Vec<u64> = vec![];
        let mut axes: Vec<usize> = vec![];
        let mut digest = String::new();
        for (i, (a, k)) in args.iter().zip(kinds.bytes()).enumerate() {
            let pos = ["first", "second", "third", "fourth"].get(i).copied().unwrap_or("last");
            match k {
                b't' => {
                    self.cur = q.clone();
                    ins.push(self.expr(&a.e)?);
                }
                b'i' => match &a.e {
                    Expr::Int(n, _) if (0..=i32::MAX as i128).contains(n) => ints.push(*n as i64),
                    _ => return Err(Finding::new(line, "W-GRAPH-2", format!("the {pos} operand of `{q}`, an axis, is an integer literal from 0")).into()),
                },
                b'f' => match &a.e {
                    Expr::Float(x, _) if x.is_finite() && *x >= 0.0 => floats.push(*x),
                    Expr::Int(n, _) if *n >= 0 && *n < 1 << 53 => floats.push(*n as f64),
                    _ => return Err(Finding::new(line, "W-GRAPH-2", format!("the {pos} operand of `{q}`, an epsilon, is a float literal of at least 0")).into()),
                },
                b'h' => match &a.e {
                    Expr::Str(d, _) if rce_fgm::parse_hex(d).is_some() => digest = d.clone(),
                    _ => return Err(Finding::new(line, "W-GRAPH-2", format!("the {pos} operand of `{q}`, a digest, is a text literal of 64 lowercase hexadecimal digits")).into()),
                },
                b'p' => match &a.e {
                    Expr::List(xs, _) if !xs.is_empty() && xs.iter().all(|x| matches!(x, Expr::Int(n, _) if (0..64).contains(n))) => {
                        axes = xs.iter().map(|x| if let Expr::Int(n, _) = x { *n as usize } else { 0 }).collect();
                    }
                    _ => return Err(Finding::new(line, "W-GRAPH-2", format!("the {pos} operand of `{q}`, a permutation of axes, is a list of integer literals from 0")).into()),
                },
                _ => match &a.e {
                    Expr::List(xs, _) if !xs.is_empty() && xs.iter().all(|x| matches!(x, Expr::Int(n, _) if (1..=i32::MAX as i128).contains(n))) => {
                        shape = xs.iter().map(|x| if let Expr::Int(n, _) = x { *n as u64 } else { 0 }).collect();
                    }
                    _ => return Err(Finding::new(line, "W-GRAPH-2", format!("the {pos} operand of `{q}`, a shape, is a list of integer literals of at least 1")).into()),
                },
            }
        }
        if let Some(elem) = op.strip_prefix("blob_").and_then(Elem::of) {
            let out = TTy { elem, shape: shape.clone() };
            let d = self.lit(digest);
            let w = out.numel();
            return Ok((self.custom("blob", &[d], &out, w), out));
        }
        let f1 = |m: String| -> BuildErr<E> { Finding::new(line, "W-GRAPH-1", m).into() };
        let f2 = |m: String| -> BuildErr<E> { Finding::new(line, "W-GRAPH-2", m).into() };
        // every tensor operand of these operations has one element type
        if !matches!(op, "embed" | "rope" | "matmul_q8_0" | "matmul_q4_0" | "matmul_q4_k" | "matmul_q5_k" | "matmul_q6_k") {
            for t in ins.iter().skip(1) {
                same_elem(line, &q, &ins[0].1, &t.1)?;
            }
        }
        let float_ops = !matches!(op, "transpose" | "permute" | "reshape" | "concat" | "add" | "mul" | "sum" | "embed" | "rope" | "matmul_q8_0" | "matmul_q4_0" | "matmul_q4_k" | "matmul_q5_k" | "matmul_q6_k");
        if float_ops && let Some((_, t)) = ins.iter().find(|(_, t)| !t.elem.is_float()) {
            return Err(f2(format!("`{q}` takes tensors of f32 or f64, found {}", t.show())));
        }
        let (x, tx) = ins[0].clone();
        let rank = |t: &TTy, r: usize, what: &str| -> Result<(), BuildErr<E>> {
            if t.rank() == r { Ok(()) } else { Err(Finding::new(line, "W-GRAPH-1", format!("`{q}` takes {what}, found {}", t.show())).into()) }
        };
        let numel = tx.numel();
        let r = match op {
            "matmul" => {
                let (y, ty) = ins[1].clone();
                rank(&tx, 2, "two matrices")?;
                rank(&ty, 2, "two matrices")?;
                let (m, k, k2, n) = (tx.shape[0], tx.shape[1], ty.shape[0], ty.shape[1]);
                if k != k2 {
                    return Err(f1(format!("`{q}`: the first operand {} has {k} columns and the second {} has {k2} rows; they must be equal", tx.show(), ty.show())));
                }
                let out = TTy { elem: tx.elem, shape: vec![m, n] };
                (self.custom("matmul", &[x, y], &out, 2 * m * n * k), out)
            }
            "matvec" => {
                let (y, ty) = ins[1].clone();
                rank(&tx, 2, "a matrix and a vector")?;
                rank(&ty, 1, "a matrix and a vector")?;
                let (m, k, k2) = (tx.shape[0], tx.shape[1], ty.shape[0]);
                if k != k2 {
                    return Err(f1(format!("`{q}`: the matrix {} has {k} columns and the vector {} has {k2} elements; they must be equal", tx.show(), ty.show())));
                }
                let col = TTy { elem: tx.elem, shape: vec![k, 1] };
                let s1 = self.lit(format!("[{k},1]"));
                let c = self.custom("reshape", &[y, s1], &col, 0);
                let prod = TTy { elem: tx.elem, shape: vec![m, 1] };
                let mm = self.custom("matmul", &[x, c], &prod, 2 * m * k);
                let out = TTy { elem: tx.elem, shape: vec![m] };
                let s2 = self.lit(format!("[{m}]"));
                (self.custom("reshape", &[mm, s2], &out, 0), out)
            }
            "transpose" => {
                rank(&tx, 2, "a matrix")?;
                let out = TTy { elem: tx.elem, shape: vec![tx.shape[1], tx.shape[0]] };
                let axes = self.lit("[1,0]".into());
                (self.custom("transpose", &[x, axes], &out, numel), out)
            }
            "permute" => {
                let mut seen = axes.clone();
                seen.sort();
                if seen != (0..tx.rank()).collect::<Vec<_>>() {
                    return Err(f1(format!("`{q}`: {axes:?} is not a permutation of the axes of {}, 0 to {}", tx.show(), tx.rank() - 1)));
                }
                let out = TTy { elem: tx.elem, shape: axes.iter().map(|a| tx.shape[*a]).collect() };
                let ax: Vec<String> = axes.iter().map(|a| a.to_string()).collect();
                let l = self.lit(format!("[{}]", ax.join(",")));
                (self.custom("transpose", &[x, l], &out, numel), out)
            }
            "reshape" => {
                let out = TTy { elem: tx.elem, shape: shape.clone() };
                if out.numel() != numel {
                    return Err(f1(format!("`{q}`: {} has {numel} elements and the shape {:?} holds {}", tx.show(), shape, out.numel())));
                }
                let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                let s = self.lit(format!("[{}]", dims.join(",")));
                (self.custom("reshape", &[x, s], &out, 0), out)
            }
            "concat" => {
                let (y, ty) = ins[1].clone();
                let a = ints[0] as usize;
                let same = tx.rank() == ty.rank() && (0..tx.rank()).all(|i| i == a || tx.shape[i] == ty.shape[i]);
                if a >= tx.rank() || !same {
                    return Err(f1(format!("`{q}`: {} and {} along axis {a}: the ranks and every other length must be equal", tx.show(), ty.show())));
                }
                let mut out = tx.clone();
                out.shape[a] += ty.shape[a];
                let ax = self.lit_i32(a as i32);
                let w = out.numel();
                (self.custom("tensor_concat", &[x, y, ax], &out, w), out)
            }
            "add" | "mul" => {
                let (y, ty) = ins[1].clone();
                let sh = broadcast(&tx.shape, &ty.shape).ok_or_else(|| f1(format!("`{q}`: {} and {} do not broadcast; aligned from the last, each pair of lengths must be equal or one of them 1", tx.show(), ty.show())))?;
                let out = TTy { elem: tx.elem, shape: sh };
                let k = if op == "add" { OpKind::Add } else { OpKind::Mul };
                let w = out.numel();
                (self.apply(k, &[x, y], &out, w), out)
            }
            "sum" => {
                let axes: Vec<String> = (0..tx.rank()).map(|i| i.to_string()).collect();
                let a = self.lit(format!("[{}]", axes.join(",")));
                let keep = self.push(GraphNode::Literal { value: LiteralValue::Bool(false), data_type: "bool".into() });
                let out = TTy { elem: tx.elem, shape: vec![] };
                (self.custom("reduce_sum", &[x, a, keep], &out, numel), out)
            }
            "relu" | "round_bf16" | "exp" | "tanh" | "sigmoid" | "silu" | "gelu" | "gelu_tanh" => {
                let ir = op_named(op).map(|o| o.ir).unwrap_or(op);
                (self.custom(ir, &[x], &tx, numel), tx.clone())
            }
            "round_q4_k" | "round_q5_k" | "round_q6_k" => {
                if tx.elem != Elem::F32 || tx.shape.last().is_none_or(|l| l % 256 != 0) {
                    return Err(f1(format!("`{q}` takes a tensor of f32 whose last length is a multiple of 256, found {}", tx.show())));
                }
                (self.custom(op, &[x], &tx, numel), tx.clone())
            }
            "rope" => {
                let ((pn, tp), (fnn, tf)) = (ins[1].clone(), ins[2].clone());
                if !tx.elem.is_float() || tp.elem != Elem::I32 || tf.elem != tx.elem {
                    return Err(f2(format!("`{q}` takes x of f32 or f64, pos of i32 and freqs of x's type, found {}, {} and {}", tx.show(), tp.show(), tf.show())));
                }
                let d = tx.shape.last().copied().unwrap_or(0);
                if tx.rank() < 2 || d % 2 != 0 || tp.shape != [tx.shape[0]] || tf.shape != [d / 2] {
                    return Err(f1(format!("`{q}`: x {}, pos {}, freqs {}: x [s, ..., d] with d even, pos [i32; s] and freqs [e; d/2]", tx.show(), tp.show(), tf.show())));
                }
                (self.custom("rope", &[x, pn, fnn], &tx, 6 * numel), tx.clone())
            }
            "matmul_q4_k" | "matmul_q5_k" | "matmul_q6_k" => {
                let sub = if op == "matmul_q6_k" { 16 } else { 32 };
                let (qn, tq) = ins[1].clone();
                let rest: Vec<(usize, TTy)> = ins[2..].to_vec();
                if tx.elem != Elem::F32 || tq.elem != Elem::I8 || rest.iter().any(|(_, t)| t.elem != Elem::F32) {
                    return Err(f2(format!("`{q}` takes x of f32, q of i8 and scales of f32, found {}, {} and {}", tx.show(), tq.show(), rest[0].1.show())));
                }
                let ok = tx.rank() == 2 && tq.rank() == 2 && tx.shape[1] == tq.shape[1] && tx.shape[1] % 256 == 0 && rest.iter().all(|(_, t)| t.shape == [tq.shape[0], tq.shape[1] / sub]);
                if !ok {
                    return Err(f1(format!("`{q}`: x {}, q {}, s {}: x [[f32; k]; m], q [[i8; k]; n] and each scale [[f32; k/{sub}]; n], k a multiple of 256", tx.show(), tq.show(), rest[0].1.show())));
                }
                let (m, k, n) = (tx.shape[0], tx.shape[1], tq.shape[0]);
                let out = TTy { elem: Elem::F32, shape: vec![m, n] };
                let mut wires = vec![x, qn];
                wires.extend(rest.iter().map(|(w, _)| *w));
                (self.custom(op, &wires, &out, 2 * m * n * k + 4 * m * n * (k / sub)), out)
            }
            "round_q8_0" | "round_q4_0" => {
                if tx.elem != Elem::F32 || tx.shape.last().is_none_or(|l| l % 32 != 0) {
                    return Err(f1(format!("`{q}` takes a tensor of f32 whose last length is a multiple of 32, found {}", tx.show())));
                }
                (self.custom(op, &[x], &tx, numel), tx.clone())
            }
            "softmax" => {
                let a = ints[0] as usize;
                if a >= tx.rank() {
                    return Err(f1(format!("`{q}`: axis {a} of {}, which has rank {}", tx.show(), tx.rank())));
                }
                let ax = self.lit_i32(a as i32);
                (self.custom("softmax", &[x, ax], &tx, 3 * numel), tx.clone())
            }
            "rmsnorm" | "layernorm" => {
                let d = *tx.shape.last().unwrap_or(&1);
                let row = TTy { elem: tx.elem, shape: vec![d] };
                for (_, p) in &ins[1..] {
                    if p.shape != row.shape {
                        return Err(f1(format!("`{q}`: the parameters of {} are of the shape {}, found {}", tx.show(), row.show(), p.show())));
                    }
                }
                let groups = numel / d;
                if groups > i32::MAX as u64 {
                    return Err(f1(format!("`{q}`: {} has more than {} rows", tx.show(), i32::MAX)));
                }
                let mut wires: Vec<usize> = ins.iter().map(|(n, _)| *n).collect();
                wires.push(self.lit_i32(groups as i32));
                wires.push(self.lit_float(tx.elem, floats[0]));
                let ir = if op == "rmsnorm" { "rms_norm" } else { "layer_norm" };
                (self.custom(ir, &wires, &tx, 4 * numel), tx.clone())
            }
            "attention" | "attention_causal" => {
                let ((k, tk), (v, tv)) = (ins[1].clone(), ins[2].clone());
                let r = tx.rank();
                let ok = (r == 2 || r == 3) && tk.rank() == r && tv.rank() == r;
                if !ok {
                    return Err(f1(format!("`{q}` takes q, k and v of rank 2 ([s, d]) or 3 ([h, s, d]), found {}, {}, {}", tx.show(), tk.show(), tv.show())));
                }
                let (hq, hk, hv) = if r == 3 { (tx.shape[0], tk.shape[0], tv.shape[0]) } else { (1, 1, 1) };
                let (sq, d, sk, kd, vs, dv) = (tx.shape[r - 2], tx.shape[r - 1], tk.shape[r - 2], tk.shape[r - 1], tv.shape[r - 2], tv.shape[r - 1]);
                if kd != d || vs != sk || hk != hv || hq % hk != 0 {
                    return Err(f1(format!("`{q}`: q {}, k {}, v {}: k has q's last length, v as many rows as k, and the heads of q are a multiple of those of k and v", tx.show(), tk.show(), tv.show())));
                }
                let causal = op == "attention_causal";
                if causal && sq > sk {
                    return Err(f1(format!("`{q}`: q has {sq} rows and k {sk}; a causal attention needs at most as many queries as keys")));
                }
                let scale = self.lit_float(tx.elem, 1.0 / (d as f64).sqrt());
                let win = self.lit_i32(0);
                let mut wires = vec![x, k, v, scale, win];
                if causal {
                    wires.push(self.push(GraphNode::Literal { value: LiteralValue::Bool(true), data_type: "bool".into() }));
                }
                let mut out = tx.clone();
                out.shape[r - 1] = dv;
                let w = 2 * hq * sq * sk * (d + dv) + 3 * hq * sq * sk;
                (self.custom("attention", &wires, &out, w), out)
            }
            "embed" => {
                let (ids, ti) = ins[1].clone();
                if !tx.elem.is_float() || ti.elem != Elem::I32 {
                    return Err(f2(format!("`{q}` takes a table of f32 or f64 and ids of i32, found {} and {}", tx.show(), ti.show())));
                }
                rank(&tx, 2, "a table [[e; d]; n]")?;
                let mut out = TTy { elem: tx.elem, shape: ti.shape.clone() };
                out.shape.push(tx.shape[1]);
                let w = out.numel();
                (self.custom("embedding_lookup", &[x, ids], &out, w), out)
            }
            "matmul_q8_0" | "matmul_q4_0" => {
                let ((qn, tq), (dn, td)) = (ins[1].clone(), ins[2].clone());
                if tx.elem != Elem::F32 || tq.elem != Elem::I8 || td.elem != Elem::F32 {
                    return Err(f2(format!("`{q}` takes x of f32, q of i8 and d of f32, found {}, {} and {}", tx.show(), tq.show(), td.show())));
                }
                let ok = tx.rank() == 2 && tq.rank() == 2 && td.rank() == 2 && tx.shape[1] == tq.shape[1] && tx.shape[1] % 32 == 0 && td.shape == [tq.shape[0], tq.shape[1] / 32];
                if !ok {
                    return Err(f1(format!("`{q}`: x {}, q {}, d {}: x [[f32; k]; m], q [[i8; k]; n] and d [[f32; k/32]; n], k a multiple of 32", tx.show(), tq.show(), td.show())));
                }
                let (m, k, n) = (tx.shape[0], tx.shape[1], tq.shape[0]);
                let out = TTy { elem: Elem::F32, shape: vec![m, n] };
                (self.custom(op, &[x, qn, dn], &out, 2 * m * n * k), out)
            }
            _ => unreachable!("every name of OPS is matched"),
        };
        Ok(r)
    }

    fn declared(&mut self, level: Level, e: &Expr) -> Result<(usize, TTy), BuildErr<E>> {
        let outer = std::mem::replace(&mut self.level, level);
        let r = self.expr(e);
        self.level = outer;
        r
    }
}

fn lane_op(op: &str) -> Option<OpKind> {
    Some(match op {
        "+" => OpKind::Add,
        "-" => OpKind::Sub,
        "*" => OpKind::Mul,
        "/" => OpKind::Div,
        _ => return None,
    })
}

/// The shape two shapes broadcast to: aligned from the last length, each
/// pair equal or one of them 1.
fn broadcast(a: &[u64], b: &[u64]) -> Option<Vec<u64>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let x = if i < n - a.len() { 1 } else { a[i - (n - a.len())] };
        let y = if i < n - b.len() { 1 } else { b[i - (n - b.len())] };
        out[i] = if x == y || y == 1 { x } else if x == 1 { y } else { return None };
    }
    Some(out)
}

fn same_elem(line: usize, what: &str, a: &TTy, b: &TTy) -> Result<(), Finding> {
    if a.elem == b.elem {
        Ok(())
    } else {
        Err(Finding::new(line, "W-GRAPH-2", format!("the operands of `{what}` have two element types, {} and {}; convert one with `as`", a.elem.name(), b.elem.name())))
    }
}

/// Build the graph term rooted at the `tensor` call `op(args)` on `line`,
/// `name` being the visible name of `std.tensor`. `leaf` gives the type of
/// each operand that is not part of the term, in the order the term meets
/// them (left to right, depth first), which is the order they are
/// evaluated in (GT-5).
pub fn build<E>(op: &str, args: &[Arg], line: usize, name: &str, leaf: &mut dyn FnMut(&Expr) -> Result<TTy, E>) -> Result<Term, BuildErr<E>> {
    let mut b = B { name, cur: format!("{name}.{op}"), nodes: vec![], wires: vec![], levels: vec![], params: vec![], level: Level::Exact, work: 0, leaf };
    let (out, result) = b.op(op, args, line)?;
    let graph = FunctionGraph { name: String::new(), nodes: b.nodes, wires: b.wires, output_node: out, return_type: result.code() };
    let types = TermTypes { params: b.params, result: result.spec(), levels: b.levels };
    let key = key_of(&graph, &types).map_err(|e| Finding::new(line, "W-GRAPH-2", format!("internal: the term has no .fg encoding: {e}")))?;
    Ok(Term { graph, types, key, result, work: b.work, line, op: op.into() })
}

/// The key of a graph term: the content hash of its `.fg` file (GT-2).
pub fn key_of(graph: &FunctionGraph, types: &TermTypes) -> Result<String, String> {
    rce_format::graph_section::term_content_hash(graph, types).map(|h| rce_format::hash::hex_encode(&h)).map_err(|e| e.to_string())
}

/// The keys of the graph terms of `m`, in ascending order: the terms the
/// checker builds on `m` linked with the library (4.1), as `rce check` does.
pub fn keys_of(m: &Module) -> Vec<String> {
    if visible_name(m).is_none() {
        return vec![];
    }
    let lk = crate::link::link(m, &[m]);
    let mut c = crate::check::Checker::new();
    c.imported_fns = lk.imported_fns.clone();
    c.imported_types = lk.imported_types.clone();
    c.check_module(&lk.module, &Default::default());
    let mut ks: Vec<String> = c.graph_terms.into_values().map(|t| t.key).collect();
    ks.sort();
    ks.dedup();
    ks
}

/// The trap text of a graph term whose evaluation leaves the finite floats
/// (6.3 N3), as every float operation of RCE.
pub const TRAP_NON_FINITE: &str = "non-finite float";

/// Whether an error of [`evaluate`] is a trap of the program (a value of
/// 6.3: `non-finite float`, an integer overflow or division by zero, an
/// index out of range) rather than a failure of the evaluation itself,
/// whose text starts `graph term:`.
pub fn is_trap(m: &str) -> bool {
    !m.starts_with("graph term:")
}

/// The reference evaluation of `term` on the values of its operands
/// (GT-6): through `rce_vm_backend_cpu::interpret_reference`, in its fixed
/// order. Err is the trap text.
pub fn evaluate(term: &Term, vals: &[V]) -> Result<V, String> {
    evaluate_graph(&term.graph, &term.types, vals)
}

/// [`evaluate`] of a graph and its typing.
pub fn evaluate_graph(graph: &FunctionGraph, types: &TermTypes, vals: &[V]) -> Result<V, String> {
    let g = materialise(graph, false)?;
    evaluate_resolved(&g, types, vals)
}

// ---- weights by digest (6.13, GT-9)

fn stores() -> &'static std::sync::Mutex<Vec<std::path::PathBuf>> {
    static S: std::sync::OnceLock<std::sync::Mutex<Vec<std::path::PathBuf>>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::sync::Mutex::new(vec![]))
}

fn blob_cache() -> &'static std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<[u8]>>> {
    static C: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<[u8]>>>> = std::sync::OnceLock::new();
    C.get_or_init(Default::default)
}

/// Read blobs from the store in `dir` too (GT-9). The toolchain adds the
/// store `blobs` beside each program file it reads. A blob is the bytes its
/// digest names wherever it is found, so the order of the stores does not
/// change a result.
pub fn add_store(dir: &std::path::Path) {
    let mut s = stores().lock().unwrap_or_else(|e| e.into_inner());
    if !s.iter().any(|d| d == dir) {
        s.push(dir.to_path_buf());
    }
}

/// The bytes of the blob `hex` from the first store that holds it, checked
/// against the digest (once per process).
pub fn blob_bytes(hex: &str) -> Result<std::sync::Arc<[u8]>, String> {
    if let Some(b) = blob_cache().lock().unwrap_or_else(|e| e.into_inner()).get(hex) {
        return Ok(b.clone());
    }
    let d = rce_fgm::parse_hex(hex).ok_or_else(|| format!("graph term: `{hex}` is not a digest"))?;
    let dirs = stores().lock().unwrap_or_else(|e| e.into_inner()).clone();
    for dir in dirs {
        match rce_fgm::Store::new(dir).get(&d) {
            Ok(b) => {
                let b: std::sync::Arc<[u8]> = b.into();
                blob_cache().lock().unwrap_or_else(|e| e.into_inner()).insert(hex.to_string(), b.clone());
                return Ok(b);
            }
            Err(rce_fgm::StoreError::Missing(_)) => continue,
            Err(e) => return Err(format!("graph term: {e}")),
        }
    }
    Err(format!("graph term: no store holds the blob {hex} (a program's store is the directory `blobs` beside its file)"))
}

/// The digests of the blobs the graph terms of `m` name, ascending.
pub fn blob_digests(m: &Module) -> Vec<String> {
    let mut ds: Vec<String> = terms_of(m).iter().flat_map(|t| term_blobs(&t.graph)).collect();
    ds.sort();
    ds.dedup();
    ds
}

fn term_blobs(g: &FunctionGraph) -> Vec<String> {
    let mut out = vec![];
    for (i, n) in g.nodes.iter().enumerate() {
        if matches!(n, GraphNode::Apply { op: OpKind::Custom(c), .. } if c == "blob")
            && let Some(w) = g.wires.iter().find(|w| w.to == i)
            && let GraphNode::Literal { value: LiteralValue::String(d), .. } = &g.nodes[w.from]
        {
            out.push(d.clone());
        }
    }
    out
}

/// The graph with each `blob` node replaced by the tensor the store holds
/// under its digest, or, with `zeros`, by zeros of its type (for typing
/// alone, W7). A blob is its elements in row-major order, little-endian;
/// its length must be the type's (an `i8` tensor is widened to `i32` lanes).
fn materialise(graph: &FunctionGraph, zeros: bool) -> Result<FunctionGraph, String> {
    let mut g = graph.clone();
    for i in 0..g.nodes.len() {
        let GraphNode::Apply { op: OpKind::Custom(c), output_type } = &g.nodes[i] else { continue };
        if c != "blob" {
            continue;
        }
        let hex = g
            .wires
            .iter()
            .find(|w| w.to == i)
            .and_then(|w| if let GraphNode::Literal { value: LiteralValue::String(d), .. } = &g.nodes[w.from] { Some(d.clone()) } else { None })
            .ok_or("graph term: a blob node without its digest")?;
        let t = TTy::of_code(output_type).ok_or_else(|| format!("graph term: a blob of type `{output_type}`"))?;
        let size = rce_fgm::elem_size(t.elem.name()).unwrap_or(4);
        let n = t.numel() as usize;
        let bytes: std::sync::Arc<[u8]> = if zeros { vec![0u8; n * size].into() } else { blob_bytes(&hex)? };
        if bytes.len() != n * size {
            return Err(format!("graph term: the blob {hex} holds {} bytes, and {} takes {}", bytes.len(), t.show(), n * size));
        }
        let (data, dtype): (std::sync::Arc<[u8]>, &str) = match t.elem {
            Elem::I8 => (bytes.iter().flat_map(|b| (*b as i8 as i32).to_le_bytes()).collect::<Vec<u8>>().into(), "i32"),
            e => (bytes, e.name()),
        };
        g.nodes[i] = GraphNode::TensorLiteral(rce_graph::graph::TensorLiteralData {
            data,
            shape: t.shape.iter().map(|d| *d as usize).collect(),
            dtype: dtype.into(),
            name: hex,
        });
        g.wires.retain(|w| w.to != i);
    }
    Ok(g)
}

fn evaluate_resolved(graph: &FunctionGraph, types: &TermTypes, vals: &[V]) -> Result<V, String> {
    if vals.len() != types.params.len() {
        return Err(format!("graph term: {} operands for {} parameters", vals.len(), types.params.len()));
    }
    let mut inputs = rce_vm_backend_cpu::Inputs::new();
    for (i, (v, t)) in vals.iter().zip(&types.params).enumerate() {
        let tt = TTy::of_spec(t).ok_or("graph term: a parameter of an unknown element type")?;
        inputs.insert(format!("p{i}"), BValue::Tensor(to_host(v, &tt)?));
    }
    let out = match rce_vm_backend_cpu::interpret_reference(graph, &inputs) {
        Ok(v) => v,
        Err(InterpretError::NonFinite(_) | InterpretError::DivByZero(_)) => return Err(TRAP_NON_FINITE.into()),
        Err(InterpretError::Trap(_, t)) => return Err(t),
        Err(e) => return Err(format!("graph term: {e}")),
    };
    let rt = TTy::of_spec(&types.result).ok_or("graph term: a result of an unknown element type")?;
    match out {
        BValue::Tensor(t) => of_host(&t, &rt),
        BValue::Scalar(s) => Err(format!("graph term: a scalar result {s}")),
        BValue::DeviceTensor(_) => Err("graph term: a device result".into()),
    }
}

/// The reference evaluation of a graph term node by node (GT-6), as
/// [`evaluate_graph`] makes it: its result (Err the trap text) and the
/// value of every node that completed, by node index. A realisation is
/// judged against these values node by node at each node's level (EW-4).
pub fn reference_nodes(graph: &FunctionGraph, types: &TermTypes, vals: &[V]) -> Result<(Result<V, String>, Vec<(usize, HostTensor)>), String> {
    let g = materialise(graph, false)?;
    if vals.len() != types.params.len() {
        return Err(format!("graph term: {} operands for {} parameters", vals.len(), types.params.len()));
    }
    let order = rce_vm_backend_cpu::reference_order(&g).map_err(|e| format!("graph term: {e}"))?;
    let mut values: std::collections::HashMap<usize, BValue> = Default::default();
    let mut done = vec![];
    for n in order {
        let mut ws: Vec<_> = g.wires.iter().filter(|w| w.to == n).collect();
        ws.sort_by_key(|w| w.port_in_idx);
        let args: Vec<BValue> = ws.iter().filter_map(|w| values.get(&w.from).cloned()).collect();
        let r = match &g.nodes[n] {
            GraphNode::Param(p) => {
                let i: usize = p.name.strip_prefix('p').and_then(|x| x.parse().ok()).ok_or("graph term: a parameter name")?;
                let tt = TTy::of_spec(&types.params[i]).ok_or("graph term: a parameter of an unknown element type")?;
                Ok(BValue::Tensor(to_host(&vals[i], &tt)?))
            }
            node => rce_vm_backend_cpu::reference_node(n, node, args),
        };
        let v = match r {
            Ok(v) => v,
            Err(InterpretError::NonFinite(_) | InterpretError::DivByZero(_)) => return Ok((Err(TRAP_NON_FINITE.into()), done)),
            Err(InterpretError::Trap(_, t)) => return Ok((Err(t), done)),
            Err(e) => return Err(format!("graph term: {e}")),
        };
        if let BValue::Tensor(t) = &v {
            done.push((n, t.clone()));
        }
        values.insert(n, v);
    }
    let rt = TTy::of_spec(&types.result).ok_or("graph term: a result of an unknown element type")?;
    let out = match values.remove(&g.output_node) {
        Some(BValue::Tensor(t)) => of_host(&t, &rt),
        _ => Err("graph term: a result that is not a tensor".into()),
    };
    Ok((out, done))
}

/// The host tensor of a value of type `t`: its elements in row-major order.
pub fn to_host(v: &V, t: &TTy) -> Result<HostTensor, String> {
    let mut flat = vec![];
    flatten(v, &t.shape, &mut flat)?;
    let shape: Vec<usize> = t.shape.iter().map(|d| *d as usize).collect();
    let ints = || -> Result<Vec<i32>, String> {
        flat.iter()
            .map(|x| match x {
                Leaf::I(n) if t.elem == Elem::I8 && i8::try_from(*n).is_ok() => Ok(*n as i32),
                Leaf::I(n) if t.elem == Elem::I32 && i32::try_from(*n).is_ok() => Ok(*n as i32),
                _ => Err(format!("graph term: an element of an operand is not of {}", t.elem.name())),
            })
            .collect()
    };
    let floats = || -> Vec<f64> { flat.iter().map(|x| if let Leaf::F(f) = x { *f } else { 0.0 }).collect() };
    let r = match t.elem {
        Elem::F64 => HostTensor::from_f64(shape, &floats()),
        Elem::F32 => HostTensor::from_f32(shape, &floats().iter().map(|x| *x as f32).collect::<Vec<_>>()),
        Elem::I8 | Elem::I32 => HostTensor::from_i32(shape, &ints()?),
    };
    r.map_err(|e| format!("graph term: {e}"))
}

/// An element of a tensor value.
enum Leaf {
    F(f64),
    I(i128),
}

fn flatten(v: &V, shape: &[u64], out: &mut Vec<Leaf>) -> Result<(), String> {
    match (v, shape) {
        (V::F(x) | V::F32(x), []) => {
            out.push(Leaf::F(*x));
            Ok(())
        }
        (V::Int(n, _), []) => {
            out.push(Leaf::I(*n));
            Ok(())
        }
        (V::List(xs), [d, rest @ ..]) if xs.len() as u64 == *d => {
            for x in xs.iter() {
                flatten(x, rest, out)?;
            }
            Ok(())
        }
        _ => Err("graph term: an operand does not have its declared shape".into()),
    }
}

/// The value of type `t` a host tensor holds.
pub fn of_host(h: &HostTensor, t: &TTy) -> Result<V, String> {
    let shape: Vec<usize> = t.shape.iter().map(|d| *d as usize).collect();
    let n = if shape.is_empty() { 1 } else { shape.iter().product() };
    let flat: Vec<V> = match (t.elem, h.dtype) {
        (Elem::F64, DType::F64) => h.as_f64().map_err(|e| e.to_string())?.into_iter().map(V::F).collect(),
        (Elem::F32, DType::F32) => h.as_f32().map_err(|e| e.to_string())?.into_iter().map(|x| V::F32(x as f64)).collect(),
        (Elem::I8 | Elem::I32, DType::I32) => h.as_i32().map_err(|e| e.to_string())?.into_iter().map(|x| V::Int(x as i128, t.elem.name())).collect(),
        _ => return Err(format!("graph term: a result of element type {} for {}", h.dtype.name(), t.show())),
    };
    if flat.len() != n {
        return Err(format!("graph term: a result of {} elements for {}", flat.len(), t.show()));
    }
    if shape.is_empty() {
        return Ok(flat.into_iter().next().unwrap_or(V::Unit));
    }
    fn nest(flat: &mut std::vec::IntoIter<V>, shape: &[usize]) -> V {
        match shape {
            [] => flat.next().unwrap_or(V::Unit),
            [d, rest @ ..] => V::List((0..*d).map(|_| nest(flat, rest)).collect::<Vec<V>>().into()),
        }
    }
    Ok(nest(&mut flat.into_iter(), &shape))
}

/// W7 of the core (6.13, GT-2 and GT-3) on a term read back from its JSON:
/// its key is the content hash of its body and typing; every operation
/// node carries a level no stronger than its operation admits; the graph
/// is well formed and its shapes type. Ok is the term.
pub fn check_term(key: &str, graph: &FunctionGraph, types: &TermTypes) -> Result<Term, String> {
    let h = key_of(graph, types)?;
    if h != key {
        return Err(format!("W7 graph term {key} requires change: the key is not the content hash of its body and typing, which is {h}"));
    }
    graph.validate().map_err(|e| format!("W7 graph term {key} requires change: the graph is not well formed: {e:?}"))?;
    let applies: Vec<usize> = graph.nodes.iter().enumerate().filter(|(_, n)| matches!(n, GraphNode::Apply { .. })).map(|(i, _)| i).collect();
    let levelled: Vec<usize> = types.levels.iter().map(|l| l.node as usize).collect();
    if applies != levelled {
        return Err(format!("W7 graph term {key} requires change: every operation node carries one declared level, in node order; the operation nodes are {applies:?} and the levels name {levelled:?}"));
    }
    for (l, i) in types.levels.iter().zip(&applies) {
        if let GraphNode::Apply { op: OpKind::Custom(n), .. } = &graph.nodes[*i] {
            let floor = ir_level(n);
            if !floor.at_least(l.level) {
                return Err(format!("W7 graph term {key} requires change: node {i} ({n}) is declared {} and its operation admits at most {}", l.level.text(), floor.text()));
            }
        }
    }
    // the shapes: run the term on zeros of its parameter types, which
    // types every node by the backend's shape rules
    let zeros: Vec<V> = types
        .params
        .iter()
        .map(|t| TTy::of_spec(t).map(|tt| zeros_of(&tt)).ok_or_else(|| format!("W7 graph term {key} requires change: a parameter of element type {}", t.dtype)))
        .collect::<Result<_, _>>()?;
    let typed = materialise(graph, true).map_err(|e| format!("W7 graph term {key} requires change: {e}"))?;
    match evaluate_resolved(&typed, types, &zeros) {
        Ok(_) => {}
        Err(e) if is_trap(&e) => {}
        Err(e) => return Err(format!("W7 graph term {key} requires change: its shapes do not type: {e}")),
    }
    let result = TTy::of_spec(&types.result).ok_or_else(|| format!("W7 graph term {key} requires change: a result of element type {}", types.result.dtype))?;
    Ok(Term { graph: graph.clone(), types: types.clone(), key: key.into(), result, work: 0, line: 0, op: String::new() })
}

fn zeros_of(t: &TTy) -> V {
    match t.shape.split_first() {
        None => match t.elem {
            Elem::F64 => V::F(0.0),
            Elem::F32 => V::F32(0.0),
            Elem::I8 | Elem::I32 => V::Int(0, t.elem.name()),
        },
        Some((d, rest)) => {
            let inner = TTy { elem: t.elem, shape: rest.to_vec() };
            V::List((0..*d).map(|_| zeros_of(&inner)).collect::<Vec<V>>().into())
        }
    }
}

/// The table of `std.tensor` as `rce graph ops` prints it: name, level, what
/// it computes.
pub fn ops_table() -> String {
    let mut s = String::new();
    for o in OPS {
        s.push_str(&format!("{:<16} {:<6} {}\n", o.name, o.level.text(), o.doc));
    }
    s.push_str("within           -      tensor.within(N, t): every operation in t has level `within N` (N units in the last place)\n");
    s.push_str("lossy            -      tensor.lossy(t): every operation in t has level `lossy`\n");
    s
}

// ---- `rce graph`: the commands on graph terms and their `.fg` files

/// Every graph term of an admitted module, in the order of their lines.
pub fn terms_of(m: &Module) -> Vec<Term> {
    if visible_name(m).is_none() {
        return vec![];
    }
    let mut c = crate::check::Checker::new();
    c.check_module(m, &Default::default());
    let mut ts: Vec<Term> = c.graph_terms.into_values().collect();
    ts.sort_by(|a, b| (a.line, &a.key).cmp(&(b.line, &b.key)));
    ts
}

fn levels_text(t: &TermTypes) -> String {
    let mut ls: Vec<String> = t.levels.iter().map(|l| l.level.text()).collect();
    ls.dedup();
    ls.join(", ")
}

fn types_text(t: &TermTypes) -> String {
    let ps: Vec<String> = t.params.iter().map(|p| TTy::of_spec(p).map(|x| x.code()).unwrap_or_else(|| p.dtype.clone())).collect();
    let r = TTy::of_spec(&t.result).map(|x| x.code()).unwrap_or_else(|| t.result.dtype.clone());
    format!("({}) -> {r}", ps.join(", "))
}

/// `rce graph terms FILE DIR`: write each graph term of the admitted module
/// `m` as `DIR/KEY.fg` and list them: key, line, operation, typing, levels.
pub fn write_terms(m: &Module, dir: &str) -> Result<String, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    let name = visible_name(m).unwrap_or_else(|| "tensor".into());
    let mut out = String::new();
    for t in terms_of(m) {
        let mut f = rce_format::graph_section::term_file(&t.graph, &t.types).map_err(|e| e.to_string())?;
        let mut buf = std::io::Cursor::new(Vec::new());
        f.write(&mut buf).map_err(|e| e.to_string())?;
        let path = std::path::Path::new(dir).join(format!("{}.fg", t.key));
        std::fs::write(&path, buf.into_inner()).map_err(|e| format!("{}: {e}", path.display()))?;
        out.push_str(&format!("{}  line {}  {name}.{}  {}  {}\n", t.key, t.line, t.op, types_text(&t.types), levels_text(&t.types)));
    }
    Ok(out)
}

/// A `.fg` file read and checked: its content hash verified by the format,
/// and, when it holds a typing, its term checked by W7.
fn read_fg(bytes: &[u8]) -> Result<(rce_format::FgFile, FunctionGraph, Option<TermTypes>), String> {
    use rce_format::{SectionKind, graph_section};
    let f = rce_format::FgFile::read(std::io::Cursor::new(bytes)).map_err(|e| format!("requires change: not a .fg file whose content hash holds: {e}"))?;
    let gs = f.section(SectionKind::Graph).ok_or("requires change: the .fg file has no graph section")?;
    let g: FunctionGraph = graph_section::graph_from_section(gs).map_err(|e| format!("requires change: the graph section does not decode: {e}"))?;
    let ty = match f.section(SectionKind::TypeTable) {
        Some(s) => Some(rce_format::section::decode_cbor::<TermTypes>(&s.payload).map_err(|e| format!("requires change: the type table is not a graph term's typing: {e}"))?),
        None => None,
    };
    Ok((f, g, ty))
}

/// `rce graph inspect FILE`: the header, the sections and the graph term of
/// a `.fg` file.
pub fn inspect(bytes: &[u8]) -> Result<String, String> {
    let (f, g, ty) = read_fg(bytes)?;
    let key = rce_format::hash::hex_encode(&f.header.content_hash);
    let mut o = String::new();
    o.push_str(&format!("content hash   b3:{key}\nformat version {}\nsections      ", f.header.version));
    let ss: Vec<String> = f.sections.iter().map(|s| format!("{:?} ({} bytes)", s.kind, s.payload.len())).collect();
    o.push_str(&ss.join(", "));
    o.push_str(&format!("\nnodes          {}\nwires          {}\noutput node    {}\n", g.nodes.len(), g.wires.len(), g.output_node));
    match &ty {
        Some(t) => {
            check_term(&key, &g, t)?;
            o.push_str(&format!("graph term     {}  levels: {}\n", types_text(t), levels_text(t)));
        }
        None => o.push_str("graph term     none: the file has no type table, so it is a graph without a typing (W7 needs one)\n"),
    }
    for (i, n) in g.nodes.iter().enumerate() {
        let level = ty.as_ref().and_then(|t| t.levels.iter().find(|l| l.node as usize == i)).map(|l| format!("  {}", l.level.text())).unwrap_or_default();
        let ins: Vec<String> = {
            let mut w: Vec<&Wire> = g.wires.iter().filter(|w| w.to == i).collect();
            w.sort_by_key(|w| w.port_in_idx);
            w.iter().map(|w| w.from.to_string()).collect()
        };
        let what = match n {
            GraphNode::Param(p) => format!("param {} {}", p.name, p.data_type),
            GraphNode::Literal { value, .. } => format!("literal {value:?}"),
            GraphNode::Apply { op: OpKind::Custom(c), output_type } => format!("{c}({}) -> {output_type}", ins.join(", ")),
            GraphNode::Apply { op, output_type } => format!("{op:?}({}) -> {output_type}", ins.join(", ")),
            other => format!("{other:?}").chars().take(60).collect(),
        };
        o.push_str(&format!("  {i:>3}  {what}{level}\n"));
    }
    Ok(o)
}

/// The evaluation of a term `run_fg` uses: the reference, or a realisation
/// (`rce graph run wasm`, 10.18), as (key, graph, types, operands) to result.
pub type Eval<'a> = &'a dyn Fn(&str, &FunctionGraph, &TermTypes, &[V]) -> Result<Result<V, String>, String>;

/// `rce graph run FILE INPUTS`: the reference evaluation (GT-6), or `eval`, of the
/// graph term in a `.fg` file. INPUTS is a JSON array with one tensor per
/// parameter, each a nested array of numbers of the parameter's shape; the
/// result is printed in the canonical JSON of 6.1.1 (GT-8).
pub fn run_fg(bytes: &[u8], inputs: &str, eval: Option<Eval>) -> Result<String, String> {
    let (f, g, ty) = read_fg(bytes)?;
    let key = rce_format::hash::hex_encode(&f.header.content_hash);
    let ty = ty.ok_or("requires change: the file has no type table, so its graph is not a graph term (W7)")?;
    check_term(&key, &g, &ty)?;
    let j: serde_json::Value = serde_json::from_str(inputs).map_err(|e| format!("INPUTS is not JSON: {e}"))?;
    let xs = j.as_array().ok_or("INPUTS is a JSON array with one tensor per parameter")?;
    if xs.len() != ty.params.len() {
        return Err(format!("INPUTS has {} tensors and the term takes {}", xs.len(), ty.params.len()));
    }
    let mut vals = vec![];
    for (x, p) in xs.iter().zip(&ty.params) {
        let t = TTy::of_spec(p).ok_or("a parameter of an unknown element type")?;
        vals.push(of_json(x, &t)?);
    }
    let r = match eval {
        Some(e) => e(&key, &g, &ty, &vals)?,
        None => evaluate_graph(&g, &ty, &vals),
    };
    match r {
        Ok(v) => Ok(canonical(&v) + "\n"),
        Err(m) if is_trap(&m) => Ok(format!("trap: {m}\n")),
        Err(m) => Err(m),
    }
}

/// The value of type `t` a JSON tensor holds (`rce graph run` INPUTS).
pub fn value_of_json(x: &serde_json::Value, t: &TTy) -> Result<V, String> {
    of_json(x, t)
}

fn of_json(x: &serde_json::Value, t: &TTy) -> Result<V, String> {
    match t.shape.split_first() {
        None if !t.elem.is_float() => {
            let n = x.as_i64().filter(|n| if t.elem == Elem::I8 { i8::try_from(*n).is_ok() } else { i32::try_from(*n).is_ok() });
            let n = n.ok_or_else(|| format!("{x} is not an integer of {}", t.elem.name()))?;
            Ok(V::Int(n as i128, t.elem.name()))
        }
        None => {
            let f = x.as_f64().ok_or_else(|| format!("{x} is not a number of {}", t.elem.name()))?;
            if !f.is_finite() {
                return Err(format!("{x} is not finite"));
            }
            Ok(match t.elem {
                Elem::F64 => V::F(f),
                // 6.3 N7: an f32 value is the binary32 value nearest the number
                Elem::F32 => V::F32((f as f32) as f64),
                Elem::I8 | Elem::I32 => unreachable!("integers are read above"),
            })
        }
        Some((d, rest)) => {
            let a = x.as_array().filter(|a| a.len() as u64 == *d).ok_or_else(|| format!("{x} is not an array of {d} elements of {}", t.show()))?;
            let inner = TTy { elem: t.elem, shape: rest.to_vec() };
            Ok(V::List(a.iter().map(|y| of_json(y, &inner)).collect::<Result<Vec<V>, String>>()?.into()))
        }
    }
}

/// The canonical JSON of a tensor value (6.1.1, GT-8).
pub fn canonical(v: &V) -> String {
    match v {
        V::F(x) | V::F32(x) => crate::interp::float_json(*x),
        V::Int(n, _) => n.to_string(),
        V::List(xs) => format!("[{}]", xs.iter().map(canonical).collect::<Vec<_>>().join(",")),
        _ => "null".into(),
    }
}
