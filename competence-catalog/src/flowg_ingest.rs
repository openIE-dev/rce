//! flowg → RCE ingest coverage (best-case lift). flowg (joulesperbit, lux-flowg) is the predecessor;
//! RCE ingests it. The core compute kernel maps 1:1 onto the universal construct taxonomy
//! (structure::CONSTRUCTS); ops without a current construct (bitwise/shift/rounding/conversion and
//! the domain groups) are the expansion frontier, honestly listed, not invented.

use crate::structure::CONSTRUCTS;

/// Core flowg OpKind names (joulesperbit inv-ai-codegraph/src/flowg/op_kind.rs) that map onto a
/// current universal construct. (op, construct).
pub fn core_kernel() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Add", "add"), ("Sub", "sub"), ("Mul", "mul"), ("Div", "div"), ("Rem", "mod"), ("Neg", "neg"),
        ("Eq", "eq"), ("Ne", "ne"), ("Lt", "lt"), ("Le", "le"), ("Gt", "gt"), ("Ge", "ge"),
        ("And", "and"), ("Or", "or"), ("Not", "not"),
        ("FieldAccess", "member"), ("IndexAccess", "index"),
        ("Call", "call"), ("MethodCall", "call"), ("ClosureCall", "call"),
        ("Cast", "cast"), ("StructCreate", "record"),
        // near-frontier constructs added to the taxonomy (pass 87): now lifted
        ("Pow", "pow"), ("Abs", "abs"), ("Xor", "xor"),
        ("BitAnd", "bitand"), ("BitOr", "bitor"), ("BitXor", "bitxor"), ("BitNot", "bitnot"),
        ("Shl", "shl"), ("Shr", "shr"), ("RotL", "rotl"), ("RotR", "rotr"),
        ("Deref", "deref"), ("AddressOf", "addressof"),
        ("Parse", "parse"), ("Serialize", "serialize"), ("Deserialize", "deserialize"),
        ("Encode", "encode"), ("Decode", "decode"),
    ]
}

/// Core-kernel ops with no current construct. Empty after the pass-87 taxonomy expansion;
/// the remaining frontier is the domain groups (frontier_groups).
pub fn near_frontier() -> Vec<&'static str> {
    vec![]
}

/// (mapped, total) over the core kernel, measured against the real taxonomy.
pub fn core_coverage() -> (usize, usize) {
    let k = core_kernel();
    let total = k.len();
    let mapped = k.iter().filter(|(_, c)| CONSTRUCTS.contains(c)).count();
    (mapped, total)
}

/// Domain op groups beyond the current taxonomy (the far expansion frontier).
pub fn frontier_groups() -> Vec<&'static str> {
    vec!["collection", "string", "math/linalg", "DSP", "AI/ML", "visualization", "quantum",
         "cryptography", "symbolic", "geospatial", "spatial3d", "intelligence-attractors",
         "concurrency", "temporal"]
}
