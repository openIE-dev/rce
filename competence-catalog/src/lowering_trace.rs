//! Lowering trace — flag every INVENTED decision a frontend makes when lowering the same
//! construct IR to a target (calling convention, ABI, register allocation, memory layout,
//! ownership, integer width, overflow). Two languages targeting the same backend diverge here;
//! the trace makes the divergence explicit instead of leaving it inside compiler heuristics.
//! (Handoff: "flag every invented decision in the lowering trace for two languages same backend.")

#[derive(Debug, Clone, PartialEq)]
pub struct InventedDecision {
    pub lang: String,
    pub category: &'static str,
    pub decision: &'static str,
}

pub fn trace(lang: &str) -> Vec<InventedDecision> {
    let d = |category: &'static str, decision: &'static str| InventedDecision {
        lang: lang.to_string(),
        category,
        decision,
    };
    match lang {
        "rust" => vec![
            d("calling_convention", "System V / Win64 per target; extern \"C\" for FFI"),
            d("abi", "Rust ABI unstable; repr(C) opt-in for layout"),
            d("register_allocation", "LLVM regalloc; ownership moves elide copies"),
            d("memory_layout", "stack by default; Box heap; no GC; drop flags"),
            d("ownership", "borrow-checked affine ownership; lifetimes erased"),
            d("int_width", "i64 default inference; checked/wrapping explicit"),
            d("overflow", "panic in debug, wrap in release (invented policy)"),
        ],
        "python" => vec![
            d("calling_convention", "CPython vectorcall / PyObject* args"),
            d("abi", "CPython object ABI; refcounted PyObjects"),
            d("register_allocation", "bytecode stack VM; no native regalloc"),
            d("memory_layout", "heap PyObject + refcount/GC; arena alloc"),
            d("ownership", "reference counting + cycle GC (invented)"),
            d("int_width", "arbitrary-precision int (invented)"),
            d("overflow", "never wraps; bigint promotion (invented)"),
        ],
        "c" => vec![
            d("calling_convention", "System V / Win64"),
            d("abi", "C ABI stable; struct padding impl-defined"),
            d("register_allocation", "LLVM/GCC regalloc"),
            d("memory_layout", "manual malloc/free; no GC"),
            d("ownership", "none (manual; invented by programmer)"),
            d("int_width", "impl-defined int/long widths"),
            d("overflow", "signed UB, unsigned wrap (invented)"),
        ],
        _ => vec![d("calling_convention", "target-default"), d("memory_layout", "target-default")],
    }
}

/// Pairwise divergence between two languages' invented decisions on the same categories.
pub fn compare(a: &str, b: &str) -> Vec<(&'static str, String, String)> {
    let ta = trace(a);
    let tb = trace(b);
    let mut out = Vec::new();
    for x in &ta {
        if let Some(y) = tb.iter().find(|y| y.category == x.category) {
            if x.decision != y.decision {
                out.push((x.category, x.decision.to_string(), y.decision.to_string()));
            }
        }
    }
    out
}
