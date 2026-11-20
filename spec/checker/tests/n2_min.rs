//! Book 6.3 N2: `i64.MIN / -1` and `i64.MIN % -1` both trap. The quotient
//! `9223372036854775808` is outside i64. The Lean models state the same
//! vectors (`RCE/Expr.lean`, `RCE/Core/Vectors.lean`).

#[test]
fn i64_min_div_and_rem_trap_overflow() {
    let src = |op: &str| {
        format!(
            "fn main()\n    let m: i64 = -9223372036854775807 - 1\n    emit (m {op} -1)\n"
        )
    };
    for op in ["/", "%"] {
        let r = rce_spec_check::cli::run(&src(op), None).unwrap_or_else(|e| panic!("{op}: {e}"));
        let text = r.lines.join("\n");
        assert!(
            r.kept && text.contains("overflow: 9223372036854775808 is outside i64"),
            "{op}: {text}"
        );
    }
}
