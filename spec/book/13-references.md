## References

This section lists the sources the spec draws on. It has no rules. Example:
the f32 rule in 6.3 cites Figueroa, "When is double rounding innocuous?".

- The Go Programming Language Specification, go.dev/ref/spec (EBNF style, layout of the spec).
- The Rust Reference, doc.rust-lang.org/reference; R. Jung et al., RustBelt, POPL 2018.
- Zig Language Reference, ziglang.org/documentation/master (comptime, error unions).
- The Swift Programming Language, docs.swift.org (argument labels, value types, actors).
- D. Racordon et al., Implementation Strategies for Mutable Value Semantics, JOT 2022 (second-class references, Hylo).
- Python Language Reference §2.1.8, Indentation (layout algorithm).
- L. Maranget, Warnings for pattern matching, JFP 17(3), 2007 (exhaustiveness).
- Y. Gurevich, Sequential Abstract State Machines Capture Sequential Algorithms, ACM TOCL 2000 (pre-state update semantics).
- C. A. R. Hoare, Communicating Sequential Processes, CACM 1978.
- J.-P. Bernardy et al., Linear Haskell, POPL 2018. N. Mitchell, Leaking Space, CACM 56(11), 2013.
- Rust RFC 2945 ("C-unwind" ABI); The Rust Edition Guide, Rust 2024 (`unsafe extern`).
- Zig 0.15.1 release notes (wrapping `+%` and saturating `+|` operators, allocators as parameters).
- Full citations for every draft 0.4 decision are in `spec/LANGUAGE-REVIEW.md`.
- S. A. Figueroa, When is double rounding innocuous?, ACM SIGNUM Newsletter 30(3), 1995 (f32 via f64).
- IEEE Std 754-2019, IEEE Standard for Floating-Point Arithmetic, doi:10.1109/IEEESTD.2019.8766229 (6.3 N3).
- The CORE-MATH project, correctly rounded mathematical functions, https://core-math.gitlabpages.inria.fr/ (6.3 N3 F9).
- The Unicode Standard, Version 18.0.0, https://www.unicode.org/versions/Unicode18.0.0/ (6.3 N7).
- Go 1.23 release notes, range-over-func iterators (push iterators, 5.14).
- A. Rundgren, B. Jordan, S. Erdtman, JSON Canonicalization Scheme (JCS), RFC 8785, June 2020, https://www.rfc-editor.org/rfc/rfc8785 (6.1.1).
- G. Plotkin, A Structural Approach to Operational Semantics, 1981; A. Wright, M. Felleisen, A Syntactic Approach to Type Soundness, 1994 (progress and preservation).

**Read receipt.** Who wrote "Communicating Sequential Processes"? (surname)

<!-- body-sha256: 93c398eee22b1538960aca4057d67e98d642d4cbd5e999a47bffe93af21b0529 -->
