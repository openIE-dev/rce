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
- J. J. M. M. Rutten, Universal coalgebra: a theory of systems, TCS 249, 2000 (6.10: the final coalgebra of (O\* × X)^E, T57).
- J. W. de Bakker and J. I. Zucker, Processes and the denotational semantics of concurrency, Information and Control 54, 1982 (6.10: the prefix ultrametric, T59).
- A. Nerode, Linear automaton transformations, Proc. AMS 9, 1958; E. F. Moore, Gedanken-experiments on sequential machines, Automata Studies, 1956; J. Hopcroft, An n log n algorithm for minimizing states in a finite automaton, 1971 (6.10: dim(P), T60).
- C. P. Pfleeger, State reduction in incompletely specified finite-state machines, IEEE Trans. Computers C-22, 1973 (6.10: why totality gives a unique minimum).
- L. D'Antoni and M. Veanes, Minimization of symbolic automata, POPL 2014 (6.10: the control dimension).
- R. Reis, N. Moreira and R. Almeida, On the representation of finite automata, DCFS 2005 (6.10: the canonical string of `shape_ctl`).
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

- Sources for draft 0.9 (Appendix F), by name; the survey read 97.6 percent of the fetched words of the language sources (the unread parts hold no language rules), and its notes are kept with the survey: ISO C++26 contracts (P2900) and static reflection (P2996); the Koka language book; the Austral specification; the Dafny reference manual; the Lean 4 release notes (4.35); the Rust 2024 edition guide and 1.88 release notes; the Verse language reference; the Mojo manual; Swift ownership proposals; the Hylo tour; the Odin overview and FAQ; the Roc language tour; PEP 750 (template strings); the Valen language repository (group borrowing); S. Willison, "The lethal trifecta for AI agents" (F.7); FaCT, a language for constant-time code (F.5.4); the SLSA specification, level 1 (F.5.3); RFC 8032, Ed25519 (F.5.3).

**Read receipt.** Who wrote "Communicating Sequential Processes"? (surname)

<!-- body-sha256: 56d394f5aa9e1e2c5d58210375755691b91b6c000cc3ec27cfc7ca5315ac493a -->
