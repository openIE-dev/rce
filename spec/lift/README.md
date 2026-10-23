# Lift probes (informative)

A lift λ_L is built per language from three artefacts, all data: the
language's tree-sitter grammar (text → concrete syntax tree, deterministic),
a semantics table (node kind → Form construct, plus the language's edge
rules from its own specification: integer width and overflow, division sign,
float mode, string unit, map order, stackless or stackful concurrency), and,
for emission, a print table (Form IR node → syntax-tree template). This
directory holds the probes that test the construction before it is in the
checker (spec/proposals/emit-targets.md, section 1; the universality
proposition U3 of the paper).

`go/`: `lift_go.py` lifts a Go struct with one pointer-receiver method per
event, `if field { return Err }` guards and field assignments into a Form.
On `door.go` the lifted Form (`door.lifted.rce`) checks, runs, has 3
behaviours and 3 control states, and its `shape_ctl` equals that of the
hand-written `spec/examples/61_dim_minimal.rce` when the two errors are
declared in the same order: the first cross-language agreement of the
canonical control form on this project (2026-10-23). The probe needs
`pip install tree-sitter tree-sitter-go`; `tests/lift_go.rs` checks the
committed lifted file against the hand-written one without Python.
