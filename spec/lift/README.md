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

`go/`: `lift_go.py lift FILE` lifts a Go struct with one pointer-receiver
method per event, `if <condition> { return Err }` guards before any write,
field assignments and `errors.New` variables into a Form; a guard's
condition and an assigned value are fields of the receiver, literals, `!`,
`&&`, `||` and comparisons. Like the Python lift, it drops nothing: every
other construct (a function, a loop, a switch, another if, a call, a type
assertion, an array, integer arithmetic, an import other than `errors`, a
name that is an RCE keyword) requires change with one of the rules
`lift_go.py rules` prints, in the form
`requires change FILE: line N: RULE: reason`, with exit status 1 and no
Form; a construct inside one that is not carried is reported on its own
line too. `fixtures/counter.go` holds such constructs and
`fixtures/counter.requires-change` the probe's report on it.
On `door.go` the lifted Form (`door.lifted.rce`) checks, runs, has 3
behaviours and 3 control states, and its `shape_ctl` equals that of the
hand-written `spec/examples/61_dim_minimal.rce` when the two errors are
declared in the same order: the first cross-language agreement of the
canonical control form on this project (2026-10-23). The probe needs
`pip install tree-sitter tree-sitter-go`; `test_lift_go.py` checks, with
Python, that the probe still gives both committed files byte for byte, and
`tests/lift_go.rs` checks the committed lifted file against the hand-written
one, and the committed report's rules and lines, without Python.

`python/`: `lift_py.py` lifts a subset of Python 3.13 (read by Python's own
`ast`) into RCE text, route T of `spec/proposals/rce-form.md` section 16:
integers as i64 (an overflow traps, never wraps), text of code points,
lists, dictionaries kept as keys and values in insertion order, tuples,
functions (a list or dictionary the function mutates is passed `&mut`;
recursion gets `bound 1000`), `print`, f-strings with int and str format
specs, `for` over ranges and collections, and `while` loops with a count
bound from a closed list of rules (a conjunction of comparisons of
variables with limits the body does not change, every path through the
body moving them toward the limits by a positive constant; rule B4 of
section 17). Python's types are found by unification, one static type per
name. Every construct outside the subset requires change with one of the
rules `lift_py.py rules` prints, in the form
`requires change FILE: line N: RULE: reason`; nothing is dropped. The
lifted program calls `runtime.rce`, whose definitions give CPython's result
or trap: a definition that needs Unicode data the runtime does not carry
(case mapping, white space) traps on non-ASCII text, and `repr` of text
reads the non-printable ranges the lifter writes from Python's own Unicode
15.1.0 database. Aliasing is checked, not carried: a list shared by two
names, or iterated, while one of them is mutated in place requires change
(`L-PY-ALIAS`). `verify_lift.py RCE` runs the lift over
`lang/python3.13/corpus` and compares each lifted program's exit status,
standard output (the emitted texts' bytes) and standard error class with
the recorded observation; `coverage.py RCE FILE` gives function-level
coverage of one module. `fixtures/` holds positive fixtures with CPython's
recorded output and one negative fixture per rule family;
`test_lift_py.py` (Python 3.13) and `tests/lift_py.rs` (no Python) check
them. Measured results: `openIE-dev/openie-research-experiments`,
`rce/bench/lift-python/`.
