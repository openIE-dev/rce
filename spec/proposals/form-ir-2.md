# Form IR format 2: named definitions, state-passing borrows, sums

Status: proposal, branch `rce/host`. Owner: David Charlot. Derived from the D.7 lowering
results in `closing-the-loop.md` (457 stdlib bricks, 269 lowered). Each change below follows
from a stated property of the Form, not from a target language.

## 1. Content-addressed definitions (`ref`)

**Problem.** Format 1 terms have no names. A call is lowered by copying the callee's term into
the caller, so a term grows with the number of call sites times the callee size, and a
recursive function has no finite term at all. Of the 188 stdlib bricks not yet lowered, 77
are held back by size or depth alone.

**Derivation.** A term is a value of the Form (it is hashed into rule ids and the log). The
smallest change that keeps every term a finite value with one meaning is a table of
definitions keyed by content: the key of a definition is the SHA-256 of the canonical JSON of
its body (the same `vhash` used for oracle rows). A key therefore names exactly one body in
every table and every target, needs no global namespace, and two bricks that call the same
function share one entry.

**Syntax.**

```
program   = {"format": 2, "defs": {KEY: term, ...}, "main": term}
term     += {"t": "ref", "node": NODE, "def": KEY}
```

A format 1 term (a bare term object, no `format` field) is read as
`{"format": 1, "defs": {}, "main": term}`, so every existing fit, catalog and Form stays readable
and keeps its hash. A Form carries one `defs` table for all of its rules (`FormIr.defs`,
absent when empty), and its `format` is 2 exactly when the table is non-empty.

**Meaning.** `ref(node, k)` on input x is the definition body `defs[k]` on x. Evaluation in all
targets is identical to evaluating the body in place; the node attributes the call.

**Well-formedness (checked when a program is read).** (a) every key equals the digest of its
body; (b) every `ref` in `main` or in a body names a key of the table. Since a body's digest
covers the keys it refers to, (a) makes the reference graph acyclic: a cycle would need a
fixed point of SHA-256. A program that fails (a) or (b) requires change and is never evaluated. Recursion
therefore stays outside format 2: it needs a decreasing measure (6.7), which a later `fix`
construct with an explicit bound will carry; format 2 removes the size cause, not that one.

**Targets.** The evaluator, the JavaScript runtime and the Python emitter take the table with
the term; the Rust emitter emits one function per definition, named by its key; the e-graph
treats a `ref` as an opaque leaf (its key is its identity), and inlines a definition only when
its body is at most `INLINE_MAX` nodes, so rewriting never grows a term past a stated bound.

## 2. Mutable borrows by state-passing

A Form's step δ is a pure function from (state, event) to (state, outputs). A function with
`&mut` parameters is the same shape: it maps its arguments to a result and new argument
values. The lowered term of `f(p1, ..., pn) -> R` whose `&mut` parameters are `pi1 < ... < pik`
returns the tuple `[r, v_i1, ..., v_ik]` (the result, then the final value of every `&mut`
argument in parameter order); with no `&mut` parameter the result is `r` alone, as in format
1. A caller lowers `x = f(..., &mut a, ...)` to a call followed by writing the returned values
back to `a` (a local variable, a field path, or a list item, whichever place was borrowed).
Vectors record the same tuple from the reference evaluator, so agreement covers the written
back values too.

## 3. Sums: `match` and errors

Values keep the embedding of the reference evaluator (`form_value`): a variant is an object
with its fields and `"$v": NAME`; an error value of `T ! E` is `{"$err": e}`; `none` is
`null`. `match` lowers to a chain of `cond` on the tag (`$v`, `null`, a literal) with the
pattern's fields bound by projection, in arm order, and a trap when no arm matches (as the
evaluator does). `fail e` returns `{"$err": e}`; `x?` / `try` tests for an object with the
single field `$err` and returns it unchanged; `catch` binds the payload and evaluates the
handler. No new IR construct is needed for sums; only the lowering changes.

## 4. Order of work

Spec section, then: evaluator (`ref`, programs, well-formedness), JavaScript runtime and Python
emitter, Rust emitter, e-graph (opaque, bounded inlining), lowering by reference, `&mut`
state-passing, `match`, errors. Each step is a commit with tests; after each lowering step the
stdlib catalog is rebuilt and every lowered brick is checked against the reference evaluator
on generated vectors (`rce-brick-verify`).
