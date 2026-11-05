# Proposal: graph terms in WebAssembly (`rce emit wasm`)

Status: accepted (ledger row of 2026-11-04). Implemented in
`compute/crates/rce-wasm-kernels` (the kernel library),
`compute/crates/rce-emit-wasm` (the program module) and
`spec/checker/src/wasm_term.rs` (the run under `wasmi`), specified in 10.5
and 10.18 (EW-1 to EW-6) of the specification, and tested in
`spec/checker/tests/emit_wasm.rs`. Owner: David Charlot.

## 1. Terms

- *Realisation*: a computation of a graph term other than the reference
  evaluation (GT-6), judged against the reference node by node at each
  node's level (GT-4).
- *Kernel library*: the WebAssembly module
  `compute/crates/rce-emit-wasm/kernels/rce-kernels.wasm`, the CPU
  backend's one-node step `reference_node` compiled to `wasm32-unknown-unknown`.
- *Program module*: the WebAssembly text `rce emit wasm` prints; it has no
  arithmetic of its own and calls the kernel library once per node.
- *Slot*: a numbered tensor held by the kernel library.

## 2. The state before

`rce-emit-wasm` printed f32 rank-2 `add`, `mul`, `matmul`, `relu` and
`softmax` as hand-written WebAssembly, and its softmax took `exp` from a
bit-pattern approximation (`$fast_exp`, Schraudolph). That softmax
realises no `std.tensor` softmax at any level, and every other operation
of GT-4 and the blob operands of GT-9 had no form at all.

## 3. Decision

The emitter prints the term's structure and nothing else; the arithmetic
of every node is the reference's own step, compiled once.

1. `rce_vm_backend_cpu::reference_node` is the step the reference
   evaluation takes for every node but a parameter (the integer path, the
   operation, the literal, the finite check), and `interpret_reference`
   calls it, so the reference and the library share one function.
2. `rce-wasm-kernels` wraps that function behind a slot interface
   (`rce_alloc`, `rce_put`, `rce_put_bytes`, `rce_node`, `rce_copy`,
   `rce_get`, `rce_get_len`, `rce_msg`, `rce_msg_len`). A node's
   descriptor is JSON: its output slot, its operand slots, and either the
   graph node, a value already in a slot, or a blob with its digest, element
   type and shape. The status of `rce_node` is 0 done, 1 non-finite, 2
   trap, 3 cannot run, 4 bad digest.
3. The program module imports memory and the five functions it calls from
   `rce-kernels/DIGEST`, DIGEST the library's BLAKE3 digest, and exports
   `term_KEY` per term. Each node's descriptor is a passive data segment,
   copied in with `memory.init`. A failing node returns `(n + 1) << 8 |
   status`, n the node index. Tensor literals go in their own segments,
   since the `.fg` form does not carry their data in the descriptor.
4. The node order is `reference_order` on the term with the blob nodes'
   digest wires removed, the graph the reference evaluates after it
   replaces each blob node by its tensor.
5. The library is built by `scripts/build-wasm-kernels.sh` with rustc
   1.98.0 in the release profile with `--remap-path-prefix`, so it holds no local
   path; it is committed with its digest, and `rce` refuses a library whose
   bytes are not that digest.
6. `rce graph run wasm` runs the program module under `wasmi` (=1.1.0)
   only when the HC-13 float probe passes, links the library instance under
   its digest name, puts each parameter and each blob's bytes, calls the
   term, and reads every node's slot back for judgement.

## 4. Why

- Exact nodes are the reference's bits by construction: there is one
  definition of each operation, so no second kernel can drift from it.
  This includes `exp`, `tanh` and `erf`, which are `math.rce`'s correctly
  rounded functions (6.3 F9); `$fast_exp` is retired.
- Naming the import module by digest makes a link to any other library
  fail at instantiation rather than compute something else.
- Checking a blob's digest inside the library makes the check part of the
  realisation, independent of the store that supplied the bytes.
- A hand-written WebAssembly kernel per operation was not chosen: it would
  be a second implementation of every operation of GT-4, each needing its
  own judgement, and the transcendentals would need a port of `math.rce`.

## 5. Not covered

WGSL and ONNX outputs need a WebGPU device and an ONNX runtime to be
judged; neither is on the machine this was built on, so they stay next in
`compute/INTEGRATION.md`. The library is 773,145 bytes; a smaller one per
program (only the operations a program uses) is a later size change, not a
change of meaning.
