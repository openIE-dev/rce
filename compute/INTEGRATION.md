# Integration of RCE's compute crates (points 1d and 1e)

This code was first written under the name flowG.

The crates in `compute/` are RCE's own: its graph term, the backends that
evaluate it, the emitters that print it, the readers that lift weight files
into it, the placement and energy accounting of the host, the serving
stack and the model programs. This file records how each part is ingested
and unified with the rest of RCE, and its state. It states only what the
code does today; anything else is called next, with what it needs.

Unified means four things, and each row below is judged by them:

1. **One toolchain.** Every command is a command of the one `rce`
   (`spec/checker`), on the positional command line (PC-1: words, then
   operands, no flags). There is no second binary.
2. **One grammar.** Programs are written in RCE's grammar (spec section 3).
   The earlier grammar and its lowering are retired and are not in this
   tree; what it expressed moves into RCE's grammar or library.
3. **One IR.** The graph term is a construct of the core `rce-core/1`
   (spec 6.13): `vN = graph K(a…)`, K the content hash of the term's `.fg`
   file. It is built from RCE source and lowered with the rest of the core.
4. **One specification.** The graph term, its rules (GT-1 to GT-8,
   W-GRAPH-1 to W-GRAPH-3, core W7) and its commands (10.16) are in
   `spec/book`, and the ledger records the change.

## State of each part

States: **done** (built and tested in this tree), **next** (not built; what
it needs is named), **blocked** (cannot be done here; the reason is named),
**retired** (removed, with what replaces it).

| Part | Crates | Place in RCE | State | Reason, and what is left |
|---|---|---|---|---|
| Graph term | `rce-graph` (`graph.rs`, `term.rs`), `rce-format`, `rce-vm-core` | the core's graph term (6.13, GT-2) | done | A term is a FunctionGraph with its typing (shapes, element types, the level of every operation); its key is the BLAKE3 content hash of its `.fg` file (`term_content_hash`); a group's key and the program hash cover it; W7 checks it when a core module is read. Left: the core JSON holds the graph as serde JSON, not as the `.fg` bytes. |
| Source of the term | `spec/checker/src/graph.rs` | `use std.tensor`, built-in module (4.1) | done | Tensors of `f32`, `f64`, `i8`, `i32`. `matmul`, `matvec`, `transpose`, `permute`, `reshape`, `concat`, `add` and `mul` (broadcasting), `sum`, `relu`, `exp`, `tanh`, `sigmoid`, `silu`, `gelu` (erf form), `gelu_tanh`, `softmax`, `rmsnorm`, `layernorm`, `attention`, `attention_causal`, `embed`, `matmul_q8_0`, `matmul_q4_0` (exact); `round_bf16`, `round_q8_0`, `round_q4_0` (lossy); lane-wise `+ - * /`; `tensor.within(N, t)`, `tensor.lossy(t)`; W-GRAPH-1 to W-GRAPH-3. Left: bool tensors; block formats beyond Q8_0 and Q4_0 (Q4_K, Q5_K, Q6_K); rotary position embedding, which needs `sin` and `cos` of `math.rce` in the backend. |
| Reference evaluation | `rce-vm-backend-cpu` | the reference interpreter's evaluation of a term (GT-6) | done | `interpret_reference`: fixed Kahn order, CPU kernels only, finite floats checked after every node, integer nodes checked against their type (6.3 N2). The model kernels (`ml/nn.rs`) fix their order, which is that of the portable kernels of `rce-op-nn-prims`, `rce-op-attention` and `rce-op-qmm`; `exp`, `tanh` and `erf` are `math.rce`'s, computed by `crmath.rs` (Ziv over exact integers, constants read from `math.rce`), equal to MPFR on the 36,491 vectors of `stdlib/tests/math` and to a decimal oracle on 1,951 arguments. The earlier `tanh`, `sigmoid`, `gelu` and `softmax` kernels, which called the platform's `exp` and `tanh`, are replaced. `rce run`, `test`, `trace`, `host run` and `host replay` give the same bytes on every run (`spec/checker/tests/graph_term.rs`). |
| Core evaluator | `spec/checker/src/one/eval.rs` | `rce run-core` | done | Evaluates `graph` statements with the same backend; equals the reference trace on `compute/examples/dense.rce` and on the levels program of the tests. |
| Graph library of the earlier grammar | `rce-graph` (about 100 modules: `dataframe`, `distributions`, `ode`, `optimize`, `signal_extended`, …) | RCE library modules, or `std.tensor` operations with a stated level | next | These modules are numeric routines that the earlier grammar reached as built-in calls. Each moves as an RCE library module (like `num.linalg`) or as a term operation with a CPU kernel of fixed order. None is reachable from RCE source yet. |
| Lowering from the earlier grammar | `rce-graph` (`lower_expr`, `lower_stmt`, `lower_item`, `type_map`, feature `lux`), `lux-lang` | none: RCE source lowers to the core, and the core holds the term | retired | Removed when this tree was published: one grammar, RCE's. Nothing in the default build used it. The constructs below move into RCE's own grammar and library. |
| Store of tensor data | `rce-fgm` | the store a program's terms name their weights in (6.13 GT-9) | done | A store is a directory of blobs, each the bytes of one tensor named by its BLAKE3 digest and checked when read. A term names a weight as `tensor.blob_f32(DIGEST, SHAPE)` (and `f64`, `i8`, `i32`), so its key and the program hash cover the weight; the toolchain reads the store `blobs` beside the program file; the host loads every named blob at start and records the digests and lengths in D (`host.Blobs`). The earlier single-file `.fgm` artifact (a ProgramGraph with its weights) and its Mamba-2 converter are retired: the computation is the program's source, and only tensor data is stored. |
| Fusion | `rce-fusion` | a lowering pass from term to term | next | A fused term must carry, for every node it replaces, a level no stronger than theirs, and be judged against the reference at those levels by the conformance harness. Its MSL code generation is Apple-only. |
| Emitters | `rce-emit-wgsl`, `-onnx`, `-stablehlo`, `-mlir-linalg`, `-triton`, `-wasm` | targets of `rce emit TARGET FILE` | next | Each prints a FunctionGraph; as targets of `rce emit` they print the terms of a program and are judged by the conformance harness at each node's level. For `rce emit wasm` (S7, not started): `rce-emit-wasm` covers f32 rank-2 `add`, `mul`, `matmul`, `relu` and `softmax`, and its softmax takes `exp` from a bit-trick approximation (`$fast_exp`, Schraudolph), so it realises no `std.tensor` softmax at any level; it needs `math.rce`'s `exp` in WebAssembly, the operations of GT-4 and the blob operands of GT-9, and `wasmtime`, which its tests alone use, moved to its dev-dependencies so that `rce` can link it and run its output under `wasmi`. Judging WGSL needs a WebGPU device and ONNX an ONNX runtime; neither is on this machine. `rce emit rust/python/js/scala/verilog` report status 2 for a program with terms until their tables have graph rows. |
| `.fg` writer | `rce-emit-fg` | `rce graph terms FILE DIR` | done in part | `rce graph terms` writes each term's `.fg` through `rce-format`; `rce-emit-fg`'s safetensors sidecar is not used, since a term holds no weights. |
| Ingest, FX | `rce-ingest-fx`, `rce-weight-loader` | `rce lift fx FILE` | next | See the `fx-import` row below: the FX JSON lacks shapes and weights, and five of its operations have no `std.tensor` operation. |
| Ingest, GGUF | `rce-ingest-gguf` | `rce lift gguf FILE STORE` (10.17) | done in part | The architecture `gpt2`, weights F32, F16, Q8_0 and Q4_0: the program's step is the forward pass as graph terms, its weights blobs of STORE (LIFT-1 to LIFT-4). `build_gguf` writes a whole GGUF file. Checked on a two-block model built in the test (`spec/checker/tests/lift_gguf.rs`). Left: architectures with rotary position embeddings (`llama` and its kin), which need `sin` and `cos` of `math.rce` in the backend; Q4_K, Q5_K, Q6_K; sequences shorter than the context. |
| Other backends | `rce-vm-backend-wgpu`, `rce-vm-backend-wasm`, `rce-vm-scheduler`, `rce-vm-cap`, `rce-vm-web`, `rce-c` | compute adapters of `rce host` | next | A host adapter realises a term on a fabric and is held to each node's level against the CPU reference (the exact classes of HC-12). The scheduler's parallel execution is a realisation, never the reference. |
| Operation kernels | `rce-op-attention`, `-attn`, `-blas`, `-conv1d`, `-conv2d`, `-lstm`, `-moe`, `-nn-prims`, `-qmm`, `-ssm` | kernels of host adapters, or reference kernels where their order is fixed | next | The reference kernels of `rms_norm`, `layer_norm`, `attention`, `matmul_q8_0` and `matmul_q4_0` take the portable order of `-nn-prims`, `-attention` and `-qmm` (S4), so those kernels are realisations that differ from the reference only where they call the platform's `exp` or `tanh`. Each enters a host adapter as such a realisation, judged at each node's level; `-conv1d`, `-conv2d`, `-lstm`, `-moe`, `-ssm` have no operation of `std.tensor` yet. |
| Hardware-only kernels | `rce-op-metal`, `-mps`, `-amx`, `-cuda`, `-qmm-metal` | compute adapters on their hardware | blocked | They build on Linux as arranged by the import (Apple and CUDA code behind target cfgs and stubs) and run only on their hardware, which this machine does not have. |
| Meter library | `rce-energy` (joined with the IDE's `ide-energy`) | RCE's one meter library: readings and their D.11 provenance | done | One crate, `compute/crates/rce-energy`, holds every reader of both (Linux RAPL and hwmon, macOS IOReport, SMC and powermetrics, NVML behind `nvml`, battery discharge), the simulator, calibration and attribution of the IDE, and the cumulative snapshots (`EnergyTelemetry`, `Rapl`, and `FallbackTdp`, an estimate that is never a reading) the placement crates read; the RAPL snapshot takes its wraparound from `meter::wrap_delta`. The IDE depends on it under its earlier name `ide_energy`, so `rce test measure` and D.11's provenance (`measured` and `reported` J, start and end times, wraps) are unchanged. 38 tests (6 unit, 32 integration), all of both crates'. `ide/crates/ide-energy` is retired. `ide/Cargo.lock` is left as it was: this machine cannot resolve the IDE workspace offline, and its first build with the network records the new path. |
| Placement and energy | `rce-placement`, `rce-placement-adapter`, `rce-placement-demo`, `rce-energy-xover`, `rce-silicon-spec` | the host's choice of fabric for a graph term by predicted joules; readings into D and receipts | next | Needs: a cost table for the fabric that runs the reference (the only coefficients in the tree are test fixtures from one machine's `cpu:op-blas` and `gpu:op-mps`, so a prediction here would be the analytical prior, `modelled`); a second fabric whose realisation of a term can be judged at each node's level (the wasm realisation of S7, run by wasmi); then `host.Placed` per term with the prediction and the chosen fabric, and `host.Energy` with the D.11 provenance of a reading, or the reason no meter can be read (on this machine: no powercap zone, no hwmon energy sensor), never 0 J. |
| Serving | `rce-serve`, `rce-federation`, `rce-federation-http`, `rce-prefix-cache`, `rce-spec-decode`, `rce-chat` | `rce host serve` | next | Every request becomes an input recorded in D and answered by a step, as `rce host serve http` does today; an answer computed outside δ and D is not part of a program. |
| Conformance | `rce-conformance` | `conformance/` and the conformance harness | next | It compares backends with a tolerance; it becomes the harness that judges every realisation of a term at each node's declared level against the CPU reference. |
| Model programs | about 47 crates (decoders, encoders, speech, vision, diffusion and their benches) | programs that run on the toolchain | blocked | They are Rust code that calls kernels directly, not RCE programs, so the toolchain cannot check, run or replay them. Each needs `rce lift` and the operations it uses in `std.tensor` (see Ingest). Several read weights from absolute paths of one machine. `rce-vjepa` is outside the workspace: it needs `mgai-reasoner`, which is not present. |
| Second binary | `rce-cli` | the one `rce` | retired | Its commands are folded or retired as the next table records. |
| Cross-language benchmark | `rce-bench` | none | retired | It needed `lux-codegen`, which is not present, and compared the earlier grammar's code generators. |
| External IR bridge | `rce-graph/src/ia_bridge.rs` (feature `ia-backend`) | none | retired | It converted to the IR of the external crate `inv-ai-codegraph`, which is not present, so it never built. |

## The commands of the retired second binary

| Its command | In the one `rce` | State | Why |
|---|---|---|---|
| `inspect FILE` | `rce graph inspect FILE` | done | Also checks the term by W7 and prints each node's level. |
| `run FILE` with JSON inputs | `rce graph run FILE INPUTS` | done | The reference evaluation only. Its device dispatch (`auto`, `cpu`, `wgpu`) is a realisation and moves to the host's compute adapters (next); its f32 cast changes the program, so it is a declared `lossy` step in source, not an option; its weight sidecar is replaced by operands; its parallel execution is not the reference. |
| `make-sample`, `make-sample-wasm`, `make-sample-wasm-tensor`, `make-sample-attention`, `make-sample-mlp` | an RCE program, then `rce graph terms FILE DIR` | retired | The samples were graphs built in Rust; a program written in RCE replaces each (`compute/examples/dense.rce` for the dense layer). |
| `fx-import` | `rce lift fx FILE` | next | The FX graphs `rce-ingest-fx` reads carry no shapes for inputs and intermediate values and no weight bytes, and use operations `std.tensor` lacks (`conv2d`, `batch_norm`, `elu`, `leaky_relu`, `log_softmax`); 10.17 states it. |
| `keygen`, `sign`, `verify` | a signature over a term's key | next | `rce` already holds Ed25519 keys for host identities (`ring`); signing a term's key needs a command and a rule for where signatures are kept. |
| `bench` | none | retired | A measurement script, not a command of the toolchain; energy readings belong in receipts. |

## Constructs of the earlier grammar that remain to move

The earlier grammar expressed them; its parser and lowering are retired.
Each moves into RCE as this table records.

| Construct | Where it goes in RCE | State |
|---|---|---|
| functions, `let`, assignment, `if`, `for`, `while`, `match`, `return`, `break`, `continue`, lambdas, traits and impls | RCE's own constructs (sections 3 and 5) | already in RCE; nothing to move |
| numeric built-ins (the graph library above) | RCE library modules or `std.tensor` operations | next |
| `app` with a view block, `component`, `style` | the `ui` library (`ui.view`, `ui.layout`, `ui.rules`) and a step that emits a view | next |
| `server` with routes | `rce host serve http HOST:PORT` (HC-24 to HC-26) and steps on `HttpRequest` | next |
| `signal`, `memo` | state fields, and values derived from them in steps | next: a written mapping |
| string interpolation | text concatenation, or `fmt` templates | next: a written mapping |
| pipe `|>` | a call | next: a written mapping |
| `await`, `yield`, `do` | tasks and channels (6.8), iterators (5.14) | next: a written mapping |
| ranges, record literals | ranges and struct values | already in RCE |

## Points 1d and 1e

- **1d, the graph in the toolchain.** Done: the graph term in the core,
  its source module with the operations of models (S4), its rules, its
  reference evaluation with `math.rce`'s transcendentals, the core
  evaluator, the program hash, D and replay, weights by digest and the
  store (S5), `rce lift gguf` for `gpt2` (S5), one meter library (S6, first
  part), the `rce graph` commands, the specification text and the ledger
  rows; the second binary retired. Next: placement of a term by predicted
  joules with readings in D and receipts (S6, second part) and `rce emit
  wasm` (S7), as the table states.
- **1e, the rest of the compute crates.** Next or blocked, part by part,
  as the first table states; the model programs are blocked on the lifts
  and on the operations they need.

## How this was checked

- `spec/checker/tests/graph_term.rs`: 15 tests (admission, examples,
  byte-identical runs and traces, D and replay, the core and its evaluator,
  the three source rules, W7, the program hash, the `.fg` commands; the
  model operations against their definitions in RCE over `math.rce`, the
  causal attention block `compute/examples/block.rce`, integer tensors,
  the findings of the new operations, a Q8_0 product from its `.fg`).
- `spec/checker/tests/lift_gguf.rs`: 2 tests. A two-block gpt2 model of
  width 32 (Q8_0 and Q4_0 products, an F16 position table) written with
  `build_gguf` is lifted; the program is admitted, `rce test` and `rce
  trace` are the same bytes on every run, D records its 50 blobs and
  replay re-derives the state; its logits equal a forward pass by the
  reference kernels bit for bit, and each operation is judged against the
  host adapters' kernels on the same inputs: rows, layer norm, Q8_0 and
  Q4_0 products equal on every element; `rce-op-attention` equal on its
  1,024 elements; `rce-op-nn-prims`' `gelu` (the platform's `tanh`) up to
  7 units in the last place.
- `compute/crates/rce-vm-backend-cpu/tests/crmath.rs`: `exp`, `tanh` and
  `erf` against the MPFR vectors of `stdlib/tests/math` and the decimal
  oracle of `tests/crmath_oracle.py`, bit for bit.
- `rce-spec-check`: every example admitted and every change example,
  `spec/examples/change/types/graph_*.rce` included, requires change.
- The compute crates outside the model programs, on Linux x86-64 with
  `CARGO_PROFILE_DEV_DEBUG=0`: `cargo test` of the graph term, ingest,
  emitter, placement and energy, CPU backend and kernel crates, 1,816
  tests passed; `cargo check --all-targets` of the device backends (wgpu,
  wasm, scheduler, capabilities, web, C ABI, conformance) and of the
  serving crates, clean. One timing assertion,
  `rce-op-qmm::brick1_batched_q5k_bit_exact_and_faster` (batched faster
  than per-row by more than 1.5 times), fails in the unoptimised build on
  this shared machine and passes in `--release`; its bit-identity check
  passes in both.
- `rce-op-blas` links OpenBLAS only where the system has it; elsewhere
  `sgemm_into` and `sgemm_with_trans_b_into` compute the product in a
  portable loop (k ascending from +0.0) instead of stopping, so the crates
  that call them build and test on a machine without OpenBLAS, as this one
  is.
- The model programs were not built here: the disk of this machine does
  not hold their builds, and their state is blocked as the first table
  states.
