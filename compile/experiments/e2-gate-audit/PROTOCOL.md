# E2: gate audit (protocol)

Status: protocol only; nothing has been run. Tests H2 of
[HYPOTHESES.md](../../HYPOTHESES.md): gates differ by compiler and can be
classified.

## Question

When a check fails, does the compiler write an artefact? Under which flags
does that change? And for each gate, what does it state it preserves, under
which linking assumption, and on which trusted computing base (the three
things every gate of [CHARTER.md](../../CHARTER.md) section 4.4 names)?

## Method

1. **Declare the faults.** For each compiler, a fixed set of small programs,
   each with one injected fault, with a correct twin that differs from it
   only at the fault:
   - a type error (every compiler);
   - a borrow or ownership error (rustc);
   - an unused import or variable (go, where it is an error);
   - an invalid IR module: an LLVM IR file that the verifier does not accept,
     given to `llc`; a WebAssembly module that does not validate, given to
     `wasmtime compile`;
   - a failing contract: a Lean theorem whose proof does not check; an Ada
     precondition that GNATprove cannot prove, where GNATprove is available;
   - a generated-code fault where the compiler checks generated code
     (Erlang's `beam_validator`, through a malformed assembly file given to
     `erlc`).
2. **Run each faulty program and its twin** under the default flags and
   under each flag set that the compiler documents as changing the gate
   (for example `noEmitOnError` for tsc, `fullgraph=True` for TorchDynamo,
   `-Werror` for gcc and clang), in a fresh directory, as E1 does.
3. **Record**, per compiler, fault and flag set: the exit status; whether
   any artefact was written, by name and SHA-256; and the diagnostic's first
   line. The twin must build, or the case is void.
4. **Classify** each compiler and fault into one of: *emits nothing by
   default*; *emits by default, nothing when configured*; *emits always*.
5. **Record each gate's three statements** from the compiler's own
   documentation, with the passage quoted and its source, or "not stated".
6. **Compare** the observed class with the documented behaviour.

Results are written by the harness as JSON with the machine and toolchain
versions, and the write-up is generated from them, as in E1.

## Metrics

- Artefact written (yes or no) and exit status, per compiler, fault and
  flag set.
- The class of step 4, per compiler and fault.
- For each gate: property class, linking assumption and trusted computing
  base, each with a quoted source or "not stated".
- The number of cases where observed and documented behaviour differ (the
  falsifier of H2).

## What it needs

- Locally available: rustc, go, gcc, clang, gfortran, `llc` from LLVM 18,
  Wasmtime 26.0.1 and Lean 4.12.0.
- In CI: ghc, gnat, erlang, ocaml and, if a package exists, GNATprove; tsc
  needs Node.js and the TypeScript package.
- Tensor compilers (TorchDynamo, XLA, ONNX Runtime) need their Python
  packages; their fallbacks to eager execution or to a CPU provider are
  recorded as a class of their own.
