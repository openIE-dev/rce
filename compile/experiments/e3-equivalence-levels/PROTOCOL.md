# E3: equivalence-level audit (protocol)

Status: protocol only; nothing has been run. Tests H3 of
[HYPOTHESES.md](../../HYPOTHESES.md): every optimisation pass and flag can
be assigned one of the four equivalence levels of
[CHARTER.md](../../CHARTER.md) section 4.3 (exact; bitwise within an
accumulation class; within a declared error bound; declared lossy, as a
program change).

## Method

1. **Declare the passes and flags, and assign each a level before the run.**
   The first set: `-O0`, `-O2` and `-O3` (exact); `-ffp-contract=off` and
   `-ffp-contract=fast` (fused multiply-add: within a declared error bound);
   `-fassociative-math` with vectorisation, and `-ffast-math` and `-Ofast`
   (reassociation: within a declared error bound, with the bound stated);
   rustc's default and `-C target-cpu=native`; Go's fused multiply-add
   where the target has it; and, on GPU runners, split-K reductions and
   lowering to FP8 (declared lossy).
2. **Declare the corpus**: integer kernels (exact by construction) and
   floating-point kernels with sums, dot products, polynomial evaluation and
   reductions over arrays of several lengths, with fixed inputs that include
   subnormal numbers, signed zeros, infinities and NaNs.
3. **Build the reference.** The exact reference is the same computation in
   rational arithmetic (Python's `fractions`) from the same inputs, rounded
   once at the end; the declared program is the `-O0` build with strict
   IEEE 754 semantics.
4. **Differential testing.** Run every build on every input and record its
   output bits. Each record is evidence of authority *pinned*
   ([CHARTER.md](../../CHARTER.md) section 4.4), bound to the SHA-256 of the
   kernel's source, of the compiler's executable, of the built binary and of
   the input, with the flag set. It shows the level held, or failed, on
   those inputs only: an observed change beyond the assigned level is a
   counterexample, and agreement on every input certifies nothing.
5. **Check the declared level** with a static tool where one exists for the
   kernel: an SMT solver for bitwise equality on loop-free code, or an
   interval or error-analysis tool for a stated bound. Which tools are
   available is not yet established. A level is *certified* only by
   evidence of authority *proved*: the tool's verdict, bound to the SHA-256
   of the binary or IR it checked, of the tool's executable and of the
   stated level. A verdict bound to other bytes than the build of step 4
   fails closed: the level is not certified. A level that only a heuristic
   or a model assigns is inferred evidence and is recorded as not
   certified (undecided), never as certified.

Results are written by the harness as JSON with the machine and toolchain
versions, and the write-up is generated from them, as in E1.

## Metrics

Per pass or flag and kernel:

- the assigned level;
- bits differing from the declared program's output;
- the largest ULP distance from the declared program's output, and the
  largest absolute and relative error against the exact reference;
- whether a static check certified the level (evidence of authority
  proved, bound to the build's digests), failed to, was undecided, or was
  not available, with the digests its verdict is bound to;
- the number of cases whose observed change exceeds the assigned level
  (the falsifier of H3).

## What it needs

- CPU compilers run locally: gcc, clang, gfortran, rustc and go.
- Tensor compilers (XLA, TorchInductor, Triton, TVM, TensorRT, IREE) and the
  FP8 and FP4 formats need GPU runners, which this project does not have
  yet.
- A static error-analysis tool, to be chosen and pinned before the run.
