# RCE Compile charter

Status: research, draft. This charter states what RCE Compile tests and the
standard it measures compilers against. Nothing in it is a measured result
except where a line says so and names the experiment.

## 1. Question

A *compiler* here is any program that turns a program text into another
program text, down to machine code or a lower representation. RCE Compile
asks whether one *Law of Compilation* holds for every compiler: a statement
of what a correct, well-ranked compiler must satisfy, true of gcc, rustc,
Go, GHC and the rest alike, and independent of RCE. RCE (the Reality Code
Engine) is the language of this repository; for RCE Compile its tools are
one subject among many, and nothing here uses RCE's checker, its Form or its
specification.

The research report that founded the project (*Law of compilation
research*, 1 November 2026; not in this repository) found that the
candidate law, read as a *description* of existing compilers, does not hold,
and that it holds up as a *normative* standard once three of its clauses
are rewritten. This charter adopts that finding: the statement in section 4
is a standard each compiler is placed against, and the project claims a law
only for what the registered hypotheses ([HYPOTHESES.md](HYPOTHESES.md))
survive.

## 2. Evidence marks

Every statement about work outside this repository carries the mark the
report gave it:

| Mark | Meaning |
|---|---|
| [F] | read in the source itself |
| [S] | seen only in a search-result snippet or a secondary summary: a source exists and its abstract or snippet says this; the paper was not read |
| [K] | from memory, not checked in the report's sessions |
| [M] | measured on the report's machine |
| [NOT FOUND] | searched for; no source located |

Most of the report's external claims are [S] or [K], because the hosts that
hold the primary texts could not be reached from its sessions. A statement
here marked [S] or [K] is therefore unverified against its primary source.
A measurement made by this project names its experiment instead of a mark.

## 3. Principles

The project works from two principles set at its start.

1. **Deterministic first, everything gated.** At least 98% of the work is
   done by deterministic means and at most 2% by stochastic ones, and every
   output, from either kind, passes a gate before it is released. A
   *deterministic* step gives the same output for the same declared inputs;
   a *stochastic* step draws on randomness or on a source whose output is not
   a function of its inputs. The 98/2 split is a hypothesis to be measured
   for each means of deciding a compiler step (the rungs of the gear ladder,
   defined in section 6), not a claim. The report found no published study
   that measures such a split [NOT FOUND].
2. **Logical, from hindsight, physics and information theory.** Each rule of
   the standard is derived from what existing compilers show (hindsight: the
   record the report surveyed), from physical limits on computation, or from
   limits on description and decision in information theory and
   computability. A rule with none of these behind it is a proposal, not
   part of the standard.

## 4. The Compiler Form

### 4.1 Terms

- A *representation* is a program form a compiler holds. An *intermediate
  representation (IR)* is one held between source and machine code; every
  compiler the report surveyed lowers through a chain of 3 to 6 of them [K].
- *Lowering* is one step from a representation to a lower one.
- *Undefined behaviour (UB)* is source behaviour the language standard gives
  no meaning to.
- *Refinement*: a target program refines a source program when every
  observable behaviour of the target is a behaviour of the source.
- A *checker* decides whether an output meets a stated property. A *gate*
  releases an output only after a checker has accepted it, and records that
  acceptance.
- A *trace property* is a property of single runs; a *hyperproperty* relates
  several runs (non-interference is one). A *robust* property is preserved
  even when the compiled code is linked with arbitrary target-level code.
  *Full abstraction* means that source components which no source context
  can tell apart compile to target components which no target context can
  tell apart.
- The *linking assumption* of a guarantee names the other code the output
  may be combined with while the guarantee still holds.
- The *trusted computing base (TCB)* of a gate is everything its verdict
  relies on without checking: the semantics, the proof kernel, the
  assembler, the hardware models or a second compiler.
- A *stop* is a lowering step that ends with a diagnostic and no output.
- A *declared input* is an input the compiler's contract lists: source,
  flags, toolchain and any build path after remapping.
- *SMT* (satisfiability modulo theories) solvers such as Z3 decide formulas
  over arithmetic, bit-vectors and memory; they are the usual means of
  checking rewrite rules.
- *WebAssembly* (Wasm) is a portable binary program format. *Cranelift* is
  the code generator used by the Wasmtime engine; its lowering rules are
  written in the rule language ISLE.
- An *e-graph* stores many equivalent forms of a program at once, grouped
  into equivalence classes (*e-classes*).

Names of compilers, products and venues are written as their projects
write them.

### 4.2 The seven parts

The Compiler Form has seven parts. For each, the table gives the
requirement of the standard and what the report found in existing
compilers.

| Part | Requirement | What the report found |
|---|---|---|
| R, representation | a canonical representation with an equivalence relation and a name-free hash: bound names replaced by positions, the build path removed, then hashed | Canonical forms exist (GHC Core, LFortran's ASR, Lean's LCNF) and e-classes are canonical sets [K]. No compiler surveyed hashes an IR free of names; GHC's `.hi` fingerprints, rustc's stable hashes, Go's action IDs and the others hash named or textual content [K]. R is therefore an *operation* the framework applies to any IR, not a property compilers are claimed to have. |
| L, lowering | each step preserves a declared contract (below) or stops | Morris's commuting square (POPL 1973): compiling then interpreting equals interpreting then encoding [S]. CompCert's theorem is refinement under the precondition that the source has no UB [K]. Shipping compilers fall short of it. |
| C, cost | a stated cost ranks the candidates that the checker accepted | Almost every cost is a local heuristic in cycles, bytes or node counts [K]. Cranelift's acyclic e-graph is the only production case found that ranks the members of an explicit equivalence class [K]. No cost is in joules [K]. |
| K, checker | a named checker certifies the declared contract | The strongest part across the field, but it guards the source program far more than the lowering; checks of lowering exist only in pieces (Alive2 for some LLVM rewrites, SMT verification of some Cranelift ISLE rules) [K]. Only CompCert and CakeML check all of their lowering, and CompCert leaves the preprocessor, assembler and linker outside the proof [S]. |
| U, reuse | earlier results are reused, keyed by the name-free hash of R together with the contract | Content-addressed caches exist in Go, rustc, LLVM, Wasmtime, JAX, Triton and TVM [K], but their keys hash names and paths, so equal programs with different names miss the cache [K]. |
| E, energy | every released output is reported with measured energy in its regime | Absent in every compiler surveyed [K; no counterexample found]. The research language Eel (MIT, 2016) annotates energy, charging for bit creation and destruction [S]. E is a part the framework adds, supplied by an external meter. |
| G, gate | nothing is released unless G records K's acceptance; each gate states what it guarantees | Present in some compilers: rustc emits no code after a type or borrow error, Go stops on unused imports and variables, Wasm engines do not load an invalid module, Erlang's `beam_validator` checks generated code [K]. tsc emits JavaScript despite type errors unless `noEmitOnError` is set [K]. |

The letter G names both the gate part and, with a digit, a rung of the gear
ladder; the rungs are always written G0 to G5.

### 4.3 Equivalence is a declared contract

"Semantics-preserving" is replaced by a contract that each lowering
declares and its checker certifies. The contract has four levels. A
*floating-point accumulation class* is the set of evaluation orders of a
reduction that give the same bits; *ULP* (unit in the last place) is the
spacing of floating-point numbers at a value.

| Level | Meaning | What K must certify |
|---|---|---|
| Exact | integer, Boolean or fixed-point values; identical output bits | bitwise equality |
| Bitwise within an accumulation class | floating point, with the reduction order recorded in R; L and C choose only within the class | the same class, then bitwise equality |
| Within a declared error bound | floating point with an absolute, relative or ULP bound | the bound, by static error analysis or interval methods, not by sample tests |
| Declared lossy, as a program change | quantisation or lowering of precision | nothing about equality: the lowering is a change of program, recorded and admitted by G as a separate decision |

The default is exact unless a checked contract says otherwise. The levels
come from the report's notes on tensor compilers (compilers of
array programs such as neural-network models: XLA, TorchInductor, TVM,
TensorRT, Triton, IREE), which all test against a reference within an
absolute and relative tolerance [K], because fusion, tiling and reordered
reductions change floating-point sums [K]. A September 2026 preprint gives
static checkers that decide bitwise equivalence across autotuner
configurations (arXiv 2609.11356) [S, provisional: only part of its
conclusion was visible]. Halide shows the pattern at small scale: a
schedule cannot change results except through named reassociating
reductions [K].

### 4.4 A gate names three things

Secure compilation shows that a compiler can preserve semantics in
CompCert's sense and still let target-level code break source-level
abstractions; full abstraction and robust property preservation are
distinct, stronger guarantees (Patrignani, Ahmed and Clarke 2019) [S].
Patterson and Ahmed (ICFP 2019) show that "correct" is ambiguous until the
theorem names the code the output may be linked with [S]. Every gate of the
standard therefore states:

1. **the property class it preserves**: trace properties, hyperproperties,
   a robust variant, or an equivalence at one of the four levels;
2. **its linking assumption**;
3. **its trusted computing base.**

A gate that does not state all three is recorded as a gate with an
unstated guarantee. Fully abstract compilation usually needs run-time
checks or isolation, so it raises cost [K].

### 4.5 Lowering is a standard to measure against

No shipping compiler meets L as a description. Csmith found more than 325
previously unknown bugs in C compilers [S]. EMI (equivalence modulo inputs:
mutating code that a given input never runs and checking that the output
stays the same) lists 1,602 GCC and LLVM bugs, 1,007 fixed [S]. Alive2, a
bounded translation validator for LLVM, found 47 new bugs, 28 fixed [S].
Zig's self-hosted x86_64 backend, the default for Debug builds since 0.15,
passes 1,984 of 2,008 behaviour tests in one source and 1,987 in another
[S]. L is therefore a standard: a step either produces output its checker
accepts or stops, and each compiler is reported by how far it falls short.

Determinism of lowering is the first property measured. The report's probe
on its machine found every same-directory rebuild byte-identical for gcc,
clang, rustc, go and gfortran, and every cross-directory difference traced
to the build path and removed by remapping [M]. The compiler is thus a
function of (source, flags, toolchain, path), and the path is a name a
name-free R or U must strip. Experiment E1 repeats and extends that probe
([experiments/e1-determinism/](experiments/e1-determinism/)).

### 4.6 Normative statement

> **Compiler Form.** A compiler is a partial function from declared inputs
> (source, flags, toolchain, and any build path after remapping) to a chain
> of representations R₀ → … → Rₙ. Each step either stops or produces an
> output that a named checker K accepts as preserving a declared contract:
> a property class, an equivalence level, a linking assumption and a trusted
> computing base. Among accepted outputs, a stated cost C ranks the
> candidates. Reuse U is keyed by a name-free hash of the representation
> together with its contract. Nothing is released unless G records K's
> acceptance. For every released output, the framework reports its distance
> from two bounds, the bits irreversibly overwritten per run (scaled by
> k_B·T·ln 2) and a stated description-length proxy, next to the energy
> measured in the regime the hardware belongs to.

This is the report's refined statement, with a step that ends without output called a stop.
It drops "holds for any compiler" in favour of "every compiler can be
placed against it"; it replaces "semantics-preserving" with a declared
contract, "canonical" with an operation, and "Landauer floor and
Kolmogorov target" with two computable counts and a measurement, as the
next section sets out.

## 5. The convergence function

### 5.1 The two bounds, as concepts

The candidate law ranked compilers by a cost whose floor is Landauer and
whose target is Kolmogorov-minimal. Both are kept as *conceptual bounds*.

- **Landauer's limit** is the minimum heat, k_B·T·ln 2, released into
  surroundings at temperature T when one bit of information is erased:
  about 2.87 zeptojoules at 300 K [K, arithmetic]. It bounds heat per bit
  *erased*, not per operation, and is reached only in the infinitely slow
  limit; erasing faster costs more, roughly as 1/τ for erasure time τ [K].
  Single-bit experiments confirm the bound at that limit (Bérut et al.
  2012; Hong et al. 2016) [S].
- **Kolmogorov complexity** K(x) is the length of the shortest program that
  prints x.

### 5.2 Why neither is computable as stated

1. K(x) is uncomputable; compressors give only upper bounds [K]. Program
   equivalence, the condition the target sits under, is undecidable in
   general by Rice's theorem [K]. The full-employment theorem for compiler
   writers follows: no compiler can always produce the smallest equivalent
   program, since that would decide halting [K].
2. Even the exact minimum heat of one fixed circuit is intractable.
   *PP* is the class of problems a polynomial-time randomised machine
   decides correctly more than half the time; *NP-hard* problems are at
   least as hard as every problem whose solutions can be checked in
   polynomial time. Computing optimal heat
   functions is PP-complete according to the abstract of arXiv 2610.33193,
   whose body reportedly argues PP-hardness, so membership in PP is
   unverified; multiplicative approximation is NP-hard, and only additive
   approximation has a polynomial randomised algorithm [S]. Choosing among
   equivalent circuits, which is the compilation problem, sits on top of
   that.
3. The floor is far below today's costs. At commonly quoted 45 nm figures a
   32-bit integer add is about 3.5×10⁷ times k_B·T·ln 2, a read of DRAM
   (dynamic main memory) about
   2.2×10¹¹ times, and an out-of-order instruction about 2.5×10¹⁰ times
   (Horowitz, ISSCC 2014) [S; arithmetic K]. The gap is 10⁶ to 10¹¹, and
   almost all of it is wires, memory and control, not logical erasure.
4. The two terms overlap. Bennett showed that any computation can be made
   logically reversible, so the Landauer floor applies only to information
   a program discards [K]; Zurek and Bennett et al. showed that the minimum
   cost of erasing a string x is about k_B·T·ln 2 · K(x) [K]. Minimising both
   jointly counts the same quantity twice.
5. Shortness and low energy trade off (unrolling, vectorising and table
   lookup make code longer and cheaper to run) [K], so a joint objective
   without a stated weighting is a set of trade-off points, not a function.
   Time is missing from the statement.
6. Compile-time energy can outweigh the savings: in an October 2026 study,
   42 of 44 configurations of model self-refinement used 4–886% more
   execution energy than the reference solutions (arXiv 2610.32891) [S].

One result supports the idea at its core: two circuits computing the same
function can have different minimum dissipation (Wolpert and Kolchinsky,
"circuit Landauer loss", arXiv 1806.04103) [S], so circuit structure, which
a compiler chooses, changes the physical floor.

### 5.3 The measurable replacement

The convergence function is replaced by two computable counts, each
reported next to measured energy:

1. **ε, the erasure count**: the number of bits a run overwrites
   irreversibly, computed from the IR (bits overwritten or discarded
   without having been copied out). ε·k_B·T·ln 2 is the run's Landauer
   bound.
2. **λ, a declared description-length proxy**: IR node count, binary bytes,
   compressed bytes, or a two-part minimum-description-length (MDL) code
   relative to a named model class. The proxy is declared before a
   measurement and not changed after it.

Both are computed under equivalence checked on a decidable fragment, and
both are reported with the joules measured in one of three *energy
regimes*:

| Regime | Energy scales with | Typical hardware |
|---|---|---|
| A | counted operations | in-order cores, FPGAs (field-programmable gate arrays) |
| B | time multiplied by power | out-of-order CPUs (central processors), GPUs (graphics processors) |
| C | entropy production; Landauer units apply directly | simulation, or thermodynamic hardware once a measured device exists |

Regimes A and B are approximations: static power scales with time in both
[K]. The claim becomes testable (H4): does lowering ε and λ predict lower
measured energy in each regime? The report found no published measurement
of the erasure content of real compiled programs against their energy [NOT
FOUND], so the result is new evidence whichever way it comes out. The
self-published preprints that claim convergence to the Landauer limit at a
stated rate are unreviewed, and their rate concerns estimating work, not
programs approaching a floor (for example Zenodo 18922172) [S]; they are not
cited as support. Energy figures in this project are *modelled* unless a
meter reading backs them.

## 6. The gear ladder

The *gear ladder* orders the means by which a compiler step is decided,
deterministic rungs first.

| Rung | Means | Kind |
|---|---|---|
| G0 | reuse a stored result by its hash | deterministic |
| G1 | apply proven rewrite rules | deterministic |
| G2 | extract the cheapest form from an e-graph by a stated cost | deterministic |
| G3 to G5 | stochastic proposers; every output passes the gate | stochastic |

*Equality saturation* applies rewrite rules to an e-graph (section 4.1)
until no rule adds anything new or a budget runs out, then *extracts* one
form by a cost function. The report
defines G3 to G5 only as a group; their separate definitions are open. A
claim that G1 uses Z3-certified rewrites was not matched to any system
[NOT FOUND]: Alive2 and Souper check LLVM rewrites with SMT, while egg does
not certify its rules [K].

The 98/2 principle is hypothesis H6: rungs G0 to G2 decide at least 98% of
accepted decisions on a declared corpus, and every stochastic proposal is
gated. Before the first measurement the unit is declared; candidate units
are accepted decisions, accepted rewrites, CPU seconds and joules, each
reported on its own because they can disagree. What the literature shows
is the two ends, not the split [S]: equality saturation, Souper and
Cranelift decide deterministically; in LLVM a fixed policy compiled into
the compiler decides one class of choices (inline or not, which register
to evict) for gains of 0.3–7% [S]; and in
the Anthropic and Theorem work a model wrote about 180M lines of Rocq that
the Rocq checker accepts or not in full [S], so by authoring effort the
split is inverted while the gate is total. The literature supports the two
premises beneath the split. *Gate everything*: SuperCoder's best
configuration still fails 5% of its test-checked outputs [S]. *Put
deterministic reuse first*: "Compiled AI" moves all stochastic work into a
one-time compile phase and breaks even after about 17 runs (arXiv
2604.05150) [S].

## 7. Prior art, and what must be read before claiming priority

No peer-reviewed source in English, Chinese, Russian or Japanese was found
that names or proves a single law of compilation [NOT FOUND]; national
databases were not queried directly, so the negative result is open there.
The closest earlier claims each cover some of the seven parts:

| Work | Claim | Parts | Mark |
|---|---|---|---|
| Morris, "Advice on Structuring Compilers and Proving Them Correct", POPL 1973 | compiler correctness is a commuting square of algebra homomorphisms | L, K | [S] |
| Futamura, "Partial Evaluation of Computation Process", 1971 | specialising an interpreter on a program gives a compiled program; specialising the specialiser gives a compiler, then a compiler generator | L, U | [S] |
| Ershov, "On the essence of compilation", IFIP working conference, 1977 | compilation as mixed computation of an interpreter | L, U | [K], not read |
| Ershov, "Mixed computation: potential applications and problems for study", TCS 18, 1982 | the Novosibirsk school's mixed computation | L, U | [S] |
| Hoare, He and Sampaio, "Normal form approach to compiler design", Acta Informatica 30, 1993 | compilation as reduction, by algebraic laws of programming, to a normal form that mirrors the target machine; correctness follows from the laws | R, L, K | [K], not read |
| Cousot and Cousot, POPL 1977 | static analyses as fixpoints over lattices | K, C | [S] |
| Eel (Tyagi, Lynch and Demaine), 2016 | energy annotation: bit creation and destruction are costly, reversible operations free | E | [S] |
| UNCOL (1958); MLIR (2020) | one universal IR, which failed; many levels without a semantics | R | [S] |
| egg and e-graphs | canonical sets, cost-ranked extraction and sound rewrites in one structure | R, C, K | [S] |

Hoare, He and Sampaio (1993) is the closest prior work to calling
compilation governed by laws, and with Ershov (1977) it must be read and
cited before any claim of priority or novelty. Both are unverified here.
The report's reading list, in priority order, is: Hoare, He and Sampaio
(1993); Ershov (1977); the theorem statement of the CompCert manual; arXiv
2610.33193 (PP-complete or PP-hard); arXiv 2609.11356; and the Zig release
notes.

What may be new is the combination: one statement binding a physical bound
and an information-theoretic bound to checked, reuse-compounding lowering,
with energy as a ranked quantity. The framing this charter adopts is a
normative framework, or a normal form in the Hoare–He–Sampaio sense, with
two bounds and a measured distance from them, not a descriptive law that
every compiler already obeys. Calling it a law is earned only by the
hypotheses that survive their experiments.

## 8. Rules of work

- A hypothesis, its metric and its falsifier are registered in
  [HYPOTHESES.md](HYPOTHESES.md) before the measurement that tests it, and
  a change after registration is a dated row in its table of changes.
- Results are written by the experiment's harness and write-ups are
  generated from them by a script; no figure is typed.
- Every experiment states its declared inputs, and records the machine and
  toolchain versions with its results.
- Statements about outside work keep their evidence marks; anything
  unverified says so.
- RCE Compile does not depend on RCE's checker, Form or specification.
