# Core properties that close the replacement-matrix GAPs

Status: derivation, branch `rce/gaps`. Owner: David Charlot. The matrix
(`rce-replace/REPLACEMENT-MATRIX.md`) is an audit. It does not drive the design. Every feature
below traces to one core property of the Form or core language, and never to one language's
issue. Each property is derived from the stated principles: **S** soundness, **LP** least
privilege, **D** determinism, **M** one meaning per construct, **E** energy observability.

## 1. Grouping the 41 failure-class GAPs and 11 domain GAPs by the missing property

| Missing property of the core | GAP rows (class ids / domains) |
|---|---|
| A. A value's interpretation context is part of its type | SQLI (Python, C, Go, Java, SQL), XSS (Python, Java, JS); domains: web/HTTP output, browser DOM |
| B. Authority is a value | AUTHZ (11 rows); domains: kernels/MMIO (device authority), network servers (listen authority) |
| C. Secrecy is part of a value's type | SECRET (C, Java) |
| D. Bounds and predicates are declared, decided at entry, carried after | INPUTV (8 rows), DEPTYPE (Haskell, Lean); domain: proof-carrying programs |
| E. Foreign code runs only inside an isolated boundary of its own | FFI (Python, C, Rust, Zig, Mojo, Haskell, OCaml) |
| F. A failure of the environment is an event | PARTITION; domains: real network distribution, network servers, web services |
| G. Placement and target have no effect on meaning, checked per run | GPUREAL (Mojo, CUDA), EXTRACT; domains: GPU codegen, PTX/SPIR-V, ML accelerator lowering |
| (owned by rce/selfhost) | domain: compilers/self-hosting (mutual recursion, input as events) |
| (expressible from A+D+existing D.5) | domain: declarative queries/joins/indexes |

A, B and C are one shape: a type index that the value carries, plus a closed set of named,
total, logged functions that move a value between indices. A is the index *context*, B the
index *authority* (as an unforgeable value), C the index *label*. The minimal core addition is
therefore **one** mechanism, indexed types with explicit crossings (P1). D, E, F and G are
independent of it and of each other, which gives five core properties.

## 2. The five core properties

### P1. Indexed values: context, authority and label live in the type; crossing is explicit
- Derivation. **M**: a text that is HTML in one place and SQL in another has two meanings, so
  the type must name the one meaning. **LP**: an effect needs authority, and authority that is
  not a value is ambient. **S**: an index the checker enforces cannot be forgotten at run time.
  **D, E**: each crossing is a named, total, costed function, so it is visible in the trace and in C.6.
- Rules.
  - `text<c>` for a closed set of contexts c ∈ {plain, html, html_attr, url, sql, json, shell-less}.
    Literals take any context: the program text is trusted, and its author wrote it. Event payload
    text is `text<plain>`. The only map plain→c is `c.encode(t)`, which is total and escapes. `+`
    and `join` need equal contexts. A sink that is context-typed takes `text<c>` only. Nothing turns
    `text<c>` into `text<plain>` except `.raw()`, and `.raw()` goes only to plain sinks.
    W-CTX-1. Result: SQLI and XSS are **EXCLUDED**, since no well-formed program puts untrusted
    text into a sink of another context without encoding it.
  - `cap A` for an authority A declared in the module. A cap is linear/affine (5.12 machinery), not
    Copy, and cannot be built: it is minted only by an `authenticate` rule, i.e. a step from an
    event that carries a credential check of the module's own declaration, and it is logged. Every
    privileged handler, step and effect names the cap it needs (W-CAP-1). A call without the cap is
    a type error. Result: missing authorization (CWE-306/862, unauthorized handler) is **EXCLUDED**.
    Incorrect policy (863/269) becomes a declared policy, checked as written. That is the residue of
    the specification, the same as for any verified system.
  - `secret τ` is a label lattice {public ⊑ secret}. Secret values come only from event fields
    declared `secret`. A secret literal requires change (W-SEC-1: hard-coded credentials). `emit` and
    sends need public. `declassify(x, reason)` is a named effect, allowed only in contexts the
    effect table permits, and logged as `Declassify(reason, hash(x))`. Implicit flows: a branch on a
    secret raises the pc label (W-SEC-2). Result: SECRET is **EXCLUDED** (no literal credential, no
    unlabelled exposure). Lean-first: `RCE/Label.lean` theorem `noninterference` (states that agree
    on public inputs give equal public outputs) before the checker change.
- Version. Under the rules, programs that 0.7 accepts can require change (string concat into emit is still fine,
  since emit is plain; but `secret` and `cap` are new keywords and could collide with identifiers).
  New keywords change meaning, so **0.8 with gate** (C.15 rule). In ≤0.7 modules everything stays
  as it is.

### P2. Declared predicates: refinements decided at entry, carried as facts
- Derivation. **S**: validation that the type does not record can be skipped. **D**: a
  predicate that is checked once at admission has one outcome: the state is kept or the step applies. **M**: the type
  `{x: u32 | x <= 100}` has one meaning everywhere it appears.
- Rules. `type Pct = u32 where 0 <= self and self <= 100`. Predicates come from a decidable
  fragment: linear integer (Presburger) comparisons, `len()` bounds, character-class membership,
  and conjunction/disjunction. Event payload fields with refinement types are checked when the event
  is admitted, and a failure is `Kept(Invalid(field, pred), payload)` with the state unchanged
  (T8 shape), i.e. **DEFINED**. Inside the program, assigning to a refined type needs a fact the
  checker derives from guards and arithmetic in the fragment (interval plus difference-bound
  reasoning, with no SMT in the trusted base). Otherwise it needs the explicit `refine(e)` form,
  which returns `Pct?`. W-REF-1.
- Results: INPUTV is **DEFINED**. DEPTYPE is **DEFINED** for the decidable fragment: obligations are
  discharged, or the program requires change; never silent. Full dependent types stay out (**S**/**D**: an undecidable
  check would make acceptance non-deterministic in time and energy), and that is recorded as an
  exclusion by derivation.
- Lean-first: `RCE/Refine.lean`, `refine_admit_sound` (every admitted state satisfies the
  predicates of its fields) and `refine_preserved` (δ preserves them).
- Version: a pure addition (a new `where` clause on `type`, which was not accepted before), so it
  is an **extension**, all versions.

### P3. Foreign code is a component with its own memory
- Derivation. **LP**: foreign code must hold only the authority it was given. Today T6 *assumes*
  this. **S** requires it to be enforced. **D**: a fault in foreign code must be one defined outcome.
- Rules. `extern "wasm" component "x.wasm" { fn f(...) -> ... }` uses the WASM Component Model
  canonical ABI. Arguments are copied into the component's linear memory and results copied out
  (no shared borrow). The instance gets no imports beyond the ones the declaration lists (WASI
  denied by default). A trap or fuel exhaustion in the instance is `Foreign(f, args, Trap(kind))`
  and is logged. Fuel is metered (**E**). `extern "C"` stays legal in ≤0.7. In 0.8 it requires change
  unless the module declares `trust extern "C"`, which is a named, auditable exception.
- Result: FFI is **DEFINED** (deterministic trap, logged) for `extern "wasm"`. The T6 assumption
  becomes a theorem about the canonical ABI copy (`RCE/Foreign.lean`, `foreign_isolated`).
- Version: requiring change for a bare `extern "C"` changes acceptance, so **0.8 gate**. `extern "wasm"` is an
  extension.

### P4. Environment failure is an event
- Derivation. **D**: the Form is a function of its log, so the network's behaviour must be in the
  log. **M**: a lost message and a slow message are different events, not one silent behaviour.
  **LP**: a network endpoint is authority, so it is a `cap Net(addr)` (P1).
- Rules. A remote node is reached only by a send on a `remote chan` that has a declared timeout.
  Each send ends as `Delivered`, `TimedOut` or `Unreachable(node)`. The outcome is an input event of
  the sender, logged like a foreign result (T21 shape). A partition is a sequence of such events,
  and replicated state merges only through a declared merge function that must be commutative,
  associative and idempotent (checked by the existing trait-law machinery, or the program requires change). Inbound
  network traffic (servers, HTTP) is events of E, and replies are sends that need the cap.
- Result: PARTITION is **DEFINED**. Network servers and web services are expressible with no
  ambient I/O. The host adapter maps sockets to events.
- Version: `remote` is a new keyword, so **0.8**.

### P5. Target independence is checked per run, not assumed
- Derivation. **M**: decision 38 already says no build mode changes meaning, and a target is a
  build mode. **S**: an emitter without a proof is trusted unless every run is checked. **E**: the
  accelerator run reports energy, so it has to be real.
- Rules. Every emitter (Rust, Python, JS, WASM, and the new PTX emitter for device regions 5.13 and
  C.7 kernels) produces a trace in the 10.9 format. `rce run --target T --validate` compares it
  with the reference trace and logs the first divergence as `TargetMismatch(T, step)`. That outcome
  is defined: the target's result is not used and the reference result stands (translation validation). Device
  regions lower to PTX only for the fixed-size pure fragment that W-DEV-1 already enforces, and
  float reductions use the fixed T18 tree, so results match bit for bit.
- Result: GPUREAL and EXTRACT are **DEFINED**. GPU, PTX and ML-lowering domains depend on the PTX
  emitter. Tensors/autodiff depend on P5 plus a library extension (a closed tensor type with
  shapes as P2 refinements).
- Version: tooling only. No program changes meaning, so an **extension**.

## 3. Domain residue
- Kernels/MMIO: P1 `cap Device(reg)` and P4 interrupts as events. Raw memory stays EXCLUDED by
  derivation (**S**). The domain becomes "drivers over typed registers", not raw MMIO.
- Self-hosting: owned by `rce/selfhost` (mutual recursion, input as events). Not duplicated here.
- Declarative queries: D.5 joins and scans plus P1 `text<sql>` mean no SQL string is ever built
  from data. Queries are comprehensions over relations. Indexes are an engine placement (P5:
  meaning-free).

## 4. Implementation order (dependencies)
1. P2 refinements: an extension, no version gate, and the smallest checker change. Lean first.
2. P1 indexed values (context, then cap, then label), in 0.8 behind the gate. Lean noninterference first.
3. P3 wasm components (needs the wasm runtime already used by wasm-test).
4. P4 network events (needs P1 cap).
5. P5 trace validation, then the PTX emitter.
After each one, re-score the matrix rows of its group, run `python3 matrix.py` and commit.

## 5. As implemented (and where the draft above was refined by derivation)
- P2: commit aa0abd3ee. 5.1 refined types, W-REF-1, T35 (`RCE/Refine.lean`). Instead of
  inferring facts from guards, a refinement is checked at the boundary where integer ranges are
  already checked (`fit`). One meaning per construct: a refinement *is* a range check, generalised.
- P1: version 0.8, 5.16, T36 (`RCE/Label.lean`). Three refinements of section 2:
  - No new keywords. `Text[C]`, `Secret[τ]`, `Cap[A]` and `Guarded[τ, A]` are type names, so the
    grammar, LALR(1) and tree-sitter are unchanged. It is still a 0.8 gate, because those names are
    ordinary in 0.7 (C.16).
  - No `declassify`. Least privilege: the only release is the bit from `==` between two secrets.
    A release with a free-text reason would be ambient declassification.
  - The capability is minted at admission, by the host's attestation, not by an in-language
    `authenticate` rule. Authentication is an input, like any event, and the event is logged. In
    the language, authority is a value that cannot be forged or stored (W-CAP-1), and `Guarded`
    writes need it (W-CAP-2).
- P3, P4 and P5 are held by the owner until the host-runtime survey (/workspace/rce-host-survey/).
