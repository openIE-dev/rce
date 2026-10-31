# Proposal: compliance with Form

Status: open. Owner: David Charlot. Derived from 6.1 (the Form of a program), 6.2 to 6.5 (evaluation, the step function δ and its outcomes), 6.7 (bounds and fuel), 6.9 (examples and checks), section 7 (well-formedness), 10.1 to 10.7 (the `rce` tools and the language server), the theorems T2, T3, T4, T7, T8, T35 and T37, Appendix E (the host contract, with pull request #31), F.2 and F.12, CONTRIBUTING rules 1, 2, 3 and 5, and four proposals: `one-library-every-fabric.md` (the Form as the shape of all code), `one-emitter.md` (every language as one table), `one-commit-change.md` and `intent-holes.md`. Nothing in this proposal has been built. No normative chapter changes with it.

Sources, all read only:

- `openIE-dev/rce` at `origin/main` d2b28e76. Every count and line number is from that commit unless a line says otherwise. A location written `NN:line` is a line of `spec/book/NN-*.md`.
- Pull request (PR) #31, branch `claude/host-robustness-http-events`, read at 96f4538f, 12b9fb92, 44bb0afc and a374226b. It is not merged. A location written `PR16:line` is a line of chapter 16 at a374226b.
- The paper on the Reality Code Engine (RCE), `research-openie-web/src/content/papers/rce.md` at 85bc444, and the three Cleanroom decks.
- Three read-only maps, dated 27 October 2026, of every line that matches the search pattern below. They sort each use into the three classes of 3.6. A figure taken from the maps and not measured again for this proposal says so.
- The reference checker built at d91a9fc8, run on the fixtures of `spec/examples` at d2b28e76 to obtain the rule counts of 11.2.

The search pattern is `refus|reject|not accepted|is an error|turned away`, matched without regard to case. One *hit* is one text line that matches it, as `git grep -I -i -E -c` counts.

In plain words:

1. The Form P = (S, E, δ, D, λ) is the shape of all code. A program in any language, of any time, on any compute fabric has a state it keeps, events it answers, a step, a log and outputs, whether or not its text names them.
2. δ is total (T2): every state and every input has an outcome. That outcome is of exactly one of two kinds. It is *applied* when the selected steps compute the next state. It is *kept* when a step's contract holds the state as it was: λ outputs the reason, and D records the input with it. The text today calls the second kind "refused". That word names an act of turning an input away, which the Form never performs. The input is in D, hashed and chained, and the fold continues from the kept state.
3. Nothing offered to RCE is outside the Form. A text, construct, language, paradigm or fabric that the Form does not yet hold is brought into compliance with it: fitted, lifted, given its named form, bounded, held as a hole, or held at a reserved place until the Form grows to hold it through the gate of CONTRIBUTING rule 1. Every tool answers with the thing's place in the Form, or with the path to that place.
4. Some records are not the record the Form derives: a forged hash, a log that replay re-derives differently, a stale pin, a stated example that the Form does not give. Such a record is reported at the first position where it differs and is kept exactly as found. That is the Form complying with its own record, and the wording says so.
5. The language gains no construct. One keyword is renamed, and the reserved list loses a word. On the host, PR #31's `host.Kept` entry (named `host.Unfit` while PR #31 was open) also holds offered inputs, HC-21's `host.Refused` is renamed `host.StepKept`, and `host.MainInput` gains one field. The tools gain one mode, `rce fmt fit`. The hash of every δ entry of D is unchanged.
6. Today the words fill 6,254 text lines in 950 files of rce main, with 401 more on PR #31 and further lines in the paper, the site and the decks. Section 6 gives the new word for every use, section 8 every behaviour that changes, and section 11 the migration.

## 1. Definition

**Compliance with Form** is the relation between the Form P = (S, E, δ, D, λ) of 6.1 and everything offered to it: every input, program text, construct, language, paradigm, compute fabric and record. It holds when the thing has its place in P. An input has its place when δ gives it an outcome, *applied* or *kept*, and D records the input with that outcome; δ is total (T2), so every input already has one. A text has its place when it has a Form, either as written or once the checker's named path is taken: a fit, a lift, a named form, a bound, a hole, or growth of the Form through the gate of CONTRIBUTING rule 1. A language has its place when its programs enter Forms by a lift and leave them by a table (`one-emitter.md`). A paradigm has its place when a named form built from existing constructs holds it, together with the theorems that form carries. A fabric has its place when its native semantics is a named form whose results D records. A record (D, a pin, a signature, a stated example) has its place when it is the record P derives. Otherwise the first position at which it differs is reported, and the record is kept as found. Nothing offered is outside the relation. Whatever the Form does not yet hold is fitted, lifted, given its named form, bounded, held as a hole, or held at a reserved place until the Form grows to hold it.

## 2. Terms

**Notation.**

- *Form*: the machine P = (S, E, δ, D, λ) of 6.1. S is the set of states with initial state s₀, E the set of events, δ : S × E → S × O\* the step function, O\* the sequences of outputs, D the log and λ the output map.
- *δ₁*: the state component of δ.
- *ρ*: a reason (4.3). *κ(ρ)*: its cause.
- *φ(t)*: the fitted text of a text t (Definition 5).

**Label families.**

- T-n: the theorems of section 8 of the specification.
- W-…: the well-formedness rules of section 7 and of Appendices C and F.
- HC-n, HS-n and HD-n: the host-contract, host-storage and delivery-by-declaration rules of Appendix E. HD-n are on PR #31 (named HF-n while it was open).
- CC-n, XF-n and ONE-n: the rules of `one-commit-change.md`, `one-library-every-fabric.md` and `one-emitter.md`.
- COMP-n: the rules of this proposal (section 7).
- S1, E1, N2 to N9, F4, F9, O4, P10, Q16, Q17, Q28 and Q33: labelled items of the specification text.

**Abbreviations.** Each is expanded here, before its first use.

| Abbreviation | Expansion |
|---|---|
| ABI | application binary interface |
| ACP | Agent Client Protocol |
| AEAD | authenticated encryption with associated data |
| ANF | A-normal form (T34) |
| ASM | abstract state machine |
| BOM | byte order mark |
| CALM | consistency as logical monotonicity |
| CBOR | Concise Binary Object Representation |
| CI | continuous integration |
| CPU, GPU | central and graphics processing unit |
| CR, LF | carriage return, line feed |
| CRDT | conflict-free replicated data type |
| FIPS | Federal Information Processing Standard |
| FPGA | field-programmable gate array |
| HPKE | hybrid public key encryption |
| HTTP | Hypertext Transfer Protocol |
| IDE | integrated development environment |
| IDNA | internationalized domain names in applications |
| IETF | Internet Engineering Task Force |
| IFC | information-flow control |
| IR | intermediate representation (the Form IR of `rce emit form`) |
| JEPA | joint-embedding predictive architecture |
| JSON | JavaScript Object Notation; canonical JSON is that of 6.1.1 |
| LALR(1) | look-ahead, left-to-right, rightmost-derivation parsing with one token of look-ahead |
| LAPACK | Linear Algebra PACKage; DGELS and DGEEV are two of its routines |
| LSP | Language Server Protocol |
| LVar | lattice variable |
| MCP | Model Context Protocol |
| ML-KEM | module-lattice-based key-encapsulation mechanism (FIPS 203) |
| POSIX | Portable Operating System Interface |
| RFC | Request for Comments of the IETF |
| SCXML | State Chart XML of the World Wide Web Consortium (W3C) |
| SHA-256 | Secure Hash Algorithm 2 with a 256-bit digest |
| SQL | Structured Query Language |
| UTF-8 | Unicode Transformation Format, 8-bit |
| WASM | WebAssembly |

## 3. The principle

### 3.1 Statement

**Principle of compliance.**

- RCE answers every input with an outcome of P.
- It answers every text with its Form, or with the path that gives the text one.
- It answers every language with a lift into Forms and a table out of them, every paradigm with a named form, and every fabric with a named form whose results D records.
- It answers every record with the verdict that it is the record P derives, or with the first position at which it differs.
- It answers nothing offered with exclusion. What P does not yet hold is fitted, lifted, given its named form, bounded, held as a hole, or held at a reserved place until the Form grows to hold it.

The Form is rigid at a commit: one text, one meaning. It grows across commits, in the text itself, when a program needs a form the text lacks (CONTRIBUTING rule 3). Growth is therefore itself a path, and no text lies outside every path.

### 3.2 Grounds

1. **A total transducer, not an acceptor.** The words "accept" and "reject" come from the theory of acceptors. A finite automaton reads a word and ends in an accepting or a rejecting state, and its language is the set of words it accepts [Rabin and Scott 1959]. The Form is of the other kind. δ maps every state and every input to a next state and an output sequence. That is the shape of a Mealy machine [Mealy 1955] whose state set may be infinite. 6.1 requires δ to be total and T2 proves it. A total function has a value at every argument [Turner 2004]. The case the text calls a refusal is one of those values: δ(s, e) = (s, [Refused(X, e)]). A reader who meets the word sees an absence of outcome where the Form has an outcome.
2. **Totality.** T2 states that δ(s, e) is defined for every s ∈ S and e ∈ E and that its evaluation ends. T4 states that an evaluation ends in a value or a defined signal (`trap`, `fail`, `ret`, `brk`, `cont`), with "no stuck states other than defined outcomes". 6.5 adds: "Nothing is thrown past a step; no event is dropped silently."
3. **The shape of all code.** Gurevich's sequential thesis characterises a sequential algorithm by three postulates [Gurevich 2000]:
   - *sequential time*: a set of states, a set of initial states and a one-step transformation;
   - *abstract state*: states are first-order structures, and the transformation commutes with isomorphisms;
   - *bounded exploration*: a fixed finite set of terms determines each step.

   Every algorithm that satisfies the postulates is simulated step for step by an ASM. Blass and Gurevich extended the result to parallel algorithms [Blass and Gurevich 2003], and Blass, Gurevich, Rosenzweig and Rossman to algorithms that interact with an environment within a step [Blass et al. 2007]. The paper states the sequential and parallel cases as its Proposition U1. The Form instantiates each postulate:
   - S, s₀ and δ are sequential time.
   - The positional form of 6.1, under which no hash depends on a name (T7), is the isomorphism invariance of abstract state.
   - The bound B_P of T2 is bounded exploration.

   6.4 item 2 already gives a step ASM semantics ("Form law 2"). Interaction within a step is placed in D: each effect λ requests has exactly one result event (HC-3), and each foreign answer is logged (HC-4, T32). λ is a Mealy output function. Within the postulates of these theses, what a program of any language computes step by step already has the Form's shape. Languages differ in their text, not in that shape.
4. **Computations without end.** A total language represents a computation that never ends as productive codata: a sequence of steps, each of which ends [Turner 2004]. General recursion is represented by a total function into a type of possibly delayed results [Capretta 2005]. The Form takes this representation. δ ends on every input (T2), and D has no bound on its length. Behaviour over sequences of inputs is the final-coalgebra semantics of 6.10 [Rutten 2000]. The paper's lift (section 3.6, Proposition U3) is stated for the terminating fragment of a source language, and membership in that fragment is undecidable. Compliance gives that boundary a path:
   - A source computation whose termination is not established is held under a bound (6.7). Where the source ends within the bound, the Form's outcome is the source's result. Where it does not, the outcome is kept with cause `fuel`.
   - A computation meant to run without end lifts to a step on events that the host drives.
5. **Replicated services.** The state machine approach represents a fault-tolerant service as a deterministic state machine applied to an agreed sequence of inputs [Schneider 1990]. That is the pair (δ, D). The admitter and term rules of Appendix E (HC-1, T41) give D its agreed order.
6. **Determinism kept in D.** The world's nondeterminism is recorded in D as input: arrival order (T40), foreign answers (T32), scheduler choices (T21), adapter results (T37) and fabric results (XF-5). The meaning of a run is therefore the fold of its log (T3), as in record-and-replay systems [Dunlap et al. 2002]. Determinism needs no restriction on what a program, a language or a fabric does. It needs every contribution of the world to be an entry of D. A fabric's native semantics is held as a named form, and its results are entries of D.
7. **Languages.** Landin described a family of languages as one abstract core with many surface syntaxes [Landin 1966]. `one-emitter.md` holds every language as a table from the one core outward. A lift (D.8, `spec/lift/`) is the same relation inward. The criterion for when a construct of one language is held by existing constructs of another is macro-expressibility: a local translation that keeps the construct's meaning adds no expressive power [Felleisen 1991].
8. **Growth through the gate.** The Form is rigid at a commit and grows across commits (CONTRIBUTING rule 3). A construct the Form does not yet hold enters by rule 1, after rule 2 has shown that no composition of existing constructs holds the programs that need it. Until then its place is reserved (W-RES-1, F.12), or it is held as a hole (`intent-holes.md`).
9. **Claims per form.** When a program is held in a named form whose guarantees are weaker than those of the canonical form, each theorem is claimed only for the forms that have it. Gradual typing does the same: it claims type soundness for the statically typed part and attributes run-time failures to boundaries [Siek and Taha 2006; Wadler and Findler 2009]. Holding one program in a named form never weakens a theorem claimed for another program.

### 3.3 Compliance is not tolerance

The robustness principle of early Internet protocols ("be liberal in what you accept") led implementations to read one input in different ways. RFC 9413 recommends active maintenance of a protocol over tolerance [Thomson and Schinazi 2023]. Language-theoretic security asks that an input be fully recognised before it is processed [Sassaman et al. 2011].

Compliance with Form meets both:

- An input that fits no declared event is recognised as such and recorded with its bytes and the reason. Its effect on S is the identity. Nothing about it is guessed.
- A path is applied by a tool only when the text has exactly one reading under it. Otherwise every reading is reported, none is applied, and the position stays open until the author writes one.
- A tool never infers a declaration that belongs to someone else: an export of another module, a release of a label, a capability.

### 3.4 Integrity is the Form complying with its own record

D is a component of P. An entry that is not the record δ derives (a forged hash, an entry replay re-derives differently, a stored value that differs from its derivation) cannot be fitted into D without D ceasing to be the record, and then no outcome in D could be trusted. A verdict that keeps such a record exactly as found is therefore not an exception to compliance. It is the Form complying with itself.

The same holds for three other kinds of record:

- a stated claim: an `example`, a `check` or a proposed law;
- a pin: a hash in a manifest;
- a signature.

No tool rewrites a claim, a pin or a signed byte to agree with the Form. It reports where they differ and what each side holds (section 9).

### 3.5 Words are part of the interface

Three kinds of reader meet this vocabulary:

- a programmer at a terminal or in an editor;
- an agent that reads diagnostics and traces as data and acts on them;
- a reader of the specification or the paper.

The literature on how readers use diagnostics bears on all three:

- Developers read compiler error messages, and reading them takes a substantial share of the time a repair takes [Barik et al. 2017].
- The vocabulary of a message determines whether novices act on it correctly [Marceau, Fisler and Krishnamurthi 2011]. The error-message literature reports this consistently [Becker et al. 2019].
- The choice of keyword measurably affects how accurately novices read a language [Stefik and Siebert 2013].
- A machine reader acts on a diagnostic when it carries an action. The Rust compiler marks the suggestions it can apply mechanically [rustc development guide]. The LSP carries code actions that apply an edit [LSP 3.17].

A word that names an outcome as a verdict ("refused") carries no action. A word that names the outcome and its path carries one.

### 3.6 Where the text and the tools stand

The maps sort every current use into three classes, and this proposal uses them throughout:

- **(i) Word only.** The behaviour is already a defined outcome. Only the word changes.
- **(ii) Behaviour.** A tool turns away a text or an input that the Form can hold. The behaviour changes so that the text or input takes its place.
- **(iii) Integrity.** A record, pin, signature or stated claim differs from what the Form derives. The record is kept as found, and the wording changes.

Third-party text (quotations, terms of art, corpora of other projects) stays as written.

| Scope | Hit lines | Source |
|---|---|---|
| rce main, d2b28e76 | 6,254 in 950 text files, plus 3 binary files | measured |
| of which generated | 3,199: `conformance/reports` 2,807, `spec/RCE-SPEC.md` 356, `tree-sitter/src` 33, `spec/book/INDEX.md` and `receipts.json` 2, `spec/data/spec-data.json` 1 | measured |
| of which maintained by hand | 3,055 | measured |
| PR #31 at a374226b, over its merge base 2e94fb13 | +401 (6,655 in 965 files) | measured |
| Whole repository and documents, by class | (i) 3,967 rows, (ii) 414, (iii) 137, third-party 99 | maps |
| Chapters 11 to 17, CONTRIBUTING, SURFACE and the proposals | (i) 128, (ii) 178, (iii) 25 lines; with PR #31 at 96f4538f, 155, 189 and 28 | maps |
| Chapters 00 to 10 and the core checker files | (i) 112, (ii) 75 plus 9 behaviours that use none of the words, (iii) 4, data names 3 | maps |

Where the behaviour is already a defined outcome, the word misdescribes the Form. A "refused" event is in D (6.1), λ outputs its reason, and its state is the pre-state (T8). Where the behaviour departs from the Form, the departures are these:

- A payload outside its refined or ranged type never reaches D (`host.rs:1099-1104`; `interp.rs:3236-3240`). 5.1 says such an event "is refused before any step runs … with the state unchanged", and 6.5 says that "no event is dropped silently".
- `rce run`, `rce trace` and example logs stop at the first kept event (`cli.rs:93, 148-151`; `interp.rs:3279`), although δ defines every later event.
- A late input to a live `fn main` is not logged (HC-6; `host.rs:1356-1360`).
- W-STEP-2 reports "δ would not be total" for an event that no step handles (`check.rs:1497`). The frame rule of 6.4 item 1 makes δ total on that event.
- `tests/nameless.rs:75-77` compares nothing. It tests `starts_with("\"outcome\":\"refused")` on both sides, but the trace writes `"outcome":{"refused":…}` (`interp.rs:3255`), so both sides are false for every entry.

### 3.7 What the principle leaves unchanged

- Every value δ computes, and the hash and `state_hash` of every δ entry of D (section 10).
- Every theorem's content. Wording changes are listed in section 13.
- The integrity of D. Replay, verification on open and pin checks remain checks. They are worded as verdicts on records, never as outcomes of inputs (section 9).
- Every boundary a module or an authority declares: a module's `pub` surface, a label, a capability (T36). The path leaves such a declaration to its owner.

## 4. The outcome kinds of the Form

### 4.1 Setting

Let P declare regions R₁ … Rₖ, and let s ∈ S and e ∈ E.

- J(s, e) ⊆ {1, …, k} is the set of regions that have a selected step for e in s (6.4 item 1). A step whose guard traps counts as selected, with the trap as its result (change S1).
- For i ∈ J, rᵢ is the result of evaluating the step's body (6.2) on the pre-state. The result includes the fit of every written field to its type: the range check, VALUE_LIMIT, and the predicate of a refined type (5.1).
- By T2 and T4, each rᵢ is exactly one of `ok(rᵢ′, oᵢ)` (a new region record and its emitted values), `fail(Xᵢ, wᵢ)` and `trap(mᵢ)`.
- Fuel exhaustion is `trap` with the cause `fuel` (E1).

### 4.2 Applied and kept

**Definition 1 (applied).** Input e is *applied* in s when rᵢ = ok(rᵢ′, oᵢ) for every i ∈ J. Then δ(s, e) = (s′, o), where:

- s′ᵢ = rᵢ′ for i ∈ J, and s′ᵢ = sᵢ otherwise (the frame rule of 6.4);
- o = o_{i₁} ++ … ++ o_{iⱼ}, in declaration order.

**Definition 2 (kept).** Input e is *kept* in s with reason ρ when some i ∈ J has a result other than `ok`. Let i\* be the least such index in declaration order. Then:

- ρ = X_{i\*} when r_{i\*} = fail(X_{i\*}, w), and ρ = trap(m_{i\*}) when r_{i\*} = trap(m_{i\*});
- δ(s, e) = (s, [Kept(ρ, e)]).

**Proposition 1 (dichotomy).** For every s and e, exactly one of the following holds: e is applied in s, or e is kept in s with exactly one reason.

*Proof.* J(s, e) is finite, and each rᵢ has exactly one of the three forms. If every rᵢ is `ok`, Definition 1 applies and Definition 2 does not. Otherwise the least index i\* is unique, so ρ is unique. ∎

In Lean, `delta` (`RCE/Form.lean:248-251`) is one `match` on `runRegions`, and the proposition becomes the lemma `outcome_dichotomy` (section 13).

**Remarks.**

1. *A frame-only input is applied.* When J(s, e) = ∅, e is applied with s′ = s and o = []. The frame rule is the identity update, so no third kind is needed. W-STEP-2 becomes a note (8.2).
2. *Applied with s′ = s is not kept.* A step may compute the pre-state. D tells the two cases apart by the derived `outcome`, while their `state_hash` values are equal. T8 is the implication "kept ⇒ s′ = s", and its converse does not hold.
3. *The contract.* A step's contract is everything that decides between Definitions 1 and 2: its pattern and guard; its `check` statements, `fail` and `try`; the edge rules of 6.3; the type, range and refinement of each written field; the contract clauses of F.6.2; and the fuel of 6.7. "The contract holds the state" is the exact reading of Definition 2, in the sense of design by contract [Meyer 1992].
4. *An algebraic reading.* The outcome of one step is the sum A + R of an update and a reason, as in the exception monad [Moggi 1991]. δ consumes the second summand at the step boundary and never propagates it.

### 4.3 Reasons and causes

A reason ρ is a declared error name X, or `trap(m)` with m a trap text. Each reason has one cause κ(ρ). The cause set is the one PR #31 already uses (`interp::Cause`), and its words are kept.

| Cause | Reasons | Defined in |
|---|---|---|
| `fail` | a declared error X of the module left the step | 6.5 |
| `trap` | overflow, division by zero, conversion out of range, non-finite float, index out of range | 6.3 N2 to N8 |
| `trap` | `refinement: value is outside R` | 5.1, T35 |
| `trap` | the contract record {kind: "contract", clause, function, index} | F.6.2, T46 |
| `trap` | the capability record {id, kind: "capability", reason: "revoked"} | F.5.1 |
| `trap` | `check failed`; `utf8: …`; `yield outside an iterator`; `loop bound`; `recursion bound`; `budget` | 6.9, 6.3, 5.14, 6.7, F.5.5 |
| `fuel` | the fuel of one input ran out | 6.7, E1, HC-7a |
| `size` | SIZE_LIMIT | 6.7 |
| `value` | VALUE_LIMIT | 6.3 N9 |
| `depth` | the evaluation depth | 6.7 |

The list of trap texts in 6.2 becomes complete (COMP-3). Today it omits seven texts the tools produce: `check failed`, `recursion bound`, the refinement text, `evaluation depth`, the contract records, `utf8: …` and `yield outside an iterator`. Fuel exhaustion during a run is written `fuel exhausted`. The text `comptime fuel exhausted` is kept for comptime evaluation only. Today every runtime writes it at run time too: `interp.rs:438`, `conc.rs:832-834`, `rt.rs:901, 922` and `rt.py:32, 1537`.

### 4.4 What D and λ record

For an applied input nothing changes. For a kept input:

- **D entry.** The hashed fields are {event, prev_hash, seq, state_hash}, where `state_hash` is the hash of the positional form of s, the state the step kept (T8). The derived `outcome` is `{"kept": ρ}`.
- **λ record.** λ outputs `Kept(ρ, payload)`, in canonical JSON `{"Kept":{"payload":P,"reason":ρ}}`. Its type is not declared by the module, so it keeps its name in the positional form (6.1), exactly as `Refused` does today.
- **Fold.** The fold law sₙ = foldl(δ₁, s₀, D) applies the identity at a kept entry, and the fold continues from s (COMP-5).
- **Example logs.** An example log holds every event of its example. The outcome an example states is the outcome of its last event (COMP-5).

### 4.5 Choice of words

The word for Definition 2 must name what T8 guarantees and pair with *applied*. Five alternatives are inexact:

| Candidate | Why it is inexact |
|---|---|
| *refused*, *rejected* | Both name an act of turning an input away. The Form does no such thing: the input is in D, hashed and chained, λ answers it, and the fold continues from the kept state. |
| *aborted* | In transaction processing, the effects of an aborted transaction are undone [Gray 1981], which matches T8. But *abort* also names the end of a process, and the Form does not end. |
| *stuttering* | A stuttering step leaves every variable unchanged and is invisible to the specification [Lamport 1994]. A kept step is visible, because λ outputs a record. |
| *failed* | It describes the evaluation of one body, not the outcome of the input. HC-3 already uses `failed` as the status of an adapter result. |
| *held* | It is used for the relation of the Form to a construct ("held by a named form") and for the host entry of 4.7. |

*Kept* names the observable T8 guarantees: the state is kept. It pairs with *applied*, the update is applied. It reads correctly in the syntax `example Dec -> kept: Underflow`.

The whole vocabulary follows six criteria:

1. **A word names what happens.** `kept` names δ₁(s, e) = s; `fit` names a path to the Form; `stopped` names a run that ended with a reason; `differs` names a record that is not the derivation.
2. **Words pair.** `applied` with `kept` for inputs, `ok` with `fit` for texts, `holds` with `differs` for claims and records.
3. **One word has one meaning (CC-5).** Three words used here are already in the text in the same sense: "fit" in HC-3 ("data that does not fit T"), in `fit_sound` (T35) and in E.4c on PR #31; "lift" in 10.5 and D.8; "hole" in `intent-holes.md`. One conflict exists. "Named form" is used at 12:485 and `interp.rs:2954` for the name-bearing, non-positional form of a value. That use becomes "name-bearing form", so "named form" has one meaning: an existing construct, or combination of them, that holds a construct with its stated claims.
4. **One spelling everywhere**: prose, syntax, data keys, event names, identifiers, test names and directories.
5. **Other people's text stays as written** (6.9).
6. **Prose says "the kept outcome" or "kept with X"**, never "the event is kept" on its own, because "kept" is also an ordinary word of the text ("the stored `event` and the readable `state` keep their names").

### 4.6 Outcomes of runs

An actor's handler is a step on a message, so its outcome is applied or kept. The run log records `Kept(ρ)`. Under a supervisor this is followed by `Restart(child, instance, reason, tick t, k)` (C.11).

A run of `fn main` (6.8, HC-6) ends in exactly one of three *run outcomes*. They partition the ends of a configuration sequence (T4 for configurations):

| Run outcome | When |
|---|---|
| `done` | The body returned and every task ended. |
| `stopped(ρ)` | The body ended with `fail` or `trap` ρ; or a supervisor exceeded its `restart bound`, and ρ names the supervisor and the child, and the log carries `Stopped(supervisor S)`; or the tick bound `max_ticks` was reached, with cause `fuel`. |
| `deadlock` | Every live task is blocked. |

A run, unlike a step, is not atomic: the entries a run has already appended stay in D. That is why a run ends `stopped` and not `kept`.

A replay that finds a record different from the one the Form derives is not a run outcome. It is a verdict on a record (4.10). Today `conc.rs:1957, 2284` and `rt.rs:1721` report it as `Outcome::Refused("replay diverged")`.

### 4.7 Inputs offered to a host

Every input offered to a host, or to `rce run` as EVENTS, has exactly one record, of one of three kinds:

1. **The program event it names**, with that event's δ outcome. A payload of the event's shape whose value lies outside a range or a refinement is this kind: the event, kept with the trap of the field it misfits (8.1, C1).
2. **A declared event bound to it.** `Input(data)` takes a line that names no event when the module declares it, as `rce host serve` does today (`cli.rs:1444-1446`). HC-21 binds `StepKept` and HC-25 binds `HttpMalformed` in the same way (E.8).
3. **`host.Kept` {event, why}.** This is PR #31's `host.Kept` {event, why} (named `host.Unfit` while PR #31 was open), with its use extended to every offered input that no declared event takes. `event` is the whole value as offered: {payload, type} when the input names a type, and {data} when it names none. `data` is the bytes as HC-10a delivers them, up to SIZE_LIMIT; above it, their SHA-256 digest and their length. `why` is the reason. δ maps this entry to the identity (T37). When the module declares `event Kept`, the admitter delivers it, matched by field name (E.4c), within the descent bound of HC-21.

The marks of E.2a and HC-10b record how an input came (`late`, `typed`). After a kept entry the host appends `host.StepKept` {cause, seq}, and, when the module declares `StepKept(seq: u64, cause: text)`, delivers that event (HC-21 with the names of section 6).

### 4.8 Texts and paths

**Definition 3 (in Form).** A text t is *in Form* at a commit when it meets every rule of section 7 and of the implemented parts of Appendices C and F. The checker then builds its Form P_t. `rce check` prints `ok`. A text in Form may carry *notes*, which name something the author may want to see and do not change the exit status.

**Definition 4 (path).** For a rule W that a text does not yet meet at a position p, a *path* is a named way by which p takes its place in the Form. Each rule names its path, of one of the kinds below:

| Kind | What it does | Text changes | Applied by | Until it is taken |
|---|---|---|---|---|
| `rewrite` | A change with exactly one result that keeps the meaning of every part already in Form: layout, a renaming (T7), a conversion whose rounding N4 fixes, `.copy()`, `let _ = e` | yes | `rce fmt fit`; an LSP code action | the text is `fit`; tools evaluate φ(t) |
| `complete` | Writes a missing part whose outcome is defined (a trap arm, a release, a trap-bodied method) and reports what it wrote as a hole | yes | `rce fmt fit` | as `rewrite` |
| `lift` | Maps a construct of another language or paradigm to its named form, by a row of the lift table (COMP-10) | yes | `rce fmt fit`; a language front end | as `rewrite` |
| `bound` | Writes a bound that makes a cycle total; reaching it is the kept outcome with cause `fuel` (6.7) | yes | `rce fmt fit` (with LOOP_LIMIT) | as `rewrite` |
| `choice` | Two or more readings, or a statement only the author can make: a declaration, an export, a capability, a release. Every reading is listed and none is applied. The position may be held as a hole (`intent-holes.md`). | by the author | the author | the position is open |
| `gate` | The Form grows to hold the construct through CONTRIBUTING rule 1. The report names the F.12 row or the proposal. | after the gate | the gate | the position is open |
| `form` | Holds the construct in a named form whose claims are stated (COMP-9) | no | — | a note; the text is in Form |
| `reserved` | The construct's place is specified (F.12), and it runs from its definition or is held unchecked (COMP-13) | no | — | a note; the text is in Form |

A `lift` is determined when the lift table has one row for the construct. When it has several, the path is a `choice` among the rows.

**Definition 5 (fitted text).** The *fitted text* φ(t) applies every path of kind `rewrite`, `complete`, `lift` or `bound`, each at its position and in source order.

- φ is deterministic and idempotent: φ(φ(t)) = φ(t).
- Every tool builds the Form of φ(t) and reports each path it applied.
- `rce fmt fit` and the LSP code action write φ(t), so the repository keeps one spelling (CONTRIBUTING rule 2), and `rce check` reports `fit` until they have.
- The CC-1 program hash names φ(t).
- A text with an open `choice` or `gate` position is reported with every reading. The tools that evaluate build its Form once each open position is closed. Whether an open position may meanwhile be evaluated as a hole is open question 2.

The report has one line per file, as 10.2 has today, with `;` between positions, and one further line per note:

```
ok FILE
note FILE: line N: RULE message
fit FILE: line N: RULE message (fit KIND: TEXT)[; line M: RULE message (fit KIND: TEXT) ...]
```

TEXT names what the path does and contains no `;`. Mechanical, rule-named rewrites of program text are established practice: semantic patches for collateral evolutions [Padioleau et al. 2008], `gofix` for interface changes [Cox 2011], and minimum-distance error-correcting parsing, which maps every string to a nearest member of a context-free language and reports the edits [Aho and Peterson 1972].

### 4.9 Claims: holds and differs

An `example`, a `check`, a proposed law or a stated derivation *holds* when the Form gives what it states. Otherwise it *differs*: the Form gives o′ where the claim states o. The report gives both values. The text has its Form, and every tool evaluates it. Its status is outstanding (exit status 1) until claim and Form agree, because pins and emitted code state that the examples hold (6.9, CC-4). The report offers two readings: change the step, or state o′. No tool writes either.

### 4.10 Records: holds and differs at k

**Definition 6 (holds, differs).** A record *holds* when it equals what P derives. The record may be a log D, a stored state, stamp or tag, a pinned module, a signed body, or a text and its layout. Otherwise it *differs at* a first position k. The verdict names:

- what P derives at k and what the record holds there;
- that the record is kept as found;
- that nothing past k is derived from it.

```
replay differs at seq 41: the Form derives state_hash 9a3e…, D holds 3f07…; D is kept as found; entries 0..40 are the record
example differs at line 3: the example states kept: Underflow; the Form gives applied (Counter { n = 1 })
```

A verdict on a record is never an outcome of an input. D is checked, not trusted (T38, HS-4).

### 4.11 Answers to effects and to library calls

An effect that λ requests has one result event: delivered, `failed`, timed out or unreachable (HC-3). A library function returns its declared result, for example `BadInt` or `BadTag`. These are values the program receives, not outcomes of δ.

### 4.12 Outcomes of the gate

- Rewrite proposals of the engine are installed (a law, with its certificate), found not to be laws (logged with the counterexample), or deferred.
- Language proposals in `LEDGER.md` are `accepted` into the normative text, `composed` (the programs that need them are held by existing constructs, named in the row, and no primitive is added), or `open`.

### 4.13 Summary

| Layer | Kinds | Where defined |
|---|---|---|
| Input to δ | applied; kept(ρ), with cause κ(ρ) | 4.2, 4.3 |
| Run | done; stopped(ρ); deadlock | 4.6 |
| Offered input | the program event; a bound declared event; `host.Kept` | 4.7 |
| Text | `ok` (in Form, with notes); `fit`, with paths `rewrite`, `complete`, `lift`, `bound`, `choice` or `gate`; notes of kind `form` or `reserved` | 4.8 |
| Claim | holds; differs | 4.9 |
| Record | holds; differs at k | 4.10 |
| Effect or library call | the result event; the declared result | 4.11 |
| Gate | installed, not a law, deferred; accepted, composed, open | 4.12 |

## 5. What each reader meets

**A programmer running a program:**

```
$ rce run 01_counter.rce "Inc, Inc, Dec, Dec, Dec, Inc"
0 (initial) Counter { n = 0 }
1 Inc Counter { n = 1 }
2 Inc Counter { n = 2 }
3 Dec Counter { n = 1 }
4 Dec Counter { n = 0 }
5 Dec kept: Underflow Counter { n = 0 }
6 Inc Counter { n = 1 }
$ echo $?
0
```

Today line 5 reads `5 Dec refused: Underflow`, there is no line 6, and the exit status is 1 (10.3).

**A programmer checking a file:**

```
$ rce check door.rce
fit door.rce: line 4: W-LEX-1 tab in indentation (fit rewrite: indent with 4 spaces); line 9: W-NAME-1 `Opne` is not declared (fit choice: `Open`, declared at line 2, or declare `Opne`)
$ rce fmt fit door.rce
$ rce check door.rce
fit door.rce: line 9: W-NAME-1 `Opne` is not declared (fit choice: `Open`, declared at line 2, or declare `Opne`)
```

Today this prints `REFUSED door.rce: line 4: W-LEX-1 …; line 9: W-NAME-1 …`.

**A programmer writing examples:**

```
intent "a counter that never goes negative"
    example Inc, Inc, Dec      -> 1
    example Dec                -> kept: Underflow
    example Dec, Inc           -> 1
```

The third example holds after this change, because the fold continues from the kept state. Today it is reported with W-INT-1.

**An agent** receives an LSP diagnostic that carries its path, with a `quickfix` code action marked preferred:

```json
{"code":"W-LEX-1","data":{"fit":{"edits":[{"newText":"    ","range":{"end":{"character":1,"line":3},"start":{"character":0,"line":3}}}],"kind":"rewrite","text":"indent with 4 spaces"}},"message":"tab in indentation","range":{"end":{"character":1,"line":3},"start":{"character":0,"line":3}},"severity":2,"source":"rce"}
```

and a trace step, here the step of `spec/traces/01_counter.json` for `example Dec -> kept: Underflow`:

```json
{"event":{"payload":{},"type":"Dec"},"hash":"07ad897ce740329566c1740b942501d6ccf683d3b86e9de3a42c3b06c238e47c","hashed":{"event":{"payload":[],"type":1},"state":[[0]]},"outcome":{"kept":"Underflow"},"prev_hash":"0000000000000000000000000000000000000000000000000000000000000000","seq":0,"state_hash":"db407f11d7ede59abaab0e98e097ff2dae10a048207b801745d7199ef19c2387"}
```

Only the word inside `outcome` differs from the fixture today. Both hashes are today's, because the outcome is derived and not hashed (section 10).

**A reader of the specification and the paper** reads:

- "A counter whose `Dec` at zero keeps n at 0" (00:25, today "a counter that refuses to go negative").
- "**T8 Atomicity.** If e is kept in s, then δ₁(s, e) = s."
- "A text that does not yet meet a rule of section 7 is reported with the rule and its path; it has its Form once the path is taken."

## 6. Vocabulary

Every current term is mapped to its compliance term. A term in backticks is spelled exactly as written. The column "Hashed" says whether the term enters a hash. Section 10 says how each hash is kept or re-pinned.

### 6.1 Inputs to δ, D and λ

| Current | Compliance | Where | Hashed |
|---|---|---|---|
| refused (outcome); "δ refuses e"; "the event is refused" | kept; "e has the kept outcome"; "the step keeps the state" | 6.1, 6.4, 6.5, T8, chapters 00 to 17, paper | no |
| refusal (noun) | the kept outcome | same | no |
| "refused with X"; `refused(X)` | "kept with reason X"; `kept(X)` | same | no |
| "X is the refusal of the first refused region" (6.4 item 4) | "ρ is the reason of the first region, in declaration order, whose step keeps the state" | 08:28-29 | no |
| "the step is **refused**" (6.4 item 3) | "the step keeps the state" | 08:26 | no |
| "Errors, traps and refusals" (6.5 title) | "Errors, traps and kept outcomes" | 08:41 | no |
| "a trap always refuses the event" (6.2) | "a trap gives the event the kept outcome" | 07:239 | no |
| "it refuses the event exactly as a trap does" (E1) | "it gives the event the kept outcome with cause `fuel`, as a trap does" | 07:244 | no |
| "integer overflow is an error" | "overflow is a trap: the kept outcome, never a wrap" | 05:168 | no |
| "is refused before any step runs" (refined payload, 5.1) | "has the kept outcome with `trap(\"refinement: value is outside R\")` before any step runs, and is in D" | 05:86 | no |
| "`some(none)` is refused" (6.1 table) | "`τ??` has its form as a kind with three variants (W-TY-2)" | 07:84 | no |
| "A refused event (T8) … ends the example log" (6.1) | "A kept event is recorded with its unchanged state (T8), and the log continues from it" | 07:59-60 | no |
| `Refused(X, payload)` (λ record) | `Kept(ρ, payload)` | 07:29, `interp.rs:3244-3262`, emitted runtimes | no (derived) |
| `{"Refused":{"error":X,"payload":P}}` | `{"Kept":{"payload":P,"reason":ρ}}` | same | no |
| D `outcome` `{"refused": X}` | `{"kept": ρ}`; `"applied"` unchanged | 6.1, traces, `rce emit form` | no (derived, 6.1) |
| example syntax `-> refused: X`; `-> refused: trap("K")` | `-> kept: X`; `-> kept: trap("K")` | 03:34, 99 lines in 77 `.rce` files | yes: program hash (CC-1) and module hash (F.5.3) |
| reserved word `refused` | removed from the reserved list; `kept` is a contextual word (COMP-2) | 02:54, `parse.rs:13` | no |
| tree-sitter node `outcome_refused`, token `refused` | `outcome_kept`, `kept` | `tree-sitter/grammar.js:34-36` | no |
| `rce run` line `k EVENT refused: E` | `k EVENT kept: E STATE` | 10.3, `cli.rs:148` | no |
| "refusal channel" (F.4, W-IFC-4) | "outcome channel" | F.4.2, 11:145, 11:316 | no |
| "fixed refusal record" (F.6.2, T46) | "fixed contract record" | 17 | no |
| "refusal record" (Q28) | "kept record" | 11:373 | no |
| `Refused(Invalid(field,pred),payload)` | `Kept(Invalid(field,pred),payload)` | `CORE-PROPERTIES.md` | no |
| "a refusal is the rejection of an input" | "the kept outcome is the program's defined answer to an input whose step's contract holds the state; an emitted value is the answer of a step that applies" | 12:602-603, `LEDGER.md` row of 2026-10-24 | no |
| `rce dim` JSON key `refuses`; labels `refused:X`, `refused:err#i` | `keeps`; `kept:X`, `kept:err#i` | `dim.rs`, 10.11, book chapter 18 | registered rows: section 10 |
| caretaker `Verdict::Refused(record)`; `refusal_record` (W-CAP-5) | `Verdict::Kept(record)`; `capability_record` | `caretaker.rs:27-50` | no |
| "a counter that refuses to go negative" | "a counter whose `Dec` at zero keeps n at 0" | 00:25 | no |
| intent "a bank account that refuses overdrafts" | "a bank account that keeps its balance on an overdraft (Insufficient)" | `spec/examples/04_bank_account.rce`, book chapter 6, conformance copies | yes: module and program hashes |
| book chapter 6 "Errors and refusals", section "Refusing an event" | "Errors and kept outcomes", "Keeping the state" | `book/src/ch06`, `SUMMARY.md:10` | book footers |

### 6.2 Runs

| Current | Compliance | Where | Hashed |
|---|---|---|---|
| `refused: REASON` (`rce run` of `fn main`) | `stopped: REASON` | 10.3, `cli.rs` | no |
| `outcome Refused: REASON` (emitted program) | `outcome Stopped: REASON` | `rt.rs:2065` and the Python, JavaScript and Scala runtimes | no |
| `conc::Outcome::Refused(String)` | `Outcome::Stopped(String)` | `conc.rs:1592-1594` | no |
| run stop tuple `("refused", r)` | `("stopped", r)` | `rt.js`, `rt.py`, `rt.rs`, `rt.scala` | no |
| actor log `Refused(r)`; `host.Step.what` `"Refused(r)"` | `Kept(r)`; `"Kept(r)"` | C.13, `conc.rs:2262, 2370` | yes, inside `host.Step` |
| `host.MainEnd.outcome` `"refused: r"` | `"stopped: r"` | E.5, `host.rs:1301-1307` | yes |
| "the run ends `Refused`" (supervisor, C.11) | "the run ends `Stopped`" | 14 | no |
| `trap: tick limit` | reason `tick limit`, cause `fuel` | `conc.rs` | in `host.MainEnd` |
| "In `main`, a trap or unhandled failure ends the command with the refusal as its output" | "… ends the run with outcome stopped(ρ), and ρ is its last output" | 08:46-47 | no |
| `Outcome::Refused("replay diverged")`; `"trap: upgrade removed supervisor position"` as a run outcome | a record verdict `Differs { seq }`, reported beside the run outcome and never as one | `conc.rs:1957, 2284`; `rt.rs:1721`; `rt.js:1672-1673`; `rt.py:1764-1766` | no |
| `comptime fuel exhausted` at run time | `fuel exhausted`, cause `fuel`; the comptime text stays for W-CT-1 | 4.3 | where `host.MainEnd` carries it |

### 6.3 Host

| Current | Compliance | Where | Hashed |
|---|---|---|---|
| HC-21 "A refusal is an input" | "A kept outcome is an input" | E.4a (PR #31) | no |
| E.4a "Refusals and restarts" | "Kept outcomes and restarts" | PR16:465 | no |
| `host.Refused` {cause, seq} | `host.StepKept` {cause, seq} | E.5, PR16:800, `host.rs` on PR #31 | yes |
| declared event `Refused(seq: u64, cause: text)`; `host.Bind` {"event":"Refused","kind":"host.Refused"} | `StepKept(seq: u64, cause: text)`; {"event":"StepKept","kind":"host.StepKept"} | HC-21, HD-4 (PR #31) | event name no (positional); `host.Bind` payload yes |
| `host.Kept` {event, why} (PR #31, named `host.Unfit` while it was open) | `host.Kept` {event, why}, with its use extended to every offered input that no declared event takes (4.7) | E.4c, E.5, PR16:803 | yes |
| "input refused: …" (serve loop) | "input kept: …", and the input's entry in D | `cli.rs:1448` | no |
| "An input refused before it is logged" (HC-24) | "An input that names no declared event is kept (`host.Kept`) and logged" | PR16:606 | no |
| `WindowAck.rejected` | `WindowAck.held` | `host.rs:806, 1699` | no |
| "An input whose tick is not after the last logged step is refused before it is logged" (HC-6) | "… is admitted at the first tick after the last logged step and marked `late`" | 16:117-118 | no |
| HC-5 "refuses any effect without one" | "answers any effect without one with its `failed` result (HC-3)" | 16:73 | no |
| "a placement on a fabric this host cannot execute is refused before anything is logged" | "… runs on the CPU reference, which gives the same bits (HC-13), and `host.Placed` names `cpu`" | 16:463, `host.rs:1576-1585` | no |
| "A step that would change state is not a query and is refused" | "A query answers λ(s, e) on the current state and appends nothing; the change it would make applies only when the request is offered as an input" | `host.rs:1807-1809` | no |
| rcedb protocol word `REFUSED E` | `KEPT E` | book chapter 16:24, `rcedb/` | no |
| `Host::refused`, `refused_payload`, `Replayed.refusals`, `Owe::HostRefused`, `Owe::BoundRefused` | `Host::kept`, `kept_payload`, `Replayed.kept`, `Owe::HostKept`, `Owe::BoundKept` | `host.rs`, PR #31 | no |

### 6.4 Texts and the static report

| Current | Compliance | Where |
|---|---|---|
| `REFUSED FILE: line N: RULE message` | `fit FILE: line N: RULE message (fit KIND: TEXT)` (4.8) | 10.2, `cli.rs:14` |
| refused program; "refused at entry"; "refused by W-X" | "a text reported with W-X and its path"; "a position not yet in Form" | section 7, chapters 00, 01, 10 to 17 |
| accepted program | text in Form | throughout |
| "## 7. Well-formedness: what is refused at entry" | "## 7. Well-formedness: the rules a text meets to have its Form" | 09:1, `clauses.json` (`section`, 66 times) |
| "A program is refused if any rule fails" | "A text is in Form when it meets every rule; each rule it does not yet meet is reported with its path" | 09 |
| Conformance (a): "refuses every text that is not in the language …" | "reports every text not yet in the language of section 3, or not yet meeting a rule of section 7, naming the rule and its path" | 01:37 |
| "a program that an earlier commit accepted may be refused by this one" | "a program that an earlier commit held is carried to the new rule in the same commit (CONTRIBUTING rule 3)" | 00:79 |
| W-RES-1 "reserved, not implemented"; "refused by version" | "reserved (F.12): place specified, held for implementation" | 02, 09:15-17, 17:1044, `cli.rs:591` |
| `rce v09-status` ("the ones it refuses by version") | `rce reserved` | `cli.rs:591, 989` |
| W-SAFE-1 "refused with 'memory safety is by construction'" | "`Rc` has its form in RCE as …", per identifier (8.2) | 09:29-30 |
| W-STEP-2 "δ would not be total" | note: "event E changes no region: δ applies it with every region framed (6.4)" | `check.rs:1497` |
| "excluded", "non-goal", "the design refuses", "declined" (of a paradigm or construct) | "held as the named form N" (8.9) | F.2, 00:69-71, 11, 17, `LEDGER.md` |
| emitter "not supported by the emitter", status 2 | `pending FILE: target T does not yet print construct C (line N)`, status 1 (COMP-14) | `cli.rs:1004-1014` |
| "refused 12/12 syntax rejects, refused 64/64 ill-typed or inadmissible" | the summary line of 11.3 | `bin/spec_check.rs:378`, 00 |
| "bricks refused for term size" | "bricks that Form IR format 1 cannot yet hold for term size" | 15:272, `LEDGER.md`, paper 609 |
| LSP: "one diagnostic per refusal, with severity Error" | "one diagnostic per position; severity Error for `choice` and `gate`, Warning for `rewrite`, `complete`, `lift` and `bound` (each with a code action), Information for notes" | 12:397-398 |
| `fmt`: "the text is refused and left as it was" | `differs FILE: the layout change would change the tokens at line N; the text is kept as written` | 10.6, `cli.rs:235-237` |
| "fitting changes no meaning" in place of "refusing costs no expressiveness" | | 17:114 |
| "named form" in the sense of name-bearing | "name-bearing form" | 12:485, `interp.rs:2954` |

### 6.5 Fixtures, conformance and data formats

| Current | Compliance | Where | Hashed |
|---|---|---|---|
| `spec/examples/reject/`, `ill_typed/`, `inadmissible/` | `spec/examples/fit/syntax/`, `fit/types/`, `fit/admission/`; `spec/examples/claims/` for the fixtures whose claim differs; `spec/examples/notes/` for the two that become notes | `bin/spec_check.rs`, CONTRIBUTING, `claim-counts.py`, `corpus.py`, `t7-equiv.sh` | no |
| `spec/examples-0.9/refused/`; `spec/conc06/reject/`, `spec/conc06b/reject/` | `spec/examples-0.9/fit/`; `spec/conc06/fit/`, `spec/conc06b/fit/` | same | no |
| `book/listings/chNN/refused/` (8 directories); `programs/money/refused/` | `book/listings/chNN/fit/`; `programs/money/fit/` | `book/scripts/check-listings.sh:8-50`, `programs/scripts/check.sh` | no |
| conformance `# expect: accept`, `# expect: reject` | `# expect: ok`; `# expect: fit`, `# expect: differs` for claims, `# expect: ok` with `# note: RULE` for notes | `conformance/INTERPRETING.md`, `harness/` | no |
| conformance files `*-reject.rce`, and names containing "refus" | `*-fit.rce`; `kept` or `fit` in the name | 157 files named `*-reject.rce` and 9 others | no |
| conformance `# outcome: refused:X` | `# outcome: kept:X` | 5 tests | no |
| adapter verdicts `"accept"`, `"reject"`; results `{"refused": e}`; replicate field `"rejected"`; finding kind `refused-at-entry` | `"ok"`, `"fit"`, `"differs"`; `{"kept": ρ}`; `"held"`; `fit-at-entry` | `bin/conform.rs`, `bin/conform_lean.rs`, `wasm_playground.mjs` | no |
| trace format `rce-trace/2` | `rce-trace/3` | `cli.rs:477`, 7 of the 65 JSON files of `spec/traces` | no |
| catalog `lowering: {"refused": r}`; `refused_files` | `{"pending": r}`; `pending_files` | `brick.rs`, 12:540-542 | the catalog hash covers `lowering` |
| corpus `rce-corpus/1`: `verdict: "reject"`, `outcome: "refused"`, `refused_rows` | new rows under `rce-corpus/2`: `"fit"`, `"kept"`, `fit_rows`; published rows kept as published | `scripts/corpus.py` | yes (`source_sha256`) |
| verdicts of the intent derivation log, `refused:<…>` | `differs:<obligation\|example\|spectest>` | `intent-holes.md` | yes, once implemented |

### 6.6 Records and integrity

| Current | Compliance | Where |
|---|---|---|
| `REPLAY FAILED`; "replay refuses an entry" | `replay differs at seq k: …; D is kept as found` | `cli.rs:1158`, HC-6, `host.rs:1346, 1393-1408` |
| "A broken chain refuses to load" | "Open verifies the chain; the verified prefix is the record (T38)" | T38, HS-4, paper 352 |
| `take_role`: "a lower or equal term is refused" (HC-1, T41) | "term t is not above T; the admitter role stays with term T" | `host.rs:1773-1777` |
| `manifest-check` "refuses a stale manifest" | "reports each module whose text differs from its pin" | `LEDGER.md`, `stdlib/scripts/check.sh` |
| W-INT-1 "expected refused: X, but E was applied" | `example differs at line N: the example states kept: X; the Form gives applied` | `interp.rs:2600-2614` |
| "the gate refused 15 of 18 proposals"; "rejected candidate" | "the gate found 15 of 18 proposals not to be laws, each with a counterexample"; "a candidate with a counterexample, logged and not installed" | paper 372, 425, 454; decks |
| "the fold refuses the first entry whose hash does not match" | "the fold stops at the first entry that is not the record" | paper 404 |
| `fold_rejects_wrong_state`, "A fold that disagrees with the stored state is refused" | the test name is the engine's and is cited as data; "A fold that disagrees with the stored state is reported as not the record" | paper 440 |
| byte-integrity checks "reject" (tags, signatures, HTTP smuggling) | "does not verify: answers `BadTag` (or the library's declared error); nothing is derived from it" | `stdlib` |
| D.6 output `reject` | output `mismatch`: a contract mismatch is an output of an applied event, with its `error` row | 15:205, 15:210 |
| D.11 promotion status `refused` | `deferred`, with the counterexample or the staleness | 15:357 |
| "refusing configuration" (session types, C.13) | "stuck configuration" | 14:406, 14:728, `mpst_async.rs:143` |

### 6.7 Tools, process and prose

| Current | Compliance | Where |
|---|---|---|
| IDE receipt `Verdict::Refused { diagnostics }` ("`rce check` refused; the change was not applied") | `Verdict::Fit { diagnostics }` ("`rce check` reported paths; nothing was written") | `ide-receipt/src/lib.rs:84-85` |
| IDE receipt `Verdict::Rejected { reason }` ("rejected before the checker ran (bad path, policy)") | `Verdict::NotPerformed { reason }`, the reason naming the path or the policy | `ide-receipt/src/lib.rs:86-87` |
| action `proposal.reject`; button "reject N" | `proposal.decline`; "decline N" (a person's decision on a proposed change) | `rce-ide/src/vm.rs:206` |
| sandbox, jail, `rce-memory`, `rce-guest`: "refused: …" | "outside the declared capabilities: not performed (declare X to allow)" | `ide/crates/ide-sandbox`, `rce-mcp` |
| `acp-client` `refused_methods` | `methods_outside_grant` | `ide/crates/acp-client` |
| graph node field `refused: bool`; `EdgeKind::Refuses`; MCP query keyword `refused` | `kept: bool`; `EdgeKind::Keeps`; `kept` | `rce-graph3d`, `rce-mcp` |
| `mei-agent` `Concl::Reject`, `v_reject`, `set_reject`, "checker reject: …" | `Concl::Fit`, `v_fit`, `set_fit`, "checker fit: …" | `ide/crates/mei-agent` |
| playground CSS class `.refused`; "refused at the X stage" | `.fit`; "fit reported at the X stage" | `playground` |
| MCP `emit` tool: Python and JavaScript "return its refusal until the emitters land" | the four public emitters, with `rust` and `scala` in the target list | `ide/crates/rce-mcp/src/lib.rs:64` |
| "0 refused" (emitter agreement, 10.5) | "0 not translated" | 12:216, 12:225 |
| "promoted or refused" (D.11) | "promoted or deferred, with the reason" | 15:357 |
| CONTRIBUTING rule 1 and the grammar paragraph: "an accepted and a refused example"; "a refused one in `spec/examples/reject/`" | "an example in Form, and an example of a text not yet in it, shown with its report and path" | `CONTRIBUTING.md:33-34, 65` |
| "A new rule that refuses programs needs a program … that the rule refuses" | "A new rule needs a program that shows its report and its path" | `CONTRIBUTING.md:39-41` |
| "a reserved word refused with W-RES-1" | "a reserved word held with W-RES-1" | `CONTRIBUTING.md:69` |
| "A second spelling … is declined on that ground alone" | "A second spelling is held as a lift to the one spelling" | `CONTRIBUTING.md:72-73` |
| "the decline rate and its reasons" | "the share of proposals composed, and its reasons" | `CONTRIBUTING.md:90` |
| ledger decisions `declined` | `composed`, with the composition named; decisions are `accepted`, `composed` or `open` (COMP-18) | `LEDGER.md`, `claim-counts.py` |
| proposal headings "What it refuses" | "What it does not claim" | `encapsulation.md:21`, `energy-calibration.md:69`, `intent-holes.md:84`, `partial-order-log.md:108` |
| paper position items "Refused: …" | "Not claimed: …", or the named form (8.12) | paper 641-653 |
| "admitted and refused programs" | "programs in Form, and the paths of those not yet in it" | paper 687 |
| "the practices the design refuses" | "A separate library is brought in by `use`, a directive is rewritten to `use`, a feature flag is a named form" | `LEDGER.md` row of 2026-10-24 |

### 6.8 Code identifiers

| Current | Compliance |
|---|---|
| `RunOut.refused`, `Example.refused` (`ast.rs:213`) | `RunOut.kept`, `Example.kept` |
| `SAFE_REFUSED` | `SAFE_FORMS`, the table of named forms of W-SAFE-1 |
| `REFUSED` (the exit-status constant of `cli.rs`) | `OUTSTANDING` |
| `class Refuse` (`rt.js`, `rt.py`) | `class Kept` |
| `interp::Cause` | unchanged: `fail`, `trap`, `fuel`, `size`, `value`, `depth` |
| 57 functions in `spec/checker/tests` whose names contain the words (54 of them tests), and the HC-21 tests of PR #31 (`tests/host_refused.rs`) | renamed with `kept` or `fit`; `tests/host_refused.rs` becomes `tests/host_step_kept.rs`; the E.10 table follows, as `host_contract.rs::every_appendix_e_rule_is_cited_by_a_test` requires |
| Lean names | section 13 |

### 6.9 Kept verbatim

These are the words of a third party or a published record. They stay as written and are listed in `scripts/vocab-allow.tsv` (COMP-19):

- "implicit rejection" (FIPS 203) and "rejection sampling";
- JavaScript `doesNotReject`;
- the ACP option kinds `reject_once` and `reject_always`;
- systemd `SAFE_ATO_REFUSE_*`;
- comments quoted from Linux i2c and edk2;
- the SCXML, CloudEvents, IDNA and RFC test corpora under `stdlib/vectors/`, and the foreign vector `stdlib/vectors/text/UTF-8-test.txt`;
- the Raft quotation "it rejects the request" (16:283) [Ongaro and Ousterhout 2014];
- the dyld quotation in `soundness.md`;
- the mutation-analysis term "the share of mutated outputs the specification rejects" (paper 647);
- the Go figure "declined 40.9%" in `CONTRIBUTING.md`, quoted from the cited study;
- the engine test name `fold_rejects_wrong_state`, cited as data;
- `spec/LANGUAGE-REVIEW.md`, a dated review;
- this proposal and the dated rows of `LEDGER.md`, which quote the terms they replace;
- every published corpus row and every signed receipt.

## 7. Rules

Each rule is normative and names the artefact that enforces it.

**COMP-1 Two outcome kinds.** Every D entry, trace step and λ record of a program input carries exactly one outcome: `"applied"`, or `{"kept": ρ}` with the λ record `Kept(ρ, payload)`. 6.1, 6.4, 6.5 and 10.7 use only these words for the outcomes of inputs.

*Enforced by:* the writers in `interp.rs:3244-3262`, `host.rs` and `emit/rt.{rs,py,js,scala}`; `tests/form_signature.rs`; the fixtures of `spec/traces`; `tests/nameless.rs`, whose comparison at lines 75 to 77 is rewritten to parse `outcome` and compare outcome kinds; and the Lean lemma `outcome_dichotomy`.

**COMP-2 The outcome syntax.** 3.2 reads:

```
Outcome = Expr | "kept" ":" ( NAME | "trap" "(" STRING ")" ) .
```

`kept` is a contextual word, as `where` and `by` are (`lalr.rs:397-399`). In outcome position, `kept` followed by `:` is the keyword. Everywhere else it is a name, so the local `kept` of `stdlib/lib/data/cdc.rce:343-348` stays as written. `refused` leaves the reserved list of 2.4, so the list is one word shorter. A bare name `kept` as the whole outcome expression is written `(kept)`; no program of the repository has one. The earlier spelling `-> refused:` is reported as W-SYN-1 with the path `rewrite` to `kept:`.

*Enforced by:* `rce-spec-check` built with cargo's `--features tree-sitter`. It requires 0 LALR(1) conflicts, with `kept` in `lalr::CONTEXTUAL`, and agreement of the LALR(1) tables, the Rust parser and tree-sitter on every example. `tree-sitter/src/scanner.c` reads `kept` as the keyword when the next token is `:` and the outcome keyword is valid, beside its handling of `where` and `by` (`scanner.c:196-220`). The grammar gains the `trap` alternative, which on main it lacks (`grammar.js:36` accepts only `NAME`).

**COMP-3 A closed list of reasons.** 6.2 lists every trap text with its cause (4.3). Every reason written by the reference evaluator, the concurrency evaluator, the host or an emitted runtime is a declared error name or a listed trap text. Run-time fuel exhaustion is written `fuel exhausted` on every runtime.

*Enforced by:* a `traps` key in `spec/data/spec-data.json`, extracted from 6.2 by `rce-spec-data`; and `tests/trap_texts.rs`, which collects the trap texts built in `interp.rs`, `conc.rs`, `host.rs` and `emit/rt.*` and fails on any text outside that key.

**COMP-4 Every offered input has an entry.** An input offered to a host, or to `rce run` as EVENTS, has exactly one record (4.7): the program event it names, with its outcome; a declared event bound to it; or `host.Kept` {event, why}, which δ maps to the identity (T37). No input is dropped before D. A payload of the event's shape whose value lies outside a range or a refinement is the program event, kept with the trap of the field it misfits.

*Enforced by:* `Host::apply_named` (`host.rs:1094-1106`), the window acknowledgement (`host.rs:1189-1196`, with `WindowAck.rejected` renamed `WindowAck.held`), the serve loop (`cli.rs:1444-1450`) and example construction (`interp.rs:2586-2594, 3236-3240`); `tests/host_contract.rs::every_offered_input_has_an_entry`, which offers malformed, undeclared, late, oversized and out-of-range inputs to each adapter and checks one entry each and that replay equals the live run; `tests/refined.rs`; and `tests/cli.rs::run_records_every_event`.

**COMP-5 Every run folds every event.** `rce run`, `rce trace` (EVENTS and `streams`), host replay, the serve loop and every emitted runtime apply each event to the state the previous outcome left. After a kept outcome the fold continues from the kept state. An example log folds every event of its example, and the example states the outcome of its last event.

*Enforced by:* `cli.rs:93, 148-151`, `interp.rs:3279`, `cli.rs:485` and `interp.rs:2607-2614`; `tests/traces.rs`, whose pin at line 204 ("a refused first event ends the stream") becomes a pin of the full fold; `tests/cli.rs`; the outputs under `book/listings`; and `tests/emit_run.rs` for the Rust, Python, JavaScript and Scala runtimes.

**COMP-6 Run outcomes.** A run of `fn main` ends `done`, `stopped(ρ)` or `deadlock` (4.6). `conc::Outcome`, `host.MainEnd.outcome`, the emitted runtimes' stop tuples and `rce run` use these words. A replay verdict is reported as `Differs { seq }` beside the run outcome, never as one.

*Enforced by:* `conc.rs:1592`, `host.rs:1301-1307`; `tests/host_sched.rs`; `tests/emit_run.rs`.

**COMP-7 Every report names its path.** Each rule of section 7 and of Appendices C to F has a path column in its rule table, giving the kind of 4.8 and the path. Every diagnostic has the form of 4.8. The LSP diagnostic carries the same path in `data.fit` as {kind, text, edits}.

*Enforced by:* `rce-spec-data check`, which requires a `fit` field for every row of `w_rules` and `w_rules_06`; and `tests/fit_reports.rs`, which runs every fixture under `spec/examples/fit/` and checks the form of each report.

**COMP-8 One fit function.** Every tool builds the Form of the fitted text φ(t) (Definition 5) and reports each path it applied. `rce fmt fit FILE…` writes φ(t), `rce fmt fit check FILE…` lists the paths, and the language server offers each path as a code action, one action per reading of a `choice`. φ is deterministic and idempotent, and the CC-1 program hash names φ(t). A `rewrite` changes only layout or names, or is a rewrite listed in 8.2 whose fitted text `rce equiv` (6.10) shows equal to a reference written by hand. No tool applies a reading of a `choice`.

*Enforced by:* `rce fmt fit`, which takes over `scripts/fix-hygiene.py`, `fix-unused-loops.py`, `fix-embedded.py` and `rename-reserved.py`; `bin/spec_check.rs`, which requires each fixture `X.rce` under `spec/examples/fit/` with a determined path to have a sibling `X.fit.rce` that equals `rce fmt fit X.rce` and is in Form; and `tests/fit.rs`, which checks that `rce hash` agrees on each such pair and that no tool applies a reading of any `choice` fixture.

**COMP-9 Named forms carry their own claims.** Each named form has a row in a `forms` table of section 8 of the specification: the form, the rule that defines it, and the theorems claimed for it. The theorem table gains a column "claimed for", so each theorem is claimed for a definition only in a form that has it. The Form JSON lists each definition held in a named form under `forms`, as {definition hash, named form}. `rce check` prints the named forms of a module as notes, with the theorems not claimed for them.

*Enforced by:* the `forms` key of `spec-data.json`; `tests/named_forms.rs`, with one program per named form in `spec/examples/forms/` and its expected `forms` entry.

**COMP-10 Every paradigm has a named form.** Every construct or paradigm that the text names as outside its grammar has a row in `spec/lift/table.json` naming its form: the F.2 list, the W-SAFE-1 identifiers, the foreign constructs of W-SYN-1 and the table of 8.9. Each row has a program in `spec/examples/forms/` that `rce check` reports `ok`. F.2 becomes the table "Constructs held by composition", and it grows with every paradigm a program brings. The normative text states each form and no exclusion.

*Enforced by:* `rce-spec-check`, which requires a lift row and an `ok` program for every F.2 row, W-SAFE-1 identifier and row of 8.9; `tests/lift_go.rs` and the lift probes of `spec/lift/`.

**COMP-11 Fabrics are held by named forms.** A fabric's native semantics enters as a placement class (HC-12, HC-13, XF-11):

- A class whose bits the CPU reference reproduces is placed under HC-12, so placement changes cost and never bits.
- A native result that the CPU reference does not reproduce is a logged answer (HC-4, T32), from a fabric form the program names. Placement never chooses that form.
- A class that a host cannot execute runs on the CPU reference, and the host logs `host.Placed` with that choice.

*Enforced by:* `tests/host_place.rs` and the cross-architecture parity job (`scripts/xarch-parity.sh`).

**COMP-12 Lifts are checked by round trip.** A lift from a language L enters with a corpus. For every program P of the corpus that L's table prints, `rce equiv` shows lift_L(print_L(P)) equivalent to P (6.10). What a lift cannot determine becomes a hole (`intent-holes.md`) or a recorded oracle answer (D.8 L2), never a guess.

*Enforced by:* the conformance job per language, which is the gate of `one-emitter.md` section 7; and `tests/lift_go.rs` for the probe in `spec/lift/go`.

**COMP-13 Reserved constructs are held.** W-RES-1 reads "reserved, held for implementation (F.12)". F.12 gives each reserved construct one of three treatments (8.6), in a new column:

- (a) run from its definition by every tool, where the reference has one;
- (b) held with the construct marked unchecked and no theorem claimed for it, where only a check is missing, as D.8 L1 marks lifted elements "unproven";
- (c) a position with the path `gate`, where the reference has no meaning yet.

`rce reserved` prints the table.

*Enforced by:* the W-RES-1 diagnostics; F.12's table; `tests/reserved.rs`, with one case per F.12 row; `tests/v09*.rs`.

**COMP-14 Emission coverage is reported as coverage.** A construct that the Form holds and that a target does not yet print is reported as `pending FILE: target T does not yet print construct C (line N)`, with the row of `one-emitter.md` that will print it, and the status is 1. Exit status 2 stays a usage error.

*Enforced by:* `tests/emit_coverage.rs`, which prints the pending table per target.

**COMP-15 Verdicts on claims and records.** Replay, verification on open, pin checks, the layout check of `fmt`, term fencing, stored stamps and tags, signatures and example checking report `holds`, or `differs` with the first position k at which the record differs (Definitions 6 and 4.9). The verdict states what each side holds, keeps the record as found, and derives nothing past k.

*Enforced by:* `host.rs:1346, 1393-1408, 1775`; `cli.rs:237, 1158, 1486-1489`; `tests/host_contract.rs` (replay, and forged or missing host entries); `tests/v09_manifest.rs`; the chain tests of `ide-receipt`; `tests/cli.rs`.

**COMP-16 Kept outcomes and kept inputs are delivered.** HC-21 reads "A kept outcome is an input". The host entry is `host.StepKept` {cause, seq}, and the declared event is `StepKept(seq: u64, cause: text)`, matched by field name (HD-4). `host.Kept` {event, why}, PR #31's entry for a delivery the module's declaration cannot take (named `host.Unfit` while PR #31 was open), follows the same binding and the same descent bound, and also records the offered inputs of COMP-4. One name has one meaning (CC-5): `host.Kept` is only the {event, why} entry, and `host.StepKept` only the {cause, seq} entry. The names are fixed before PR #31 merges.

*Enforced by:* the PR #31 `host.rs`; `tests/host_refused.rs`, renamed `tests/host_step_kept.rs`; `tests/nameless.rs::kept_entries_are_name_free`.

**COMP-17 Exit status by kind.** 10.1 item 4 becomes:

| Status | Meaning |
|---|---|
| 0 | The command completed. Every input has its outcome, kept outcomes included, and every run has its run outcome. Every text is in Form as written; notes are allowed. Every claim and record holds. |
| 1 | Something is outstanding: a path of kind `rewrite`, `complete`, `lift` or `bound` not yet written (`rce check`, `rce fmt check`); an open `choice` or `gate` position; a claim or record that differs; a construct pending in a target (`rce emit`). |
| 2 | A usage error. |

`rce run`, `trace`, `emit`, `dim` and `host` build the Form of φ(t) when every open path is determined. They print the paths first and continue.

*Enforced by:* 10.1; `tests/cli.rs`, with one case per status and per tool; `book/scripts/check-listings.sh`; `programs/scripts/check.sh`.

**COMP-18 Growth through the gate, and the ledger.** A form the Form lacks enters under CONTRIBUTING rule 1, after rule 2 has shown that no composition of existing constructs holds the programs that need it. A row of `LEDGER.md` is decided `accepted` (into the normative text), `composed` (held by existing constructs, with the composition named, and no primitive added) or `open`. The reasons of rule 5 stay; they say why a construct is composed instead of added.

*Enforced by:* `CONTRIBUTING.md`, `spec/proposals/LEDGER.md` and the ledger section of `scripts/claim-counts.py`.

**COMP-19 One vocabulary.** Every word in the repository for an input outcome, a run outcome, a static report, a claim verdict or a record verdict is the compliance term of section 6. This covers normative text, diagnostics, command output, data keys, event names, host entry types, identifiers, test names and file names. The words of the search pattern occur only in the quotations and third-party terms listed in `scripts/vocab-allow.tsv`, one row per path glob and line pattern with its reason, and in stored records read under COMP-20.

*Enforced by:* `scripts/vocab.py check`, a step of `scripts/ci-tests.sh`. Its word `inventory` prints the line inventory per repository area.

**COMP-20 Hashes and history.** The hashes of δ entries do not change. A hashed name changes once, in the commit that renames it, together with:

- (a) every pin, fixture and golden that the name reaches, re-derived in the same commit (CONTRIBUTING rule 3);
- (b) a map from each earlier hash to its new one, committed beside the pins (`spec/proposals/compliance-repin.tsv`);
- (c) for stored records that are history (a host D, signed receipts, registered runs), a reader that reads the earlier spelling as the same kind, while the writer writes only the new one. A published or signed record is never rewritten.

*Enforced by:* `tests/form_signature.rs` (golden hashes unchanged); `stdlib/lib/MANIFEST.json` with `rce manifest-check`; `scripts/claim-counts.py check`; `conformance/tools/validate.py`, which checks clause ids against the map; `tests/host_contract.rs::stored_records_are_read_by_kind`; and the `ide-receipt` test `a_chain_written_before_the_rename_verifies`.

**COMP-21 The change keeps every Form.** Every text that has a Form before this change has the same Form after it, up to the `rewrite` of the renamed keyword, and every example that holds before it holds after it, with the same log.

*Enforced by:* `rce change` and `rce equiv` (CC-2) over the corpus, in the pattern of `scripts/t7-equiv.sh`; and `tests/examples_stable.rs` for the claim of 11.4.

## 8. Behaviour changes, case by case

This section covers every case the maps class (ii), and the static rules whose path is a declaration only the author can make. Cases of class (i) need only the words of section 6. Cases of class (iii) are in section 9. *Today* means at d2b28e76, or at a374226b for PR #31.

### 8.1 Inputs, runs and the host

| Case | Today | In compliance | Rule |
|---|---|---|---|
| C1. A payload of the event's shape outside a range or refinement | `apply_named` returns before appending (`host.rs:1094-1106`). The window's `rejected` list is "dropped, never in D" (`host.rs:806, 1189-1196, 1699`). The serve loop prints "input refused" (`cli.rs:1448`). An example with such a payload stops (`interp.rs:3236-3240`) or becomes W-INT-1 (`interp.rs:2586-2594`), so `spec/examples/51_refined_input.rce` has no example for its own intent. This contradicts 5.1 (05:86-88) and 6.5 ("no event is dropped silently"). | E is read at shapes: Ê = ⊎ₗ {Eₗ} × ⌊payload(Eₗ)⌋, where ⌊τ⌋ widens every integer type to the Form's int and every refined type to its base. For e ∈ Ê \ E, δ(s, e) = (s, [Kept(trap(m), e)]), where m is the trap text of the first misfitting field. The entry is in D, and an example may state it: `example SetVolume(pct: 101) -> kept: trap("refinement: value is outside Pct")`. This is the check 5.1 and 6.4 item 3 already perform, with its outcome now recorded. | COMP-4 |
| C1′. An undeclared event name, or a payload of another shape | Printed and dropped. `rce run` prints a `REFUSED` line and exits 1 (`cli.rs:688`, 12:92). | At a host: `Input(data)` when the module declares it, otherwise `host.Kept` {event, why}. In `rce run`: the line `k NAME kept: WHY`, the state kept, and the run continuing. | COMP-4, COMP-16 |
| C2. Runs stop at the first kept event | `rce run` stops and exits 1 (`cli.rs:93, 148-151`). Traces and streams end (`interp.rs:3279`, `cli.rs:485`, pinned by `tests/traces.rs:204`). A kept event before an example's last event fails the example (`interp.rs:2607-2614`; 08:234, "earlier events must be applied"). | Every event is folded (5, the run of the counter). An example states the outcome of its last event; an intermediate outcome is stated by an example of the prefix. By the maps, none of the 13 conformance tests that expect W-INT-1 depends on an earlier kept event. 07:59-60 reads "a kept entry carries the pre-state's hash, and the log continues from it". | COMP-5, COMP-17 |
| C3. Placement on a fabric this host cannot execute | Reported before anything is logged (`host.rs:1576-1585, 1633-1634`; 16:463). | Run on the CPU reference, which gives the same bits by HC-13, and log `host.Placed` naming `cpu`. The fabric stays a named candidate. The code already falls back this way when a registered fabric errors at run time (`host.rs:1578-1579`). | COMP-11 |
| C4. A query whose step would change state (HS-8) | "not a query and is refused" (`host.rs:1807-1821`). | A query is evaluated on the current state and never appended. It answers with λ(s, e), and reports that the change δ₁(s, e) ≠ s applies only when the request is offered as an input. | COMP-4 |
| C5. A late `fn main` input (HC-6) | An input whose tick is not after the last logged step, or that would change a logged step, is not logged (`host.rs:1356-1360, 1433-1446`; 16:117-118). | The input is admitted at the first tick after the last logged step. `host.MainInput` carries `late: true` and the requested tick as `asked`, the pattern of E.2a's `host.Input {late: true}`. Placement after the last logged step cannot change a logged step, so the second condition does not arise. | COMP-4 |
| C6. A construct an emitter does not translate | `rce emit` exits 2 (`cli.rs:1004-1014`). | `pending FILE: target T does not yet print construct C (line N)`, status 1; the construct is printed from its definition once `one-emitter.md` section 11 covers it. | COMP-14 |
| HC-24: "An input refused before it is logged" | Printed, never in D (PR16:606). | One record per offered input (4.7): the fitted event, `Input(data)` when declared, or `host.Kept`, delivered as `HttpMalformed` where HD-2 says so. | COMP-4 |
| HC-5, HS-6 | An effect without a capability, and a result that cannot be bound, are not delivered (16:73, 16:528-532). | A `failed` result with the reason (HC-3), logged; the path names the capability or the declaration. | COMP-4 |
| E.4c, delivery by declaration (PR #31) | At 96f4538f a module that did not declare the host's shapes was "Refused at start". At a374226b E.4c already started on any such module, and its rule now named HD-2 reads "`http` serves any module": PR #31 already holds these cases. | PR #31 names its entry `host.Kept` (formerly `host.Unfit`) and its rules HD-1 to HD-4 (formerly HF-1 to HF-4); the use of `host.Kept` is extended (4.7). HD-1 to HD-4 otherwise stand as written on PR #31. | COMP-16 |
| Handler-spec keywords and kinds (15:75) | Unknown keywords are not read; duration defaults apply silently. | Unknown keywords and kinds are held as opaque named declarations, with a note. Each default becomes a recorded path, as 17:13 requires ("never silently ignored"). | COMP-7 |
| An undeclared endpoint (15:105) | Not admitted. | Held as a hole, as escalation endpoints already are. | COMP-7 |
| A stale promotion record (15:321) | Not promoted, with no path. | The record carries the staleness and the path (`rce currency --mark`). | COMP-15 |
| A commit inside a tick (`conc-gap-register.md:15`) | Not admitted. | Deferred to the tick boundary. | COMP-5 |
| An opaque guard in a lift (15:291, D.8 L2) | `Lifted::step` does not step through it. | Its value is asked through the recorded oracle of D.4 and logged; it is never guessed. | COMP-12 |

`rce run` after COMP-5 and COMP-17 is shown in section 5. Its status is 0.

### 8.2 Static rules: the path of each rule

**The report channel.** For every rule, the checker today prints `line N: RULE msg`, builds no Form, prints `REFUSED file:` from `rce check` and exits 1. `run`, `trace`, `emit`, `dim` and `host` stop at the same point (`cli.rs:96-99, 501-504, 870-876, 1134-1140`). Under COMP-7 and COMP-8 each rule gives the path below, and a text whose open paths are all determined has the Form of its fitted text.

**Fixture counts.** The reference checker reports these first rules on the fixtures of `spec/examples` today:

- `reject/` (12): W-SYN-1 10, W-LEX-1 1, W-LEX-2 1.
- `ill_typed/` (48): W-TY-1 10, W-LIN-1 6, W-DEV-1 3, W-FFI-1 3, W-ITER-1 3, W-REC-1 3, W-EFF-1 2, W-OWN-2 2, W-SIMD-1 2, W-SPAWN-1 2, and one each of W-LOOP-1, W-MATCH-1, W-NAME-1, W-OWN-1 (with W-LEN-1), W-OWN-3, W-PLACE, W-REG-1, W-SAFE-1, W-STATE-2, W-STEP-2, W-STEP-4 and W-TRAIT-2. W-WORLD-1, W-UNUSED-1 and W-TY-1 accompany several.
- `inadmissible/` (16): W-INT-1 11, W-CT-1 3, W-LEX-3 1, and `loop_bound`, which reports W-UNUSED-1 and W-UNUSED-2 and, once those rewrites are written, a check that reaches its loop bound.

So the 76 fixtures become 12 syntax fixtures, 46 type fixtures, 4 admission fixtures, 12 claim fixtures and 2 notes (`unhandled_event`, `unreachable_step`). Of the 346 conformance tests that expect `reject`, 329 become `fit`, 13 become `differs`, and 4 become `ok` with a note (`unhandled-event-reject`, `unreachable-step-reject`, `w-match-2-unreachable`, `w-state-1-no-fields`).

**Lexical and syntax**

| Rule | Path |
|---|---|
| W-LEX-1 (09:11) | `rewrite`: strip the BOM; turn CR and CR LF into LF; a tab inside a literal becomes `\t`; indentation that uses only tabs, or only spaces, keeps its block structure at every tab width, so tabs become 4 spaces. `choice`: indentation that mixes tabs and spaces (the readings at widths 8 and 4); bytes that are not UTF-8 (transcode under a named encoding, or hold them as a `List[u8]` literal). |
| W-LEX-2 (09:12) | `choice`: both block readings are reported. |
| W-LEX-3 (09:13) | `lift`: a literal beyond every fixed type becomes a `core.exact` value (`mathint` once it passes the gate). A `\u{}` that is not a scalar value is held as the bytes it names. |
| W-SYN-1, constructs of other languages (09:14) | `lift` by the lift table: `class` becomes struct, impl and state; `a<b<c` becomes `a<b and b<c` (11:353, 17:775); an assignment inside an expression becomes a `let` before it; an `if` expression without `else` becomes the Option or unit form; `try` in a branch becomes `match`; a keyword used as a member name is escaped; a missing `forall` is inserted; an empty `state` becomes the unit region; an empty `kind` becomes the empty type. |
| W-SYN-1, incomplete or damaged text | `choice`: the position and the fewest edits that put the text in the language of section 3 [Aho and Peterson 1972]. A source language with a front end lifts through it (`spec/lift`). |
| W-SYN-1, `-> refused:` | `rewrite`: `kept:` (COMP-2). |
| W-RES-1 (09:15-17) | `reserved` or `gate` (8.6). |

**Names and modules**

| Rule | Path |
|---|---|
| W-NAME-1 (09:20) | `choice`: declare or import the name (`rce lib` lists library names); a near declared name is offered; the position may be held as a hole (`intent-holes.md`). `decimal` (`programs/money/fit/decimal.rce`): `lift` to `core.exact`. |
| W-NAME-2 (09:21), W-PAT-1 (09:41) | `rewrite`: rename; names carry no meaning (T7). W-PAT-1 is a `choice` of which binding to rename. |
| W-MOD-1, W-VIS-1 (09:23-24) | `choice`: use the module's `pub` surface, or have its owner export the item. The module's declared boundary is kept (T36); a tool never infers an export. |
| W-MOD-2 (09:23) | `form`: an import cycle is checked as one package. |
| W-MOD-3 (09:26) | `rewrite`: `use p.q as r`; the linker qualifies a clashing name by its module (XF-2). |
| W-SAFE-1 (09:29-30; `check.rs:106, 230, 234, 817, 821, 1099`) | `lift`: `Rc` and `Arc` become owned copy-on-write values or content ids; `RefCell` becomes a `state` field or an actor; `static` becomes `const` or `state`; `unsafe` and `transmute` go through the `extern "C"` boundary of 5.11. The message reads "`Rc` has its form in RCE as …". |
| W-FFI-1 (09:32) | `lift`: another ABI becomes a host adapter (HC-4) or a fabric's named form; a generic extern is monomorphised; a comptime extern call moves to run time. |
| W-WORLD-1 (F.4.1, 17:176) | `rewrite`: `&World` becomes the extern's first parameter. `choice` where a caller does not hold `&World`: move the call into `fn main` or a task function, or pass the handle. |

**Types**

| Rule | Path |
|---|---|
| W-TY-1 (09:37) | `rewrite` for a lossless widening, with the conversion inserted. `choice` where the conversion is a choice; the report names the candidates. Untyped code lifts at the Form's JSON-shaped value type, with checks at run time [Siek and Taha 2006]. `return` in a step body (`check.rs:2161`) is the early end of the step. |
| W-TY-2 (09:38) | `rewrite`: `τ??` becomes a kind with three variants; `&&τ` becomes `&τ`. `choice`: a borrow, region or `fn` inside data becomes an owned copy, a handle or a kind. |
| W-TY-3 (09:40) | `rewrite`: an explicit `as`, with N4 fixing the rounding; narrowing becomes a checked conversion that traps out of range. |
| W-MATCH-1 (09:42) | `complete`: the missing arms, each with `trap("match: no arm for V")`, reported as holes. δ stays total. |
| W-MATCH-2 (09:42), W-STEP-4 (09:58) | note: the arm or step never runs; the first-match order of 6.4 already gives the meaning. |
| W-PLACE (09:44) | `rewrite`: `var`. |
| W-LEN-1 | `choice`: `len_chars` or `len_bytes`; a lift reads the source language's meaning (Python `len` counts characters, Go `len` counts bytes). |
| W-DROP-1 (`check.rs:1935`) | `rewrite`: `let _ = e`, as the message already states. |

**Ownership and borrowing**

| Rule | Path |
|---|---|
| W-OWN-1 (09:47) | `rewrite`: `.copy()` of a value that is not linear. Form values have no aliasing. |
| W-OWN-2 (09:47) | `choice`: copy in and out, or iterate over a snapshot. |
| W-OWN-3, W-REG-1, W-SPAWN-1, W-STATE-3 (09:48-54) | `choice`: own a copy, or pass a handle or an index. |
| W-LIN-1 (09:75) | `complete`: insert the release on each path; a copy becomes an explicit foreign `dup`. |

**The Form**

| Rule | Path |
|---|---|
| W-STATE-1 (09:53) | note: an empty region is the unit record. |
| W-STATE-2 (09:53) | `lift`: the initialiser becomes the first input of D, so s₀ stays a constant. |
| W-STEP-1 (09:55) | `choice`: declare the event or the region, or hold it as a hole. |
| W-STEP-2 (09:56; `check.rs:1493-1498`) | note: "event E changes no region: δ applies it with every region framed (6.4)". The message today, "δ would not be total", is wrong: 6.4 item 1 and T2 make δ total on such an event. |
| W-STEP-3 (09:57) | `rewrite`: the block form naming the field; `choice` when several fields have the expression's type. |
| W-EFF-1 (09:60) | `lift`: an effectful `fn` becomes a task or a step; a read of another region becomes a payload field or a merged region. |
| W-REC-1 (09:63; 08:114, 08:140, 08:164, 08:171) | `bound`: `bound N`, with N the module's LOOP_LIMIT, on the first function in declaration order of each cycle without a measure; or a `decreases` the author writes. T2 holds by the bound. T48 is claimed only with a measure. The 16-parameter search limit (08:140) is a limit of the checker, and `bound` holds the cycle. |
| W-CT-1 (09:70; 08:46) | `choice`: the comptime evaluation's outcome (a trap, or `fuel`) is reported as that outcome; change the expression, or move the computation to the first input of D (W-STATE-2), where its outcome is recorded. |
| W-INT-1 (09:72) | a claim verdict: section 9. |
| W-SIMD-1 (09:73) | `lift`: an explicit `map` or `zip`. |
| W-DEV-1 (09:78) | `lift`: split the host and device parts. A growable value in a device region takes the fabric form of declared capacity (XF-12). Fabric-native forms are held as named forms, with their results recorded in D. |
| W-ITER-1 (09:81) | `rewrite`: `.collect()` for an iterator value used elsewhere; `lift` to a task and a channel for an iterator with effects. |
| W-REF-1 (09:84) | `form`: a predicate outside the decidable fragment becomes a run-time `check` in a constructor. T35 is claimed for predicates in the fragment. |
| W-LOOP-1 (09:89) | `rewrite`: `const LOOP_LIMIT: u64 = n`. |
| W-TRAIT-1 (09:91) | `complete`: the missing methods, with trap bodies, reported as holes. |
| W-TRAIT-2 (09:91) | `rewrite`: an orphan impl goes through a newtype. |
| W-TRAIT-3 (09:92) | `rewrite`: the impl derived from N6 equality and the total order of 6.1, the "tool step that emits source" of F.2. |

**Indexed values (5.16).** These rules guard declared contracts (T36), so the declaration stays with its author.

| Rule | Path |
|---|---|
| W-CTX-1 (09:96) | `rewrite` when the context is unique: insert `C.encode`. |
| W-SEC-1 (09:99) | `choice`: replace the operation with a digest or an equality bit. |
| W-CAP-1, W-CAP-2 (09:102-105) | `choice`: add the `Cap` field or parameter. A tool never infers authority. |

**Appendix C**

| Rule | Path |
|---|---|
| W-PAR-1 (09:118) | `form`: source order or a declared merge, with determinism kept in D. T16 is claimed for the form with disjoint footprints. |
| W-PAR-2 to W-PAR-4, W-GRP-1 to W-GRP-3 | `rewrite`: results are collected or reduced in declared order. |
| W-ND-1, W-DL-1 | `form`: the choice is logged in D (T21); `deadlock` is a defined run outcome. |
| W-SCOPE-1, W-ATOM-1, W-ATOM-2 | `rewrite`: the enclosing step becomes the scope; an atomic block is split at the receive or the effect. |
| W-SES-1 to W-SES-3, W-NODE-1, W-SUP-1, W-SUP-2, W-RT-1, W-INTR-1 | `choice`: the report names the declaration to add or change. |
| W-SES-4, W-CRDT-1, W-LVAR-1, W-LVAR-2, W-BUD-1 | `form`: 8.3. |
| A duplicate supervisor child (14:592) | `rewrite`: remove the repeat. |

**Other static cases**

| Case | Path |
|---|---|
| W-TX-1 (07:255) | `form`, or the `rewrite` `fail X(v)` → `return Err(X(v))`: 8.3. |
| A write-back call in a conditional position (07:274) | `rewrite`, in the lowering: the short-circuit or the branch becomes an `if` statement before the ANF pass, which keeps the order of evaluation (T34). No report. |
| A cyclic type alias (`check.rs:130`) | `lift` to a `kind`. |
| Q16, argument labels (11:361) | `rewrite` when the labels name the parameters; otherwise `choice`. |

### 8.3 Run-time properties held by named forms

Each named form below is made of existing constructs. Each guarantee is claimed for the form that has it (COMP-9).

| Locations | Rule or case | Named form | Claims |
|---|---|---|---|
| 07:255; 11:142, 11:346; 17:297-307 | W-TX-1 (F.4.4) | Non-transactional call: after a `fail` that follows an effect that cannot be undone, the effects stand in D and the `&mut` arguments keep their current values, as in a saga [Garcia-Molina and Salem 1987]. | T47 and T33's restore for the transactional form only |
| 11:349, 17:770 | `?`, `return` or `break` in `par` arms (C.4) | The arm's early exit is its `fail`. The `par` takes the outcome of the lowest-numbered failing arm and restores, as for contracts. | T16 |
| 14:191; 17:459, 17:923 | W-BUD-1 (C.5) | A non-atomic budget form. A model call outside a budget is wrapped in a budget equal to the input's fuel (HC-7a). | |
| 14:341, 14:344 | W-SES-4 (C.6) | Dynamic session form: a use of a closed endpoint traps, and D records the trap [Igarashi et al. 2017]. | T28l for the linear form |
| 14:380 | A cycle of waits | Held, with `deadlock` as a possible run outcome (4.6). | T28p for the acyclic form |
| 14:477, 14:480 | Merge certificates (C.9) | Uncertified merge: the laws are sampled [Claessen and Hughes 2000], any counterexample is recorded, and results follow the logged schedule. The certificate set grows through the gate. | T21 |
| 14:491 | W-CRDT-1 | The componentwise `Merge` is derived (G-counter and PN-counter [Shapiro et al. 2011]); a field without a merge takes the logged-order form. | |
| 14:553, 14:575; `check06b.rs:577` | LVar thresholds; W-LVAR-1, W-LVAR-2 (C.10) | Quasi-deterministic form: the threshold that fired is recorded in D [Kuper et al. 2014]. The run-time check every wait already makes stands. | T29 for the deterministic form; T21 for frozen reads |
| 14:787; 17:647, 17:1008, 17:1019 | Recursion without a measure | `bound` (8.2). | T2 by the bound; T48 only with a measure |
| 17:192 | `while true` | Already bounded: it traps at its bound. "Run indefinitely" lifts to a step on an event the host drives. | |
| 17:599, 17:985 | W-CON-1 (F.6.2) | The function is held, and every call takes the contract outcome, as outside the decidable fragment (Q33). | T46 |
| 17:634 | An unbounded ghost quantifier (F.6.3) | An unchecked ghost claim, or a fuel bound. | |
| 17:712; `CORE-PROPERTIES.md:106` | W-LAW-1 (F.6.5) | The impl is held and its counterexample recorded. No theorem assumes laws. | |
| 17:778 | Indexing proofs (F.6) | The out-of-range trap of N8 is the dynamic form; the static proof is the guarantee of the static form. | |
| 17:784 | A symbolic comptime assert | A run-time `check`, or a ghost claim marked unchecked. | |
| 16:561 | Queries | The fuel bound of 6.7: a result, or the `fuel` outcome, recorded and never measured in wall time. | |
| 16:658, 16:661 | K-3; K-6 (E.11) | Written-order form; bounded fixpoint iteration. | |
| 16:692, 16:701 | TU-2 (E.11) | Dynamic-shape form with a run-time shape check [Henriksen et al. 2017]. | |
| 16:708 | G-3 (E.11) | Dynamically typed pattern form, with run-time checks. | |
| `CORE-PROPERTIES.md:76` | "discharged or refused" | "discharged, or enforced at run time". | |
| `intent-holes.md:17, 38, 56` | W-HOLE-1, W-HOLE-3 | The intent is held, with its completeness score reported; an obligation that cannot be decided becomes a run-time check, never an assumption. | |

### 8.4 Authority and information flow

Labels and capabilities are erased at run time (5.16, F.4.2), so holding a module in one of these forms does not change how it runs. A tool never infers authority: every path names a declaration the author writes.

| Locations | Rule or case | Named form or path |
|---|---|---|
| 17:213, 17:218; 17:740 | W-IFC-1, W-IFC-2; W-TRI-1 | `release` under a held `Cap` that names the label, or a sink declared `@{L}`. T42 is claimed for flows in that form. |
| 17:240, 17:371, 17:514 | W-IFC-5 | `endorse` under its capability first. T43 is claimed for that form. |
| 17:227, 17:437 | Checked operations (F.4) | `rewrite` to the checked form that returns an Option. |
| 17:231 | The outcome channel (W-IFC-4) | The kept record carries the payload's label. |
| 17:333, 17:766 | A closure that captures authority | `lift`: the capability becomes a call parameter, by defunctionalisation [Reynolds 1972; Danvy and Nielsen 2001]. |
| 17:439, 17:941 | Secret-dependent control (F.5.4) | `rewrite` to `ct_choose` written in the program; otherwise the function is held as a plain `fn`, with no T45 claim. |
| 17:480, 17:485, 17:489 | Prompt and model inputs (F.5) | `rewrite`: the data is wrapped in `Prompt.encode`; a decision on untrusted data is endorsed first. |
| 17:737 | A send awaiting a grant | The broker holds the send pending a grant event, as TRIFECTA-2 already does. |
| 17:902 | Paths | `sub` is normalised against the root. A path outside it gets the capability record {id, kind: "capability", reason}. |
| `CORE-PROPERTIES.md:52` | A secret literal | `rewrite`: the literal moves into a field of an input declared secret. |

### 8.5 Hygiene rules of Appendix F

Every one of these is a `rewrite` that changes no meaning (O4).

| Locations | Rule or case | Path |
|---|---|---|
| 17:99-114; 14:820; 17:834 | W-UNUSED-1, W-UNUSED-2 and unused names | Prefix `_`, or drop the name; drop `_` from a name that is read; `let _ = e` for a dropped result; remove a dead store; fold a constant condition (Q17). 17:114 reads "fitting changes no meaning (O4)", and 17:834 "is fitted or given its named form, never silently re-read". |
| 17:135, 17:892, 17:987; 14:30, 14:107, 14:813 | Bindings named `result`; shadowing | Rename (T7). |
| 17:124-131 | Confusable characters | Write the character as `\u{…}`, the rule's own path. |
| 17:140 | An unqualified variant | Qualify the variant. |
| 17:688, 17:690, 17:704 | `if let` forms; implicit defaults | Rewrite to the `match` they desugar to, or to `let P = e`; name the chosen default. |
| 17:768 | A duplicate or unreachable arm | Remove it, with a note. |
| 17:777 | A duplicate key in a map literal | `choice`. A lift from Python or JavaScript reads last-wins as its named form. |
| 17:561, 17:562, 17:1041, 17:1042 | Template text | `{` becomes `{{`; an expression field becomes a `let` and a named field; a template used as text takes an explicit `text.t`. |
| 17:976, 17:989, 17:956 | Clause placement; a ghost name in real code; manifests | Move the clause to the head of the body; the ghost name becomes a plain `let`; canonicalise the manifest. |
| 15:173 | A D.3 join for a sink no rule writes | Held inert and reported as a note. |

### 8.6 Reserved constructs

The locations are 11:164, 14:796, 14:823, 17:13, 17:42, 17:442, 17:861, 17:947, 17:1044, 17:1055, 17:1065-1069, `CONTRIBUTING.md:69`, `cli.rs:591` and paper line 256. The place of each reserved construct is already specified in F.12. COMP-13 holds each in one of three ways:

| Way | Constructs | What happens |
|---|---|---|
| (a) The reference has the meaning | `ct fn`, contract clauses, `Prompt`, `Header` | Every tool runs or prints them from that definition. |
| (b) Only a check is missing | `law`, ghost parameters, quantifiers, `R.check`, `Template` values, structural and loop measures, `host.revoke` with no host log | The module is held with the construct marked unchecked, as D.8 L1 marks lifted elements "unproven". No theorem is claimed for the construct. A note names it. |
| (c) No meaning yet | `mathint` | A position with the path `gate`. It enters through the gate. |

Lifts and library growth:

- `declassify(x, reason)` lifts to `release` under the capability.
- `pow` on `f64` (14:796): the library gains a correctly rounded `math.pow` (6.3 F9).

Wording: 17:1065-1066 reads "Why each is still reserved, and what the Form needs to hold it". 17:1044's "refused by version" names a distinction the text no longer has, since there is one language at a commit; it reads "reserved, held for implementation".

### 8.7 Lowering and emission

| Locations | In compliance |
|---|---|
| 12:313, 12:314; 15:252 | Write-back calls nested in expressions: the order-preserving ANF lift that T34 already performs in the emitters. A call in a conditional position follows 8.2. |
| 15:261 | `try` in a lowered position: ANF into a `let`-bound `try`. |
| 15:269 | Brick lowering: `f64` is the Form's float; narrow integers become `i64` with range checks; closures are defunctionalised. |
| 15:272; `LEDGER.md:12`; `closing-the-loop.md:100`; `form-ir-2.md:11`; paper 609, 714 | Content-addressed definitions in the Form IR (`form-ir-2.md`) remove the term-size cause. Meanwhile: "bricks that Form IR format 1 cannot yet hold for term size". |
| 15:273; 12:540, 12:542 | "Every brick not yet lowered is recorded with its line and the lift it needs" (`{"pending": r}`); the brick is held as a reference brick. |
| 17:16, 17:830, 17:878, 17:937, 17:384 | Printed from definitions (`one-emitter.md`); 878 uses the copy-in restore; 384 uses positional names (T7). |
| 17:992, 17:1074 | Contract clauses already lower to guards that trap with the fixed record, and every emitter prints trap text, so contracts are emitted with no change of protocol. |
| `emit-targets.md:40, 41, 338` | Already restated by `one-emitter.md` section 11 ("How each language is held"). |
| `cli.rs:1004-1014` | COMP-14. |
| `ide/crates/rce-mcp/src/lib.rs:64` | The `emit` tool's description lists the four public emitters, `rust` and `scala` included. |

### 8.8 Engine profile and measurement records

| Locations | In compliance |
|---|---|
| 15:205, 15:210 | D.6's `reject` output is the output `mismatch`: the event is applied and the mismatch is an `error` row. |
| 15:357 | A promotion record is `promoted` or `deferred`, with the counterexample or the staleness. |
| 15:381, 15:402, 15:409, 15:415, 15:423; `energy-calibration.md:62`; `receipts-P1-P10.md:27, 29, 50` | Each term is held at the weakest label its evidence supports: a reading without provenance becomes `est`, and a figure without a basis becomes `unpriced`, with the figure kept as text, as D.11 already keeps such text. Unknown fields are kept under `ext` (P10). Terms of different amortisation or sampler go in separate columns, as `unpriced` already does, and n\* takes a defined kind when its terms cannot be compared. A calibration entry without its protocol is held as an uncalibrated entry, not used for pricing. |

### 8.9 Paradigms and constructs the text names as outside its grammar

COMP-10 gives each a named form. The rows cover F.2 (00:69-71 and 17:54-80), the ledger row on effect handlers, 11:154, 11:329, 17:393, 17:780, 17:790, `CORE-PROPERTIES.md:91`, `LEDGER.md` rows of 2026-10-22 and 2026-10-24, the synonym hits ("excluded", "forbids", "syntax error", "not offered", "not allowed", "never placed", "not admitted", "declined"), and the 33 paradigm rows of `docs/figures/src/replace/{failure-classes.csv, matrix.py}` labelled "EXCLUDED … rejected by derivation". F.2's title "Non-goals (excluded because each breaks a theorem)" becomes "Constructs held by composition, because a primitive form of each would break a theorem".

| Construct | Named form | Theorems kept |
|---|---|---|
| Effect handlers, one-shot [Plotkin and Pretnar 2009] | The operation is a λ request and the handler is a step on its result event, the shape of HC-3. | T2, T3 |
| Effect handlers, multi-shot | Each resume is a logged choice (T21) under fuel. | T2 by fuel, T3, T21; T6 for the clones of captured state is open question 5 |
| `await`; coroutines, generators, live variables | Task functions and channels, or steps on result events, as asynchronous methods compile to state machines [Bierman et al. 2012]; push `iter` (5.14), or a region with state and a resume event. | T2, T15 |
| Switchable contracts | Each configuration is its own program with its own hash (CC-1). "Observe" is a named form that logs a violation as output. | T46 for the enforcing form |
| Mutable aliasing across `par` | The source-order sequential form (17:822). | T16 |
| Runtime reflection | A comptime or tool step emits source; the concept record of 6.6 serves names as data outside the hash; closed `derive` (5.9). | T7 |
| Pointers, `alloc`, uninitialised memory, `unsafe`, shared mutable pointers | Handles or indices into regions or arenas. | T6 |
| Garbage collection | An arena or region form whose collection points are recorded in D [Tofte and Talpin 1997]. | T6, T3 |
| `bitcast`; uninitialised values | Named `core.bytes` conversions; Option. | T1; T5 |
| Overloading of the order operators | Named methods. | principle (e) |
| Inheritance, run-time casts, `any`, subtyping | Traits, struct embedding, a closed `kind` with `match`; the Form's JSON-shaped value type. | T5 |
| `defer`, `with`, `try` … `finally` | A linear handle consumed by a named function (5.12). | T12 |
| `raise` | `fail`, or a Result. | T8, T33 |
| loop-`else` | `first` returns an Option, then `match`. | T2 |
| Set and map comprehensions | A list comprehension and `from_list`. | C.6 cost |
| Glob imports | `rewrite`: an explicit list. | T7 |
| Conditional compilation, build features, flags (17:393) | One named form per configuration, each with its own hash. Fabric-native forms are held the same way. | CC-1 |
| Manifest inheritance; unknown manifest keys | Flattened; kept under `ext`. | |
| Python interop, raw pointers, C layouts, libraries found by path search (17:790) | A foreign component of the `World` with logged answers (HC-4, XF-5): copied canonical records, buffers or handles, with the library pinned by hash. | T32, T37 |
| `eval`, just-in-time compilation | An input event that carries code, held through the checker and recorded in D; the code runs as its own Form. | T3 |
| Escape hatches | Named `extern "C"` forms. | T6 for the RCE side |
| Partial definitions | Held under a fuel bound. | T2 |
| SQL and command injection | W-CTX-1 encodes the input automatically [Su and Wassermann 2006]. | T36 |
| Comparison chains (11:353, 17:775) | `lift` to `a<b and b<c`. | T10 |
| User `derive` (11:365, 17:783) | A tool step that emits source. | T7 |
| Statements in expression position (11:367) | A `let` lift. | |
| A second spelling of a construct (`CONTRIBUTING.md:72-73`) | A lift table reads it, and `fmt` writes the one spelling. | principle (e) |
| A separate library, a `#!lib` directive, a feature flag (`LEDGER.md`, 2026-10-24) | The library is brought in by `use`, the directive is rewritten to `use`, and the flag is a named form. | XF-2 |
| Leases (16:326) | A named form that states its clock-skew assumption. | T41 under that assumption |
| A device without binary64 (16:473, "never placed") | Its native float is held as a named fabric form, and its results are recorded in D as logged answers, so replay reads them and does not recompute them. | T32 |
| Floating-point addition in K-8 (16:663) | A fixed reduction tree with a logged order (C.8, XF-8). | T18 |

The ledger row on effect handlers is read the same way. A resumable handler is not added as a primitive, because T2 and T6 would not hold for it. Programs that use handlers are held by the named forms above, and the decision becomes `composed`.

### 8.10 Standard library

| Location | Today | In compliance |
|---|---|---|
| `stdlib/lib/num/linalg.rce:737, 1478` | Inputs outside the scaling range answer `Shape`. | Reproduce the rescaling of LAPACK's DGELS and DGEEV. |
| `stdlib/lib/net/http.rce:7` | obs-fold answers an error. | Replace it with a space, as RFC 9112 section 5.2 permits. |
| `stdlib/lib/net/http.rce:10`; `stdlib/vectors/SOURCE.md:387` | obs-text bytes answer an error. | Hold the field value as bytes (the UTF-8 carry of PR #31, HD-3). |
| `stdlib/lib/enc/enc.rce:277` | CBOR floats answer an error. | A float form for CBOR [RFC 8949] within the finite `f64` of 6.3. |
| `stdlib/lib/time/time.rce:7, 156` | `23:59:60` answers an error. | A named form for the leap second [ITU-R TF.460-6]; POSIX instants are unchanged. |
| `stdlib/README.md:22`, and about 40 lines of library error wording | "overflow refused" | "answers `BadInt` on overflow"; each function's declared error is named as its answer (4.11). |

### 8.11 Grammar and tools

| Case | In compliance |
|---|---|
| tree-sitter on main has no alternative for the `trap` form, although 3.2 and the Rust parser (`parse.rs:1062-1075`) read it; editors mark a valid form of the specification as an error, and the cross-check agrees only because no file on main uses that form | The rule of COMP-2, with the alternative PR #31 adds (`grammar.js:36`, `scanner.c:196`, corpus `declarations.txt:98`), carried with the rename. |
| The report channel (12:65-69, 12:397-398; 17:221, 17:793) | Each line names the rule, the line and the path (4.8). `rce fmt fit` or an LSP code action writes determined paths. LSP severity follows the kind. 17:793 reads "A diagnostic names the true rule and its path." |
| `tests/nameless.rs:75-77` | Parses the trace JSON and compares outcome kinds (COMP-1). |
| `scripts/fix-hygiene.py`, `fix-unused-loops.py`, `fix-embedded.py`, `rename-reserved.py` | Their determined rewrites move into `rce fmt fit`, so one checker applies them (10.1 item 3). |
| `rce v09-status` | Becomes `rce reserved` (COMP-13). |
| The book (chapters 1, 2, 3, 5, 7, 9 and 13) | The listings that show a refusal show the path: chapter 1's tab is W-LEX-1 `rewrite`; chapter 2's mixed numeric types are W-TY-3; chapter 3's chained comparison is a `lift`; chapter 5's unhandled event is a W-STEP-2 note; chapter 7 is W-MATCH-1 `complete`; chapter 9 is W-OWN-1 and W-OWN-2; chapter 13 is W-REC-1 `bound`. `ch05/refused/unhandled_event` moves to a listing in Form, with its note. |

### 8.12 The paper and the decks

The rows of class (ii) in `research-openie-web/src/content/papers/rce.md` (eight by the maps), with line 124, which states the behaviour that C1 changes:

| Line | Compliant text |
|---|---|
| 124 | "Malformed payloads … and undeclared events each have their entry in D: the event kept with its trap, or `host.Kept`" |
| 256 | "held as reserved, not yet implemented; it enters through the gate (W-RES-1)" |
| 334 | "one program whose projection the checker reported with its renaming path" |
| 609 | "the 77 standard-library bricks that Form IR format 1 cannot yet hold for term size" |
| 643 | "RFC 8785 is held as a named output projection; canonical JSON stays the hashed bytes" |
| 645 | "the linearisation of a non-monotone event is an input recorded in D, never chosen silently by a target" |
| 647 | "a rule that reports an intent whose obligations exclude no behaviour and holds it as an open hole"; "promotion by score is held as a ranking input, not as the admission rule" |
| 714 | "a definition that the Form IR cannot yet hold for size" |

The decks:

- Technical Appendix (532b173c), slide `appform`: "the `Dec` step split over two lines is a layout `rewrite` (W-SYN-1)".
- Send Deck (a32996f4), proof slide: "fitted by the unused-name rules (W-UNUSED-1)".
- The 76 fixtures read "76 fixtures, each reported with its rule: 62 with a path, 12 claims that differ, 2 notes".

## 9. What stays integrity, in compliance wording

These checks keep their behaviour. Their wording follows 4.9 and Definition 6, and it never uses the vocabulary of inputs. Each verdict keeps the record as found, names the first position at which it differs, and names a way forward.

| Locations | What is compared | Compliance wording | Way forward |
|---|---|---|---|
| `host.rs:1346, 1393-1395, 1406-1408`; `cli.rs:1158, 1486-1489`; 16:115, 16:311, 16:447; HC-21 replay on PR #31 | Replay against D (HC-8) | "replay differs at seq k: the Form derives X, D holds Y; D is kept as found; entries 0..k−1 are the record (host fault)" | Replay under the program that wrote D; CC-1 names it in the genesis entry. |
| `conc.rs:1957, 2284`; `rt.rs:1721`; `rt.js:1672-1673`; `rt.py:1764-1766` | Actor and supervisor replay, upgrades | `Differs { seq }`, with the supervisor position, beside the run outcome (4.6) | As above. |
| T38, HS-4; `rcedb-host-contract.md:96` | Verification on open | "Open verifies the chain; the verified prefix is the record; an entry rewritten, reordered or removed is reported at its position" | |
| `host.rs:1775` (`take_role`); 16:252 | Terms (HC-1, T41) | "term t is not above T; the admitter role stays with term T"; "an admitter whose barrier finds a higher term granted has been succeeded" | A new term is granted by the vote rule. |
| 16:222 | Follower append (E.2b) | "a follower appends only after a matching (prev_idx, prev_term); otherwise it answers with the index it holds, and the leader resends from there". The quotation at 16:283 stays as written. | The protocol's own repair. |
| 16:639 | Frame signatures | "a frame from a key that is not pinned is not counted" | Pin the key. |
| 14:660, 14:666 | A log under a changed supervisor tree (C.11) | "the log is not a record under this program's supervisor tree: the first position that differs is reported, and nothing is replayed" | Record it again under the new program; the conversion is specified. |
| 15:124, 15:354; `conc-gap-register.md:51`; paper 208 | Stamps and derived tags | "a stored stamp or tag that differs from its derivation is reported with both, as not this Form's record; the derived one stands" | Re-derive. |
| 15:178, 15:426 | CALM tags and n\* | "re-derived; a stored value that differs is reported, and the derived value stands" | Re-derive. |
| 17:399, 17:407, 17:957; `LEDGER.md` rows of 2026-10-24; `one-library-every-fabric.md:27` | Pins (F.5.3, W-PIN-1) | "the module's hash differs from its pin: reported with both; nothing is built from it" | Re-pin with `rce manifest`. Expanding a glob, qualifying a name and holding an import cycle as one group are paths of 8.2. |
| `cli.rs:237`; 12:339 | Layout check of `fmt` | "the formatter writes only a text with the same tokens; otherwise it writes nothing and reports the first token that differs" | Edit the text. |
| W-INT-1 (09:72; 12 fixtures of `spec/examples`; 13 conformance tests) | Stated claims | "example differs at line N: the example states o; the Form gives o′". The module has its Form and runs, and `rce check` exits 1 (COMP-17). | The author states o′ or changes the step. |
| `CORE-PROPERTIES.md:119` (`TargetMismatch`) | A target's trace | "the target's trace is not the Form's record at step k; the reference result stands" | Correct the target. |
| `intent-holes.md` D_I | Derivations | `differs:<obligation\|example\|spectest>` | Correct the derivation. |
| `form-ir-2.md:39`; `one-emitter.md:257, 259`; `rcedb-host-contract.md:51` | Readers of formats | "not a Form IR program: reported, nothing evaluated"; "a table row that does not parse is reported"; "a reader reports a header format it does not yet read and makes no claim about the log" | The reader grows. |
| `stdlib`, 76 rows: Ed25519, HPKE and AEAD tags, ML-KEM input checks, non-canonical CBOR, energy envelopes, HTTP requests carrying both Transfer-Encoding and Content-Length, energy samples | Byte integrity | "does not verify: answers the declared error (`BadTag`); nothing is derived from the bytes". These are declared results of library functions (4.11). The terms of art of FIPS 203 stay as written. | The sender sends bytes that verify. |
| `ide-energy` samples | Energy envelopes | "a sample that does not verify gives no reading (never 0 J)" | |
| Signed IDE receipts | Signed bodies | Stored chains verify as stored (section 10). | |
| Paper 352, 372, 380, 404, 425, 440, 454; decks (`tloop`, `appgate`) | Gate and engine fold | "found 15 of 18 proposals not to be laws, each with a counterexample, and installed none of them"; "reload reports the first entry whose hash or link does not match: that D is not the Form's record, and nothing past it is folded"; "every promotion, deferral, non-promotion with a counterexample and install is a mutation in an append-only log" | A proposal without a counterexample is installed. |

One case is a boundary, not an integrity check. W-MOD-1 and W-VIS-1, and the label and capability rules of 8.4, protect a declared contract of another module or authority (T36). Their path leaves the declaration to its owner, and the wording names the boundary: "`f` is private to module m; its `pub` surface offers g".

## 10. Hashed names, and how each hash is kept or re-pinned

The outcome words of δ are derived, so they enter no hash of a program entry. 6.1 defines `hash` and `state_hash` over seq, prev_hash, the positional event and the positional state (`interp.rs:3269-3273`). Renaming the outcome words therefore changes no hash of a δ entry in Form logs, example logs, host logs or emitted runtimes, and no conformance `trace_head`. The table lists every name that does enter a hash, and how each is treated under COMP-20.

| Name | Hashed in | Treatment |
|---|---|---|
| D `outcome` and `out` of a δ entry | nothing: derived (6.1) | Renamed. The trace bytes change in 7 of the 65 JSON files of `spec/traces` (12 kept entries), in the stream pin of `tests/traces.rs:204` and in `tests/refined.rs:111`. They are regenerated, and the format becomes `rce-trace/3`, as at the T7 erratum. |
| The outcome keyword in examples | the program hash (CC-1): unnamed items, examples among them, are hashed by their masked text (`apidiff.rs:180-202`), which contains the keyword (`unparse.rs:534-538`); and the module hash (F.5.3, `manifest.rs:43-47`) | 77 files and 99 example lines on main. Their program hashes, the emitted provenance line (CC-4) and the emit goldens are re-derived in the same commit, and `spec/proposals/compliance-repin.tsv` maps each earlier hash to its new one. No committed file pins a program hash. No module of `stdlib/lib` has an example with the keyword, so `stdlib/lib/MANIFEST.json` and the library hash are unchanged. Because `kept` is contextual (COMP-2), `stdlib/lib/data/cdc.rce` keeps its local `kept`. See open question 1. |
| Intents and string literals | module and program hashes | Reworded only where they describe an RCE outcome: 46 intent lines (38 in `conformance`, 4 in `spec`, 2 in `programs`, 1 in `book`, 1 in `ide`), `"#refused"` in 2 conformance tests, and `"ack refused"`. They go in the same map. Domain data stays as its author wrote it, for example `rejected: u64` in `apps/chat.rce` and `kid: "refused"` in `stdlib/tests/energy/energy_checks.rce:75-76`. |
| `host.MainEnd.outcome` `"refused: r"`; `host.Step.what` `"Refused(r)"` | the hash of the host entry, and every hash after it | The writer writes `"stopped: r"` and `"Kept(r)"`. No `.json`, `.jsonl` or `.out` fixture on main holds either entry. A stored D is kept as found: replay compares through a closed table in `host.rs` that reads each earlier spelling as the same kind (COMP-20 (c)). |
| `host.Refused` {cause, seq}; the `host.Bind` payload {"event":"Refused","kind":"host.Refused"}; the HD-4 shape | host entries (PR #31) | Renamed on the branch before it merges, to `host.StepKept`, {"event":"StepKept","kind":"host.StepKept"} and `StepKept(seq: u64, cause: text)`, so that no D written by main carries the earlier names. The cause words `fail`, `trap`, `fuel`, `size`, `value` and `depth` stay. Renaming after the merge would change the hash of every host D with a kept entry, a log-format erratum of the kind of the T7 change. |
| `host.Unfit` {event, why} | host entries (PR #31) | Named `host.Kept` on the branch before the merge; no D written by main carries `host.Unfit`. |
| The declared event `StepKept(seq, cause)` | its name is not hashed (positional event index); its payload is | HD-4 binds it by name. |
| Catalog `lowering` | the `rce-catalog/1` hash covers its entries | Becomes `{"pending": r}`, and `spec/checker/tests/brick/catalog.json` is regenerated in the same commit. `refused_files` is outside the hash and becomes `pending_files`. |
| IDE receipts: the `verdict` tags `refused` and `rejected`; the action `proposal.reject` | BLAKE3 over the canonical body, signed with Ed25519; `verify` re-serialises the body (`ide-receipt/src/lib.rs`) | `Verdict` gains `Fit` and `NotPerformed`, which the writer uses. The two earlier tags stay as variants that serialise to their own bytes, so every stored chain verifies; the writer never produces them. |
| Conformance clause ids `CHnn-M-<sha256[:8]>` | identity from content (`conformance/tools/extract_clauses.py`) | By the maps, 7 modal clauses contain the words (CH00-M-d5ed1b66, CH02-M-69371f07, CH03-M-8435434a, CH08-M-51f39d7d, CH09-M-82e65964, one in chapter 11 and one in chapter 12). They get new ids; their 409 references are rewritten from `conformance/clause-repin.tsv`; `id_set_sha256` is regenerated. |
| Book footers `body-sha256`; `INDEX.md`, `receipts.json`, `RCE-SPEC.md`, `spec-data.json`, `spec-tables.json` | generated hashes | Regenerated by `rce-book build` and `rce-spec-data`. The read receipt of chapter 9 ("Which rule refuses …", 09:122) is reworded; its answer, W-STEP-4, is unchanged. |
| Published records: `rce-corpus/1` (`forms.jsonl`, `negatives.jsonl`); registered `rce dim` runs (`dim-r1.jsonl`, key `refuses`, hashed sidecar) | source hashes and the sidecar | History: kept as published under their format identifiers. New rows use the words of section 6 under `rce-corpus/2` and `rce-dim/2`. |
| The `refused:` verdicts of D_I | the derivation log of `intent-holes.md`, not implemented | Written as `differs:<…>` from the start. |
| The engine's D.6 output `reject`; its D.11 status `refused`; the log mutation kind "refusal" | the D.6 `error` row is in Σd, so `state_hash` covers it | The private engine's owner renames them in one engine commit (SURFACE rule 4), with the reader table of COMP-20 for stored engine logs. Chapter 15 names the compliance words. |

Not hashed, but pinned byte for byte, and changed in the same commit as the names they mirror: the command-line strings; the run-log prefixes `Refused(` in the four runtimes and in `tests/emit_run.rs:965, 1025`; the conformance protocol (`expect`, adapter verdicts, harness strings, finding kinds); the fixture directories read by `bin/spec_check.rs`, `claim-counts.py`, `t7-equiv.sh` and `corpus.py`; the `rce dim` and `rce index` keys; the rcedb word `REFUSED E`; the IDE and playground strings; and the titles of section 7 and of book chapter 6, which appear 66 times as `section` values in `clauses.json`. `scripts/claim-counts.py` checks the counts in 00-overview against the summary line of `rce-spec-check`, so the two change together.

## 11. Migration

### 11.1 Order

1. **PR #31, before it merges:** the names of COMP-16 (`host.StepKept`, the declared event `StepKept`, `host.Kept`, the last already on the branch), with `tests/host_step_kept.rs`. It adds 401 hit lines over its merge base (spec/checker 250, spec/book 43, the generated `RCE-SPEC.md` 43, conformance 39, book 13, programs 7, tree-sitter 5, proposals 1), and 102 example lines in 80 files, 2 of them in the `trap` form.
2. **`openIE-dev/rce` main, one commit** (CONTRIBUTING rule 3), with its ledger row and a C.15 row. It carries the outcome words (COMP-1 to COMP-3, COMP-6, COMP-19, COMP-20), every input with its outcome (COMP-4, COMP-5, COMP-16, COMP-17), the paths (COMP-7, COMP-8, COMP-15) with the fixture moves, and the named forms, lifts and reserved constructs (COMP-9 to COMP-14), each with its fixtures, theorem-table rows and programs. Should the commit prove too large to review, it splits along these four groups, each a commit that leaves every check passing.
3. **`research-openie-web`, one commit**, built from the rce commit.
4. **The three decks**, one republish each.
5. **`openie-research-experiments`, one commit**: a README line stating that published records stay as found, and new rows from the corpus generator.
6. **The engine**, one commit by its owner.

The bulk of the change is mechanical. A token-aware rename map, which extends `scripts/rename-reserved.py`, and `rce fmt fit` apply it. Prose and messages are reworded by hand from section 6. The same discipline of tool-driven change in one repository is documented for interface migration [Cox 2011], for large-scale refactoring [Wright et al. 2013] and for atomic changes in a single repository [Potvin and Levenberg 2016].

### 11.2 `openIE-dev/rce` main

Hit lines at d2b28e76, with the number of files in parentheses. The class splits are the maps'.

| Area | Lines (files) | How it moves |
|---|---|---|
| `conformance` | 3,633 (446) | 2,807 lines in generated `reports/*.json`, regenerated by the harness. Of the 826 maintained by hand, by the maps (i) 459, (ii) 338, (iii) 29: 346 `expect: reject` tests become 329 `fit`, 13 `differs` and 4 `ok` with a note; 91 `expect: accept` become `ok`; 157 files `*-reject.rce` and 9 other file names move; 5 `# outcome:` lines and the outcome examples change; 7 clause ids and their 409 references are re-pinned; `INTERPRETING.md`, the adapters, `harness/run.py` and `tools/*.py` change. |
| `spec/checker` | 715 (114): `src` 395 (43), `tests` 318 (69), 2 others | Messages; the `REFUSED` constant; verdict words; the identifiers of 6.8; the behaviours of 8.1 and 8.2; the emitted runtimes `emit/rt.{rs,py,js,scala}`, whose stop words change and whose folds continue. The largest files are `cli.rs` 56, `conc.rs` 28, `emit/rt.rs` 25, `rt.py` 25, `rt.js` 23, `rt.scala` 21, `dim.rs` 21, `v09.rs` 19, `bin/conform.rs` 19, `interp.rs` 18, `bin/spec_check.rs` 15, `host.rs` 12, `parse.rs` 11 and `check.rs` 10. 57 test functions are renamed; `tests/nameless.rs:75-77` is corrected; `tests/fit.rs`, `fit_reports.rs`, `trap_texts.rs`, `named_forms.rs`, `reserved.rs`, `emit_coverage.rs` and `examples_stable.rs` are added. |
| `spec/book` | 358 (19): chapters 00 to 10 97, 11 to 17 259, `INDEX.md` and `receipts.json` 2 | Reworded by hand, chapter by chapter, from section 6, with the path column of section 7, the `forms` table and "claimed for" column of section 8, and F.2's table. Chapter 16 takes in the PR #31 lines when that PR has merged. |
| `spec/RCE-SPEC.md` | 356 (1) | Generated by `rce-book build`. |
| `ide` | 291 (74) | (i) 276, (iii) 11, third-party 4. Receipts as in section 10; the strings and keywords of `rce-mcp` and `rce-graph3d`; `mei-agent`; `acp-client`; `rce-memory` and `rce-guest`. |
| `stdlib` | 222 (97), plus 1 binary vector | (i) 64, (ii) 11, (iii) 64, third-party 83. Library comments are reworded, which changes no hash; 8.10; the test helpers `rejects()` and `refused_*`; 1 outcome example in `tests/rand/det_checks.rce`. `MANIFEST.json` is unchanged, as `stdlib/scripts/check.sh` and `rce manifest-check` confirm. |
| `book` | 120 (50), plus 43 paths | Chapter titles; 8 `listings/chNN/refused/` directories; outcome examples; run outputs re-blessed (`counter_run.out`, `account_run.out`, `cart_run.out`, `counter_js.out`, `counter_python.out`, `door_refused.out`); `scripts/check-listings.sh:8-50`. |
| `spec/lean` | 110 (24) | Section 13. |
| `spec/fuzz` | 90 (9) | The wording of the near-miss counts. |
| `programs` | 79 (36) | 37 outcome examples in 20 files; `money/refused/` becomes `money/fit/` (`decimal.rce`, a `lift` to `core.exact`); `scripts/check.sh`. |
| `spec/proposals` | 65 (15) | Wording; the D_I verdicts; the "What it refuses" headings; this proposal's ledger row. |
| `docs` | 56 (9) | (i) 12, (ii) 41, third-party 3. The 33 paradigm rows of 8.9; figure sources regenerated. |
| `tree-sitter` | 47 (10), plus 1 binary | `grammar.js` with the rule of COMP-2; `scanner.c`; `queries/highlights.scm`; the corpus; regenerated with `tree-sitter generate --js-runtime native` (`parser.c` 26, `grammar.json` 4, `node-types.json` 3). |
| `scripts` | 31 (6) | `claim-counts.py` (the summary regex and the ledger section); `corpus.py`; the new `vocab.py` and `vocab-allow.tsv`. |
| Others | 66 | `spec/examples` 10, `apps` 9, `spec/LANGUAGE-REVIEW.md` 9 (history, kept), `spec/traces` 7, `rcedb` 6, `spec/wasm-test` 6, `CONTRIBUTING.md` 6, `lib` 5, `README.md` 5, `playground` 4, `examples` 3, `spec/conc06b` 2, `spec/README.md` 2, `.githooks` 2, `spec/lift` 1, `spec/examples-0.9` 1, `spec/data` 1, `spec/checker-wasm` 1, `SURFACE.md` 1. |
| **Total** | **6,254 lines in 950 text files, and 3 binary files** | Generated, moving by regeneration: 3,199. Third-party text and dated history, staying: about 105. Moved by script or by hand: about 2,950. |
| Paths | 333 paths whose names contain the words: conformance 166, `spec/examples-0.9` 88, `book/listings` 43, `spec/examples` 14, `spec/conc06b` 9, `spec/conc06` 8, `stdlib/tests` 2, `programs/money` 2, `spec/checker` 1; plus the other 62 files of `ill_typed/` and `inadmissible/` | Moved to the directories of 6.5. The 2 third-party vector names in `stdlib/tests/crypto` stay. |

### 11.3 The checks the commit passes

- `cargo test --release` and `cargo clippy --release --all-targets` in `spec/checker`;
- `rce-spec-check` built with cargo's `--features tree-sitter`;
- `rce-book build`, which must leave no difference;
- `rce-spec-data check`;
- `scripts/claim-counts.py check`;
- `lake build` and `scripts/lean-audit.sh`;
- the conformance harness against the reference adapter;
- `stdlib/scripts/check.sh` with `rce manifest-check`;
- `programs/scripts/check.sh`;
- `book/scripts/check-listings.sh`;
- `scripts/t7-equiv.sh`;
- the `ide` tests and the playground build;
- `scripts/vocab.py check`.

The summary line of `rce-spec-check`, today `checker: accepted 62/62 examples, refused 12/12 syntax rejects, refused 64/64 ill-typed or inadmissible (16 of them inadmissible)`, becomes:

```
checker: 64/64 examples in Form (2 with a note); 12/12 syntax, 46/46 type and 4/4 admission fixtures reported with rule and path; 12/12 claim fixtures reported as differing
```

00-overview and `claim-counts.py` move with this line.

### 11.4 The migration claim

Every example that holds at d2b28e76 holds after the change, with the same log. `tests/examples_stable.rs` checks it. The argument:

- An example that holds today has no kept outcome before its last event; otherwise it would be W-INT-1.
- So its fold is unchanged.
- A last event with the kept outcome has the same entry hash, because the outcome is derived and not hashed.

### 11.5 `research-openie-web`

One commit, at 85bc444 today:

| Item | Size | Action |
|---|---|---|
| The paper, `src/content/papers/rce.md` | 34 lines, 52 occurrences: by the maps (i) 30, (ii) 8, (iii) 12, terms of art 2 | Restated by sections 6, 8.12 and 9. For example, 291: "T8 atomicity: an event with the kept outcome leaves the state unchanged"; 687: "every program has its place in the Form: as written, or fitted, lifted or grown into it". |
| The derived copy `scripts/i18n/.rce_en_body.md` | 34 lines | Regenerated. |
| The 8 translations `src/content/translations/*/rce.md` | | Translated again from the revised text. |
| RCE components: `RceJepaPaper.astro` 8, `RceJepaProposer.astro` 7, `RceJepaConvergence.astro` 3, `RceJepaThesis.astro` 1 | 19 lines | "accepts or rejects each candidate" becomes "gives each candidate its verdict: `ok`, or the rule with its path". The quotations and the description of other systems stay as written. |
| RCE pages and data: `rce/spec/[page].astro` 4, `about/rce-jepa.astro` 2, `rce/spec/index.astro` 2, `about/rce.astro` 1, `rce/download.astro` 1, `data/rce-facts.json` 2 | 12 lines | Restated. |
| The specification mirror `spec-src/rce/spec.json` | 398 occurrences | Regenerated from the rce commit by `spec-src/rce/gen.mjs`, whose own 3 lines are restated. |
| `public/pdfs/rce.pdf` | | Regenerated by `scripts/generate-pdfs.sh`. |

The site's checks (`lint-order.mjs`, `lint-claims.mjs`) run in `pnpm build`. Other papers on the site use the words in unrelated senses and are out of scope.

### 11.6 The decks

By the maps, 67 occurrences: the Technical Appendix (532b173c) 39, the Pitch (5f2a1c35) 18 and the Send Deck (a32996f4) 10, of which (i) 61, (ii) 1 and (iii) 5. Each deck is republished with the wording of section 6 and 8.12. The image on slide `hm1` and its alt text carry "refuse at zero", which becomes "keep at zero".

### 11.7 `openie-research-experiments`

At 79f5057 the `rce/` subtree has 3,784 hit lines in 114 files. The published form-space results (`rce/bench/form-space/results/negatives.jsonl` 2,941 lines, `forms.jsonl` 47, `summary.json` 127) and the registered `rce dim` runs (`dim-r1.jsonl` 49, `pilot-2026-10-22.jsonl` 43) are records, and the surveys and notes are dated history. All stay as found, and a README line says so. New rows come from `scripts/corpus.py` in rce under `rce-corpus/2`. The `agency-run` results outside `rce/` use the words in an unrelated sense.

### 11.8 The former satellite repositories

`rce-stdlib`, `rce-conformance`, `tree-sitter-rce`, `rce-book` and `rce-ide` moved into rce main on 23 October 2026. Their pointer commits (12dfee9, 84c3af3, 02b3bc7, db097b6 and f48d079) are on branch `claude/great-bardeen-jxdm9n` of each. Their content is the history of the move and is not edited; local clones are stale copies.

### 11.9 The engine

The private engine's owner renames the D.6 output, the D.11 status, the log mutation kind "refusal" and the test `fold_rejects_wrong_state` in one commit, under COMP-20. Until then the paper cites the test by its current name, as data.

## 12. Composition before addition

CONTRIBUTING rule 2 asks a proposal to show the same program written with existing constructs before it adds a construct. The criterion is macro-expressibility: a form that is a local translation into existing constructs adds no expressive power, so it is a path and not a construct [Felleisen 1991].

| Need | Held by existing constructs | Added |
|---|---|---|
| The kept outcome | The second case of δ in 6.4 item 4, and T8 | Nothing: one keyword replaces another, and the reserved list is one word shorter |
| A frame-only input | Applied, with the identity update of the frame rule | Nothing: W-STEP-2 becomes a note |
| Every event folded | T3, and δ total (T2) | Nothing |
| Outcomes of intermediate events in an example | Prefix examples: `example Dec -> kept: Underflow` and `example Dec, Inc -> 1` | Nothing |
| A payload outside a range or refinement | The trap of 5.1 and 6.4 item 3, with E read at shapes | Nothing: a definitional widening of E |
| An input that names no event, or has another shape | `Input(data)` (`rce host serve`); HC-21 and HC-25 bindings; PR #31's `host.Kept` (formerly `host.Unfit`), which already holds the host's whole value and the reason | One extension: `host.Kept`, with a `data` alternative for bytes that name no type |
| A late `fn main` input | The `late` mark of E.2a (`host.Input {late: true}`) | One field, `asked`, on `host.MainInput`, beside `late` |
| Placement fallback | `host.Placed` and the CPU reference (HC-12, HC-13) | Nothing |
| A query whose step would change state | Queries already append nothing (HS-8) | Nothing |
| Delivery of a kept outcome | HC-21 of PR #31 | Nothing: a respelling |
| Run outcomes | `Done`, `Deadlock`, and the `Stopped(supervisor S)` log entry | Nothing: a respelling |
| Replay divergence | The existing replay check | `Differs { seq }`, kept apart from run outcomes, which `Outcome::Refused("replay diverged")` conflates today |
| Determined paths | `rce fmt` (10.6) and the existing `scripts/fix-*.py` and `rename-reserved.py` | A tool mode, `--fit`, and `data.fit` in diagnostics; no language construct |
| Lifts of other languages' constructs | The semantics tables of `spec/lift` (the Go probe and `tests/lift_go.rs`) | Data: `spec/lift/table.json` |
| A default bound for recursion | `bound` and LOOP_LIMIT (6.7) | Its use for recursion cycles |
| Named forms | A Result return (W-TX-1); `bound` (W-REC-1); a run-time `check` (W-REF-1, W-CON-1); Option; traits; `iter`; actors and regions; `extern "C"`; logged choices (T21, T32); `Prompt.encode`, `release`, `endorse`, `ct_choose` | Nothing: each form is a rule's reading of existing constructs, recorded in the `forms` table and the "claimed for" column |
| Paradigms of 8.9 | Regions, actors, `extern "C"`, Option, `kind` with `match`, traits, `iter`, tasks and channels, list comprehensions with `from_list`, `use`, fuel bounds | Data: one program per row in `spec/examples/forms/` |
| Reserved constructs | F.12 | A column (a), (b) or (c) |
| Verdicts on records | The existing replay, verify, pin and layout checks | Nothing: wording |
| Exit status | The three codes of 10.1 | Nothing: a redistribution |
| Ledger decisions | The reasons of rule 5 | The word `composed` |

The language gains no construct. The host gains one entry name with one payload alternative, and one field. Bijective renaming of a terminal keeps the shape of the grammar; moving `kept` to the contextual words changes the lexer's treatment of one word, so `rce-spec-check` re-establishes 0 LALR(1) conflicts (T10).

## 13. Theorems and Lean names

### 13.1 Theorems

No theorem changes its content, and no theorem loses its claim for a program that has a Form today (COMP-21).

| Theorem | Effect |
|---|---|
| T1 | Unchanged. The logged choices of the named forms (W-ND-1, W-LVAR, W-DL-1) keep a run determined by D (T21). |
| T2, T4 | Unchanged. T4's "no stuck states other than defined outcomes" already states the principle. W-STEP-2's message is corrected, and the `bound` path of W-REC-1 gives T2 by 6.7. |
| T3 | Unchanged. COMP-5 brings `rce run`, `rce trace` and the serve loop into agreement with T3 for logs that hold kept entries, which today they stop before. |
| T6 | Wording: write-while-borrowed, overlapping `&mut`, dangling borrows and reads after a partial move "have no typing derivation as written; W-OWN-1, W-OWN-2 and W-OWN-3 report each with its path". Foreign functions stay outside the statement, so the `extern "C"` form of W-SAFE-1 keeps T6 for the RCE side. |
| T7 | Unchanged for every hashed form. `Kept` is a record of a type the module does not declare and keeps its name, as `Refused` does. The derived outcome names a declared error by name (open question 3). |
| T8 | Restated: "**T8 Atomicity.** If e is kept in s (Definition 2), then δ₁(s, e) = s." The proof `atomicity` is unchanged apart from the constructor name. |
| T9 | The emitted runtimes print the words of section 6, and the differential tests compare runs folded to the end (COMP-5). |
| T10 | Re-established by `rce-spec-check` (section 12). |
| T12 | The comment "`none` when refused" in `RCE/Linear.lean` reads "`none` when the count has no value (W-LIN-1)". |
| T16, T28l, T28p, T29, T47, T48 | Gain the "claimed for" scope of COMP-9: T16 for `par` with disjoint footprints, T28l for the linear session form, T28p for the acyclic wait form, T29 for deterministic LVars, T47 for transactional calls, T48 for definitions with a written measure. Each is claimed for the form it was proved or stated for, so none is weakened. |
| T35 | Restated: "… an event whose payload or resulting state breaks a predicate has the kept outcome, with the state unchanged, and is in D (COMP-4)". New lemma `admit_kept_unchanged`: the checked δ applied to a payload that fails `admit` is the identity. It follows from `checked` as `refine_kept_unchanged` does. |
| T37 | `host.StepKept` and `host.Kept` are host entries, and `host_entry_identity` (`RCE/Host.lean:24`) is stated for every host entry, so no new proof is needed. `tests/host_contract.rs` re-folds logs that hold them. |
| T38 to T41 | Wording only (section 9). T38's "cuts" already describes a verdict on a record. |
| T42 | "including the refusal outcome" reads "including the kept outcome and its reason". |
| T46 | "otherwise the call's event is refused with the fixed record" reads "otherwise the call's event has the kept outcome, with the fixed contract record". |
| T28, T29 (proposals) | "refusing configuration" reads "stuck configuration". |

New lemma `outcome_dichotomy` in `RCE/Form.lean`: for every P, s and e, either (delta P s e).2 = [Out.kept x] for exactly one x and (delta P s e).1 = s, or every element of (delta P s e).2 is `Out.emit v`. It is Proposition 1, and COMP-1 cites it.

### 13.2 Lean names

The names move in the same commit as `spec/lean/audit/Audit.lean`, the theorem table of section 8, `scripts/lean-audit.sh`, `tests/lean_*.rs`, and the citations at 05:102, 10:34 and in the proposals. `lake build` and `scripts/lean-audit.sh` re-establish the audit count with standard axioms only.

| File | Current | Compliance |
|---|---|---|
| `RCE/Form.lean` | `inductive Refusal` (`user`, `trap`) at line 17; `Refusal.rename` at line 314 | `Reason`, with the constructors unchanged; `Reason.rename` |
| `RCE/Form.lean` | `Out.refused` (line 219); `SRes.ref` | `Out.kept`; `SRes.kept` |
| `RCE/Form.lean` | `atomicity` (line 255, T8) | unchanged; its hypothesis reads `Out.kept x ∈ (delta P s e).2`, and its docstring "a kept outcome leaves every region unchanged" |
| `RCE/Bound.lean` | `SCfg.refused` (line 102); `refused_nostep` (149); `SStep.seqRef`, `SRun.seqRef` (122, 171); `Except Refusal` | `SCfg.kept`; `kept_nostep`; `SStep.seqKept`, `SRun.seqKept`; `Except Reason` |
| `RCE/Core/Events.lean` | `StepR.refuse` (line 18), `DeltaR.refuse`; `deltaS_refuse_atomic` (69) | `StepR.kept`, `DeltaR.kept`; `deltaS_kept_atomic` |
| `RCE/Core/Events.lean`, `Driver.lean`, `Lang.lean` | `.oof` (out of fuel) | unchanged; documented as the kept outcome with cause `fuel` |
| `RCE/Core/Events.lean`, `Driver.lean`, `Lang.lean` | `.stuck` | unchanged; unreachable for typed terms by T4 |
| `RCE/Core/Log.lean` | the comment "a refused event leaves the state unchanged" | "a kept event leaves the state unchanged" |
| `RCE/Refine.lean` | `refine_refused_unchanged` (line 108); the docstring of `admit` | `refine_kept_unchanged`, and the new `admit_kept_unchanged`; "an event whose payload fields fail their predicates has the kept outcome before any step runs" |
| `RCE/Borrow.lean` | `write_while_borrowed_refused`, `overlapping_mut_refused`, `dangling_refused`, `read_after_partial_move_refused` (lines 131-162) | `write_while_borrowed_no_typing`, `overlapping_mut_no_typing`, `dangling_no_typing`, `read_after_partial_move_no_typing` |
| `RCE/Ownership.lean` | `use_after_move_refused` (91), `mut_alias_refused` (103) | `use_after_move_no_typing`, `mut_alias_no_typing` |
| `RCE/NameFree.lean` | the `.refused` case (lines 37-38, 72) | `.kept` |
| `obligations/Obligations.lean` | `CallResult.refused` (lines 60, 76) | `CallResult.kept` |
| `RCE/Emit.lean`, `EmitPrint.lean`, `emit-test/Gen.lean` | `Refusal::User`, `Refusal::Trap` in emitted Rust; `pRef` (`Gen.lean:47`) | `Reason::User`, `Reason::Trap`; `pReason`; the emit-test vectors are regenerated |
| `CoreMain.lean` | the JSON key `"refused"` (lines 206-207); `"comptime fuel exhausted"` for `.oof` at run time (207, 229) | `"kept"`; `"fuel exhausted"` |
| `RCE/Admit.lean:111`; `RCE/Discharged.lean:72` | comments "refused" | "not appended: a higher term was promised"; "a log of another length differs" |

The suffix `_no_typing` states what those theorems prove: the typing judgement has no derivation for the program as written. The program's place in the Form is then given by its rule's path.

## 14. Open questions

1. **Program hashes independent of spelling.** CC-1 hashes unnamed items, examples among them, by their masked text, so respelling a keyword changes program hashes. The alternative encodes an example's outcome kind, and keywords in general, by position, as T7 does for names. Then no later respelling would move a program hash, at the cost of one more change of the `rce-program/1` identifier. The same question arises for the outcome words and trap texts inside the hashed payloads of `host.MainEnd` and `host.Step`: carried positionally, they would leave the hashes of D unchanged under a later rewording, at the cost of payloads that are read through the table of 6.2. Whether to make either change in the commit of 11.1 step 2, or separately, is open.
2. **Evaluating an open position as a hole.** `rce run` could hold an unresolved name as a hole whose evaluation is the kept outcome `trap("hole: NAME")`, typed at the JSON-shaped value type [Siek and Taha 2006]. Whether T6 holds for a hole of unknown type depends on the typing rules of `intent-holes.md`, and is open.
3. **A name-free derived outcome.** The derived `outcome` and the λ record name a declared error by name. They are not hashed, so T7 holds either way. `rce equiv` and `rce dim control` already compare a kept outcome by the error's declaration index (10.11). Whether D's derived field carries the index, the name, or both is open: readability favours the name, uniformity with the hashed forms the index.
4. **Universality for each source language.** The ASM theses give step-for-step simulation for algorithms that satisfy their postulates [Gurevich 2000; Blass and Gurevich 2003], and COMP-12 checks round trips on a corpus. For source constructs that interact without bound inside one step, such as a blocking call inside an expression, it is open whether behavioural equivalence (6.10) holds between the source and its lift, or only equivalence up to the logged answers and the `fuel` outcome. Its general statement, a proof for any one language, and the criterion for when a source language counts as held (agreement of `shape_ctl` with a Form written by hand, as the Go probe shows; replay of the source program's own tests through the lifted Form) are open.
5. **Claims of named forms not yet proved.** The main case is the multi-shot handler composition of 8.9: T6 for the clones of captured state that each logged resume makes, the ground of the ledger row of 22 October 2026. Until each is proved, the "claimed for" column reads "stated".
6. **The size of D under adversarial input.** COMP-4 records every offered input. SIZE_LIMIT bounds the size of one `host.Kept` entry, but not their number. Whether the kept entries of one admission window may be compacted into one entry that carries a Merkle root of their bytes (HS-12), without losing replay equality (T37, T40), is open.

## References

- Aho, A. V., and Peterson, T. G. 1972. A minimum distance error-correcting parser for context-free languages. *SIAM Journal on Computing* 1(4), 305–312.
- Barik, T., Smith, J., Lubick, K., Holmes, E., Feng, J., Murphy-Hill, E., and Parnin, C. 2017. Do developers read compiler error messages? *Proceedings of ICSE 2017*.
- Becker, B. A., Denny, P., Pettit, R., et al. 2019. Compiler error messages considered unhelpful: the landscape of text-based programming error message research. *ITiCSE Working Group Reports 2019*.
- Bierman, G., Russo, C., Mainland, G., Meijer, E., and Torgersen, M. 2012. Pause 'n' Play: formalizing asynchronous C♯. *Proceedings of ECOOP 2012*, LNCS 7313.
- Blass, A., and Gurevich, Y. 2003. Abstract state machines capture parallel algorithms. *ACM Transactions on Computational Logic* 4(4), 578–651.
- Blass, A., Gurevich, Y., Rosenzweig, D., and Rossman, B. 2007. Interactive small-step algorithms I: axiomatization. *Logical Methods in Computer Science* 3(4:3).
- Capretta, V. 2005. General recursion via coinductive types. *Logical Methods in Computer Science* 1(2).
- Claessen, K., and Hughes, J. 2000. QuickCheck: a lightweight tool for random testing of Haskell programs. *Proceedings of ICFP 2000*.
- Cox, R. 2011. Introducing Gofix. *The Go Blog*, 15 April 2011.
- Danvy, O., and Nielsen, L. R. 2001. Defunctionalization at work. *Proceedings of PPDP 2001*.
- Dunlap, G. W., King, S. T., Cinar, S., Basrai, M. A., and Chen, P. M. 2002. ReVirt: enabling intrusion analysis through virtual-machine logging and replay. *Proceedings of OSDI 2002*.
- Felleisen, M. 1991. On the expressive power of programming languages. *Science of Computer Programming* 17(1–3), 35–75.
- Garcia-Molina, H., and Salem, K. 1987. Sagas. *Proceedings of SIGMOD 1987*.
- Gray, J. 1981. The transaction concept: virtues and limitations. *Proceedings of VLDB 1981*, 144–154.
- Gurevich, Y. 2000. Sequential abstract-state machines capture sequential algorithms. *ACM Transactions on Computational Logic* 1(1), 77–111.
- Henriksen, T., Serup, N. G. W., Elsman, M., Henglein, F., and Oancea, C. E. 2017. Futhark: purely functional GPU-programming with nested parallelism and in-place array updates. *Proceedings of PLDI 2017*.
- Igarashi, A., Thiemann, P., Vasconcelos, V. T., and Wadler, P. 2017. Gradual session types. *Proceedings of the ACM on Programming Languages* 1 (ICFP).
- International Telecommunication Union. 2002. Recommendation ITU-R TF.460-6: Standard-frequency and time-signal emissions.
- Kuper, L., Turon, A., Krishnaswami, N. R., and Newton, R. R. 2014. Freeze after writing: quasi-deterministic parallel programming with LVars. *Proceedings of POPL 2014*.
- Lamport, L. 1994. The temporal logic of actions. *ACM Transactions on Programming Languages and Systems* 16(3), 872–923.
- Landin, P. J. 1966. The next 700 programming languages. *Communications of the ACM* 9(3), 157–166.
- Marceau, G., Fisler, K., and Krishnamurthi, S. 2011. Measuring the effectiveness of error messages designed for novice programmers. *Proceedings of SIGCSE 2011*.
- Mealy, G. H. 1955. A method for synthesizing sequential circuits. *Bell System Technical Journal* 34(5), 1045–1079.
- Meyer, B. 1992. Applying "design by contract". *IEEE Computer* 25(10), 40–51.
- Microsoft. 2022. *Language Server Protocol Specification 3.17* (diagnostics and code actions).
- Moggi, E. 1991. Notions of computation and monads. *Information and Computation* 93(1), 55–92.
- Ongaro, D., and Ousterhout, J. 2014. In search of an understandable consensus algorithm. *Proceedings of USENIX ATC 2014*.
- Padioleau, Y., Lawall, J., Hansen, R. R., and Muller, G. 2008. Documenting and automating collateral evolutions in Linux device drivers. *Proceedings of EuroSys 2008*.
- Plotkin, G., and Pretnar, M. 2009. Handlers of algebraic effects. *Proceedings of ESOP 2009*, LNCS 5502.
- Potvin, R., and Levenberg, J. 2016. Why Google stores billions of lines of code in a single repository. *Communications of the ACM* 59(7).
- Rabin, M. O., and Scott, D. 1959. Finite automata and their decision problems. *IBM Journal of Research and Development* 3(2), 114–125.
- Reynolds, J. C. 1972. Definitional interpreters for higher-order programming languages. *Proceedings of the ACM Annual Conference 1972*.
- RFC 8785 (JSON Canonicalization Scheme), RFC 8949 (CBOR), RFC 9112 (HTTP/1.1). Internet Engineering Task Force.
- Rust compiler development guide. Errors and lints: suggestions and their applicability.
- Rutten, J. J. M. M. 2000. Universal coalgebra: a theory of systems. *Theoretical Computer Science* 249(1), 3–80.
- Sassaman, L., Patterson, M. L., Bratus, S., and Shubina, A. 2011. The halting problems of network stack insecurity. *;login:* 36(6).
- Schneider, F. B. 1990. Implementing fault-tolerant services using the state machine approach: a tutorial. *ACM Computing Surveys* 22(4), 299–319.
- Shapiro, M., Preguiça, N., Baquero, C., and Zawirski, M. 2011. Conflict-free replicated data types. *Proceedings of SSS 2011*, LNCS 6976.
- Siek, J. G., and Taha, W. 2006. Gradual typing for functional languages. *Scheme and Functional Programming Workshop 2006*.
- Stefik, A., and Siebert, S. 2013. An empirical investigation into programming language syntax. *ACM Transactions on Computing Education* 13(4).
- Su, Z., and Wassermann, G. 2006. The essence of command injection attacks in web applications. *Proceedings of POPL 2006*.
- Thomson, M., and Schinazi, D. 2023. Maintaining robust protocols. RFC 9413, Internet Engineering Task Force.
- Tofte, M., and Talpin, J.-P. 1997. Region-based memory management. *Information and Computation* 132(2), 109–176.
- Turner, D. A. 2004. Total functional programming. *Journal of Universal Computer Science* 10(7), 751–768.
- Wadler, P., and Findler, R. B. 2009. Well-typed programs can't be blamed. *Proceedings of ESOP 2009*.
- Wright, H. K., Jasper, D., Klimek, M., Carruth, C., and Wan, Z. 2013. Large-scale automated refactoring using ClangMR. *Proceedings of ICSM 2013*.
