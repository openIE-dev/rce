# The 79 clauses without a test, sorted by the evidence that decides them

Work item 4.5. Each clause that `tools/validate.py` reported as untested is
sorted by what can decide it, read off its quoted text:

- **a**: decidable on a finite trace by the reference. One program, one run or
  one command decides it, so a conformance test is the evidence.
- **b**: a statement over every program or every value. No finite set of runs
  proves it, but every counterexample is a finite trace, so a test can refute
  it. The evidence is a bounded exhaustive search or seeded generation with its
  bounds stated, or a proof; a Lean proof on a model is cited as evidence about
  the model, not about the implementation.
- **c**: a physical claim about the host. The evidence for a value is a
  measurement with its unit, device and uncertainty, labelled by LL-2. A test
  without particular hardware checks form, units, ranges and order; any value
  check needs hardware.
- **d**: not a duty on an implementation (a derivation, a description, an
  instruction to the book's authors, or a removed rule). The proposal is a
  change of classification in `tools/extract_clauses.py`; no rule is weakened.

## The table

| Clause | Class | Reason (quoted) | Status |
|---|---|---|---|
| CH00-M-263cf831 | a | "must still be accepted by the reference checker": one accept per SURFACE.md program | open: accept tests |
| R10 | a | "everywhere else each is a NAME": accept a program naming a value `always` | open: accept test |
| CH07-M-9b2d649e | b | "the bytes must decode to that value and to no other": injectivity over all values; seeded round trips falsify | open: property test |
| CH07-M-dec2665b | a | "an RFC 8785 library does not produce these bytes": one value where rules 2, 3, 4, 6 differ decides it | open: trace test |
| CH08-M-30a83208 | d | "Changing the outcome to the law's turns the line into the example": states an effect of an edit; no implementation duty | propose: descriptive |
| W-MUT-2 | a | "`rce mutants` (10.14) prints each one with its edit": finite cli output | open: cli test |
| CH08-M-cad7de4b | d | "They are the levels of the RCE Compile study's experiment E3": a lead-in; the levels below carry the duties | propose: descriptive |
| CH08-M-abd54dca | a | "their number must be the shape's element count times the element's size": one byte count | open: change test |
| T42 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T43 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T44 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T45 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T46 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T47 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T48 | b | "proved on a model of the measure rule: a step that strictly lowers a n..."; Lean: `t48` (`RCE/Discharged.lean`) | evidence cited (proof on a model, not the implementation); property test still open |
| T49 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T50 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T51 | b | "proved on a log model: a log with a revoke entry for a caretaker admit..."; Lean: `t51` (`RCE/Discharged.lean`) | evidence cited (proof on a model, not the implementation); property test still open |
| T52 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T53 | b | "proved on a model of programs that ask an oracle: replay from the log ..."; Lean: `t53` (`RCE/Discharged.lean`) | evidence cited (proof on a model, not the implementation); property test still open |
| T54 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T55 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T56 | b | "stated, unproved (Appendix F.9; F.12 lists what is implement": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T57 | b | "proved for the morphism equation (one-step unfolding of the behaviour ..."; Lean: `beh_cons` (`RCE/Behaviour.lean`) | evidence cited (proof on a model, not the implementation); property test still open |
| T58 | b | "proved on the Form of `RCE/Form.lean`, for one program and for two: a ..."; Lean: `bisim_beh`, `beh_bisim`, `behEq_iff_bisim`, `bisim2_beh`, `beh_bisim2` | evidence cited (proof on a model, not the implementation); property test still open |
| T59 | b | "proved over an explicit finite alphabet (the explored alphabet of 10.1..."; Lean: `words`, `AgreeAt`, `firstDiff`, `firstDiff_disagrees`, `firstDiff_le`, `firstDiff_none`, `trace_ball`, `ball_trace` | evidence cited (proof on a model, not the implementation); property test still open |
| T60 | b | "stated, unproved (a model proof by partition refinement is t": a statement over every program; a counterexample is a finite trace | open: property test (bounded search) |
| T61 | b | "tested for renaming only (`tests/dim.rs`, `shape_ctl_is_name_free`, against `rce fmt namel" | partial evidence (Rust test); property test open |
| T62 | b | "proved on a model (`RCE/LogPO.lean`, spec/proposals/partial-order-log...."; Lean: `commute`, `fold_swap`, `fold_perm_eq` | evidence cited (proof on a model, not the implementation); property test still open |
| T63 | b | "proved on the Form of `RCE/Form.lean` (`RCE/Law.lean`): with a complet..."; Lean: `reachBy`, `ClosedOn`, `reachBy_mem`, `law_always_holds`, `law_step_holds`, `law_always_fails`, `law_alway | evidence cited (proof on a model, not the implementation); property test still open |
| CH11-M-efbb7a78 | b | "The four Form runtimes ... must agree on the new ops": agreement over all inputs; differential generation falsifies | open: differential property test |
| T28l | b | "proved on a calculus of bodies (send, receive, close, move, branch over message protocols); the checker agrees..." | evidence cited (proof on a model, see 14-draft06 table row); property test still open |
| T28d | b | "proved on a model of networks (processes blocked on one endpoint or finished); the tree is enforced by W-SES-5..." | evidence cited (proof on a model, see 14-draft06 table row); property test still open |
| T28p | b | "proved on a model of networks (rendezvous, spawns as one-step channels); priorities computed by relaxation are..." | evidence cited (proof on a model, see 14-draft06 table row); property test still open |
| T28g | b | "proved on a synchronous calculus of global and local types with messages and binary choice; the checker's proj..." | evidence cited (proof on a model, see 14-draft06 table row); property test still open |
| T28a | b | "proved on an asynchronous calculus of global and local types with binary choice and one FIFO queue per ordered..." | evidence cited (proof on a model, see 14-draft06 table row); property test still open |
| HC-5a | a | "The printed target is the installed target": one run's printed line | open: host test |
| HC-10a | a | "delivers exactly the bytes it received, in order": a non-UTF-8 delivery and its D entry | open: needs a byte-input harness op |
| HC-10b | a | "Typed inputs are inputs": one admitted stdin line | open: host test |
| HC-21 | a | "the admitter appends the input with the state unchanged, then host.StepKept" | tested: hc-21-kept-step-is-input |
| HC-22 | a | "A restart is an input, and ids are never reused": two opens of one D | open: two-command host test |
| HC-23 | a | "a request that names an address not granted ... fails": one denied request | open: needs net harness op |
| HC-24 | a | "HTTP requests are inputs": one serve run | open: needs http harness op |
| HC-25 | a | "The syntax is RFC 9112 with the strict choices": finite malformed requests | open: needs http harness op |
| HC-26 | a | "Responses are a function of D": replay gives the same bytes | open: needs http harness op |
| HC-27 | a | "A program whose module hash differs from D.program requires change on that log": two runs on one D | open: two-command cli test |
| HC-28 | a | "write their status to stderr as records of OSC 7501": stderr of one run | open: cli test |
| HC-29 | a | "Frames are derived from D": one run with frames FILE | open: cli test |
| CH16-M-acc6f048 | a | "the finding must not depend on the author opting in": a finding with no opt-in | open: change test |
| HC-30 | a | "each as its exact bytes ... in byte order": one listing | open: needs fs harness op |
| HC-31 | a | "lstat and readlink do not follow the entry": one symlink | open: needs fs harness op |
| HC-32 | c | "answers the wall clock in nanoseconds since the Unix epoch": a physical reading; form and order checkable, value not | open: form test; value needs hardware |
| HC-33 | a | "Account names need the `names` grant": one denied call | open: host test |
| HC-34 | a | "a bind that fails stops the host": one serve run | open: needs rap harness op |
| HC-35 | a | "durable before it is acknowledged": order of append and reply | open: needs rap harness op |
| HC-36 | a | "A result is a function of D, delivered at least once" | open: needs rap harness op |
| HC-37 | a | "whose state hash is the state hash of entry N - 1": compact then replay | open: two-command cli test |
| HC-38 | c | "A package counter reading enters D as host.Energy, in the D.11 form": form decidable, joules a measurement (LL-2) | tested (form): hc-38-energy-reading-form; value needs hardware |
| CH17-M-99b56574 | d | "a rule that the checker enforces must be enough for the theorem": a duty on the spec's rules, not on an implementation | propose: editorial |
| CH17-M-a1395e23 | d | "`rce-spec-check` must report 0 conflicts before this appendix loses the word": a gate on the book's own text | propose: tooling |
| CH17-M-ed78a540 | d | "Examples and the fuzz generator must be written clean before the rules are enabled": an order of work for the authors | propose: editorial |
| CH17-M-6076c934 | d | "This is a requirement on the lexer of 2.3": points at a rule of 2.3, adds none | propose: restatement |
| CH17-M-f213d82f | b | "the meaning of ... a name must not change with a tool, a font or a position": invariance over all renderings | open: property test |
| CH17-M-f7baba5b | a | "`==` and `<` on -0 and +0 must be shown to agree with the total order": four comparisons, exhaustive | open: run test |
| CH17-M-f3aecd9d | d | "is not written with a loop; it is a total step function": describes the model of 6.1 and Appendix E | propose: descriptive |
| CH17-M-21af0c5d | d | "Derivation: least privilege (a)": the reason for the W-WORLD rules, which carry the duty | propose: derivation |
| CH17-M-d10b1b54 | b | "the hash must not depend on names": invariance under every renaming; seeded renamings falsify | open: property test |
| CH17-M-67022294 | a | "authority held by a closure must be visible in its capture list": W-CAP-1 on one closure | tested: ch17-closure-authority-in-capture |
| CH17-M-07908645 | d | "Derivation: totality (d)": the reason for W-BUD-1, tested under S6 | propose: derivation |
| CH17-M-3c3abed1 | d | "Implementation cost is the largest of the tier": a cost remark; the counting rule is 10.5 | propose: descriptive |
| CH17-M-a720cd01 | a | "otherwise the call is checked at run time by the enforcement": a violated requires traps on one run | open: run test |
| CH17-M-0c74f453 | a | "an unbounded quantifier requires change": one program | open: change test |
| CH17-M-e53c7f6e | d | "Implement each W-rule of this appendix with accept and requires-change examples": instruction to the reference's authors | propose: tooling |
| S6 | a | "a call outside `with budget` requires change (W-BUD-1)" | tested: s6-model-call-outside-budget |
| P2 | a | "a clause elsewhere requires change (W-CON-2)" | tested: p2-contract-after-code |
| P4 | a | "W-DEC-1 decides this fragment" | tested: p4-decreases-outside-fragment |
| P5 | a | "A chain with a second `let` is reserved, not implemented" | tested: p5-iflet-chain-reserved |
| P6 | a | "Only the two rendered forms are implemented" | tested: p6-template-unknown-context |
| W-VER-1 | d | "the former W-VER-1": a mention of a removed rule, caught by the extractor's second pass | propose: extractor fix |

Totals: a 34, b 31, c 2, d 12. Tested in this change: 8 (S6, P2, P4, P5, P6,
CH17-M-67022294, HC-21, HC-38 form).

## Notes on the evidence

- **HC-38**: the test runs `rce host run energy` and requires the run's own
  replay to pass. Replay applies `energy_check` (D.11): exactly six members,
  basis `reported` or `measured`, joules finite and at least 0 (a counter
  delta is never negative), entries `[first, last]` before the entry's seq. On
  a host with no readable counter the run says why and records nothing, so the
  test asserts no joule value. A run on the M5 Max (2026-11-04) recorded 0.0345 J
  over 0.0176 s (reported IOReport reading; the reading file is not kept in this repository) with the counter text "IOReport package = cpu + gpu + ane",
  each part in nJ. The payload holds only the package sum, so the conservation
  check cpu + gpu + ane = package can be made only inside `rce-energy`, not
  from D; a value check of that sum needs hardware.
- **HC-32**: a clock reading is a physical answer. Its form (an integer of
  nanoseconds) and the order of two answers in one run are testable; the value
  is not.
- **P5**: the conform adapter reports W-SYN-1 for the chained `if let`, where
  `spec/examples-0.9/change/iflet_chain_let.rce` expects W-RES-1 from
  `rce check`. The clause fixes only that the chain is not accepted, so the test
  asks for a change with any code. The two codes disagreeing is a finding.
- **Lean evidence** (T48, T51, T53, T57, T58, T59, T62, T63, T28l, T28d,
  T28p, T28g, T28a): each row says the proof is on a model, so it does not
  cover the checker or the host. A property test against the implementation is
  still open for each.

## Proposed extractor changes (need David's review; not applied)

1. **Removed rules.** The second pass of `extract_clauses.py` makes a clause of
   every `W-` name it sees. W-VER-1 comes from "the former W-VER-1", which names
   a rule that no longer exists. Proposal: skip a name preceded by the word
   `former`. Reason: the sentence asserts the rule's absence, so it has no
   content to test.
2. **Derivations.** CH17-M-21af0c5d and CH17-M-07908645 begin "Derivation:".
   Their modal verb sits inside the reason for a rule (W-WORLD, W-BUD-1), not
   in a duty of its own; the rule they justify is already a clause. Proposal: a
   sentence that begins `Derivation:` gets profile `editorial`.
3. **Duties on the book's authors.** CH17-M-99b56574, CH17-M-a1395e23,
   CH17-M-ed78a540, CH17-M-e53c7f6e: their subject is a rule of the spec, the
   spec's own text, the examples, or "Implement each W-rule", so the one bound
   is the author of the reference, not an implementation. Proposal: profile
   `tooling` (a1395e23, e53c7f6e) or `editorial` (99b56574, ed78a540).
4. **Pointers and descriptions.** CH17-M-6076c934 ("This is a requirement on
   the lexer of 2.3") restates a rule stated in 2.3; CH08-M-30a83208,
   CH08-M-cad7de4b, CH17-M-f3aecd9d and CH17-M-3c3abed1 describe an effect, a
   list, a programming model or a cost, with no subject that an implementation
   could fail. Proposal: profile `editorial`, with the restated rule named.

If all four are accepted, 12 clauses leave the language profile and none of
their duties is lost: each is either absent (W-VER-1) or carried by a rule
that stays in the language profile.
