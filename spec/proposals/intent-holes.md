# Proposal: intent with holes, non-vacuity, completeness and the derivation log

Status: proposal. Owner: David Charlot. Derived from 6.9 (examples and
checks as admission), F.6.2 (contracts), T35 (the decidable predicate
fragment), `closing-the-loop.md` §5.1 (the `.intent` sketch) and section
2.2 of the paper (gears as entropy removal). It makes intent a first-class
object with a measured completeness, and makes the choice of a program
from an intent a logged derivation.

In plain words:

1. An intent is a program with holes: it says what must hold, and leaves
   some of how to the engine.
2. A verifier only checks that a program keeps its promises; it cannot
   tell a promise from an empty one. `ensures true` verifies. So every
   intent carries a measured score of how much it rules out, and an
   intent that rules out nothing is refused at the door.
3. Every hole that gets filled leaves a record of what was tried, what was
   rejected and why, and how many bits of choice the engine resolved. That
   record is logged like any event, so a synthesis is replayable.

## 1. Syntax (section 3 additions, not yet in the LALR(1) check)

    Hole     = "?" NAME ":" Type [ "{" Pred "}" ] .
    Obligation = "ensures" Pred NEWLINE | "invariant" Pred NEWLINE
               | "uses" NAME { "," NAME } NEWLINE | "budget" INT NEWLINE .

A hole may stand where an expression, a step body or a function body
stands. `Pred` is the T35 fragment (linear integer comparisons, `len`
bounds, character classes, conjunction and disjunction) over the state,
the event payload, the primed state and the outputs. `uses` names the
bricks (D.7) a fill may draw on; `budget` bounds the term size and the
`est_j` of a fill. A module with zero holes is an ordinary module and
every rule of this proposal is vacuous for it.

## 2. Rules

- **W-HOLE-1 (non-vacuity).** An intent is refused when its obligations and
  examples together reject no behaviour: when every obligation is
  satisfiable by the constant program that leaves the state unchanged and
  emits nothing, and no example constrains an output. The check is the
  T35 decision procedure on the obligations plus a run of the examples
  against the identity fill.
- **W-HOLE-2 (completeness is measured, not assumed).** Each intent carries
  a completeness score: the share of *spectests* it rejects. A spectest is
  a mutant output: for each example and each generated input, the
  reference output with one field changed, one output dropped, or one
  refusal flipped. The score is the fraction of spectests that fail at
  least one obligation or example. It is reported with the intent and
  recorded in the derivation; it is not a gate, because a low score is
  information the author needs, not an error. (Spectests follow SpecRL's
  executed-mutant construction, 2026; they cost a run each, not a solver
  call.)
- **W-HOLE-3 (no `assume`).** An obligation is never assumed; there is no
  construct that admits a predicate without a check. An obligation the
  fragment cannot decide is refused as outside T35, not admitted as
  trusted.

## 3. The derivation log

Filling a hole appends entries to a second log D_I, hash-chained like D
and keyed by the intent's shape hash, one entry per candidate:

    {hole, stage, candidate, verdict, evidence, bits, est_j, hash}

- `stage`: G0 lookup, G1 rule, G2 synthesis, G3 to G5 model (4.2 of the
  paper); the gear that proposed the candidate.
- `candidate`: the content hash of the proposed term.
- `verdict`: `admitted`, `refused:<obligation>`, `refused:<example>`,
  `refused:spectest`.
- `evidence`: the counterexample (an event sequence and the obligation it
  broke) or the certificate hash of the gate.
- `bits`: log₂(candidates alive before) − log₂(candidates alive after),
  the entropy this step removed (section 2.2 of the paper, made a number
  per step); for a lookup hit, the rank of the hit in the index.
- `est_j`: the estimate of the candidate's run, labeled as in 4.4.

The sum of `bits` over a hole is the intent bits the engine supplied; the
model calls in D_I are the paper's primary convergence endpoint (8.6)
read off the log instead of counted beside it. A program whose holes were
all filled carries the hash of its D_I in its concept record (6.6), so
"why is it this program" is answerable from the artefact.

## 4. What it refuses

- Score-gated promotion: a fill is admitted by the gate (tests, coverage,
  certificate), never by a model's confidence or a benchmark score.
- Self-refinement inside a gear: a candidate is proposed once and judged;
  a gear does not iterate on its own output (cost-of-pass, Erol et al.
  2025, finds this rarely pays).
- A Turing-complete intent language: `Pred` stays the decidable fragment.

## 5. What would show it wrong

If, on the ladder corpora, the completeness score does not predict the
plausible-versus-correct gap (Qi et al. 2015) — that is, if intents with
high scores admit as many wrong-but-passing fills as intents with low
scores — the spectest construction is not measuring what it claims and
W-HOLE-2 is replaced by whatever does.
