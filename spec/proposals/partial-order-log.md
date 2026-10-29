# Proposal: D as a partial order (chains, deps, cuts) and T62

Status: proposal. Owner: David Charlot. Derived from 6.1 (the log), 6.6
(name-free ids), E.2a (canonical admission), T3 (fold law), T16 (the `par`
diamond) and Proposition 6 of the paper (CALM). Nothing here changes δ or
the admitter's order; it changes what an entry's hash covers and where a
state hash is recorded, so that D's hash is invariant under every
linearization the semantics admits.

In plain words:

1. Today D is one chain: every entry hashes the entry before it, so two
   hosts that admit the same independent inputs in different orders have
   different logs even when the program cannot tell the difference.
2. The ticks of 6.8 already group entries into layers inside which order
   does not matter for independent events (T16). Make the log say that:
   each entry hashes what it depends on, not what happened to come before
   it, and the state hash is recorded at the cuts where order is fixed.
3. Then any order that respects the dependencies folds to the same state
   at every cut, which is a theorem (T62), and a replica may run the
   independent entries of a layer in any order it likes.

## 1. What changes in an entry

An entry of D (6.1) becomes

    {chain, seq, event, deps, hash}

- `chain`: the name-free key of the source (E.2a source id for inputs; the
  task or actor path of 6.6 for internal events).
- `seq`: the entry's index on its chain (program order on the chain).
- `deps`: the set of `hash`es of the entries this one causally observed:
  its chain predecessor, and for every location it reads, the last entry
  that wrote it (the write footprint of W-PAR-1). `deps` is a frontier:
  no element is an ancestor of another.
- `hash`: SHA-256 of the canonical JSON of {chain, deps, event, seq}.

`prev_hash` and `state_hash` leave the entry. A **cut entry**
`{cut, upto, state_hash, hash}` is appended at every point where the
total order is fixed by the semantics: the close of a window (HC-17
`host.Window`), a `Tick`, the end of an example, a `RESOLVE` (D.3), a
`Restart` (C.11.11), and before any entry whose event is not monotone
(below). Its `deps` is the whole frontier; its `state_hash` is the SHA-256
of the canonical state after folding every entry it dominates in the
canonical order of E.2a.

The admitter still appends in the canonical total order of HC-15, so T3
and Proposition 4 hold unchanged for the admitter's own replay. What the
change buys is that any other dep-consistent order gives the same cut
states, and that a log's hash does not depend on which of the admissible
orders the admitter happened to use.

## 2. Which events are ordered

The monotone tag of Proposition 6 is derived per rule for writes. It is
extended to reads: a read is monotone when it is threshold-shaped (it can
only observe a value growing past a bound, as LVar `wait` does); any other
read of a location another chain writes is a coordination point. The
events that must be ordered by a cut are exactly those whose rule is not
monotone under this extension: `select` across chains, `resolve`, a
read-modify-write of a shared location, `atomic`, `freeze`, `restart`, a
deadline. Everything else — `par` branches, `for … par` chunks, `pscan`,
workgroup lanes, LVar `put`, single-reader channels, and every step of a
`replicated state` region whose rules are all monotone — commutes with its
independent neighbours inside a layer, and T62 says the fold does not see
the order.

A rotating `select` (decision 48) that chooses among arms on different
chains logs its taken arm as part of its event, as `nondet` already does
(C.5), so that replay does not depend on the arrival order of the arms.

## 3. Theorem

**T62 (linearization invariance).** Let L be a layer of D between two cuts
and let `indep(a, b)` hold when `a` and `b` are on different chains and
neither is in the other's `deps` closure. For any two orders of L that
agree with every chain's `seq` order and every `deps` edge, the fold of δ₁
from the lower cut's state gives the same state at the upper cut, and the
multiset of outputs is the same (the output *order* is fixed by content
hash, as 6.8 already does).

Proof: the two orders differ by a sequence of swaps of adjacent
independent entries (Mazurkiewicz); an adjacent independent pair commutes
in δ₁ because their write footprints are disjoint and neither reads the
other's writes (the T16 diamond lemma, lifted from `par` branches to log
entries). Induction on the number of swaps.

The Lean model is `RCE/LogPO.lean`: entries as (chain, seq, footprint),
an abstract step on a store, the commutation lemma for independent
entries, and `fold_perm_eq`: two dep-consistent orderings of one layer
give equal stores. It is a model in the sense of chapter 8: it says the
rule is coherent, not that the admitter implements it; the link to the
admitter is by the HC-15 permutation tests of T40, which already show
byte-identical D for per-window permutations and become the test of T62
when entries are hashed by `deps`.

## 4. What it costs

- One more entry kind (`cut`) and a `deps` list per entry; a chain of n
  independent entries has n hashes instead of one, which is what makes
  them independent.
- Replay of a prefix that ends between cuts has no recorded state hash to
  check against; T38's "cut back to the last valid entry" becomes "cut
  back to the last cut", which is the same rule with a coarser grain.
- Nothing for δ, the emitters, or the examples: an example's log is one
  chain with one cut at its end, and its hash is unchanged in form.

## 5. What it does not claim

Target-chosen linearization for non-monotone events. A target may reorder
only inside a layer and only among entries T62 covers; everything else is
the admitter's order, as now. This is the CALM boundary (Hellerstein and
Alvaro 2020) and Flo's eager-execution criterion (POPL 2025), not a new
rule.

## 6. Order of work

1. `RCE/LogPO.lean` with `fold_perm_eq` (this proposal ships it).
2. Spec text for 6.1 (entry shape, cut entries) and E.2a (cuts at window
   close), gated on 1.
3. Admitter: write `deps` and cut entries; T40's permutation tests
   extended to assert equal `hash` per entry across arrival orders.
4. Runtimes: read the new shape; replay checks state at cuts.
5. Monotone tag on reads (Proposition 6 extension) in the checker, with
   the Appendix D table updated.

## Appendix: the T7 resolution A, sized

Making `state_hash` name-free (LEDGER, 2026-10-22) is the same kind of
change as the cut entry above: the bytes an entry hashes change, δ does
not. Touch points found on 2026-10-23:

- `spec/checker/src/interp.rs` `state_json` (one function) and `form_value`
  (`"$v"` variant tags by name, two sites).
- `spec/checker/src/host.rs`: five uses of `state_json` (admission, snapshot,
  snapshot check, replay check).
- The four emitted runtimes each rebuild the chain: `emit/rt.rs`,
  `rt.py` (line 1058 onward), `rt.js`, `rt.scala`. Any change to the
  hashed bytes is made four more times, which is the strongest argument in
  this repository for one ANF lowering and generated runtimes (roadmap
  phase 1).
- Fixtures: `spec/traces/*.json` (62), `tests/form_signature.rs`,
  `tests/host_*.rs` (eight files), `tests/canonical_json.rs` (77 vectors),
  `spec/data/canonical-json-vectors.json`.

Positional embedding: a region is its declaration index, a field its index
within the region, a variant its index within its kind, an event its index
among events; the named JSON stays as the derived `state` field of a trace
(6.1 already says derived fields are not hashed). Order of work: one
function in `interp.rs` behind a feature flag, the fixtures re-blessed
under the flag, the four runtimes, then the flag removed and C.15's
erratum row written. Estimated at one focused day with the suite as the
oracle; not started.
