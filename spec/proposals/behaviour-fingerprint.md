# Proposal: a behaviour fingerprint, so reuse finds a program by what it does

Status: open. The tool is implemented (`rce fingerprint`, 10.11); this
proposal records its definition and what it guarantees, and asks whether
the index it makes possible belongs in the normative text.

## The problem

Reuse today finds a program by its program hash (CC-1 of
`one-commit-change.md`): the SHA-256 over the definition hashes of its
name-free projection (10.6). Two programs equal up to names and layout
share it; two programs with the same behaviour and a different shape (a
loop and a bounded recursion, a counter kept as an integer and as two
booleans, guards written as separate steps and as an `if` in one step) do
not. `rce equiv A B` (10.11, CC-2) decides behaviour equality of two given
programs on the explored alphabet, but nothing finds, among many programs,
the ones that behave like a given one. This is the reuse form of the
objection from Rice's theorem: no total procedure decides behaviour
equality of arbitrary programs, so a shape key can only ever find shapes.

Every RCE program is a deterministic machine whose steps end (T2), and
6.10 defines its behaviour by its outputs (λ: what a step emits, or the
error that kept the state). On a bounded set of event sequences, behaviour
is therefore computable, and a key computed from it can be compared.

## Rule proposed

**BF-1 Behaviour fingerprint.** The fingerprint of a program P to depth n
(n ≥ 1) is the SHA-256 of:

- the text `rce-fingerprint/1 depth n alphabet k`, then the name-free key
  of each of the k event instances that `rce equiv` explores (the
  declaration index of the event and its payload values from the boundary
  sets of 6.3, in the canonical order of 6.1), each prefixed by its byte
  length; then
- h_n(s₀), where h₀(s) is the SHA-256 of `rce-fingerprint/1 leaf` and
  h_d(s), for d ≥ 1, is the SHA-256 of, for each instance e in canonical
  order, the length-prefixed name-free output of the step of e at s (the
  label `rce equiv` compares: emitted values in their positional form, a
  kept step by its error's declaration index, T7) followed by
  h_{d−1}(δ(s, e)) (a kept step leaves s unchanged, T8).

h_n(s₀) is a hash of the outputs of every event sequence of length 1 to n
over that alphabet: the depth-n tree of P's outputs, hashed bottom up.
Computing it steps each state within n − 1 steps of s₀ once per instance,
so the cost is one hash per (state, instance), never one per sequence
(there are k + k² + … + kⁿ sequences).

**What it guarantees.** Two programs with the same output on every event
sequence of length at most n have the same fingerprint at depth n; this
follows from the definition by induction on n. In particular, when
`rce equiv A B` judges A and B equivalent and both fingerprints are
computed, A and B have the same fingerprint at every depth: equivalence is
bisimilarity on the explored graph (T58), and the partition it computes is
stable, so the outputs of every sequence that stays within both explored
graphs agree. Equal fingerprints do not imply equivalence: two programs
that differ only after n events share the fingerprint at depth n. A shared
fingerprint is therefore a candidate, confirmed or refuted by
`rce equiv`. Different alphabets never share a fingerprint, as `rce equiv`
never calls them equivalent.

**The limit.** Equality is up to the bound in two senses: the alphabet is
the boundary set of 6.3, not every payload value, and the sequences are at
most n long. Two programs that differ on a payload outside the boundary
set, or only after more than n events, share a fingerprint and are called
equivalent by `rce equiv` alike. For two machines whose reachable graphs
are fully explored with a and b states, depth a + b − 1 separates every
pair that `rce equiv` separates (Moore 1956); a smaller depth may not.
`--fuel` bounds the states stepped; a fingerprint that would need a state
beyond the bound is not computed and the command says so.

## What it adds and what it does not

It adds one command, `rce fingerprint [--depth N] [--fuel N] [--json]
FILE...` (default depth 8, fuel 100,000; one report for one file, one line
`HASH  FILE` per file for several), and the JSON format
`rce-fingerprint/1`. It adds no construct and no admission rule; no
verdict depends on it (it is informative, as `rce dim` and `rce equiv`
are). It does not decide behaviour equality of arbitrary programs, and
does not claim to: Rice's theorem stands. It turns reuse by behaviour into
a lookup for the bounded behaviour that 6.10 defines. The registered
measurement (`openie-research-experiments/rce/bench/behaviour-index/`)
found all 86 behaviour-equal pairs with distinct program hashes in its
population (the program hash found none), missed none of the 187 pairs
`rce equiv` judged equivalent, and left 2.0% of the pairs sharing a
depth-8 fingerprint undecided by `rce equiv` at its bound, none with a
distinguishing sequence. It covers state machines only: 750 of its 1,079
program texts are commands, which have no event alphabet.

## What stays open

Whether `rce index` should carry the fingerprint per program so that
`rce search` can retrieve by it; a Lean statement of the guarantee above
next to T58 and T59 in `RCE/Behaviour.lean`; a fingerprint over the
refinement fragment of T35 instead of the base type's boundary set.
