# The encapsulation criterion (U4 made measurable)

Status: proposal. Owner: David Charlot. Derived from U1 to U4 of the paper
(section 3.6) and section 6.10 of the spec.

**Claim.** RCE encapsulates a language L when, for every program p of L in
a pinned corpus, the lifted Form F = λ_L(p) is no worse than p on each of
the five coordinates below and strictly better on at least one. "Better
than all programming languages" is the conjunction of this claim over the
atlas, one preregistered run per language, and nothing else: the Form is
not a better surface than L and does not claim to be.

| Coordinate | Measure | No worse means | Decides |
|---|---|---|---|
| Joules | `measured_j` (8.3) of π_L(F) and of p on the same task set and host | J(π_L(F)) ≤ J(p) within the meter's interval | RCE-F3 |
| Redundancy | redundancy bits of F (6.10, `rce dim`) against the state bits p declares | bits(F) ≤ bits(p) | the dimension bench |
| Decidability | which of {equivalence, termination, replay} are decided for F and for p by their own tools | every property decided for p is decided for F | by construction: T1, T2, T3, T58 |
| Holonomy | beh(π_L(F)) = beh(p) on the corpus's traces to length k, and hash(λ_L(π_L(F))) = hash(F) | both hold | T59, the lens law |
| Entropy to author | cost-of-pass (joules and dollars) for a model to reach a gate-passing program from the same intent, in F and in L | cost(F) ≤ cost(L) | the ladder, section 6 |

**What it refuses to claim.** Expressiveness: a Form cannot say more than p
(U2) and the surface language is one chart among many. Readability: that
is a property of a projection, and projections are free. Adoption: the
market does not prune (CONTRIBUTING rule 6).

**What would show it wrong.** Any language on which, after two frozen
runs, F is strictly worse on any coordinate: then RCE does not encapsulate
L and the atlas entry for L records what L's chart carries that the Form
does not. That entry is a work item for the Form, not for the surface.

**Order of languages.** By the lift's cost (atlas, 2026-10-22): Go and
Python first (cheapest ASTs), TypeScript and Rust next (after macro
expansion), then C and Zig (after comptime), then Fortran (arrays), then
the dynamic and logic lineages. Each run is a PREREG in
`openie-research-experiments/rce/bench/encapsulation/<L>/`.
