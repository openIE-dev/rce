import RCE.Form
/-! Laws (spec 6.11): exhaustive exploration decides a law exactly.

    A law states a property of every state the program reaches (`always p`)
    or of every step taken from such a state (`every step … -> o`). The
    checker's method D1 explores breadth first from s₀ over a complete
    alphabet and stops when no new state appears. **T63** is the ground of
    its two verdicts: a closed exploration on which the property holds
    proves it for every event sequence of every length, and a sequence after
    which it is false refutes it. States are reached by the fold of T3
    (`List.foldl (fun s e => (delta P s e).1) s₀`).

    In `lake build` and the standing axiom audit (scripts/lean-audit.sh,
    v4.12.0, no Batteries, no Mathlib). -/
namespace RCE

/-- The state after the event sequence `w` from `s` (the fold of T3). -/
def reachBy (P : Program) (s : State) (w : List Event) : State :=
  w.foldl (fun s e => (delta P s e).1) s

/-- A finite set of explored states (`seen`) is closed under a list of
    event instances (`evs`) when every instance takes every explored state
    to an explored state: the exploration has no frontier. -/
def ClosedOn (P : Program) (seen : List State) (evs : List Event) : Prop :=
  ∀ s, s ∈ seen → ∀ e, e ∈ evs → (delta P s e).1 ∈ seen

/-- Every state reached from an explored state by events of a complete
    alphabet is explored. -/
theorem reachBy_mem (P : Program) (seen : List State) (evs : List Event)
    (hc : ClosedOn P seen evs) (hall : ∀ e, e ∈ evs) :
    ∀ (w : List Event) (s : State), s ∈ seen → reachBy P s w ∈ seen
  | [], _, h => h
  | e :: w, s, h => reachBy_mem P seen evs hc hall w _ (hc s h e (hall e))

/-- **T63 (always, holds)**: with a complete alphabet, a closed exploration
    from s₀ on whose every state p holds proves p after every event
    sequence of every length. -/
theorem law_always_holds (P : Program) (s0 : State) (seen : List State) (evs : List Event)
    (p : State → Prop) (h0 : s0 ∈ seen) (hc : ClosedOn P seen evs) (hall : ∀ e, e ∈ evs)
    (hp : ∀ s, s ∈ seen → p s) : ∀ w, p (reachBy P s0 w) :=
  fun w => hp _ (reachBy_mem P seen evs hc hall w s0 h0)

/-- **T63 (every step, holds)**: the same for a property of a step, stated
    over the state before it, the event and δ's result. -/
theorem law_step_holds (P : Program) (s0 : State) (seen : List State) (evs : List Event)
    (q : State → Event → State × List Out → Prop) (h0 : s0 ∈ seen)
    (hc : ClosedOn P seen evs) (hall : ∀ e, e ∈ evs)
    (hq : ∀ s, s ∈ seen → ∀ e, q s e (delta P s e)) :
    ∀ w e, q (reachBy P s0 w) e (delta P (reachBy P s0 w) e) :=
  fun w e => hq _ (reachBy_mem P seen evs hc hall w s0 h0) e

/-- **T63 (fails)**: one sequence after which the property is false refutes
    the law, whatever bound the exploration reached: the counterexample is
    exact. -/
theorem law_always_fails (P : Program) (s0 : State) (p : State → Prop) (w : List Event)
    (hw : ¬ p (reachBy P s0 w)) : ¬ ∀ v, p (reachBy P s0 v) :=
  fun h => hw (h w)

/-- **T63 (exactness)**: when every explored state is reached from s₀ (each
    is found by stepping) and the exploration is closed over a complete
    alphabet, the law holds for every sequence exactly when it holds on the
    explored states. -/
theorem law_always_iff (P : Program) (s0 : State) (seen : List State) (evs : List Event)
    (p : State → Prop) (h0 : s0 ∈ seen) (hc : ClosedOn P seen evs) (hall : ∀ e, e ∈ evs)
    (hreach : ∀ s, s ∈ seen → ∃ w, reachBy P s0 w = s) :
    (∀ w, p (reachBy P s0 w)) ↔ (∀ s, s ∈ seen → p s) := by
  constructor
  · intro h s hs
    obtain ⟨w, hw⟩ := hreach s hs
    rw [← hw]
    exact h w
  · intro h
    exact law_always_holds P s0 seen evs p h0 hc hall h

end RCE
