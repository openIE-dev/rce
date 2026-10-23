import RCE.Form
/-! Behaviour, bisimulation and distance for the Form of `RCE/Form.lean`
    (spec 6.10). T57 (one-step unfolding) and T58 (equivalence is
    bisimilarity) are proved here; T59 (trace evidence is a ball) is stated
    as an obligation that needs a finite event type.

    Not yet in `lake build` (see `lakefile.lean`): add `RCE.Behaviour` to
    the library once the obligation below is discharged or moved to
    `obligations/`. Written without the toolchain at hand; the three proofs
    use only `rfl`, induction on the event list and `simp` on the
    definitional unfolding, so they should check as written on v4.12.0. -/
namespace RCE

/-- The per-step output sequence along an event sequence (6.10.1). -/
def beh (P : Program) : State → List Event → List (List Out)
  | _, [] => []
  | s, e :: w =>
    let r := delta P s e
    r.2 :: beh P r.1 w

/-- A bisimulation for P (6.10.3): related states give equal outputs and
    related successors on every event. -/
def Bisim (P : Program) (R : State → State → Prop) : Prop :=
  ∀ s t, R s t → ∀ e, (delta P s e).2 = (delta P t e).2 ∧ R (delta P s e).1 (delta P t e).1

/-- **T57** (the morphism equation): one step of behaviour unfolds to the
    step's outputs followed by the successor's behaviour. -/
theorem beh_cons (P : Program) (s : State) (e : Event) (w : List Event) :
    beh P s (e :: w) = (delta P s e).2 :: beh P (delta P s e).1 w := rfl

/-- **T58 (⇐)**: a bisimulation forces equal behaviours. Induction on the
    event sequence, generalising the related pair. -/
theorem bisim_beh (P : Program) (R : State → State → Prop) (hR : Bisim P R) :
    ∀ (w : List Event) (s t : State), R s t → beh P s w = beh P t w := by
  intro w
  induction w with
  | nil => intro s t _; rfl
  | cons e w ih =>
    intro s t hst
    obtain ⟨ho, hr⟩ := hR s t hst e
    rw [beh_cons, beh_cons, ho, ih _ _ hr]

/-- **T58 (⇒)**: behavioural equality is itself a bisimulation, hence the
    largest one. -/
theorem beh_bisim (P : Program) :
    Bisim P (fun s t => ∀ w, beh P s w = beh P t w) := by
  intro s t h e
  refine ⟨?_, ?_⟩
  · have h1 := h [e]
    simp only [beh_cons, beh] at h1
    exact (List.cons.inj h1).1
  · intro w
    have h1 := h (e :: w)
    simp only [beh_cons] at h1
    exact (List.cons.inj h1).2

/-- Behavioural equivalence of two states (6.10.2 for one program). -/
def BehEq (P : Program) (s t : State) : Prop := ∀ w, beh P s w = beh P t w

/-- T58 as one statement. -/
theorem behEq_iff_bisim (P : Program) (s t : State) :
    BehEq P s t ↔ ∃ R, Bisim P R ∧ R s t :=
  ⟨fun h => ⟨_, beh_bisim P, h⟩,
   fun ⟨R, hR, hst⟩ w => bisim_beh P R hR w s t hst⟩

/-! ### Obligation for T59

`Event := String × List Val` is not finite as a Lean type, so the shortest
distinguishing sequence cannot be searched by enumeration here. The
obligation is: given a finite list `evs : List Event` (the declared events
over the T35 payload fragment, as `rce dim` enumerates them), define

    firstDiff P s t evs k : Option Nat

as the least `n ≤ k` such that some `w` of length `n` over `evs` has
`beh P s w ≠ beh P t w`, and prove

    (∀ w, w.length ≤ k → (∀ x ∈ w, x ∈ evs) → beh P s w = beh P t w)
      → ∀ n, firstDiff P s t evs k = some n → k < n.

Together with 6.10.4 this is T59 restricted to the explored alphabet, which
is exactly the claim `rce dim` can make. -/

end RCE
