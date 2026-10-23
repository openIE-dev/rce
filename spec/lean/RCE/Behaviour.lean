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

/-! ### T59 over an explored alphabet

`Event := String × List Val` is not finite as a Lean type, so the
distance of 6.10.4 is stated over an explicit finite alphabet `evs`, the
declared events over the boundary sets that `rce dim` enumerates. This is
exactly the claim the tool can make, and it is what T59 means in the spec
once "every trace of length ≤ k" is read as "every trace over the
explored alphabet". -/

/-- All words of length `n` over the alphabet `evs`. -/
def words (evs : List Event) : Nat → List (List Event)
  | 0 => [[]]
  | n + 1 => (words evs n).flatMap fun w => evs.map fun e => e :: w

theorem mem_words_length (evs : List Event) : ∀ n w, w ∈ words evs n → w.length = n := by
  intro n
  induction n with
  | zero => intro w h; simp [words] at h; subst h; rfl
  | succ n ih =>
    intro w h
    simp only [words, List.mem_flatMap, List.mem_map] at h
    obtain ⟨w', hw', e, _, rfl⟩ := h
    simp [ih w' hw']

/-- Two states agree on every word of length `n` over `evs`. -/
def AgreeAt (P Q : Program) (evs : List Event) (s t : State) (n : Nat) : Prop :=
  ∀ w ∈ words evs n, beh P s w = beh Q t w

instance (P Q : Program) (evs : List Event) (s t : State) (n : Nat) :
    Decidable (AgreeAt P Q evs s t n) := by
  unfold AgreeAt; infer_instance

/-- The first length at which the two behaviours part, searched up to `k`
    (6.10.4, n(b₁, b₂) restricted to `evs`); `none` when they agree to `k`. -/
def firstDiff (P Q : Program) (evs : List Event) (s t : State) : Nat → Option Nat
  | 0 => if AgreeAt P Q evs s t 0 then none else some 0
  | k + 1 =>
    match firstDiff P Q evs s t k with
    | some n => some n
    | none => if AgreeAt P Q evs s t (k + 1) then none else some (k + 1)

/-- `firstDiff` only ever answers a length at which the states disagree. -/
theorem firstDiff_disagrees (P Q : Program) (evs : List Event) (s t : State) :
    ∀ k n, firstDiff P Q evs s t k = some n → ¬ AgreeAt P Q evs s t n := by
  intro k
  induction k with
  | zero =>
    intro n h
    simp only [firstDiff] at h
    split at h
    · simp at h
    · simp at h; subst h; assumption
  | succ k ih =>
    intro n h
    simp only [firstDiff] at h
    split at h
    · rename_i n' heq
      simp at h; subst h; exact ih _ heq
    · split at h
      · simp at h
      · simp at h; subst h; assumption

/-- **T59** (over the explored alphabet): agreement on every word of every
    length ≤ k means the first difference, if any, lies beyond k. Hence
    d(beh_P s, beh_Q t) ≤ 2^-(k+1) in the sense of 6.10.4. -/
theorem trace_ball (P Q : Program) (evs : List Event) (s t : State) (k : Nat)
    (h : ∀ n ≤ k, AgreeAt P Q evs s t n) :
    ∀ n, firstDiff P Q evs s t k = some n → k < n := by
  intro n hd
  by_contra hle
  push_neg at hle
  exact firstDiff_disagrees P Q evs s t k n hd (h n hle)

end RCE
