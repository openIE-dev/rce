/-
T32 Foreign replay. A step may consult a foreign function (5.11) or a
nondeterministic choice; its answer is an input and is logged with the
event. Replaying the log reproduces the live run for EVERY oracle: the
statement is quantified over all oracles, so no property of the foreign
code is assumed. A contract on answers (a predicate P every answer
satisfies) holds of every logged answer, so it is available on replay.
-/
namespace RCE.Replay

variable {S E A : Type}

/-- The live run: each event consults the oracle, and the log records
the event with the answer. -/
def live (δ : S → E → A → S) (o : S → E → A) : S → List E → S × List (E × A)
  | s, [] => (s, [])
  | s, e :: es =>
    let a := o s e
    let r := live δ o (δ s e a) es
    (r.1, (e, a) :: r.2)

/-- Replay reads answers from the log; it has no oracle. -/
def replay (δ : S → E → A → S) : S → List (E × A) → S
  | s, [] => s
  | s, (e, a) :: l => replay δ (δ s e a) l

theorem replay_live (δ : S → E → A → S) (o : S → E → A) (s : S) (es : List E) :
    replay δ s (live δ o s es).2 = (live δ o s es).1 := by
  induction es generalizing s with
  | nil => rfl
  | cons e es ih => simp only [live, replay]; exact ih _

theorem live_events (δ : S → E → A → S) (o : S → E → A) (s : S) (es : List E) :
    (live δ o s es).2.map Prod.fst = es := by
  induction es generalizing s with
  | nil => rfl
  | cons e es ih => simp only [live, List.map]; rw [ih]

theorem live_answers_satisfy (δ : S → E → A → S) (o : S → E → A) (P : A → Prop)
    (h : ∀ s e, P (o s e)) (s : S) (es : List E) :
    ∀ x ∈ (live δ o s es).2, P x.2 := by
  induction es generalizing s with
  | nil => intro x hx; cases hx
  | cons e es ih =>
    intro x hx
    simp only [live] at hx
    cases hx with
    | head => exact h s e
    | tail _ hx => exact ih _ x hx

/-- T32: two runs with different oracles that logged the same answers
reach the same state; replay alone decides it. -/
theorem t32 (δ : S → E → A → S) (o₁ o₂ : S → E → A) (s : S) (es₁ es₂ : List E)
    (h : (live δ o₁ s es₁).2 = (live δ o₂ s es₂).2) :
    (live δ o₁ s es₁).1 = (live δ o₂ s es₂).1 := by
  rw [← replay_live δ o₁, ← replay_live δ o₂, h]

end RCE.Replay
