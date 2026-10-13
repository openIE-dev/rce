/-
  Iterator termination (spec 6.8, draft 0.5).
  An iterator body is ordinary bounded RCE code (T2), so each step from one
  `yield` to the next is a total function `step : σ → Option (α × σ)`
  (`none`: the body finished). The consumer is a `for` loop that may
  `break`. The run counts yields against the loop limit `L`; yield number
  L + 1 traps "loop bound".
  Theorems:
  * `drive_total`: the run always ends, with at most `L` values delivered,
    and it traps only after exactly `L` values.
  * `drive_break`: when the consumer breaks, the values delivered are a
    prefix of the full run, so `break` only cuts the sequence short.
  * `gen_prefix`: running with a smaller cap gives a prefix of a run with a
    larger cap (the reference evaluator's re-run strategy is sound).
-/
namespace RCE.Iter

inductive Out (α : Type) where
  | done (xs : List α)          -- iterator finished
  | stopped (xs : List α)       -- consumer broke
  | bound (xs : List α)         -- loop bound trap

def Out.vals {α} : Out α → List α
  | .done xs | .stopped xs | .bound xs => xs

variable {σ α : Type}

/-- Collect up to `fuel` yields; the `fuel + 1`-th yield traps. `brk x` is
    the consumer's decision to break after receiving `x`. -/
def drive (step : σ → Option (α × σ)) (brk : α → Bool) : Nat → σ → Out α
  | fuel, s =>
    match step s with
    | none => .done []
    | some (x, s') =>
      match fuel with
      | 0 => .bound []
      | f + 1 =>
        if brk x then .stopped [x]
        else match drive step brk f s' with
          | .done xs => .done (x :: xs)
          | .stopped xs => .stopped (x :: xs)
          | .bound xs => .bound (x :: xs)

theorem drive_total (step : σ → Option (α × σ)) (brk : α → Bool) :
    ∀ (L : Nat) (s : σ), (drive step brk L s).vals.length ≤ L ∧
      (∀ xs, drive step brk L s = .bound xs → xs.length = L) := by
  intro L
  induction L with
  | zero =>
    intro s
    unfold drive
    cases step s with
    | none => simp [Out.vals]
    | some p => simp [Out.vals]
  | succ f ih =>
    intro s
    unfold drive
    cases step s with
    | none => simp [Out.vals]
    | some p =>
      obtain ⟨x, s'⟩ := p
      simp only
      split
      · simp [Out.vals]
      · obtain ⟨h1, h2⟩ := ih s'
        cases hd : drive step brk f s' with
        | done xs => simp [hd, Out.vals] at h1 ⊢; omega
        | stopped xs => simp [hd, Out.vals] at h1 ⊢; omega
        | bound xs =>
          simp [hd, Out.vals] at h1 ⊢
          exact ⟨by omega, h2 xs hd⟩

/-- The full run (no break). -/
def full (step : σ → Option (α × σ)) (L : Nat) (s : σ) : Out α := drive step (fun _ => false) L s

theorem drive_break (step : σ → Option (α × σ)) (brk : α → Bool) :
    ∀ (L : Nat) (s : σ), (drive step brk L s).vals <+: (full step L s).vals := by
  intro L
  induction L with
  | zero => intro s; unfold full drive; cases step s <;> simp [Out.vals]
  | succ f ih =>
    intro s
    unfold full drive
    cases step s with
    | none => simp [Out.vals]
    | some p =>
      obtain ⟨x, s'⟩ := p
      simp only
      have h := ih s'
      unfold full at h
      have hv : ∀ o : Out α, (match o with
          | .done xs => Out.done (x :: xs) | .stopped xs => .stopped (x :: xs) | .bound xs => .bound (x :: xs)).vals
          = x :: o.vals := by intro o; cases o <;> rfl
      split
      · simp only [Bool.false_eq_true, ↓reduceIte, hv]
        exact List.cons_prefix_cons.mpr ⟨rfl, List.nil_prefix⟩
      · simp only [Bool.false_eq_true, ↓reduceIte, hv]
        exact List.cons_prefix_cons.mpr ⟨rfl, h⟩

theorem gen_prefix (step : σ → Option (α × σ)) :
    ∀ (c : Nat) (s : σ), (full step c s).vals <+: (full step (c + 1) s).vals := by
  intro c
  induction c with
  | zero => intro s; unfold full drive; cases step s <;> simp [Out.vals]
  | succ f ih =>
    intro s
    unfold full drive
    cases step s with
    | none => simp [Out.vals]
    | some p =>
      obtain ⟨x, s'⟩ := p
      have h := ih s'
      unfold full at h
      have hv : ∀ o : Out α, (match o with
          | .done xs => Out.done (x :: xs) | .stopped xs => .stopped (x :: xs) | .bound xs => .bound (x :: xs)).vals
          = x :: o.vals := by intro o; cases o <;> rfl
      simp only [Bool.false_eq_true, ↓reduceIte, hv]
      exact List.cons_prefix_cons.mpr ⟨rfl, h⟩

end RCE.Iter
