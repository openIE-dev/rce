import RCE.Core.Lang
/-! S1 fuel laws: more fuel never changes a result that did not run out.
    So exhaustion is the only fuel-dependent outcome, and an
    implementation that meters differently but agrees on budgets agrees
    on every completed result (the property the conformance suite's
    differential runs rely on). -/
namespace RCE.Core

theorem eval_mono (P : Prog) : ∀ n ρ e, eval P n ρ e ≠ .oof → eval P (n+1) ρ e = eval P n ρ e := by
  intro n
  induction n with
  | zero => intro ρ e h; simp [eval] at h
  | succ n ih =>
    intro ρ e h
    conv at h => simp only [eval]
    conv => rhs; simp only [eval]
    generalize _hm : n + 1 = m at ih ⊢
    conv => lhs; simp only [eval]
    have ih' : ∀ ρ e r, eval P n ρ e = r → r ≠ .oof → eval P m ρ e = r :=
      fun ρ e r hr hn => by rw [ih ρ e (by rw [hr]; exact hn), hr]
    cases e <;> simp only at h ⊢
    all_goals
      repeat (first
        | (split at h <;> rename_i hx <;> (try rw [ih' _ _ _ hx (by simp)]) <;> (try simp only [hx]) <;> simp_all; done)
        | (split at h <;> rename_i hx <;> (try rw [ih' _ _ _ hx (by simp)]) <;> (try simp only [hx])))
    all_goals first | exact ih _ _ h | (rw [ih _ _ h]; split <;> simp_all)

end RCE.Core

namespace RCE.Core

/-- Fuel monotonicity in general form: any larger budget gives the same
    completed result. -/
theorem eval_mono_le (P : Prog) {n k ρ e} (h : eval P n ρ e ≠ .oof) :
    eval P (n + k) ρ e = eval P n ρ e := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [← Nat.add_assoc, eval_mono P _ _ _ (by rw [ih]; exact h), ih]

/-- Two budgets that both complete give the same result. -/
theorem eval_fuel_indep (P : Prog) {n m ρ e} (hn : eval P n ρ e ≠ .oof)
    (hm : eval P m ρ e ≠ .oof) : eval P n ρ e = eval P m ρ e := by
  rcases Nat.le_total n m with h | h
  · obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h; exact (eval_mono_le P hn).symm
  · obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h; exact eval_mono_le P hm

end RCE.Core
