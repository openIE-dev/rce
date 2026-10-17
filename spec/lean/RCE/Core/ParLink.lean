import RCE.Core.Frame
import RCE.ParFuel
/-! S5: the formal link from the core's counted interpreter to `ParFuel`.
    `ParFuel` models a branch's use as a number fixed by the branch; here
    the uses are the core branches' isolated tick counts. When the
    reference's sum-checked account (`sumTrap`, each branch on a copy of
    the budget) reports no trap, the threaded sequential run of the
    branches on the one budget gives every branch its isolated result and
    leaves the budget less the summed uses. -/
namespace RCE.Core

theorem foldr_add_init (xs : List Nat) (k : Nat) :
    xs.foldr (· + ·) k = xs.foldr (· + ·) 0 + k := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp only [List.foldr]; rw [ih]; omega

/-- No trap in the sequential account means the summed use fits. -/
theorem seqTrap_none (F : Nat) :
    ∀ (us : List Nat) (c : Nat), RCE.ParFuel.seqTrap F c us = none → c ≤ F →
      c + us.foldr (· + ·) 0 ≤ F
  | [], c, _, h => by simpa using h
  | u :: us, c, h, _ => by
    simp only [RCE.ParFuel.seqTrap] at h
    by_cases hu : c + u > F
    · simp [hu] at h
    · simp only [hu, if_false, Option.map_eq_none'] at h
      have := seqTrap_none F us (c + u) h (by omega)
      simp only [List.foldr]; omega

/-- **The link.** Branches `(ρ, e, u, r)`, each completing alone on exactly
    its use `u` with result `r`; if the reference's account
    (`ParFuel.sumTrap`) finds no trap on budget `F`, the threaded run on
    `F` returns the isolated results and `F - Σ u` ticks. -/
theorem par_core_link (P : Prog) (d j F : Nat)
    (bs : List (List Val × Expr × Nat × Res))
    (hb : ∀ b ∈ bs, evalC P d b.2.2.1 b.1 b.2.1 = (b.2.2.2, 0) ∧ b.2.2.2 ≠ .oof)
    (hacc : RCE.ParFuel.sumTrap F 0 (bs.map (·.2.2.1)) = none) :
    seqC P (d + j) F (bs.map (fun b => (b.1, b.2.1))) =
      (bs.map (·.2.2.2), F - (bs.map (·.2.2.1)).foldr (· + ·) 0) := by
  rw [RCE.ParFuel.sumcheck_eq_seq] at hacc
  have hle := seqTrap_none F _ 0 hacc (Nat.zero_le _)
  have := seqC_isolated P d j bs (F - (bs.map (·.2.2.1)).foldr (· + ·) 0) hb
  rw [foldr_add_init] at this
  rwa [show (bs.map (·.2.2.1)).foldr (· + ·) 0 + (F - (bs.map (·.2.2.1)).foldr (· + ·) 0) = F by omega] at this

end RCE.Core
