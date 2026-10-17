import RCE.Par

/-! Fuel accounting of `par` (book/12-tooling.md, 10.5; book/14 C.4).
    A branch run alone on fuel F either completes using u units (u ≤ F)
    or is exhausted (u > F). The sequential run of branches 1..k on fuel F
    traps at the first branch whose cumulative use passes F. The reference
    runs every branch on a copy of F and then sums the uses in source
    order, trapping at the first branch that is exhausted or whose running
    sum passes F. Proved: the two trap points are the same branch
    (`sumcheck_eq_seq`), and the work evaluated by the threaded run is at
    most k * F (`overrun_bound`). Uses are modelled as numbers fixed by the
    branch and its input, which is what schedule independence of a wait
    (no fuel while blocked) gives. Branch uses combined in the C.8 tree
    (power-of-two split, `RCE.Par.split`) give the same total as the
    source-order sum (`tree_use_eq_sum`, `tree_work_eq`), so the account
    does not depend on which fabric reduces the uses. -/
namespace RCE.ParFuel

/-- index of the first trapping branch, counting from 0, in the sequential
    run: `none` if all complete within F. `c` is the use so far. -/
def seqTrap (F : Nat) : Nat → List Nat → Option Nat
  | _, [] => none
  | c, u :: us => if c + u > F then some 0 else (seqTrap F (c + u) us).map (· + 1)

/-- the sum-checked account: branch i is exhausted on its own copy when
    u > F; else its use is added to the running sum. -/
def sumTrap (F : Nat) : Nat → List Nat → Option Nat
  | _, [] => none
  | c, u :: us => if u > F ∨ c + u > F then some 0 else (sumTrap F (c + u) us).map (· + 1)

theorem sumcheck_eq_seq (F : Nat) : ∀ (us : List Nat) (c : Nat), sumTrap F c us = seqTrap F c us
  | [], _ => rfl
  | u :: us, c => by
    simp only [sumTrap, seqTrap]
    by_cases h : c + u > F
    · simp [h]
    · have hu : ¬ u > F := by omega
      simp [h, hu, sumcheck_eq_seq F us (c + u)]

/-- the work the threaded run evaluates: each branch runs until it
    completes or exhausts its copy of F -/
def work (F : Nat) (us : List Nat) : Nat := (us.map (fun u => min u F)).foldl (· + ·) 0

theorem foldl_add_init (xs : List Nat) (a : Nat) : xs.foldl (· + ·) a = a + xs.foldl (· + ·) 0 := by
  induction xs generalizing a with
  | nil => simp
  | cons x xs ih => simp only [List.foldl]; rw [ih (a + x), ih (0 + x)]; omega

theorem overrun_bound (F : Nat) : ∀ us : List Nat, work F us ≤ us.length * F
  | [] => by simp [work]
  | u :: us => by
    have ih := overrun_bound F us
    simp only [work, List.map, List.foldl, List.length] at *
    rw [foldl_add_init]
    have : min u F ≤ F := Nat.min_le_right u F
    rw [Nat.succ_mul]
    omega

/-- the total use, reduced in the C.8 tree, is the source-order sum -/
theorem tree_use_eq_sum (us : List Nat) :
    RCE.Par.tree (· + ·) 0 us = us.foldl (· + ·) 0 :=
  RCE.Par.tree_eq_foldl _ _ Nat.add_assoc Nat.zero_add Nat.add_zero us

/-- the threaded work, reduced in the C.8 tree, obeys the overrun bound -/
theorem tree_work_eq (F : Nat) (us : List Nat) :
    RCE.Par.tree (· + ·) 0 (us.map (fun u => min u F)) = work F us ∧
    RCE.Par.tree (· + ·) 0 (us.map (fun u => min u F)) ≤ us.length * F := by
  rw [tree_use_eq_sum]; exact ⟨rfl, overrun_bound F us⟩

end RCE.ParFuel
