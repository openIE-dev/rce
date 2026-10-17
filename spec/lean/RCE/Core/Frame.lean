import RCE.Core.Count
namespace RCE.Core

set_option maxHeartbeats 2000000 in
theorem evalC_frame (P : Prog) : ∀ d f ρ e k j, (evalC P d f ρ e).1 ≠ .oof →
    evalC P (d + j) (f + k) ρ e = ((evalC P d f ρ e).1, (evalC P d f ρ e).2 + k) := by
  intro d
  induction d with
  | zero => intro f ρ e k j h; simp [evalC] at h
  | succ d ih =>
    intro f ρ e k j h
    cases f with
    | zero => simp [evalC] at h
    | succ f =>
    rw [show d + 1 + j = (d + j) + 1 by omega, show f + 1 + k = (f + k) + 1 by omega]
    have ih' : ∀ ρ e r f', evalC P d f ρ e = (r, f') → r ≠ .oof → evalC P (d + j) (f + k) ρ e = (r, f' + k) :=
      fun ρ e r f' hr hn => by rw [ih f ρ e k j (by rw [hr]; exact hn), hr]
    have ih2 : ∀ f1 ρ e r f', evalC P d f1 ρ e = (r, f') → r ≠ .oof → evalC P (d + j) (f1 + k) ρ e = (r, f' + k) :=
      fun f1 ρ e r f' hr hn => by rw [ih f1 ρ e k j (by rw [hr]; exact hn), hr]
    cases e <;> simp only [evalC] at h ⊢
    all_goals
      repeat (first
        | (split at h <;> rename_i hx <;> (try rw [ih' _ _ _ _ hx (by simp)]) <;> (try rw [ih2 _ _ _ _ _ hx (by simp)]) <;> (try simp only [hx]) <;> simp_all; done)
        | (split at h <;> rename_i hx <;> (try rw [ih' _ _ _ _ hx (by simp)]) <;> (try rw [ih2 _ _ _ _ _ hx (by simp)]) <;> (try simp only [hx])))
    all_goals first
      | (rw [ih _ _ _ k j h]; done)
      | (rw [ih _ _ _ k j h]; split <;> simp_all)
      | (rw [ih _ _ _ k j h]; simp_all)
    all_goals simp_all [Prod.ext_iff]


/-- Branches run one after another on one tick counter (the reference's
    sequential charge of `par`, book 12 / C.4). -/
def seqC (P : Prog) (d : Nat) : Nat → List (List Val × Expr) → List Res × Nat
  | f, [] => ([], f)
  | f, (ρ, e) :: bs =>
    let r := evalC P d f ρ e
    let q := seqC P d r.2 bs
    (r.1 :: q.1, q.2)

/-- **S5 composition (fuel).** If each branch, run alone on exactly its use
    `u`, completes with result `r`, then running the branches in sequence on
    the summed budget plus any slack `k` gives the same results and leaves
    `k`. With `eval_fuel_indep` this is the hypothesis `ParFuel` assumes:
    a branch's use is fixed by the branch and its input, and the threaded
    account equals the sum of isolated uses (`ParFuel.sumcheck_eq_seq`). -/
theorem seqC_isolated (P : Prog) (d j : Nat) :
    ∀ (bs : List (List Val × Expr × Nat × Res)) (k : Nat),
      (∀ b ∈ bs, evalC P d b.2.2.1 b.1 b.2.1 = (b.2.2.2, 0) ∧ b.2.2.2 ≠ .oof) →
      seqC P (d + j) ((bs.map (·.2.2.1)).foldr (· + ·) k) (bs.map (fun b => (b.1, b.2.1))) =
        (bs.map (·.2.2.2), k)
  | [], _, _ => rfl
  | (ρ, e, u, r) :: bs, k, h => by
    have h0 := h _ (List.mem_cons_self _ _)
    have hf := evalC_frame P d u ρ e ((bs.map (·.2.2.1)).foldr (· + ·) k) j (by rw [h0.1]; exact h0.2)
    rw [h0.1] at hf
    simp only [List.map, List.foldr, seqC]
    rw [hf]
    simp only [Nat.zero_add]
    rw [seqC_isolated P d j bs k (fun b m => h b (List.mem_cons_of_mem _ m))]

end RCE.Core
