/-
  Select fairness (spec 6.8, draft 0.5).
  A select with `n` arms is executed repeatedly. Execution number `k` scans the
  arms cyclically starting at `k % n` and takes the first ready one.
  Theorem `select_bounded_wait`: an arm that is ready at every one of `n`
  consecutive executions is taken by one of them.
-/
namespace RCE.Select

/-- The arm taken at execution `k`, given the readiness of each arm. -/
def pick (n k : Nat) (ready : Nat → Bool) : Option Nat :=
  ((List.range n).find? (fun d => ready ((k + d) % n))).map (fun d => (k + d) % n)

theorem pick_start (n k : Nat) (ready : Nat → Bool) (hn : 0 < n) (h : ready (k % n) = true) :
    pick n k ready = some (k % n) := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  unfold pick
  rw [List.range_succ_eq_map]
  simp [List.find?, h]

/-- The taken arm is always a ready arm below `n`. -/
theorem pick_ready (n k : Nat) (ready : Nat → Bool) (i : Nat) (h : pick n k ready = some i) :
    ready i = true ∧ i < n := by
  unfold pick at h
  cases hf : (List.range n).find? (fun d => ready ((k + d) % n)) with
  | none => simp [hf] at h
  | some d =>
    simp [hf] at h
    have hd := List.find?_some hf
    have hn : 0 < n := by
      have := List.mem_of_find?_eq_some hf
      have := List.mem_range.mp this
      omega
    subst h
    exact ⟨by simpa using hd, Nat.mod_lt _ hn⟩

/-- Within any `n` consecutive executions there is one that starts its scan at arm `a`. -/
theorem exists_start (n k a : Nat) (ha : a < n) : ∃ t, t < n ∧ (k + t) % n = a := by
  have hn : 0 < n := by omega
  have hk := Nat.div_add_mod k n
  have hr := Nat.mod_lt k hn
  by_cases h : k % n ≤ a
  · refine ⟨a - k % n, by omega, ?_⟩
    have e : k + (a - k % n) = n * (k / n) + a := by omega
    rw [e, Nat.mul_add_mod, Nat.mod_eq_of_lt ha]
  · refine ⟨a + n - k % n, by omega, ?_⟩
    have e : k + (a + n - k % n) = n * (k / n + 1) + a := by rw [Nat.mul_succ]; omega
    rw [e, Nat.mul_add_mod, Nat.mod_eq_of_lt ha]

/-- Bounded wait: if arm `a` is ready at each of the executions `k, …, k+n-1`,
    one of them takes it. -/
theorem select_bounded_wait (n k a : Nat) (ready : Nat → Nat → Bool) (ha : a < n)
    (hready : ∀ t, t < n → ready (k + t) a = true) :
    ∃ t, t < n ∧ pick n (k + t) (ready (k + t)) = some a := by
  obtain ⟨t, ht, hmod⟩ := exists_start n k a ha
  refine ⟨t, ht, ?_⟩
  have h0 : ready (k + t) ((k + t) % n) = true := by rw [hmod]; exact hready t ht
  rw [pick_start n (k + t) _ (by omega) h0, hmod]

end RCE.Select
