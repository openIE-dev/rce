import RCE.Core.F64Round
/-! Comparisons and the f64 → int truncation are exact (S3).
    `cmp` compares the exact values `±m·2^e`: the result is the same at every
    common unit `2^u` below both exponents, so it is the order of the reals.
    `trunc` is truncation toward zero of the exact value. -/
namespace RCE.F64

theorem ipow_add (a b : Nat) : (2:Int) ^ (a + b) = 2 ^ a * 2 ^ b := by
  rw [show (2:Int) = ((2:Nat):Int) from rfl, ← Int.natCast_pow, ← Int.natCast_pow, ← Int.natCast_pow,
    Nat.pow_add, Int.ofNat_mul]

theorem atUnit_scale (s : Bool) (m : Nat) (e u d : Int) (hu : u ≤ e) (hd : 0 ≤ d) :
    atUnit s m e (u - d) = atUnit s m e u * 2 ^ d.toNat := by
  have t1 := Int.toNat_of_nonneg (show 0 ≤ e - (u - d) by omega)
  have t2 := Int.toNat_of_nonneg (show 0 ≤ e - u by omega)
  have t3 := Int.toNat_of_nonneg hd
  have h : (e - (u - d)).toNat = (e - u).toNat + d.toNat := by omega
  unfold atUnit
  rw [h, ipow_add]
  cases s <;> simp [Int.mul_assoc, Int.neg_mul]

theorem compare_mul_pos (a b c : Int) (hc : 0 < c) : compare (a * c) (b * c) = compare a b := by
  simp only [compare, compareOfLessAndEq]
  have h1 : a * c < b * c ↔ a < b :=
    ⟨fun h => Int.lt_of_mul_lt_mul_right h (by omega), fun h => Int.mul_lt_mul_of_pos_right h hc⟩
  have h2 : a * c = b * c ↔ a = b :=
    ⟨fun h => Int.eq_of_mul_eq_mul_right (Int.ne_of_gt hc) h, fun h => by rw [h]⟩
  by_cases p : a < b <;> by_cases q : a = b <;> simp [h1, h2, p, q]

theorem pow2_pos (n : Nat) : (0:Int) < 2 ^ n := by
  rw [show (2:Int) = ((2:Nat):Int) from rfl, ← Int.natCast_pow]
  exact Int.ofNat_lt.mpr (Nat.pos_pow_of_pos _ (by decide))

/-- **Comparison is exact**: `cmp x y` is the order of the exact values,
    computed at any common unit `2^u` at or below both exponents. -/
theorem cmp_exact (x y : Nat) (u : Int) (h1 : u ≤ (decode x).2.2) (h2 : u ≤ (decode y).2.2) :
    cmp x y = compare (atUnit (decode x).1 (decode x).2.1 (decode x).2.2 u)
      (atUnit (decode y).1 (decode y).2.1 (decode y).2.2 u) := by
  simp only [cmp]
  generalize decode x = dx at *; generalize decode y = dy at *
  obtain ⟨s1, m1, e1⟩ := dx; obtain ⟨s2, m2, e2⟩ := dy
  simp only at h1 h2 ⊢
  have hm : u ≤ min e1 e2 := by omega
  have := atUnit_scale s1 m1 e1 (min e1 e2) (min e1 e2 - u) (by omega) (by omega)
  have h' := atUnit_scale s2 m2 e2 (min e1 e2) (min e1 e2 - u) (by omega) (by omega)
  rw [show min e1 e2 - (min e1 e2 - u) = u by omega] at this h'
  rw [this, h', compare_mul_pos _ _ _ (pow2_pos _)]

/-- `lt` and `eq` read the exact order (`-0 = +0`). -/
theorem lt_exact (x y : Nat) (u : Int) (h1 : u ≤ (decode x).2.2) (h2 : u ≤ (decode y).2.2) :
    lt x y = decide (atUnit (decode x).1 (decode x).2.1 (decode x).2.2 u <
      atUnit (decode y).1 (decode y).2.1 (decode y).2.2 u) := by
  unfold lt; rw [cmp_exact x y u h1 h2]
  generalize atUnit _ _ _ u = a; generalize atUnit _ _ _ u = b
  simp only [compare, compareOfLessAndEq]
  by_cases p : a < b <;> by_cases q : a = b <;> simp [p, q] <;> first | rfl | omega

theorem eq_exact (x y : Nat) (u : Int) (h1 : u ≤ (decode x).2.2) (h2 : u ≤ (decode y).2.2) :
    eq x y = decide (atUnit (decode x).1 (decode x).2.1 (decode x).2.2 u =
      atUnit (decode y).1 (decode y).2.1 (decode y).2.2 u) := by
  unfold eq; rw [cmp_exact x y u h1 h2]
  generalize atUnit _ _ _ u = a; generalize atUnit _ _ _ u = b
  simp only [compare, compareOfLessAndEq]
  by_cases p : a < b <;> by_cases q : a = b <;> simp [p, q] <;> first | rfl | omega

/-- **Truncation is exact**: for a nonnegative exponent the value is the
    integer `m·2^e`; otherwise `|trunc x|` is `⌊m/2^(-e)⌋`, i.e.
    `|t|·2^k ≤ m < (|t|+1)·2^k`, with the sign of `x` (toward zero). -/
theorem trunc_exact (x : Nat) :
    let d := decode x
    (trunc x = if d.1 then -((trunc x).natAbs : Int) else (trunc x).natAbs) ∧
    (0 ≤ d.2.2 → (trunc x).natAbs = d.2.1 * 2 ^ d.2.2.toNat) ∧
    (d.2.2 < 0 → (trunc x).natAbs * 2 ^ (-d.2.2).toNat ≤ d.2.1 ∧
                 d.2.1 < ((trunc x).natAbs + 1) * 2 ^ (-d.2.2).toNat) := by
  intro d
  simp only [trunc, d]
  generalize decode x = dx
  obtain ⟨s, m, e⟩ := dx
  simp only
  have hp : 0 < 2 ^ (-e).toNat := Nat.pos_pow_of_pos _ (by omega)
  refine ⟨?_, ?_, ?_⟩
  · cases s <;> simp
  · intro he; cases s <;> simp [he, Int.natAbs_mul]
  · intro he
    have hne : ¬ (e ≥ 0) := by omega
    have hdm := Nat.div_add_mod m (2 ^ (-e).toNat)
    have hml := Nat.mod_lt m hp
    have key : m / 2 ^ (-e).toNat * 2 ^ (-e).toNat ≤ m ∧
        m < (m / 2 ^ (-e).toNat + 1) * 2 ^ (-e).toNat := by
      generalize 2 ^ (-e).toNat = k at *
      generalize m / k = q at *; generalize m % k = r at *
      rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm q k]; omega
    cases s <;> simpa [hne] using key

end RCE.F64
