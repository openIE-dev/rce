import RCE.Core.F64
/-! S3: `RCE.F64.round` is round-to-nearest, ties to even, of the exact
    value, over every finite binary64 value; and it overflows exactly when
    IEEE 754 roundTiesToEven does. -/
namespace RCE.F64

theorem lgBits_spec : ∀ (f n acc : Nat), 0 < n → n < 2 ^ (f + 1) →
    acc ≤ lgBits n f acc ∧ 2 ^ (lgBits n f acc - acc) ≤ n ∧ n < 2 ^ (lgBits n f acc - acc + 1) := by
  intro f
  induction f with
  | zero => intro n acc h0 h1; simp [lgBits]; omega
  | succ f ih =>
    intro n acc h0 h1
    simp only [lgBits]
    split
    · simp; omega
    · rename_i h2
      have hn2 : 0 < n / 2 := (Nat.le_div_iff_mul_le (by omega)).mpr (by omega)
      have hlt : n / 2 < 2 ^ (f + 1) := by
        rw [Nat.div_lt_iff_lt_mul (by omega)]; rw [Nat.pow_succ] at h1; exact h1
      obtain ⟨a, b, c⟩ := ih (n / 2) (acc + 1) hn2 hlt
      generalize lgBits (n / 2) f (acc + 1) = g at *
      refine ⟨by omega, ?_, ?_⟩
      · have : g - acc = (g - (acc + 1)) + 1 := by omega
        rw [this, Nat.pow_succ]
        exact (Nat.le_div_iff_mul_le (by omega : 0 < 2)).mp b
      · have : g - acc + 1 = (g - (acc + 1) + 1) + 1 := by omega
        rw [this, Nat.pow_succ]
        exact (Nat.div_lt_iff_lt_mul (by omega : 0 < 2)).mp c

theorem lgWords_spec : ∀ (f n acc : Nat), 0 < n → n < 2 ^ (64 * f + 64) →
    acc ≤ lgWords n f acc ∧ 2 ^ (lgWords n f acc - acc) ≤ n ∧ n < 2 ^ (lgWords n f acc - acc + 1) := by
  intro f
  induction f with
  | zero =>
    intro n acc h0 h1
    simp only [lgWords]
    exact lgBits_spec 64 n acc h0 (Nat.lt_of_lt_of_le h1 (Nat.pow_le_pow_right (by omega) (by omega)))
  | succ f ih =>
    intro n acc h0 h1
    simp only [lgWords]
    split
    · rename_i h2
      exact lgBits_spec 64 n acc h0 (Nat.lt_of_lt_of_le h2 (Nat.pow_le_pow_right (by omega) (by omega)))
    · rename_i h2
      have hp : 0 < 2 ^ 64 := Nat.pos_pow_of_pos _ (by omega)
      have hn2 : 0 < n / 2 ^ 64 := (Nat.le_div_iff_mul_le hp).mpr (by omega)
      have hlt : n / 2 ^ 64 < 2 ^ (64 * f + 64) := by
        rw [Nat.div_lt_iff_lt_mul hp, ← Nat.pow_add]
        have : 64 * f + 64 + 64 = 64 * (f + 1) + 64 := by omega
        rw [this]; exact h1
      obtain ⟨a, b, c⟩ := ih (n / 2 ^ 64) (acc + 64) hn2 hlt
      generalize lgWords (n / 2 ^ 64) f (acc + 64) = g at *
      refine ⟨by omega, ?_, ?_⟩
      · have : g - acc = (g - (acc + 64)) + 64 := by omega
        rw [this, Nat.pow_add]
        exact (Nat.le_div_iff_mul_le hp).mp b
      · have : g - acc + 1 = (g - (acc + 64) + 1) + 64 := by omega
        rw [this, Nat.pow_add]
        exact (Nat.div_lt_iff_lt_mul hp).mp c

/-- `lg` is `⌊log2⌋` below `2^6464`. -/
theorem lg_spec (n : Nat) (h0 : 0 < n) (h1 : n < 2 ^ 6464) :
    2 ^ lg n ≤ n ∧ n < 2 ^ (lg n + 1) := by
  have := lgWords_spec 100 n 0 h0 h1
  simp only [lg, Nat.sub_zero] at this ⊢
  exact ⟨this.2.1, this.2.2⟩

theorem rq_cases (num den : Nat) :
    (rq num den = num / den + 1 ∧ (2 * (num % den) > den ∨ (2 * (num % den) = den ∧ (num / den) % 2 = 1))) ∨
    (rq num den = num / den ∧ ¬(2 * (num % den) > den ∨ (2 * (num % den) = den ∧ (num / den) % 2 = 1))) := by
  unfold rq; split
  · left; exact ⟨rfl, by assumption⟩
  · right; exact ⟨rfl, by assumption⟩

/-- **Nearest.** `rq num den` is a nearest integer to `num/den`. -/
theorem rq_near (num den q' : Nat) (hd : 0 < den) :
    ndist num (rq num den * den) ≤ ndist num (q' * den) := by
  have hdm := Nat.div_add_mod num den
  have hr := Nat.mod_lt num hd
  rcases rq_cases num den with ⟨h1, hc⟩ | ⟨h1, hc⟩ <;> rw [h1] <;> clear h1 <;>
  generalize num / den = q at * <;>
  generalize num % den = r at * <;>
  have hq1 : (q + 1) * den = q * den + den := Nat.succ_mul q den <;>
  rw [Nat.mul_comm den q] at hdm <;>
  unfold ndist
  · rw [hq1]
    rcases Nat.lt_or_ge q' (q + 1) with h | h
    · have : q' * den ≤ q * den := Nat.mul_le_mul_right _ (by omega)
      generalize q' * den = a at *; generalize q * den = b at *
      split <;> split <;> omega
    · have : (q + 1) * den ≤ q' * den := Nat.mul_le_mul_right _ h
      rw [hq1] at this
      generalize q' * den = a at *; generalize q * den = b at *
      split <;> split <;> omega
  · rcases Nat.lt_or_ge q' (q + 1) with h | h
    · have : q' * den ≤ q * den := Nat.mul_le_mul_right _ (by omega)
      generalize q' * den = a at *; generalize q * den = b at *
      split <;> split <;> omega
    · have : (q + 1) * den ≤ q' * den := Nat.mul_le_mul_right _ h
      rw [hq1] at this
      generalize q' * den = a at *; generalize q * den = b at *
      split <;> split <;> omega

/-- **Ties to even.** -/
theorem rq_tie_even (num den : Nat) (hd : 0 < den)
    (ht : 2 * ndist num (rq num den * den) = den) : rq num den % 2 = 0 := by
  have hdm := Nat.div_add_mod num den
  have hr := Nat.mod_lt num hd
  rcases rq_cases num den with ⟨h1, hc⟩ | ⟨h1, hc⟩ <;> rw [h1] at ht ⊢ <;> clear h1 <;>
  generalize num / den = q at * <;>
  generalize num % den = r at * <;>
  have hq1 : (q + 1) * den = q * den + den := Nat.succ_mul q den <;>
  rw [Nat.mul_comm den q] at hdm <;>
  unfold ndist at ht
  · rw [hq1] at ht; generalize q * den = b at *; split at ht <;> omega
  · generalize q * den = b at *; split at ht <;> omega


theorem rq_half (num den : Nat) (hd : 0 < den) : 2 * ndist num (rq num den * den) ≤ den := by
  have hdm := Nat.div_add_mod num den
  have hr := Nat.mod_lt num hd
  rcases rq_cases num den with ⟨h1, hc⟩ | ⟨h1, hc⟩ <;> rw [h1] <;> clear h1 <;>
  generalize num / den = q at * <;>
  generalize num % den = r at * <;>
  have hq1 : (q + 1) * den = q * den + den := Nat.succ_mul q den <;>
  rw [Nat.mul_comm den q] at hdm <;>
  unfold ndist
  · rw [hq1]; generalize q * den = b at *; split <;> omega
  · generalize q * den = b at *; split <;> omega

theorem rq_tie (num den q' : Nat) (hd : 0 < den)
    (he : ndist num (rq num den * den) = ndist num (q' * den))
    (hne : q' * den ≠ rq num den * den) : 2 * ndist num (rq num den * den) = den := by
  have h1 := rq_half num den hd
  generalize rq num den = q at *
  rcases Nat.lt_or_ge q' q with h | h
  · have : (q' + 1) * den ≤ q * den := Nat.mul_le_mul_right _ h
    rw [Nat.succ_mul] at this
    unfold ndist at *
    generalize q' * den = a at *; generalize q * den = b at *
    by_cases hb : b ≤ num <;> by_cases ha : a ≤ num <;> simp only [hb, ha, if_true, if_false] at he h1 ⊢ <;> omega
  · rcases Nat.eq_or_lt_of_le h with h | h
    · subst h; exact absurd rfl hne
    · have : (q + 1) * den ≤ q' * den := Nat.mul_le_mul_right _ h
      rw [Nat.succ_mul] at this
      unfold ndist at *
      generalize q' * den = a at *; generalize q * den = b at *
      by_cases hb : b ≤ num <;> by_cases ha : a ≤ num <;> simp only [hb, ha, if_true, if_false] at he h1 ⊢ <;> omega

theorem rq_le (num den : Nat) : rq num den ≤ num / den + 1 := by
  unfold rq; split <;> omega

theorem rq_ge (num den : Nat) : num / den ≤ rq num den := by
  unfold rq; split <;> omega


/-! Patterns. -/

theorem p63 : (2:Nat) ^ 63 = 2048 * 2 ^ 52 := by
  rw [show (63:Nat) = 11 + 52 from rfl, Nat.pow_add]
theorem p53 : (2:Nat) ^ 53 = 2 * 2 ^ 52 := by
  rw [show (53:Nat) = 1 + 52 from rfl, Nat.pow_add]

/-- A nonnegative finite binary64 pattern (+0, subnormal or normal). -/
def NonnegFin (c : Nat) : Prop := c < 2 ^ 63 ∧ isFinite c = true

theorem decode_sub (q : Nat) (hq : q < 2 ^ 52) : decode q = (false, q, -1074) := by
  have h1 : q / 2 ^ 52 = 0 := Nat.div_eq_of_lt hq
  have h2 : q % 2 ^ 52 = q := Nat.mod_eq_of_lt hq
  have h3 : ¬ q ≥ 2 ^ 63 := by have : (2:Nat) ^ 52 ≤ 2 ^ 63 := Nat.pow_le_pow_right (by omega) (by omega); omega
  simp [decode, h1, h2, h3]

theorem decode_norm (ex r : Nat) (h1 : 1 ≤ ex) (h2 : ex < 2047) (hr : r < 2 ^ 52) :
    decode (ex * 2 ^ 52 + r) = (false, r + 2 ^ 52, (ex : Int) - 1075) := by
  have hp : 0 < 2 ^ 52 := Nat.pos_pow_of_pos _ (by omega)
  have hd : (ex * 2 ^ 52 + r) / 2 ^ 52 = ex := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hr]; omega
  have hm : (ex * 2 ^ 52 + r) % 2 ^ 52 = r := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]
  have hlt : ex * 2 ^ 52 + r < 2 ^ 63 := by
    have : ex * 2 ^ 52 ≤ 2046 * 2 ^ 52 := Nat.mul_le_mul_right _ (by omega)
    have := p63
    omega
  have h3 : ¬ ex * 2 ^ 52 + r ≥ 2 ^ 63 := by omega
  have h4 : ex % 2048 = ex := Nat.mod_eq_of_lt (by omega)
  have h5 : ex ≠ 0 := by omega
  simp [decode, hd, hm, h3, h4, h5]

theorem norm_fin (ex r : Nat) (h2 : ex < 2047) (hr : r < 2 ^ 52) : NonnegFin (ex * 2 ^ 52 + r) := by
  have hp : 0 < 2 ^ 52 := Nat.pos_pow_of_pos _ (by omega)
  have hd : (ex * 2 ^ 52 + r) / 2 ^ 52 = ex := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hr]; omega
  have : ex * 2 ^ 52 ≤ 2046 * 2 ^ 52 := Nat.mul_le_mul_right _ (by omega)
  refine ⟨by omega, ?_⟩
  simp [isFinite, hd, Nat.mod_eq_of_lt (show ex < 2048 by omega)]; omega

theorem sub_fin (q : Nat) (hq : q < 2 ^ 52) : NonnegFin q := by
  refine ⟨by have : (2:Nat) ^ 52 ≤ 2 ^ 63 := Nat.pow_le_pow_right (by omega) (by omega); omega, ?_⟩
  simp [isFinite, Nat.div_eq_of_lt hq]

/-- What a nonnegative finite pattern decodes to: sign clear, significand
    below `2^53`, biased exponent at least the subnormal unit. -/
theorem decode_fin (c : Nat) (h : NonnegFin c) :
    (decode c).1 = false ∧ (decode c).2.1 < 2 ^ 53 ∧ 1126 ≤ ((decode c).2.2 + bias).toNat := by
  obtain ⟨h1, h2⟩ := h
  have hp : 0 < 2 ^ 52 := Nat.pos_pow_of_pos _ (by omega)
  have hlt : c / 2 ^ 52 < 2048 := by
    rw [Nat.div_lt_iff_lt_mul hp]; have := p63
    omega
  have hm := Nat.mod_lt c hp
  have h3 : ¬ c ≥ 2 ^ 63 := by omega
  have hb := p53
  simp only [isFinite, Nat.mod_eq_of_lt hlt] at h2
  simp only [decode, Nat.mod_eq_of_lt hlt, h3, decide_False, bias]
  simp only [bne_iff_ne, ne_eq] at h2
  generalize c / 2 ^ 52 = x at *
  generalize c % 2 ^ 52 = r at *
  split
  · simp; omega
  · simp only [true_and]
    have := Int.toNat_of_nonneg (show (0:Int) ≤ (x:Int) - 1075 + ((2200:Nat):Int) by omega)
    omega


/-! The rounding. -/

/-- `round` after the exact value is put over the biased unit. -/
def roundA (A D : Nat) : Option Nat :=
  let Eb := max (lg (A / D) - 52) 1126
  let q := rq A (D * 2 ^ Eb)
  if q < 2 ^ 52 then some q
  else if q = 2 ^ 53 then
    (if Eb + 1 - 1125 ≥ 2047 then none else some ((Eb + 1 - 1125) * 2 ^ 52))
  else if Eb - 1125 ≥ 2047 then none else some ((Eb - 1125) * 2 ^ 52 + (q - 2 ^ 52))

theorem round_eq (N D : Nat) (k : Int) : round N D k = roundA (N * 2 ^ (k + bias).toNat) D := rfl

/-- The binade facts: the exact value is below `2^53` grid units, and at
    least `2^52` of them when the unit is above the subnormal one. -/
theorem binade (A D : Nat) (hD : 0 < D) (hT : A / D < 2 ^ 6464) :
    A < 2 ^ 53 * (D * 2 ^ max (lg (A / D) - 52) 1126) ∧
    (1126 < max (lg (A / D) - 52) 1126 → 2 ^ 52 * (D * 2 ^ max (lg (A / D) - 52) 1126) ≤ A) := by
  generalize hL : lg (A / D) = L
  have hup : A < D * 2 ^ (L + 1) := by
    rcases Nat.eq_zero_or_pos (A / D) with h0 | h0
    · have : A < D := by
        rcases Nat.lt_or_ge A D with h | h
        · exact h
        · exact absurd h0 (Nat.pos_iff_ne_zero.mp ((Nat.le_div_iff_mul_le hD).mpr (by omega)))
      have : 1 ≤ 2 ^ (L + 1) := Nat.one_le_two_pow
      have : D * 1 ≤ D * 2 ^ (L + 1) := Nat.mul_le_mul_left _ this
      omega
    · have := (lg_spec _ h0 hT).2
      rw [hL] at this
      rw [Nat.mul_comm]; exact (Nat.div_lt_iff_lt_mul hD).mp this
  constructor
  · have : 2 ^ (L + 1) ≤ 2 ^ 53 * 2 ^ max (L - 52) 1126 := by
      rw [← Nat.pow_add]; exact Nat.pow_le_pow_right (by omega) (by omega)
    have := Nat.mul_le_mul_left D this
    rw [Nat.mul_left_comm] at this
    omega
  · intro hE
    have hEq : max (L - 52) 1126 = L - 52 := by omega
    rw [hEq]
    have h0 : 0 < A / D := by
      rcases Nat.eq_zero_or_pos (A / D) with h0 | h0
      · rw [h0] at hL; simp [lg, lgWords, lgBits] at hL; omega
      · exact h0
    have := (lg_spec _ h0 hT).1
    rw [hL] at this
    have h2 : D * 2 ^ L ≤ A := by
      have := (Nat.le_div_iff_mul_le hD).mp this; rw [Nat.mul_comm]; exact this
    rw [Nat.mul_left_comm, ← Nat.pow_add, show 52 + (L - 52) = L by omega]
    exact h2


/-- The value of a pattern over the biased unit, scaled by `D`:
    `Cv D c = c·D·2^bias` as a natural number. -/
def Cv (D c : Nat) : Nat := (decode c).2.1 * D * 2 ^ ((decode c).2.2 + bias).toNat

theorem toNat_b (x : Int) (h : 0 ≤ x) : ((x.toNat : Nat) : Int) = x := Int.toNat_of_nonneg h

set_option maxRecDepth 20000 in
/-- What `roundA` returns: a nonnegative finite pattern whose value is the
    rounded significand `q` on the grid `D·2^Eb`, and whose significand is
    even when `q` is. -/
theorem roundA_result (A D p : Nat) (hD : 0 < D) (hT : A / D < 2 ^ 6464) (h : roundA A D = some p) :
    NonnegFin p ∧
    Cv D p = rq A (D * 2 ^ max (lg (A / D) - 52) 1126) * (D * 2 ^ max (lg (A / D) - 52) 1126) ∧
    (rq A (D * 2 ^ max (lg (A / D) - 52) 1126) % 2 = 0 → (decode p).2.1 % 2 = 0) := by
  obtain ⟨hF1, hF2⟩ := binade A D hD hT
  unfold roundA at h
  simp only at h
  have hEb : 1126 ≤ max (lg (A / D) - 52) 1126 := Nat.le_max_right _ _
  generalize max (lg (A / D) - 52) 1126 = Eb at *
  have hden : 0 < D * 2 ^ Eb := Nat.mul_pos hD (Nat.pos_pow_of_pos _ (by omega))
  have hq53 : rq A (D * 2 ^ Eb) ≤ 2 ^ 53 := by
    have h1 := rq_le A (D * 2 ^ Eb)
    have h2 : A / (D * 2 ^ Eb) < 2 ^ 53 := (Nat.div_lt_iff_lt_mul hden).mpr hF1
    generalize A / (D * 2 ^ Eb) = t at *
    omega
  have hq52 : 1126 < Eb → 2 ^ 52 ≤ rq A (D * 2 ^ Eb) := by
    intro hE
    have h1 := rq_ge A (D * 2 ^ Eb)
    have h2 : 2 ^ 52 ≤ A / (D * 2 ^ Eb) := (Nat.le_div_iff_mul_le hden).mpr (hF2 hE)
    generalize A / (D * 2 ^ Eb) = t at *
    omega
  generalize rq A (D * 2 ^ Eb) = q at *
  split at h
  · rename_i hq
    cases h
    have hE : Eb = 1126 := by
      rcases Nat.lt_or_ge 1126 Eb with h1 | h1
      · have := hq52 h1; omega
      · omega
    subst hE
    refine ⟨sub_fin _ hq, ?_, ?_⟩
    · simp only [Cv, decode_sub _ hq, bias]; rw [Nat.mul_assoc]; rfl
    · simp only [decode_sub _ hq]; exact id
  · rename_i hq
    split at h
    · rename_i h53
      split at h
      · cases h
      · rename_i hex
        cases h
        have hd := decode_norm (Eb + 1 - 1125) 0 (by omega) (by omega) (by decide)
        rw [Nat.add_zero] at hd
        refine ⟨by have := norm_fin (Eb + 1 - 1125) 0 (by omega) (by decide); rwa [Nat.add_zero] at this, ?_, ?_⟩
        · unfold Cv; rw [hd]; dsimp only
          have e1 : (((Eb + 1 - 1125 : Nat) : Int) - 1075 + ((2200 : Nat) : Int)).toNat = Eb + 1 := by
            have := toNat_b (((Eb + 1 - 1125 : Nat) : Int) - 1075 + ((2200 : Nat) : Int)) (by omega)
            omega
          unfold bias; rw [e1, Nat.zero_add, h53, show (2:Nat) ^ (Eb + 1) = 2 ^ Eb * 2 from Nat.pow_succ 2 Eb, p53]
          ac_rfl
        · intro _; rw [hd]; exact (by decide : (0 + 2 ^ 52) % 2 = 0)
    · rename_i h53
      split at h
      · cases h
      · rename_i hex
        have h := Option.some.inj h; subst h
        clear hT hF1 hF2 hden
        have hr : q - 2 ^ 52 < 2 ^ 52 := by
          have := p53; generalize (2:Nat) ^ 52 = P at *; generalize (2:Nat) ^ 53 = Q at *; omega
        have hd := decode_norm (Eb - 1125) (q - 2 ^ 52) (by omega) (by omega) hr
        refine ⟨norm_fin _ _ (by omega) hr, ?_, ?_⟩
        · unfold Cv; rw [hd]; dsimp only; unfold bias
          have e1 : (((Eb - 1125 : Nat) : Int) - 1075 + ((2200 : Nat) : Int)).toNat = Eb := by
            have := toNat_b (((Eb - 1125 : Nat) : Int) - 1075 + ((2200 : Nat) : Int)) (by omega)
            omega
          rw [e1, Nat.sub_add_cancel (by omega), Nat.mul_assoc]
        · intro h2; rw [hd]; dsimp only; rw [Nat.sub_add_cancel (by omega)]; exact h2


/-- **Nearest, ties to even.** Among all nonnegative finite binary64
    values, the one `roundA` returns is nearest the exact value `A/D`
    (both scaled by `D·2^bias`), and on a tie with a different value its
    significand is even. -/
theorem roundA_nearest (A D p : Nat) (hD : 0 < D) (hT : A / D < 2 ^ 6464)
    (h : roundA A D = some p) (c : Nat) (hc : NonnegFin c) :
    ndist A (Cv D p) ≤ ndist A (Cv D c) ∧
    (ndist A (Cv D p) = ndist A (Cv D c) → Cv D c ≠ Cv D p → (decode p).2.1 % 2 = 0) := by
  obtain ⟨_, hv, hpar⟩ := roundA_result A D p hD hT h
  obtain ⟨_, hF2⟩ := binade A D hD hT
  obtain ⟨_, hm, he⟩ := decode_fin c hc
  rw [hv]
  have hEb : 1126 ≤ max (lg (A / D) - 52) 1126 := Nat.le_max_right _ _
  generalize max (lg (A / D) - 52) 1126 = Eb at *
  have hden : 0 < D * 2 ^ Eb := Nat.mul_pos hD (Nat.pos_pow_of_pos _ (by omega))
  have hcv : Cv D c = (decode c).2.1 * D * 2 ^ ((decode c).2.2 + bias).toNat := rfl
  rw [hcv]
  generalize (decode c).2.1 = m at *
  generalize ((decode c).2.2 + bias).toNat = ec at *
  clear hT
  rcases Nat.lt_or_ge ec Eb with hlt | hge
  · -- below the binade: farther than the grid point 2^52
    have hE : 1126 < Eb := by omega
    have h52 := hF2 hE
    have hcl : m * D * 2 ^ ec < 2 ^ 52 * (D * 2 ^ Eb) := by
      have h1 : m * (D * 2 ^ ec) < 2 ^ 53 * (D * 2 ^ ec) :=
        Nat.mul_lt_mul_of_pos_right hm (Nat.mul_pos hD (Nat.pos_pow_of_pos _ (by omega)))
      have h2 : 2 ^ 53 * (D * 2 ^ ec) ≤ 2 ^ 52 * (D * 2 ^ Eb) := by
        have e : 2 ^ 53 * (D * 2 ^ ec) = 2 ^ 52 * (D * 2 ^ (ec + 1)) := by
          rw [Nat.pow_succ 2 ec, p53]; ac_rfl
        rw [e]
        exact Nat.mul_le_mul_left _ (Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by omega) hlt))
      rw [Nat.mul_assoc]; omega
    have hn := rq_near A (D * 2 ^ Eb) (2 ^ 52) hden
    generalize rq A (D * 2 ^ Eb) * (D * 2 ^ Eb) = v at *
    generalize 2 ^ 52 * (D * 2 ^ Eb) = g at *
    generalize m * D * 2 ^ ec = w at *
    unfold ndist at *
    refine ⟨?_, ?_⟩
    · by_cases hv1 : v ≤ A <;> by_cases hg : g ≤ A <;> simp only [hv1, hg, if_true, if_false] at hn ⊢ <;>
        split <;> omega
    · intro he1
      by_cases hv1 : v ≤ A <;> by_cases hg : g ≤ A <;> simp only [hv1, hg, if_true, if_false] at hn he1 <;>
        split at he1 <;> omega
  · -- on the grid: the nearest integer
    have hc' : m * D * 2 ^ ec = (m * 2 ^ (ec - Eb)) * (D * 2 ^ Eb) := by
      rw [show ec = Eb + (ec - Eb) by omega, Nat.pow_add, Nat.add_sub_cancel_left]; ac_rfl
    rw [hc']
    refine ⟨rq_near _ _ _ hden, ?_⟩
    intro he1 hne
    exact hpar (rq_tie_even A _ hden (rq_tie A _ _ hden he1 hne))


/-- The IEEE 754 overflow threshold for roundTiesToEven: the exact value
    is at least `2^1024 - 2^970` (halfway between the largest finite value
    and `2^1024`), scaled by `D·2^bias`. -/
def Overflows (A D : Nat) : Prop := (2 ^ 54 - 1) * D * 2 ^ 3170 ≤ A

theorem pred_mul (Q d : Nat) (h : 1 ≤ Q) : (Q - 1) * d + d = Q * d := by
  rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel h]

/-- The floor of `A/d` pinned between `Q-1/2` and `Q`. -/
theorem floor_pin (A d t r Q : Nat) (hdm : t * d + r = A) (hr : r < d)
    (hup : A < Q * d) (hlo : 2 * (Q * d) ≤ 2 * A + d) (hQ : 2 ≤ Q) : t = Q - 1 ∧ d ≤ 2 * r := by
  have e1 := pred_mul Q d (by omega)
  have e2 := pred_mul (Q - 1) d (by omega)
  have ht1 : t ≤ Q - 1 := by
    rcases Nat.lt_or_ge t Q with h | h
    · omega
    · have := Nat.mul_le_mul_right d h; omega
  have ht2 : Q - 1 ≤ t := by
    rcases Nat.lt_or_ge t (Q - 1) with h | h
    · have := Nat.mul_le_mul_right d (show t ≤ Q - 1 - 1 by omega); omega
    · exact h
  have : t = Q - 1 := by omega
  subst this
  omega

theorem ovf_iff_gen (A D P M : Nat) : M * D * P ≤ A ↔ M * (D * (P * 2)) ≤ 2 * A := by
  rw [show M * (D * (P * 2)) = 2 * (M * D * P) by ac_rfl]
  generalize M * D * P = V; omega

theorem p3171 : (2:Nat) ^ 3171 = 2 ^ 3170 * 2 := Nat.pow_succ 2 3170
theorem p3172 : (2:Nat) ^ 3172 = 2 ^ 3170 * 2 ^ 2 := Nat.pow_add 2 3170 2
theorem p3223 : (2:Nat) ^ 3223 = 2 ^ 53 * 2 ^ 3170 := Nat.pow_add 2 53 3170

theorem ovf_iff (A D : Nat) : Overflows A D ↔ (2 ^ 54 - 1) * (D * 2 ^ 3171) ≤ 2 * A := by
  unfold Overflows; rw [p3171]; exact ovf_iff_gen _ _ _ _

theorem m54 (d : Nat) : (2 ^ 54 - 1) * d + d = 2 * (2 ^ 53 * d) := by
  rw [pred_mul _ _ (by decide), show (54:Nat) = 1 + 53 from rfl, Nat.pow_add, Nat.mul_assoc]

theorem mono_gen (Q P D M A : Nat) (hq : Q ≤ M) (h : M * D * P ≤ A) : Q * P * D ≤ A :=
  calc Q * P * D = Q * (D * P) := by ac_rfl
    _ ≤ M * (D * P) := Nat.mul_le_mul_right _ hq
    _ = M * D * P := by ac_rfl
    _ ≤ A := h

theorem big_gen (M K P D A : Nat) (hq : M ≤ K * 2 ^ 2) (h : K * (D * (P * 2 ^ 2)) ≤ A) : M * D * P ≤ A :=
  calc M * D * P = M * (D * P) := by ac_rfl
    _ ≤ (K * 2 ^ 2) * (D * P) := Nat.mul_le_mul_right _ hq
    _ = K * (D * (P * 2 ^ 2)) := by ac_rfl
    _ ≤ A := h

/-- Over the threshold the value is at least `2^3223` units of `D`. -/
theorem ovf_lg (A D : Nat) (hD : 0 < D) (hT : A / D < 2 ^ 6464) (ho : Overflows A D) :
    3223 ≤ lg (A / D) := by
  have h1 : 2 ^ 3223 ≤ A / D := by
    apply (Nat.le_div_iff_mul_le hD).mpr
    rw [p3223]
    exact mono_gen _ _ _ _ _ (by decide) ho
  have h2 := (lg_spec _ (Nat.lt_of_lt_of_le (Nat.pos_pow_of_pos _ (by omega)) h1) hT).2
  have := Nat.lt_of_le_of_lt h1 h2
  exact Nat.le_of_lt_succ ((Nat.pow_lt_pow_iff_right (by omega)).mp this)

/-- `2^52` grid units at `2^3172` or more are over the threshold. -/
theorem ovf_big (A D Eb : Nat) (hE : 3172 ≤ Eb) (h : 2 ^ 52 * (D * 2 ^ Eb) ≤ A) : Overflows A D := by
  unfold Overflows
  have h1 : 2 ^ 52 * (D * 2 ^ 3172) ≤ 2 ^ 52 * (D * 2 ^ Eb) :=
    Nat.mul_le_mul_left _ (Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by omega) hE))
  rw [p3172] at h1
  exact big_gen _ _ _ _ _ (by decide) (Nat.le_trans h1 h)

/-- **Overflow.** `roundA` gives no finite pattern exactly when the exact
    value reaches the IEEE overflow threshold. -/
theorem roundA_none (A D : Nat) (hD : 0 < D) (hT : A / D < 2 ^ 6464) :
    roundA A D = none ↔ Overflows A D := by
  obtain ⟨hF1, hF2⟩ := binade A D hD hT
  have hL := ovf_lg A D hD hT
  unfold roundA
  simp only
  have hEb : 1126 ≤ max (lg (A / D) - 52) 1126 := Nat.le_max_right _ _
  have hEL : lg (A / D) - 52 ≤ max (lg (A / D) - 52) 1126 := Nat.le_max_left _ _
  generalize lg (A / D) = L at *
  generalize max (L - 52) 1126 = Eb at *
  clear hT
  have hden : 0 < D * 2 ^ Eb := Nat.mul_pos hD (Nat.pos_pow_of_pos _ (by omega))
  have hq52 : 1126 < Eb → 2 ^ 52 ≤ rq A (D * 2 ^ Eb) := by
    intro hE
    have h1 := rq_ge A (D * 2 ^ Eb)
    have h2 : 2 ^ 52 ≤ A / (D * 2 ^ Eb) := (Nat.le_div_iff_mul_le hden).mpr (hF2 hE)
    generalize A / (D * 2 ^ Eb) = t at *
    omega
  have hhalf := rq_half A (D * 2 ^ Eb) hden
  have hdm := Nat.div_add_mod A (D * 2 ^ Eb)
  have hrm := Nat.mod_lt A hden
  have hcases := rq_cases A (D * 2 ^ Eb)
  generalize rq A (D * 2 ^ Eb) = q at *
  constructor
  · intro h
    split at h
    · cases h
    · split at h
      · rename_i h53
        split at h
        · rename_i hex
          rcases Nat.lt_or_ge Eb 3172 with h1 | h1
          · have hE : Eb = 3171 := by omega
            subst hE
            rw [ovf_iff]
            have hm := m54 (D * 2 ^ 3171)
            unfold ndist at hhalf
            rw [h53] at hhalf
            generalize D * 2 ^ 3171 = d at *
            generalize 2 ^ 53 * d = W at *
            generalize (2 ^ 54 - 1) * d = V at *
            split at hhalf <;> omega
          · exact ovf_big A D Eb h1 (hF2 (by omega))
        · cases h
      · split at h
        · rename_i hex; exact ovf_big A D Eb (by omega) (hF2 (by omega))
        · cases h
  · intro ho
    have hL' := hL ho
    have hE : 3171 ≤ Eb := by omega
    have hq := hq52 (by omega)
    split
    · omega
    · split
      · rename_i h53; rw [if_pos (by omega)]
      · rename_i h53
        rw [if_pos]
        rcases Nat.lt_or_ge Eb 3172 with h1 | h1
        · exfalso
          have hE' : Eb = 3171 := by omega
          subst hE'
          rw [ovf_iff] at ho
          have hm := m54 (D * 2 ^ 3171)
          rw [Nat.mul_comm (D * 2 ^ 3171)] at hdm
          generalize D * 2 ^ 3171 = d at *
          generalize A / d = t at *
          generalize A % d = r at *
          obtain ⟨ht, hr2⟩ := floor_pin A d t r (2 ^ 53) hdm hrm hF1
            (by generalize 2 ^ 53 * d = W at *; generalize (2 ^ 54 - 1) * d = V at *; omega) (by decide)
          subst ht
          have hodd : (2 ^ 53 - 1) % 2 = 1 := by decide
          rcases hcases with ⟨h1, h2⟩ | ⟨h1, h2⟩
          · exact h53 (by rw [h1])
          · exact h2 (by omega)
        · omega


/-! The operations. -/

/-- `o` is the roundTiesToEven result for the positive exact value
    `x = N/D·2^k` with `A = N·2^(k+bias)` (so `A = x·D·2^bias`): a pattern
    nearest `x` among all nonnegative finite binary64 values, with an even
    significand on a tie, or `none` exactly when IEEE 754 overflows. -/
def RNE (N D : Nat) (k : Int) (o : Option Nat) : Prop :=
  let A := N * 2 ^ (k + bias).toNat
  (∀ p, o = some p → NonnegFin p ∧ ∀ c, NonnegFin c →
      ndist A (Cv D p) ≤ ndist A (Cv D c) ∧
      (ndist A (Cv D p) = ndist A (Cv D c) → Cv D c ≠ Cv D p → (decode p).2.1 % 2 = 0)) ∧
  (o = none ↔ Overflows A D)

/-- **`round` is correctly rounded.** -/
theorem round_rne (N D : Nat) (k : Int) (hD : 0 < D) (hT : N * 2 ^ (k + bias).toNat / D < 2 ^ 6464) :
    RNE N D k (round N D k) := by
  rw [round_eq]
  refine ⟨fun p hp => ⟨(roundA_result _ _ p hD hT hp).1, fun c hc => roundA_nearest _ _ p hD hT hp c hc⟩,
    roundA_none _ _ hD hT⟩

/-- Ranges of a finite pattern's decoding (either sign). -/
theorem decode_range (b : Nat) (hf : isFinite b = true) :
    (decode b).2.1 < 2 ^ 53 ∧ -1074 ≤ (decode b).2.2 ∧ (decode b).2.2 ≤ 971 := by
  have hp : 0 < 2 ^ 52 := Nat.pos_pow_of_pos _ (by omega)
  have hm := Nat.mod_lt b hp
  have hx := Nat.mod_lt (b / 2 ^ 52) (show 0 < 2048 by omega)
  simp only [isFinite, bne_iff_ne, ne_eq] at hf
  simp only [decode]
  generalize (b / 2 ^ 52) % 2048 = x at *
  generalize b % 2 ^ 52 = r at *
  split
  · simp; omega
  · simp only; refine ⟨by omega, by omega, by omega⟩

theorem pow_toNat_le (a b : Int) (h : a ≤ b) (h0 : 0 ≤ a) : 2 ^ a.toNat ≤ 2 ^ b.toNat :=
  Nat.pow_le_pow_right (by omega) (by
    have := Int.toNat_of_nonneg h0; have := Int.toNat_of_nonneg (Int.le_trans h0 h); omega)

/-- **Multiplication** of finite nonzero operands is the correctly rounded
    exact product, with the sign of the product. -/
theorem mul_rne (x y : Nat) (hx : isFinite x = true) (hy : isFinite y = true)
    (hm : (decode x).2.1 * (decode y).2.1 ≠ 0) :
    mul x y = (round ((decode x).2.1 * (decode y).2.1) 1 ((decode x).2.2 + (decode y).2.2)).map
      (withSign ((decode x).1 != (decode y).1)) ∧
    RNE ((decode x).2.1 * (decode y).2.1) 1 ((decode x).2.2 + (decode y).2.2)
      (round ((decode x).2.1 * (decode y).2.1) 1 ((decode x).2.2 + (decode y).2.2)) := by
  obtain ⟨m1, l1, u1⟩ := decode_range x hx
  obtain ⟨m2, l2, u2⟩ := decode_range y hy
  refine ⟨?_, round_rne _ _ _ (by omega) ?_⟩
  · simp only [mul]; rw [if_neg hm]
  · rw [Nat.div_one]
    generalize (decode x).2.1 = a at *; generalize (decode y).2.1 = b at *
    generalize (decode x).2.2 = e at *; generalize (decode y).2.2 = f at *
    have h1 : a * b < 2 ^ 53 * 2 ^ 53 :=
      Nat.lt_of_le_of_lt (Nat.mul_le_mul_left a (Nat.le_of_lt m2))
        (Nat.mul_lt_mul_of_pos_right m1 (Nat.pos_pow_of_pos _ (by omega)))
    have h2 : 2 ^ (e + f + bias).toNat ≤ 2 ^ (4142 : Int).toNat :=
      pow_toNat_le _ _ (by unfold bias; omega) (by unfold bias; omega)
    have h3 : a * b * 2 ^ (e + f + bias).toNat < 2 ^ 53 * 2 ^ 53 * 2 ^ (4142 : Int).toNat :=
      Nat.lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right h1 (Nat.pos_pow_of_pos _ (by omega)))
        (Nat.mul_le_mul_left _ h2)
    refine Nat.lt_of_lt_of_le h3 ?_
    rw [← Nat.pow_add, ← Nat.pow_add]
    exact Nat.pow_le_pow_right (by omega) (by decide)


/-- **Division** of a finite nonzero operand by a finite nonzero operand is
    the correctly rounded exact quotient, with the sign of the quotient. -/
theorem div_rne (x y : Nat) (hx : isFinite x = true) (hy : isFinite y = true)
    (h1 : (decode x).2.1 ≠ 0) (h2 : (decode y).2.1 ≠ 0) :
    div x y = (round (decode x).2.1 (decode y).2.1 ((decode x).2.2 - (decode y).2.2)).map
      (withSign ((decode x).1 != (decode y).1)) ∧
    RNE (decode x).2.1 (decode y).2.1 ((decode x).2.2 - (decode y).2.2)
      (round (decode x).2.1 (decode y).2.1 ((decode x).2.2 - (decode y).2.2)) := by
  obtain ⟨m1, l1, u1⟩ := decode_range x hx
  obtain ⟨m2, l2, u2⟩ := decode_range y hy
  refine ⟨?_, round_rne _ _ _ (by omega) ?_⟩
  · simp only [div]; rw [if_neg h2, if_neg h1]
  · generalize (decode x).2.1 = a at *; generalize (decode y).2.1 = b at *
    generalize (decode x).2.2 = e at *; generalize (decode y).2.2 = f at *
    refine Nat.lt_of_le_of_lt (Nat.div_le_self _ _) ?_
    have h2 : 2 ^ (e - f + bias).toNat ≤ 2 ^ (4245 : Int).toNat :=
      pow_toNat_le _ _ (by unfold bias; omega) (by unfold bias; omega)
    have h3 : a * 2 ^ (e - f + bias).toNat < 2 ^ 53 * 2 ^ (4245 : Int).toNat :=
      Nat.lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right m1 (Nat.pos_pow_of_pos _ (by omega)))
        (Nat.mul_le_mul_left _ h2)
    refine Nat.lt_of_lt_of_le h3 ?_
    rw [← Nat.pow_add]
    exact Nat.pow_le_pow_right (by omega) (by decide)

theorem natAbs_mp (m n : Nat) : ((m:Int) * 2 ^ n).natAbs = m * 2 ^ n := by
  rw [show ((m:Int) * 2 ^ n) = ((m * 2 ^ n : Nat) : Int) by rw [Int.ofNat_mul, Int.natCast_pow]; rfl]
  exact Int.natAbs_ofNat _

theorem atUnit_abs (s : Bool) (m : Nat) (e u : Int) : (atUnit s m e u).natAbs = m * 2 ^ (e - u).toNat := by
  unfold atUnit; cases s <;> simp only [Int.natAbs_neg, natAbs_mp, ite_true, ite_false, cond_true, cond_false] <;> first | rfl | simp [Int.natAbs_neg, natAbs_mp]

/-- **Addition** (and so subtraction, `sub x y = add x (neg y)`) of finite
    operands: an exact zero sum is `+0` unless both are negative (IEEE 754
    6.3), otherwise the correctly rounded exact sum `S·2^u` with its sign. -/
theorem add_rne (x y : Nat) (hx : isFinite x = true) (hy : isFinite y = true) :
    let u := min (decode x).2.2 (decode y).2.2
    let S := atUnit (decode x).1 (decode x).2.1 (decode x).2.2 u +
      atUnit (decode y).1 (decode y).2.1 (decode y).2.2 u
    add x y = (if S = 0 then some (withSign ((decode x).1 && (decode y).1) 0)
      else (round S.natAbs 1 u).map (withSign (decide (S < 0)))) ∧
    RNE S.natAbs 1 u (round S.natAbs 1 u) := by
  intro u S
  obtain ⟨m1, l1, u1⟩ := decode_range x hx
  obtain ⟨m2, l2, u2⟩ := decode_range y hy
  refine ⟨rfl, round_rne _ _ _ (by omega) ?_⟩
  rw [Nat.div_one]
  have hS : S.natAbs ≤ (decode x).2.1 * 2 ^ ((decode x).2.2 - u).toNat +
      (decode y).2.1 * 2 ^ ((decode y).2.2 - u).toNat := by
    have := Int.natAbs_add_le (atUnit (decode x).1 (decode x).2.1 (decode x).2.2 u)
      (atUnit (decode y).1 (decode y).2.1 (decode y).2.2 u)
    rw [atUnit_abs, atUnit_abs] at this; exact this
  have hu1 : -1074 ≤ u := by simp only [u]; omega
  have hu2 : u ≤ 971 := by simp only [u]; omega
  have hd1 : (decode x).2.2 - u ≤ 2045 := by simp only [u]; omega
  have hd2 : (decode y).2.2 - u ≤ 2045 := by simp only [u]; omega
  have hd1' : 0 ≤ (decode x).2.2 - u := by simp only [u]; omega
  have hd2' : 0 ≤ (decode y).2.2 - u := by simp only [u]; omega
  generalize S.natAbs = n at *
  generalize (decode x).2.1 = a at *; generalize (decode y).2.1 = b at *
  generalize (decode x).2.2 = e at *; generalize (decode y).2.2 = f at *
  have p1 := pow_toNat_le (e - u) 2045 hd1 hd1'
  have p2 := pow_toNat_le (f - u) 2045 hd2 hd2'
  have p3 := pow_toNat_le (u + bias) 3171 (by unfold bias; omega) (by unfold bias; omega)
  have q1 : a * 2 ^ (e - u).toNat ≤ 2 ^ 53 * 2 ^ (2045 : Int).toNat :=
    Nat.mul_le_mul (Nat.le_of_lt m1) p1
  have q2 : b * 2 ^ (f - u).toNat ≤ 2 ^ 53 * 2 ^ (2045 : Int).toNat :=
    Nat.mul_le_mul (Nat.le_of_lt m2) p2
  have hn : n ≤ 2 * (2 ^ 53 * 2 ^ (2045 : Int).toNat) := by omega
  have := Nat.mul_le_mul hn p3
  refine Nat.lt_of_le_of_lt this ?_
  rw [← Nat.pow_add, show 2 * 2 ^ (53 + (2045:Int).toNat) = 2 ^ (1 + (53 + (2045:Int).toNat)) by
    rw [Nat.pow_add 2 1], ← Nat.pow_add]
  exact Nat.pow_lt_pow_of_lt (by omega) (by decide)

/-- **Integer to float** (`x as f64`) is the correctly rounded integer. -/
theorem ofInt_rne (x : Int) (hx : x ≠ 0) (hb : x.natAbs < 2 ^ 4264) :
    ofInt x = (round x.natAbs 1 0).map (withSign (decide (x < 0))) ∧
    RNE x.natAbs 1 0 (round x.natAbs 1 0) := by
  refine ⟨by simp [ofInt, hx], round_rne _ _ _ (by omega) ?_⟩
  rw [Nat.div_one]
  have : ((0:Int) + bias).toNat = 2200 := rfl
  rw [this]
  refine Nat.lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right hb (Nat.pos_pow_of_pos _ (by omega))) ?_
  rw [← Nat.pow_add]
  exact Nat.pow_le_pow_right (by omega) (by decide)

end RCE.F64
