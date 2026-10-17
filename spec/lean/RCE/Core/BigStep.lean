import RCE.Core.Fuel
/-! S1: a relational big-step semantics with no fuel, and its agreement
    with the fuel-indexed interpreter: `eval` with any fuel gives a value
    or trap only if the relation does (`eval_bigstep`), and every relation
    derivation is reached by `eval` at some fuel (`bigstep_eval`). So the
    fuel is a bound on a fuel-free meaning, never a source of it. -/
namespace RCE.Core

/-- Values and traps: the results the relation derives. -/
def Final : Res → Prop
  | .ok _ => True
  | .err _ => True
  | _ => False

inductive BS (P : Prog) : List Val → Expr → Res → Prop where
  | var {ρ i v} : ρ.get? i = some v → BS P ρ (.var i) (.ok v)
  | lit {ρ v} : BS P ρ (.lit v) (.ok v)
  | bin {ρ op ta a b x y} : BS P ρ a (.ok x) → BS P ρ b (.ok y) → Final (binop op ta x y) →
      BS P ρ (.bin op ta a b) (binop op ta x y)
  | bin_ea {ρ op ta a b t} : BS P ρ a (.err t) → BS P ρ (.bin op ta a b) (.err t)
  | bin_eb {ρ op ta a b x t} : BS P ρ a (.ok x) → BS P ρ b (.err t) → BS P ρ (.bin op ta a b) (.err t)
  | ite_t {ρ c a b r} : BS P ρ c (.ok (.bool true)) → BS P ρ a r → BS P ρ (.ite c a b) r
  | ite_f {ρ c a b r} : BS P ρ c (.ok (.bool false)) → BS P ρ b r → BS P ρ (.ite c a b) r
  | ite_e {ρ c a b t} : BS P ρ c (.err t) → BS P ρ (.ite c a b) (.err t)
  | let_ {ρ a b x r} : BS P ρ a (.ok x) → BS P (x :: ρ) b r → BS P ρ (.let_ a b) r
  | let_e {ρ a b t} : BS P ρ a (.err t) → BS P ρ (.let_ a b) (.err t)
  | pair {ρ a b x y} : BS P ρ a (.ok x) → BS P ρ b (.ok y) → BS P ρ (.pair a b) (.ok (.pair x y))
  | pair_ea {ρ a b t} : BS P ρ a (.err t) → BS P ρ (.pair a b) (.err t)
  | pair_eb {ρ a b x t} : BS P ρ a (.ok x) → BS P ρ b (.err t) → BS P ρ (.pair a b) (.err t)
  | fst {ρ a x y} : BS P ρ a (.ok (.pair x y)) → BS P ρ (.fst a) (.ok x)
  | fst_e {ρ a t} : BS P ρ a (.err t) → BS P ρ (.fst a) (.err t)
  | snd {ρ a x y} : BS P ρ a (.ok (.pair x y)) → BS P ρ (.snd a) (.ok y)
  | snd_e {ρ a t} : BS P ρ a (.err t) → BS P ρ (.snd a) (.err t)
  | inl {ρ a x} : BS P ρ a (.ok x) → BS P ρ (.inl a) (.ok (.inl x))
  | inl_e {ρ a t} : BS P ρ a (.err t) → BS P ρ (.inl a) (.err t)
  | inr {ρ a x} : BS P ρ a (.ok x) → BS P ρ (.inr a) (.ok (.inr x))
  | inr_e {ρ a t} : BS P ρ a (.err t) → BS P ρ (.inr a) (.err t)
  | case_l {ρ s l r x q} : BS P ρ s (.ok (.inl x)) → BS P (x :: ρ) l q → BS P ρ (.case s l r) q
  | case_r {ρ s l r x q} : BS P ρ s (.ok (.inr x)) → BS P (x :: ρ) r q → BS P ρ (.case s l r) q
  | case_e {ρ s l r t} : BS P ρ s (.err t) → BS P ρ (.case s l r) (.err t)
  | raise {ρ c} : BS P ρ (.raise c) (.err (.user c))
  | catch_h {ρ a h c r} : BS P ρ a (.err (.user c)) → BS P ρ h r → BS P ρ (.catch_ a h) r
  | catch_ok {ρ a h v} : BS P ρ a (.ok v) → BS P ρ (.catch_ a h) (.ok v)
  | catch_t {ρ a h t} : BS P ρ a (.err t) → (∀ c, t ≠ .user c) → BS P ρ (.catch_ a h) (.err t)
  | call {ρ f a x ta t body r} : BS P ρ a (.ok x) → P.get? f = some (ta, t, body) →
      BS P [x] (.tick body) r → BS P ρ (.call f a) r
  | call_e {ρ f a t} : BS P ρ a (.err t) → BS P ρ (.call f a) (.err t)
  | tick {ρ e r} : BS P ρ e r → BS P ρ (.tick e) r
  | try_r {ρ e c v} : BS P ρ e (.ok (.inr v)) → BS P ρ (.try_ e c) (.ok v)
  | try_l {ρ e c v} : BS P ρ e (.ok (.inl v)) → BS P ρ (.try_ e c) (.err (.user c))
  | try_e {ρ e c t} : BS P ρ e (.err t) → BS P ρ (.try_ e c) (.err t)
  | cast {ρ lo hi e x} : BS P ρ e (.ok (.int x)) → BS P ρ (.cast lo hi e) (chk lo hi x)
  | cast_e {ρ lo hi e t} : BS P ρ e (.err t) → BS P ρ (.cast lo hi e) (.err t)
  | fcast {ρ lo hi e b} : BS P ρ e (.ok (.f64 b)) → BS P ρ (.cast lo hi e) (chk lo hi (RCE.F64.trunc b))
  | tof {ρ e x} : BS P ρ e (.ok (.int x)) → BS P ρ (.tof e) (fres (RCE.F64.ofInt x))
  | tof_e {ρ e t} : BS P ρ e (.err t) → BS P ρ (.tof e) (.err t)
  | un {ρ op ta e x} : BS P ρ e (.ok x) → Final (unop op ta x) → BS P ρ (.un op ta e) (unop op ta x)
  | un_e {ρ op ta e t} : BS P ρ e (.err t) → BS P ρ (.un op ta e) (.err t)

theorem BS.final {P ρ e r} (h : BS P ρ e r) : Final r := by
  induction h <;> first | exact True.intro | assumption | (simp only [chk]; split <;> exact True.intro) | (simp only [fres]; split <;> exact True.intro)

/-- **Interpreter to relation.** -/
theorem eval_bigstep (P : Prog) :
    ∀ n ρ e, Final (eval P n ρ e) → BS P ρ e (eval P n ρ e) := by
  intro n
  induction n with
  | zero => intro ρ e h; simp [eval, Final] at h
  | succ n ih =>
    intro ρ e hf
    have IH : ∀ ρ a r, eval P n ρ a = r → Final r → BS P ρ a r := by
      intro ρ a r h hr; subst h; exact ih ρ a hr
    cases e with
    | var i =>
      simp only [eval] at hf ⊢
      cases hv : ρ.get? i with
      | none => simp only [hv, Final] at hf
      | some v => simp only [hv]; exact .var hv
    | lit v => simp only [eval]; exact .lit
    | bin op ta a b =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ a <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · rename_i x
        cases hb : eval P n ρ b <;> simp only [hb] at hf ⊢ <;> try simp [Final] at hf
        · exact .bin (IH _ _ _ ha trivial) (IH _ _ _ hb trivial) hf
        · exact .bin_eb (IH _ _ _ ha trivial) (IH _ _ _ hb trivial)
      · exact .bin_ea (IH _ _ _ ha trivial)
    | ite c a b =>
      simp only [eval] at hf ⊢
      cases hc : eval P n ρ c <;> simp only [hc] at hf ⊢ <;> try simp [Final] at hf
      · rename_i v
        cases v <;> (try simp only at hf ⊢) <;> try simp [Final] at hf
        rename_i bb
        cases bb <;> (try simp only at hf ⊢)
        · exact .ite_f (IH _ _ _ hc trivial) (ih _ _ hf)
        · exact .ite_t (IH _ _ _ hc trivial) (ih _ _ hf)
      · exact .ite_e (IH _ _ _ hc trivial)
    | let_ a b =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ a <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · exact .let_ (IH _ _ _ ha trivial) (ih _ _ hf)
      · exact .let_e (IH _ _ _ ha trivial)
    | pair a b =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ a <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · cases hb : eval P n ρ b <;> simp only [hb] at hf ⊢ <;> try simp [Final] at hf
        · exact .pair (IH _ _ _ ha trivial) (IH _ _ _ hb trivial)
        · exact .pair_eb (IH _ _ _ ha trivial) (IH _ _ _ hb trivial)
      · exact .pair_ea (IH _ _ _ ha trivial)
    | fst a =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ a <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · rename_i v; cases v <;> simp [Final] at hf ⊢; exact .fst (IH _ _ _ ha trivial)
      · exact .fst_e (IH _ _ _ ha trivial)
    | snd a =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ a <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · rename_i v; cases v <;> simp [Final] at hf ⊢; exact .snd (IH _ _ _ ha trivial)
      · exact .snd_e (IH _ _ _ ha trivial)
    | inl a =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ a <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · exact .inl (IH _ _ _ ha trivial)
      · exact .inl_e (IH _ _ _ ha trivial)
    | inr a =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ a <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · exact .inr (IH _ _ _ ha trivial)
      · exact .inr_e (IH _ _ _ ha trivial)
    | case s l r =>
      simp only [eval] at hf ⊢
      cases hs : eval P n ρ s <;> simp only [hs] at hf ⊢ <;> try simp [Final] at hf
      · rename_i v
        cases v <;> (try simp only at hf ⊢) <;> try simp [Final] at hf
        · exact .case_l (IH _ _ _ hs trivial) (ih _ _ hf)
        · exact .case_r (IH _ _ _ hs trivial) (ih _ _ hf)
      · exact .case_e (IH _ _ _ hs trivial)
    | raise c => simp only [eval]; exact .raise
    | catch_ a h =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ a <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · exact .catch_ok (IH _ _ _ ha trivial)
      · rename_i t
        cases t with
        | user c => simp only at hf ⊢; exact .catch_h (IH _ _ _ ha trivial) (ih _ _ hf)
        | _ => (try simp only at hf ⊢); exact .catch_t (IH _ _ _ ha trivial) (by intro c h; cases h)
    | call f a =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ a with
      | ok x =>
        simp only [ha] at hf ⊢
        cases hp : P.get? f with
        | none => simp only [hp, Final] at hf
        | some fb =>
          obtain ⟨ta, t, body⟩ := fb
          simp only [hp] at hf ⊢
          exact .call (IH _ _ _ ha trivial) hp (ih _ _ hf)
      | err t => simp only [ha]; exact .call_e (IH _ _ _ ha trivial)
      | oof => simp only [ha, Final] at hf
      | stuck => simp only [ha, Final] at hf
    | tick e => simp only [eval] at hf ⊢; exact .tick (ih _ _ hf)
    | try_ e c =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ e <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · rename_i v
        cases v <;> (try simp only at hf ⊢) <;> try simp [Final] at hf
        · exact .try_l (IH _ _ _ ha trivial)
        · exact .try_r (IH _ _ _ ha trivial)
      · exact .try_e (IH _ _ _ ha trivial)
    | cast lo hi e =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ e <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · rename_i v
        cases v <;> (try simp only at hf ⊢) <;> try simp [Final] at hf
        · exact .cast (IH _ _ _ ha trivial)
        · exact .fcast (IH _ _ _ ha trivial)
      · exact .cast_e (IH _ _ _ ha trivial)
    | tof e =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ e <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · rename_i v
        cases v <;> (try simp only at hf ⊢) <;> try simp [Final] at hf
        exact .tof (IH _ _ _ ha trivial)
      · exact .tof_e (IH _ _ _ ha trivial)
    | un op ta e =>
      simp only [eval] at hf ⊢
      cases ha : eval P n ρ e <;> simp only [ha] at hf ⊢ <;> try simp [Final] at hf
      · exact .un (IH _ _ _ ha trivial) hf
      · exact .un_e (IH _ _ _ ha trivial)

theorem lift {P n ρ e r} (h : eval P n ρ e = r) (hr : Final r) (m : Nat) (hm : n ≤ m) :
    eval P m ρ e = r := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hm
  rw [eval_mono_le P (by rw [h]; intro c; subst c; exact hr)]; exact h

/-- **Relation to interpreter.** -/
theorem bigstep_eval {P ρ e r} (h : BS P ρ e r) : ∃ n, eval P n ρ e = r := by
  induction h with
  | var hv => exact ⟨1, by simp only [eval, hv]⟩
  | lit => exact ⟨1, rfl⟩
  | raise => exact ⟨1, rfl⟩
  | bin ha hb _ iha ihb =>
    obtain ⟨n1, h1⟩ := iha; obtain ⟨n2, h2⟩ := ihb
    exact ⟨n1 + n2 + 1, by simp only [eval, lift h1 ha.final (n1 + n2) (by omega), lift h2 hb.final (n1 + n2) (by omega)]⟩
  | bin_ea ha iha =>
    obtain ⟨n1, h1⟩ := iha
    exact ⟨n1 + 1, by simp only [eval, lift h1 ha.final n1 (by omega)]⟩
  | bin_eb ha hb iha ihb =>
    obtain ⟨n1, h1⟩ := iha; obtain ⟨n2, h2⟩ := ihb
    exact ⟨n1 + n2 + 1, by simp only [eval, lift h1 ha.final (n1 + n2) (by omega), lift h2 hb.final (n1 + n2) (by omega)]⟩
  | ite_t hc ha ihc iha =>
    obtain ⟨n1, h1⟩ := ihc; obtain ⟨n2, h2⟩ := iha
    exact ⟨n1 + n2 + 1, by simp only [eval, lift h1 hc.final (n1 + n2) (by omega), lift h2 ha.final (n1 + n2) (by omega)]⟩
  | ite_f hc ha ihc iha =>
    obtain ⟨n1, h1⟩ := ihc; obtain ⟨n2, h2⟩ := iha
    exact ⟨n1 + n2 + 1, by simp only [eval, lift h1 hc.final (n1 + n2) (by omega), lift h2 ha.final (n1 + n2) (by omega)]⟩
  | ite_e hc ihc =>
    obtain ⟨n1, h1⟩ := ihc
    exact ⟨n1 + 1, by simp only [eval, lift h1 hc.final n1 (by omega)]⟩
  | let_ ha hb iha ihb =>
    obtain ⟨n1, h1⟩ := iha; obtain ⟨n2, h2⟩ := ihb
    exact ⟨n1 + n2 + 1, by simp only [eval, lift h1 ha.final (n1 + n2) (by omega), lift h2 hb.final (n1 + n2) (by omega)]⟩
  | let_e ha iha =>
    obtain ⟨n1, h1⟩ := iha
    exact ⟨n1 + 1, by simp only [eval, lift h1 ha.final n1 (by omega)]⟩
  | pair ha hb iha ihb =>
    obtain ⟨n1, h1⟩ := iha; obtain ⟨n2, h2⟩ := ihb
    exact ⟨n1 + n2 + 1, by simp only [eval, lift h1 ha.final (n1 + n2) (by omega), lift h2 hb.final (n1 + n2) (by omega)]⟩
  | pair_ea ha iha =>
    obtain ⟨n1, h1⟩ := iha
    exact ⟨n1 + 1, by simp only [eval, lift h1 ha.final n1 (by omega)]⟩
  | pair_eb ha hb iha ihb =>
    obtain ⟨n1, h1⟩ := iha; obtain ⟨n2, h2⟩ := ihb
    exact ⟨n1 + n2 + 1, by simp only [eval, lift h1 ha.final (n1 + n2) (by omega), lift h2 hb.final (n1 + n2) (by omega)]⟩
  | fst ha iha | fst_e ha iha | snd ha iha | snd_e ha iha | inl ha iha | inl_e ha iha
  | inr ha iha | inr_e ha iha | case_e ha iha | call_e ha iha | try_r ha iha | try_l ha iha
  | try_e ha iha | cast ha iha | cast_e ha iha | fcast ha iha | tof ha iha | tof_e ha iha
  | un ha _ iha | un_e ha iha
  | catch_ok ha iha =>
    obtain ⟨n1, h1⟩ := iha
    exact ⟨n1 + 1, by simp only [eval, lift h1 ha.final n1 (by omega)]⟩
  | catch_t ha hn iha =>
    obtain ⟨n1, h1⟩ := iha
    refine ⟨n1 + 1, ?_⟩
    simp only [eval, lift h1 ha.final n1 (by omega)]
    rename_i t
    cases t with
    | user c => exact absurd rfl (hn c)
    | _ => rfl
  | case_l hs hl ihs ihl | case_r hs hl ihs ihl | catch_h hs hl ihs ihl =>
    obtain ⟨n1, h1⟩ := ihs; obtain ⟨n2, h2⟩ := ihl
    exact ⟨n1 + n2 + 1, by simp only [eval, lift h1 hs.final (n1 + n2) (by omega), lift h2 hl.final (n1 + n2) (by omega)]⟩
  | call ha hp hb iha ihb =>
    obtain ⟨n1, h1⟩ := iha; obtain ⟨n2, h2⟩ := ihb
    exact ⟨n1 + n2 + 1, by simp only [eval, lift h1 ha.final (n1 + n2) (by omega), hp, lift h2 hb.final (n1 + n2) (by omega)]⟩
  | tick ha iha =>
    obtain ⟨n1, h1⟩ := iha
    exact ⟨n1 + 1, by simp only [eval, lift h1 ha.final n1 (by omega)]⟩

/-- Agreement: a final result is the interpreter's at some fuel iff the
    relation derives it. -/
theorem eval_iff_bigstep (P : Prog) (ρ : List Val) (e : Expr) (r : Res) (hr : Final r) :
    (∃ n, eval P n ρ e = r) ↔ BS P ρ e r :=
  ⟨fun ⟨n, h⟩ => h ▸ eval_bigstep P n ρ e (h ▸ hr), bigstep_eval⟩

end RCE.Core
