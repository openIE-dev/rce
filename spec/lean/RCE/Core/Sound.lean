import RCE.Core.Lang
/-! S1 theorems for the core: type soundness of the definitional
    interpreter (a well-typed program is never `stuck`, and a value it
    returns has the static type), totality (the result is a value, a
    defined trap or exhaustion), and determinism. -/
namespace RCE.Core

theorem env_lookup : ∀ {Γ ρ} (i : Nat) {t}, WtEnv Γ ρ → Γ.get? i = some t →
    ∃ v, ρ.get? i = some v ∧ VTy v t
  | _, _, _, _, .nil, h => by simp at h
  | _, _, 0, _, .cons hv _, h => by simp at h; subst h; exact ⟨_, rfl, hv⟩
  | _, _, i+1, _, .cons _ hr, h => by simpa using env_lookup i hr (by simpa using h)

theorem sig_lookup : ∀ (P : Prog) (f : Nat) {ta t}, (sig P).get? f = some (ta, t) →
    ∃ b, P.get? f = some (ta, t, b)
  | [], _, _, _, h => by simp [sig] at h
  | d :: _, 0, _, _, h => by
      simp [sig] at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨d.2.2, rfl⟩
  | _ :: P, f+1, _, _, h => by simpa using sig_lookup P f (by simpa [sig] using h)

theorem chk_ty {lo hi r} : RTy (chk lo hi r) (.int lo hi) := by
  unfold chk; split
  · rename_i h; exact .int h.1 h.2
  · trivial

theorem fres_ty (o : Option Nat) : RTy (fres o) .f64 := by
  cases o
  · trivial
  · exact .f64

theorem fres32_ty (o : Option Nat) : RTy (fres32 o) .f32 := by
  cases o
  · trivial
  · exact .f32

set_option maxHeartbeats 4000000 in
theorem binop_sound {op ta t x y} (ho : OpTy op ta t) (hx : VTy x ta) (hy : VTy y ta) :
    RTy (binop op ta x y) t := by
  cases ho <;> cases hx <;> cases hy <;> simp only [binop, ibinop, fbinop, fbinop32]
  all_goals first
    | exact fres_ty _
    | exact fres32_ty _
    | exact chk_ty
    | exact .bool
    | exact .f64
    | exact .text
    | (split <;> first | trivial | exact chk_ty)
    | (split <;> (try split) <;> first | trivial | exact chk_ty)

theorem unop_sound {op ta t x} (ho : UnTy op ta t) (hx : VTy x ta) : RTy (unop op ta x) t := by
  cases ho <;> cases hx <;> simp only [unop]
  all_goals first | exact fres_ty _ | exact fres32_ty _ | exact chk_ty | exact .f64 | exact .f32

/-- Results that are not values pass through any context. -/
theorem rty_pass {r t t'} (h : RTy r t) (hn : ∀ v, r ≠ .ok v) : RTy r t' := by
  cases r <;> simp_all [RTy]

/-- **Type soundness.** In a well-typed program, a well-typed expression
    evaluated in an agreeing environment on any fuel is never stuck, and
    a value it yields has the expression's type. -/
theorem eval_sound (P : Prog) (hP : WtProg P) :
    ∀ n {Γ ρ e t}, WtEnv Γ ρ → HasTy (sig P) Γ e t → RTy (eval P n ρ e) t := by
  intro n
  induction n with
  | zero => intros; simp [eval, RTy]
  | succ n ih =>
    intro Γ ρ e t hE ht
    cases ht with
    | var h =>
      obtain ⟨v, hv, hvt⟩ := env_lookup _ hE h
      simp only [eval]; rw [hv]; exact hvt
    | lit hv => simpa [eval] using hv
    | bin ho ha hb =>
      have iha := ih hE ha; have ihb := ih hE hb
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      cases hy : eval P n ρ _ <;> rw [hy] at ihb <;> try simp [RTy] at ihb ⊢
      exact binop_sound ho iha ihb
    | ite hc ha hb =>
      have ihc := ih hE hc
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at ihc <;> try simp [RTy] at ihc ⊢
      cases ihc with
      | bool => rename_i b; cases b
                · exact ih hE hb
                · exact ih hE ha
    | let_ ha hb =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      exact ih (.cons iha hE) hb
    | pair ha hb =>
      have iha := ih hE ha; have ihb := ih hE hb
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      cases hy : eval P n ρ _ <;> rw [hy] at ihb <;> try simp [RTy] at ihb ⊢
      exact .pair iha ihb
    | fst ha =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      cases iha; assumption
    | snd ha =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      cases iha; assumption
    | inl ha =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      exact .inl iha
    | inr ha =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      exact .inr iha
    | case hs hl hr =>
      have ihs := ih hE hs
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at ihs <;> try simp [RTy] at ihs ⊢
      cases ihs with
      | inl h => exact ih (.cons h hE) hl
      | inr h => exact ih (.cons h hE) hr
    | raise => simp [eval, RTy]
    | catch_ ha hh =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      all_goals first | exact iha | (rename_i m; cases m <;> first | exact True.intro | exact ih hE hh)
    | call hf ha =>
      have iha := ih hE ha
      obtain ⟨b, hb⟩ := sig_lookup P _ hf
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      have hb' := hb
      simp only [List.get?_eq_getElem?] at hb'
      rw [hb']
      exact ih (.cons iha .nil) (.tick (hP _ _ _ _ hb))
    | tick he => simp only [eval]; exact ih hE he
    | try_ ha =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      cases iha <;> first | assumption | trivial
    | cast ha =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      cases iha; exact chk_ty
    | fcast ha =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      cases iha; exact chk_ty
    | tof ha =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      cases iha; exact fres_ty _
    | un ho ha =>
      have iha := ih hE ha
      simp only [eval]
      cases hx : eval P n ρ _ <;> rw [hx] at iha <;> try simp [RTy] at iha ⊢
      exact unop_sound ho iha

/-- **Totality.** A well-typed program on any fuel ends in a value of its
    type, a defined trap, or exhaustion. -/
theorem eval_total (P : Prog) (hP : WtProg P) (n : Nat) {Γ ρ e t}
    (hE : WtEnv Γ ρ) (ht : HasTy (sig P) Γ e t) :
    (∃ v, eval P n ρ e = .ok v ∧ VTy v t) ∨ (∃ m, eval P n ρ e = .err m) ∨
      eval P n ρ e = .oof := by
  have h := eval_sound P hP n hE ht
  cases hr : eval P n ρ e <;> rw [hr] at h <;> simp_all [RTy]

/-- **Determinism.** Evaluation is a function of program, fuel,
    environment and expression; two runs on the same inputs agree. -/
theorem eval_det (P : Prog) (n : Nat) (ρ : List Val) (e : Expr) {r1 r2 : Res}
    (h1 : eval P n ρ e = r1) (h2 : eval P n ρ e = r2) : r1 = r2 := h1 ▸ h2

end RCE.Core
