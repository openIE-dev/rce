import RCE.Core.Sound
/-! S1/S6: the counted interpreter. `eval`'s fuel bounds nesting depth;
    the reference evaluator instead counts ticks across the whole run
    (book 12 Fuel), so a second interpreter threads a tick counter: every
    node costs one tick from the counter `f` and returns what is left.
    `d` bounds depth only so the definition is structural; with `d ≥ f`
    it never binds first (each node costs a tick). -/
namespace RCE.Core

def evalC (P : Prog) : Nat → Nat → List Val → Expr → Res × Nat
  | 0, f, _, _ => (.oof, f)
  | _ + 1, 0, _, _ => (.oof, 0)
  | d + 1, f + 1, ρ, e =>
    match e with
    | .var i => (match ρ.get? i with | some v => .ok v | none => .stuck, f)
    | .lit v => (.ok v, f)
    | .bin op ta a b =>
      match evalC P d f ρ a with
      | (.ok x, f1) => match evalC P d f1 ρ b with
        | (.ok y, f2) => (binop op ta x y, f2)
        | r => r
      | r => r
    | .ite c a b =>
      match evalC P d f ρ c with
      | (.ok (.bool true), f1) => evalC P d f1 ρ a
      | (.ok (.bool false), f1) => evalC P d f1 ρ b
      | (.ok _, f1) => (.stuck, f1)
      | r => r
    | .let_ a b =>
      match evalC P d f ρ a with
      | (.ok x, f1) => evalC P d f1 (x :: ρ) b
      | r => r
    | .pair a b =>
      match evalC P d f ρ a with
      | (.ok x, f1) => match evalC P d f1 ρ b with
        | (.ok y, f2) => (.ok (.pair x y), f2)
        | r => r
      | r => r
    | .fst a => match evalC P d f ρ a with
      | (.ok (.pair x _), f1) => (.ok x, f1)
      | (.ok _, f1) => (.stuck, f1)
      | r => r
    | .snd a => match evalC P d f ρ a with
      | (.ok (.pair _ y), f1) => (.ok y, f1)
      | (.ok _, f1) => (.stuck, f1)
      | r => r
    | .inl a => match evalC P d f ρ a with
      | (.ok x, f1) => (.ok (.inl x), f1)
      | r => r
    | .inr a => match evalC P d f ρ a with
      | (.ok x, f1) => (.ok (.inr x), f1)
      | r => r
    | .case s l r =>
      match evalC P d f ρ s with
      | (.ok (.inl x), f1) => evalC P d f1 (x :: ρ) l
      | (.ok (.inr x), f1) => evalC P d f1 (x :: ρ) r
      | (.ok _, f1) => (.stuck, f1)
      | q => q
    | .raise c => (.err (.user c), f)
    | .catch_ a h =>
      match evalC P d f ρ a with
      | (.err (.user _), f1) => evalC P d f1 ρ h
      | r => r
    | .call g a =>
      match evalC P d f ρ a with
      | (.ok x, f1) => match P.get? g with
        | some (_, _, body) => evalC P d f1 [x] (.tick body)
        | none => (.stuck, f1)
      | r => r
    | .tick e => evalC P d f ρ e
    | .try_ e c => match evalC P d f ρ e with
      | (.ok (.inr v), f1) => (.ok v, f1)
      | (.ok (.inl _), f1) => (.err (.user c), f1)
      | (.ok _, f1) => (.stuck, f1)
      | r => r
    | .cast lo hi e => match evalC P d f ρ e with
      | (.ok (.int x), f1) => (chk lo hi x, f1)
      | (.ok (.f64 b), f1) => (chk lo hi (RCE.F64.trunc b), f1)
      | (.ok _, f1) => (.stuck, f1)
      | r => r
    | .tof e => match evalC P d f ρ e with
      | (.ok (.int x), f1) => (fres (RCE.F64.ofInt x), f1)
      | (.ok _, f1) => (.stuck, f1)
      | r => r
    | .un op ta e => match evalC P d f ρ e with
      | (.ok x, f1) => (unop op ta x, f1)
      | r => r

/-- Run on a tick budget: depth bound equal to the budget. -/
def runC (P : Prog) (fuel : Nat) (ρ : List Val) (e : Expr) : Res × Nat :=
  evalC P fuel fuel ρ e

/-- Ticks used by a run. -/
def ticksUsed (P : Prog) (fuel : Nat) (ρ : List Val) (e : Expr) : Nat :=
  fuel - (runC P fuel ρ e).2

/-- **Type soundness of the counted interpreter.** -/
theorem evalC_sound (P : Prog) (hP : WtProg P) :
    ∀ d f {Γ ρ e t}, WtEnv Γ ρ → HasTy (sig P) Γ e t → RTy (evalC P d f ρ e).1 t := by
  intro d
  induction d with
  | zero => intros; simp [evalC, RTy]
  | succ d ih =>
    intro f Γ ρ e t hE ht
    cases f with
    | zero => simp [evalC, RTy]
    | succ f =>
    cases ht with
    | var h =>
      obtain ⟨v, hv, hvt⟩ := env_lookup _ hE h
      simp only [evalC]; rw [hv]; exact hvt
    | lit hv => simpa [evalC] using hv
    | raise => simp [evalC, RTy]
    | tick he => simp only [evalC]; exact ih _ hE he
    | bin ho ha hb =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      have ib := ih fa hE hb
      rcases eqb : evalC P d fa ρ _ with ⟨rb, fb⟩
      rw [eqb] at ib
      cases rb <;> simp [RTy] at ib ⊢
      exact binop_sound ho ia ib
    | ite hc ha hb =>
      simp only [evalC]
      have ia := ih f hE hc
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      cases ia with
      | bool => rename_i b; cases b
                · exact ih _ hE hb
                · exact ih _ hE ha
    | let_ ha hb =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      exact ih _ (.cons ia hE) hb
    | pair ha hb =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      have ib := ih fa hE hb
      rcases eqb : evalC P d fa ρ _ with ⟨rb, fb⟩
      rw [eqb] at ib
      cases rb <;> simp [RTy] at ib ⊢
      exact .pair ia ib
    | fst ha =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      cases ia; assumption
    | snd ha =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      cases ia; assumption
    | inl ha =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      exact .inl ia
    | inr ha =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      exact .inr ia
    | case hs hl hr =>
      simp only [evalC]
      have ia := ih f hE hs
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      cases ia with
      | inl h => exact ih _ (.cons h hE) hl
      | inr h => exact ih _ (.cons h hE) hr
    | catch_ ha hh =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases hx : evalC P d f ρ _ with ⟨r, f1⟩
      rw [hx] at ia
      cases r <;> simp [RTy] at ia ⊢
      all_goals first | exact ia | (rename_i m; cases m <;> first | exact True.intro | exact ih _ hE hh)
    | call hf ha =>
      obtain ⟨b, hb⟩ := sig_lookup P _ hf
      have hb' := hb
      simp only [List.get?_eq_getElem?] at hb'
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      rw [hb']
      exact ih _ (.cons ia .nil) (.tick (hP _ _ _ _ hb))
    | try_ ha =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      cases ia <;> first | assumption | trivial
    | cast ha =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      cases ia; exact chk_ty
    | fcast ha =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      cases ia; exact chk_ty
    | tof ha =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      cases ia; exact fres_ty _
    | un ho ha =>
      simp only [evalC]
      have ia := ih f hE ha
      rcases eqa : evalC P d f ρ _ with ⟨ra, fa⟩
      rw [eqa] at ia
      cases ra <;> simp [RTy] at ia ⊢
      exact unop_sound ho ia

end RCE.Core

namespace RCE.Core

/-- **Totality of the counted run.** -/
theorem runC_total (P : Prog) (hP : WtProg P) (fuel : Nat) {Γ ρ e t}
    (hE : WtEnv Γ ρ) (ht : HasTy (sig P) Γ e t) :
    (∃ v, (runC P fuel ρ e).1 = .ok v ∧ VTy v t) ∨ (∃ m, (runC P fuel ρ e).1 = .err m) ∨
      (runC P fuel ρ e).1 = .oof := by
  have h := evalC_sound P hP fuel fuel hE ht
  unfold runC
  cases hr : (evalC P fuel fuel ρ e).1 <;> rw [hr] at h <;> simp_all [RTy]

/-- Tick vectors: `f(1)` with `f(x) = x` costs 4 ticks (call node,
    argument, entry tick, body); a budget of 3 is exhausted. -/
example : runC [(.int 0 255, .int 0 255, .var 0)] 4 [] (.call 0 (.lit (.int 1))) = (.ok (.int 1), 0) := by decide
example : (runC [(.int 0 255, .int 0 255, .var 0)] 3 [] (.call 0 (.lit (.int 1)))).1 = .oof := by decide
example : ticksUsed [] 100 [] (.bin .add (.int 0 255) (.lit (.int 1)) (.lit (.int 2))) = 3 := by decide

end RCE.Core
