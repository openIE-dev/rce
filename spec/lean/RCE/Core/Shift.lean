import RCE.Core.Lang
/-! S2: de Bruijn shifting and the weakening lemma. `shift c e` makes room
    for one new variable at position `c`; `ins c u Γ` is the context with
    `u` inserted there. Weakening: a typing in Γ is a typing of the
    shifted term in the larger context. -/
namespace RCE.Core

def shift (c : Nat) : Expr → Expr
  | .var i => if i < c then .var i else .var (i + 1)
  | .lit v => .lit v
  | .bin op ta a b => .bin op ta (shift c a) (shift c b)
  | .ite x a b => .ite (shift c x) (shift c a) (shift c b)
  | .let_ a b => .let_ (shift c a) (shift (c + 1) b)
  | .pair a b => .pair (shift c a) (shift c b)
  | .fst e => .fst (shift c e)
  | .snd e => .snd (shift c e)
  | .inl e => .inl (shift c e)
  | .inr e => .inr (shift c e)
  | .case s l r => .case (shift c s) (shift (c + 1) l) (shift (c + 1) r)
  | .raise k => .raise k
  | .catch_ a h => .catch_ (shift c a) (shift c h)
  | .call f a => .call f (shift c a)
  | .tick e => .tick (shift c e)
  | .try_ e k => .try_ (shift c e) k
  | .cast lo hi e => .cast lo hi (shift c e)
  | .tof e => .tof (shift c e)
  | .un op ta e => .un op ta (shift c e)

def ins : Nat → Ty → List Ty → List Ty
  | 0, u, Γ => u :: Γ
  | _ + 1, u, [] => [u]
  | c + 1, u, t :: Γ => t :: ins c u Γ

theorem ins_get : ∀ (c : Nat) (u : Ty) (Γ : List Ty) (i : Nat) {t},
    Γ.get? i = some t → (ins c u Γ).get? (if i < c then i else i + 1) = some t
  | 0, _, _, _, _, h => by simpa [ins] using h
  | _ + 1, _, [], _, _, h => by simp at h
  | _ + 1, _, _ :: _, 0, _, h => by simpa [ins] using h
  | c + 1, u, _ :: Γ, j + 1, _, h => by
    have ih := ins_get c u Γ j (by simpa using h)
    by_cases hj : j < c
    · have : j + 1 < c + 1 := by omega
      simp only [hj, if_true] at ih; simpa [ins, this] using ih
    · have : ¬ j + 1 < c + 1 := by omega
      simp only [hj, if_false] at ih; simpa [ins, this] using ih

/-- **Weakening.** -/
theorem weaken {S} : ∀ {Γ e t}, HasTy S Γ e t → ∀ (c : Nat) (u : Ty),
    HasTy S (ins c u Γ) (shift c e) t := by
  intro Γ e t h
  induction h with
  | var h => intro c u; have := ins_get c u _ _ h; simp only [shift]; split <;> simp_all <;> exact .var (by simpa using this)
  | lit hv => intro _ _; exact .lit hv
  | bin ho _ _ iha ihb => intro c u; exact .bin ho (iha c u) (ihb c u)
  | ite _ _ _ ihc iha ihb => intro c u; exact .ite (ihc c u) (iha c u) (ihb c u)
  | let_ _ _ iha ihb => intro c u; exact .let_ (iha c u) (ihb (c + 1) u)
  | pair _ _ iha ihb => intro c u; exact .pair (iha c u) (ihb c u)
  | fst _ ih => intro c u; exact .fst (ih c u)
  | snd _ ih => intro c u; exact .snd (ih c u)
  | inl _ ih => intro c u; exact .inl (ih c u)
  | inr _ ih => intro c u; exact .inr (ih c u)
  | case _ _ _ ihs ihl ihr => intro c u; exact .case (ihs c u) (ihl (c + 1) u) (ihr (c + 1) u)
  | raise => intro _ _; exact .raise
  | catch_ _ _ iha ihh => intro c u; exact .catch_ (iha c u) (ihh c u)
  | call hf _ ih => intro c u; exact .call hf (ih c u)
  | tick _ ih => intro c u; exact .tick (ih c u)
  | try_ _ ih => intro c u; exact .try_ (ih c u)
  | cast _ ih => intro c u; exact .cast (ih c u)
  | fcast _ ih => intro c u; exact .fcast (ih c u)
  | tof _ ih => intro c u; exact .tof (ih c u)
  | un ho _ ih => intro c u; exact .un ho (ih c u)

end RCE.Core
