import RCE.Core.Lang
/-! S2: ownership in the core, book 5.3 [Var-Copy]/[Var-Move] and 5.5 O1.
    A context entry is a type and a liveness flag. `HasTyO Γ e t Γ'` threads
    the context left to right: reading a live Copy variable leaves it live,
    reading a live Move variable marks it moved. Branches must agree on
    the output context (as `Linear.lin` requires both branches to consume
    alike). Proved: erasing liveness gives the core typing `HasTy`, so
    `eval_sound` covers every ownership-typed program (`erase`); and a
    moved variable is never read again (`moved_unread`, O1). -/
namespace RCE.Core

/-- Move types (book 5.5): text and anything containing it. -/
def isMove : Ty → Bool
  | .text => true
  | .prod a b => isMove a || isMove b
  | .sum a b => isMove a || isMove b
  | _ => false

abbrev OCtx := List (Ty × Bool)

inductive HasTyO (S : Sig) : OCtx → Expr → Ty → OCtx → Prop where
  | varC {Γ i t} : Γ[i]? = some (t, true) → isMove t = false → HasTyO S Γ (.var i) t Γ
  | varM {Γ i t} : Γ[i]? = some (t, true) → isMove t = true →
      HasTyO S Γ (.var i) t (Γ.set i (t, false))
  | lit {Γ v t} : VTy v t → HasTyO S Γ (.lit v) t Γ
  | bin {Γ Γ1 Γ2 op ta a b t} : OpTy op ta t → HasTyO S Γ a ta Γ1 → HasTyO S Γ1 b ta Γ2 →
      HasTyO S Γ (.bin op ta a b) t Γ2
  | ite {Γ Γ1 Γ2 c a b t} : HasTyO S Γ c .bool Γ1 → HasTyO S Γ1 a t Γ2 → HasTyO S Γ1 b t Γ2 →
      HasTyO S Γ (.ite c a b) t Γ2
  | let_ {Γ Γ1 Γ2 a b ta t x} : HasTyO S Γ a ta Γ1 → HasTyO S ((ta, true) :: Γ1) b t (x :: Γ2) →
      HasTyO S Γ (.let_ a b) t Γ2
  | pair {Γ Γ1 Γ2 a b ta tb} : HasTyO S Γ a ta Γ1 → HasTyO S Γ1 b tb Γ2 →
      HasTyO S Γ (.pair a b) (.prod ta tb) Γ2
  | fst {Γ Γ1 e ta tb} : HasTyO S Γ e (.prod ta tb) Γ1 → HasTyO S Γ (.fst e) ta Γ1
  | snd {Γ Γ1 e ta tb} : HasTyO S Γ e (.prod ta tb) Γ1 → HasTyO S Γ (.snd e) tb Γ1
  | inl {Γ Γ1 e ta tb} : HasTyO S Γ e ta Γ1 → HasTyO S Γ (.inl e) (.sum ta tb) Γ1
  | inr {Γ Γ1 e ta tb} : HasTyO S Γ e tb Γ1 → HasTyO S Γ (.inr e) (.sum ta tb) Γ1
  | case {Γ Γ1 Γ2 s l r ta tb t x y} : HasTyO S Γ s (.sum ta tb) Γ1 →
      HasTyO S ((ta, true) :: Γ1) l t (x :: Γ2) → HasTyO S ((tb, true) :: Γ1) r t (y :: Γ2) →
      HasTyO S Γ (.case s l r) t Γ2
  | raise {Γ c t} : HasTyO S Γ (.raise c) t Γ
  | catch_ {Γ Γ1 a h t} : HasTyO S Γ a t Γ1 → HasTyO S Γ1 h t Γ1 → HasTyO S Γ (.catch_ a h) t Γ1
  | call {Γ Γ1 f a ta t} : S.get? f = some (ta, t) → HasTyO S Γ a ta Γ1 →
      HasTyO S Γ (.call f a) t Γ1
  | tick {Γ Γ1 e t} : HasTyO S Γ e t Γ1 → HasTyO S Γ (.tick e) t Γ1
  | try_ {Γ Γ1 e ta t c} : HasTyO S Γ e (.sum ta t) Γ1 → HasTyO S Γ (.try_ e c) t Γ1
  | cast {Γ Γ1 e lo hi lo' hi'} : HasTyO S Γ e (.int lo' hi') Γ1 →
      HasTyO S Γ (.cast lo hi e) (.int lo hi) Γ1
  | fcast {Γ Γ1 e lo hi} : HasTyO S Γ e .f64 Γ1 → HasTyO S Γ (.cast lo hi e) (.int lo hi) Γ1
  | tof {Γ Γ1 e lo hi} : HasTyO S Γ e (.int lo hi) Γ1 → HasTyO S Γ (.tof e) .f64 Γ1
  | un {Γ Γ1 op ta t e} : UnTy op ta t → HasTyO S Γ e ta Γ1 → HasTyO S Γ (.un op ta e) t Γ1

/-- Ownership only changes liveness, never types. -/
theorem types_kept {S Γ e t Γ'} (h : HasTyO S Γ e t Γ') : Γ'.map Prod.fst = Γ.map Prod.fst := by
  induction h with
  | @varM Γ i t hg _ =>
    have hl : i < Γ.length := by
      rcases Nat.lt_or_ge i Γ.length with h | h
      · exact h
      · rw [List.getElem?_eq_none h] at hg; cases hg
    apply List.ext_getElem?
    intro j
    simp only [List.getElem?_map, List.getElem?_set]
    by_cases hj : i = j
    · subst hj; simp [hg, hl]
    · simp [hj]
  | let_ _ _ iha ihb => simp only [List.map] at ihb; rw [(List.cons.inj ihb).2, iha]
  | case _ _ _ ihs ihl _ => simp only [List.map] at ihl; rw [(List.cons.inj ihl).2, ihs]
  | bin _ _ _ iha ihb => rw [ihb, iha]
  | ite _ _ _ ihc iha _ => rw [iha, ihc]
  | pair _ _ iha ihb => rw [ihb, iha]
  | catch_ _ _ iha _ => exact iha
  | _ => simp_all

/-- Erasing liveness gives the core typing. -/
theorem erase {S Γ e t Γ'} (h : HasTyO S Γ e t Γ') : HasTy S (Γ.map Prod.fst) e t := by
  induction h with
  | varC hg _ => exact .var (by simp [List.get?_eq_getElem?, List.getElem?_map, hg])
  | varM hg _ => exact .var (by simp [List.get?_eq_getElem?, List.getElem?_map, hg])
  | lit hv => exact .lit hv
  | bin ho ha hb iha ihb => exact .bin ho iha (by rw [← types_kept ha]; exact ihb)
  | ite hc ha hb ihc iha ihb =>
    exact .ite ihc (by rw [← types_kept hc]; exact iha) (by rw [← types_kept hc]; exact ihb)
  | let_ ha hb iha ihb => exact .let_ iha (by simpa [types_kept ha] using ihb)
  | pair ha hb iha ihb => exact .pair iha (by rw [← types_kept ha]; exact ihb)
  | fst _ ih => exact .fst ih
  | snd _ ih => exact .snd ih
  | inl _ ih => exact .inl ih
  | inr _ ih => exact .inr ih
  | case hs hl hr ihs ihl ihr =>
    exact .case ihs (by simpa [types_kept hs] using ihl) (by simpa [types_kept hs] using ihr)
  | raise => exact .raise
  | catch_ ha hh iha ihh => exact .catch_ iha (by rw [← types_kept ha]; exact ihh)
  | call hf _ ih => exact .call hf ih
  | tick _ ih => exact .tick ih
  | try_ _ ih => exact .try_ ih
  | cast _ ih => exact .cast ih
  | fcast _ ih => exact .fcast ih
  | tof _ ih => exact .tof ih
  | un ho _ ih => exact .un ho ih

/-- Free occurrence of de Bruijn index `i`. -/
def occurs (i : Nat) : Expr → Bool
  | .var j => j == i
  | .lit _ => false
  | .bin _ _ a b => occurs i a || occurs i b
  | .ite c a b => occurs i c || occurs i a || occurs i b
  | .let_ a b => occurs i a || occurs (i + 1) b
  | .pair a b => occurs i a || occurs i b
  | .fst e | .snd e | .inl e | .inr e | .tick e => occurs i e
  | .case s l r => occurs i s || occurs (i + 1) l || occurs (i + 1) r
  | .raise _ => false
  | .catch_ a h => occurs i a || occurs i h
  | .call _ a => occurs i a
  | .try_ e _ => occurs i e
  | .cast _ _ e => occurs i e
  | .tof e => occurs i e
  | .un _ _ e => occurs i e

/-- A moved variable stays moved. -/
theorem dead_stays {S Γ e t Γ'} (h : HasTyO S Γ e t Γ') :
    ∀ (i : Nat) (τ : Ty), Γ[i]? = some (τ, false) → Γ'[i]? = some (τ, false) := by
  induction h with
  | @varM Γ i t hg _ =>
    intro j τ hj
    simp only [List.getElem?_set]
    by_cases hij : i = j
    · subst hij; rw [hg] at hj; cases hj
    · simp [hij, hj]
  | let_ _ _ iha ihb =>
    intro j τ hj
    have := ihb (j + 1) τ (by simpa using iha j τ hj)
    simpa using this
  | case _ _ _ ihs ihl _ =>
    intro j τ hj
    have := ihl (j + 1) τ (by simpa using ihs j τ hj)
    simpa using this
  | bin _ _ _ iha ihb => intro j τ hj; exact ihb j τ (iha j τ hj)
  | ite _ _ _ ihc iha _ => intro j τ hj; exact iha j τ (ihc j τ hj)
  | pair _ _ iha ihb => intro j τ hj; exact ihb j τ (iha j τ hj)
  | catch_ _ _ iha _ => intro j τ hj; exact iha j τ hj
  | _ => first | assumption | simp_all

/-- **O1.** A moved variable is not read again. -/
theorem moved_unread {S Γ e t Γ'} (h : HasTyO S Γ e t Γ') :
    ∀ (i : Nat) (τ : Ty), Γ[i]? = some (τ, false) → occurs i e = false := by
  induction h with
  | @varC Γ i t hg _ =>
    intro j τ hj; simp only [occurs, beq_eq_false_iff_ne]
    intro e; subst e; rw [hg] at hj; cases hj
  | @varM Γ i t hg _ =>
    intro j τ hj; simp only [occurs, beq_eq_false_iff_ne]
    intro e; subst e; rw [hg] at hj; cases hj
  | bin _ ha _ iha ihb =>
    intro j τ hj; simp [occurs, iha j τ hj, ihb j τ (dead_stays ha j τ hj)]
  | ite hc ha _ ihc iha ihb =>
    intro j τ hj
    have h1 := dead_stays hc j τ hj
    simp [occurs, ihc j τ hj, iha j τ h1, ihb j τ h1]
  | let_ ha _ iha ihb =>
    intro j τ hj
    simp [occurs, iha j τ hj, ihb (j + 1) τ (by simpa using dead_stays ha j τ hj)]
  | pair ha _ iha ihb =>
    intro j τ hj; simp [occurs, iha j τ hj, ihb j τ (dead_stays ha j τ hj)]
  | case hs _ _ ihs ihl ihr =>
    intro j τ hj
    simp [occurs, ihs j τ hj, ihl (j + 1) τ (by simpa using dead_stays hs j τ hj),
      ihr (j + 1) τ (by simpa using dead_stays hs j τ hj)]
  | catch_ ha _ iha ihh =>
    intro j τ hj; simp [occurs, iha j τ hj, ihh j τ (dead_stays ha j τ hj)]
  | _ => simp only [occurs]; first | assumption | simp_all

end RCE.Core
