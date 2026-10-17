import RCE.Core.Driver
/-! S6 `check`: a decidable type checker for the core, proved sound against
    the static semantics `HasTy` (Lang.lean). `inf S Γ e h` synthesises a
    type for `e`, using the hint `h` where the syntax does not determine one
    (integer literals, `raise`, `inl`/`inr`). It is incomplete by design: a
    `none` means "not shown well typed", never "ill typed", so the conformance
    adapter answers `unsupported` on it, and claims `accept` only on `some`. -/
namespace RCE.Core

/-- Does a value have a type? (decides `VTy`) -/
def vty : Val → Ty → Bool
  | .int n, .int lo hi => decide (lo ≤ n) && decide (n ≤ hi)
  | .bool _, .bool => true
  | .text _, .text => true
  | .unit, .unit => true
  | .pair a b, .prod ta tb => vty a ta && vty b tb
  | .inl v, .sum ta _ => vty v ta
  | .inr v, .sum _ tb => vty v tb
  | .f64 _, .f64 => true
  | .f32 _, .f32 => true
  | _, _ => false

theorem vty_sound : ∀ v t, vty v t = true → VTy v t
  | .int n, .int lo hi, h => by
    simp [vty] at h; exact .int h.1 h.2
  | .bool _, .bool, _ => .bool
  | .text _, .text, _ => .text
  | .unit, .unit, _ => .unit
  | .pair a b, .prod ta tb, h => by
    simp [vty] at h; exact .pair (vty_sound a ta h.1) (vty_sound b tb h.2)
  | .inl v, .sum ta _, h => .inl (vty_sound v ta (by simpa [vty] using h))
  | .inr v, .sum _ tb, h => .inr (vty_sound v tb (by simpa [vty] using h))
  | .f64 _, .f64, _ => .f64
  | .f32 _, .f32, _ => .f32
  | .int _, .bool, h | .int _, .text, h | .int _, .unit, h | .int _, .prod _ _, h
  | .int _, .sum _ _, h | .int _, .f64, h | .int _, .f32, h => by simp [vty] at h
  | .bool _, .int _ _, h | .bool _, .text, h | .bool _, .unit, h | .bool _, .prod _ _, h
  | .bool _, .sum _ _, h | .bool _, .f64, h | .bool _, .f32, h => by simp [vty] at h
  | .text _, .int _ _, h | .text _, .bool, h | .text _, .unit, h | .text _, .prod _ _, h
  | .text _, .sum _ _, h | .text _, .f64, h | .text _, .f32, h => by simp [vty] at h
  | .unit, .int _ _, h | .unit, .bool, h | .unit, .text, h | .unit, .prod _ _, h
  | .unit, .sum _ _, h | .unit, .f64, h | .unit, .f32, h => by simp [vty] at h
  | .pair _ _, .int _ _, h | .pair _ _, .bool, h | .pair _ _, .text, h | .pair _ _, .unit, h
  | .pair _ _, .sum _ _, h | .pair _ _, .f64, h | .pair _ _, .f32, h => by simp [vty] at h
  | .inl _, .int _ _, h | .inl _, .bool, h | .inl _, .text, h | .inl _, .unit, h
  | .inl _, .prod _ _, h | .inl _, .f64, h | .inl _, .f32, h => by simp [vty] at h
  | .inr _, .int _ _, h | .inr _, .bool, h | .inr _, .text, h | .inr _, .unit, h
  | .inr _, .prod _ _, h | .inr _, .f64, h | .inr _, .f32, h => by simp [vty] at h
  | .f64 _, .int _ _, h | .f64 _, .bool, h | .f64 _, .text, h | .f64 _, .unit, h
  | .f64 _, .prod _ _, h | .f64 _, .sum _ _, h | .f64 _, .f32, h => by simp [vty] at h
  | .f32 _, .int _ _, h | .f32 _, .bool, h | .f32 _, .text, h | .f32 _, .unit, h
  | .f32 _, .prod _ _, h | .f32 _, .sum _ _, h | .f32 _, .f64, h => by simp [vty] at h

/-- The type a literal has on its own (no hint). Integers and sums need one. -/
def litTy : Val → Option Ty
  | .bool _ => some .bool
  | .text _ => some .text
  | .unit => some .unit
  | .f64 _ => some .f64
  | .f32 _ => some .f32
  | _ => none

theorem litTy_sound : ∀ v t, litTy v = some t → VTy v t := by
  intro v t h; cases v <;> simp [litTy] at h <;> subst h <;> constructor

/-- Result type of a binary operator at operand type `ta` (decides `OpTy`). -/
def opRes (op : BinOp) (ta : Ty) : Option Ty :=
  match op with
  | .eq => some .bool
  | .lt => match ta with
    | .int _ _ => some .bool | .f64 => some .bool | .f32 => some .bool | _ => none
  | .and | .or => match ta with | .bool => some .bool | _ => none
  | .cat => match ta with | .text => some .text | _ => none
  | .add | .sub | .mul | .div => match ta with
    | .int lo hi => some (.int lo hi) | .f64 => some .f64 | .f32 => some .f32 | _ => none
  | .min | .max => match ta with
    | .int lo hi => some (.int lo hi) | .f64 => some .f64 | _ => none
  | .rem | .band | .bor | .bxor | .shl | .shr | .sadd | .ssub | .smul
  | .wadd | .wsub | .wmul => match ta with | .int lo hi => some (.int lo hi) | _ => none

theorem opRes_sound : ∀ op ta t, opRes op ta = some t → OpTy op ta t := by
  intro op ta t h
  cases op <;> cases ta <;> (first | (cases h; constructor) | cases h)

/-- Result type of a unary operator at operand type `ta` (decides `UnTy`). -/
def unRes (op : UnOp) (ta : Ty) : Option Ty :=
  match op with
  | .iabs => match ta with | .int lo hi => some (.int lo hi) | _ => none
  | .fabs | .fneg => match ta with | .f64 => some .f64 | .f32 => some .f32 | _ => none
  | .sqrt | .floor | .ceil | .round => match ta with | .f64 => some .f64 | _ => none
  | .ineg lo hi => match ta with | .int _ _ => some (.int lo hi) | _ => none
  | .tof32 => match ta with | .int _ _ => some .f32 | .f64 => some .f32 | _ => none
  | .widen => match ta with | .f32 => some .f64 | _ => none
  | .trunc32 lo hi => match ta with | .f32 => some (.int lo hi) | _ => none

theorem unRes_sound : ∀ op ta t, unRes op ta = some t → UnTy op ta t := by
  intro op ta t h
  cases op <;> cases ta <;> (first | (cases h; constructor) | cases h)

def hintL : Option Ty → Option Ty
  | some (.prod a _) => some a
  | some (.sum a _) => some a
  | _ => none
def hintR : Option Ty → Option Ty
  | some (.prod _ b) => some b
  | some (.sum _ b) => some b
  | _ => none

/-- Type synthesis with a hint. -/
def inf (S : Sig) : List Ty → Expr → Option Ty → Option Ty
  | Γ, .var i, _ => Γ.get? i
  | _, .lit v, h => match h with
    | some t => if vty v t then some t else litTy v
    | none => litTy v
  | Γ, .bin op ta a b, _ =>
    if inf S Γ a (some ta) = some ta ∧ inf S Γ b (some ta) = some ta then opRes op ta else none
  | Γ, .ite c a b, h =>
    if inf S Γ c (some .bool) = some .bool then
      match inf S Γ a h with
      | some t => if inf S Γ b (some t) = some t then some t else none
      | none => none
    else none
  | Γ, .let_ a b, h => match inf S Γ a none with
    | some ta => inf S (ta :: Γ) b h
    | none => none
  | Γ, .pair a b, h => match inf S Γ a (hintL h), inf S Γ b (hintR h) with
    | some ta, some tb => some (.prod ta tb)
    | _, _ => none
  | Γ, .fst e, _ => match inf S Γ e none with
    | some (.prod ta _) => some ta
    | _ => none
  | Γ, .snd e, _ => match inf S Γ e none with
    | some (.prod _ tb) => some tb
    | _ => none
  | Γ, .inl e, h => match h with
    | some (.sum ta tb) => if inf S Γ e (some ta) = some ta then some (.sum ta tb) else none
    | _ => none
  | Γ, .inr e, h => match h with
    | some (.sum ta tb) => if inf S Γ e (some tb) = some tb then some (.sum ta tb) else none
    | _ => none
  | Γ, .case s l r, h => match inf S Γ s none with
    | some (.sum ta tb) => match inf S (ta :: Γ) l h with
      | some t => if inf S (tb :: Γ) r (some t) = some t then some t else none
      | none => none
    | _ => none
  | _, .raise _, h => h
  | Γ, .catch_ a k, h => match inf S Γ a h with
    | some t => if inf S Γ k (some t) = some t then some t else none
    | none => none
  | Γ, .call f a, _ => match S.get? f with
    | some (ta, t) => if inf S Γ a (some ta) = some ta then some t else none
    | none => none
  | Γ, .tick e, h => inf S Γ e h
  | Γ, .try_ e _, _ => match inf S Γ e none with
    | some (.sum _ t) => some t
    | _ => none
  | Γ, .cast lo hi e, _ => match inf S Γ e none with
    | some (.int _ _) => some (.int lo hi)
    | some .f64 => some (.int lo hi)
    | _ => none
  | Γ, .tof e, _ => match inf S Γ e none with
    | some (.int _ _) => some .f64
    | _ => none
  | Γ, .un op ta e, _ => if inf S Γ e (some ta) = some ta then unRes op ta else none

theorem inf_sound (S : Sig) : ∀ e Γ h t, inf S Γ e h = some t → HasTy S Γ e t := by
  intro e
  induction e with
  | var i => intro Γ h t H; exact .var H
  | lit v =>
    intro Γ h t H
    cases h with
    | none => exact .lit (litTy_sound v t H)
    | some u =>
      simp only [inf] at H
      split at H
      · cases H; exact .lit (vty_sound _ _ (by assumption))
      · exact .lit (litTy_sound v t H)
  | bin op ta a b iha ihb =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i hc; exact .bin (opRes_sound _ _ _ H) (iha _ _ _ hc.1) (ihb _ _ _ hc.2)
    · cases H
  | ite c a b ihc iha ihb =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i hc; split at H
      · rename_i u hu; split at H
        · rename_i hb; cases H; exact .ite (ihc _ _ _ hc) (iha _ _ _ hu) (ihb _ _ _ hb)
        · cases H
      · cases H
    · cases H
  | let_ a b iha ihb =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i ta hta; exact .let_ (iha _ _ _ hta) (ihb _ _ _ H)
    · cases H
  | pair a b iha ihb =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i ta tb hta htb; cases H; exact .pair (iha _ _ _ hta) (ihb _ _ _ htb)
    · cases H
  | fst e ih =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i ta tb he; cases H; exact .fst (ih _ _ _ he)
    · cases H
  | snd e ih =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i ta tb he; cases H; exact .snd (ih _ _ _ he)
    · cases H
  | inl e ih =>
    intro Γ h t H; simp only [inf] at H; split at H
    · split at H
      · rename_i he; cases H; exact .inl (ih _ _ _ he)
      · cases H
    · cases H
  | inr e ih =>
    intro Γ h t H; simp only [inf] at H; split at H
    · split at H
      · rename_i he; cases H; exact .inr (ih _ _ _ he)
      · cases H
    · cases H
  | case s l r ihs ihl ihr =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i ta tb hs; split at H
      · rename_i u hl; split at H
        · rename_i hr; cases H; exact .case (ihs _ _ _ hs) (ihl _ _ _ hl) (ihr _ _ _ hr)
        · cases H
      · cases H
    · cases H
  | raise c => intro Γ h t _; exact .raise
  | catch_ a k iha ihk =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i u ha; split at H
      · rename_i hk; cases H; exact .catch_ (iha _ _ _ ha) (ihk _ _ _ hk)
      · cases H
    · cases H
  | call f a iha =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i ta u hf; split at H
      · rename_i ha; cases H; exact .call hf (iha _ _ _ ha)
      · cases H
    · cases H
  | tick e ih => intro Γ h t H; exact .tick (ih _ _ _ H)
  | try_ e c ih =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i ta u he; cases H; exact .try_ (ih _ _ _ he)
    · cases H
  | cast lo hi e ih =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i a b he; cases H; exact .cast (ih _ _ _ he)
    · rename_i he; cases H; exact .fcast (ih _ _ _ he)
    · cases H
  | tof e ih =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i a b he; cases H; exact .tof (ih _ _ _ he)
    · cases H
  | un op ta e ih =>
    intro Γ h t H; simp only [inf] at H; split at H
    · rename_i he; exact .un (unRes_sound _ _ _ H) (ih _ _ _ he)
    · cases H

/-- Check a program's functions: each body has its declared result type. -/
def checkProg (P : Prog) : Bool :=
  P.all fun (ta, t, b) => inf (sig P) [ta] b (some t) == some t

theorem checkProg_sound (P : Prog) : checkProg P = true → WtProg P := by
  intro H f ta t b hf
  have hm := List.get?_mem hf
  have := List.all_eq_true.mp H _ hm
  simp at this
  exact inf_sound _ _ _ _ _ this

end RCE.Core

