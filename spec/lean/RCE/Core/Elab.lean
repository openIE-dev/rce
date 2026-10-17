import RCE.Core.Shift
/-! S2: elaboration of structs, enums and `match` into the core. A struct
    is the right-nested product ending in `unit`; an enum is an exact
    right-nested sum (`enumTy`); variant `i` is `inj i`; a `match` is a
    chain of `case`, later arms shifted past each remainder variable
    (`RCE.Core.Shift.weaken`). Proved: each elaboration is well-typed when
    its parts are, and exhaustiveness both statically (one arm per
    variant) and dynamically (every enum value is one variant's
    injection). -/
namespace RCE.Core

def structTy : List Ty → Ty
  | [] => .unit
  | t :: ts => .prod t (structTy ts)

/-- Enums are exact: one variant is its payload type, more are a
    right-nested sum with no padding, so every value of the enum is the
    injection of exactly one variant (`enum_val_inv`). An enum with no
    variants is refused by the checker and elaborates to `unit`. -/
def enumTy : List Ty → Ty
  | [] => .unit
  | [t] => t
  | t :: t' :: ts => .sum t (enumTy (t' :: ts))

def mkStruct : List Expr → Expr
  | [] => .lit .unit
  | e :: es => .pair e (mkStruct es)

def proj : Nat → Expr → Expr
  | 0, e => .fst e
  | i+1, e => proj i (.snd e)

def inj : Nat → List Ty → Expr → Expr
  | _, [], e => e
  | _, [_], e => e
  | 0, _ :: _ :: _, e => .inl e
  | i+1, _ :: t' :: ts, e => .inr (inj i (t' :: ts) e)

def injV : Nat → List Ty → Val → Val
  | _, [], v => v
  | _, [_], v => v
  | 0, _ :: _ :: _, v => .inl v
  | i+1, _ :: t' :: ts, v => .inr (injV i (t' :: ts) v)

/-- `match s { arm₀, arm₁, ... }` with each arm binding its payload as
    variable 0 in the arm's own context. Later arms are shifted past the
    remainder variable each `case` introduces. -/
def matchS : Expr → List Expr → Expr
  | _, [] => .raise 0
  | s, [a] => .let_ s a
  | s, a :: b :: as => .case s a (matchS (.var 0) ((b :: as).map (shift 1)))
termination_by _ arms => arms.length

inductive AllTy (S : Sig) (Γ : List Ty) : List Expr → List Ty → Prop where
  | nil : AllTy S Γ [] []
  | cons {e es t ts} : HasTy S Γ e t → AllTy S Γ es ts → AllTy S Γ (e :: es) (t :: ts)

/-- Arm `i` is typed with its payload `tᵢ` bound in front of Γ. -/
inductive ArmsTy (S : Sig) (Γ : List Ty) (t : Ty) : List Ty → List Expr → Prop where
  | nil : ArmsTy S Γ t [] []
  | cons {t0 ts a as} : HasTy S (t0 :: Γ) a t → ArmsTy S Γ t ts as → ArmsTy S Γ t (t0 :: ts) (a :: as)

theorem mkStruct_ty {S Γ} : ∀ {es ts}, AllTy S Γ es ts → HasTy S Γ (mkStruct es) (structTy ts)
  | _, _, .nil => .lit .unit
  | _, _, .cons h hs => .pair h (mkStruct_ty hs)

theorem proj_ty {S Γ} : ∀ (i : Nat) {e ts t}, ts.get? i = some t →
    HasTy S Γ e (structTy ts) → HasTy S Γ (proj i e) t
  | _, _, [], _, h, _ => by simp at h
  | 0, _, _ :: _, _, h, he => by simp at h; subst h; exact .fst he
  | i+1, _, _ :: ts, _, h, he => proj_ty i (by simpa using h) (.snd he)

theorem inj_ty {S Γ} : ∀ (i : Nat) (ts : List Ty) {e t}, ts.get? i = some t →
    HasTy S Γ e t → HasTy S Γ (inj i ts e) (enumTy ts)
  | _, [], _, _, h, _ => by simp at h
  | 0, [_], _, _, h, he => by simp at h; subst h; exact he
  | _+1, [_], _, _, h, _ => by simp at h
  | 0, _ :: _ :: _, _, _, h, he => by simp at h; subst h; exact .inl he
  | i+1, _ :: t' :: ts, _, _, h, he => .inr (inj_ty i (t' :: ts) (by simpa using h) he)

theorem injV_ty : ∀ (i : Nat) (ts : List Ty) {v t}, ts.get? i = some t →
    VTy v t → VTy (injV i ts v) (enumTy ts)
  | _, [], _, _, h, _ => by simp at h
  | 0, [_], _, _, h, hv => by simp at h; subst h; exact hv
  | _+1, [_], _, _, h, _ => by simp at h
  | 0, _ :: _ :: _, _, _, h, hv => by simp at h; subst h; exact .inl hv
  | i+1, _ :: t' :: ts, _, _, h, hv => .inr (injV_ty i (t' :: ts) (by simpa using h) hv)

theorem arms_shift {S Γ t u} : ∀ {ts as}, ArmsTy S Γ t ts as →
    ArmsTy S (u :: Γ) t ts (as.map (shift 1))
  | _, _, .nil => .nil
  | _, _, .cons ha hs => .cons (weaken ha 1 u) (arms_shift hs)

/-- **[Match] typing**: an exhaustive arm list (one arm per variant, in
    order) over a scrutinee of the enum type gives a well-typed match. -/
theorem matchS_ty {S} : ∀ (ts : List Ty) {Γ t s arms}, ts ≠ [] → ArmsTy S Γ t ts arms →
    HasTy S Γ s (enumTy ts) → HasTy S Γ (matchS s arms) t
  | [], _, _, _, _, hn, _, _ => absurd rfl hn
  | [_], _, _, _, _, _, .cons ha .nil, h => by simp only [matchS]; exact .let_ h ha
  | t0 :: t1 :: ts, Γ, t, s, _, _, .cons ha (.cons hb hs), h => by
    simp only [matchS]
    exact .case h ha (matchS_ty (t1 :: ts) (by simp) (arms_shift (.cons hb hs)) (.var rfl))

/-- **Exhaustiveness (static)**: an arm list that types has exactly one
    arm per variant. -/
theorem arms_exhaustive {S Γ t} : ∀ {ts as}, ArmsTy S Γ t ts as → as.length = ts.length
  | _, _, .nil => rfl
  | _, _, .cons _ hs => by simp [arms_exhaustive hs]

/-- **Exhaustiveness (dynamic)**: every value of a nonempty enum is the
    injection of exactly one variant's payload, so the `case` chain always
    reaches an arm and never the fallback. -/
theorem enum_val_inv : ∀ (ts : List Ty) {v}, ts ≠ [] → VTy v (enumTy ts) →
    ∃ i t u, ts.get? i = some t ∧ VTy u t ∧ v = injV i ts u
  | [], _, hn, _ => absurd rfl hn
  | [t], v, _, h => ⟨0, t, v, rfl, h, rfl⟩
  | t0 :: t1 :: ts, _, _, h => by
    cases h with
    | inl h => exact ⟨0, t0, _, rfl, h, rfl⟩
    | inr h =>
      obtain ⟨i, t, u, hi, hu, rfl⟩ := enum_val_inv (t1 :: ts) (by simp) h
      exact ⟨i + 1, t, u, by simpa using hi, hu, rfl⟩

end RCE.Core
