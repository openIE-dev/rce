/-! P1 (spec/proposals/CORE-PROPERTIES.md): indexed values, version 0.8
    (spec 5.16). Three models, one per index:
    * context: a `Text[C]` value is a sequence of literal pieces and
      encoded pieces; `encode` emits no metacharacter of C, so every
      metacharacter of the rendered text comes from a literal of the
      program (`ctx_meta_from_literals`): untrusted text cannot form
      markup or query syntax (SQL injection, XSS excluded).
    * secrecy: `Secret[τ]` is opaque; the only operation is equality,
      whose bit is the one release. Two runs whose public inputs agree and
      whose secret-equality bits agree emit the same outputs
      (`noninterference`).
    * authority: a field of type `Guarded[τ, A]` is written only by a step
      of an event that carries `Cap[A]`; any log of events without the
      capability leaves the guarded part unchanged (`guarded_needs_cap`). -/
namespace RCE.Label

/-! ## Context-typed text -/
section Ctx
variable (isMeta : Char → Bool) (enc : Char → List Char)

inductive Piece | lit (s : List Char) | enc (s : List Char)

def render : List Piece → List Char
  | [] => []
  | .lit s :: d => s ++ render d
  | .enc s :: d => s.bind enc ++ render d

def litMeta : List Piece → List Char
  | [] => []
  | .lit s :: d => s.filter isMeta ++ litMeta d
  | .enc _ :: d => litMeta d

theorem enc_clean (h : ∀ c, ∀ x ∈ enc c, isMeta x = false) (s : List Char) :
    (s.bind enc).filter isMeta = [] := by
  induction s with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.bind_cons, List.filter_append, ih, List.append_nil]
    exact List.filter_eq_nil.mpr (fun x hx => by simp [h c x hx])

/-- Every metacharacter of a rendered `Text[C]` value comes from a literal. -/
theorem ctx_meta_from_literals (h : ∀ c, ∀ x ∈ enc c, isMeta x = false) (d : List Piece) :
    (render enc d).filter isMeta = litMeta isMeta d := by
  induction d with
  | nil => rfl
  | cons p d ih =>
    cases p with
    | lit s => simp [render, litMeta, List.filter_append, ih]
    | enc s => simp [render, litMeta, List.filter_append, ih, enc_clean isMeta enc h s]

/-- HTML encoding of 5.16: `& < > " '` become entities. -/
def htmlMeta (c : Char) : Bool := c = '&' || c = '<' || c = '>' || c = '"' || c = '\''
def htmlEnc (c : Char) : List Char :=
  if c = '&' then "&amp;".toList else if c = '<' then "&lt;".toList
  else if c = '>' then "&gt;".toList else if c = '"' then "&quot;".toList
  else if c = '\'' then "&#39;".toList else [c]

/-- The HTML encoder meets the hypothesis, except for the `&` that starts
    each entity, which is the encoder's own escape and is followed by an
    entity name; stated for the structural metacharacters `< > " '`. -/
def htmlStruct (c : Char) : Bool := c = '<' || c = '>' || c = '"' || c = '\''
theorem html_enc_clean : ∀ c, ∀ x ∈ htmlEnc c, htmlStruct x = false := by
  intro c x hx
  unfold htmlEnc at hx
  split at hx
  · simp at hx; rcases hx with h | h | h | h | h <;> subst h <;> decide
  split at hx
  · simp at hx; rcases hx with h | h | h | h <;> subst h <;> decide
  split at hx
  · simp at hx; rcases hx with h | h | h | h <;> subst h <;> decide
  split at hx
  · simp at hx; rcases hx with h | h | h | h | h | h <;> subst h <;> decide
  split at hx
  · simp at hx; rcases hx with h | h | h | h | h <;> subst h <;> decide
  · simp at hx; subst hx; unfold htmlStruct; simp_all
end Ctx

/-! ## Secrecy -/
inductive Lbl | pub | sec deriving DecidableEq

/-- Expressions of the public fragment: secret variables appear only
    under `eqS`, the equality of two secrets (the one release). -/
inductive PE
  | lit (n : Int)
  | pvar (i : Nat)
  | add (a b : PE)
  | eqS (i j : Nat)
  | ite (c a b : PE)

structure Env where
  pub : Nat → Int
  sec : Nat → Int

def eval (ρ : Env) : PE → Int
  | .lit n => n
  | .pvar i => ρ.pub i
  | .add a b => eval ρ a + eval ρ b
  | .eqS i j => if ρ.sec i = ρ.sec j then 1 else 0
  | .ite c a b => if eval ρ c ≠ 0 then eval ρ a else eval ρ b

/-- Low equivalence: same public inputs and same secret-equality bits. -/
def LowEq (ρ₁ ρ₂ : Env) : Prop :=
  (∀ i, ρ₁.pub i = ρ₂.pub i) ∧ (∀ i j, (ρ₁.sec i = ρ₁.sec j ↔ ρ₂.sec i = ρ₂.sec j))

theorem noninterference (ρ₁ ρ₂ : Env) (h : LowEq ρ₁ ρ₂) (e : PE) : eval ρ₁ e = eval ρ₂ e := by
  induction e with
  | lit n => rfl
  | pvar i => exact h.1 i
  | add a b iha ihb => simp [eval, iha, ihb]
  | eqS i j =>
    simp only [eval]
    by_cases q : ρ₁.sec i = ρ₁.sec j
    · simp [q, (h.2 i j).mp q]
    · have : ¬ ρ₂.sec i = ρ₂.sec j := fun r => q ((h.2 i j).mpr r)
      simp [q, this]
  | ite c a b ihc iha ihb => simp [eval, ihc, iha, ihb]

/-- The emitted outputs of a run are public expressions (W-SEC-1). -/
theorem outputs_noninterfere (ρ₁ ρ₂ : Env) (h : LowEq ρ₁ ρ₂) (outs : List PE) :
    outs.map (eval ρ₁) = outs.map (eval ρ₂) := by
  induction outs with
  | nil => rfl
  | cons e es ih => simp [noninterference ρ₁ ρ₂ h e, ih]

/-! ## Authority -/
section Cap
variable {G O E : Type} (hasCap : E → Bool)

/-- A state is a guarded part and the rest. W-CAP-2: a step of an event
    without the capability does not write the guarded part. -/
def CapWF (d : G × O → E → G × O) : Prop := ∀ s e, hasCap e = false → (d s e).1 = s.1

theorem guarded_needs_cap (d : G × O → E → G × O) (h : CapWF hasCap d)
    (s : G × O) (log : List E) (hl : ∀ e ∈ log, hasCap e = false) :
    (log.foldl d s).1 = s.1 := by
  induction log generalizing s with
  | nil => rfl
  | cons e es ih =>
    simp only [List.foldl]
    rw [ih (d s e) (fun x hx => hl x (List.mem_cons_of_mem _ hx))]
    exact h s e (hl e (List.mem_cons_self _ _))
end Cap

end RCE.Label
