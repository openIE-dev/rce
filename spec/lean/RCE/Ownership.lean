/-! T6 Memory safety, ownership core (spec 5.5–5.7).
    An instrumented store tags every variable with its validity, ownership
    kind and region. Access to an invalid (moved, dropped, or freed-region)
    variable, or a call whose mutable borrows alias, is a *fault*. Theorem:
    a well-typed program never faults, and the static context tracks the
    dynamic store exactly. -/
namespace RCE.Own

inductive Kind where
  | copy | move
  deriving DecidableEq, Repr

/-- A live cell: ownership kind and the arena it was allocated in (if any). -/
structure Cell where
  kind : Kind
  region : Option Nat
  deriving DecidableEq, Repr

abbrev Var := Nat
abbrev Store := Var → Option Cell

def upd (σ : Store) (x : Var) (c : Option Cell) : Store := fun z => if z = x then c else σ z

inductive OStmt where
  | alloc (x : Var) (k : Kind) (r : Option Nat)   -- let x = <new value> [@r]
  | use (x : Var)                                 -- read x
  | mv (x y : Var)                                -- let y = x (move or copy)
  | drop (x : Var)                                -- end of x's scope
  | call (shr mu : List Var)                     -- f(&shr…, &mu mu…)
  | endArena (r : Nat)                            -- end of `with arena r`
  | seq (a b : OStmt)

/-- Shared XOR mutable at a call: mutable borrows are pairwise distinct and
    distinct from every shared borrow. -/
def aliasFree (shr mu : List Var) : Bool :=
  decide mu.Nodup && mu.all (fun v => !(shr.contains v))

def live (σ : Store) (xs : List Var) : Bool := xs.all fun v => (σ v).isSome

def freeArena (σ : Store) (r : Nat) : Store :=
  fun z => match σ z with
    | some c => if c.region = some r then none else some c
    | none => none

/-- Instrumented semantics; `none` is a fault. -/
def oexec (σ : Store) : OStmt → Option Store
  | .alloc x k r => some (upd σ x (some ⟨k, r⟩))
  | .use x => if (σ x).isSome then some σ else none
  | .mv x y =>
    match σ x with
    | none => none
    | some c => some (upd (if c.kind = .move then upd σ x none else σ) y (some c))
  | .drop x => if (σ x).isSome then some (upd σ x none) else none
  | .call shr mu => if live σ (shr ++ mu) && aliasFree shr mu then some σ else none
  | .endArena r => some (freeArena σ r)
  | .seq a b => match oexec σ a with
    | none => none
    | some σ' => oexec σ' b

/-- Ownership typing Γ ⊢ s ⊣ Γ' (spec 5.5 O1–O4, 5.6, 5.7). The context has
    the same shape as the store; the checker computes it statically. -/
inductive OWt : Store → OStmt → Store → Prop where
  | alloc {Γ x k r} : OWt Γ (.alloc x k r) (upd Γ x (some ⟨k, r⟩))
  | use {Γ x} : (Γ x).isSome → OWt Γ (.use x) Γ
  | mv {Γ x y c} : Γ x = some c →
      OWt Γ (.mv x y) (upd (if c.kind = .move then upd Γ x none else Γ) y (some c))
  | drop {Γ x} : (Γ x).isSome → OWt Γ (.drop x) (upd Γ x none)
  | call {Γ shr mu} : live Γ (shr ++ mu) = true → aliasFree shr mu = true → OWt Γ (.call shr mu) Γ
  | endArena {Γ r} : OWt Γ (.endArena r) (freeArena Γ r)
  | seq {Γ Γ' Γ'' a b} : OWt Γ a Γ' → OWt Γ' b Γ'' → OWt Γ (.seq a b) Γ''

/-- **T6 Memory safety (ownership core)**: a well-typed program never uses a
    moved, dropped or arena-freed variable and never passes aliasing mutable
    borrows; the store after it is exactly the context the checker computed. -/
theorem memory_safety {Γ s Γ'} (h : OWt Γ s Γ') : ∀ σ, σ = Γ → oexec σ s = some Γ' := by
  induction h with
  | alloc => intro σ e; subst e; rfl
  | use hx => intro σ e; subst e; simp [oexec, hx]
  | mv hx => intro σ e; subst e; simp [oexec, hx]
  | drop hx => intro σ e; subst e; simp [oexec, hx]
  | call hl ha => intro σ e; subst e; simp [oexec, hl, ha]
  | endArena => intro σ e; subst e; rfl
  | seq _ _ iha ihb => intro σ e; simp [oexec, iha σ e, ihb _ rfl]

/-- Corollary: no fault is ever reached. -/
theorem no_fault {Γ s Γ'} (h : OWt Γ s Γ') : oexec Γ s ≠ none := by
  rw [memory_safety h Γ rfl]; simp

/-- Use-after-move is refused statically: after `let y = x` with a Move
    value, x is not live, so `use x` is ill-typed. -/
theorem use_after_move_no_typing {Γ x y c} (hx : Γ x = some c) (hk : c.kind = .move) (hxy : x ≠ y) :
    ¬ ∃ Γ', OWt Γ (.seq (.mv x y) (.use x)) Γ' := by
  intro ⟨Γ', h⟩
  cases h with
  | seq h1 h2 =>
    cases h1 with
    | mv hx' =>
      rw [hx] at hx'; cases hx'
      cases h2 with
      | use hl => simp [upd, hk, hxy] at hl

/-- Mutable aliasing is refused statically. -/
theorem mut_alias_no_typing {Γ x} : ¬ ∃ Γ', OWt Γ (.call [] [x, x]) Γ' := by
  intro ⟨Γ', h⟩; cases h with
  | call _ ha => simp [aliasFree] at ha

end RCE.Own
