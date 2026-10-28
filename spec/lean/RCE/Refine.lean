/-! P2 (spec/proposals/CORE-PROPERTIES.md): declared predicates, spec 5.1
    "Refined types" and W-REF-1. A refined type `type R = τ where p` has as
    values the values of τ that satisfy p. p is in a decidable fragment
    (comparisons of `self` or `self.len()` with integer constants, character
    classes of text, and/or/not), so `holds` is a total Bool function.
    A refined value is checked at the same boundaries as an integer range
    (6.3 `fit`): event admission, state write-back (6.4 step 3), typed
    bindings. A failed check gives the event the kept outcome with the state unchanged.
    Theorems: `holds_total` (the check decides), `fit_sound` (a value that
    passes the boundary satisfies p), `refine_preserved` (every state
    reachable by the checked δ from an initial state satisfying the field
    predicates satisfies them), `refine_kept_unchanged`. -/
namespace RCE.Refine

inductive Cmp | lt | le | gt | ge | eq | ne
  deriving DecidableEq, Repr

inductive Cls | digit | alpha | space | empty
  deriving DecidableEq, Repr

inductive Term | self | len | lit (n : Int)
  deriving DecidableEq, Repr

inductive Pred where
  | cmp (op : Cmp) (a b : Term)
  | cls (c : Cls)
  | and (p q : Pred)
  | or (p q : Pred)
  | not (p : Pred)
  deriving DecidableEq, Repr

/-- The observable parts of a value the fragment can read. -/
structure Obs where
  num : Int
  len : Nat
  cls : Cls → Bool

def term (v : Obs) : Term → Int
  | .self => v.num
  | .len => v.len
  | .lit n => n

def cmp : Cmp → Int → Int → Bool
  | .lt, a, b => decide (a < b)
  | .le, a, b => decide (a ≤ b)
  | .gt, a, b => decide (a > b)
  | .ge, a, b => decide (a ≥ b)
  | .eq, a, b => decide (a = b)
  | .ne, a, b => decide (a ≠ b)

def holds (v : Obs) : Pred → Bool
  | .cmp op a b => cmp op (term v a) (term v b)
  | .cls c => v.cls c
  | .and p q => holds v p && holds v q
  | .or p q => holds v p || holds v q
  | .not p => !holds v p

/-- The check is a function: it decides, and the same value always gives
    the same answer (determinism of admission). -/
theorem holds_total (v : Obs) (p : Pred) : holds v p = true ∨ holds v p = false := by
  cases holds v p <;> simp

/-- The boundary check (`fit` for a refined type). -/
def fit (p : Pred) (v : Obs) : Option Obs := if holds v p then some v else none

theorem fit_sound (p : Pred) (v w : Obs) (h : fit p v = some w) : holds w p = true := by
  unfold fit at h
  split at h
  · cases h; assumption
  · cases h

/-- A state is an assignment of observable values to fields, each field
    with its predicate (fields of unrefined type carry `.cmp .eq .len .len`,
    which always holds). -/
def Inv (ps : List Pred) (s : List Obs) : Prop :=
  ∀ i (h1 : i < ps.length) (h2 : i < s.length), holds (s.get ⟨i, h2⟩) (ps.get ⟨i, h1⟩) = true

instance (ps : List Pred) (s : List Obs) : Decidable (Inv ps s) := by
  unfold Inv; infer_instance

/-- δ checked at write-back (6.4 step 3): the raw step `d` may refuse
    (`none`); a result whose fields fail their predicates has the kept outcome too;
    a kept event leaves the state unchanged (T8). -/
def checked {E : Type} (ps : List Pred) (d : List Obs → E → Option (List Obs))
    (s : List Obs) (e : E) : List Obs :=
  match d s e with
  | some s' => if Inv ps s' then s' else s
  | none => s

theorem refine_step {E : Type} (ps : List Pred) (d : List Obs → E → Option (List Obs))
    (s : List Obs) (e : E) (h : Inv ps s) : Inv ps (checked ps d s e) := by
  unfold checked
  split
  · split
    · assumption
    · exact h
  · exact h

/-- Every state reachable from the initial state by the log satisfies the
    field predicates (the fold law of 6.1). -/
theorem refine_preserved {E : Type} (ps : List Pred) (d : List Obs → E → Option (List Obs))
    (s0 : List Obs) (log : List E) (h : Inv ps s0) :
    Inv ps (log.foldl (checked ps d) s0) := by
  induction log generalizing s0 with
  | nil => exact h
  | cons e es ih => exact ih _ (refine_step ps d s0 e h)

theorem refine_kept_unchanged {E : Type} (ps : List Pred)
    (d : List Obs → E → Option (List Obs)) (s s' : List Obs) (e : E)
    (hd : d s e = some s') (hn : ¬ Inv ps s') : checked ps d s e = s := by
  unfold checked; rw [hd]; simp [hn]

/-- Admission (6.1, 6.4): an event whose payload fields fail their
    predicates has the kept outcome before any step runs. -/
def admit (ps : List Pred) (payload : List Obs) : Bool := decide (Inv ps payload)

theorem admit_sound (ps : List Pred) (payload : List Obs) (h : admit ps payload = true) :
    Inv ps payload := by
  unfold admit at h; exact of_decide_eq_true h

end RCE.Refine
