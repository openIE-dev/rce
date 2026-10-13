/-
  Structural recursion (spec 6.7, W-REC-1 as revised in draft 0.4).
  Model: a recursive kind with a leaf constructor and a binary constructor;
  a structurally recursive function is given by its leaf case `fl` and its
  combining case `fc`, each total (they are ordinary bounded RCE bodies, T2).
  Evaluation is run with explicit fuel, one unit per call.
  Theorems: with fuel ≥ size t the evaluation always returns a value
  (termination with an explicit bound), the result does not depend on
  how much extra fuel is given (determinism of the bound), and the number of
  calls made equals size t.
-/
namespace RCE.Struct

inductive Tree where
  | leaf (v : Int)
  | node (l r : Tree)

def size : Tree → Nat
  | .leaf _ => 1
  | .node l r => size l + size r + 1

def evalF (fl : Int → Int) (fc : Int → Int → Int) : Nat → Tree → Option Int
  | 0, _ => none
  | _ + 1, .leaf v => some (fl v)
  | n + 1, .node l r =>
    match evalF fl fc n l, evalF fl fc n r with
    | some a, some b => some (fc a b)
    | _, _ => none

/-- The value a structurally recursive function denotes (no fuel). -/
def den (fl : Int → Int) (fc : Int → Int → Int) : Tree → Int
  | .leaf v => fl v
  | .node l r => fc (den fl fc l) (den fl fc r)

/-- Termination with an explicit bound: fuel `size t` always suffices,
    and the result is the denotation. -/
theorem struct_total (fl : Int → Int) (fc : Int → Int → Int) :
    ∀ (t : Tree) (fuel : Nat), size t ≤ fuel → evalF fl fc fuel t = some (den fl fc t) := by
  intro t
  induction t with
  | leaf v =>
    intro fuel h
    cases fuel with
    | zero => simp [size] at h
    | succ n => simp [evalF, den]
  | node l r ihl ihr =>
    intro fuel h
    cases fuel with
    | zero => simp [size] at h
    | succ n =>
      have hl : size l ≤ n := by simp [size] at h; omega
      have hr : size r ≤ n := by simp [size] at h; omega
      simp [evalF, den, ihl n hl, ihr n hr]

/-- Number of calls a structurally recursive evaluation makes. -/
def calls : Tree → Nat
  | .leaf _ => 1
  | .node l r => calls l + calls r + 1

theorem calls_eq_size : ∀ t, calls t = size t := by
  intro t; induction t with
  | leaf v => rfl
  | node l r ihl ihr => simp [calls, size, ihl, ihr]

/-- Extra fuel never changes the result (the bound is not observable). -/
theorem struct_fuel_irrelevant (fl : Int → Int) (fc : Int → Int → Int) (t : Tree) (m n : Nat)
    (hm : size t ≤ m) (hn : size t ≤ n) : evalF fl fc m t = evalF fl fc n t := by
  rw [struct_total fl fc t m hm, struct_total fl fc t n hn]

end RCE.Struct

#print axioms RCE.Struct.struct_total
#print axioms RCE.Struct.calls_eq_size
