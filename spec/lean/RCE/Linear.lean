/-
  Linear handles (spec 5.12, W-LIN-1, draft 0.5).
  Model: a body over one linear handle. `use` consumes it; `ite` branches on a
  run-time condition; `rep n s` is a bounded loop. The checker's count `lin`
  requires both branches to consume the same number of times and a loop body
  to consume nothing (a move inside a loop is reported).
  Theorem `linear_exactly_once`: if the checker accepts (lin s = some 1),
  every execution, whatever its conditions, consumes the handle exactly once.
-/
namespace RCE.Linear

inductive Stmt where
  | use
  | skip
  | seq (a b : Stmt)
  | ite (c : Nat) (a b : Stmt)   -- condition number `c`, decided at run time
  | rep (n : Nat) (body : Stmt)

/-- The checker's count of consumptions, `none` when the count has no value (W-LIN-1). -/
def lin : Stmt → Option Nat
  | .use => some 1
  | .skip => some 0
  | .seq a b => match lin a, lin b with
    | some x, some y => some (x + y)
    | _, _ => none
  | .ite _ a b => match lin a, lin b with
    | some x, some y => if x = y then some x else none
    | _, _ => none
  | .rep _ body => if lin body = some 0 then some 0 else none

/-- Consumptions on the path chosen by the environment `env`
    (`env i c` is the value of condition `c` at its `i`-th evaluation; the
    path may differ between loop iterations). -/
def uses (env : Nat → Nat → Bool) : Stmt → Nat
  | .use => 1
  | .skip => 0
  | .seq a b => uses env a + uses env b
  | .ite c a b => if env 0 c then uses env a else uses env b
  | .rep n body => (List.range n).foldl (fun acc i => acc + uses (fun j => env (i + j)) body) 0

theorem lin_sound : ∀ (s : Stmt) (k : Nat), lin s = some k → ∀ env, uses env s = k := by
  intro s
  induction s with
  | use => intro k h env; simp [lin] at h; simp [uses, h]
  | skip => intro k h env; simp [lin] at h; simp [uses, h]
  | seq a b iha ihb =>
    intro k h env
    simp only [lin] at h
    cases ha : lin a <;> cases hb : lin b <;> simp [ha, hb] at h
    simp [uses, iha _ ha env, ihb _ hb env, h]
  | ite c a b iha ihb =>
    intro k h env
    simp only [lin] at h
    cases ha : lin a <;> cases hb : lin b <;> simp [ha, hb] at h
    rename_i x y
    obtain ⟨hxy, hk⟩ := h
    simp only [uses]
    split
    · rw [iha _ ha env]; exact hk
    · rw [ihb _ hb env]; omega
  | rep n body ih =>
    intro k h env
    simp only [lin] at h
    split at h
    · rename_i h0
      simp at h; subst h
      simp only [uses]
      have : ∀ (l : List Nat) (acc : Nat), l.foldl (fun acc i => acc + uses (fun j => env (i + j)) body) acc = acc := by
        intro l
        induction l with
        | nil => intro acc; rfl
        | cons x xs ihl => intro acc; simp only [List.foldl]; rw [ih 0 h0]; simpa using ihl acc
      exact this _ 0
    · simp at h

/-- Exactly once on every path. -/
theorem linear_exactly_once (s : Stmt) (h : lin s = some 1) (env : Nat → Nat → Bool) : uses env s = 1 :=
  lin_sound s 1 h env

/-- A body that consumes on one branch only is reported. -/
example : lin (.ite 0 .use .skip) = none := by decide

/-- A move inside a loop is reported. -/
example : lin (.rep 3 .use) = none := by decide

end RCE.Linear
