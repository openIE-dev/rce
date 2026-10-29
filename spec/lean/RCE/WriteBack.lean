/-
T33 `&mut` restore on failure (F.4.4). A call `f(&mut x)` runs the callee on
the value of x and ends with an outcome (a value or a failure). A call that
returns writes the final value of x back; a call that ends by `fail` leaves x
as it was at the call. Both are proved of the model below (copy in, run on
the copy, copy out only when the outcome is a value). The earlier rule, which
wrote back on every outcome, is retired and is not stated in this tree.
Whether every effect since entry is undoable (W-TX-1) is a check of the
checker, not of this model. A trap still keeps the state for the whole event (T8).
-/
namespace RCE.WriteBack

variable {L V O : Type} [DecidableEq L]

def upd (σ : L → V) (l : L) (v : V) : L → V := fun k => if k = l then v else σ k

/-- Copy in, run on the copy, copy out only when the outcome is a value
(`none` = no failure). -/
def call (σ : L → V) (l : L) (f : V → V × Option O) : (L → V) × Option O :=
  let r := f (σ l)
  match r.2 with
  | none => (upd σ l r.1, none)
  | some e => (σ, some e)

/-- A call that ends by `fail` leaves the caller's state as at the call. -/
theorem t33_failure_restores (σ : L → V) (l : L) (f : V → V × Option O) (e : O)
    (h : (f (σ l)).2 = some e) : (call σ l f).1 = σ := by
  simp [call, h]

/-- A call that returns writes the callee's final value back. -/
theorem t33_return_writes_back (σ : L → V) (l : L) (f : V → V × Option O)
    (h : (f (σ l)).2 = none) : (call σ l f).1 = upd σ l (f (σ l)).1 := by
  simp [call, h]

end RCE.WriteBack
