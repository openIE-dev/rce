/-
T33 `&mut` write-back on failure. A call `f(&mut x)` runs the callee on
the value of x and ends with an outcome (a value or a failure). The
emitted Rust updates x in place, so writes made before a `fail` remain.
Copy-in/copy-out that writes back on every outcome equals in-place
update; writing back only on success does not (there is a callee where
they differ). So the reference evaluator writes back on failure too, and
a trap still refuses the whole event (T8), which restores the state.
-/
namespace RCE.WriteBack

variable {L V O : Type} [DecidableEq L]

def upd (σ : L → V) (l : L) (v : V) : L → V := fun k => if k = l then v else σ k

/-- In place: the callee mutates the location as it runs. -/
def inplace (σ : L → V) (l : L) (f : V → V × Option O) : (L → V) × Option O :=
  (upd σ l (f (σ l)).1, (f (σ l)).2)

/-- Copy in, run on the copy, copy out on every outcome (the rule). -/
def copyAll (σ : L → V) (l : L) (f : V → V × Option O) : (L → V) × Option O :=
  let r := f (σ l)
  (upd σ l r.1, r.2)

/-- Copy out only when the outcome is a value (`none` = no failure). -/
def copyOk (σ : L → V) (l : L) (f : V → V × Option O) : (L → V) × Option O :=
  let r := f (σ l)
  match r.2 with
  | none => (upd σ l r.1, none)
  | some e => (σ, some e)

theorem t33 (σ : L → V) (l : L) (f : V → V × Option O) :
    copyAll σ l f = inplace σ l f := rfl

/-- Writing back only on success disagrees with in place. -/
theorem copyOk_differs : ∃ (σ : Unit → Nat) (f : Nat → Nat × Option Unit),
    copyOk σ () f ≠ inplace σ () f := by
  refine ⟨fun _ => 0, fun v => (v + 1, some ()), ?_⟩
  intro h
  have := congrArg (fun p => p.1 ()) h
  simp [copyOk, inplace, upd] at this

end RCE.WriteBack
