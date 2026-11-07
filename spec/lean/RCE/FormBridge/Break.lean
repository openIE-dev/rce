import RCE.Form
import RCE.Iter

/-!
  Break as a carried flag, as a definition in this directory.

  `rce-form.md` §6 runs a count-bounded Iterate by testing, then raising
  the counter, then trapping "loop bound" when the next iteration would
  pass the bound, and only then running the body. A break in the body
  returns false. §13 says the schedule writes `if not k { break 0 }` in
  that place, and that construct ∘ schedule is the identity on canonical
  Forms while schedule ∘ construct is not the identity on the core.
  ONE-15 (`one-emitter.md`) carries an exit count when the depth is above
  0. A single boolean has two values, so it does not name three depths and
  it does not name the three outcomes of `RCE.Iter.Out`.

  `RCE.iter` in `spec/lean/RCE/Form.lean` has no break. These definitions
  do not add one. The flag relation below is a projection: the same boolean
  is the image of more than one result, so a flag is not by itself an
  equivalence. No theorem says these functions equal
  `spec/checker/src/form/bridge.rs`.
-/

namespace RCE.FormBridge.Break

/-- Order context for one loop. `log` records effects in the order they
    run. `fuel` is a count in this definition, not a `fuel.tick` node. -/
structure Ω where
  fuel : Nat
  log : List Nat
  deriving DecidableEq, Repr

/-- How a count loop ends. `.done` is a false test. `.stopped` is a false
    body flag. `.bound` is the trap before the body. `.fuel` is a test or a
    body that stopped for fuel. -/
inductive LoopRes where
  | done (v : Nat) (ω : Ω)
  | stopped (v : Nat) (ω : Ω)
  | bound (v : Nat) (ω : Ω)
  | fuel (v : Nat) (ω : Ω)
  deriving DecidableEq, Repr

/-- §6, with the body flag. `n` is how many body runs are still allowed.
    The test runs first. On true, `n = 0` traps before the body. On
    `n + 1` the body runs, and a false flag returns `.stopped` without
    another test. -/
def countLoop
    (test : Nat → Ω → Option (Bool × Nat × Ω))
    (body : Nat → Ω → Option (Bool × Nat × Ω)) :
    Nat → Nat → Ω → LoopRes
  | n, v, ω =>
    match test v ω with
    | none => .fuel v ω
    | some (false, v1, ω1) => .done v1 ω1
    | some (true, v1, ω1) =>
      match n with
      | 0 => .bound v1 ω1
      | n + 1 =>
        match body v1 ω1 with
        | none => .fuel v1 ω1
        | some (false, v2, ω2) => .stopped v2 ω2
        | some (true, v2, ω2) => countLoop test body n v2 ω2

/-- The same shape with the body flag ignored, so a break falls into the
    next iteration. This is the late test named by the carried-flag note:
    a break is then able to meet the bound trap, and the next test runs. -/
def countLoopLate
    (test : Nat → Ω → Option (Bool × Nat × Ω))
    (body : Nat → Ω → Option (Bool × Nat × Ω)) :
    Nat → Nat → Ω → LoopRes
  | n, v, ω =>
    match test v ω with
    | none => .fuel v ω
    | some (false, v1, ω1) => .done v1 ω1
    | some (true, v1, ω1) =>
      match n with
      | 0 => .bound v1 ω1
      | n + 1 =>
        match body v1 ω1 with
        | none => .fuel v1 ω1
        | some (_, v2, ω2) => countLoopLate test body n v2 ω2

theorem bound_before_body
    (test : Nat → Ω → Option (Bool × Nat × Ω))
    (body : Nat → Ω → Option (Bool × Nat × Ω))
    (v v1 : Nat) (ω ω1 : Ω)
    (ht : test v ω = some (true, v1, ω1)) :
    countLoop test body 0 v ω = .bound v1 ω1 := by
  simp [countLoop, ht]

theorem done_skips_body
    (test : Nat → Ω → Option (Bool × Nat × Ω))
    (body : Nat → Ω → Option (Bool × Nat × Ω))
    (n v v1 : Nat) (ω ω1 : Ω)
    (ht : test v ω = some (false, v1, ω1)) :
    countLoop test body n v ω = .done v1 ω1 := by
  simp [countLoop, ht]

theorem break_skips_next_iteration
    (test : Nat → Ω → Option (Bool × Nat × Ω))
    (body : Nat → Ω → Option (Bool × Nat × Ω))
    (n v v1 v2 : Nat) (ω ω1 ω2 : Ω)
    (ht : test v ω = some (true, v1, ω1))
    (hb : body v1 ω1 = some (false, v2, ω2)) :
    countLoop test body (n + 1) v ω = .stopped v2 ω2 := by
  simp [countLoop, ht, hb]

theorem fuel_test_skips_body
    (test : Nat → Ω → Option (Bool × Nat × Ω))
    (body : Nat → Ω → Option (Bool × Nat × Ω))
    (n v : Nat) (ω : Ω)
    (ht : test v ω = none) :
    countLoop test body n v ω = .fuel v ω := by
  simp [countLoop, ht]

/-- Why a false flag was produced. The boolean does not carry this. -/
inductive Why where
  | testFalse
  | bodyFalse
  deriving DecidableEq, Repr

/-- Projection from a finished loop to a flag witness. `.bound` and
    `.fuel` have no witness: they are not a false flag. -/
inductive Projects : LoopRes → Why → Bool → Nat → Ω → Prop where
  | done {v ω} : Projects (.done v ω) .testFalse false v ω
  | stopped {v ω} : Projects (.stopped v ω) .bodyFalse false v ω

theorem projects_bound_has_no_witness (v : Nat) (ω : Ω)
    (why : Why) (b : Bool) (c : Nat) (ω' : Ω) :
    ¬ Projects (.bound v ω) why b c ω' := by
  intro h
  cases h

theorem projects_fuel_has_no_witness (v : Nat) (ω : Ω)
    (why : Why) (b : Bool) (c : Nat) (ω' : Ω) :
    ¬ Projects (.fuel v ω) why b c ω' := by
  intro h
  cases h

/-- One boolean and one value are the image of two results that are not
    equal. The witness `Why` is what tells them apart. -/
theorem flag_is_not_equivalence (v : Nat) (ω : Ω) :
    Projects (.done v ω) .testFalse false v ω ∧
    Projects (.stopped v ω) .bodyFalse false v ω ∧
    (.done v ω : LoopRes) ≠ .stopped v ω ∧
    Why.testFalse ≠ Why.bodyFalse := by
  refine ⟨.done, .stopped, ?_, ?_⟩
  · intro h
    cases h
  · intro h
    cases h

theorem projects_why_is_unique (v : Nat) (ω : Ω) :
    ¬ Projects (.done v ω) .bodyFalse false v ω := by
  intro h
  cases h

/-- `RCE.Iter.Out` keeps finished, broken and bound apart. The flag
    projection below identifies the first two and leaves bound out. -/
def projectIter : RCE.Iter.Out Nat → Option (Bool × List Nat)
  | .done xs => some (false, xs)
  | .stopped xs => some (false, xs)
  | .bound _ => none

theorem iter_outcomes_distinct :
    (RCE.Iter.Out.done [0] : RCE.Iter.Out Nat) ≠ .stopped [0] ∧
    (RCE.Iter.Out.stopped [0]) ≠ .bound [0] ∧
    (RCE.Iter.Out.done [0]) ≠ .bound [0] := by
  refine ⟨?_, ?_, ?_⟩ <;> intro h <;> cases h

theorem projectIter_collapses_done_and_stopped :
    projectIter (.done [1]) = some (false, [1]) ∧
    projectIter (.stopped [1]) = some (false, [1]) ∧
    projectIter (.bound [1]) = none ∧
    projectIter (.done [1]) = projectIter (.stopped [1]) := by
  refine ⟨rfl, rfl, rfl, rfl⟩

theorem projectIter_not_injective :
    ¬ (∀ a b : RCE.Iter.Out Nat, projectIter a = projectIter b → a = b) := by
  intro h
  have heq := h (.done [1]) (.stopped [1]) rfl
  cases heq

/-- The existing `RCE.iter` traps when the bound is already spent and the
    test is true, and the body is not the result. That body would have
    written 9. -/
theorem existing_iter_traps_before_body :
    RCE.iter
      (fun _ _ => .ok [.int 9] [])
      (fun _ => .ok (.bool true))
      0 [.int 0] [] =
    .ref (.trap .loopBound) := rfl

/-- A test that records `0` and spends one unit of fuel. Fuel 0 stops
    before a boolean is produced. -/
def effTest (v : Nat) (ω : Ω) : Option (Bool × Nat × Ω) :=
  if ω.fuel = 0 then none
  else some (true, v, ⟨ω.fuel - 1, ω.log ++ [0]⟩)

/-- A body that breaks, records `1`, and does not spend fuel. -/
def brkBody (v : Nat) (ω : Ω) : Option (Bool × Nat × Ω) :=
  some (false, v + 1, ⟨ω.fuel, ω.log ++ [1]⟩)

/-- A body that continues and records `1` without spending fuel. -/
def goBody (v : Nat) (ω : Ω) : Option (Bool × Nat × Ω) :=
  some (true, v, ⟨ω.fuel, ω.log ++ [1]⟩)

/-- The one legal iteration breaks. The result is `.stopped`, the test ran
    once, and the bound trap is not the result. -/
theorem break_at_final_iteration :
    countLoop effTest brkBody 1 0 ⟨3, []⟩ =
      .stopped 1 ⟨2, [0, 1]⟩ := rfl

/-- Continuing from that same final iteration runs the next test and then
    traps, before a second body. -/
theorem continue_past_final_iteration_bounds :
    countLoop effTest goBody 1 0 ⟨3, []⟩ =
      .bound 0 ⟨1, [0, 1, 0]⟩ := rfl

theorem break_at_final_is_not_bound :
    countLoop effTest brkBody 1 0 ⟨3, []⟩ ≠
      countLoop effTest goBody 1 0 ⟨3, []⟩ := by
  intro h
  simp [break_at_final_iteration, continue_past_final_iteration_bounds] at h

/-- A test with no fuel does not reach the breaking body. -/
theorem effectful_test_traps_before_break :
    countLoop effTest brkBody 1 0 ⟨0, []⟩ = .fuel 0 ⟨0, []⟩ := rfl

/-- Ignoring the flag on the final iteration runs the test again and traps.
    The log gains a second `0`. The fuel is one lower. -/
theorem late_flag_retests_and_bounds :
    countLoopLate effTest brkBody 1 0 ⟨3, []⟩ =
      .bound 1 ⟨1, [0, 1, 0]⟩ ∧
    countLoop effTest brkBody 1 0 ⟨3, []⟩ =
      .stopped 1 ⟨2, [0, 1]⟩ ∧
    countLoopLate effTest brkBody 1 0 ⟨3, []⟩ ≠
      countLoop effTest brkBody 1 0 ⟨3, []⟩ := by
  refine ⟨rfl, rfl, ?_⟩
  intro h
  cases h

/-- Setting the flag to a literal does not change ω. A flag that spends
    fuel does. -/
def flagLiteral (b : Bool) (ω : Ω) : Bool × Ω := (b, ω)

def flagTick (b : Bool) (ω : Ω) : Option (Bool × Ω) :=
  if ω.fuel = 0 then none
  else some (b, ⟨ω.fuel - 1, ω.log ++ [9]⟩)

theorem flag_literal_keeps_omega (b : Bool) (ω : Ω) :
    (flagLiteral b ω).2 = ω := rfl

theorem flag_tick_changes_omega :
    flagTick false ⟨2, [1]⟩ = some (false, ⟨1, [1, 9]⟩) ∧
    flagTick false ⟨2, [1]⟩ ≠ some (false, ⟨2, [1]⟩) := by
  refine ⟨rfl, ?_⟩
  intro h
  cases h

/-- Actions inside a body. `break d` ends the list. Later actions do not
    run. `tick` spends one unit of fuel or stops with the fuel text. -/
inductive Act where
  | note (n : Nat)
  | tick
  | break (d : Nat)
  deriving DecidableEq, Repr

def noBreak : List Act → Bool
  | [] => true
  | .break _ :: _ => false
  | .note _ :: xs => noBreak xs
  | .tick :: xs => noBreak xs

def run (fuel : Nat) : List Act → Except String (List Nat × Nat × Option Nat)
  | [] => .ok ([], fuel, none)
  | .note n :: rest =>
    match run fuel rest with
    | .ok (log, f, e) => .ok (n :: log, f, e)
    | .error s => .error s
  | .tick :: rest =>
    if fuel = 0 then .error "comptime fuel exhausted"
    else
      match run (fuel - 1) rest with
      | .ok (log, f, e) => .ok (log, f, e)
      | .error s => .error s
  | .break d :: _ => .ok ([], fuel, some d)

theorem run_note_cong {n fuel : Nat} {a b : List Act}
    (h : run fuel a = run fuel b) :
    run fuel (.note n :: a) = run fuel (.note n :: b) := by
  simp only [run]
  rw [h]

theorem run_tick_cong {fuel : Nat} {a b : List Act}
    (h : run (fuel - 1) a = run (fuel - 1) b) :
    run fuel (.tick :: a) = run fuel (.tick :: b) := by
  simp only [run]
  split
  · rfl
  · rw [h]

theorem run_break_drops_tail :
    ∀ (pre post : List Act) (d fuel : Nat),
      noBreak pre = true →
      run fuel (pre ++ .break d :: post) = run fuel (pre ++ [.break d])
  | [], _, _, _, _ => rfl
  | .note n :: as, post, d, fuel, h => by
      have h' : noBreak as = true := by simpa [noBreak] using h
      have ih := run_break_drops_tail as post d fuel h'
      exact run_note_cong ih
  | .tick :: as, post, d, fuel, h => by
      have h' : noBreak as = true := by simpa [noBreak] using h
      have ih := run_break_drops_tail as post d (fuel - 1) h'
      exact run_tick_cong ih
  | .break _ :: _, _, _, _, h => by
      simp [noBreak, Bool.false_eq_true] at h

/-- ONE-15. Depth 0 stops at this construct. Depth `d + 1` leaves it and
    still has `d` to propagate. -/
def propagate : Nat → Option Nat
  | 0 => none
  | d + 1 => some d

theorem propagate_zero : propagate 0 = none := rfl

theorem propagate_succ (d : Nat) : propagate (d + 1) = some d := rfl

/-- Three depths cannot land in two booleans without a collision. -/
theorem bool_cannot_separate_three_depths (encode : Nat → Bool) :
    encode 0 = encode 1 ∨ encode 0 = encode 2 ∨ encode 1 = encode 2 := by
  cases encode 0 <;> cases encode 1 <;> cases encode 2 <;> simp_all

inductive Leave where
  | inner (log : List Nat) (fuel : Nat)
  | outer (log : List Nat) (fuel : Nat)
  | fell (log : List Nat) (fuel : Nat)
  | stuck (text : String)
  deriving DecidableEq, Repr

def runTwo (fuel : Nat) (inner : List Act) : Leave :=
  match run fuel inner with
  | .error s => .stuck s
  | .ok (log, f, none) => .fell log f
  | .ok (log, f, some d) =>
    match propagate d with
    | none => .inner log f
    | some _ => .outer log f

theorem depth0_suppresses_later_effects :
    runTwo 3 [.note 1, .break 0, .note 9] = .inner [1] 3 := rfl

theorem depth1_propagates_and_suppresses_later_effects :
    runTwo 3 [.note 1, .break 1, .note 9] = .outer [1] 3 := rfl

theorem prior_tick_kept_later_tick_not_run :
    runTwo 3 [.tick, .note 1, .break 0, .tick] = .inner [1] 2 := rfl

theorem depth0_ne_depth1 :
    runTwo 3 [.break 0] ≠ runTwo 3 [.break 1] := by
  intro h
  cases h

/-- A measure continue path raises the index by one. The break path may
    leave the index. A continue path that raises the index only on some
    values is not that shape. -/
structure Idx where
  i : Nat
  j : Nat
  deriving DecidableEq, Repr

def stepContinue (m : Idx) : Idx := ⟨m.i + 1, m.j⟩

def stepBreak (m : Idx) : Idx := m

def condInc (m : Idx) : Idx :=
  if m.i % 2 = 0 then ⟨m.i + 1, m.j⟩ else m

theorem break_path_keeps_index (m : Idx) : stepBreak m = m := rfl

theorem cond_inc_is_not_measure_continue :
    condInc ⟨1, 4⟩ = ⟨1, 4⟩ ∧
    stepContinue ⟨1, 4⟩ = ⟨2, 4⟩ ∧
    condInc ⟨1, 4⟩ ≠ stepContinue ⟨1, 4⟩ := by
  refine ⟨rfl, rfl, ?_⟩
  intro h
  cases h

/-- Names a branch arm can capture. ω is a name, not a boolean on the side. -/
inductive Name where
  | v (n : Nat)
  | omega
  deriving DecidableEq, Repr

def mem : Name → List Name → Bool
  | _, [] => false
  | x, y :: ys => if x = y then true else mem x ys

def stripOmega : List Name → List Name
  | [] => []
  | .omega :: xs => stripOmega xs
  | .v n :: xs => .v n :: stripOmega xs

def dedup : List Name → List Name
  | [] => []
  | x :: xs =>
    let ys := dedup xs
    if mem x ys then ys else x :: ys

/-- Captures, with ω last when either list holds it. Plain names keep the
    first-seen order of `dedup` after ω is removed. This is not an order by
    Merkle digest. -/
def align (xs : List Name) : List Name :=
  let base := dedup (stripOmega xs)
  if mem .omega xs then base ++ [.omega] else base

/-- Both arms are handed this one list, built from the union of the uses.
    It is not the live set left after the first arm runs. -/
def armInputs (u0 u1 : List Name) : List Name :=
  align (u0 ++ u1)

def dropName (x : Name) : List Name → List Name
  | [] => []
  | y :: ys => if x = y then dropName x ys else y :: dropName x ys

/-- What the second arm would see if the first arm's consumption of ω were
    left in the list. -/
def buggyArm1 (u0 u1 : List Name) : List Name :=
  dropName .omega (armInputs u0 u1)

theorem mem_append (x : Name) :
    ∀ a b, mem x (a ++ b) = (mem x a || mem x b)
  | [], b => by simp [mem]
  | y :: ys, b => by
      have ih := mem_append x ys b
      simp only [List.cons_append, mem]
      split
      · simp
      · simp [ih]

theorem omega_from_either_arm :
    armInputs [.v 1] [.omega] = [.v 1, .omega] ∧
    armInputs [.omega] [.v 2] = [.v 2, .omega] ∧
    armInputs [.v 1] [.v 2] = [.v 1, .v 2] ∧
    armInputs [.omega, .v 1] [.v 1, .omega] = [.v 1, .omega] := by
  refine ⟨rfl, rfl, rfl, rfl⟩

theorem shared_omega_is_on_both_arms :
    mem .omega (armInputs [.omega] [.omega]) = true ∧
    mem .omega (buggyArm1 [.omega] [.omega]) = false := by
  refine ⟨rfl, rfl⟩

def rename (ρ : Nat → Nat) : Name → Name
  | .v n => .v (ρ n)
  | .omega => .omega

def renameAll (ρ : Nat → Nat) (xs : List Name) : List Name :=
  xs.map (rename ρ)

/-- Renaming both arms by the same successor renames the shared capture
    list. ω stays ω. -/
theorem arm_rename_succ :
    let ρ : Nat → Nat := fun n => n + 1
    armInputs (renameAll ρ [.v 0, .omega]) (renameAll ρ [.v 2]) =
      renameAll ρ (armInputs [.v 0, .omega] [.v 2]) := rfl

end RCE.FormBridge.Break
