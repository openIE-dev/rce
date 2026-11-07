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

/-- One run of a test or a body. `.fuel` keeps the order value at the
    trap, including effects already recorded. It is not a request to
    restore the value from before the step. -/
inductive StepRes where
  | cont (go : Bool) (v : Nat) (ω : Ω)
  | fuel (v : Nat) (ω : Ω)
  deriving DecidableEq, Repr

/-- §6, with the body flag. `n` is how many body runs are still allowed.
    The test runs first. On true, `n = 0` traps before the body. On
    `n + 1` the body runs, and a false flag returns `.stopped` without
    another test. A fuel result keeps the ω that step reported. -/
def countLoop
    (test : Nat → Ω → StepRes)
    (body : Nat → Ω → StepRes) :
    Nat → Nat → Ω → LoopRes
  | n, v, ω =>
    match test v ω with
    | .fuel v1 ω1 => .fuel v1 ω1
    | .cont false v1 ω1 => .done v1 ω1
    | .cont true v1 ω1 =>
      match n with
      | 0 => .bound v1 ω1
      | n + 1 =>
        match body v1 ω1 with
        | .fuel v2 ω2 => .fuel v2 ω2
        | .cont false v2 ω2 => .stopped v2 ω2
        | .cont true v2 ω2 => countLoop test body n v2 ω2

/-- The same shape with the body flag ignored, so a break falls into the
    next iteration. This is the late test named by the carried-flag note:
    a break is then able to meet the bound trap, and the next test runs. -/
def countLoopLate
    (test : Nat → Ω → StepRes)
    (body : Nat → Ω → StepRes) :
    Nat → Nat → Ω → LoopRes
  | n, v, ω =>
    match test v ω with
    | .fuel v1 ω1 => .fuel v1 ω1
    | .cont false v1 ω1 => .done v1 ω1
    | .cont true v1 ω1 =>
      match n with
      | 0 => .bound v1 ω1
      | n + 1 =>
        match body v1 ω1 with
        | .fuel v2 ω2 => .fuel v2 ω2
        | .cont _ v2 ω2 => countLoopLate test body n v2 ω2

theorem bound_before_body
    (test : Nat → Ω → StepRes)
    (body : Nat → Ω → StepRes)
    (v v1 : Nat) (ω ω1 : Ω)
    (ht : test v ω = .cont true v1 ω1) :
    countLoop test body 0 v ω = .bound v1 ω1 := by
  simp [countLoop, ht]

theorem done_skips_body
    (test : Nat → Ω → StepRes)
    (body : Nat → Ω → StepRes)
    (n v v1 : Nat) (ω ω1 : Ω)
    (ht : test v ω = .cont false v1 ω1) :
    countLoop test body n v ω = .done v1 ω1 := by
  simp [countLoop, ht]

theorem break_skips_next_iteration
    (test : Nat → Ω → StepRes)
    (body : Nat → Ω → StepRes)
    (n v v1 v2 : Nat) (ω ω1 ω2 : Ω)
    (ht : test v ω = .cont true v1 ω1)
    (hb : body v1 ω1 = .cont false v2 ω2) :
    countLoop test body (n + 1) v ω = .stopped v2 ω2 := by
  simp [countLoop, ht, hb]

theorem fuel_test_keeps_reported_state
    (test : Nat → Ω → StepRes)
    (body : Nat → Ω → StepRes)
    (n v v1 : Nat) (ω ω1 : Ω)
    (ht : test v ω = .fuel v1 ω1) :
    countLoop test body n v ω = .fuel v1 ω1 := by
  simp [countLoop, ht]

theorem fuel_body_keeps_reported_state
    (test : Nat → Ω → StepRes)
    (body : Nat → Ω → StepRes)
    (n v v1 v2 : Nat) (ω ω1 ω2 : Ω)
    (ht : test v ω = .cont true v1 ω1)
    (hb : body v1 ω1 = .fuel v2 ω2) :
    countLoop test body (n + 1) v ω = .fuel v2 ω2 := by
  simp [countLoop, ht, hb]

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
def effTest (v : Nat) (ω : Ω) : StepRes :=
  if ω.fuel = 0 then .fuel v ω
  else .cont true v ⟨ω.fuel - 1, ω.log ++ [0]⟩

/-- A body that breaks, records `1`, and does not spend fuel. -/
def brkBody (v : Nat) (ω : Ω) : StepRes :=
  .cont false (v + 1) ⟨ω.fuel, ω.log ++ [1]⟩

/-- A body that continues and records `1` without spending fuel. -/
def goBody (v : Nat) (ω : Ω) : StepRes :=
  .cont true v ⟨ω.fuel, ω.log ++ [1]⟩

/-- The body records `7` and then reports fuel. The loop result keeps that
    log. It does not restore the order value from after the test alone. -/
def bodyRecordsThenFuel (v : Nat) (ω : Ω) : StepRes :=
  .fuel (v + 1) ⟨ω.fuel, ω.log ++ [7]⟩

theorem body_fuel_keeps_partial_log :
    countLoop effTest bodyRecordsThenFuel 1 0 ⟨3, []⟩ =
      .fuel 1 ⟨2, [0, 7]⟩ := rfl

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

/-- A finished trace. `log` is the notes already run. `fuel` is what
    remains. `exit` is a break depth when the list stopped for a break. -/
structure Trace where
  log : List Nat
  fuel : Nat
  exit : Option Nat
  deriving DecidableEq, Repr

/-- `.fuel` keeps the trace at the trap: notes already run stay, and fuel
    already spent stays spent. -/
inductive RunEnd where
  | done (t : Trace)
  | fuel (t : Trace)
  deriving DecidableEq, Repr

def run (fuel : Nat) : List Act → RunEnd
  | [] => .done ⟨[], fuel, none⟩
  | .note n :: rest =>
    match run fuel rest with
    | .done ⟨log, f, e⟩ => .done ⟨n :: log, f, e⟩
    | .fuel ⟨log, f, e⟩ => .fuel ⟨n :: log, f, e⟩
  | .tick :: rest =>
    if fuel = 0 then .fuel ⟨[], fuel, none⟩
    else
      match run (fuel - 1) rest with
      | .done t => .done t
      | .fuel t => .fuel t
  | .break d :: _ => .done ⟨[], fuel, some d⟩

/-- One unit of fuel, a note, a tick that spends it, and a second tick.
    The note stays, and the fuel left is 0. -/
theorem fuel_keeps_prior_note :
    run 1 [.note 1, .tick, .tick] = .fuel ⟨[1], 0, none⟩ := rfl

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

/-- A frame the exit count can meet. `eval.rs` decrements `Break` on
    `Block` and `Loop` only. `If` runs one arm and does not decrement. -/
inductive Frame where
  | block
  | loop
  | ite
  deriving DecidableEq, Repr

/-- Structural depth counts enclosing blocks and loops. An `if` adds none.
    This is the source nesting, not the residual count after one exit. -/
def structuralDepth : List Frame → Nat
  | [] => 0
  | .ite :: fs => structuralDepth fs
  | .block :: fs => structuralDepth fs + 1
  | .loop :: fs => structuralDepth fs + 1

/-- Decrement once when the frame is a block or a loop. An `if` returns
    the same count. `none` means this frame consumed a 0 and stopped. -/
def stepExit : Frame → Nat → Option Nat
  | .ite, d => some d
  | .block, 0 => none
  | .loop, 0 => none
  | .block, d + 1 => some d
  | .loop, d + 1 => some d

theorem if_does_not_decrement (d : Nat) : stepExit .ite d = some d := rfl

theorem block_decrements_once (d : Nat) : stepExit .block (d + 1) = some d := rfl

theorem source_depth_is_not_the_residual :
    structuralDepth [.block, .ite, .loop] = 2 ∧
    stepExit .block 2 = some 1 ∧
    (1 ≠ 2) := by
  refine ⟨rfl, rfl, ?_⟩
  intro h
  cases h

inductive Leave where
  | stopHere (log : List Nat) (fuel : Nat)
  | propagate (residual : Nat) (log : List Nat) (fuel : Nat)
  | fell (log : List Nat) (fuel : Nat)
  | fuelHalt (log : List Nat) (fuel : Nat)
  deriving DecidableEq, Repr

/-- `break 0` stops at this construct. `break (d + 1)` leaves it with
    residual `d`. The residual is kept, so `break 1` and `break 2` do not
    become one result. -/
def runTwo (fuel : Nat) (inner : List Act) : Leave :=
  match run fuel inner with
  | .fuel ⟨log, f, _⟩ => .fuelHalt log f
  | .done ⟨log, f, none⟩ => .fell log f
  | .done ⟨log, f, some 0⟩ => .stopHere log f
  | .done ⟨log, f, some (d + 1)⟩ => .propagate d log f

/-- Run `tail` after `log`, keeping a fuel trap's partial trace. A `break`
    in `tail` is still pending: this function does not consume it. -/
def resume (fuel : Nat) (log : List Nat) (tail : List Act) : Leave :=
  match run fuel tail with
  | .done ⟨log2, f2, none⟩ => .fell (log ++ log2) f2
  | .done ⟨log2, f2, some 0⟩ => .stopHere (log ++ log2) f2
  | .done ⟨log2, f2, some (d + 1)⟩ => .propagate d (log ++ log2) f2
  | .fuel ⟨log2, f2, _⟩ => .fuelHalt (log ++ log2) f2

/-- A block, as `eval.rs` does for `Block`. Fallthrough is `Next`. `Break 0`
    is also `Next`, so the continuation after the block runs. `Break (d+1)`
    becomes residual `d` and that continuation does not run. Statements
    after a `break` in the same list are already absent from `run`. -/
def runEnclosing (fuel : Nat) (inner tail : List Act) : Leave :=
  match runTwo fuel inner with
  | .fell log f => resume f log tail
  | .stopHere log f => resume f log tail
  | .propagate d log f => .propagate d log f
  | .fuelHalt log f => .fuelHalt log f

/-- A loop, as `eval.rs` does for `Loop`. Fallthrough repeats the body, so
    the continuation after the loop does not run on that result. `Break 0`
    leaves the loop and that continuation runs. This function does not
    re-enter the body. -/
def runLoopFrame (fuel : Nat) (inner tail : List Act) : Leave :=
  match runTwo fuel inner with
  | .stopHere log f => resume f log tail
  | .fell log f => .fell log f
  | .propagate d log f => .propagate d log f
  | .fuelHalt log f => .fuelHalt log f

/-- An `if`, as `eval.rs` does for `If`: the arm's control is returned
    unchanged. `Break 0` is not consumed, so the continuation after the
    `if` does not run. -/
def runIte (fuel : Nat) (inner tail : List Act) : Leave :=
  match runTwo fuel inner with
  | .fell log f => resume f log tail
  | r => r

/-- The parent block around `runEnclosing`. Residual 0 and a pending
    `break 0` are each consumed once, and then `outer` runs. Residual
    `n + 1` is decremented and `outer` does not run. -/
def runParentBlock (fuel : Nat) (inner mid outer : List Act) : Leave :=
  match runEnclosing fuel inner mid with
  | .fell log f => resume f log outer
  | .stopHere log f => resume f log outer
  | .propagate 0 log f => resume f log outer
  | .propagate (n + 1) log f => .propagate n log f
  | .fuelHalt log f => .fuelHalt log f

theorem depth0_suppresses_later_effects :
    runTwo 3 [.note 1, .break 0, .note 9] = .stopHere [1] 3 := rfl

theorem depth1_keeps_residual_zero :
    runTwo 3 [.note 1, .break 1, .note 9] = .propagate 0 [1] 3 := rfl

theorem depth2_keeps_residual_one :
    runTwo 3 [.break 2] = .propagate 1 [] 3 := rfl

theorem break1_ne_break2 :
    runTwo 3 [.break 1] ≠ runTwo 3 [.break 2] := by
  intro h
  cases h

theorem enclosing_tail_not_run :
    runEnclosing 3 [.break 1] [.note 9] = .propagate 0 [] 3 ∧
    runEnclosing 3 [.note 1] [.note 9] = .fell [1, 9] 3 := by
  refine ⟨rfl, rfl⟩

theorem break0_runs_after_block :
    runEnclosing 3 [.break 0] [.note 9] = .fell [9] 3 := rfl

theorem break0_in_list_still_drops_later_note :
    runTwo 3 [.break 0, .note 9] = .stopHere [] 3 ∧
    runEnclosing 3 [.break 0, .note 9] [.note 8] = .fell [8] 3 := by
  refine ⟨rfl, rfl⟩

theorem break1_skips_mid_parent_runs_outer :
    runParentBlock 4 [.break 1] [.note 1] [.note 2] = .fell [2] 4 ∧
    runParentBlock 4 [.note 1] [.note 8] [.note 2] = .fell [1, 8, 2] 4 ∧
    runParentBlock 4 [.break 0] [.note 1] [.note 2] = .fell [1, 2] 4 := by
  refine ⟨rfl, rfl, rfl⟩

theorem residual0_consumed_once :
    runEnclosing 4 [.break 1] [.note 1] = .propagate 0 [] 4 ∧
    runParentBlock 4 [.break 1] [.note 1] [.note 2] = .fell [2] 4 ∧
    runParentBlock 4 [.break 2] [.note 1] [.note 2] = .propagate 0 [] 4 := by
  refine ⟨rfl, rfl, rfl⟩

theorem loop_fallthrough_is_not_the_exit :
    runLoopFrame 3 [.note 1] [.note 9] = .fell [1] 3 ∧
    runLoopFrame 3 [.break 0] [.note 9] = .fell [9] 3 ∧
    runIte 3 [.break 0] [.note 9] = .stopHere [] 3 := by
  refine ⟨rfl, rfl, rfl⟩

theorem prior_tick_kept_later_tick_not_run :
    runTwo 3 [.tick, .note 1, .break 0, .tick] = .stopHere [1] 2 := rfl

theorem depth0_ne_depth1 :
    runTwo 3 [.break 0] ≠ runTwo 3 [.break 1] := by
  intro h
  cases h

/-- ONE-15 writes `if e > 0` around the decrement. `eval.rs` does not
    decrement on `If`. This file follows `eval.rs` for `stepExit`. It does
    not prove that a printed `if e > 0 { set e = e − 1; break 0 }` is free
    of an extra block. That encoding is not identified with `stepExit`. -/
theorem one15_if_is_not_proved_here :
    stepExit .ite 1 = some 1 ∧ stepExit .block 1 = some 0 :=
  ⟨rfl, rfl⟩

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

/-- Left to right, the first occurrence of each name. A later copy is
    dropped. -/
def dedup (xs : List Name) : List Name :=
  xs.foldl (fun acc x => if mem x acc then acc else acc ++ [x]) []

theorem dedup_keeps_first :
    dedup [.v 1, .v 2, .v 1] = [.v 1, .v 2] ∧
    dedup [.v 1, .v 2, .v 1] ≠ [.v 2, .v 1] := by
  refine ⟨rfl, ?_⟩
  intro h
  cases h

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
