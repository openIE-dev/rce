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

/-- Control at a boundary.
    `.break d` is a raw pending exit depth: the next block or loop has not
    consumed it. `.next` is normal continuation, including a `Break 0` that
    a block or loop has already consumed. `.again` is a loop body that fell
    through; it is not an exit. `.ret` passes through every frame.
    `.fuel` is a failure and keeps the partial trace. A sequence and an
    `if` do not change a `.break` depth. -/
inductive Ctrl where
  | next (log : List Nat) (fuel : Nat)
  | break (depth : Nat) (log : List Nat) (fuel : Nat)
  | again (log : List Nat) (fuel : Nat)
  | ret (log : List Nat) (fuel : Nat)
  | fuel (log : List Nat) (fuel : Nat)
  deriving DecidableEq, Repr

/-- A statement list. The depth in `.break` is the depth the `break` wrote.
    This is not a frame, so the depth is not decremented. `run` does not
    produce `.ret` or `.again`: `Act` has no return. -/
def rawCtrl : RunEnd → Ctrl
  | .fuel ⟨log, f, _⟩ => .fuel log f
  | .done ⟨log, f, none⟩ => .next log f
  | .done ⟨log, f, some d⟩ => .break d log f

/-- Continuation of a sequence. A pending break, a return, a loop
    fallthrough, and a fuel trap do not run `tail`, and a break depth is
    not decremented. -/
def resumeCtrl : Ctrl → List Act → Ctrl
  | .next log f, tail =>
    match rawCtrl (run f tail) with
    | .next log2 f2 => .next (log ++ log2) f2
    | .break d log2 f2 => .break d (log ++ log2) f2
    | .fuel log2 f2 => .fuel (log ++ log2) f2
    | .again log2 f2 => .again (log ++ log2) f2
    | .ret log2 f2 => .ret (log ++ log2) f2
  | c, _ => c

/-- One frame boundary. A block or a loop decrements a pending break once.
    An `if` returns the control unchanged. A loop body that falls through
    becomes `.again`, not `.next`, so the continuation after the loop is
    not run by `resumeCtrl`. -/
def cross : Frame → Ctrl → Ctrl
  | .ite, c => c
  | .block, .break 0 log f => .next log f
  | .loop, .break 0 log f => .next log f
  | .block, .break (d + 1) log f => .break d log f
  | .loop, .break (d + 1) log f => .break d log f
  | .loop, .next log f => .again log f
  | _, c => c

/-- A block. `Break 0` becomes `Next`, so `tail` runs. `Break (d + 1)`
    becomes raw depth `d`, and `tail` does not run. -/
def runEnclosing (fuel : Nat) (inner tail : List Act) : Ctrl :=
  resumeCtrl (cross .block (rawCtrl (run fuel inner))) tail

/-- A loop. Body fallthrough stays inside (`.again`) and `tail` does not
    run. `Break 0` leaves the loop and `tail` runs. This does not re-enter
    the body. -/
def runLoopFrame (fuel : Nat) (inner tail : List Act) : Ctrl :=
  match cross .loop (rawCtrl (run fuel inner)) with
  | .next log f => resumeCtrl (.next log f) tail
  | c => c

/-- An `if`. The arm's break depth is unchanged, so `tail` runs only on
    fallthrough. -/
def runIte (fuel : Nat) (inner tail : List Act) : Ctrl :=
  resumeCtrl (cross .ite (rawCtrl (run fuel inner))) tail

/-- Inner block, then `mid` still inside the parent, then the parent
    boundary, then `outer`. Each block decrements once. `mid` is a
    sequence, so a break there keeps its raw depth until the parent
    boundary. -/
def runParentBlock (fuel : Nat) (inner mid outer : List Act) : Ctrl :=
  resumeCtrl (cross .block (runEnclosing fuel inner mid)) outer

theorem depth0_suppresses_later_effects :
    rawCtrl (run 3 [.note 1, .break 0, .note 9]) = .break 0 [1] 3 := rfl

theorem raw_break_is_not_decremented :
    rawCtrl (run 3 [.note 1, .break 1, .note 9]) = .break 1 [1] 3 ∧
    rawCtrl (run 3 [.break 2]) = .break 2 [] 3 := by
  refine ⟨rfl, rfl⟩

theorem break1_ne_break2 :
    rawCtrl (run 3 [.break 1]) ≠ rawCtrl (run 3 [.break 2]) := by
  intro h
  cases h

theorem enclosing_tail_not_run :
    runEnclosing 3 [.break 1] [.note 9] = .break 0 [] 3 ∧
    runEnclosing 3 [.note 1] [.note 9] = .next [1, 9] 3 := by
  refine ⟨rfl, rfl⟩

theorem break0_runs_after_block :
    runEnclosing 3 [.break 0] [.note 9] = .next [9] 3 := rfl

theorem break0_in_list_still_drops_later_note :
    rawCtrl (run 3 [.break 0, .note 9]) = .break 0 [] 3 ∧
    runEnclosing 3 [.break 0, .note 9] [.note 8] = .next [8] 3 := by
  refine ⟨rfl, rfl⟩

theorem break1_skips_mid_parent_runs_outer :
    runParentBlock 4 [.break 1] [.note 1] [.note 2] = .next [2] 4 ∧
    runParentBlock 4 [.note 1] [.note 8] [.note 2] = .next [1, 8, 2] 4 ∧
    runParentBlock 4 [.break 0] [.note 1] [.note 2] = .next [1, 2] 4 := by
  refine ⟨rfl, rfl, rfl⟩

/-- `break 1` in the parent, after the inner block, is still raw depth 1
    until the parent boundary. The parent emits raw depth 0 and does not
    run the outer note. -/
theorem mid_break_is_raw_until_parent :
    runParentBlock 3 [.note 1] [.break 1] [.note 2] = .break 0 [1] 3 ∧
    runParentBlock 3 [.note 1] [.break 2] [.note 2] = .break 1 [1] 3 := by
  refine ⟨rfl, rfl⟩

theorem residual0_consumed_once :
    runEnclosing 4 [.break 1] [.note 1] = .break 0 [] 4 ∧
    runParentBlock 4 [.break 1] [.note 1] [.note 2] = .next [2] 4 ∧
    runParentBlock 4 [.break 2] [.note 1] [.note 2] = .break 0 [] 4 := by
  refine ⟨rfl, rfl, rfl⟩

theorem if_keeps_raw_depth :
    runIte 3 [.break 0] [.note 9] = .break 0 [] 3 ∧
    runIte 3 [.break 1] [.note 9] = .break 1 [] 3 ∧
    runIte 3 [.break 2] [.note 9] = .break 2 [] 3 ∧
    runIte 3 [.note 1] [.note 9] = .next [1, 9] 3 := by
  refine ⟨rfl, rfl, rfl, rfl⟩

theorem if_inside_block_and_loop :
    let arm := rawCtrl (run 5 [.note 4, .break 2])
    let afterIf := cross .ite arm
    let afterBlock := cross .block afterIf
    let afterLoop := cross .loop afterBlock
    afterIf = .break 2 [4] 5 ∧
    afterBlock = .break 1 [4] 5 ∧
    afterLoop = .break 0 [4] 5 := by
  refine ⟨rfl, rfl, rfl⟩

theorem loop_fallthrough_is_not_the_exit :
    runLoopFrame 3 [.note 1] [.note 9] = .again [1] 3 ∧
    runLoopFrame 3 [.break 0] [.note 9] = .next [9] 3 ∧
    runLoopFrame 3 [.break 1] [.note 9] = .break 0 [] 3 := by
  refine ⟨rfl, rfl, rfl⟩

theorem block_in_loop :
    let inner0 := runEnclosing 5 [.break 0] [.note 7]
    let inner1 := runEnclosing 5 [.break 1] [.note 7]
    cross .loop inner0 = .again [7] 5 ∧
    cross .loop inner1 = .next [] 5 := by
  refine ⟨rfl, rfl⟩

theorem prior_tick_kept_later_tick_not_run :
    rawCtrl (run 3 [.tick, .note 1, .break 0, .tick]) = .break 0 [1] 2 := rfl

theorem depth0_ne_depth1 :
    rawCtrl (run 3 [.break 0]) ≠ rawCtrl (run 3 [.break 1]) := by
  intro h
  cases h

theorem cross_keeps_ret (fr : Frame) (log : List Nat) (f : Nat) :
    cross fr (.ret log f) = .ret log f := by
  cases fr <;> rfl

theorem cross_keeps_fuel (fr : Frame) (log : List Nat) (f : Nat) :
    cross fr (.fuel log f) = .fuel log f := by
  cases fr <;> rfl

theorem cross_agrees_with_stepExit (fr : Frame) (d : Nat) (log : List Nat) (f : Nat) :
    cross fr (.break d log f) =
      match stepExit fr d with
      | none => .next log f
      | some d' => .break d' log f := by
  cases fr <;> cases d <;> rfl

def nCons : List Frame → Nat
  | [] => 0
  | .ite :: fs => nCons fs
  | .block :: fs => nCons fs + 1
  | .loop :: fs => nCons fs + 1

/-- Innermost frame first. `none` means a block or loop consumed a raw 0
    and the later frames in the list are not applied. -/
def crossAll : List Frame → Nat → Option Nat
  | [], d => some d
  | .ite :: fs, d => crossAll fs d
  | .block :: _, 0 => none
  | .loop :: _, 0 => none
  | .block :: fs, d + 1 => crossAll fs d
  | .loop :: fs, d + 1 => crossAll fs d

theorem depth_conserved :
    ∀ (fs : List Frame) (d : Nat), nCons fs ≤ d →
      crossAll fs d = some (d - nCons fs)
  | [], _, _ => by simp [crossAll, nCons]
  | .ite :: fs, d, h => by
      simp [crossAll, nCons] at h ⊢
      exact depth_conserved fs d h
  | .block :: _, 0, h => by simp [nCons] at h
  | .loop :: _, 0, h => by simp [nCons] at h
  | .block :: fs, d + 1, h => by
      simp [crossAll, nCons] at h ⊢
      have h' : nCons fs ≤ d := by omega
      rw [depth_conserved fs d h']
  | .loop :: fs, d + 1, h => by
      simp [crossAll, nCons] at h ⊢
      have h' : nCons fs ≤ d := by omega
      rw [depth_conserved fs d h']

def crossFrames : List Frame → Ctrl → Ctrl
  | [], c => c
  | fr :: fs, c => crossFrames fs (cross fr c)

theorem crossFrames_pending :
    ∀ (fs : List Frame) (d : Nat) (log : List Nat) (f : Nat), nCons fs ≤ d →
      crossFrames fs (.break d log f) = .break (d - nCons fs) log f
  | [], _, _, _, _ => by simp [crossFrames, nCons]
  | .ite :: fs, d, log, f, h => by
      simp [crossFrames, cross, nCons] at h ⊢
      exact crossFrames_pending fs d log f h
  | .block :: _, 0, _, _, h => by simp [nCons] at h
  | .loop :: _, 0, _, _, h => by simp [nCons] at h
  | .block :: fs, d + 1, log, f, h => by
      simp [crossFrames, cross, nCons] at h ⊢
      have h' : nCons fs ≤ d := by omega
      rw [crossFrames_pending fs d log f h']
  | .loop :: fs, d + 1, log, f, h => by
      simp [crossFrames, cross, nCons] at h ⊢
      have h' : nCons fs ≤ d := by omega
      rw [crossFrames_pending fs d log f h']

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
