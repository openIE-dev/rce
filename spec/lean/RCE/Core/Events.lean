import RCE.Core.Driver
/-! S4 executable: event runs over the core (book 6.4 δ, interp.rs
    `region_step`/`delta`). A region's state is its field values in
    environment order (the last field is de Bruijn index 0). A step is the
    event it fires on, an optional guard and a statement body run with the
    fields as the environment. δ runs every region on the pre-state; the
    first refusal (trap, declared error, exhaustion) refuses the event
    atomically: the state is unchanged (T8, `delta_atomic` in Log.lean is
    the expression-level statement). -/
namespace RCE.Core

/-- A step: event tag, guard, body. -/
abbrev StepD := Nat × Option Expr × List Stmt

inductive StepR where
  | skip
  | ok (fs : List Val) (outs : List String)
  | kept (t : Trap)
  | oof
  | stuck

/-- The first step of a region that fires on `ev` (its guard holds). The
    event's payload `pv` (environment order) is bound above the fields. -/
def regionStep (P : Prog) (fuel : Nat) (fs : List Val) (ev : Nat) (pv : List Val) : List StepD → StepR
  | [] => .skip
  | (tag, g, body) :: rest =>
    if tag ≠ ev then regionStep P fuel fs ev pv rest else
    let fire : StepR :=
      let r := execS P (2 * fuel + 2) fuel (pv ++ fs) body
      match r.2.1 with
      | .norm | .brk | .cont => .ok (leave fs.length r.2.2.1) r.1
      | .trap t => .kept t
      | .oof => .oof
      | .stuck => .stuck
    match g with
    | none => fire
    | some g => match (runC P fuel (pv ++ fs) g).1 with
      | .ok (.bool true) => fire
      | .ok (.bool false) => regionStep P fuel fs ev pv rest
      | .ok _ => .stuck
      | .err t => .kept t
      | .oof => .oof
      | .stuck => .stuck

/-- How δ ended on one event. -/
inductive DeltaR where
  | ok (s : List (List Val)) (outs : List String)
  | kept (t : Trap)
  | oof
  | stuck

/-- δ over all regions on the pre-state. -/
def deltaS (P : Prog) (fuel : Nat) (ev : Nat) (pv : List Val) :
    List (List Val × List StepD) → DeltaR
  | [] => .ok [] []
  | (fs, steps) :: rs =>
    match regionStep P fuel fs ev pv steps with
    | .kept t => .kept t
    | .oof => .oof
    | .stuck => .stuck
    | r =>
      let (fs', o) := match r with | .ok fs' o => (fs', o) | _ => (fs, [])
      match deltaS P fuel ev pv rs with
      | .ok s o2 => .ok (fs' :: s) (o ++ o2)
      | e => e

/-- **T8 for δ over regions**: a refused event is refused as a whole; no
    region's new state is returned. -/
theorem deltaS_kept_atomic (P : Prog) (fuel ev : Nat) (pv : List Val) (rs : List (List Val × List StepD))
    (t : Trap) (h : deltaS P fuel ev pv rs = .kept t) : ∀ s o, deltaS P fuel ev pv rs ≠ .ok s o := by
  intro s o h2; rw [h] at h2; cases h2

end RCE.Core
