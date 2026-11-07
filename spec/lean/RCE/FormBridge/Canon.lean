import RCE.Form

/-!
  What this development does not decide, and the identity it does state.

  C7 (`rce-form.md` §10) removes a carried value that neither region uses
  and both return unchanged. §6 says a count-bounded Iterate may halt, so
  it always takes ω. FG7 (§11) says every node that may halt consumes and
  returns ω. On a pure count those sentences disagree. The note
  `break-carried-flag.md` records the same disagreement as obligation O8
  and does not pick a side. Neither does this file.

  `RCE.Stmt` has no break. A normalization relation on the small terms
  below is not raw equality, and schedule ∘ construct is not the identity
  on a core term that still has a break. Neither relation is a proof about
  `spec/checker/src/form/canon.rs` or `bridge.rs`.
-/

namespace RCE.FormBridge.Canon

inductive Carry where
  | omega
  | plain (n : Nat)
  deriving DecidableEq, Repr

/-- Whether a region uses ω, and whether it returns that carrier unchanged.
    A plain carrier is outside this record. -/
structure RegionSpec where
  usesOmega : Bool
  returnsOmegaUnchanged : Bool
  deriving DecidableEq, Repr

/-- The C7 deletion test, read as written, applied to ω. A plain carrier
    is not removed by this function. -/
def c7RemovesOmega (test body : RegionSpec) : Bool :=
  (!test.usesOmega) && (!body.usesOmega) &&
    test.returnsOmegaUnchanged && body.returnsOmegaUnchanged

/-- The FG7 / §6 demand, read as written: a count-bounded Iterate takes ω.
    This function does not delete a carrier. -/
def fg7DemandsOmega (countBounded : Bool) : Bool :=
  countBounded

def pureCountTest : RegionSpec where
  usesOmega := false
  returnsOmegaUnchanged := true

def pureCountBody : RegionSpec where
  usesOmega := false
  returnsOmegaUnchanged := true

/-- Both sentences hold of one pure count. This theorem does not delete ω
    and does not keep it. -/
theorem c7_fg7_disagree :
    c7RemovesOmega pureCountTest pureCountBody = true ∧
    fg7DemandsOmega true = true := by
  decide

/-- A carried flag that is returned unchanged and unused meets the same
    deletion test. A break path that returns the literal false does not
    return the incoming flag unchanged, so the test does not remove it. -/
def c7RemovesFlag (usesFlag returnsUnchanged : Bool) : Bool :=
  (!usesFlag) && returnsUnchanged

theorem passthrough_flag_removed :
    c7RemovesFlag false true = true := rfl

theorem replaced_break_flag_not_removed :
    c7RemovesFlag false false = false := rfl

/-- `Stmt` in `RCE/Form.lean` cannot mention a break. The flag definitions
    in `Break.lean` are not a constructor of this inductive. -/
def mentionsBreak : RCE.Stmt → Bool
  | .skip => false
  | .assign _ _ => false
  | .seq a b => mentionsBreak a || mentionsBreak b
  | .ifs _ a b => mentionsBreak a || mentionsBreak b
  | .loop _ _ b => mentionsBreak b
  | .fail _ => false
  | .emit _ => false

theorem mentionsBreak_false (s : RCE.Stmt) : mentionsBreak s = false := by
  induction s with
  | skip => rfl
  | assign => rfl
  | seq _ _ iha ihb => simp [mentionsBreak, iha, ihb]
  | ifs _ _ _ iha ihb => simp [mentionsBreak, iha, ihb]
  | loop _ _ _ ih => simp [mentionsBreak, ih]
  | fail => rfl
  | emit => rfl

inductive CoreTerm where
  | sourceBreak
  | scheduledNoExit
  deriving DecidableEq, Repr

inductive FormTerm where
  | iter (carries : List Carry)
  deriving DecidableEq, Repr

/-- The schedule of an Iterate in this file writes the loop the schedule
    writes: no source break. The carrier list is forgotten here. Restoring
    it is not this function. -/
def schedule : FormTerm → CoreTerm
  | .iter _ => .scheduledNoExit

/-- The construction turns both a source break and the scheduled loop into
    an Iterate whose carrier list is only ω. That fixed list is not C7. -/
def construct : CoreTerm → FormTerm
  | .sourceBreak => .iter [.omega]
  | .scheduledNoExit => .iter [.omega]

theorem reverse_not_core_identity :
    schedule (construct .sourceBreak) = .scheduledNoExit ∧
    (.sourceBreak : CoreTerm) ≠ .scheduledNoExit := by
  refine ⟨rfl, ?_⟩
  intro h
  cases h

theorem construct_schedule_fixed (cs : List Carry) :
    construct (schedule (.iter cs)) = .iter [.omega] := rfl

theorem construct_schedule_not_raw :
    construct (schedule (.iter [.plain 0, .omega])) ≠
      .iter [.plain 0, .omega] := by
  intro h
  simp [construct, schedule] at h

/-- Drop a leading plain carrier. ω is not a clause of this relation.
    Dropping or keeping a pure-count ω would choose a side of
    `c7_fg7_disagree`, and this relation does not. -/
inductive CanonRel : FormTerm → FormTerm → Prop where
  | refl (t : FormTerm) : CanonRel t t
  | dropPlain (n : Nat) (rest : List Carry) :
      CanonRel (.iter (.plain n :: rest)) (.iter rest)

theorem canon_rel_is_not_raw_equality :
    CanonRel (.iter [.plain 0]) (.iter []) ∧
    (FormTerm.iter [.plain 0]) ≠ .iter [] := by
  refine ⟨.dropPlain 0 [], ?_⟩
  intro h
  cases h

theorem canon_rel_does_not_drop_omega :
    ¬ CanonRel (.iter [.omega]) (.iter []) := by
  intro h
  cases h

end RCE.FormBridge.Canon
