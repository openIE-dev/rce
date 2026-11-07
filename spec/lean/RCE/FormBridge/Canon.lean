import RCE.Form

/-!
  What this development does not decide, and the identity it does state.

  The predicates below are readings of sentences in `rce-form.md`. They
  are not `spec/checker/src/form/canon.rs`. On main `7e4e968c`,
  `c7_carried` removes a carried slot of an Iterate with two regions when
  both regions return that slot's input at output `1 + k` and neither uses
  it otherwise. Output 0 is the flag and is not that slot. The function has
  no clause that names ω. Pull request 153 is open and is not this main;
  its text says C7 and FG7 are unchanged. This file does not adopt an
  erratum and does not prove that the Rust function deletes ω.

  `CanonRel` drops a leading plain carrier in this file's term. It is not
  `canon.rs`.

  `RCE.Stmt` has no break. schedule ∘ construct is not the identity on a
  core term that still has a break.
-/

namespace RCE.FormBridge.Canon

inductive Carry where
  | omega
  | plain (n : Nat)
  deriving DecidableEq, Repr

/-- One arm of the two-region test in `c7_carried`, abstracted. `usesSlot`
    is a use other than returning the input. `returnsInputAt` is the output
    index of that return, when the arm returns the slot input. -/
structure ArmCarry where
  usesSlot : Bool
  returnsInputAt : Option Nat
  deriving DecidableEq, Repr

/-- Removal test shaped like `c7_carried`: exactly two regions, the output
    index is at least 1 (a carried slot, not the flag at output 0), and both
    arms return the slot input there and do not use it otherwise. This
    function does not rewrite a body and is not `canon.rs`. -/
def c7RemovesCarried (arms : List ArmCarry) (outputIndex : Nat) : Bool :=
  decide (arms.length = 2) && decide (outputIndex ≥ 1) &&
    arms.all fun a => !a.usesSlot && a.returnsInputAt == some outputIndex

theorem flag_output_is_not_removed :
    c7RemovesCarried
      [⟨false, some 0⟩, ⟨false, some 0⟩] 0 = false := rfl

theorem two_arms_unused_slot_removed :
    c7RemovesCarried
      [⟨false, some 1⟩, ⟨false, some 1⟩] 1 = true := rfl

theorem one_arm_is_not_removed :
    c7RemovesCarried [⟨false, some 1⟩] 1 = false := rfl

theorem a_use_keeps_the_slot :
    c7RemovesCarried
      [⟨true, some 1⟩, ⟨false, some 1⟩] 1 = false := rfl

/-- The proposal sentences, as two predicates on a record. Both hold of
    this record. That is not a proof that `canon.rs` and the gate
    contradict, and it does not delete or keep ω. -/
def proposalRemovesUnusedOmega (uses returnsUnchanged : Bool) : Bool :=
  (!uses) && returnsUnchanged

def proposalDemandsOmega (countBounded : Bool) : Bool :=
  countBounded

theorem proposal_sentences_both_hold :
    proposalRemovesUnusedOmega false true = true ∧
    proposalDemandsOmega true = true := by
  decide

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

/-- Drop a leading plain carrier in this term language. ω is not a clause.
    This relation is not `canon.rs`. -/
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
