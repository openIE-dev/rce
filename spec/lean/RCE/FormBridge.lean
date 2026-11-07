import RCE.FormBridge.Catch
import RCE.FormBridge.Break
import RCE.FormBridge.Canon

/-!
  Checked lemmas for the Form bridge catch adapter and for break as a
  carried flag. The definitions live under `RCE/FormBridge/` so a change
  to `spec/checker/src/form/bridge.rs` does not edit them.

  Nothing here is a theorem about that Rust file. On the commit this
  library was added, `bridge.rs` refuses a catch with the reason
  `catch: a catch and the fuel-trap branch are not in this slice`, and a
  loop-body break is the flag of option B in
  `spec/proposals/break-carried-flag.md`. The lemmas state the value
  relations those texts describe. They do not prove the Rust implements
  them.

  Unproved obligations, left open on purpose:

  1. `Catch.coreCatch` matches the value cases of `P::Catch` in `eval.rs`
     only as a definition in this file. It is not proved equal to that
     Rust. `Catch.adapt` is the schedule rethrow, and `schedule_rethrow_is_not_the_core_catch`
     separates the two. Neither is proved equal to `bridge.rs`, which still
     refuses catch.
  2. Three-way eval is not a statement of this library.
  3. `construct ∘ schedule` as canonical bytes is not proved.
     `Canon.construct_schedule_fixed` returns an Iterate whose only carrier
     is ω. `Canon.construct_schedule_not_raw` says that is not raw equality
     for a longer carrier list. `Canon.reverse_not_core_identity` says
     schedule ∘ construct is not the identity on `CoreTerm.sourceBreak`.
  4. `Canon.proposal_sentences_both_hold` says two predicates read from the
     proposal both hold. It is not a proof about `canon.rs` or the gate.
     `Canon.c7RemovesCarried` is the two-region slot test, and it is not
     `c7_carried`. Pull request 153 is not on this main.
  5. `Break.armInputs` puts ω last and hands both arms one list, in
     first-seen order. It is not C7's order by Merkle digest, and it is
     not a bridge scope environment.
  6. The further Branch that a lift of RCE writes, which halts again on a
     trap text other than the budget trap, is not `Catch.adapt`.
  7. `Break.cross` decrements a raw `.break` once on a block or a loop and
     does not decrement on an `if`. `resumeCtrl` does not decrement.
     `one15_if_is_not_proved_here` records that the printed ONE-15 `if` is
     not identified with `stepExit`. These drivers are one list plus
     explicit frames, not the recursive `eval.rs` walker, and a loop is
     not re-entered. `Act` has no return, so `run` does not produce
     `.ret`; `cross` preserves a `.ret` it is given. A break that does not
     name the loop still requires change in the bridge.
  8. Fuel in these files is a `Nat`. It is not a `fuel.tick` node and not
     the reference limit. `run` and `countLoop` keep the trace they are
     given. They are not the reference fuel accounting.
  9. `Canon.CanonRel` drops one leading plain carrier. It is not proved to
     preserve `Break.countLoop`, and it is not `form/canon.rs`.
  10. `Canon.mentionsBreak_false` says `RCE.Stmt` has no break. No
      constructor was added, and `ExecR` is unchanged.
  11. `Catch.dispatch` applies the handler function it is given. It does
      not prove a handler written in the checker.
  12. `coreSliceAdmits` states the arity `sig_of` refuses. It is not proved
      equal to that function. Zero outputs and several outputs have no
      core sum in this file.
-/
