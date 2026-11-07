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

  1. `Catch.adapt` and `Catch.dispatch` are not proved equal to any
     function in `bridge.rs`. The fuel-trap Branch and the Form `catch`
     node are not a proved pair of inverses on core syntax.
  2. Three-way eval (the core evaluator, the Form evaluator, and the core
     evaluator on the schedule) is not a statement of this library.
  3. `construct ∘ schedule` as canonical bytes is not proved.
     `Canon.construct_schedule_fixed` returns an Iterate whose only carrier
     is ω. `Canon.construct_schedule_not_raw` says that is not raw equality
     for a longer carrier list. `Canon.reverse_not_core_identity` says
     schedule ∘ construct is not the identity on `CoreTerm.sourceBreak`.
  4. C7 and FG7 disagree on a pure count (`Canon.c7_fg7_disagree`). No
     definition here deletes that ω, and none keeps it. An erratum that
     would exempt a required order carrier is not applied.
  5. `Break.armInputs` puts ω last and hands both arms one list. It is not
     C7's order by Merkle digest, and it is not a bridge scope environment.
  6. The further Branch that a lift of RCE writes, which halts again on a
     trap text other than the budget trap, is not `Catch.adapt`. `adapt`
     rethrows only the fuel trap text `comptime fuel exhausted`.
  7. `Break.propagate` is the exit count of ONE-15 on `Break.Act`. It is
     not a proof that the bridge carries depths. A break that does not
     name the loop still requires change in the bridge.
  8. Fuel in these files is a `Nat`. It is not a `fuel.tick` node and not
     the reference limit.
  9. `Canon.CanonRel` drops one leading plain carrier. It is not proved to
     preserve `Break.countLoop`, and it is not `form/canon.rs`.
  10. `Canon.mentionsBreak_false` says `RCE.Stmt` has no break. No
      constructor was added, and `ExecR` is unchanged.
  11. `Catch.dispatch` applies the handler function it is given. It does
      not prove a handler written in the checker.
-/
