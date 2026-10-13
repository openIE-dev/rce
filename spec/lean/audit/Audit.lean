/-
Standing axiom audit (run by `make check` through scripts/lean-audit.sh).
Every theorem of the RCE library may use only the standard axioms
propext, Quot.sound and Classical.choice. `sorryAx` (a `sorry`) or any
added axiom fails the check.
-/
import Lean
import RCE
open Lean Elab Command

elab "#rce_audit" : command => do
  let env ← getEnv
  let allowed : List Name := [``propext, ``Quot.sound, ``Classical.choice]
  let mut n := 0
  let mut bad : Array String := #[]
  for (nm, ci) in env.constants.toList do
    if let .thmInfo _ := ci then
      if let some i := env.getModuleIdxFor? nm then
        if (env.header.moduleNames[i.toNat]!).getRoot == `RCE then
          n := n + 1
          let axs ← Lean.collectAxioms nm
          let extra := axs.filter (fun a => !allowed.contains a)
          if !extra.isEmpty then
            bad := bad.push s!"{nm} uses {extra.toList}"
  if bad.isEmpty then
    logInfo m!"lean audit: {n} RCE theorems, standard axioms only"
  else
    for b in bad do logError m!"{b}"
    throwError m!"lean audit: {bad.size} theorem(s) use sorry or non-standard axioms"
  if n < 100 then throwError m!"lean audit: only {n} RCE theorems found"

#rce_audit
