import RCE.Core.Fuel
import RCE.ParFuel
import RCE.LVarAlloc
import RCE.Canon
import RCE.SessionRec
import RCE.SessionEnc
/-! S5 wiring: the concurrency results the core reuses. Each `#check`
    breaks the build if the lemma it names is renamed or removed, so the
    soundness plan cannot drift from the proofs it relies on. -/
namespace RCE.Core.Conc

-- par fuel: sum-checked trap point equals the sequential one; overrun bound
#check @RCE.ParFuel.sumcheck_eq_seq
#check @RCE.ParFuel.overrun_bound
-- LVar store: the executable step is exactly the relation
#check @RCE.LVarAlloc.stepAt_sound
#check @RCE.LVarAlloc.stepAt_complete
#check @RCE.LVarAlloc.exec_sound
#check @RCE.LVarAlloc.alloc_quasi_det
#check @RCE.LVarAlloc.alloc_det_no_err
-- canonical Set and Map values
#check @RCE.Canon.set_canon
#check @RCE.Canon.map_canon
-- sessions: duality, preservation, progress, encoding
#check @RCE.SessionRec.t28r
#check @RCE.SessionRec.t28r_preserve
#check @RCE.SessionRec.t28r_progress
#check @RCE.SessionRec.enc_forward
#check @RCE.SessionRec.enc_backward

/-- The hypothesis `ParFuel` states in prose ("a branch's use is fixed by
    the branch and its input") holds for core branches: a branch that
    completes on its copy of F gives the same result on any larger budget,
    in particular on the summed budget the reference charges. -/
theorem branch_result_stable (P : Prog) {F : Nat} {ρ e} (h : eval P F ρ e ≠ .oof) (k : Nat) :
    eval P (F + k) ρ e = eval P F ρ e := eval_mono_le P h

/-- Branch results agree across every pair of completing budgets, so the
    order in which `par` charges branches cannot change any result. -/
theorem branches_budget_indep (P : Prog) (bs : List (List Val × Expr)) (F G : Nat)
    (hF : ∀ b ∈ bs, eval P F b.1 b.2 ≠ .oof) (hG : ∀ b ∈ bs, eval P G b.1 b.2 ≠ .oof) :
    bs.map (fun b => eval P F b.1 b.2) = bs.map (fun b => eval P G b.1 b.2) :=
  List.map_congr_left (fun b m => eval_fuel_indep P (hF b m) (hG b m))

end RCE.Core.Conc
