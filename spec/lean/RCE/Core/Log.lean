import RCE.Core.Sound
import RCE.Replay
/-! S4 seed: the step function δ of a core program and the log. The
    handler is function 0 of the program; it takes `((state, event),
    answer)`, where the answer is the logged result of the event's effect
    (book 16 host contract), and returns `(state', output)`. A trap, an
    error or exhaustion gives the event the kept outcome and leaves the state (T8). The
    log machinery is `RCE.Replay`, reused, not restated. -/
namespace RCE.Core

def delta (P : Prog) (n : Nat) (s e a : Val) : Val :=
  match eval P n [] (.call 0 (.lit (.pair (.pair s e) a))) with
  | .ok (.pair s' _) => s'
  | _ => s

/-- **T8 Atomicity**: a kept event leaves the state unchanged. -/
theorem delta_atomic (P : Prog) (n : Nat) (s e a : Val)
    (h : ∀ s' o, eval P n [] (.call 0 (.lit (.pair (.pair s e) a))) ≠ .ok (.pair s' o)) :
    delta P n s e a = s := by
  unfold delta; split
  · rename_i s' o hx; exact absurd hx (h s' o)
  · rfl

/-- A handler signature: state, event, answer, output types. -/
def WtHandler (P : Prog) (St Ev An Out : Ty) : Prop :=
  (sig P).get? 0 = some (.prod (.prod St Ev) An, .prod St Out)

/-- **T5 for δ**: a well-typed handler maps a typed state to a typed state. -/
theorem delta_wt (P : Prog) (hP : WtProg P) {St Ev An Out} (hh : WtHandler P St Ev An Out)
    (n : Nat) {s e a} (hs : VTy s St) (he : VTy e Ev) (ha : VTy a An) :
    VTy (delta P n s e a) St := by
  have hr := eval_sound P hP n .nil
    (HasTy.call hh (.lit (.pair (.pair hs he) ha)))
  unfold delta; split
  · rename_i s' o hx; rw [hx] at hr; cases hr; assumption
  · exact hs

/-- **T3 Fold law**: replaying a log is the left fold of δ over it. -/
theorem replay_fold (P : Prog) (n : Nat) (s : Val) (D : List (Val × Val)) :
    Replay.replay (fun s e a => delta P n s e a) s D =
      D.foldl (fun s ea => delta P n s ea.1 ea.2) s := by
  induction D generalizing s with
  | nil => rfl
  | cons ea D ih => simp only [Replay.replay, List.foldl]; exact ih _

/-- **T32 for the core**: the live run with any effect oracle is the fold
    of δ over its own log (reuses `Replay.replay_live`). -/
theorem live_is_fold (P : Prog) (n : Nat) (o : Val → Val → Val) (s : Val) (es : List Val) :
    (Replay.live (fun s e a => delta P n s e a) o s es).1 =
      (Replay.live (fun s e a => delta P n s e a) o s es).2.foldl
        (fun s ea => delta P n s ea.1 ea.2) s := by
  rw [← replay_fold, Replay.replay_live]

/-- **Determinism of the log**: the same start state and log give the
    same state, whatever oracles produced the logs. -/
theorem log_det (P : Prog) (n : Nat) (o₁ o₂ : Val → Val → Val) (s : Val) (es₁ es₂ : List Val)
    (h : (Replay.live (fun s e a => delta P n s e a) o₁ s es₁).2 =
         (Replay.live (fun s e a => delta P n s e a) o₂ s es₂).2) :
    (Replay.live (fun s e a => delta P n s e a) o₁ s es₁).1 =
      (Replay.live (fun s e a => delta P n s e a) o₂ s es₂).1 :=
  Replay.t32 _ o₁ o₂ s es₁ es₂ h

/-- Preservation along the whole log. -/
theorem fold_wt (P : Prog) (hP : WtProg P) {St Ev An Out} (hh : WtHandler P St Ev An Out)
    (n : Nat) : ∀ (D : List (Val × Val)) (s : Val), VTy s St →
      (∀ ea ∈ D, VTy ea.1 Ev ∧ VTy ea.2 An) →
      VTy (D.foldl (fun s ea => delta P n s ea.1 ea.2) s) St
  | [], _, hs, _ => hs
  | ea :: D, s, hs, hD => by
    have h1 := hD ea (List.mem_cons_self _ _)
    exact fold_wt P hP hh n D _ (delta_wt P hP hh n hs h1.1 h1.2)
      (fun x m => hD x (List.mem_cons_of_mem _ m))

end RCE.Core
