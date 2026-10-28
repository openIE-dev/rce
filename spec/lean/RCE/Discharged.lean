/-
Obligations of Appendix F that are proved here, each of a model written below, not of the
checker or the evaluator: the model says what the rule says and the theorem is about the model.
The statements that are not proved are in obligations/Obligations.lean (outside this build).
T48 termination by a written measure, T51 revocation, T53 world replay.
-/
namespace RCE.Discharged

/-! T48 (F.6.4). A configuration step that strictly lowers a natural-number measure
reaches the end of the run, whatever the start. -/
structure MeasureModel where
  Config : Type
  step : Config → Option Config
  measure : Config → Nat
  decreases : ∀ c c', step c = some c' → measure c' < measure c

/-- `n` steps from an optional configuration (`none` is the end of the run and stays). -/
def runN {C : Type} (step : C → Option C) : Nat → Option C → Option C
  | 0, o => o
  | n + 1, o => runN step n (o.bind step)

theorem runN_none {C : Type} (step : C → Option C) (n : Nat) : runN step n none = none := by
  induction n with
  | zero => rfl
  | succ n ih => simpa [runN] using ih

theorem runN_add {C : Type} (step : C → Option C) (n k : Nat) (o : Option C) :
    runN step (n + k) o = runN step k (runN step n o) := by
  induction n generalizing o with
  | zero => simp [runN]
  | succ n ih => rw [Nat.succ_add]; simp [runN, ih]

theorem t48 (m : MeasureModel) (c : m.Config) : ∃ n, runN m.step n (some c) = none := by
  suffices h : ∀ k (c : m.Config), m.measure c = k → ∃ n, runN m.step n (some c) = none from
    h _ c rfl
  intro k
  induction k using Nat.strongRecOn with
  | _ k ih =>
    intro c hk
    cases hs : m.step c with
    | none => exact ⟨1, by simp [runN, hs]⟩
    | some c' =>
      obtain ⟨n, hn⟩ := ih (m.measure c') (hk ▸ m.decreases c c' hs) c' rfl
      refine ⟨1 + n, ?_⟩
      rw [runN_add]
      simp [runN, hs, hn]

/-! T51 (F.5.1). After a revoke entry for a caretaker, no later use by it is admitted. -/
inductive Entry (C : Type) where
  | revoke (c : C)
  | use (c : C)

/-- A use by `c` is admitted by a log iff the log holds no revoke of `c`. -/
def admitted {C : Type} (log : List (Entry C)) (c : C) : Prop := Entry.revoke c ∉ log

theorem t51 {C : Type} (pre post : List (Entry C)) (c : C) :
    ¬ admitted (pre ++ Entry.revoke c :: post) c := by
  simp [admitted]

/-! T53 (F.4.1). A program that asks an oracle questions (foreign calls through a `World`)
replays from its log of answers: the replay returns what the live run returned. -/
inductive Prog (Q A R : Type) where
  | ret (r : R)
  | ask (q : Q) (k : A → Prog Q A R)

/-- Live run against an oracle: the answers asked for, in order, and the result. -/
def run {Q A R : Type} (o : Q → A) : Prog Q A R → List A × R
  | .ret r => ([], r)
  | .ask q k => ((o q :: (run o (k (o q))).1), (run o (k (o q))).2)

/-- Replay from a log: answers are read from the log, the oracle is not consulted;
a log of another length differs. -/
def replay {Q A R : Type} : Prog Q A R → List A → Option R
  | .ret r, [] => some r
  | .ask _ k, a :: l => replay (k a) l
  | _, _ => none

theorem t53 {Q A R : Type} (o : Q → A) (p : Prog Q A R) :
    replay p (run o p).1 = some (run o p).2 := by
  induction p with
  | ret r => rfl
  | ask q k ih => simpa [run, replay] using ih (o q)

end RCE.Discharged
