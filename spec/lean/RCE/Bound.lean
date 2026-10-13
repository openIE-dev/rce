import RCE.Form
/-! T2 with an explicit step bound, and T4 (progress) for statements and δ.
    Expressions reduce by the small-step relation `Step` of Expr.lean;
    statements by a small-step relation whose steps that evaluate an
    expression carry the exact number of expression steps taken.  We prove
    that from any statement configuration the machine reaches the result of
    the reference semantics `exec` in at most `B s` steps, where `B` is
    computed from the program text (loop bounds and expression sizes); the
    same for a region's step selection and for δ, with
    `B(P) = Σ (size guard + 1 + B body)` per region. -/
namespace RCE

def Expr.size : Expr → Nat
  | .lit _ => 1
  | .trap _ => 1
  | .field _ => 1
  | .arg _ => 1
  | .bin _ _ a b => a.size + b.size + 1
  | .ite c a b => c.size + a.size + b.size + 1

/-- n expression steps. -/
inductive NSteps (E : Env) : Nat → Expr → Expr → Prop where
  | refl {e} : NSteps E 0 e e
  | step {n e e' e''} : Step E e e' → NSteps E n e' e'' → NSteps E (n + 1) e e''

theorem NSteps.trans {E n m a b c} (h1 : NSteps E n a b) (h2 : NSteps E m b c) : NSteps E (n + m) a c := by
  induction h1 with
  | refl => simpa using h2
  | step s _ ih => have := NSteps.step s (ih h2); rwa [Nat.add_right_comm] at this

theorem NSteps.binL {E n op ta a a' b} (h : NSteps E n a a') : NSteps E n (.bin op ta a b) (.bin op ta a' b) := by
  induction h with
  | refl => exact .refl
  | step s _ ih => exact .step (.binL s) ih
theorem NSteps.binR {E n op ta v b b'} (h : NSteps E n b b') :
    NSteps E n (.bin op ta (.lit v) b) (.bin op ta (.lit v) b') := by
  induction h with
  | refl => exact .refl
  | step s _ ih => exact .step (.binR s) ih
theorem NSteps.iteC {E n c c' a b} (h : NSteps E n c c') : NSteps E n (.ite c a b) (.ite c' a b) := by
  induction h with
  | refl => exact .refl
  | step s _ ih => exact .step (.iteC s) ih

theorem NSteps.one {E e e'} (h : Step E e e') : NSteps E 1 e e' := .step h .refl

/-- Expressions reach their value or trap in at most `size e` steps. -/
theorem eval_bound (E : Env) (e : Expr) : ∃ n, n ≤ e.size ∧ NSteps E n e (Res.toExpr (eval E e)) := by
  induction e with
  | lit v => exact ⟨0, by simp [Expr.size], .refl⟩
  | trap m => exact ⟨0, by simp [Expr.size], .refl⟩
  | field i =>
    simp only [eval]; cases h : E.fields.get? i with
    | none => exact ⟨1, by simp [Expr.size], .one (.fieldBad h)⟩
    | some v => exact ⟨1, by simp [Expr.size], .one (.field h)⟩
  | arg i =>
    simp only [eval]; cases h : E.args.get? i with
    | none => exact ⟨1, by simp [Expr.size], .one (.argBad h)⟩
    | some v => exact ⟨1, by simp [Expr.size], .one (.arg h)⟩
  | bin op ta a b iha ihb =>
    obtain ⟨na, hna, sa⟩ := iha; obtain ⟨nb, hnb, sb⟩ := ihb
    simp only [eval, bind, Except.bind]
    cases ha : eval E a with
    | error m =>
      rw [ha] at sa; dsimp only
      exact ⟨na + 1, by simp [Expr.size]; omega, (NSteps.binL sa).trans (.one .binTrapL)⟩
    | ok x =>
      rw [ha] at sa
      cases hb : eval E b with
      | error m =>
        rw [hb] at sb; dsimp only
        exact ⟨na + nb + 1, by simp [Expr.size]; omega, ((NSteps.binL sa).trans (NSteps.binR sb)).trans (.one .binTrapR)⟩
      | ok y =>
        rw [hb] at sb; dsimp only
        cases hr : binop op ta x y with
        | ok r => exact ⟨na + nb + 1, by simp [Expr.size]; omega, ((NSteps.binL sa).trans (NSteps.binR sb)).trans (.one (.binOk hr))⟩
        | error m => exact ⟨na + nb + 1, by simp [Expr.size]; omega, ((NSteps.binL sa).trans (NSteps.binR sb)).trans (.one (.binErr hr))⟩
  | ite c a b ihc iha ihb =>
    obtain ⟨nc, hnc, sc⟩ := ihc; obtain ⟨na, hna, sa⟩ := iha; obtain ⟨nb, hnb, sb⟩ := ihb
    simp only [eval]
    cases hc : eval E c with
    | error m => rw [hc] at sc; exact ⟨nc + 1, by simp [Expr.size]; omega, (NSteps.iteC sc).trans (.one .iteTrap)⟩
    | ok v =>
      rw [hc] at sc
      cases v with
      | int k => exact ⟨nc + 1, by simp [Expr.size]; omega, (NSteps.iteC sc).trans (.one .iteBad)⟩
      | bool bb =>
        cases bb
        · exact ⟨nc + (nb + 1), by simp [Expr.size]; omega, (NSteps.iteC sc).trans (.step .iteF sb)⟩
        · exact ⟨nc + (na + 1), by simp [Expr.size]; omega, (NSteps.iteC sc).trans (.step .iteT sa)⟩

/-- An expression evaluated in `n` small steps. -/
def EvalN (E : Env) (n : Nat) (e : Expr) (r : Except Trap Val) : Prop := NSteps E n e (Res.toExpr r)

theorem evalN_exists (E : Env) (e : Expr) : ∃ n, n ≤ e.size ∧ EvalN E n e (eval E e) := eval_bound E e

/-! ### Statements -/

inductive SCfg where
  | run (s : Stmt) (fs os : List Val)
  | done (fs os : List Val)
  | refused (r : Refusal)

def SRes.toCfg : SRes → SCfg
  | .ok fs os => .done fs os
  | .ref r => .refused r

/-- Small-step semantics of statements; `SStep args c k c'` takes cost k
    (the expression steps it performs, plus one). -/
inductive SStep (args : List Val) : SCfg → Nat → SCfg → Prop where
  | skip {fs os} : SStep args (.run .skip fs os) 1 (.done fs os)
  | assign {i e fs os n v} : EvalN ⟨fs, args⟩ n e (.ok v) → i < fs.length →
      SStep args (.run (.assign i e) fs os) (n + 1) (.done (fs.set i v) os)
  | assignBad {i e fs os n v} : EvalN ⟨fs, args⟩ n e (.ok v) → ¬ i < fs.length →
      SStep args (.run (.assign i e) fs os) (n + 1) (.refused (.trap .badIndex))
  | assignTrap {i e fs os n m} : EvalN ⟨fs, args⟩ n e (.error m) →
      SStep args (.run (.assign i e) fs os) (n + 1) (.refused (.trap m))
  | seqL {a a' b fs os fs' os' k} : SStep args (.run a fs os) k (.run a' fs' os') →
      SStep args (.run (.seq a b) fs os) k (.run (.seq a' b) fs' os')
  | seqDone {a b fs os fs' os' k} : SStep args (.run a fs os) k (.done fs' os') →
      SStep args (.run (.seq a b) fs os) k (.run b fs' os')
  | seqRef {a b fs os r k} : SStep args (.run a fs os) k (.refused r) →
      SStep args (.run (.seq a b) fs os) k (.refused r)
  | ifT {c a b fs os n} : EvalN ⟨fs, args⟩ n c (.ok (.bool true)) → SStep args (.run (.ifs c a b) fs os) (n + 1) (.run a fs os)
  | ifF {c a b fs os n} : EvalN ⟨fs, args⟩ n c (.ok (.bool false)) → SStep args (.run (.ifs c a b) fs os) (n + 1) (.run b fs os)
  | ifBad {c a b fs os n k} : EvalN ⟨fs, args⟩ n c (.ok (.int k)) → SStep args (.run (.ifs c a b) fs os) (n + 1) (.refused (.trap .badIndex))
  | ifTrap {c a b fs os n m} : EvalN ⟨fs, args⟩ n c (.error m) → SStep args (.run (.ifs c a b) fs os) (n + 1) (.refused (.trap m))
  | loopStop {k c b fs os n} : EvalN ⟨fs, args⟩ n c (.ok (.bool false)) → SStep args (.run (.loop k c b) fs os) (n + 1) (.done fs os)
  | loopBound {c b fs os n} : EvalN ⟨fs, args⟩ n c (.ok (.bool true)) → SStep args (.run (.loop 0 c b) fs os) (n + 1) (.refused (.trap .loopBound))
  | loopGo {k c b fs os n} : EvalN ⟨fs, args⟩ n c (.ok (.bool true)) →
      SStep args (.run (.loop (k + 1) c b) fs os) (n + 1) (.run (.seq b (.loop k c b)) fs os)
  | loopBad {k c b fs os n j} : EvalN ⟨fs, args⟩ n c (.ok (.int j)) → SStep args (.run (.loop k c b) fs os) (n + 1) (.refused (.trap .badIndex))
  | loopTrap {k c b fs os n m} : EvalN ⟨fs, args⟩ n c (.error m) → SStep args (.run (.loop k c b) fs os) (n + 1) (.refused (.trap m))
  | fail {x fs os} : SStep args (.run (.fail x) fs os) 1 (.refused (.user x))
  | emit {e fs os n v} : EvalN ⟨fs, args⟩ n e (.ok v) → SStep args (.run (.emit e) fs os) (n + 1) (.done fs (os ++ [v]))
  | emitTrap {e fs os n m} : EvalN ⟨fs, args⟩ n e (.error m) → SStep args (.run (.emit e) fs os) (n + 1) (.refused (.trap m))

/-- A run of total cost k. -/
inductive SRun (args : List Val) : Nat → SCfg → SCfg → Prop where
  | refl {c} : SRun args 0 c c
  | step {c c' c'' k n} : SStep args c k c' → SRun args n c' c'' → SRun args (k + n) c c''

theorem SRun.trans {args n m a b c} (h1 : SRun args n a b) (h2 : SRun args m b c) : SRun args (n + m) a c := by
  induction h1 with
  | refl => simpa using h2
  | step s _ ih => have := SRun.step s (ih h2); rwa [Nat.add_assoc]

theorem done_nostep {args fs os k c} : ¬ SStep args (.done fs os) k c := by intro h; cases h
theorem refused_nostep {args r k c} : ¬ SStep args (.refused r) k c := by intro h; cases h

/-- Running `a` to completion inside `seq a b` continues with `b`. -/
theorem SRun.seqDone {args n a b fs os fs' os'} (h : SRun args n (.run a fs os) (.done fs' os')) :
    SRun args n (.run (.seq a b) fs os) (.run b fs' os') := by
  generalize hx : SCfg.run a fs os = x at h
  generalize hy : SCfg.done fs' os' = y at h
  induction h generalizing a fs os with
  | refl => subst hx; cases hy
  | @step c c' c'' k n s r ih =>
    subst hx
    cases c' with
    | run a' fs1 os1 => exact .step (.seqL s) (ih rfl hy)
    | done fs1 os1 =>
      cases r with
      | refl => cases hy; simpa using SRun.step (.seqDone s) .refl
      | step s2 _ => exact absurd s2 done_nostep
    | refused r0 =>
      cases r with
      | refl => cases hy
      | step s2 _ => exact absurd s2 refused_nostep

theorem SRun.seqRef {args n a b fs os r} (h : SRun args n (.run a fs os) (.refused r)) :
    SRun args n (.run (.seq a b) fs os) (.refused r) := by
  generalize hx : SCfg.run a fs os = x at h
  generalize hy : SCfg.refused r = y at h
  induction h generalizing a fs os with
  | refl => subst hx; cases hy
  | @step c c' c'' k n s rr ih =>
    subst hx
    cases c' with
    | run a' fs1 os1 => exact .step (.seqL s) (ih rfl hy)
    | done fs1 os1 =>
      cases rr with
      | refl => cases hy
      | step s2 _ => exact absurd s2 done_nostep
    | refused r0 =>
      cases rr with
      | refl => cases hy; simpa using SRun.step (.seqRef s) .refl
      | step s2 _ => exact absurd s2 refused_nostep

/-- The explicit step bound of a statement (spec 6.7, T2). -/
def B : Stmt → Nat
  | .skip => 1
  | .assign _ e => e.size + 1
  | .seq a b => B a + B b
  | .ifs c a b => c.size + 1 + max (B a) (B b)
  | .loop n c b => (n + 1) * (c.size + 1) + n * B b
  | .fail _ => 1
  | .emit e => e.size + 1

theorem iter_eq_exec_loop (args : List Val) (n : Nat) (c : Expr) (b : Stmt) (fs os : List Val) :
    exec args (.loop n c b) fs os = iter (exec args b) (fun fs => eval ⟨fs, args⟩ c) n fs os := rfl

/-- **T2 (explicit bound) and adequacy for statements**: from any statement
    configuration the small-step machine reaches exactly the result of `exec`
    with total cost at most `B s`. -/
theorem stmt_bound (args : List Val) (s : Stmt) : ∀ fs os, ∃ k, k ≤ B s ∧ SRun args k (.run s fs os) (exec args s fs os).toCfg := by
  induction s with
  | skip => intro fs os; exact ⟨1, by simp [B], by simpa using SRun.step .skip .refl⟩
  | assign i e =>
    intro fs os
    obtain ⟨n, hn, he⟩ := evalN_exists ⟨fs, args⟩ e
    simp only [exec]
    revert he
    cases eval ⟨fs, args⟩ e with
    | ok v =>
      intro he
      by_cases hi : i < fs.length
      · simp only [hi, ite_true, SRes.toCfg]; exact ⟨n + 1, by simp [B]; omega, by simpa using SRun.step (.assign he hi) .refl⟩
      · simp only [hi, ite_false, SRes.toCfg]; exact ⟨n + 1, by simp [B]; omega, by simpa using SRun.step (.assignBad he hi) .refl⟩
    | error m => intro he; exact ⟨n + 1, by simp [B]; omega, by simpa using SRun.step (.assignTrap he) .refl⟩
  | seq a b iha ihb =>
    intro fs os
    obtain ⟨ka, hka, ra⟩ := iha fs os
    simp only [exec]
    revert ra
    cases exec args a fs os with
    | ok fs' os' =>
      intro ra
      obtain ⟨kb, hkb, rb⟩ := ihb fs' os'
      exact ⟨ka + kb, by simp [B]; omega, (SRun.seqDone ra).trans rb⟩
    | ref x => intro ra; exact ⟨ka, by simp [B]; omega, SRun.seqRef ra⟩
  | ifs c a b iha ihb =>
    intro fs os
    obtain ⟨n, hn, hc⟩ := evalN_exists ⟨fs, args⟩ c
    simp only [exec]
    revert hc
    cases eval ⟨fs, args⟩ c with
    | error m => intro hc; exact ⟨n + 1, by simp [B]; omega, by simpa using SRun.step (.ifTrap hc) .refl⟩
    | ok v =>
      cases v with
      | int j => intro hc; exact ⟨n + 1, by simp [B]; omega, by simpa using SRun.step (.ifBad hc) .refl⟩
      | bool bb =>
        cases bb
        · intro hc; obtain ⟨k, hk, r⟩ := ihb fs os
          exact ⟨n + 1 + k, by simp [B]; omega, .step (.ifF hc) r⟩
        · intro hc; obtain ⟨k, hk, r⟩ := iha fs os
          exact ⟨n + 1 + k, by simp [B]; omega, .step (.ifT hc) r⟩
  | loop n c b ihb =>
    induction n with
    | zero =>
      intro fs os
      obtain ⟨m, hm, hc⟩ := evalN_exists ⟨fs, args⟩ c
      rw [iter_eq_exec_loop]; simp only [iter]
      revert hc
      cases eval ⟨fs, args⟩ c with
      | error e => intro hc; exact ⟨m + 1, by simp [B]; omega, by simpa using SRun.step (.loopTrap hc) .refl⟩
      | ok v =>
        cases v with
        | int j => intro hc; exact ⟨m + 1, by simp [B]; omega, by simpa using SRun.step (.loopBad hc) .refl⟩
        | bool bb =>
          cases bb
          · intro hc; exact ⟨m + 1, by simp [B]; omega, by simpa using SRun.step (.loopStop hc) .refl⟩
          · intro hc; exact ⟨m + 1, by simp [B]; omega, by simpa using SRun.step (.loopBound hc) .refl⟩
    | succ k ihk =>
      intro fs os
      obtain ⟨m, hm, hc⟩ := evalN_exists ⟨fs, args⟩ c
      rw [iter_eq_exec_loop]; simp only [iter]
      revert hc
      cases eval ⟨fs, args⟩ c with
      | error e => intro hc; exact ⟨m + 1, by simp [B, Nat.succ_mul]; omega, by simpa using SRun.step (.loopTrap hc) .refl⟩
      | ok v =>
        cases v with
        | int j => intro hc; exact ⟨m + 1, by simp [B, Nat.succ_mul]; omega, by simpa using SRun.step (.loopBad hc) .refl⟩
        | bool bb =>
          cases bb
          · intro hc; exact ⟨m + 1, by simp [B, Nat.succ_mul]; omega, by simpa using SRun.step (.loopStop hc) .refl⟩
          · intro hc
            obtain ⟨kb, hkb, rb⟩ := ihb fs os
            simp only
            revert rb
            cases exec args b fs os with
            | ref x =>
              intro rb
              exact ⟨m + 1 + kb, by simp [B, Nat.succ_mul, Nat.add_mul] at hkb ⊢; omega, .step (.loopGo hc) (SRun.seqRef rb)⟩
            | ok fs' os' =>
              intro rb
              obtain ⟨kl, hkl, rl⟩ := ihk fs' os'
              rw [iter_eq_exec_loop] at rl
              exact ⟨m + 1 + (kb + kl), by simp [B, Nat.succ_mul, Nat.add_mul] at hkl ⊢; omega, .step (.loopGo hc) ((SRun.seqDone rb).trans rl)⟩
  | fail x => intro fs os; exact ⟨1, by simp [B], by simpa using SRun.step .fail .refl⟩
  | emit e =>
    intro fs os
    obtain ⟨n, hn, he⟩ := evalN_exists ⟨fs, args⟩ e
    simp only [exec]
    revert he
    cases eval ⟨fs, args⟩ e with
    | ok v => intro he; exact ⟨n + 1, by simp [B]; omega, by simpa using SRun.step (.emit he) .refl⟩
    | error m => intro he; exact ⟨n + 1, by simp [B]; omega, by simpa using SRun.step (.emitTrap he) .refl⟩

/-- **T4 for statements**: every running configuration takes a step; the
    only configurations that do not are results (`done`) and refusals. -/
theorem stmt_progress (args : List Val) (s : Stmt) (fs os : List Val) :
    ∃ k c', SStep args (.run s fs os) k c' := by
  obtain ⟨k, _, h⟩ := stmt_bound args s fs os
  generalize hy : (exec args s fs os).toCfg = y at h
  have hne : y ≠ .run s fs os := by
    subst hy; cases exec args s fs os <;> simp [SRes.toCfg]
  generalize hx : SCfg.run s fs os = x at h
  cases h with
  | refl => exact absurd hx.symm (by rw [← hx] at hne; exact fun h => hne (h ▸ rfl))
  | step st _ => subst hx; exact ⟨_, _, st⟩

/-! ### Region step selection and δ -/

/-- Selecting and running the step of region i for an event, with cost. -/
inductive RRun (args : List Val) (ev : String) (i : Nat) (fs : List Val) : List StepDecl → Nat → SRes → Prop where
  | nil : RRun args ev i fs [] 0 (.ok fs [])
  | other {d ds n r} : ¬ (d.region = i ∧ d.event = ev) → RRun args ev i fs ds n r → RRun args ev i fs (d :: ds) n r
  | guardF {d ds n m r} : d.region = i ∧ d.event = ev → EvalN ⟨fs, args⟩ n d.guard (.ok (.bool false)) →
      RRun args ev i fs ds m r → RRun args ev i fs (d :: ds) (n + 1 + m) r
  | guardT {d ds n k r} : d.region = i ∧ d.event = ev → EvalN ⟨fs, args⟩ n d.guard (.ok (.bool true)) →
      SRun args k (.run d.body fs []) r.toCfg → RRun args ev i fs (d :: ds) (n + 1 + k) r
  | guardBad {d ds n j} : d.region = i ∧ d.event = ev → EvalN ⟨fs, args⟩ n d.guard (.ok (.int j)) →
      RRun args ev i fs (d :: ds) (n + 1) (.ref (.trap .badIndex))
  | guardTrap {d ds n m} : d.region = i ∧ d.event = ev → EvalN ⟨fs, args⟩ n d.guard (.error m) →
      RRun args ev i fs (d :: ds) (n + 1) (.ref (.trap m))

/-- B(P): the per-region bound, computed from the program text. -/
def BR : List StepDecl → Nat
  | [] => 0
  | d :: ds => d.guard.size + 1 + B d.body + BR ds

theorem region_bound (args : List Val) (ev : String) (i : Nat) (fs : List Val) :
    ∀ ds, ∃ n, n ≤ BR ds ∧ RRun args ev i fs ds n (regionStep args ev i fs ds)
  | [] => ⟨0, Nat.le_refl _, .nil⟩
  | d :: ds => by
    obtain ⟨m, hm, hr⟩ := region_bound args ev i fs ds
    simp only [regionStep]
    by_cases hd : d.region = i ∧ d.event = ev
    · rw [if_pos hd]
      obtain ⟨n, hn, hg⟩ := evalN_exists ⟨fs, args⟩ d.guard
      revert hg
      cases eval ⟨fs, args⟩ d.guard with
      | error e => intro hg; exact ⟨n + 1, by simp [BR]; omega, .guardTrap hd hg⟩
      | ok v =>
        cases v with
        | int j => intro hg; exact ⟨n + 1, by simp [BR]; omega, .guardBad hd hg⟩
        | bool bb =>
          cases bb
          · intro hg; exact ⟨n + 1 + m, by simp [BR]; omega, .guardF hd hg hr⟩
          · intro hg
            obtain ⟨k, hk, hb⟩ := stmt_bound args d.body fs []
            exact ⟨n + 1 + k, by simp [BR]; omega, .guardT hd hg hb⟩
    · rw [if_neg hd]; exact ⟨m, by simp [BR]; omega, .other hd hr⟩

/-- δ with cost: every region, in declaration order, on the pre-state. -/
inductive DRun (P : Program) (e : Event) : Nat → State → Nat → Except Refusal (State × List Val) → Prop where
  | nil {i} : DRun P e i [] 0 (.ok ([], []))
  | ref {i fs rs n x} : RRun e.2 e.1 i fs P.steps n (.ref x) → DRun P e i (fs :: rs) n (.error x)
  | okRef {i fs rs n m fs' o x} : RRun e.2 e.1 i fs P.steps n (.ok fs' o) → DRun P e (i + 1) rs m (.error x) →
      DRun P e i (fs :: rs) (n + m) (.error x)
  | okOk {i fs rs n m fs' o rs' os} : RRun e.2 e.1 i fs P.steps n (.ok fs' o) → DRun P e (i + 1) rs m (.ok (rs', os)) →
      DRun P e i (fs :: rs) (n + m) (.ok (fs' :: rs', o ++ os))

/-- **T2 for δ (explicit bound) and T4 for δ**: for every state and event,
    the small-step machine computes exactly δ's region results, with total
    cost at most |s| · B(P), where B(P) = Σ over declared steps of
    (size of guard + 1 + B body). -/
theorem delta_bound (P : Program) (e : Event) : ∀ (s : State) (i : Nat),
    ∃ n, n ≤ s.length * BR P.steps ∧ DRun P e i s n (runRegions P e i s)
  | [], i => ⟨0, by simp, by simp [runRegions]; exact .nil⟩
  | fs :: rs, i => by
    obtain ⟨n, hn, hr⟩ := region_bound e.2 e.1 i fs P.steps
    simp only [runRegions]
    revert hr
    cases regionStep e.2 e.1 i fs P.steps with
    | ref x => intro hr; exact ⟨n, by simp [Nat.succ_mul]; omega, .ref hr⟩
    | ok fs' o =>
      intro hr
      obtain ⟨m, hm, hd⟩ := delta_bound P e rs (i + 1)
      revert hd
      cases runRegions P e (i + 1) rs with
      | error x => intro hd; exact ⟨n + m, by simp [Nat.succ_mul]; omega, .okRef hr hd⟩
      | ok p => obtain ⟨rs', os⟩ := p; intro hd; exact ⟨n + m, by simp [Nat.succ_mul]; omega, .okOk hr hd⟩

end RCE
