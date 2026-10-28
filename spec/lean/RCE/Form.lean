import RCE.Expr
/-! RCE core: step bodies, the step function δ of spec 6.4, the Form
    P = (S, E, δ, D, λ), and theorems T1 T2 T3 T5(state) T7 T8. -/
namespace RCE

inductive Stmt where
  | skip
  | assign (i : Nat) (e : Expr)
  | seq (a b : Stmt)
  | ifs (c : Expr) (a b : Stmt)
  /-- `while c bound n` (spec 6.7) -/
  | loop (n : Nat) (c : Expr) (body : Stmt)
  | fail (x : String)
  | emit (e : Expr)
  deriving Repr

inductive Reason where
  | user (x : String)
  | trap (m : Trap)
  deriving DecidableEq, Repr

inductive SRes where
  | ok (fs : List Val) (os : List Val)
  | ref (r : Reason)
  deriving DecidableEq, Repr

/-- A bounded loop: at most `n` iterations; starting iteration n+1 traps. -/
def iter (f : List Val → List Val → SRes) (c : List Val → Except Trap Val) :
    Nat → List Val → List Val → SRes
  | 0, fs, os =>
    match c fs with
    | .error m => .ref (.trap m)
    | .ok (.int _) => .ref (.trap .badIndex)
    | .ok (.bool false) => .ok fs os
    | .ok (.bool true) => .ref (.trap .loopBound)
  | k + 1, fs, os =>
    match c fs with
    | .error m => .ref (.trap m)
    | .ok (.int _) => .ref (.trap .badIndex)
    | .ok (.bool false) => .ok fs os
    | .ok (.bool true) =>
      match f fs os with
      | .ok fs' os' => iter f c k fs' os'
      | r => r

/-- Reference semantics of step bodies. Total by construction. -/
def exec (args : List Val) : Stmt → List Val → List Val → SRes
  | .skip, fs, os => .ok fs os
  | .assign i e, fs, os =>
    match eval ⟨fs, args⟩ e with
    | .ok v => if i < fs.length then .ok (fs.set i v) os else .ref (.trap .badIndex)
    | .error m => .ref (.trap m)
  | .seq a b, fs, os =>
    match exec args a fs os with
    | .ok fs' os' => exec args b fs' os'
    | r => r
  | .ifs c a b, fs, os =>
    match eval ⟨fs, args⟩ c with
    | .ok (.bool true) => exec args a fs os
    | .ok (.bool false) => exec args b fs os
    | .ok (.int _) => .ref (.trap .badIndex)
    | .error m => .ref (.trap m)
  | .loop n c body, fs, os => iter (exec args body) (fun fs => eval ⟨fs, args⟩ c) n fs os
  | .fail x, _, _ => .ref (.user x)
  | .emit e, fs, os =>
    match eval ⟨fs, args⟩ e with
    | .ok v => .ok fs (os ++ [v])
    | .error m => .ref (.trap m)

/-- Relational (big-step) semantics of step bodies, rule by rule. -/
inductive ExecR (args : List Val) : Stmt → List Val → List Val → SRes → Prop where
  | skip {fs os} : ExecR args .skip fs os (.ok fs os)
  | assign {i e fs os v} : eval ⟨fs, args⟩ e = .ok v → i < fs.length →
      ExecR args (.assign i e) fs os (.ok (fs.set i v) os)
  | assignBad {i e fs os v} : eval ⟨fs, args⟩ e = .ok v → ¬ i < fs.length →
      ExecR args (.assign i e) fs os (.ref (.trap .badIndex))
  | assignTrap {i e fs os m} : eval ⟨fs, args⟩ e = .error m →
      ExecR args (.assign i e) fs os (.ref (.trap m))
  | seq {a b fs os fs' os' r} : ExecR args a fs os (.ok fs' os') → ExecR args b fs' os' r →
      ExecR args (.seq a b) fs os r
  | seqKept {a b fs os x} : ExecR args a fs os (.ref x) → ExecR args (.seq a b) fs os (.ref x)
  | ifT {c a b fs os r} : eval ⟨fs, args⟩ c = .ok (.bool true) → ExecR args a fs os r →
      ExecR args (.ifs c a b) fs os r
  | ifF {c a b fs os r} : eval ⟨fs, args⟩ c = .ok (.bool false) → ExecR args b fs os r →
      ExecR args (.ifs c a b) fs os r
  | ifBad {c a b fs os n} : eval ⟨fs, args⟩ c = .ok (.int n) →
      ExecR args (.ifs c a b) fs os (.ref (.trap .badIndex))
  | ifTrap {c a b fs os m} : eval ⟨fs, args⟩ c = .error m →
      ExecR args (.ifs c a b) fs os (.ref (.trap m))
  | loopStop {n c b fs os} : eval ⟨fs, args⟩ c = .ok (.bool false) →
      ExecR args (.loop n c b) fs os (.ok fs os)
  | loopBound {c b fs os} : eval ⟨fs, args⟩ c = .ok (.bool true) →
      ExecR args (.loop 0 c b) fs os (.ref (.trap .loopBound))
  | loopStep {n c b fs os fs' os' r} : eval ⟨fs, args⟩ c = .ok (.bool true) →
      ExecR args b fs os (.ok fs' os') → ExecR args (.loop n c b) fs' os' r →
      ExecR args (.loop (n+1) c b) fs os r
  | loopRef {n c b fs os x} : eval ⟨fs, args⟩ c = .ok (.bool true) →
      ExecR args b fs os (.ref x) → ExecR args (.loop (n+1) c b) fs os (.ref x)
  | loopBad {n c b fs os k} : eval ⟨fs, args⟩ c = .ok (.int k) →
      ExecR args (.loop n c b) fs os (.ref (.trap .badIndex))
  | loopTrap {n c b fs os m} : eval ⟨fs, args⟩ c = .error m →
      ExecR args (.loop n c b) fs os (.ref (.trap m))
  | fail {x fs os} : ExecR args (.fail x) fs os (.ref (.user x))
  | emit {e fs os v} : eval ⟨fs, args⟩ e = .ok v → ExecR args (.emit e) fs os (.ok fs (os ++ [v]))
  | emitTrap {e fs os m} : eval ⟨fs, args⟩ e = .error m → ExecR args (.emit e) fs os (.ref (.trap m))

/-- Soundness of the relational rules w.r.t. the reference function. -/
theorem execR_sound {args s fs os r} (h : ExecR args s fs os r) : exec args s fs os = r := by
  induction h with
  | skip => rfl
  | assign he hi => simp [exec, he, hi]
  | assignBad he hi => simp [exec, he, hi]
  | assignTrap he => simp [exec, he]
  | seq _ _ iha ihb => simp [exec, iha, ihb]
  | seqKept _ iha => simp [exec, iha]
  | ifT hc _ ih => simp [exec, hc, ih]
  | ifF hc _ ih => simp [exec, hc, ih]
  | ifBad hc => simp [exec, hc]
  | ifTrap hc => simp [exec, hc]
  | @loopStop n _ _ _ _ hc => cases n <;> simp [exec, iter, hc]
  | loopBound hc => simp [exec, iter, hc]
  | loopStep hc _ _ ihb ihl =>
    simp only [exec] at ihl ⊢; simp [iter, hc, ihb, ihl]
  | loopRef hc _ ihb => simp [exec, iter, hc, ihb]
  | @loopBad n _ _ _ _ _ hc => cases n <;> simp [exec, iter, hc]
  | @loopTrap n _ _ _ _ _ hc => cases n <;> simp [exec, iter, hc]
  | fail => rfl
  | emit he => simp [exec, he]
  | emitTrap he => simp [exec, he]

/-- Completeness: the relation derives the reference result. -/
theorem execR_complete (args : List Val) (s : Stmt) : ∀ fs os, ExecR args s fs os (exec args s fs os) := by
  induction s with
  | skip => intro fs os; exact .skip
  | assign i e =>
    intro fs os; simp only [exec]
    cases he : eval ⟨fs, args⟩ e with
    | ok v => by_cases hi : i < fs.length
              · simp [hi]; exact .assign he hi
              · simp [hi]; exact .assignBad he hi
    | error m => exact .assignTrap he
  | seq a b iha ihb =>
    intro fs os; simp only [exec]
    have ha := iha fs os
    cases hr : exec args a fs os with
    | ok fs' os' => rw [hr] at ha; exact .seq ha (ihb fs' os')
    | ref x => rw [hr] at ha; exact .seqKept ha
  | ifs c a b iha ihb =>
    intro fs os; simp only [exec]
    cases hc : eval ⟨fs, args⟩ c with
    | error m => exact .ifTrap hc
    | ok v => cases v with
      | int n => exact .ifBad hc
      | bool bb => cases bb
                   · exact .ifF hc (ihb fs os)
                   · exact .ifT hc (iha fs os)
  | loop n c body ih =>
    induction n with
    | zero =>
      intro fs os; simp only [exec, iter]
      cases hc : eval ⟨fs, args⟩ c with
      | error m => exact .loopTrap hc
      | ok v =>
        cases v with
        | int k => exact .loopBad hc
        | bool bb =>
          cases bb with
          | false => exact .loopStop hc
          | true => exact .loopBound hc
    | succ k ihk =>
      intro fs os; simp only [exec, iter]
      cases hc : eval ⟨fs, args⟩ c with
      | error m => exact .loopTrap hc
      | ok v =>
        cases v with
        | int j => exact .loopBad hc
        | bool bb =>
          cases bb with
          | false => exact .loopStop hc
          | true =>
            dsimp only
            have hb := ih fs os
            cases hr : exec args body fs os with
            | ok fs' os' =>
              rw [hr] at hb
              have h2 := ihk fs' os'
              simp only [exec] at h2
              exact .loopStep hc hb h2
            | ref x => rw [hr] at hb; exact .loopRef hc hb
  | fail x => intro fs os; exact .fail
  | emit e =>
    intro fs os; simp only [exec]
    cases he : eval ⟨fs, args⟩ e with
    | ok v => exact .emit he
    | error m => exact .emitTrap he

/-- **T1 Determinism + T2 Totality of step bodies**: every body, from every
    state and event payload, has exactly one outcome. -/
theorem exec_unique (args : List Val) (s : Stmt) (fs os : List Val) :
    ∃ r, ExecR args s fs os r ∧ ∀ r', ExecR args s fs os r' → r' = r :=
  ⟨_, execR_complete args s fs os, fun _ h => execR_sound h ▸ rfl⟩

/-! ## Programs and δ (spec 6.4) -/

structure StepDecl where
  region : Nat
  event : String
  guard : Expr
  body : Stmt

structure Program where
  steps : List StepDecl

abbrev State := List (List Val)
abbrev Event := String × List Val

inductive Out where
  | emit (v : Val)
  | kept (r : Reason)
  deriving DecidableEq, Repr

/-- The step of region `i` for event `ev`: the first declared step for this
    region and event whose guard holds (spec 6.4 item 1); none = frame. -/
def regionStep (args : List Val) (ev : String) (i : Nat) (fs : List Val) : List StepDecl → SRes
  | [] => .ok fs []
  | d :: ds =>
    if d.region = i ∧ d.event = ev then
      match eval ⟨fs, args⟩ d.guard with
      | .ok (.bool true) => exec args d.body fs []
      | .ok (.bool false) => regionStep args ev i fs ds
      | .ok (.int _) => .ref (.trap .badIndex)
      | .error m => .ref (.trap m)
    else regionStep args ev i fs ds

/-- Every region steps on the pre-state; the first kept outcome (declaration
    order) gives the whole event the kept outcome. -/
def runRegions (P : Program) (e : Event) : Nat → State → Except Reason (State × List Val)
  | _, [] => .ok ([], [])
  | i, fs :: rs =>
    match regionStep e.2 e.1 i fs P.steps with
    | .ref x => .error x
    | .ok fs' o =>
      match runRegions P e (i+1) rs with
      | .error x => .error x
      | .ok (rs', os) => .ok (fs' :: rs', o ++ os)

/-- δ : S × E → S × O* -/
def delta (P : Program) (s : State) (e : Event) : State × List Out :=
  match runRegions P e 0 s with
  | .ok (s', o) => (s', o.map .emit)
  | .error x => (s, [.kept x])

/-- **T8 Atomicity**: a kept outcome leaves every region unchanged, and an
    output list containing a kept record only arises from a kept outcome. -/
theorem atomicity (P : Program) (s : State) (e : Event) (x : Reason)
    (h : Out.kept x ∈ (delta P s e).2) : (delta P s e).1 = s := by
  unfold delta at h ⊢
  cases hr : runRegions P e 0 s with
  | error y => rfl
  | ok p => rw [hr] at h; simp at h

/-! ## The Form's log and the fold law (spec 6.1, T3) -/

structure Machine where
  state : State
  log : List (Event × List Out)

def admit (P : Program) (m : Machine) (e : Event) : Machine :=
  let r := delta P m.state e
  ⟨r.1, m.log ++ [(e, r.2)]⟩

def run (P : Program) (s0 : State) (es : List Event) : Machine :=
  es.foldl (admit P) ⟨s0, []⟩

theorem run_state_aux (P : Program) : ∀ (es : List Event) (m : Machine),
    (es.foldl (admit P) m).state = es.foldl (fun s e => (delta P s e).1) m.state := by
  intro es; induction es with
  | nil => intro m; rfl
  | cons e es ih => intro m; simp only [List.foldl]; rw [ih]; rfl

/-- **T3 Fold law**: state = foldl(δ, s₀, D). -/
theorem fold_law (P : Program) (s0 : State) (es : List Event) :
    (run P s0 es).state = es.foldl (fun s e => (delta P s e).1) s0 :=
  run_state_aux P es ⟨s0, []⟩

theorem run_log_aux (P : Program) : ∀ (es : List Event) (m : Machine),
    ((es.foldl (admit P) m).log.map Prod.fst) = m.log.map Prod.fst ++ es := by
  intro es; induction es with
  | nil => intro m; simp
  | cons e es ih => intro m; simp only [List.foldl]; rw [ih]; simp [admit]

/-- The log records exactly the admitted events. -/
theorem log_is_events (P : Program) (s0 : State) (es : List Event) :
    (run P s0 es).log.map Prod.fst = es := by
  have := run_log_aux P es ⟨s0, []⟩; simpa using this

/-- **T3 Replay**: folding the log again reproduces the machine exactly,
    every state and every output entry. -/
theorem replay (P : Program) (s0 : State) (es : List Event) :
    run P s0 ((run P s0 es).log.map Prod.fst) = run P s0 es := by
  rw [log_is_events]

/-! ## Name independence (T7): event and error names are opaque. -/

def Stmt.rename (ρ : String → String) : Stmt → Stmt
  | .skip => .skip
  | .assign i e => .assign i e
  | .seq a b => .seq (a.rename ρ) (b.rename ρ)
  | .ifs c a b => .ifs c (a.rename ρ) (b.rename ρ)
  | .loop n c b => .loop n c (b.rename ρ)
  | .fail x => .fail (ρ x)
  | .emit e => .emit e

def Reason.rename (ρ : String → String) : Reason → Reason
  | .user x => .user (ρ x)
  | .trap m => .trap m

def SRes.rename (ρ : String → String) : SRes → SRes
  | .ok fs os => .ok fs os
  | .ref x => .ref (x.rename ρ)

def Out.rename (ρ : String → String) : Out → Out
  | .emit v => .emit v
  | .kept x => .kept (x.rename ρ)

def Program.rename (ρ : String → String) (P : Program) : Program :=
  ⟨P.steps.map fun d => ⟨d.region, ρ d.event, d.guard, d.body.rename ρ⟩⟩

theorem iter_rename (ρ : String → String) (f g : List Val → List Val → SRes)
    (hfg : ∀ fs os, g fs os = (f fs os).rename ρ) (c) :
    ∀ n fs os, iter g c n fs os = (iter f c n fs os).rename ρ := by
  intro n; induction n with
  | zero => intro fs os; simp only [iter]; split <;> simp_all [SRes.rename, Reason.rename]
  | succ k ih =>
    intro fs os; simp only [iter]
    split <;> simp_all [SRes.rename, Reason.rename]
    cases hf : f fs os with
    | ok fs' os' => simp [hf, SRes.rename, ih]
    | ref x => simp [hf, SRes.rename]

theorem exec_rename (ρ : String → String) (args : List Val) (s : Stmt) :
    ∀ fs os, exec args (s.rename ρ) fs os = (exec args s fs os).rename ρ := by
  induction s with
  | skip => intro fs os; rfl
  | assign i e => intro fs os; simp only [Stmt.rename, exec]; split <;> (try split) <;> rfl
  | seq a b iha ihb =>
    intro fs os; simp only [Stmt.rename, exec]; rw [iha]
    cases exec args a fs os with
    | ok fs' os' => simp [SRes.rename, ihb]
    | ref x => rfl
  | ifs c a b iha ihb => intro fs os; simp only [Stmt.rename, exec]; split <;> simp_all [SRes.rename, Reason.rename]
  | loop n c b ih => intro fs os; simp only [Stmt.rename, exec]; exact iter_rename ρ _ _ ih _ n fs os
  | fail x => intro fs os; rfl
  | emit e => intro fs os; simp only [Stmt.rename, exec]; split <;> rfl

theorem regionStep_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b)
    (args ev i fs) : ∀ ds : List StepDecl,
    regionStep args (ρ ev) i fs (ds.map fun d => ⟨d.region, ρ d.event, d.guard, d.body.rename ρ⟩)
      = (regionStep args ev i fs ds).rename ρ := by
  intro ds; induction ds with
  | nil => rfl
  | cons d ds ih =>
    simp only [List.map, regionStep]
    have hiff : (ρ d.event = ρ ev) ↔ (d.event = ev) := ⟨hρ _ _, fun h => h ▸ rfl⟩
    by_cases hc : d.region = i ∧ d.event = ev
    · have hc' : d.region = i ∧ ρ d.event = ρ ev := ⟨hc.1, hiff.2 hc.2⟩
      simp only [hc, hc', and_self, if_true]
      split <;> simp_all [SRes.rename, Reason.rename, exec_rename]
    · have hc' : ¬ (d.region = i ∧ ρ d.event = ρ ev) := fun h => hc ⟨h.1, hiff.1 h.2⟩
      simp only [hc, hc', if_false]; exact ih

theorem runRegions_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b)
    (P : Program) (e : Event) : ∀ (s : State) i,
    runRegions (P.rename ρ) (ρ e.1, e.2) i s
      = match runRegions P e i s with
        | .ok r => .ok r
        | .error x => .error (x.rename ρ) := by
  intro s; induction s with
  | nil => intro i; rfl
  | cons fs rs ih =>
    intro i
    simp only [runRegions, Program.rename]
    rw [regionStep_rename ρ hρ]
    cases regionStep e.2 e.1 i fs P.steps with
    | ref x => rfl
    | ok fs' o =>
      simp only [SRes.rename]
      have := ih (i+1); simp only [Program.rename] at this; rw [this]
      cases runRegions P e (i+1) rs <;> rfl

/-- **T7 Name independence**: consistently renaming event and error names
    (any injective ρ) commutes with δ. Variables and fields are positional
    in the core (De Bruijn), so their names cannot affect δ at all. -/
theorem name_independence (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b)
    (P : Program) (s : State) (e : Event) :
    delta (P.rename ρ) s (ρ e.1, e.2) =
      ((delta P s e).1, (delta P s e).2.map (Out.rename ρ)) := by
  unfold delta
  rw [runRegions_rename ρ hρ]
  cases runRegions P e 0 s with
  | ok r => simp [Out.rename]
  | error x => simp [Out.rename]

end RCE
