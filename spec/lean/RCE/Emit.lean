import RCE.Form
/-! T9 for the Rust emitter on the core: a verified translation.

`compile` maps core step bodies (Stmt, Expr) to MiniRust, a deep embedding
of the Rust subset the emitter prints (`RE`, `RS`), with its own big-step
semantics written the way Rust executes it: a mutable store with one loop
counter variable `k_d` per loop nesting depth, `loop { … }` with `break`,
`return Err(…)` for refusals, `?` propagation of traps, and `fs.get(i)` /
`fs[i] = v` with bounds checks.  `toRust` prints MiniRust as Rust source.

Proved: for every core statement, the compiled MiniRust program has exactly
one outcome, and it is the core semantics' outcome (`exec`); the same for
expressions (`reval_compile`) and for the selection of a region's step.
Trusted (not proved): that rustc executes the printed text as the MiniRust
semantics says (the model of Rust's `i128` arithmetic, `Vec::get`, `?`,
`loop`/`break`), and the hand-written prelude functions `binop`/`truthy`;
these are checked by differential testing against the Lean evaluator. -/
namespace RCE.Emit

/-- MiniRust expressions: `Result<Val, Trap>`-valued Rust expressions. -/
inductive RE where
  | ok (v : Val)                         -- Ok(Val::…)
  | err (m : Trap)                       -- Err(Trap::…)
  | fld (i : Nat)                        -- get(&fs, i)
  | arg (i : Nat)                        -- get(&args, i)
  | bin (op : BinOp) (ta : Ty) (a b : RE) -- binop(op, ta, (a)?, (b)?)
  | ite (c a b : RE)                     -- if truthy((c)?)? { a } else { b }

/-- Model of the prelude's `truthy`. -/
def truthy : Val → Except Trap Bool
  | .bool b => .ok b
  | .int _ => .error .badIndex

/-- Rust evaluation: left to right, `?` returns the first `Err`. -/
def reval (E : Env) : RE → Except Trap Val
  | .ok v => .ok v
  | .err m => .error m
  | .fld i => match E.fields.get? i with | some v => .ok v | none => .error .badIndex
  | .arg i => match E.args.get? i with | some v => .ok v | none => .error .badIndex
  | .bin op ta a b => match reval E a with
    | .error m => .error m
    | .ok x => match reval E b with
      | .error m => .error m
      | .ok y => binop op ta x y
  | .ite c a b => match reval E c with
    | .error m => .error m
    | .ok v => match truthy v with
      | .error m => .error m
      | .ok true => reval E a
      | .ok false => reval E b

def compileE : Expr → RE
  | .lit v => .ok v
  | .trap m => .err m
  | .field i => .fld i
  | .arg i => .arg i
  | .bin op ta a b => .bin op ta (compileE a) (compileE b)
  | .ite c a b => .ite (compileE c) (compileE a) (compileE b)

theorem reval_compile (E : Env) (e : Expr) : reval E (compileE e) = eval E e := by
  induction e with
  | lit v => rfl
  | trap m => rfl
  | field i => rfl
  | arg i => rfl
  | bin op ta a b iha ihb =>
    simp only [compileE, reval, eval, iha, ihb, bind, Except.bind]
    cases eval E a <;> cases eval E b <;> rfl
  | ite c a b ihc iha ihb =>
    simp only [compileE, reval, eval, ihc, iha, ihb]
    cases eval E c with
    | error m => rfl
    | ok v => cases v with
      | int n => rfl
      | bool bb => cases bb <;> rfl

/-- MiniRust statements. -/
inductive RS where
  | skip                         -- {}
  | set (i : Nat) (e : RE)       -- set(&mut fs, i, (e)?)?;
  | seq (a b : RS)               -- a b
  | ifelse (c : RE) (a b : RS)   -- if truthy((c)?)? { a } else { b }
  | loop (body : RS)             -- loop { body }
  | brk                          -- break;
  | fail (x : String)            -- return Err(Refusal::User("x"));
  | trap (m : Trap)              -- return Err(Refusal::Trap(Trap::m));
  | letK (d n : Nat)             -- let mut k_d: u64 = n;
  | decK (d : Nat)               -- k_d -= 1;
  | ifKZero (d : Nat) (s : RS)   -- if k_d == 0 { s }
  | ifNot (c : RE) (s : RS)      -- if !truthy((c)?)? { s }
  | push (e : RE)                -- os.push((e)?);

/-- The Rust store: fields, outputs, loop counters. -/
structure St where
  fs : List Val
  os : List Val
  ks : Nat → Nat

def updK (ks : Nat → Nat) (d n : Nat) : Nat → Nat := fun j => if j = d then n else ks j

inductive Flow where
  | normal (st : St)
  | brk (st : St)
  | ret (r : Refusal)

/-- Big-step semantics of MiniRust, as Rust runs it. -/
inductive RExec (args : List Val) : RS → St → Flow → Prop where
  | skip {st} : RExec args .skip st (.normal st)
  | set {i e st v} : reval ⟨st.fs, args⟩ e = .ok v → i < st.fs.length →
      RExec args (.set i e) st (.normal { st with fs := st.fs.set i v })
  | setBad {i e st v} : reval ⟨st.fs, args⟩ e = .ok v → ¬ i < st.fs.length →
      RExec args (.set i e) st (.ret (.trap .badIndex))
  | setErr {i e st m} : reval ⟨st.fs, args⟩ e = .error m → RExec args (.set i e) st (.ret (.trap m))
  | seqN {a b st st' f} : RExec args a st (.normal st') → RExec args b st' f → RExec args (.seq a b) st f
  | seqB {a b st st'} : RExec args a st (.brk st') → RExec args (.seq a b) st (.brk st')
  | seqR {a b st r} : RExec args a st (.ret r) → RExec args (.seq a b) st (.ret r)
  | ifT {c a b st f} : reval ⟨st.fs, args⟩ c = .ok (.bool true) → RExec args a st f → RExec args (.ifelse c a b) st f
  | ifF {c a b st f} : reval ⟨st.fs, args⟩ c = .ok (.bool false) → RExec args b st f → RExec args (.ifelse c a b) st f
  | ifBad {c a b st n} : reval ⟨st.fs, args⟩ c = .ok (.int n) → RExec args (.ifelse c a b) st (.ret (.trap .badIndex))
  | ifErr {c a b st m} : reval ⟨st.fs, args⟩ c = .error m → RExec args (.ifelse c a b) st (.ret (.trap m))
  | loopN {b st st' f} : RExec args b st (.normal st') → RExec args (.loop b) st' f → RExec args (.loop b) st f
  | loopB {b st st'} : RExec args b st (.brk st') → RExec args (.loop b) st (.normal st')
  | loopR {b st r} : RExec args b st (.ret r) → RExec args (.loop b) st (.ret r)
  | brk {st} : RExec args .brk st (.brk st)
  | fail {x st} : RExec args (.fail x) st (.ret (.user x))
  | trap {m st} : RExec args (.trap m) st (.ret (.trap m))
  | letK {d n st} : RExec args (.letK d n) st (.normal { st with ks := updK st.ks d n })
  | decK {d st} : RExec args (.decK d) st (.normal { st with ks := updK st.ks d (st.ks d - 1) })
  | kZero {d s st f} : st.ks d = 0 → RExec args s st f → RExec args (.ifKZero d s) st f
  | kPos {d s st} : st.ks d ≠ 0 → RExec args (.ifKZero d s) st (.normal st)
  | notF {c s st f} : reval ⟨st.fs, args⟩ c = .ok (.bool false) → RExec args s st f → RExec args (.ifNot c s) st f
  | notT {c s st} : reval ⟨st.fs, args⟩ c = .ok (.bool true) → RExec args (.ifNot c s) st (.normal st)
  | notBad {c s st n} : reval ⟨st.fs, args⟩ c = .ok (.int n) → RExec args (.ifNot c s) st (.ret (.trap .badIndex))
  | notErr {c s st m} : reval ⟨st.fs, args⟩ c = .error m → RExec args (.ifNot c s) st (.ret (.trap m))
  | push {e st v} : reval ⟨st.fs, args⟩ e = .ok v → RExec args (.push e) st (.normal { st with os := st.os ++ [v] })
  | pushErr {e st m} : reval ⟨st.fs, args⟩ e = .error m → RExec args (.push e) st (.ret (.trap m))

/-- The emitter on step bodies; `d` is the loop nesting depth. -/
def compile (d : Nat) : Stmt → RS
  | .skip => .skip
  | .assign i e => .set i (compileE e)
  | .seq a b => .seq (compile d a) (compile d b)
  | .ifs c a b => .ifelse (compileE c) (compile d a) (compile d b)
  | .loop n c b => .seq (.letK d n)
      (.loop (.seq (.ifNot (compileE c) .brk) (.seq (.ifKZero d (.trap .loopBound)) (.seq (.decK d) (compile (d + 1) b)))))
  | .fail x => .fail x
  | .emit e => .push (compileE e)

/-- How a core result is observed in Rust: the fields and outputs (counters are internal). -/
def Obs : SRes → Flow → Prop
  | .ok fs os, .normal st => st.fs = fs ∧ st.os = os
  | .ref r, .ret r' => r = r'
  | _, _ => False

theorem RExec.det {args s st f1 f2} (h1 : RExec args s st f1) (h2 : RExec args s st f2) : f1 = f2 := by
  induction h1 generalizing f2 with
  | seqN _ _ iha ihb => cases h2 with
    | seqN ha' hb' => cases iha ha'; exact ihb hb'
    | seqB ha' => cases iha ha'
    | seqR ha' => cases iha ha'
  | seqB _ iha => cases h2 with
    | seqN ha' _ => cases iha ha'
    | seqB ha' => exact iha ha'
    | seqR ha' => cases iha ha'
  | seqR _ iha => cases h2 with
    | seqN ha' _ => cases iha ha'
    | seqB ha' => cases iha ha'
    | seqR ha' => exact iha ha'
  | loopN _ _ iha ihb => cases h2 with
    | loopN ha' hb' => cases iha ha'; exact ihb hb'
    | loopB ha' => cases iha ha'
    | loopR ha' => cases iha ha'
  | loopB _ iha => cases h2 with
    | loopN ha' _ => cases iha ha'
    | loopB ha' => cases iha ha'; rfl
    | loopR ha' => cases iha ha'
  | loopR _ iha => cases h2 with
    | loopN ha' _ => cases iha ha'
    | loopB ha' => cases iha ha'
    | loopR ha' => exact iha ha'
  | ifT hc _ iha => cases h2 <;> first | exact iha ‹_› | simp_all
  | ifF hc _ iha => cases h2 <;> first | exact iha ‹_› | simp_all
  | kZero hk _ iha => cases h2 <;> first | exact iha ‹_› | simp_all
  | notF hc _ iha => cases h2 <;> first | exact iha ‹_› | simp_all
  | _ => cases h2 <;> simp_all

def Frame (d : Nat) (st : St) : Flow → Prop
  | .normal st' => ∀ j < d, st'.ks j = st.ks j
  | _ => True

/-- The loop as emitted: `loop { if !c { break } if k_d == 0 { return Err(LoopBound) } k_d -= 1; body }`. -/
def loopBody (d : Nat) (c : RE) (cb : RS) : RS :=
  .seq (.ifNot c .brk) (.seq (.ifKZero d (.trap .loopBound)) (.seq (.decK d) cb))

theorem loop_correct (args : List Val) (d : Nat) (c : Expr) (b : Stmt) (cb : RS)
    (hb : ∀ st : St, ∃ f, RExec args cb st f ∧ Obs (exec args b st.fs st.os) f ∧ Frame (d+1) st f) :
    ∀ n (st : St), st.ks d = n → ∃ f, RExec args (.loop (loopBody d (compileE c) cb)) st f ∧
      Obs (iter (exec args b) (fun fs => eval ⟨fs, args⟩ c) n st.fs st.os) f ∧ Frame d st f := by
  intro n
  induction n with
  | zero =>
    intro st hk
    cases hc : eval ⟨st.fs, args⟩ c with
    | error m =>
      exact ⟨_, .loopR (.seqR (.notErr (by rw [reval_compile]; exact hc))), by simp [iter, hc, Obs], trivial⟩
    | ok v => cases v with
      | int z =>
        exact ⟨_, .loopR (.seqR (.notBad (by rw [reval_compile]; exact hc))), by simp [iter, hc, Obs], trivial⟩
      | bool bb => cases bb with
        | false =>
          exact ⟨_, .loopB (.seqB (.notF (by rw [reval_compile]; exact hc) .brk)), by simp [iter, hc, Obs],
            fun _ _ => rfl⟩
        | true =>
          exact ⟨_, .loopR (.seqN (.notT (by rw [reval_compile]; exact hc)) (.seqR (.kZero hk .trap))),
            by simp [iter, hc, Obs], trivial⟩
  | succ k ih =>
    intro st hk
    cases hc : eval ⟨st.fs, args⟩ c with
    | error m =>
      exact ⟨_, .loopR (.seqR (.notErr (by rw [reval_compile]; exact hc))), by simp [iter, hc, Obs], trivial⟩
    | ok v => cases v with
      | int z =>
        exact ⟨_, .loopR (.seqR (.notBad (by rw [reval_compile]; exact hc))), by simp [iter, hc, Obs], trivial⟩
      | bool bb => cases bb with
        | false =>
          exact ⟨_, .loopB (.seqB (.notF (by rw [reval_compile]; exact hc) .brk)), by simp [iter, hc, Obs],
            fun _ _ => rfl⟩
        | true =>
          have hT : reval ⟨st.fs, args⟩ (compileE c) = .ok (.bool true) := by rw [reval_compile]; exact hc
          have hpos : st.ks d ≠ 0 := by omega
          obtain ⟨f, hr, ho, hf⟩ := hb { st with ks := updK st.ks d (st.ks d - 1) }
          cases f with
          | normal st2 =>
            cases he : exec args b st.fs st.os with
            | ref r => simp [he, Obs] at ho
            | ok fs' os' =>
              simp only [he, Obs] at ho
              have hk2 : st2.ks d = k := by
                rw [hf d (by omega)]; simp [updK]; omega
              obtain ⟨f', hr', ho', hf'⟩ := ih st2 hk2
              refine ⟨f', .loopN (.seqN (.notT hT) (.seqN (.kPos hpos) (.seqN .decK hr))) hr', ?_, ?_⟩
              · simp only [iter, hc, he]; rw [← ho.1, ← ho.2]; exact ho'
              · cases f' with
                | normal st3 =>
                  intro j hj; rw [hf' j hj, hf j (by omega)]; simp [updK]; omega
                | _ => trivial
          | ret r =>
            refine ⟨_, .loopR (.seqN (.notT hT) (.seqN (.kPos hpos) (.seqN .decK hr))), ?_, trivial⟩
            cases he : exec args b st.fs st.os with
            | ok _ _ => simp [he, Obs] at ho
            | ref r' => simp [he, Obs] at ho; simp [iter, hc, he, Obs, ho]
          | brk st2 =>
            cases he : exec args b st.fs st.os <;> simp [he, Obs] at ho

/-- **T9 (Rust emitter, core), existence**: the emitted code runs to an outcome, and
    that outcome is the core semantics' outcome. -/
theorem compile_correct (args : List Val) (s : Stmt) :
    ∀ d (st : St), ∃ f, RExec args (compile d s) st f ∧ Obs (exec args s st.fs st.os) f ∧ Frame d st f := by
  induction s with
  | skip => intro d st; exact ⟨_, .skip, ⟨rfl, rfl⟩, fun _ _ => rfl⟩
  | assign i e =>
    intro d st
    cases h : eval ⟨st.fs, args⟩ e with
    | error m => exact ⟨_, .setErr (by rw [reval_compile]; exact h), by simp [exec, h, Obs], trivial⟩
    | ok v =>
      by_cases hi : i < st.fs.length
      · exact ⟨_, .set (by rw [reval_compile]; exact h) hi, by simp [exec, h, hi, Obs], fun _ _ => rfl⟩
      · exact ⟨_, .setBad (by rw [reval_compile]; exact h) hi, by simp [exec, h, hi, Obs], trivial⟩
  | seq a b iha ihb =>
    intro d st
    obtain ⟨f1, h1, o1, r1⟩ := iha d st
    cases f1 with
    | normal st1 =>
      cases he : exec args a st.fs st.os with
      | ref r => simp [he, Obs] at o1
      | ok fs os =>
        simp only [he, Obs] at o1
        obtain ⟨f2, h2, o2, r2⟩ := ihb d st1
        refine ⟨f2, .seqN h1 h2, ?_, ?_⟩
        · simp only [exec, he]; rw [← o1.1, ← o1.2]; exact o2
        · cases f2 with
          | normal st2 => intro j hj; rw [r2 j hj, r1 j hj]
          | _ => trivial
    | ret r =>
      refine ⟨_, .seqR h1, ?_, trivial⟩
      cases he : exec args a st.fs st.os with
      | ok _ _ => simp [he, Obs] at o1
      | ref r' => simp [he, Obs] at o1; simp [exec, he, Obs, o1]
    | brk st1 => cases he : exec args a st.fs st.os <;> simp [he, Obs] at o1
  | ifs c a b iha ihb =>
    intro d st
    cases h : eval ⟨st.fs, args⟩ c with
    | error m => exact ⟨_, .ifErr (by rw [reval_compile]; exact h), by simp [exec, h, Obs], trivial⟩
    | ok v => cases v with
      | int z => exact ⟨_, .ifBad (by rw [reval_compile]; exact h), by simp [exec, h, Obs], trivial⟩
      | bool bb => cases bb with
        | true =>
          obtain ⟨f, hr, ho, hf⟩ := iha d st
          exact ⟨f, .ifT (by rw [reval_compile]; exact h) hr, by simp only [exec, h]; exact ho, hf⟩
        | false =>
          obtain ⟨f, hr, ho, hf⟩ := ihb d st
          exact ⟨f, .ifF (by rw [reval_compile]; exact h) hr, by simp only [exec, h]; exact ho, hf⟩
  | loop n c b ihb =>
    intro d st
    obtain ⟨f, hr, ho, hf⟩ := loop_correct args d c b (compile (d+1) b) (fun st => ihb (d+1) st) n
      { st with ks := updK st.ks d n } (by simp [updK])
    refine ⟨f, .seqN .letK hr, by simp only [exec]; exact ho, ?_⟩
    cases f with
    | normal st2 => intro j hj; rw [hf j hj]; simp [updK]; omega
    | _ => trivial
  | fail x => intro d st; exact ⟨_, .fail, by simp [exec, Obs], trivial⟩
  | emit e =>
    intro d st
    cases h : eval ⟨st.fs, args⟩ e with
    | error m => exact ⟨_, .pushErr (by rw [reval_compile]; exact h), by simp [exec, h, Obs], trivial⟩
    | ok v => exact ⟨_, .push (by rw [reval_compile]; exact h), by simp [exec, h, Obs], fun _ _ => rfl⟩

/-- **T9 (Rust emitter, core)**: every run of the emitted step body, from any
    counter store, ends in the outcome the core semantics gives. -/
theorem emit_agrees (args : List Val) (s : Stmt) (fs os : List Val) (ks : Nat → Nat) (f : Flow)
    (h : RExec args (compile 0 s) ⟨fs, os, ks⟩ f) : Obs (exec args s fs os) f := by
  obtain ⟨f', h', o', _⟩ := compile_correct args s 0 ⟨fs, os, ks⟩
  rw [RExec.det h h']; exact o'

theorem emit_total (args : List Val) (s : Stmt) (fs os : List Val) (ks : Nat → Nat) :
    ∃ f, RExec args (compile 0 s) ⟨fs, os, ks⟩ f := by
  obtain ⟨f, h, _⟩ := compile_correct args s 0 ⟨fs, os, ks⟩; exact ⟨f, h⟩

/-- Emitted step selection for one region: `for d in steps { if d matches { match truthy(guard?)? … } }`. -/
inductive RRegion (args : List Val) (ev : String) (i : Nat) : List StepDecl → St → Flow → Prop where
  | nil {st} : RRegion args ev i [] st (.normal st)
  | skipD {d ds st f} : ¬ (d.region = i ∧ d.event = ev) → RRegion args ev i ds st f → RRegion args ev i (d :: ds) st f
  | gT {d ds st f} : d.region = i ∧ d.event = ev → reval ⟨st.fs, args⟩ (compileE d.guard) = .ok (.bool true) →
      RExec args (compile 0 d.body) st f → RRegion args ev i (d :: ds) st f
  | gF {d ds st f} : d.region = i ∧ d.event = ev → reval ⟨st.fs, args⟩ (compileE d.guard) = .ok (.bool false) →
      RRegion args ev i ds st f → RRegion args ev i (d :: ds) st f
  | gBad {d ds st z} : d.region = i ∧ d.event = ev → reval ⟨st.fs, args⟩ (compileE d.guard) = .ok (.int z) →
      RRegion args ev i (d :: ds) st (.ret (.trap .badIndex))
  | gErr {d ds st m} : d.region = i ∧ d.event = ev → reval ⟨st.fs, args⟩ (compileE d.guard) = .error m →
      RRegion args ev i (d :: ds) st (.ret (.trap m))

/-- **T9 for step selection**: the emitted region step exists and agrees with `regionStep`. -/
theorem region_agrees (args : List Val) (ev : String) (i : Nat) (ds : List StepDecl) (fs : List Val)
    (ks : Nat → Nat) : (∃ f, RRegion args ev i ds ⟨fs, [], ks⟩ f) ∧
      ∀ f, RRegion args ev i ds ⟨fs, [], ks⟩ f → Obs (regionStep args ev i fs ds) f := by
  induction ds with
  | nil =>
    refine ⟨⟨_, .nil⟩, fun f h => ?_⟩
    cases h; exact ⟨rfl, rfl⟩
  | cons d ds ih =>
    by_cases hm : d.region = i ∧ d.event = ev
    · cases hg : eval ⟨fs, args⟩ d.guard with
      | error m =>
        have hg' : reval ⟨fs, args⟩ (compileE d.guard) = .error m := by rw [reval_compile]; exact hg
        refine ⟨⟨_, .gErr hm hg'⟩, fun f h => ?_⟩
        cases h <;> simp_all [regionStep, Obs]
      | ok v =>
        have hg' : reval ⟨fs, args⟩ (compileE d.guard) = .ok v := by rw [reval_compile]; exact hg
        cases v with
        | int z =>
          refine ⟨⟨_, .gBad hm hg'⟩, fun f h => ?_⟩
          cases h <;> simp_all [regionStep, Obs]
        | bool bb => cases bb with
          | false =>
            obtain ⟨⟨f0, h0⟩, ih2⟩ := ih
            refine ⟨⟨f0, .gF hm hg' h0⟩, fun f h => ?_⟩
            cases h with
            | skipD hn _ => exact absurd hm hn
            | gT _ h1 _ => simp_all
            | gF _ _ h1 => simp only [regionStep, hm, and_self, if_true, hg]; exact ih2 f h1
            | gBad _ h1 => simp_all
            | gErr _ h1 => simp_all
          | true =>
            obtain ⟨f0, h0, _⟩ := compile_correct args d.body 0 ⟨fs, [], ks⟩
            refine ⟨⟨f0, .gT hm hg' h0⟩, fun f h => ?_⟩
            cases h with
            | skipD hn _ => exact absurd hm hn
            | gT _ _ h1 =>
              simp only [regionStep, hm, and_self, if_true, hg]; exact emit_agrees args d.body fs [] ks f h1
            | gF _ h1 _ => simp_all
            | gBad _ h1 => simp_all
            | gErr _ h1 => simp_all
    · obtain ⟨⟨f0, h0⟩, ih2⟩ := ih
      refine ⟨⟨f0, .skipD hm h0⟩, fun f h => ?_⟩
      cases h with
      | skipD _ h1 => simp only [regionStep, hm, if_false]; exact ih2 f h1
      | gT h' _ _ => exact absurd h' hm
      | gF h' _ _ => exact absurd h' hm
      | gBad h' _ => exact absurd h' hm
      | gErr h' _ => exact absurd h' hm

end RCE.Emit
