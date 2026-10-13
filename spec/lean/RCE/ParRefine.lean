import RCE.ParSem
/-! A refinement proof from the small-step semantics of `ParSem` to a Lean
    model of the reference evaluator on core plus `par`, with bounded
    integers handled explicitly.

    The evaluator model `evalB` follows `interp`/`conc::run_main` on this
    fragment: statements in order, `par` in source order (C.4.2), and every
    `+`/`*` result checked against the i64 range, an overflow being a trap
    (`none`). The bounded small-step semantics `StepB` interleaves `par`
    branches like `Step`, but an assignment or test whose expression
    overflows cannot step: the configuration `Traps`.

    Proved, for every W-PAR-1 program: a step preserves the evaluator's
    outcome (`stepB_evalB`), a trapping configuration has outcome `none`
    (`traps_evalB`), every configuration can step, has trapped, or is
    `skip` (`progressB`), and runs are finite. Hence every maximal
    interleaving ends in `skip` with exactly the evaluator's store when the
    evaluator succeeds, and traps when the evaluator traps (`refine_ok`,
    `refine_trap`, `refine`). A successful evaluation also equals the
    unbounded meaning `exec` (`evalB_exec`), which ties this to `par_det`.
    The evaluator model is checked against the Rust evaluator by the
    generated vectors in `ParVectors.lean`, including programs where the
    Rust evaluator traps on overflow. -/
namespace RCE.ParSem

def lo : Int := -9223372036854775808
def hi : Int := 9223372036854775807

def chk (v : Int) : Option Int := if lo ≤ v ∧ v ≤ hi then some v else none

def Ex.evalB (σ : Store) : Ex → Option Int
  | .lit n => chk n
  | .var x => some (σ x)
  | .add a b => do let x ← a.evalB σ; let y ← b.evalB σ; chk (x + y)
  | .mul a b => do let x ← a.evalB σ; let y ← b.evalB σ; chk (x * y)
  | .lt a b => do let x ← a.evalB σ; let y ← b.evalB σ; pure (if x < y then 1 else 0)

/-- the evaluator model: source order, overflow traps -/
def evalB : St → Store → Option Store
  | .skip, σ => some σ
  | .assign x e, σ => (e.evalB σ).map (upd σ x)
  | .seq a b, σ => (evalB a σ).bind (evalB b)
  | .ite c a b, σ => (c.evalB σ).bind (fun v => if v ≠ 0 then evalB a σ else evalB b σ)
  | .par a b, σ => (evalB a σ).bind (evalB b)

inductive StepB : St → Store → St → Store → Prop
  | assign {x e σ v} : e.evalB σ = some v → StepB (.assign x e) σ .skip (upd σ x v)
  | seqL {a a' b σ σ'} : StepB a σ a' σ' → StepB (.seq a b) σ (.seq a' b) σ'
  | seqSkip {b σ} : StepB (.seq .skip b) σ b σ
  | iteT {c a b σ v} : c.evalB σ = some v → v ≠ 0 → StepB (.ite c a b) σ a σ
  | iteF {c a b σ} : c.evalB σ = some 0 → StepB (.ite c a b) σ b σ
  | parL {a a' b σ σ'} : StepB a σ a' σ' → StepB (.par a b) σ (.par a' b) σ'
  | parR {a b b' σ σ'} : StepB b σ b' σ' → StepB (.par a b) σ (.par a b') σ'
  | parSkip {σ} : StepB (.par .skip .skip) σ .skip σ

inductive StepsB : Nat → St → Store → St → Store → Prop
  | refl {s σ} : StepsB 0 s σ s σ
  | head {n s σ s' σ' s'' σ''} : StepB s σ s' σ' → StepsB n s' σ' s'' σ'' →
      StepsB (n + 1) s σ s'' σ''

/-- the next step of some thread overflows -/
inductive Traps : St → Store → Prop
  | assign {x e σ} : e.evalB σ = none → Traps (.assign x e) σ
  | ite {c a b σ} : c.evalB σ = none → Traps (.ite c a b) σ
  | seq {a b σ} : Traps a σ → Traps (.seq a b) σ
  | parL {a b σ} : Traps a σ → Traps (.par a b) σ
  | parR {a b σ} : Traps b σ → Traps (.par a b) σ

/-! ### the bounded evaluator agrees with the unbounded meaning when it succeeds -/

theorem chk_some {v w : Int} (h : chk v = some w) : w = v := by
  unfold chk at h; split at h <;> simp_all

theorem Ex.evalB_eval (σ : Store) : ∀ (e : Ex) {v : Int}, e.evalB σ = some v → v = e.eval σ
  | .lit n, v, h => by simp only [Ex.evalB] at h; exact chk_some h
  | .var x, v, h => by simp only [Ex.evalB, Option.some.injEq] at h; exact h.symm
  | .add a b, v, h | .mul a b, v, h | .lt a b, v, h => by
    simp only [Ex.evalB, Option.bind_eq_bind, Option.bind_eq_some] at h
    obtain ⟨x, hx, y, hy, h⟩ := h
    have ex := Ex.evalB_eval σ a hx
    have ey := Ex.evalB_eval σ b hy
    subst ex ey
    first
      | (simp only [Ex.eval]; exact chk_some h)
      | (simp only [pure, Option.some.injEq] at h; simp only [Ex.eval]; exact h.symm)

theorem evalB_exec : ∀ (s : St) (σ : Store) {σ' : Store}, evalB s σ = some σ' → σ' = exec s σ
  | .skip, σ, σ', h => by simp only [evalB, Option.some.injEq] at h; exact h.symm
  | .assign x e, σ, σ', h => by
    simp only [evalB, Option.map_eq_some'] at h
    obtain ⟨v, hv, e2⟩ := h
    subst e2; simp only [exec]; rw [Ex.evalB_eval σ e hv]
  | .seq a b, σ, σ', h | .par a b, σ, σ', h => by
    simp only [evalB, Option.bind_eq_some] at h
    obtain ⟨τ, ha, hb⟩ := h
    simp only [exec]; rw [evalB_exec b τ hb, evalB_exec a σ ha]
  | .ite c a b, σ, σ', h => by
    simp only [evalB, Option.bind_eq_some] at h
    obtain ⟨v, hv, h⟩ := h
    have ev := Ex.evalB_eval σ c hv
    simp only [exec]; rw [← ev]
    split at h
    · rw [if_pos (by assumption)]; exact evalB_exec a σ h
    · rw [if_neg (by assumption)]; exact evalB_exec b σ h

/-! ### footprints for the bounded evaluator -/

theorem Ex.evalB_agree (σ τ : Store) : ∀ e : Ex, (∀ y, y ∈ e.vars → σ y = τ y) →
    e.evalB σ = e.evalB τ
  | .lit _, _ => rfl
  | .var x, h => by simp only [Ex.evalB]; rw [h x (by simp [Ex.vars])]
  | .add a b, h | .mul a b, h | .lt a b, h => by
    have ha := Ex.evalB_agree σ τ a (fun y hy => h y (by simp [Ex.vars, hy]))
    have hb := Ex.evalB_agree σ τ b (fun y hy => h y (by simp [Ex.vars, hy]))
    simp only [Ex.evalB, ha, hb]

theorem evalB_frame : ∀ (s : St) (σ : Store) {σ' : Store}, evalB s σ = some σ' →
    ∀ x, x ∉ W s → σ' x = σ x
  | .skip, σ, σ', h, x, _ => by simp only [evalB, Option.some.injEq] at h; rw [h]
  | .assign y e, σ, σ', h, x, hx => by
    simp only [evalB, Option.map_eq_some'] at h
    obtain ⟨v, _, e2⟩ := h
    simp [W] at hx; subst e2; simp [upd, hx]
  | .seq a b, σ, σ', h, x, hx | .par a b, σ, σ', h, x, hx => by
    simp only [evalB, Option.bind_eq_some] at h
    obtain ⟨τ, ha, hb⟩ := h
    simp only [W, List.mem_append, not_or] at hx
    rw [evalB_frame b τ hb x hx.2, evalB_frame a σ ha x hx.1]
  | .ite c a b, σ, σ', h, x, hx => by
    simp only [evalB, Option.bind_eq_some] at h
    obtain ⟨v, _, h⟩ := h
    simp only [W, List.mem_append, not_or] at hx
    split at h
    · exact evalB_frame a σ h x hx.1
    · exact evalB_frame b σ h x hx.2

/-- outcomes that agree on a set: both trap, or both succeed agreeing there -/
def OAgree (S : Nat → Prop) : Option Store → Option Store → Prop
  | none, none => True
  | some a, some b => ∀ y, S y → a y = b y
  | _, _ => False

theorem evalB_agree (S : Nat → Prop) : ∀ (s : St) (σ τ : Store), (∀ y, y ∈ FV s → S y) →
    (∀ y, S y → σ y = τ y) → OAgree S (evalB s σ) (evalB s τ)
  | .skip, σ, τ, _, h => h
  | .assign x e, σ, τ, hS, h => by
    have he : e.evalB σ = e.evalB τ :=
      Ex.evalB_agree σ τ e (fun z hz => h z (hS z (by simp [FV, hz])))
    simp only [evalB, he]
    cases e.evalB τ with
    | none => trivial
    | some v =>
      intro y hy; simp only [upd]; split
      · rfl
      · exact h y hy
  | .seq a b, σ, τ, hS, h | .par a b, σ, τ, hS, h => by
    have ha := evalB_agree S a σ τ (fun z hz => hS z (by simp [FV, hz])) h
    simp only [evalB]
    cases h1 : evalB a σ <;> cases h2 : evalB a τ <;> rw [h1, h2] at ha
    · trivial
    · exact ha.elim
    · exact ha.elim
    · exact evalB_agree S b _ _ (fun z hz => hS z (by simp [FV, hz])) ha
  | .ite c a b, σ, τ, hS, h => by
    have hc : c.evalB σ = c.evalB τ :=
      Ex.evalB_agree σ τ c (fun z hz => h z (hS z (by simp [FV, hz])))
    simp only [evalB, hc]
    cases c.evalB τ with
    | none => trivial
    | some v =>
      simp only [Option.some_bind]; split
      · exact evalB_agree S a σ τ (fun z hz => hS z (by simp [FV, hz])) h
      · exact evalB_agree S b σ τ (fun z hz => hS z (by simp [FV, hz])) h

/-- two statements that do not write what the other uses commute, traps included -/
theorem evalB_comm (a b : St) (hab : Disj (W a) (FV b)) (hba : Disj (W b) (FV a)) (σ : Store) :
    (evalB a σ).bind (evalB b) = (evalB b σ).bind (evalB a) := by
  cases ha : evalB a σ with
  | none =>
    cases hb : evalB b σ with
    | none => rfl
    | some σb =>
      have := evalB_agree (· ∈ FV a) a σ σb (fun _ h => h)
        (fun y hy => (evalB_frame b σ hb y (fun h => hba y h hy)).symm)
      rw [ha] at this
      simp only [Option.none_bind, Option.some_bind]
      cases h' : evalB a σb with
      | none => rfl
      | some _ => rw [h'] at this; exact this.elim
  | some σa =>
    have agb := evalB_agree (· ∈ FV b) b σ σa (fun _ h => h)
      (fun y hy => (evalB_frame a σ ha y (fun h => hab y h hy)).symm)
    cases hb : evalB b σ with
    | none =>
      rw [hb] at agb
      simp only [Option.none_bind, Option.some_bind]
      cases h' : evalB b σa with
      | none => rfl
      | some _ => rw [h'] at agb; exact agb.elim
    | some σb =>
      have aga := evalB_agree (· ∈ FV a) a σ σb (fun _ h => h)
        (fun y hy => (evalB_frame b σ hb y (fun h => hba y h hy)).symm)
      rw [hb] at agb; rw [ha] at aga
      simp only [Option.some_bind]
      cases hab' : evalB b σa with
      | none => rw [hab'] at agb; exact agb.elim
      | some σab =>
        cases hba' : evalB a σb with
        | none => rw [hba'] at aga; exact aga.elim
        | some σba =>
          rw [hab'] at agb; rw [hba'] at aga
          congr 1; funext x
          by_cases hxb : x ∈ W b
          · have hxa : x ∉ W a := fun h => hab x h (W_sub_FV b x hxb)
            rw [evalB_frame a σb hba' x hxa]
            exact (agb x (W_sub_FV b x hxb)).symm
          · by_cases hxa : x ∈ W a
            · rw [evalB_frame b σa hab' x hxb]
              exact aga x (W_sub_FV a x hxa)
            · rw [evalB_frame b σa hab' x hxb, evalB_frame a σb hba' x hxa,
                evalB_frame a σ ha x hxa, evalB_frame b σ hb x hxb]

/-! ### the simulation -/

/-- a bounded step is a step of the unbounded semantics -/
theorem stepB_step {s σ s' σ'} (h : StepB s σ s' σ') : Step s σ s' σ' := by
  induction h with
  | assign hv => rename_i x e σ v; rw [Ex.evalB_eval σ e hv]; exact .assign
  | seqL _ ih => exact .seqL ih
  | seqSkip => exact .seqSkip
  | iteT hv h0 => rename_i c a b σ v; exact .iteT (by rw [← Ex.evalB_eval σ c hv]; exact h0)
  | iteF hv => rename_i c a b σ; exact .iteF (by rw [← Ex.evalB_eval σ c hv])
  | parL _ ih => exact .parL ih
  | parR _ ih => exact .parR ih
  | parSkip => exact .parSkip

/-- every bounded step preserves the evaluator's outcome -/
theorem stepB_evalB {s σ s' σ'} (h : StepB s σ s' σ') (hw : WF s) : evalB s' σ' = evalB s σ := by
  induction h with
  | assign hv => simp [evalB, hv]
  | seqL _ ih => simp only [evalB]; rw [ih hw.1]
  | seqSkip => simp [evalB]
  | iteT hv h0 => simp [evalB, hv, h0]
  | iteF hv => simp [evalB, hv]
  | parL _ ih => simp only [evalB]; rw [ih hw.1]
  | parR hs ih =>
    rename_i a b b' σ σ'
    have m := step_mono (stepB_step hs)
    have h1 : Disj (W a) (FV b') := fun x hx hf => hw.2.2.1 x hx (m.2 x hf)
    have h2 : Disj (W b') (FV a) := fun x hx => hw.2.2.2 x (m.1 x hx)
    show (evalB a σ').bind (evalB b') = (evalB a σ).bind (evalB b)
    rw [evalB_comm a b' h1 h2, ih hw.2.1, evalB_comm a b hw.2.2.1 hw.2.2.2]
  | parSkip => simp [evalB]

theorem stepB_wf {s σ s' σ'} (h : StepB s σ s' σ') (hw : WF s) : WF s' :=
  step_wf (stepB_step h) hw

/-- a trapping configuration: the evaluator traps -/
theorem traps_evalB {s σ} (h : Traps s σ) (hw : WF s) : evalB s σ = none := by
  induction h with
  | assign hv => simp [evalB, hv]
  | ite hv => simp [evalB, hv]
  | seq _ ih => simp [evalB, ih hw.1]
  | parL _ ih => simp [evalB, ih hw.1]
  | parR ht ih =>
    rename_i a b σ
    show (evalB a σ).bind (evalB b) = none
    rw [evalB_comm a b hw.2.2.1 hw.2.2.2, ih hw.2.1]; rfl

/-- every configuration is finished, trapped, or can step -/
theorem progressB : ∀ s σ, s = .skip ∨ Traps s σ ∨ ∃ s' σ', StepB s σ s' σ'
  | .skip, _ => Or.inl rfl
  | .assign x e, σ => by
    cases h : e.evalB σ with
    | none => exact Or.inr (Or.inl (.assign h))
    | some v => exact Or.inr (Or.inr ⟨_, _, .assign h⟩)
  | .seq a b, σ => by
    rcases progressB a σ with ha | ha | ⟨a', σ', ha⟩
    · subst ha; exact Or.inr (Or.inr ⟨_, _, .seqSkip⟩)
    · exact Or.inr (Or.inl (.seq ha))
    · exact Or.inr (Or.inr ⟨_, _, .seqL ha⟩)
  | .ite c a b, σ => by
    cases h : c.evalB σ with
    | none => exact Or.inr (Or.inl (.ite h))
    | some v =>
      by_cases h0 : v = 0
      · subst h0; exact Or.inr (Or.inr ⟨_, _, .iteF h⟩)
      · exact Or.inr (Or.inr ⟨_, _, .iteT h h0⟩)
  | .par a b, σ => by
    rcases progressB a σ with ha | ha | ⟨a', σ', ha⟩
    · rcases progressB b σ with hb | hb | ⟨b', σ', hb⟩
      · subst ha; subst hb; exact Or.inr (Or.inr ⟨_, _, .parSkip⟩)
      · exact Or.inr (Or.inl (.parR hb))
      · exact Or.inr (Or.inr ⟨_, _, .parR hb⟩)
    · exact Or.inr (Or.inl (.parL ha))
    · exact Or.inr (Or.inr ⟨_, _, .parL ha⟩)

theorem stepsB_evalB {n s σ s' σ'} (h : StepsB n s σ s' σ') (hw : WF s) :
    evalB s' σ' = evalB s σ ∧ WF s' := by
  induction h with
  | refl => exact ⟨rfl, hw⟩
  | head hs _ ih =>
    have ⟨e, w⟩ := ih (stepB_wf hs hw)
    exact ⟨e.trans (stepB_evalB hs hw), w⟩

/-- runs are finite: no run is longer than the program -/
theorem stepsB_bound {n s σ s' σ'} (h : StepsB n s σ s' σ') : n + size s' ≤ size s := by
  induction h with
  | refl => omega
  | head hs _ ih => have := step_size (stepB_step hs); omega

/-- **refinement, success**: every complete interleaving ends with the store
    the evaluator computes -/
theorem refine_ok {n s σ σ'} (h : StepsB n s σ .skip σ') (hw : WF s) : evalB s σ = some σ' := by
  have := (stepsB_evalB h hw).1
  simp only [evalB] at this; exact this.symm

/-- **refinement, trap**: an interleaving that reaches an overflow means the
    evaluator traps -/
theorem refine_trap {n s σ s' σ'} (h : StepsB n s σ s' σ') (ht : Traps s' σ') (hw : WF s) :
    evalB s σ = none := by
  have ⟨e, w⟩ := stepsB_evalB h hw
  rw [← e]; exact traps_evalB ht w

/-- a maximal run: it cannot step any further -/
def Maximal (s : St) (σ : Store) : Prop := ¬ ∃ s' σ', StepB s σ s' σ'

/-- **refinement**: for a W-PAR-1 program, every maximal interleaving
    either ends in `skip` with the evaluator's store, when the evaluator
    succeeds, or stops at an overflow trap, when the evaluator traps; it
    never gets stuck otherwise. -/
theorem refine {n s σ s' σ'} (h : StepsB n s σ s' σ') (hm : Maximal s' σ') (hw : WF s) :
    (s' = .skip ∧ evalB s σ = some σ') ∨ (Traps s' σ' ∧ evalB s σ = none) := by
  rcases progressB s' σ' with hs | ht | hst
  · subst hs; exact Or.inl ⟨rfl, refine_ok h hw⟩
  · exact Or.inr ⟨ht, refine_trap h ht hw⟩
  · exact absurd hst hm

/-- and when the evaluator succeeds its store is the unbounded meaning, so
    the bounded runs agree with `par_det` -/
theorem refine_exec {n s σ σ'} (h : StepsB n s σ .skip σ') (hw : WF s) : σ' = exec s σ :=
  evalB_exec s σ (refine_ok h hw)

end RCE.ParSem
