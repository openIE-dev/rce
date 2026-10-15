/-! A small-step operational semantics for a core of RCE with `par`
    (Appendix C, book/14-draft06-concurrency.md C.3, C.4), and the theorems about `par` proved on
    it rather than on a separate model:
    * T16 (`par_det`): under W-PAR-1 every complete interleaving gives the
      source-order (big-step) result.
    * T17 (`diamond`): steps of two branches of a well-formed `par` commute
      (strong local confluence), the formal content of race freedom.
    * T19(a) (`progress`, `steps_bound`): a non-`skip` configuration always
      steps, and every run is shorter than the program, so `par` neither
      deadlocks nor diverges.
    * T20 (`steps_count`, `work_par`): every complete run has the same
      length, a function of program and store, and the work of `par` is
      the sum of the branches plus a fork step.
    * T23 (`group_det`): a workgroup (phases separated by barriers, lanes
      in each phase) gives the phase-then-lane sequential result.
    The core: integer variables, assignment, sequence, `if`, binary `par`.
    Footprints are variable sets (the C.3 paths restricted to roots); loops,
    calls, data structures and traps are not in this core. -/
namespace RCE.ParSem

abbrev Store := Nat → Int

inductive Ex where
  | lit (n : Int)
  | var (x : Nat)
  | add (a b : Ex)
  | mul (a b : Ex)
  | lt (a b : Ex)

def Ex.eval (σ : Store) : Ex → Int
  | .lit n => n
  | .var x => σ x
  | .add a b => a.eval σ + b.eval σ
  | .mul a b => a.eval σ * b.eval σ
  | .lt a b => if a.eval σ < b.eval σ then 1 else 0

def Ex.vars : Ex → List Nat
  | .lit _ => []
  | .var x => [x]
  | .add a b | .mul a b | .lt a b => a.vars ++ b.vars

inductive St where
  | skip
  | assign (x : Nat) (e : Ex)
  | seq (a b : St)
  | ite (c : Ex) (a b : St)
  | par (a b : St)

/-- write footprint W (C.3) -/
def W : St → List Nat
  | .skip => []
  | .assign x _ => [x]
  | .seq a b | .ite _ a b | .par a b => W a ++ W b

/-- all variables used, R ∪ W -/
def FV : St → List Nat
  | .skip => []
  | .assign x e => x :: e.vars
  | .seq a b | .par a b => FV a ++ FV b
  | .ite c a b => c.vars ++ (FV a ++ FV b)

def upd (σ : Store) (x : Nat) (v : Int) : Store := fun y => if y = x then v else σ y

/-- the source-order meaning (C.4, T16): `par a b` is `a; b` -/
def exec : St → Store → Store
  | .skip, σ => σ
  | .assign x e, σ => upd σ x (e.eval σ)
  | .seq a b, σ => exec b (exec a σ)
  | .ite c a b, σ => if c.eval σ ≠ 0 then exec a σ else exec b σ
  | .par a b, σ => exec b (exec a σ)

def Disj (l m : List Nat) : Prop := ∀ x, x ∈ l → x ∉ m

/-- W-PAR-1 at every `par`: a branch writes nothing the other uses -/
def WF : St → Prop
  | .skip | .assign _ _ => True
  | .seq a b | .ite _ a b => WF a ∧ WF b
  | .par a b => WF a ∧ WF b ∧ Disj (W a) (FV b) ∧ Disj (W b) (FV a)

/-- small steps; `par` interleaves the steps of its branches -/
inductive Step : St → Store → St → Store → Prop
  | assign {x e σ} : Step (.assign x e) σ .skip (upd σ x (e.eval σ))
  | seqL {a a' b σ σ'} : Step a σ a' σ' → Step (.seq a b) σ (.seq a' b) σ'
  | seqSkip {b σ} : Step (.seq .skip b) σ b σ
  | iteT {c a b σ} : c.eval σ ≠ 0 → Step (.ite c a b) σ a σ
  | iteF {c a b σ} : c.eval σ = 0 → Step (.ite c a b) σ b σ
  | parL {a a' b σ σ'} : Step a σ a' σ' → Step (.par a b) σ (.par a' b) σ'
  | parR {a b b' σ σ'} : Step b σ b' σ' → Step (.par a b) σ (.par a b') σ'
  | parSkip {σ} : Step (.par .skip .skip) σ .skip σ

/-- runs with their length -/
inductive Steps : Nat → St → Store → St → Store → Prop
  | refl {s σ} : Steps 0 s σ s σ
  | head {n s σ s' σ' s'' σ''} : Step s σ s' σ' → Steps n s' σ' s'' σ'' → Steps (n + 1) s σ s'' σ''

/-! ### footprint lemmas -/

theorem W_sub_FV : ∀ s x, x ∈ W s → x ∈ FV s
  | .skip, _, h => by simp [W] at h
  | .assign y e, x, h => by simp [W] at h; simp [FV, h]
  | .seq a b, x, h | .par a b, x, h => by
    simp only [W, FV, List.mem_append] at h ⊢
    rcases h with h | h
    · exact Or.inl (W_sub_FV a x h)
    · exact Or.inr (W_sub_FV b x h)
  | .ite c a b, x, h => by
    simp only [W, FV, List.mem_append] at h ⊢
    rcases h with h | h
    · exact Or.inr (Or.inl (W_sub_FV a x h))
    · exact Or.inr (Or.inr (W_sub_FV b x h))

theorem eval_agree (σ τ : Store) : ∀ e : Ex, (∀ y, y ∈ e.vars → σ y = τ y) → e.eval σ = e.eval τ
  | .lit _, _ => rfl
  | .var x, h => h x (by simp [Ex.vars])
  | .add a b, h | .mul a b, h | .lt a b, h => by
    have ha := eval_agree σ τ a (fun y hy => h y (by simp [Ex.vars, hy]))
    have hb := eval_agree σ τ b (fun y hy => h y (by simp [Ex.vars, hy]))
    simp [Ex.eval, ha, hb]

theorem exec_frame : ∀ s σ x, x ∉ W s → exec s σ x = σ x
  | .skip, _, _, _ => rfl
  | .assign y e, σ, x, h => by
    simp [W] at h; simp [exec, upd, h]
  | .seq a b, σ, x, h | .par a b, σ, x, h => by
    simp only [W, List.mem_append, not_or] at h
    simp only [exec]; rw [exec_frame b _ x h.2, exec_frame a _ x h.1]
  | .ite c a b, σ, x, h => by
    simp only [W, List.mem_append, not_or] at h
    simp only [exec]; split
    · exact exec_frame a σ x h.1
    · exact exec_frame b σ x h.2

/-- the result on a set S ⊇ FV s depends only on the store on S -/
theorem exec_agree (S : Nat → Prop) : ∀ s σ τ, (∀ y, y ∈ FV s → S y) →
    (∀ y, S y → σ y = τ y) → ∀ y, S y → exec s σ y = exec s τ y
  | .skip, _, _, _, h, y, hy => h y hy
  | .assign x e, σ, τ, hS, h, y, hy => by
    have he : e.eval σ = e.eval τ :=
      eval_agree σ τ e (fun z hz => h z (hS z (by simp [FV, hz])))
    simp only [exec, upd]; split
    · exact he
    · exact h y hy
  | .seq a b, σ, τ, hS, h, y, hy | .par a b, σ, τ, hS, h, y, hy => by
    have ha := exec_agree S a σ τ (fun z hz => hS z (by simp [FV, hz])) h
    exact exec_agree S b _ _ (fun z hz => hS z (by simp [FV, hz])) ha y hy
  | .ite c a b, σ, τ, hS, h, y, hy => by
    have hc : c.eval σ = c.eval τ :=
      eval_agree σ τ c (fun z hz => h z (hS z (by simp [FV, hz])))
    simp only [exec, hc]; split
    · exact exec_agree S a σ τ (fun z hz => hS z (by simp [FV, hz])) h y hy
    · exact exec_agree S b σ τ (fun z hz => hS z (by simp [FV, hz])) h y hy

/-- two statements that do not write what the other uses commute -/
theorem exec_comm (a b : St) (hab : Disj (W a) (FV b)) (hba : Disj (W b) (FV a)) (σ : Store) :
    exec b (exec a σ) = exec a (exec b σ) := by
  funext x
  by_cases hb : x ∈ W b
  · have hxa : x ∉ W a := fun h => hab x h (W_sub_FV b x hb)
    rw [exec_frame a _ x hxa]
    exact exec_agree (· ∈ FV b) b _ _ (fun _ h => h)
      (fun y hy => exec_frame a σ y (fun h => hab y h hy)) x (W_sub_FV b x hb)
  · by_cases ha : x ∈ W a
    · rw [exec_frame b _ x hb]
      exact (exec_agree (· ∈ FV a) a _ _ (fun _ h => h)
        (fun y hy => exec_frame b σ y (fun h => hba y h hy)) x (W_sub_FV a x ha)).symm
    · rw [exec_frame b _ x hb, exec_frame a _ x ha, exec_frame a _ x ha, exec_frame b _ x hb]

/-! ### steps shrink footprints and keep W-PAR-1 -/

theorem step_mono {s σ s' σ'} (h : Step s σ s' σ') :
    (∀ x, x ∈ W s' → x ∈ W s) ∧ (∀ x, x ∈ FV s' → x ∈ FV s) := by
  induction h with
  | assign => exact ⟨fun _ h => by simp [W] at h, fun _ h => by simp [FV] at h⟩
  | seqL _ ih =>
    exact ⟨fun x h => by
      simp only [W, List.mem_append] at h ⊢; exact h.imp (ih.1 x) id,
      fun x h => by simp only [FV, List.mem_append] at h ⊢; exact h.imp (ih.2 x) id⟩
  | parL _ ih =>
    exact ⟨fun x h => by
      simp only [W, List.mem_append] at h ⊢; exact h.imp (ih.1 x) id,
      fun x h => by simp only [FV, List.mem_append] at h ⊢; exact h.imp (ih.2 x) id⟩
  | parR _ ih =>
    exact ⟨fun x h => by
      simp only [W, List.mem_append] at h ⊢; exact h.imp id (ih.1 x),
      fun x h => by simp only [FV, List.mem_append] at h ⊢; exact h.imp id (ih.2 x)⟩
  | seqSkip =>
    exact ⟨fun x h => by simp [W, h], fun x h => by simp [FV, h]⟩
  | iteT _ =>
    exact ⟨fun x h => by simp [W, h], fun x h => by simp [FV, h]⟩
  | iteF _ =>
    exact ⟨fun x h => by simp [W, h], fun x h => by simp [FV, h]⟩
  | parSkip => exact ⟨fun _ h => by simp [W] at h, fun _ h => by simp [FV] at h⟩

theorem step_wf {s σ s' σ'} (h : Step s σ s' σ') (hw : WF s) : WF s' := by
  induction h with
  | assign => trivial
  | seqL _ ih => exact ⟨ih hw.1, hw.2⟩
  | seqSkip => exact hw.2
  | iteT _ => exact hw.1
  | iteF _ => exact hw.2
  | parL hs ih =>
    have m := step_mono hs
    exact ⟨ih hw.1, hw.2.1, fun x hx => hw.2.2.1 x (m.1 x hx),
      fun x hx hf => hw.2.2.2 x hx (m.2 x hf)⟩
  | parR hs ih =>
    have m := step_mono hs
    exact ⟨hw.1, ih hw.2.1, fun x hx hf => hw.2.2.1 x hx (m.2 x hf),
      fun x hx => hw.2.2.2 x (m.1 x hx)⟩
  | parSkip => trivial

/-- every step preserves the source-order meaning -/
theorem step_exec {s σ s' σ'} (h : Step s σ s' σ') (hw : WF s) : exec s' σ' = exec s σ := by
  induction h with
  | assign => rfl
  | seqL _ ih => simp only [exec]; rw [ih hw.1]
  | seqSkip => rfl
  | iteT hc => simp [exec, hc]
  | iteF hc => simp [exec, hc]
  | parL _ ih => simp only [exec]; rw [ih hw.1]
  | parR hs ih =>
    rename_i a b b' σ σ'
    have m := step_mono hs
    have h1 : Disj (W a) (FV b') := fun x hx hf => hw.2.2.1 x hx (m.2 x hf)
    have h2 : Disj (W b') (FV a) := fun x hx => hw.2.2.2 x (m.1 x hx)
    show exec b' (exec a σ') = exec b (exec a σ)
    rw [exec_comm a b' h1 h2, ih hw.2.1, exec_comm a b hw.2.2.1 hw.2.2.2]
  | parSkip => rfl

/-- **T16**: under W-PAR-1, every complete run (any interleaving of the
    branches) ends in the source-order result. -/
theorem par_det {n s σ σ'} (h : Steps n s σ .skip σ') (hw : WF s) : σ' = exec s σ := by
  generalize hk : St.skip = k at h
  induction h with
  | refl => subst hk; rfl
  | head hs _ ih => rw [ih (step_wf hs hw) hk, step_exec hs hw]

/-! ### T19(a): progress and termination -/

theorem progress : ∀ s σ, s ≠ .skip → ∃ s' σ', Step s σ s' σ'
  | .skip, _, h => absurd rfl h
  | .assign _ _, _, _ => ⟨_, _, .assign⟩
  | .seq a b, σ, _ => by
    by_cases ha : a = .skip
    · subst ha; exact ⟨_, _, .seqSkip⟩
    · obtain ⟨a', σ', h⟩ := progress a σ ha; exact ⟨_, _, .seqL h⟩
  | .ite c a b, σ, _ => by
    by_cases hc : c.eval σ = 0
    · exact ⟨_, _, .iteF hc⟩
    · exact ⟨_, _, .iteT hc⟩
  | .par a b, σ, _ => by
    by_cases ha : a = .skip
    · by_cases hb : b = .skip
      · subst ha; subst hb; exact ⟨_, _, .parSkip⟩
      · obtain ⟨b', σ', h⟩ := progress b σ hb; exact ⟨_, _, .parR h⟩
    · obtain ⟨a', σ', h⟩ := progress a σ ha; exact ⟨_, _, .parL h⟩

def size : St → Nat
  | .skip => 1
  | .assign _ _ => 2
  | .seq a b | .ite _ a b | .par a b => size a + size b + 1

theorem step_size {s σ s' σ'} (h : Step s σ s' σ') : size s' < size s := by
  induction h <;> simp only [size] at * <;> omega

/-- **T19(a)**: every run is shorter than the program (no divergence), and
    with `progress` a run that cannot continue has reached `skip`. -/
theorem steps_bound {n s σ s' σ'} (h : Steps n s σ s' σ') : n + size s' ≤ size s := by
  induction h with
  | refl => omega
  | head hs _ ih => have := step_size hs; omega

/-! ### T20: the work of every complete run -/

/-- number of steps of a complete run (`if` counts the branch taken) -/
def work : St → Store → Nat
  | .skip, _ => 0
  | .assign _ _, _ => 1
  | .seq a b, σ => work a σ + work b (exec a σ) + 1
  | .ite c a b, σ => (if c.eval σ ≠ 0 then work a σ else work b σ) + 1
  | .par a b, σ => work a σ + work b (exec a σ) + 1

theorem work_agree : ∀ s σ τ, (∀ y, y ∈ FV s → σ y = τ y) → work s σ = work s τ
  | .skip, _, _, _ | .assign _ _, _, _, _ => rfl
  | .seq a b, σ, τ, h | .par a b, σ, τ, h => by
    have ha : ∀ y, y ∈ FV a → σ y = τ y := fun y hy => h y (by simp [FV, hy])
    have hb := exec_agree (fun y => y ∈ FV a ∨ y ∈ FV b) a σ τ (fun y hy => Or.inl hy)
      (fun y hy => h y (by simp [FV]; exact hy))
    simp only [work]
    rw [work_agree a σ τ ha, work_agree b _ _ (fun y hy => hb y (Or.inr hy))]
  | .ite c a b, σ, τ, h => by
    have hc : c.eval σ = c.eval τ := eval_agree σ τ c (fun z hz => h z (by simp [FV, hz]))
    simp only [work, hc]; split
    · rw [work_agree a σ τ (fun y hy => h y (by simp [FV, hy]))]
    · rw [work_agree b σ τ (fun y hy => h y (by simp [FV, hy]))]

/-- a step leaves the store unchanged outside the write footprint -/
theorem step_frame {s σ s' σ'} (h : Step s σ s' σ') : ∀ x, x ∉ W s → σ' x = σ x := by
  induction h with
  | assign => intro x hx; simp [W] at hx; simp [upd, hx]
  | seqL _ ih | parL _ ih =>
    intro x hx; simp only [W, List.mem_append, not_or] at hx; exact ih x hx.1
  | parR _ ih =>
    intro x hx; simp only [W, List.mem_append, not_or] at hx; exact ih x hx.2
  | seqSkip | iteT _ | iteF _ | parSkip => intro _ _; rfl

theorem step_work {s σ s' σ'} (h : Step s σ s' σ') (hw : WF s) : work s' σ' + 1 = work s σ := by
  induction h with
  | assign => rfl
  | seqL hs ih => simp only [work]; rw [step_exec hs hw.1]; have := ih hw.1; omega
  | seqSkip => simp [work, exec]
  | iteT hc => simp [work, hc]
  | iteF hc => simp [work, hc]
  | parL hs ih => simp only [work]; rw [step_exec hs hw.1]; have := ih hw.1; omega
  | parR hs ih =>
    rename_i a b b' σ σ'
    have m := step_mono hs
    have ha : work a σ' = work a σ := work_agree a σ' σ (fun y hy =>
      step_frame hs y (fun h => hw.2.2.2 y h hy))
    have e1 : work b' (exec a σ') = work b' σ' := work_agree b' _ _ (fun y hy =>
      exec_frame a σ' y (fun h => hw.2.2.1 y h (m.2 y hy)))
    have e2 : work b (exec a σ) = work b σ := work_agree b _ _ (fun y hy =>
      exec_frame a σ y (fun h => hw.2.2.1 y h hy))
    simp only [work]; rw [ha, e1, e2]; have := ih hw.2.1; omega
  | parSkip => rfl

/-- **T20**: every complete run of a well-formed program has the same
    length, `work s σ`, whatever the interleaving. -/
theorem steps_count {n s σ σ'} (h : Steps n s σ .skip σ') (hw : WF s) : n = work s σ := by
  generalize hk : St.skip = k at h
  induction h with
  | refl => subst hk; rfl
  | head hs _ ih => have := step_work hs hw; rw [ih (step_wf hs hw) hk]; omega

/-- the work law of `par` (C.6, T20): the branches' work plus one fork step;
    under W-PAR-1 the second branch's work does not depend on the first -/
theorem work_par (a b : St) (σ : Store) (hw : WF (.par a b)) :
    work (.par a b) σ = work a σ + work b σ + 1 := by
  have e : work b (exec a σ) = work b σ := work_agree b _ _ (fun y hy =>
    exec_frame a σ y (fun h => hw.2.2.1 y h hy))
  simp only [work]; rw [e]

/-! ### T17: steps of different branches commute -/

/-- a step depends only on the store on FV s -/
theorem step_transport {s σ s' σ'} (h : Step s σ s' σ') : ∀ τ, (∀ y, y ∈ FV s → σ y = τ y) →
    ∃ τ', Step s τ s' τ' ∧ (∀ y, y ∈ FV s → σ' y = τ' y) ∧ (∀ y, y ∉ W s → τ' y = τ y) := by
  induction h with
  | assign =>
    rename_i x e σ
    intro τ hτ
    have he : e.eval σ = e.eval τ := eval_agree σ τ e (fun z hz => hτ z (by simp [FV, hz]))
    refine ⟨upd τ x (e.eval τ), .assign, fun y hy => ?_, fun y hy => ?_⟩
    · simp only [upd]; split
      · exact he
      · exact hτ y hy
    · simp [W] at hy; simp [upd, hy]
  | seqL hs ih =>
    rename_i a a' b σ σ'
    intro τ hτ
    obtain ⟨τ', st, ag, fr⟩ := ih τ (fun y hy => hτ y (by simp [FV, hy]))
    refine ⟨τ', .seqL st, fun y hy => ?_, fun y hy => ?_⟩
    · by_cases hya : y ∈ FV a
      · exact ag y hya
      · have hw : y ∉ W a := fun h => hya (W_sub_FV a y h)
        rw [step_frame hs y hw, fr y hw]; exact hτ y hy
    · simp only [W, List.mem_append, not_or] at hy; exact fr y hy.1
  | parL hs ih =>
    rename_i a a' b σ σ'
    intro τ hτ
    obtain ⟨τ', st, ag, fr⟩ := ih τ (fun y hy => hτ y (by simp [FV, hy]))
    refine ⟨τ', .parL st, fun y hy => ?_, fun y hy => ?_⟩
    · by_cases hya : y ∈ FV a
      · exact ag y hya
      · have hw : y ∉ W a := fun h => hya (W_sub_FV a y h)
        rw [step_frame hs y hw, fr y hw]; exact hτ y hy
    · simp only [W, List.mem_append, not_or] at hy; exact fr y hy.1
  | parR hs ih =>
    rename_i a b b' σ σ'
    intro τ hτ
    obtain ⟨τ', st, ag, fr⟩ := ih τ (fun y hy => hτ y (by simp [FV, hy]))
    refine ⟨τ', .parR st, fun y hy => ?_, fun y hy => ?_⟩
    · by_cases hyb : y ∈ FV b
      · exact ag y hyb
      · have hw : y ∉ W b := fun h => hyb (W_sub_FV b y h)
        rw [step_frame hs y hw, fr y hw]; exact hτ y hy
    · simp only [W, List.mem_append, not_or] at hy; exact fr y hy.2
  | seqSkip => intro τ hτ; exact ⟨τ, .seqSkip, hτ, fun _ _ => rfl⟩
  | iteT hc =>
    rename_i c a b σ
    intro τ hτ
    have e : c.eval σ = c.eval τ := eval_agree σ τ c (fun z hz => hτ z (by simp [FV, hz]))
    exact ⟨τ, .iteT (e ▸ hc), hτ, fun _ _ => rfl⟩
  | iteF hc =>
    rename_i c a b σ
    intro τ hτ
    have e : c.eval σ = c.eval τ := eval_agree σ τ c (fun z hz => hτ z (by simp [FV, hz]))
    exact ⟨τ, .iteF (e ▸ hc), hτ, fun _ _ => rfl⟩
  | parSkip => intro τ hτ; exact ⟨τ, .parSkip, hτ, fun _ _ => rfl⟩

/-- **T17**: in a well-formed `par`, a step of the left branch and a step of
    the right branch, both enabled in one configuration, commute: doing them
    in either order reaches the same configuration. No location is written
    by one and used by the other. -/
theorem diamond {a a' b b' σ σ1 σ2} (hw : WF (.par a b))
    (ha : Step a σ a' σ1) (hb : Step b σ b' σ2) :
    ∃ σ3, Step (.par a' b) σ1 (.par a' b') σ3 ∧ Step (.par a b') σ2 (.par a' b') σ3 := by
  obtain ⟨τ, sb, agb, frb⟩ := step_transport hb σ1 (fun y hy =>
    (step_frame ha y (fun h => hw.2.2.1 y h hy)).symm)
  obtain ⟨ρ, sa, aga, fra⟩ := step_transport ha σ2 (fun y hy =>
    (step_frame hb y (fun h => hw.2.2.2 y h hy)).symm)
  have e : τ = ρ := by
    funext y
    by_cases hyb : y ∈ W b
    · have hya : y ∉ W a := fun h => hw.2.2.1 y h (W_sub_FV b y hyb)
      rw [fra y hya]; exact (agb y (W_sub_FV b y hyb)).symm
    · by_cases hya : y ∈ W a
      · rw [frb y hyb]; exact aga y (W_sub_FV a y hya)
      · rw [frb y hyb, fra y hya, step_frame ha y hya, step_frame hb y hyb]
  exact ⟨τ, .parR sb, e ▸ .parL sa⟩

/-! ### T23: workgroups -/

/-- the lanes of one phase as nested `par` (lane order = source order) -/
def lanes : List St → St
  | [] => .skip
  | l :: ls => .par l (lanes ls)

/-- phases separated by barriers -/
def group : List (List St) → St
  | [] => .skip
  | p :: ps => .seq (lanes p) (group ps)

/-- the phase-then-lane sequential order -/
def phaseSeq : List (List St) → Store → Store
  | [], σ => σ
  | p :: ps, σ => phaseSeq ps (p.foldl (fun τ l => exec l τ) σ)

theorem exec_lanes : ∀ (p : List St) σ, exec (lanes p) σ = p.foldl (fun τ l => exec l τ) σ
  | [], _ => rfl
  | l :: ls, σ => by simp only [lanes, exec, List.foldl]; exact exec_lanes ls (exec l σ)

theorem exec_group : ∀ (ps : List (List St)) σ, exec (group ps) σ = phaseSeq ps σ
  | [], _ => rfl
  | p :: ps, σ => by simp only [group, exec, phaseSeq]; rw [exec_lanes]; exact exec_group ps _

/-- **T23**: if every phase satisfies W-PAR-1 between its lanes (W-GRP-1 to
    W-GRP-3), every schedule of the lanes, with barriers between phases,
    gives the phase-then-lane sequential result. -/
theorem group_det {n ps σ σ'} (hw : WF (group ps)) (h : Steps n (group ps) σ .skip σ') :
    σ' = phaseSeq ps σ := by
  rw [par_det h hw, exec_group]

end RCE.ParSem
