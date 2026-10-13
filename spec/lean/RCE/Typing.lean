import RCE.Form
/-! T5 Preservation for δ: every value δ writes to state has its field's
    static type (including the integer range). -/
namespace RCE

def WtFields (ts : List Ty) (fs : List Val) : Prop :=
  fs.length = ts.length ∧ ∀ i v t, fs.get? i = some v → ts.get? i = some t → v.hasTy t

inductive WtStmt (Γ : Ctx) : Stmt → Prop where
  | skip : WtStmt Γ .skip
  | assign {i e t} : Γ.fields.get? i = some t → HasTy Γ e t → WtStmt Γ (.assign i e)
  | seq {a b} : WtStmt Γ a → WtStmt Γ b → WtStmt Γ (.seq a b)
  | ifs {c a b} : HasTy Γ c .bool → WtStmt Γ a → WtStmt Γ b → WtStmt Γ (.ifs c a b)
  | loop {n c b} : HasTy Γ c .bool → WtStmt Γ b → WtStmt Γ (.loop n c b)
  | fail {x} : WtStmt Γ (.fail x)
  | emit {e t} : HasTy Γ e t → WtStmt Γ (.emit e)

theorem wtEnv_of {ts as fs args} (hf : WtFields ts fs) (ha : WtFields as args) :
    WtEnv ⟨ts, as⟩ ⟨fs, args⟩ := ⟨hf.2, ha.2⟩

theorem wtFields_set {ts fs i v t} (hf : WtFields ts fs) (hi : ts.get? i = some t) (hv : v.hasTy t) :
    WtFields ts (fs.set i v) := by
  refine ⟨by simp [hf.1], ?_⟩
  intro j w u hj hu
  simp only [List.get?_eq_getElem?, List.getElem?_set] at hj
  split at hj
  · rename_i hij; subst hij
    split at hj
    · cases hj; rw [hi] at hu; cases hu; exact hv
    · cases hj
  · exact hf.2 j w u (by simpa [List.get?_eq_getElem?] using hj) hu

theorem iter_pres (Q : List Val → Prop) (f : List Val → List Val → SRes) (c)
    (hf : ∀ fs os fs' os', Q fs → f fs os = .ok fs' os' → Q fs') :
    ∀ n fs os fs' os', Q fs → iter f c n fs os = .ok fs' os' → Q fs' := by
  intro n; induction n with
  | zero => intro fs os fs' os' hq h; simp only [iter] at h; split at h <;> cases h; exact hq
  | succ k ih =>
    intro fs os fs' os' hq h; simp only [iter] at h
    split at h
    · cases h
    · cases h
    · cases h; exact hq
    · split at h
      · rename_i a b hab; exact ih _ _ _ _ (hf _ _ _ _ hq hab) h
      · simp_all

theorem exec_wt {ts as args} (ha : WtFields as args) (s : Stmt) (hs : WtStmt ⟨ts, as⟩ s) :
    ∀ fs os fs' os', WtFields ts fs → exec args s fs os = .ok fs' os' → WtFields ts fs' := by
  induction hs with
  | skip => intro fs os fs' os' hf h; cases h; exact hf
  | @assign i e t hi he =>
    intro fs os fs' os' hf h; simp only [exec] at h
    split at h
    · rename_i v hv
      split at h
      · cases h; exact wtFields_set hf hi (eval_ty (wtEnv_of hf ha) he hv)
      · cases h
    · cases h
  | seq _ _ iha ihb =>
    intro fs os fs' os' hf h; simp only [exec] at h
    split at h
    · rename_i a b hab; exact ihb _ _ _ _ (iha _ _ _ _ hf hab) h
    · simp_all
  | ifs _ _ _ iha ihb =>
    intro fs os fs' os' hf h; simp only [exec] at h
    split at h
    · exact iha _ _ _ _ hf h
    · exact ihb _ _ _ _ hf h
    · cases h
    · cases h
  | loop _ _ ih =>
    intro fs os fs' os' hf h; simp only [exec] at h
    exact iter_pres (WtFields ts) _ _ (fun a b c d q e => ih a b c d q e) _ _ _ _ _ hf h
  | fail => intro fs os fs' os' _ h; cases h
  | emit _ =>
    intro fs os fs' os' hf h; simp only [exec] at h
    split at h
    · cases h; exact hf
    · cases h

/-- Program signature: field types per region, payload types per event. -/
structure Sig where
  regions : List (List Ty)
  events : String → List Ty

def WtStep (S : Sig) (d : StepDecl) : Prop :=
  ∃ ts, S.regions.get? d.region = some ts ∧
    HasTy ⟨ts, S.events d.event⟩ d.guard .bool ∧ WtStmt ⟨ts, S.events d.event⟩ d.body

def WtProg (S : Sig) (P : Program) : Prop := ∀ d ∈ P.steps, WtStep S d

def WtState (S : Sig) (s : State) : Prop :=
  s.length = S.regions.length ∧
  ∀ k fs ts, s.get? k = some fs → S.regions.get? k = some ts → WtFields ts fs

theorem regionStep_wt (S : Sig) {args ev i fs ts} (ha : WtFields (S.events ev) args)
    (hts : S.regions.get? i = some ts) (hf : WtFields ts fs) :
    ∀ ds : List StepDecl, (∀ d ∈ ds, WtStep S d) → ∀ fs' o,
      regionStep args ev i fs ds = .ok fs' o → WtFields ts fs' := by
  intro ds; induction ds with
  | nil => intro _ fs' o h; simp [regionStep] at h; rw [← h.1]; exact hf
  | cons d ds ih =>
    intro hd fs' o h
    simp only [regionStep] at h
    split at h
    · rename_i hc
      obtain ⟨ts', hr, _, hb⟩ := hd d (List.mem_cons_self _ _)
      rw [hc.1, hts] at hr; cases hr
      rw [hc.2] at hb
      split at h
      · exact exec_wt ha d.body hb _ _ _ _ hf h
      · exact ih (fun d' m => hd d' (List.mem_cons_of_mem _ m)) _ _ h
      · cases h
      · cases h
    · exact ih (fun d' m => hd d' (List.mem_cons_of_mem _ m)) _ _ h

theorem runRegions_wt (S : Sig) (P : Program) (hP : WtProg S P) (e : Event)
    (ha : WtFields (S.events e.1) e.2) :
    ∀ (s : State) (i : Nat) s' o,
      (∀ k fs ts, s.get? k = some fs → S.regions.get? (i + k) = some ts → WtFields ts fs) →
      runRegions P e i s = .ok (s', o) →
      s'.length = s.length ∧
      ∀ k fs ts, s'.get? k = some fs → S.regions.get? (i + k) = some ts → WtFields ts fs := by
  intro s; induction s with
  | nil => intro i s' o _ h; simp [runRegions] at h; obtain ⟨rfl, rfl⟩ := h; simp
  | cons fs rs ih =>
    intro i s' o hw h
    simp only [runRegions] at h
    split at h
    · cases h
    · rename_i fs1 o1 hr
      split at h
      · cases h
      · rename_i rs' os hrest
        cases h
        have hrs : ∀ k fs ts, rs.get? k = some fs → S.regions.get? (i + 1 + k) = some ts → WtFields ts fs := by
          intro k f t hk ht
          exact hw (k+1) f t (by simpa using hk) (by rw [show i + (k+1) = i + 1 + k by omega]; exact ht)
        obtain ⟨hl, hw'⟩ := ih (i+1) rs' os hrs hrest
        refine ⟨by simp [hl], ?_⟩
        intro k f t hk ht
        cases k with
        | zero =>
          simp at hk; subst hk
          simp at ht
          exact regionStep_wt S ha (by simpa [List.get?_eq_getElem?] using ht) (hw 0 fs t (by simp) (by simpa using ht)) P.steps hP _ _ hr
        | succ k =>
          exact hw' k f t (by simpa using hk) (by rw [show i + 1 + k = i + (k+1) by omega]; exact ht)

/-- **T5 Preservation for δ**: from a well-typed state and a well-typed
    event, δ yields a well-typed state. -/
theorem delta_wt (S : Sig) (P : Program) (hP : WtProg S P) (s : State) (e : Event)
    (hs : WtState S s) (ha : WtFields (S.events e.1) e.2) : WtState S (delta P s e).1 := by
  unfold delta
  split
  · rename_i s' o h
    obtain ⟨hl, hw⟩ := runRegions_wt S P hP e ha s 0 s' o (by simpa using hs.2) h
    exact ⟨by rw [hl]; exact hs.1, by simpa using hw⟩
  · exact hs

/-- Preservation along the whole log (with the fold law, T3). -/
theorem run_wt (S : Sig) (P : Program) (hP : WtProg S P) :
    ∀ (es : List Event) (s : State), WtState S s → (∀ e ∈ es, WtFields (S.events e.1) e.2) →
      WtState S (es.foldl (fun s e => (delta P s e).1) s) := by
  intro es; induction es with
  | nil => intro s hs _; exact hs
  | cons e es ih =>
    intro s hs he
    exact ih _ (delta_wt S P hP s e hs (he e (List.mem_cons_self _ _)))
      (fun e' m => he e' (List.mem_cons_of_mem _ m))

end RCE
