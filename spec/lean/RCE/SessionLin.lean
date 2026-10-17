import RCE.SessionRec
/-! Linearity of session endpoints (W-SES-2, W-SES-4).

    An environment gives each endpoint name a state: `live p` (the
    protocol still to run, `RCE.SessionRec.P`) or `gone` (moved, sent or
    closed). `Ty Γ p Δ` is a linear typing judgment for a body `p` over
    endpoint operations: each operation needs its endpoint live and
    advances or consumes it; a `move` hands the endpoint to a new name and
    makes the old one gone; at the join of a branch both arms must agree
    up to dropping a finished endpoint (`Wk`). A body is well typed when
    its final environment is `Ended` (every endpoint gone or done).

    `chk` is the algorithm of `check06b` restricted to this fragment: it
    returns the rule it refuses with (`W-SES-1`, `W-SES-2`, `W-SES-4`).
    Proved: `chk_sound` (an accepted body has a derivation, so W-SES-2
    and W-SES-4 are sound for the judgment) and `ty_safe` (a typed body
    never uses a gone endpoint, never takes an action its protocol does
    not enable, never closes an unfinished endpoint, along every path, and
    ends with every endpoint gone or done).

    Fragment: message steps (send/recv) and `done`; branching is a local
    `if` (both arms possible). Choice/offer labels, loops and recursion
    are covered by the automaton link (`RCE.SessionVectors`), not here. -/
namespace RCE.SessionLin
open RCE.SessionRec

deriving instance DecidableEq for P

inductive Op | send | recv
  deriving DecidableEq

inductive St | live (p : P) | gone
  deriving DecidableEq

abbrev Env := List St

/-- a body: a sequence of endpoint statements; `branch a b k` runs one of
    `a`, `b` and then `k` -/
inductive Prog
  | nil
  | act (x : Nat) (o : Op) (k : Prog)
  | close (x : Nat) (k : Prog)
  | move (x : Nat) (k : Prog)
  | branch (a b k : Prog)

def get : Env → Nat → Option St
  | [], _ => none
  | s :: _, 0 => some s
  | _ :: Γ, n + 1 => get Γ n

def set : Env → Nat → St → Env
  | [], _, _ => []
  | _ :: Γ, 0, t => t :: Γ
  | s :: Γ, n + 1, t => s :: set Γ n t

def stepOp : Op → P → Option P
  | .send, .send k => some k
  | .recv, .recv k => some k
  | _, _ => none

/-- one slot weakens to another: unchanged, or a finished endpoint dropped -/
inductive WkS : St → St → Prop
  | refl (s : St) : WkS s s
  | drop : WkS (.live .done) .gone

inductive Wk : Env → Env → Prop
  | nil : Wk [] []
  | cons {s t Γ Δ} : WkS s t → Wk Γ Δ → Wk (s :: Γ) (t :: Δ)

def endS : St → Bool
  | .gone => true
  | .live .done => true
  | _ => false

def Ended (Δ : Env) : Prop := Δ.all endS = true

/-- the linear typing judgment -/
inductive Ty : Env → Prog → Env → Prop
  | nil {Γ} : Ty Γ .nil Γ
  | act {Γ Δ x o p p' k} : get Γ x = some (.live p) → stepOp o p = some p' →
      Ty (set Γ x (.live p')) k Δ → Ty Γ (.act x o k) Δ
  | close {Γ Δ x k} : get Γ x = some (.live .done) →
      Ty (set Γ x .gone) k Δ → Ty Γ (.close x k) Δ
  | move {Γ Δ x p k} : get Γ x = some (.live p) →
      Ty (set Γ x .gone ++ [.live p]) k Δ → Ty Γ (.move x k) Δ
  | branch {Γ Γa Γb Γj Δ a b k} : Ty Γ a Γa → Ty Γ b Γb → Wk Γa Γj → Wk Γb Γj →
      Ty Γj k Δ → Ty Γ (.branch a b k) Δ

def TyBody (Γ : Env) (p : Prog) : Prop := ∃ Δ, Ty Γ p Δ ∧ Ended Δ

/-! The checker's algorithm -/

def joinS : St → St → Except String St
  | .gone, .gone => .ok .gone
  | .gone, .live q => if q = .done then .ok .gone else .error "W-SES-4"
  | .live p, .gone => if p = .done then .ok .gone else .error "W-SES-4"
  | .live p, .live q => if p = q then .ok (.live p) else .error "W-SES-1"

def join : Env → Env → Except String Env
  | [], [] => .ok []
  | s :: Γ, t :: Δ => match joinS s t, join Γ Δ with
    | .ok u, .ok r => .ok (u :: r)
    | .error e, _ => .error e
    | _, .error e => .error e
  | _, _ => .error "W-SES-1"

def chk : Env → Prog → Except String Env
  | Γ, .nil => .ok Γ
  | Γ, .act x o k => match get Γ x with
    | some (.live p) => match stepOp o p with
      | some p' => chk (set Γ x (.live p')) k
      | none => .error "W-SES-1"
    | some .gone => .error "W-SES-4"
    | none => .error "unbound"
  | Γ, .close x k => match get Γ x with
    | some (.live .done) => chk (set Γ x .gone) k
    | some (.live _) => .error "W-SES-2"
    | some .gone => .error "W-SES-4"
    | none => .error "unbound"
  | Γ, .move x k => match get Γ x with
    | some (.live p) => chk (set Γ x .gone ++ [.live p]) k
    | some .gone => .error "W-SES-4"
    | none => .error "unbound"
  | Γ, .branch a b k => match chk Γ a, chk Γ b with
    | .ok Γa, .ok Γb => match join Γa Γb with
      | .ok Γj => chk Γj k
      | .error e => .error e
    | .error e, _ => .error e
    | _, .error e => .error e

def chkBody (Γ : Env) (p : Prog) : Except String Unit :=
  match chk Γ p with
  | .ok Δ => if Δ.all endS then .ok () else .error "W-SES-2"
  | .error e => .error e

/-! Soundness of the checker -/

theorem joinS_sound {s t u : St} (h : joinS s t = .ok u) : WkS s u ∧ WkS t u := by
  cases s with
  | gone =>
    cases t with
    | gone => simp only [joinS] at h; cases h; exact ⟨.refl _, .refl _⟩
    | live q =>
      simp only [joinS] at h
      split at h
      · rename_i e; subst e; cases h; exact ⟨.refl _, .drop⟩
      · cases h
  | live p =>
    cases t with
    | gone =>
      simp only [joinS] at h
      split at h
      · rename_i e; subst e; cases h; exact ⟨.drop, .refl _⟩
      · cases h
    | live q =>
      simp only [joinS] at h
      split at h
      · rename_i e; subst e; cases h; exact ⟨.refl _, .refl _⟩
      · cases h

theorem join_sound : ∀ {Γ Δ J : Env}, join Γ Δ = .ok J → Wk Γ J ∧ Wk Δ J
  | [], [], J, h => by cases h; exact ⟨.nil, .nil⟩
  | s :: Γ, t :: Δ, J, h => by
    unfold join at h
    split at h
    · rename_i u r hu hr
      cases h
      have ⟨a, b⟩ := joinS_sound hu
      have ⟨c, d⟩ := join_sound hr
      exact ⟨.cons a c, .cons b d⟩
    · cases h
    · cases h
  | [], _ :: _, _, h => by simp [join] at h
  | _ :: _, [], _, h => by simp [join] at h

theorem chk_sound : ∀ {Γ : Env} {p : Prog} {Δ : Env}, chk Γ p = .ok Δ → Ty Γ p Δ
  | Γ, .nil, Δ, h => by cases h; exact .nil
  | Γ, .act x o k, Δ, h => by
    unfold chk at h
    split at h
    · rename_i p hg
      split at h
      · rename_i p' hs; exact .act hg hs (chk_sound h)
      · cases h
    · cases h
    · cases h
  | Γ, .close x k, Δ, h => by
    unfold chk at h
    split at h
    · rename_i hg; exact .close hg (chk_sound h)
    · cases h
    · cases h
    · cases h
  | Γ, .move x k, Δ, h => by
    unfold chk at h
    split at h
    · rename_i p hg; exact .move hg (chk_sound h)
    · cases h
    · cases h
  | Γ, .branch a b k, Δ, h => by
    unfold chk at h
    split at h
    · rename_i Γa Γb ha hb
      split at h
      · rename_i Γj hj
        have ⟨wa, wb⟩ := join_sound hj
        exact .branch (chk_sound ha) (chk_sound hb) wa wb (chk_sound h)
      · cases h
    · cases h
    · cases h

/-- W-SES-2 and W-SES-4 are sound: a body the checker accepts is typed -/
theorem chkBody_sound {Γ : Env} {p : Prog} (h : chkBody Γ p = .ok ()) : TyBody Γ p := by
  unfold chkBody at h
  split at h
  · rename_i Δ hc
    split at h
    · rename_i he; exact ⟨Δ, chk_sound hc, he⟩
    · cases h
  · cases h

/-! Safety: every path of a typed body runs without a linearity fault -/

/-- big-step runs; `none` is a fault (a gone endpoint used, an action the
    protocol does not enable, an unfinished endpoint closed) -/
inductive Run : Env → Prog → Option Env → Prop
  | nil {Γ} : Run Γ .nil (some Γ)
  | act {Γ x o p p' k r} : get Γ x = some (.live p) → stepOp o p = some p' →
      Run (set Γ x (.live p')) k r → Run Γ (.act x o k) r
  | actF {Γ x o k} : (∀ p p', get Γ x = some (.live p) → stepOp o p = some p' → False) →
      Run Γ (.act x o k) none
  | close {Γ x k r} : get Γ x = some (.live .done) → Run (set Γ x .gone) k r → Run Γ (.close x k) r
  | closeF {Γ x k} : get Γ x ≠ some (.live .done) → Run Γ (.close x k) none
  | move {Γ x p k r} : get Γ x = some (.live p) →
      Run (set Γ x .gone ++ [.live p]) k r → Run Γ (.move x k) r
  | moveF {Γ x k} : (∀ p, get Γ x ≠ some (.live p)) → Run Γ (.move x k) none
  | left {Γ Γ' a b k r} : Run Γ a (some Γ') → Run Γ' k r → Run Γ (.branch a b k) r
  | right {Γ Γ' a b k r} : Run Γ b (some Γ') → Run Γ' k r → Run Γ (.branch a b k) r
  | leftF {Γ a b k} : Run Γ a none → Run Γ (.branch a b k) none
  | rightF {Γ a b k} : Run Γ b none → Run Γ (.branch a b k) none

theorem wkS_trans : ∀ {a b c : St}, WkS a b → WkS b c → WkS a c
  | _, _, _, .refl _, h => h
  | _, _, _, .drop, .refl _ => .drop

theorem wk_trans : ∀ {a b c : Env}, Wk a b → Wk b c → Wk a c
  | _, _, _, .nil, .nil => .nil
  | _, _, _, .cons s r, .cons s' r' => .cons (wkS_trans s s') (wk_trans r r')

theorem wk_refl : ∀ Γ : Env, Wk Γ Γ
  | [] => .nil
  | s :: Γ => .cons (.refl s) (wk_refl Γ)

theorem wk_get_live : ∀ {Γ0 Γ : Env} {x : Nat} {p : P}, Wk Γ0 Γ →
    get Γ x = some (.live p) → get Γ0 x = some (.live p)
  | _, _, _, _, .nil, h => by simp [get] at h
  | _, _, 0, _, .cons s _, h => by
    simp [get] at h ⊢; subst h; cases s; rfl
  | _, _, _ + 1, _, .cons _ r, h => by simp [get] at h ⊢; exact wk_get_live r h

theorem wk_set : ∀ {Γ0 Γ : Env} (x : Nat) (t : St), Wk Γ0 Γ → Wk (set Γ0 x t) (set Γ x t)
  | _, _, _, _, .nil => .nil
  | _, _, 0, t, .cons _ r => .cons (.refl t) r
  | _, _, _ + 1, t, .cons s r => .cons s (wk_set _ t r)

theorem wk_snoc : ∀ {Γ0 Γ : Env} (t : St), Wk Γ0 Γ → Wk (Γ0 ++ [t]) (Γ ++ [t])
  | _, _, t, .nil => .cons (.refl t) .nil
  | _, _, t, .cons s r => .cons s (wk_snoc t r)

theorem ty_safe : ∀ {Γ : Env} {p : Prog} {Δ : Env}, Ty Γ p Δ →
    ∀ {Γ0 : Env} {r : Option Env}, Wk Γ0 Γ → Run Γ0 p r → ∃ Δ', r = some Δ' ∧ Wk Δ' Δ := by
  intro Γ p Δ ht
  induction ht with
  | nil => intro Γ0 r w hr; cases hr; exact ⟨_, rfl, w⟩
  | act hg hs _ ih =>
    intro Γ0 r w hr
    have hg0 := wk_get_live w hg
    cases hr with
    | act hg' hs' hk =>
      rw [hg0] at hg'; cases hg'; rw [hs] at hs'; cases hs'
      exact ih (wk_set _ _ w) hk
    | actF hf => exact (hf _ _ hg0 hs).elim
  | close hg _ ih =>
    intro Γ0 r w hr
    have hg0 := wk_get_live w hg
    cases hr with
    | close _ hk => exact ih (wk_set _ _ w) hk
    | closeF hf => exact (hf hg0).elim
  | move hg _ ih =>
    intro Γ0 r w hr
    have hg0 := wk_get_live w hg
    cases hr with
    | move hg' hk =>
      rw [hg0] at hg'; cases hg'
      exact ih (wk_snoc _ (wk_set _ _ w)) hk
    | moveF hf => exact (hf _ hg0).elim
  | branch _ _ wa wb _ iha ihb ihk =>
    intro Γ0 r w hr
    cases hr with
    | left ha hk =>
      have ⟨_, e, w'⟩ := iha w ha; cases e
      exact ihk (wk_trans w' wa) hk
    | right hb hk =>
      have ⟨_, e, w'⟩ := ihb w hb; cases e
      exact ihk (wk_trans w' wb) hk
    | leftF ha => have ⟨_, e, _⟩ := iha w ha; cases e
    | rightF hb => have ⟨_, e, _⟩ := ihb w hb; cases e

theorem ended_wk : ∀ {Δ' Δ : Env}, Wk Δ' Δ → Ended Δ → Ended Δ'
  | _, _, .nil, _ => rfl
  | _, _, .cons s r, h => by
    simp only [Ended, List.all_cons, Bool.and_eq_true] at h ⊢
    refine ⟨?_, ended_wk r h.2⟩
    cases s with
    | refl => exact h.1
    | drop => rfl

/-- a body the checker accepts runs every path without a linearity fault
    and ends with every endpoint gone or done -/
theorem chkBody_safe {Γ : Env} {p : Prog} {r : Option Env} (h : chkBody Γ p = .ok ())
    (hr : Run Γ p r) : ∃ Δ', r = some Δ' ∧ Ended Δ' := by
  have ⟨Δ, ht, he⟩ := chkBody_sound h
  have ⟨Δ', e, w⟩ := ty_safe ht (wk_refl Γ) hr
  exact ⟨Δ', e, ended_wk w he⟩

end RCE.SessionLin
