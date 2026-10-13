/-! T28 with recursion. Session types with `mu` binders and de Bruijn
    variables. A protocol `loop` whose `break` continues with `rest` is
    encoded as `mu` of the body with `cont` as `var 0` and `break` as
    `rest`; `RCE.SessionEnc` mechanizes that translation.
    Proved: the dual is compatible and an involution; compatibility is
    preserved by unfolding (substitution); session reduction preserves
    compatibility (subject reduction); and closed compatible endpoints are
    either both done or can take a step (progress). -/
namespace RCE.SessionRec

inductive P where
  | done
  | send : P → P
  | recv : P → P
  | choose : P → P → P
  | offer : P → P → P
  | mu : P → P
  | var : Nat → P

def dual : P → P
  | .done => .done
  | .send k => .recv (dual k)
  | .recv k => .send (dual k)
  | .choose a b => .offer (dual a) (dual b)
  | .offer a b => .choose (dual a) (dual b)
  | .mu k => .mu (dual k)
  | .var n => .var n

inductive Comp : P → P → Prop
  | done : Comp .done .done
  | sr {a b} : Comp a b → Comp (.send a) (.recv b)
  | rs {a b} : Comp a b → Comp (.recv a) (.send b)
  | co {a b c d} : Comp a c → Comp b d → Comp (.choose a b) (.offer c d)
  | oc {a b c d} : Comp a c → Comp b d → Comp (.offer a b) (.choose c d)
  | mu {a b} : Comp a b → Comp (.mu a) (.mu b)
  | var {n} : Comp (.var n) (.var n)

theorem t28r : ∀ p, Comp p (dual p) ∧ dual (dual p) = p
  | .done => ⟨.done, rfl⟩
  | .send k => let ⟨h, e⟩ := t28r k; ⟨.sr h, by simp [dual, e]⟩
  | .recv k => let ⟨h, e⟩ := t28r k; ⟨.rs h, by simp [dual, e]⟩
  | .choose a b => let ⟨h, e⟩ := t28r a; let ⟨h', e'⟩ := t28r b; ⟨.co h h', by simp [dual, e, e']⟩
  | .offer a b => let ⟨h, e⟩ := t28r a; let ⟨h', e'⟩ := t28r b; ⟨.oc h h', by simp [dual, e, e']⟩
  | .mu k => let ⟨h, e⟩ := t28r k; ⟨.mu h, by simp [dual, e]⟩
  | .var _ => ⟨.var, rfl⟩

/-- replace variable `k` by the closed type `c` (under a binder `k` grows;
    `c` is closed where it is used, so no shifting is needed) -/
def subst (k : Nat) (c : P) : P → P
  | .done => .done
  | .send a => .send (subst k c a)
  | .recv a => .recv (subst k c a)
  | .choose a b => .choose (subst k c a) (subst k c b)
  | .offer a b => .offer (subst k c a) (subst k c b)
  | .mu a => .mu (subst (k + 1) c a)
  | .var n => if n = k then c else .var n

def unfold : P → P
  | .mu a => subst 0 (.mu a) a
  | p => p

theorem comp_subst {a b c d : P} (k : Nat) (hab : Comp a b) (hcd : Comp c d) :
    Comp (subst k c a) (subst k d b) := by
  induction hab generalizing k with
  | done => exact .done
  | sr _ ih => exact .sr (ih k)
  | rs _ ih => exact .rs (ih k)
  | co _ _ ih1 ih2 => exact .co (ih1 k) (ih2 k)
  | oc _ _ ih1 ih2 => exact .oc (ih1 k) (ih2 k)
  | mu _ ih => exact .mu (ih (k + 1))
  | var => simp only [subst]; split <;> first | exact hcd | exact .var

theorem comp_unfold {a b : P} (h : Comp a b) : Comp (unfold a) (unfold b) := by
  cases h with
  | mu h' => exact comp_subst 0 h' (.mu h')
  | _ => simp only [unfold]; constructor <;> assumption

/-- one interaction step between the two endpoints -/
inductive Red : P → P → P → P → Prop
  | msg {a b} : Red (.send a) (.recv b) a b
  | msg' {a b} : Red (.recv a) (.send b) a b
  | pickL {a b c d} : Red (.choose a b) (.offer c d) a c
  | pickR {a b c d} : Red (.choose a b) (.offer c d) b d
  | pickL' {a b c d} : Red (.offer a b) (.choose c d) a c
  | pickR' {a b c d} : Red (.offer a b) (.choose c d) b d
  | unf {a b} : Red (.mu a) (.mu b) (subst 0 (.mu a) a) (subst 0 (.mu b) b)

theorem t28r_preserve {a b a' b' : P} (h : Comp a b) (r : Red a b a' b') : Comp a' b' := by
  cases r <;> cases h
  all_goals first
    | assumption
    | (rename_i h; exact comp_subst 0 h (.mu h))

/-- free variables bounded by `n` -/
def closedAt : Nat → P → Prop
  | _, .done => True
  | n, .send a | n, .recv a => closedAt n a
  | n, .choose a b | n, .offer a b => closedAt n a ∧ closedAt n b
  | n, .mu a => closedAt (n + 1) a
  | n, .var m => m < n

theorem t28r_progress {a b : P} (h : Comp a b) (hc : closedAt 0 a) :
    (a = .done ∧ b = .done) ∨ ∃ a' b', Red a b a' b' := by
  cases h with
  | done => exact Or.inl ⟨rfl, rfl⟩
  | sr => exact Or.inr ⟨_, _, .msg⟩
  | rs => exact Or.inr ⟨_, _, .msg'⟩
  | co => exact Or.inr ⟨_, _, .pickL⟩
  | oc => exact Or.inr ⟨_, _, .pickL'⟩
  | mu => exact Or.inr ⟨_, _, .unf⟩
  | var => exact absurd hc (Nat.not_lt_zero _)

theorem closedAt_mono : ∀ {n m : Nat} (p : P), n ≤ m → closedAt n p → closedAt m p
  | _, _, .done, _, h => h
  | _, _, .send a, hl, h | _, _, .recv a, hl, h => closedAt_mono a hl h
  | _, _, .choose a b, hl, h | _, _, .offer a b, hl, h =>
    ⟨closedAt_mono a hl h.1, closedAt_mono b hl h.2⟩
  | _, _, .mu a, hl, h => closedAt_mono a (Nat.succ_le_succ hl) h
  | _, _, .var _, hl, h => Nat.lt_of_lt_of_le h hl

theorem closedAt_subst : ∀ (k : Nat) {c : P} (p : P), closedAt 0 c →
    closedAt (k + 1) p → closedAt k (subst k c p)
  | _, _, .done, _, _ => trivial
  | k, _, .send a, hc, h | k, _, .recv a, hc, h => closedAt_subst k a hc h
  | k, _, .choose a b, hc, h | k, _, .offer a b, hc, h =>
    ⟨closedAt_subst k a hc h.1, closedAt_subst k b hc h.2⟩
  | k, _, .mu a, hc, h => closedAt_subst (k + 1) a hc h
  | k, c, .var m, hc, h => by
    simp only [subst]; split
    · exact closedAt_mono c (Nat.zero_le _) hc
    · simp only [closedAt] at h ⊢; omega

/-- closedness survives a step, so progress holds along every run -/
theorem closed_preserve {a b a' b' : P} (r : Red a b a' b') (hc : closedAt 0 a) :
    closedAt 0 a' := by
  cases r with
  | msg | msg' => exact hc
  | pickL | pickL' => exact hc.1
  | pickR | pickR' => exact hc.2
  | unf => exact closedAt_subst 0 _ hc hc

end RCE.SessionRec
