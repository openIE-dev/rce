import RCE.SessionPrio
/-! T28g: multiparty global types with projection (Honda, Yoshida and
    Carbone, POPL 2008), synchronous.

    A global type says who sends to whom and in which order: `msg p q k`
    (role `p` sends role `q` a message, then `k`) and `br p q a b` (`p`
    chooses `a` or `b` and tells `q`). `proj G r` is the local type of role
    `r`: a send or select for the sender, a receive or branch for the
    receiver, and for every other role the continuation, where at a branch
    the two continuations must project to the same local type (plain merge,
    HYC Def. 4.1); otherwise the projection is undefined. A configuration
    gives each role a local type, and two roles step together when one
    sends (or selects) to the other and the other receives (or branches)
    from it.

    `sr`: a step of a configuration projected from `G` lands in a
    configuration projected from some `G'` (session fidelity in the
    synchronous setting; the step may be a later message of `G` between
    roles not involved in the earlier ones, as in HYC's reduction of global
    types up to independent prefixes). `progress`: a projected configuration
    is finished or steps. `deadlock_free`: every configuration reachable
    from a projection of a well-formed `G` is finished or steps.

    `global_prio` relates global types to priorities (Padovani, CSL-LICS
    2014, App. C, where the event order of a global type gives the
    priorities): for a global type of messages only, give the `i`-th
    message its own channel `i` and priority `i`; the projections are
    increasing in priority, so the session satisfies `SessionPrio.Inv`, and
    it composes with any other channels by `SessionPrio.deadlock_free`. -/
namespace RCE.SessionGlobal

inductive G where
  | done
  | msg (p q : Nat) (k : G)
  | br (p q : Nat) (a b : G)
  deriving Repr

inductive L where
  | done
  | send (q : Nat) (k : L)
  | recv (p : Nat) (k : L)
  | sel (q : Nat) (a b : L)
  | bra (p : Nat) (a b : L)
  deriving DecidableEq, Repr

def proj : G → Nat → Option L
  | .done, _ => some .done
  | .msg p q k, r =>
    match proj k r with
    | some x => if r = p then some (.send q x) else if r = q then some (.recv p x) else some x
    | none => none
  | .br p q a b, r =>
    match proj a r, proj b r with
    | some x, some y =>
      if r = p then some (.sel q x y) else if r = q then some (.bra p x y)
      else if x = y then some x else none
    | _, _ => none

/-- no role talks to itself -/
def WF : G → Prop
  | .done => True
  | .msg p q k => p ≠ q ∧ WF k
  | .br p q a b => p ≠ q ∧ WF a ∧ WF b

abbrev C := Nat → L

def Proj (g : G) (c : C) : Prop := ∀ r, proj g r = some (c r)

def upd (c : C) (p : Nat) (l : L) : C := fun i => if i = p then l else c i

/-- what an interaction leaves to sender and receiver (`d` picks the
    branch of a selection) -/
def fire (d : Bool) (p q : Nat) : L → L → Option (L × L)
  | .send q' k1, .recv p' k2 => if q' = q ∧ p' = p then some (k1, k2) else none
  | .sel q' a1 b1, .bra p' a2 b2 => if q' = q ∧ p' = p then (if d then some (a1, a2) else some (b1, b2)) else none
  | _, _ => none

inductive Step (c : C) : C → Prop
  | go {d : Bool} {p q : Nat} {k1 k2 : L} : p ≠ q → fire d p q (c p) (c q) = some (k1, k2) →
      Step c (upd (upd c p k1) q k2)

def Final (c : C) : Prop := ∀ r, c r = .done

inductive Reach (c0 : C) : C → Prop
  | refl : Reach c0 c0
  | step {c c'} : Reach c0 c → Step c c' → Reach c0 c'

theorem proj_msg {p q r : Nat} {k : G} {l : L} :
    proj (.msg p q k) r = some l ↔
      ∃ x, proj k r = some x ∧ l = (if r = p then .send q x else if r = q then .recv p x else x) := by
  simp only [proj]
  cases proj k r with
  | none => simp
  | some x =>
    by_cases h1 : r = p
    · simp only [if_pos h1]; simp [eq_comm]
    · by_cases h2 : r = q
      · simp only [if_neg h1, if_pos h2]; simp [eq_comm]
      · simp only [if_neg h1, if_neg h2]; simp [eq_comm]

theorem proj_br {p q r : Nat} {a b : G} {l : L} :
    proj (.br p q a b) r = some l ↔
      ∃ x y, proj a r = some x ∧ proj b r = some y ∧
        (if r = p then l = .sel q x y else if r = q then l = .bra p x y else x = y ∧ l = x) := by
  simp only [proj]
  cases proj a r with
  | none => simp
  | some x =>
    cases proj b r with
    | none => simp
    | some y =>
      by_cases h1 : r = p
      · simp only [if_pos h1]; simp [eq_comm]
      · by_cases h2 : r = q
        · simp only [if_neg h1, if_pos h2]; simp [eq_comm]
        · simp only [if_neg h1, if_neg h2]
          by_cases h3 : x = y
          · subst h3; simp [eq_comm]
          · simp [h3]

theorem fire_shape {d : Bool} {p q : Nat} {x y : L} {k1 k2 : L} (h : fire d p q x y = some (k1, k2)) :
    (∃ k k', x = .send q k ∧ y = .recv p k') ∨ (∃ a b a' b', x = .sel q a b ∧ y = .bra p a' b') := by
  cases x <;> cases y <;> simp only [fire] at h
  all_goals first
    | (cases h; done)
    | (split at h
       · next hc =>
         obtain ⟨rfl, rfl⟩ := hc
         first | exact Or.inl ⟨_, _, rfl, rfl⟩ | exact Or.inr ⟨_, _, _, _, rfl, rfl⟩
       · cases h)

theorem upd2 (c : C) (p q : Nat) (k1 k2 : L) (r : Nat) :
    upd (upd c p k1) q k2 r = if r = q then k2 else if r = p then k1 else c r := by
  simp [upd]

/-- the configuration of a projectable global type -/
def cfg (g : G) : C := fun r => (proj g r).getD .done

theorem cfg_of {g : G} (h : ∀ r, ∃ x, proj g r = some x) : Proj g (cfg g) := by
  intro r; have ⟨x, hx⟩ := h r; simp [cfg, hx]

/-- the step of a configuration projected from `msg a b k` or `br a b ..`
    is the head interaction or one between two other roles -/
theorem head_or_apart {c : C} {a b : Nat} (hab : a ≠ b)
    (ha : (∃ k, c a = .send b k) ∨ (∃ x y, c a = .sel b x y))
    (hb : (∃ k, c b = .recv a k) ∨ (∃ x y, c b = .bra a x y))
    {d : Bool} {p q : Nat} {k1 k2 : L} (hf : fire d p q (c p) (c q) = some (k1, k2)) :
    (p = a ∧ q = b) ∨ (p ≠ a ∧ p ≠ b ∧ q ≠ a ∧ q ≠ b) := by
  have hs := fire_shape hf
  by_cases hpa : p = a
  · subst hpa
    left; refine ⟨rfl, ?_⟩
    rcases hs with ⟨k, k', e1, _⟩ | ⟨x, y, x', y', e1, _⟩ <;>
      rcases ha with ⟨k0, e0⟩ | ⟨x0, y0, e0⟩ <;> rw [e0] at e1 <;> cases e1 <;> rfl
  · right
    have hpb : p ≠ b := by
      intro e; subst e
      rcases hs with ⟨k, k', e1, _⟩ | ⟨x, y, x', y', e1, _⟩ <;>
        rcases hb with ⟨k0, e0⟩ | ⟨x0, y0, e0⟩ <;> rw [e0] at e1 <;> cases e1
    have hqa : q ≠ a := by
      intro e; subst e
      rcases hs with ⟨k, k', _, e1⟩ | ⟨x, y, x', y', _, e1⟩ <;>
        rcases ha with ⟨k0, e0⟩ | ⟨x0, y0, e0⟩ <;> rw [e0] at e1 <;> cases e1
    have hqb : q ≠ b := by
      intro e; subst e
      rcases hs with ⟨k, k', _, e1⟩ | ⟨x, y, x', y', _, e1⟩ <;>
        rcases hb with ⟨k0, e0⟩ | ⟨x0, y0, e0⟩ <;> rw [e0] at e1 <;> cases e1 <;> exact hpa rfl
    exact ⟨hpa, hpb, hqa, hqb⟩

/-- session fidelity: a step of a projected configuration is a step of the
    global type -/
theorem sr : ∀ (g : G) {c c' : C}, WF g → Proj g c → Step c c' → ∃ g', WF g' ∧ Proj g' c'
  | .done, c, _, _, hp, ⟨_, hf⟩ => by
    next p q _ _ _ =>
    have e1 : c p = .done := by have := hp p; simp [proj] at this; exact this.symm
    rw [e1] at hf; cases c q <;> simp [fire] at hf
  | .msg a b k, c, _, hw, hp, @Step.go _ d p q k1 k2 hpq hf => by
    have hk : ∀ r, ∃ x, proj k r = some x := fun r => by
      have ⟨x, hx, _⟩ := proj_msg.1 (hp r); exact ⟨x, hx⟩
    have hck := cfg_of hk
    have cr : ∀ r, c r = (if r = a then .send b (cfg k r) else if r = b then .recv a (cfg k r) else cfg k r) := by
      intro r; have ⟨x, hx, e⟩ := proj_msg.1 (hp r); rw [hck r] at hx; cases hx; exact e
    have ca : c a = .send b (cfg k a) := by rw [cr a]; simp
    have cb : c b = .recv a (cfg k b) := by rw [cr b]; simp [Ne.symm hw.1]
    rcases head_or_apart hw.1 (Or.inl ⟨_, ca⟩) (Or.inl ⟨_, cb⟩) hf with ⟨e1, e2⟩ | ⟨h1, h2, h3, h4⟩
    · subst e1; subst e2
      rw [ca, cb] at hf; simp [fire] at hf
      obtain ⟨rfl, rfl⟩ := hf
      refine ⟨k, hw.2, fun r => ?_⟩
      rw [upd2, hck r]
      by_cases r1 : r = q <;> by_cases r2 : r = p <;> simp [r1, r2, cr r, hpq]
    · have hp' : c p = cfg k p := by rw [cr p]; simp [h1, h2]
      have hq' : c q = cfg k q := by rw [cr q]; simp [h3, h4]
      rw [hp', hq'] at hf
      have ⟨k', hw', hp''⟩ := sr k hw.2 hck (Step.go hpq hf)
      refine ⟨.msg a b k', ⟨hw.1, hw'⟩, fun r => proj_msg.2 ⟨_, hp'' r, ?_⟩⟩
      rw [upd2, upd2, cr r]
      by_cases r1 : r = q <;> by_cases r2 : r = p <;> by_cases r3 : r = a <;> by_cases r4 : r = b <;>
        simp_all
  | .br a b x y, c, _, hw, hp, @Step.go _ d p q k1 k2 hpq hf => by
    have hx : ∀ r, ∃ v, proj x r = some v := fun r => by
      have ⟨v, _, hv, _⟩ := proj_br.1 (hp r); exact ⟨v, hv⟩
    have hy : ∀ r, ∃ v, proj y r = some v := fun r => by
      have ⟨_, v, _, hv, _⟩ := proj_br.1 (hp r); exact ⟨v, hv⟩
    have hcx := cfg_of hx
    have hcy := cfg_of hy
    have cr : ∀ r, (if r = a then c r = .sel b (cfg x r) (cfg y r) else if r = b then c r = .bra a (cfg x r) (cfg y r)
        else cfg x r = cfg y r ∧ c r = cfg x r) := by
      intro r; have ⟨u, v, hu, hv, e⟩ := proj_br.1 (hp r)
      rw [hcx r] at hu; rw [hcy r] at hv; cases hu; cases hv; exact e
    have ca : c a = .sel b (cfg x a) (cfg y a) := by have := cr a; simpa using this
    have cb : c b = .bra a (cfg x b) (cfg y b) := by have := cr b; simpa [Ne.symm hw.1] using this
    have co : ∀ r, r ≠ a → r ≠ b → cfg x r = cfg y r ∧ c r = cfg x r := by
      intro r h1 h2; have := cr r; simpa [h1, h2] using this
    rcases head_or_apart hw.1 (Or.inr ⟨_, _, ca⟩) (Or.inr ⟨_, _, cb⟩) hf with ⟨e1, e2⟩ | ⟨h1, h2, h3, h4⟩
    · subst e1; subst e2
      rw [ca, cb] at hf
      cases d
      · simp [fire] at hf
        obtain ⟨rfl, rfl⟩ := hf
        refine ⟨y, hw.2.2, fun r => ?_⟩
        rw [upd2, hcy r]
        by_cases r1 : r = q <;> by_cases r2 : r = p <;> simp_all [(co r · ·)]
      · simp [fire] at hf
        obtain ⟨rfl, rfl⟩ := hf
        refine ⟨x, hw.2.1, fun r => ?_⟩
        rw [upd2, hcx r]
        by_cases r1 : r = q <;> by_cases r2 : r = p <;> simp_all [(co r · ·)]
    · have hp' : c p = cfg x p ∧ c p = cfg y p := by
        have := co p h1 h2; exact ⟨this.2, this.2.trans this.1⟩
      have hq' : c q = cfg x q ∧ c q = cfg y q := by
        have := co q h3 h4; exact ⟨this.2, this.2.trans this.1⟩
      have fx := hf; rw [hp'.1, hq'.1] at fx
      have fy := hf; rw [hp'.2, hq'.2] at fy
      have ⟨x', wx, px⟩ := sr x hw.2.1 hcx (Step.go hpq fx)
      have ⟨y', wy, py⟩ := sr y hw.2.2 hcy (Step.go hpq fy)
      refine ⟨.br a b x' y', ⟨hw.1, wx, wy⟩, fun r => proj_br.2 ⟨_, _, px r, py r, ?_⟩⟩
      rw [upd2, upd2, upd2]
      by_cases r3 : r = a
      · subst r3; simp [h1, h3, Ne.symm h1, Ne.symm h3, ca]
      · by_cases r4 : r = b
        · subst r4; simp [h2, h4, Ne.symm h2, Ne.symm h4, r3, cb]
        · simp only [r3, r4, if_false]
          have := co r r3 r4
          by_cases r1 : r = q <;> by_cases r2 : r = p <;> simp [r1, r2, this.1, this.2]

theorem progress : ∀ (g : G) {c : C}, WF g → Proj g c → Final c ∨ ∃ c', Step c c'
  | .done, c, _, hp => Or.inl fun r => by have := hp r; simp [proj] at this; exact this.symm
  | .msg a b k, c, hw, hp => by
    have ⟨x, _, ex⟩ := proj_msg.1 (hp a)
    have ⟨y, _, ey⟩ := proj_msg.1 (hp b)
    simp at ex; simp [Ne.symm hw.1] at ey
    right; exact ⟨_, Step.go (d := true) (k1 := x) (k2 := y) hw.1 (by rw [ex, ey]; simp [fire])⟩
  | .br a b x y, c, hw, hp => by
    have ⟨u, v, _, _, ex⟩ := proj_br.1 (hp a)
    have ⟨u', v', _, _, ey⟩ := proj_br.1 (hp b)
    simp at ex; simp [Ne.symm hw.1] at ey
    right; exact ⟨_, Step.go (d := true) (k1 := u) (k2 := u') hw.1 (by rw [ex, ey]; simp [fire])⟩

/-- T28g -/
theorem deadlock_free {g : G} {c0 c : C} (hw : WF g) (hp : Proj g c0) (r : Reach c0 c) :
    Final c ∨ ∃ c', Step c c' := by
  have : ∃ g', WF g' ∧ Proj g' c := by
    induction r with
    | refl => exact ⟨g, hw, hp⟩
    | step _ st ih => have ⟨g1, w1, p1⟩ := ih; exact sr g1 w1 p1 st
  have ⟨g', w', p'⟩ := this
  exact progress g' w' p'

/-! ## Global types give priorities -/

/-- role `r` takes part in the `i`-th message -/
def inv (evs : List (Nat × Nat)) (r i : Nat) : Bool :=
  (evs.getD i (0, 1)).1 == r || (evs.getD i (0, 1)).2 == r

open RCE.SessionPrio in
/-- the `i`-th message `(p, q)` of a message-only global type is channel
    `i`, done once (`(i, 0)`), at priority `i`; role `r` does the messages
    it takes part in, in order -/
def netOf (evs : List (Nat × Nat)) : State :=
  fun r => ((List.range evs.length).filter (inv evs r)).map (fun i => (i, 0))

open RCE.SessionPrio in
def endsOf (evs : List (Nat × Nat)) : Ends := fun c => evs.getD c (0, 1)

open RCE.SessionPrio in
theorem filt_map0 (c : Nat) : ∀ l : List Nat,
    filt c (l.map (fun i => (i, 0))) = (l.filter (fun i => i == c)).map (fun i => (i, 0))
  | [] => rfl
  | j :: l => by
    have ih := filt_map0 c l
    simp only [filt] at ih ⊢
    simp only [List.map_cons, List.filter_cons]
    by_cases h : j = c
    · simp [h, ih]
    · simp [h, ih]

theorem filter_one {f : Nat → Bool} {c : Nat} : ∀ l : List Nat, l.Nodup →
    (l.filter f).filter (fun i => i == c) = if c ∈ l ∧ f c = true then [c] else []
  | [], _ => by simp
  | j :: l, nd => by
    have ⟨hj, nd'⟩ := List.nodup_cons.1 nd
    have ih := filter_one (f := f) (c := c) l nd'
    by_cases h : j = c
    · subst h
      have e : (l.filter f).filter (fun i => i == j) = [] := by rw [ih]; simp [hj]
      by_cases hf : f j = true
      · simp [List.filter_cons, hf, e]
      · simp [List.filter_cons, hf, e]
    · by_cases hf : f j = true
      · simp [List.filter_cons, hf, h, ih, Ne.symm h]
      · simp [List.filter_cons, hf, ih, Ne.symm h]

open RCE.SessionPrio in
theorem global_prio {evs : List (Nat × Nat)} (hw : ∀ e ∈ evs, e.1 ≠ e.2) :
    Inv (endsOf evs) (fun a => a.1) (netOf evs) := by
  have hd : ∀ c, (evs.getD c (0, 1)).1 ≠ (evs.getD c (0, 1)).2 := by
    intro c
    by_cases h : c < evs.length
    · simp [List.getD, List.getElem?_eq_getElem h]; exact hw _ (List.getElem_mem _ _ _)
    · simp [List.getD, List.getElem?_eq_none (by omega : evs.length ≤ c)]
  refine ⟨?_, fun c => hd c, ?_, ?_⟩
  · intro p a ha
    simp only [netOf, List.mem_map, List.mem_filter, inv, Bool.or_eq_true, beq_iff_eq] at ha
    obtain ⟨i, ⟨_, h⟩, rfl⟩ := ha
    rcases h with h | h
    · exact Or.inl h.symm
    · exact Or.inr h.symm
  · intro c
    simp only [netOf, endsOf]
    rw [filt_map0, filt_map0, filter_one _ (List.nodup_range _), filter_one _ (List.nodup_range _)]
    simp [inv]
  · intro r
    simp only [netOf, List.pairwise_map]
    exact List.Pairwise.filter _ (List.pairwise_lt_range _)

end RCE.SessionGlobal
