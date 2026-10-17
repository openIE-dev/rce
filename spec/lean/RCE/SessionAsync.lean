/-! T28a: multiparty global types with full merge and asynchronous,
    order-preserving delivery.

    Extends T28g (SessionGlobal, synchronous, plain merge) in two ways.

    Full merge (Hou, Yoshida and Kuhn, arXiv:2402.16741v6, Def. 3.1): a role
    not involved in a choice may have different receive behaviour in the two
    branches; its projection merges them into one branching that accepts the
    labels of both (sends must still agree). Projection is then not preserved
    by steps, so, as in their Def. 3.7, the invariant is association: each
    role's local type is a *wider* version of the projection of the current
    global type (it may accept more labels, `Wide`).

    Asynchrony (Honda, Yoshida and Carbone, POPL 2008, Sec. 2: a send is never
    blocked and two messages from one sender on one channel arrive in sending
    order): one FIFO queue per ordered pair of roles. A send appends its label
    to the queue `Q p q`; a receive takes the head. Global types carry the
    in-flight state: `comm p q (some d) ..` is an interaction whose label `d`
    has been sent but not received. `queues g` reads the queues off the global
    type, and association requires the configuration's queues to equal them.

    Labels are `Bool` (two branches, `ha`/`hb` say which are offered); a plain
    message is a one-label choice. Payloads and recursion are left out, as in
    T28g.

    Theorems: `sr` (every asynchronous step of an associated configuration is
    a step of the global type and stays associated), `progress`, `safe` (a
    receiver always accepts the label at the head of its queue), and
    `deadlock_free` from the projection of any well-formed global type with
    empty queues. -/
namespace RCE.SessionAsync

inductive G where
  | done
  | comm (p q : Nat) (s : Option Bool) (ha hb : Bool) (a b : G)
  deriving Repr

inductive L where
  | done
  | snd (q : Nat) (ha hb : Bool) (a b : L)
  | rcv (p : Nat) (ha hb : Bool) (a b : L)
  deriving DecidableEq, Repr

/-- the branch for label `d` -/
def pick {α : Type} (d : Bool) (a b : α) : α := if d then a else b

/-- full merge: sends agree, receives take the union of labels -/
def merge : L → L → Option L
  | .done, .done => some .done
  | .snd q ha hb a b, .snd q' ha' hb' a' b' =>
    if q = q' ∧ ha = ha' ∧ hb = hb' then
      match (if ha then merge a a' else some .done), (if hb then merge b b' else some .done) with
      | some x, some y => some (.snd q ha hb x y)
      | _, _ => none
    else none
  | .rcv p ha hb a b, .rcv p' ha' hb' a' b' =>
    if p = p' then
      match (if ha then (if ha' then merge a a' else some a) else (if ha' then some a' else some .done)),
            (if hb then (if hb' then merge b b' else some b) else (if hb' then some b' else some .done)) with
      | some x, some y => some (.rcv p (ha || ha') (hb || hb') x y)
      | _, _ => none
    else none
  | _, _ => none

/-- `Wide x y`: `x` behaves as `y` but may accept more labels -/
inductive Wide : L → L → Prop
  | done : Wide .done .done
  | snd {q ha hb a b a' b'} : (ha = true → Wide a a') → (hb = true → Wide b b') →
      Wide (.snd q ha hb a b) (.snd q ha hb a' b')
  | rcv {p ha hb a b ha' hb' a' b'} : (ha' = true → ha = true) → (hb' = true → hb = true) →
      (ha' = true → Wide a a') → (hb' = true → Wide b b') → Wide (.rcv p ha hb a b) (.rcv p ha' hb' a' b')

theorem Wide.refl : ∀ x : L, Wide x x
  | .done => .done
  | .snd _ _ _ a b => .snd (fun _ => Wide.refl a) (fun _ => Wide.refl b)
  | .rcv _ _ _ a b => .rcv id id (fun _ => Wide.refl a) (fun _ => Wide.refl b)

theorem Wide.trans {x y z : L} (h1 : Wide x y) (h2 : Wide y z) : Wide x z := by
  induction h1 generalizing z with
  | done => exact h2
  | snd ha hb iha ihb =>
    cases h2 with
    | snd ha2 hb2 => exact .snd (fun h => iha h (ha2 h)) (fun h => ihb h (hb2 h))
  | rcv fa fb _ _ iha ihb =>
    cases h2 with
    | rcv fa2 fb2 wa2 wb2 =>
      exact .rcv (fa ∘ fa2) (fb ∘ fb2) (fun h => iha (fa2 h) (wa2 h)) (fun h => ihb (fb2 h) (wb2 h))

/-- what a merged receive branch is -/
def MB (h h' : Bool) (a a' u : L) : Prop :=
  (h = true → h' = true → merge a a' = some u) ∧ (h = true → h' = false → u = a) ∧
  (h = false → h' = true → u = a')

theorem merge_inv {x y z : L} (hm : merge x y = some z) :
    (x = .done ∧ y = .done ∧ z = .done) ∨
    (∃ q ha hb a b a' b' u v, x = .snd q ha hb a b ∧ y = .snd q ha hb a' b' ∧ z = .snd q ha hb u v ∧
      (ha = true → merge a a' = some u) ∧ (hb = true → merge b b' = some v)) ∨
    (∃ p ha hb a b ha' hb' a' b' u v, x = .rcv p ha hb a b ∧ y = .rcv p ha' hb' a' b' ∧
      z = .rcv p (ha || ha') (hb || hb') u v ∧ MB ha ha' a a' u ∧ MB hb hb' b b' v) := by
  cases x <;> cases y <;> simp only [merge] at hm <;> try (cases hm; done)
  · cases hm; exact Or.inl ⟨rfl, rfl, rfl⟩
  · rename_i q ha hb a b q' ha' hb' a' b'
    split at hm
    · next hc =>
      obtain ⟨rfl, rfl, rfl⟩ := hc
      split at hm
      · next u v e1 e2 =>
        cases hm
        refine Or.inr (Or.inl ⟨q, ha, hb, a, b, a', b', u, v, rfl, rfl, rfl, ?_, ?_⟩)
        · intro h; simpa [h] using e1
        · intro h; simpa [h] using e2
      · cases hm
    · cases hm
  · rename_i p ha hb a b p' ha' hb' a' b'
    split at hm
    · next hc =>
      subst hc
      split at hm
      · next u v e1 e2 =>
        cases hm
        refine Or.inr (Or.inr ⟨p, ha, hb, a, b, ha', hb', a', b', u, v, rfl, rfl, rfl, ?_, ?_⟩)
        · refine ⟨fun h h' => ?_, fun h h' => ?_, fun h h' => ?_⟩ <;> simp_all
        · refine ⟨fun h h' => ?_, fun h h' => ?_, fun h h' => ?_⟩ <;> simp_all
      · cases hm
    · cases hm

/-- a merge accepts everything either side accepts (HYK Lem. A.7) -/
theorem merge_wide : ∀ (x : L) {y z : L}, merge x y = some z → Wide z x ∧ Wide z y
  | .done, _, _, hm => by
    rcases merge_inv hm with ⟨_, rfl, rfl⟩ | ⟨_, _, _, _, _, _, _, _, _, h, _⟩ | ⟨_, _, _, _, _, _, _, _, _, _, _, h, _⟩ <;>
      first | exact ⟨.done, .done⟩ | cases h
  | .snd q ha hb a b, _, _, hm => by
    rcases merge_inv hm with ⟨h, _⟩ | ⟨_, _, _, _, _, a', b', u, v, h, rfl, rfl, ea, eb⟩ | ⟨_, _, _, _, _, _, _, _, _, _, _, h, _⟩ <;>
      try (cases h; done)
    cases h
    exact ⟨.snd (fun h => (merge_wide a (ea h)).1) (fun h => (merge_wide b (eb h)).1),
           .snd (fun h => (merge_wide a (ea h)).2) (fun h => (merge_wide b (eb h)).2)⟩
  | .rcv p ha hb a b, _, _, hm => by
    rcases merge_inv hm with ⟨h, _⟩ | ⟨_, _, _, _, _, _, _, _, _, h, _⟩ | ⟨_, _, _, _, _, ha', hb', a', b', u, v, h, rfl, rfl, ma, mb⟩ <;>
      try (cases h; done)
    cases h
    obtain ⟨ma1, ma2, ma3⟩ := ma
    obtain ⟨mb1, mb2, mb3⟩ := mb
    refine ⟨.rcv (by simp_all) (by simp_all) (fun h => ?_) (fun h => ?_),
            .rcv (by simp_all) (by simp_all) (fun h => ?_) (fun h => ?_)⟩
    · cases h2 : ha'
      · rw [ma2 h h2]; exact Wide.refl _
      · exact (merge_wide a (ma1 h h2)).1
    · cases h2 : hb'
      · rw [mb2 h h2]; exact Wide.refl _
      · exact (merge_wide b (mb1 h h2)).1
    · cases h2 : ha
      · rw [ma3 h2 h]; exact Wide.refl _
      · exact (merge_wide a (ma1 h2 h)).2
    · cases h2 : hb
      · rw [mb3 h2 h]; exact Wide.refl _
      · exact (merge_wide b (mb1 h2 h)).2

theorem wide_done {x : L} (h : Wide .done x) : x = .done := by cases h; rfl

/-- one receive branch of a merge of narrowed sides -/
theorem mb_mono {h h' f f' : Bool} {a a' u a1 a1' : L}
    (ih : ∀ {y z x' y'}, merge a y = some z → Wide a x' → Wide y y' → ∃ z', merge x' y' = some z' ∧ Wide z z')
    (mb : MB h h' a a' u) (ff : f = true → h = true) (ff' : f' = true → h' = true)
    (wa : f = true → Wide a a1) (wa' : f' = true → Wide a' a1') :
    ∃ u', (if f then (if f' then merge a1 a1' else some a1) else (if f' then some a1' else some .done)) = some u' ∧
      ((f || f') = true → Wide u u') := by
  obtain ⟨m1, m2, m3⟩ := mb
  cases f <;> cases f'
  · exact ⟨_, rfl, by simp⟩
  · refine ⟨a1', rfl, fun _ => ?_⟩
    have h1 := ff' rfl
    cases h2 : h
    · rw [m3 h2 h1]; exact wa' rfl
    · exact (merge_wide a (m1 h2 h1)).2.trans (wa' rfl)
  · refine ⟨a1, rfl, fun _ => ?_⟩
    have h1 := ff rfl
    cases h2 : h'
    · rw [m2 h1 h2]; exact wa rfl
    · exact (merge_wide a (m1 h1 h2)).1.trans (wa rfl)
  · have ⟨z', e, w⟩ := ih (m1 (ff rfl) (ff' rfl)) (wa rfl) (wa' rfl)
    exact ⟨z', by simpa using e, fun _ => w⟩

/-- merging narrowed types is defined and narrows the merge -/
theorem merge_mono : ∀ (x : L) {y z x' y' : L}, merge x y = some z → Wide x x' → Wide y y' →
    ∃ z', merge x' y' = some z' ∧ Wide z z'
  | .done, _, _, _, _, hm, wx, wy => by
    rcases merge_inv hm with ⟨_, rfl, rfl⟩ | ⟨_, _, _, _, _, _, _, _, _, h, _⟩ | ⟨_, _, _, _, _, _, _, _, _, _, _, h, _⟩ <;>
      try (cases h; done)
    cases wx; cases wy; exact ⟨.done, rfl, .done⟩
  | .snd q ha hb a b, _, _, _, _, hm, wx, wy => by
    rcases merge_inv hm with ⟨h, _⟩ | ⟨_, _, _, _, _, a', b', u, v, h, rfl, rfl, ea, eb⟩ | ⟨_, _, _, _, _, _, _, _, _, _, _, h, _⟩ <;>
      try (cases h; done)
    cases h
    cases wx with
    | snd wa wb =>
    cases wy with
    | snd wa' wb' =>
    rename_i a1 b1 a1' b1'
    have ka : ∃ u', (if ha then merge a1 a1' else some .done) = some u' ∧ (ha = true → Wide u u') := by
      cases h : ha
      · exact ⟨_, rfl, by simp⟩
      · have ⟨z', e, w⟩ := merge_mono a (ea h) (wa h) (wa' h); exact ⟨z', by simpa using e, fun _ => w⟩
    have kb : ∃ v', (if hb then merge b1 b1' else some .done) = some v' ∧ (hb = true → Wide v v') := by
      cases h : hb
      · exact ⟨_, rfl, by simp⟩
      · have ⟨z', e, w⟩ := merge_mono b (eb h) (wb h) (wb' h); exact ⟨z', by simpa using e, fun _ => w⟩
    obtain ⟨u', e1, w1⟩ := ka
    obtain ⟨v', e2, w2⟩ := kb
    refine ⟨.snd q ha hb u' v', ?_, .snd w1 w2⟩
    simp only [merge, and_self, if_true]
    rw [e1, e2]
  | .rcv p ha hb a b, _, _, _, _, hm, wx, wy => by
    rcases merge_inv hm with ⟨h, _⟩ | ⟨_, _, _, _, _, _, _, _, _, h, _⟩ | ⟨_, _, _, _, _, ha', hb', a', b', u, v, h, rfl, rfl, ma, mb⟩ <;>
      try (cases h; done)
    cases h
    cases wx with
    | rcv fa fb wa wb =>
    cases wy with
    | rcv fa' fb' wa' wb' =>
    rename_i f1 g1 a1 b1 f2 g2 a1' b1'
    obtain ⟨u', e1, w1⟩ := mb_mono (fun hm wx wy => merge_mono a hm wx wy) ma fa fa' wa wa'
    obtain ⟨v', e2, w2⟩ := mb_mono (fun hm wx wy => merge_mono b hm wx wy) mb fb fb' wb wb'
    refine ⟨.rcv p (f1 || f2) (g1 || g2) u' v', ?_, .rcv ?_ ?_ w1 w2⟩
    · simp only [merge, if_true]; rw [e1, e2]
    · intro h; cases f1 <;> cases f2 <;> simp_all
    · intro h; cases g1 <;> cases g2 <;> simp_all

/-! ## Global types, projection, queues -/

def pj (h : Bool) (o : Option L) : Option L := if h then o else some .done

/-- the projection onto a role outside the interaction: the present branches, merged -/
def join (ha hb : Bool) (x y : L) : Option L :=
  if ha then (if hb then merge x y else some x) else (if hb then some y else none)

/-- projection of `comm p q s ha hb ..` onto `r`, given the branch projections -/
def pbody (p q : Nat) (s : Option Bool) (ha hb : Bool) (r : Nat) (x y : L) : Option L :=
  if r = q then some (.rcv p ha hb x y)
  else match s with
    | none => if r = p then some (.snd q ha hb x y) else join ha hb x y
    | some d => if pick d ha hb then some (pick d x y) else none

def proj : G → Nat → Option L
  | .done, _ => some .done
  | .comm p q s ha hb a b, r =>
    match pj ha (proj a r), pj hb (proj b r) with
    | some x, some y => pbody p q s ha hb r x y
    | _, _ => none

theorem proj_comm {p q : Nat} {s : Option Bool} {ha hb : Bool} {a b : G} {r : Nat} {l : L} :
    proj (.comm p q s ha hb a b) r = some l ↔
      ∃ x y, pj ha (proj a r) = some x ∧ pj hb (proj b r) = some y ∧ pbody p q s ha hb r x y = some l := by
  simp only [proj]
  constructor
  · intro h
    split at h
    · next x y e1 e2 => exact ⟨x, y, e1, e2, h⟩
    · cases h
  · rintro ⟨x, y, e1, e2, h⟩
    rw [e1, e2]; exact h

def WF : G → Prop
  | .done => True
  | .comm p q s ha hb a b => p ≠ q ∧ (ha || hb) = true ∧ (∀ d, s = some d → pick d ha hb = true) ∧
      (ha = true → WF a) ∧ (hb = true → WF b)

abbrev Qs := Nat → Nat → List Bool

/-- the messages in flight, read off a global type: a sent interaction
    contributes its label, before everything sent later on that path -/
def queues : G → Qs
  | .done, _, _ => []
  | .comm _ _ none ha _ a b, x, y => if ha then queues a x y else queues b x y
  | .comm p q (some d) _ _ a b, x, y =>
    (if x = p ∧ y = q then [d] else []) ++ (if d then queues a x y else queues b x y)

/-- the queue invariant: before `p` sends at `comm p q none ..`, both branches
    hold the same messages and none from `p` -/
def Inv : G → Prop
  | .done => True
  | .comm p _ none ha hb a b => (ha = true → Inv a) ∧ (hb = true → Inv b) ∧
      (ha = true → hb = true → queues a = queues b) ∧
      (ha = true → ∀ x, queues a p x = []) ∧ (hb = true → ∀ x, queues b p x = [])
  | .comm _ _ (some _) ha hb a b => (ha = true → Inv a) ∧ (hb = true → Inv b)

inductive Act where
  | snd (p q : Nat) (d : Bool)
  | rcv (p q : Nat) (d : Bool)
  deriving DecidableEq, Repr

/-- the role that acts -/
def Act.subj : Act → Nat
  | .snd p _ _ => p
  | .rcv _ q _ => q

def updQ : Act → Qs → Qs
  | .snd p q d, Q => fun x y => if x = p ∧ y = q then Q x y ++ [d] else Q x y
  | .rcv p q _, Q => fun x y => if x = p ∧ y = q then (Q x y).tail else Q x y

/-- global steps: `p` sends (the label goes in flight), `q` receives, or a
    step of a role not blocked at the head happens inside the continuation
    (inside every present branch while the choice is not yet sent) -/
inductive GStep : G → Act → G → Prop
  | send {p q d ha hb a b} : pick d ha hb = true →
      GStep (.comm p q none ha hb a b) (.snd p q d) (.comm p q (some d) ha hb a b)
  | recv {p q d ha hb a b} : GStep (.comm p q (some d) ha hb a b) (.rcv p q d) (pick d a b)
  | ctxU {p q ha hb a b a' b' α} : α.subj ≠ p → α.subj ≠ q →
      (ha = true → GStep a α a') → (hb = true → GStep b α b') → (ha = false → a' = a) → (hb = false → b' = b) →
      GStep (.comm p q none ha hb a b) α (.comm p q none ha hb a' b')
  | ctxS {p q d ha hb a b x α} : α.subj ≠ q → GStep (pick d a b) α x →
      GStep (.comm p q (some d) ha hb a b) α (.comm p q (some d) ha hb (if d then x else a) (if d then b else x))

theorem join_wide {ha hb : Bool} {x y l : L} {d : Bool} (hj : join ha hb x y = some l) (hd : pick d ha hb = true) :
    Wide l (pick d x y) := by
  unfold join at hj
  cases ha <;> cases hb <;> cases d <;> simp [pick] at hj hd ⊢
  all_goals first
    | (subst hj; exact Wide.refl _)
    | exact (merge_wide _ hj).1
    | exact (merge_wide _ hj).2

theorem join_mono {ha hb : Bool} {x y x' y' l : L} (hj : join ha hb x y = some l)
    (wx : ha = true → Wide x x') (wy : hb = true → Wide y y') : ∃ l', join ha hb x' y' = some l' ∧ Wide l l' := by
  unfold join at hj ⊢
  cases ha <;> cases hb <;> simp at hj ⊢
  · subst hj; exact wy rfl
  · subst hj; exact wx rfl
  · exact merge_mono _ hj (wx rfl) (wy rfl)

theorem pj_true {o : Option L} {x : L} (h : pj true o = some x) : o = some x := h
theorem pj_false {o : Option L} {x : L} (h : pj false o = some x) : x = .done := by
  simp [pj] at h; exact h.symm

/-- a step leaves every other role's projection defined, and the old one
    accepts at least what the new one does (a merge resolved to a branch) -/
theorem proj_ns {g : G} {α : Act} {g' : G} (st : GStep g α g') :
    WF g → ∀ {r : Nat} {l : L}, r ≠ α.subj → proj g r = some l → ∃ l', proj g' r = some l' ∧ Wide l l' := by
  induction st with
  | @send p q d ha hb a b hd =>
    intro _ r l hr hp
    obtain ⟨x, y, e1, e2, hbody⟩ := proj_comm.1 hp
    simp only [Act.subj] at hr
    unfold pbody at hbody
    by_cases hq : r = q
    · rw [if_pos hq] at hbody; cases hbody
      exact ⟨_, proj_comm.2 ⟨x, y, e1, e2, by rw [pbody, if_pos hq]⟩, Wide.refl _⟩
    · simp only [if_neg hq, if_neg hr] at hbody
      exact ⟨pick d x y, proj_comm.2 ⟨x, y, e1, e2, by simp only [pbody, if_neg hq, hd, if_true]⟩, join_wide hbody hd⟩
  | @recv p q d ha hb a b =>
    intro _ r l hr hp
    obtain ⟨x, y, e1, e2, hbody⟩ := proj_comm.1 hp
    simp only [Act.subj] at hr
    simp only [pbody, hr, if_false] at hbody
    split at hbody
    · next hd =>
      cases hbody
      refine ⟨_, ?_, Wide.refl _⟩
      cases d <;> simp [pick] at hd ⊢
      · subst hd; exact e2
      · subst hd; exact e1
    · cases hbody
  | @ctxU p q ha hb a b a' b' α h1 h2 _ _ ea eb iha ihb =>
    intro hw r l hr hp
    obtain ⟨x, y, e1, e2, hbody⟩ := proj_comm.1 hp
    have kx : ∃ x', pj ha (proj a' r) = some x' ∧ (ha = true → Wide x x') := by
      cases h : ha
      · rw [h] at e1; exact ⟨.done, rfl, by simp⟩
      · rw [h] at e1
        have ⟨x', e, w⟩ := iha h (hw.2.2.2.1 h) hr (pj_true e1)
        exact ⟨x', e, fun _ => w⟩
    have ky : ∃ y', pj hb (proj b' r) = some y' ∧ (hb = true → Wide y y') := by
      cases h : hb
      · rw [h] at e2; exact ⟨.done, rfl, by simp⟩
      · rw [h] at e2
        have ⟨y', e, w⟩ := ihb h (hw.2.2.2.2 h) hr (pj_true e2)
        exact ⟨y', e, fun _ => w⟩
    obtain ⟨x', ex, wx⟩ := kx
    obtain ⟨y', ey, wy⟩ := ky
    unfold pbody at hbody
    by_cases hq : r = q
    · rw [if_pos hq] at hbody; cases hbody
      exact ⟨_, proj_comm.2 ⟨x', y', ex, ey, by rw [pbody, if_pos hq]⟩, .rcv id id wx wy⟩
    · by_cases hpp : r = p
      · simp only [if_neg hq, if_pos hpp] at hbody; cases hbody
        exact ⟨_, proj_comm.2 ⟨x', y', ex, ey, by simp only [pbody, if_neg hq, if_pos hpp]⟩, .snd wx wy⟩
      · simp only [if_neg hq, if_neg hpp] at hbody
        have ⟨l', ej, w⟩ := join_mono hbody wx wy
        exact ⟨l', proj_comm.2 ⟨x', y', ex, ey, by simp only [pbody, if_neg hq, if_neg hpp, ej]⟩, w⟩
  | @ctxS p q d ha hb a b x α hs _ ih =>
    intro hw r l hr hp
    obtain ⟨xa, xb, e1, e2, hbody⟩ := proj_comm.1 hp
    have hd := hw.2.2.1 d rfl
    cases d
    · simp [pick] at hd ih hbody; subst hd
      have ⟨x', ex, wx⟩ := ih (hw.2.2.2.2 rfl) hr (pj_true e2)
      by_cases hq : r = q
      · simp [pbody, hq] at hbody; subst hbody
        exact ⟨_, proj_comm.2 ⟨xa, x', e1, ex, by simp [pbody, hq]⟩, .rcv id id (fun _ => Wide.refl _) (fun _ => wx)⟩
      · simp [pbody, hq, pick] at hbody; subst hbody
        exact ⟨x', proj_comm.2 ⟨xa, x', e1, ex, by simp [pbody, hq, pick]⟩, wx⟩
    · simp [pick] at hd ih hbody; subst hd
      have ⟨x', ex, wx⟩ := ih (hw.2.2.2.1 rfl) hr (pj_true e1)
      by_cases hq : r = q
      · simp [pbody, hq] at hbody; subst hbody
        exact ⟨_, proj_comm.2 ⟨x', xb, ex, e2, by simp [pbody, hq]⟩, .rcv id id (fun _ => wx) (fun _ => Wide.refl _)⟩
      · simp [pbody, hq, pick] at hbody; subst hbody
        exact ⟨x', proj_comm.2 ⟨x', xb, ex, e2, by simp [pbody, hq, pick]⟩, wx⟩

theorem wf_step {g : G} {α : Act} {g' : G} (st : GStep g α g') : WF g → WF g' := by
  induction st with
  | send hd => intro hw; exact ⟨hw.1, hw.2.1, fun d' e => by cases e; exact hd, hw.2.2.2⟩
  | @recv p q d ha hb a b =>
    intro hw; have hd := hw.2.2.1 d rfl
    cases d <;> simp [pick] at hd ⊢
    · exact hw.2.2.2.2 hd
    · exact hw.2.2.2.1 hd
  | ctxU _ _ _ _ ea eb iha ihb =>
    intro hw
    refine ⟨hw.1, hw.2.1, hw.2.2.1, fun h => iha h (hw.2.2.2.1 h), fun h => ihb h (hw.2.2.2.2 h)⟩
  | @ctxS p q d ha hb a b x α _ _ ih =>
    intro hw; have hd := hw.2.2.1 d rfl
    refine ⟨hw.1, hw.2.1, hw.2.2.1, fun h => ?_, fun h => ?_⟩ <;> cases d <;> simp [pick] at hd ih ⊢
    · exact hw.2.2.2.1 h
    · exact ih (hw.2.2.2.1 hd)
    · exact ih (hw.2.2.2.2 hd)
    · exact hw.2.2.2.2 h

theorem queues_step {g : G} {α : Act} {g' : G} (st : GStep g α g') :
    WF g → Inv g → queues g' = updQ α (queues g) := by
  induction st with
  | @send p q d ha hb a b hd =>
    intro hw hi
    obtain ⟨_, _, i1, i2a, i2b⟩ := hi
    funext x y
    simp only [queues, updQ]
    by_cases hx : x = p ∧ y = q
    · obtain ⟨rfl, rfl⟩ := hx
      cases d <;> cases ha <;> cases hb <;> simp_all [pick]
    · simp only [hx, if_false, List.nil_append]
      cases d <;> cases ha <;> cases hb <;> simp_all [pick]
  | @recv p q d ha hb a b =>
    intro _ _
    funext x y
    simp only [queues, updQ]
    by_cases hx : x = p ∧ y = q
    · simp only [hx, and_self, if_true, List.singleton_append, List.tail_cons]
      cases d <;> rfl
    · simp only [hx, if_false, List.nil_append]; cases d <;> rfl
  | @ctxU p q ha hb a b a' b' α h1 h2 _ _ ea eb iha ihb =>
    intro hw hi
    obtain ⟨ia, ib, _, _, _⟩ := hi
    funext x y
    simp only [queues]
    cases h : ha
    · have hb' : hb = true := by have := hw.2.1; simp_all
      rw [if_neg (by simp)]; rw [ihb hb' (hw.2.2.2.2 hb') (ib hb')]; rfl
    · rw [if_pos rfl]; rw [iha h (hw.2.2.2.1 h) (ia h)]; rfl
  | @ctxS p q d ha hb a b x α hs _ ih =>
    intro hw hi
    have hd := hw.2.2.1 d rfl
    have e : queues x = updQ α (queues (pick d a b)) := by
      cases d <;> simp [pick] at hd ih ⊢
      · exact ih (hw.2.2.2.2 hd) (hi.2 hd)
      · exact ih (hw.2.2.2.1 hd) (hi.1 hd)
    have hg : queues (G.comm p q (some d) ha hb a b) =
        fun u v => (if u = p ∧ v = q then [d] else []) ++ queues (pick d a b) u v := by
      funext u v; cases d <;> rfl
    have hg' : queues (G.comm p q (some d) ha hb (if d then x else a) (if d then b else x)) =
        fun u v => (if u = p ∧ v = q then [d] else []) ++ queues x u v := by
      funext u v; cases d <;> rfl
    rw [hg, hg', e]
    funext u v
    cases α with
    | snd p' q' d' =>
      simp only [updQ]
      by_cases hx : u = p' ∧ v = q' <;> simp [hx, List.append_assoc]
    | rcv p' q' d' =>
      simp only [Act.subj] at hs
      simp only [updQ]
      by_cases hx : u = p' ∧ v = q'
      · obtain ⟨rfl, rfl⟩ := hx
        have hv : ¬ (u = p ∧ v = q) := fun h => hs h.2
        simp [hv]
      · simp [hx]

theorem updQ_other {α : Act} {Q : Qs} {p x : Nat} (hp : α.subj ≠ p) (h : Q p x = []) : updQ α Q p x = [] := by
  cases α with
  | snd p' q' d' =>
    simp only [Act.subj] at hp
    simp only [updQ]; have : ¬ (p = p' ∧ x = q') := fun e => hp e.1.symm
    simp [this, h]
  | rcv p' q' d' => simp only [updQ]; split <;> simp [h]

theorem inv_step {g : G} {α : Act} {g' : G} (st : GStep g α g') : WF g → Inv g → Inv g' := by
  induction st with
  | send _ => intro _ hi; exact ⟨hi.1, hi.2.1⟩
  | @recv p q d ha hb a b =>
    intro hw hi; have hd := hw.2.2.1 d rfl
    cases d <;> simp [pick] at hd ⊢
    · exact hi.2 hd
    · exact hi.1 hd
  | @ctxU p q ha hb a b a' b' α h1 h2 sa sb ea eb iha ihb =>
    intro hw hi
    obtain ⟨ia, ib, i1, i2a, i2b⟩ := hi
    have qa : ha = true → queues a' = updQ α (queues a) := fun h => queues_step (sa h) (hw.2.2.2.1 h) (ia h)
    have qb : hb = true → queues b' = updQ α (queues b) := fun h => queues_step (sb h) (hw.2.2.2.2 h) (ib h)
    refine ⟨fun h => iha h (hw.2.2.2.1 h) (ia h), fun h => ihb h (hw.2.2.2.2 h) (ib h),
      fun h h' => by rw [qa h, qb h', i1 h h'], fun h x => ?_, fun h x => ?_⟩
    · rw [qa h]; exact updQ_other h1 (i2a h x)
    · rw [qb h]; exact updQ_other h1 (i2b h x)
  | @ctxS p q d ha hb a b x α hs _ ih =>
    intro hw hi; have hd := hw.2.2.1 d rfl
    refine ⟨fun h => ?_, fun h => ?_⟩ <;> cases d <;> simp [pick] at hd ih ⊢
    · exact hi.1 h
    · exact ih (hw.2.2.2.1 hd) (hi.1 hd)
    · exact ih (hw.2.2.2.2 hd) (hi.2 hd)
    · exact hi.2 h

theorem pick_eq {α : Type} {d : Bool} {a b : α} : pick d a b = if d then a else b := rfl

/-- a role whose local type (projected from `g`) offers a send can do it: the
    global type has the matching send, somewhere not blocked -/
theorem lsend : ∀ (g : G) {p q : Nat} {ha hb : Bool} {y1 y2 : L} {d : Bool}, WF g →
    proj g p = some (.snd q ha hb y1 y2) → pick d ha hb = true →
    ∃ g', GStep g (.snd p q d) g' ∧ proj g' p = some (pick d y1 y2)
  | .done, _, _, _, _, _, _, _, _, hp, _ => by simp [proj] at hp
  | .comm s t st fa fb a b, p, q, ha, hb, y1, y2, d, hw, hp, hd => by
    obtain ⟨x, y, e1, e2, hbody⟩ := proj_comm.1 hp
    by_cases ht : p = t
    · simp only [pbody, if_pos ht] at hbody; cases hbody
    cases st with
    | none =>
      by_cases hs : p = s
      · simp only [pbody, if_neg ht, if_pos hs] at hbody; cases hbody; subst hs
        refine ⟨_, GStep.send hd, proj_comm.2 ⟨y1, y2, e1, e2, ?_⟩⟩
        simp only [pbody, if_neg ht]; rw [if_pos hd]
      · simp only [pbody, if_neg ht, if_neg hs] at hbody
        have sp : (Act.snd p q d).subj = p := rfl
        unfold join at hbody
        cases hfa : fa <;> cases hfb : fb <;> simp only [hfa, hfb, if_true, if_false, Bool.false_eq_true] at hbody e1 e2
        · cases hbody
        · cases hbody
          have ⟨b', sb, pb⟩ := lsend b (hw.2.2.2.2 hfb) (pj_true e2) hd
          refine ⟨.comm s t none fa fb a b', ?_, proj_comm.2 ⟨x, pick d y1 y2, ?_, ?_, ?_⟩⟩
          · subst hfa hfb
            exact GStep.ctxU (by simp [Act.subj, hs]) (by simp [Act.subj, ht]) (fun h => by cases h) (fun _ => sb) (fun _ => rfl) (fun h => by cases h)
          · rw [hfa]; exact e1
          · rw [hfb]; exact pb
          · simp [pbody, ht, hs, join, hfa, hfb]
        · cases hbody
          have ⟨a', sa, pa⟩ := lsend a (hw.2.2.2.1 hfa) (pj_true e1) hd
          refine ⟨.comm s t none fa fb a' b, ?_, proj_comm.2 ⟨pick d y1 y2, y, ?_, ?_, ?_⟩⟩
          · subst hfa hfb
            exact GStep.ctxU (by simp [Act.subj, hs]) (by simp [Act.subj, ht]) (fun _ => sa) (fun h => by cases h) (fun h => by cases h) (fun _ => rfl)
          · rw [hfa]; exact pa
          · rw [hfb]; exact e2
          · simp [pbody, ht, hs, join, hfa, hfb]
        · rcases merge_inv hbody with ⟨_, _, h⟩ | ⟨_, _, _, a1, b1, a2, b2, u, v, ex, ey, ez, mu, mv⟩ | ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, h, _⟩ <;>
            try (cases h; done)
          cases ez; subst ex; subst ey
          have ⟨a', sa, pa⟩ := lsend a (hw.2.2.2.1 hfa) (pj_true e1) hd
          have ⟨b', sb, pb⟩ := lsend b (hw.2.2.2.2 hfb) (pj_true e2) hd
          refine ⟨.comm s t none fa fb a' b', ?_, proj_comm.2 ⟨pick d a1 b1, pick d a2 b2, ?_, ?_, ?_⟩⟩
          · subst hfa hfb
            exact GStep.ctxU (by simp [Act.subj, hs]) (by simp [Act.subj, ht]) (fun _ => sa) (fun _ => sb) (fun h => by cases h) (fun h => by cases h)
          · rw [hfa]; exact pa
          · rw [hfb]; exact pb
          · simp only [pbody, if_neg ht, if_neg hs, join, hfa, hfb, if_true]
            cases d <;> simp [pick] at hd ⊢
            · exact mv hd
            · exact mu hd
    | some e =>
      by_cases he : pick e fa fb = true
      · simp only [pbody, if_neg ht, he, if_true, Option.some.injEq] at hbody
        cases e
        · simp only [pick, Bool.false_eq_true, if_false] at he hbody
          subst he hbody
          have ⟨b', sb, pb⟩ := lsend b (hw.2.2.2.2 rfl) (pj_true e2) hd
          refine ⟨_, GStep.ctxS (d := false) (a := a) (b := b) (by simp [Act.subj, ht]) sb, proj_comm.2 ⟨x, pick d y1 y2, ?_, ?_, ?_⟩⟩
          · exact e1
          · exact pb
          · simp [pbody, ht, pick]
        · simp only [pick, if_true] at he hbody
          subst he hbody
          have ⟨a', sa, pa⟩ := lsend a (hw.2.2.2.1 rfl) (pj_true e1) hd
          refine ⟨_, GStep.ctxS (d := true) (a := a) (b := b) (by simp [Act.subj, ht]) sa, proj_comm.2 ⟨pick d y1 y2, y, ?_, ?_, ?_⟩⟩
          · exact pa
          · exact e2
          · simp [pbody, ht, pick]
      · simp only [pbody, if_neg ht, he, if_false, Bool.false_eq_true] at hbody; cases hbody

/-- a role whose local type (projected from `g`) receives from `p` while the
    `p → q` queue has `d` at its head: the label is accepted, and the global
    type has the matching receive, somewhere not blocked -/
theorem lrecv : ∀ (g : G) {p q : Nat} {ha hb : Bool} {y1 y2 : L} {d : Bool} {rest : List Bool}, WF g → Inv g →
    proj g q = some (.rcv p ha hb y1 y2) → queues g p q = d :: rest →
    pick d ha hb = true ∧ ∃ g', GStep g (.rcv p q d) g' ∧ proj g' q = some (pick d y1 y2)
  | .done, _, _, _, _, _, _, _, _, _, _, hp, _ => by simp [proj] at hp
  | .comm s t st fa fb a b, p, q, ha, hb, y1, y2, d, rest, hw, hi, hp, hq => by
    obtain ⟨x, y, e1, e2, hbody⟩ := proj_comm.1 hp
    cases st with
    | none =>
      by_cases ht : q = t
      · exfalso
        simp only [pbody, if_pos ht] at hbody
        cases hbody
        simp only [queues] at hq
        cases fa
        · have hfb : fb = true := by simpa using hw.2.1
          simp only [if_false, Bool.false_eq_true] at hq
          rw [hi.2.2.2.2 hfb] at hq; cases hq
        · simp only [if_true] at hq
          rw [hi.2.2.2.1 rfl] at hq; cases hq
      by_cases hs : q = s
      · simp only [pbody, if_neg ht, if_pos hs] at hbody; cases hbody
      simp only [pbody, if_neg ht, if_neg hs] at hbody
      obtain ⟨ia, ib, i1, -, -⟩ := hi
      simp only [queues] at hq
      unfold join at hbody
      cases hfa : fa <;> cases hfb : fb <;> simp only [hfa, hfb, if_true, if_false, Bool.false_eq_true] at hbody e1 e2 hq
      · cases hbody
      · cases hbody
        obtain ⟨hd, b', sb, pb⟩ := lrecv b (hw.2.2.2.2 hfb) (ib hfb) (pj_true e2) hq
        refine ⟨hd, .comm s t none fa fb a b', ?_, proj_comm.2 ⟨x, pick d y1 y2, ?_, ?_, ?_⟩⟩
        · subst hfa hfb
          exact GStep.ctxU (by simp [Act.subj, hs]) (by simp [Act.subj, ht]) (fun h => by cases h) (fun _ => sb) (fun _ => rfl) (fun h => by cases h)
        · rw [hfa]; exact e1
        · rw [hfb]; exact pb
        · simp [pbody, ht, hs, join, hfa, hfb]
      · cases hbody
        obtain ⟨hd, a', sa, pa⟩ := lrecv a (hw.2.2.2.1 hfa) (ia hfa) (pj_true e1) hq
        refine ⟨hd, .comm s t none fa fb a' b, ?_, proj_comm.2 ⟨pick d y1 y2, y, ?_, ?_, ?_⟩⟩
        · subst hfa hfb
          exact GStep.ctxU (by simp [Act.subj, hs]) (by simp [Act.subj, ht]) (fun _ => sa) (fun h => by cases h) (fun h => by cases h) (fun _ => rfl)
        · rw [hfa]; exact pa
        · rw [hfb]; exact e2
        · simp [pbody, ht, hs, join, hfa, hfb]
      · have hqb : queues b p q = d :: rest := by rw [← i1 hfa hfb]; exact hq
        rcases merge_inv hbody with ⟨_, _, h⟩ | ⟨_, _, _, _, _, _, _, _, _, _, _, h, _⟩ |
            ⟨p', ha1, hb1, a1, b1, ha2, hb2, a2, b2, u, v, ex, ey, ez, mu, mv⟩ <;>
          try (cases h; done)
        cases ez; subst ex; subst ey
        obtain ⟨hd1, a', sa, pa⟩ := lrecv a (hw.2.2.2.1 hfa) (ia hfa) (pj_true e1) hq
        obtain ⟨hd2, b', sb, pb⟩ := lrecv b (hw.2.2.2.2 hfb) (ib hfb) (pj_true e2) hqb
        refine ⟨?_, .comm s t none fa fb a' b', ?_, proj_comm.2 ⟨pick d a1 b1, pick d a2 b2, ?_, ?_, ?_⟩⟩
        · cases d <;> simp [pick] at hd1 ⊢ <;> simp [hd1]
        · subst hfa hfb
          exact GStep.ctxU (by simp [Act.subj, hs]) (by simp [Act.subj, ht]) (fun _ => sa) (fun _ => sb) (fun h => by cases h) (fun h => by cases h)
        · rw [hfa]; exact pa
        · rw [hfb]; exact pb
        · simp only [pbody, if_neg ht, if_neg hs, join, hfa, hfb, if_true]
          cases d <;> simp [pick] at hd1 hd2 ⊢
          · exact mv.1 hd1 hd2
          · exact mu.1 hd1 hd2
    | some e =>
      by_cases ht : q = t
      · simp only [pbody, if_pos ht] at hbody
        cases hbody
        subst ht
        simp [queues] at hq
        obtain ⟨rfl, -⟩ := hq
        have hd := hw.2.2.1 _ rfl
        refine ⟨hd, _, GStep.recv, ?_⟩
        cases e <;> simp [pick] at hd ⊢
        · simpa [pj, hd] using e2
        · simpa [pj, hd] using e1
      · have hq' : queues (pick e a b) p q = d :: rest := by
          have : ¬ (p = s ∧ q = t) := fun h => ht h.2
          simp only [queues, if_neg this, List.nil_append] at hq; cases e <;> simpa [pick] using hq
        by_cases he : pick e fa fb = true
        · simp only [pbody, if_neg ht, he, if_true, Option.some.injEq] at hbody
          cases e
          · simp only [pick, Bool.false_eq_true, if_false] at he hbody hq'
            subst he hbody
            obtain ⟨hd, b', sb, pb⟩ := lrecv b (hw.2.2.2.2 rfl) (hi.2 rfl) (pj_true e2) hq'
            refine ⟨hd, _, GStep.ctxS (d := false) (a := a) (b := b) (by simp [Act.subj, ht]) sb, proj_comm.2 ⟨x, pick d y1 y2, e1, pb, ?_⟩⟩
            simp [pbody, ht, pick]
          · simp only [pick, if_true] at he hbody hq'
            subst he hbody
            obtain ⟨hd, a', sa, pa⟩ := lrecv a (hw.2.2.2.1 rfl) (hi.1 rfl) (pj_true e1) hq'
            refine ⟨hd, _, GStep.ctxS (d := true) (a := a) (b := b) (by simp [Act.subj, ht]) sa, proj_comm.2 ⟨pick d y1 y2, y, pa, e2, ?_⟩⟩
            simp [pbody, ht, pick]
        · simp only [pbody, if_neg ht, he, if_false, Bool.false_eq_true] at hbody; cases hbody

/-! ## Configurations and asynchronous steps -/

/-- a configuration: each role's local type, and the queues -/
structure Cfg where
  c : Nat → L
  Q : Qs

def setc (c : Nat → L) (r : Nat) (l : L) : Nat → L := fun x => if x = r then l else c x

/-- asynchronous steps of a configuration: a role sends one of its offered
    labels (appended to its queue towards the receiver), or a role waiting on
    `p` takes the head of the `p → q` queue, if it offers that label -/
inductive AStep : Cfg → Act → Cfg → Prop
  | send {c : Nat → L} {Q : Qs} {p q : Nat} {ha hb : Bool} {k1 k2 : L} {d : Bool} :
      c p = .snd q ha hb k1 k2 → pick d ha hb = true →
      AStep ⟨c, Q⟩ (.snd p q d) ⟨setc c p (pick d k1 k2), updQ (.snd p q d) Q⟩
  | recv {c : Nat → L} {Q : Qs} {p q : Nat} {ha hb : Bool} {k1 k2 : L} {d : Bool} {rest : List Bool} :
      c q = .rcv p ha hb k1 k2 → Q p q = d :: rest → pick d ha hb = true →
      AStep ⟨c, Q⟩ (.rcv p q d) ⟨setc c q (pick d k1 k2), updQ (.rcv p q d) Q⟩

/-- association (after Hou, Yoshida and Kuhn, Def. 3.7): every local type is
    a wider version of the current projection, and the queues are the ones the
    global type records -/
def Assoc (g : G) (C : Cfg) : Prop :=
  WF g ∧ Inv g ∧ (∀ r, ∃ y, proj g r = some y ∧ Wide (C.c r) y) ∧ C.Q = queues g

theorem wide_snd {q : Nat} {ha hb : Bool} {a b y : L} (h : Wide (.snd q ha hb a b) y) :
    ∃ a' b', y = .snd q ha hb a' b' ∧ (ha = true → Wide a a') ∧ (hb = true → Wide b b') := by
  cases h with
  | snd wa wb => exact ⟨_, _, rfl, wa, wb⟩

theorem wide_rcv {p : Nat} {ha hb : Bool} {a b y : L} (h : Wide (.rcv p ha hb a b) y) :
    ∃ ha' hb' a' b', y = .rcv p ha' hb' a' b' ∧ (ha' = true → ha = true) ∧ (hb' = true → hb = true) ∧
      (ha' = true → Wide a a') ∧ (hb' = true → Wide b b') := by
  cases h with
  | rcv fa fb wa wb => exact ⟨_, _, _, _, rfl, fa, fb, wa, wb⟩

theorem wide_to_done {x : L} (h : Wide x .done) : x = .done := by cases h; rfl

theorem wide_pick {ha hb : Bool} {a b a' b' : L} {d : Bool} (wa : ha = true → Wide a a')
    (wb : hb = true → Wide b b') (hd : pick d ha hb = true) : Wide (pick d a b) (pick d a' b') := by
  cases d <;> simp [pick] at hd ⊢
  · exact wb hd
  · exact wa hd

theorem pick_mono {ha hb ha' hb' : Bool} {d : Bool} (fa : ha' = true → ha = true) (fb : hb' = true → hb = true)
    (hd : pick d ha' hb' = true) : pick d ha hb = true := by
  cases d <;> simp [pick] at hd ⊢
  · exact fb hd
  · exact fa hd

/-- association is kept by the step, given the stepping role's new local type -/
theorem assoc_step {g g' : G} {α : Act} {C : Cfg} {l y : L} (hA : Assoc g C) (st : GStep g α g')
    (hp : proj g' α.subj = some y) (hw : Wide l y) :
    Assoc g' ⟨setc C.c α.subj l, updQ α C.Q⟩ := by
  obtain ⟨w, i, pr, hq⟩ := hA
  refine ⟨wf_step st w, inv_step st w i, fun r => ?_, by rw [hq, queues_step st w i]⟩
  by_cases hr : r = α.subj
  · subst hr; exact ⟨y, hp, by simp [setc]; exact hw⟩
  · obtain ⟨y0, e0, w0⟩ := pr r
    obtain ⟨y', e', w'⟩ := proj_ns st w hr e0
    exact ⟨y', e', by simp only [setc, if_neg hr]; exact Wide.trans w0 w'⟩

/-- subject reduction: every step of an associated configuration is a step
    of the global type, and the result is associated with the new global type -/
theorem sr {g : G} {C C' : Cfg} {α : Act} (hA : Assoc g C) (st : AStep C α C') :
    ∃ g', GStep g α g' ∧ Assoc g' C' := by
  cases st with
  | @send c Q p q ha hb k1 k2 d hc hd =>
    obtain ⟨y, ey, wy⟩ := hA.2.2.1 p
    change Wide (c p) y at wy; rw [hc] at wy
    obtain ⟨y1, y2, rfl, w1, w2⟩ := wide_snd wy
    obtain ⟨g', sg, pg⟩ := lsend g hA.1 ey hd
    exact ⟨g', sg, assoc_step hA sg pg (wide_pick w1 w2 hd)⟩
  | @recv c Q p q ha hb k1 k2 d rest hc hQ hd =>
    obtain ⟨y, ey, wy⟩ := hA.2.2.1 q
    change Wide (c q) y at wy; rw [hc] at wy
    obtain ⟨ha', hb', y1, y2, rfl, fa, fb, w1, w2⟩ := wide_rcv wy
    have hQ' : queues g p q = d :: rest := by rw [← hA.2.2.2]; exact hQ
    obtain ⟨hd', g', sg, pg⟩ := lrecv g hA.1 hA.2.1 ey hQ'
    exact ⟨g', sg, assoc_step hA sg pg (wide_pick w1 w2 hd')⟩

/-- safety: a role waiting on `p` accepts whatever label heads the `p → q` queue -/
theorem safe {g : G} {C : Cfg} {p q : Nat} {ha hb : Bool} {k1 k2 : L} {d : Bool} {rest : List Bool}
    (hA : Assoc g C) (hc : C.c q = .rcv p ha hb k1 k2) (hQ : C.Q p q = d :: rest) : pick d ha hb = true := by
  obtain ⟨y, ey, wy⟩ := hA.2.2.1 q
  rw [hc] at wy
  obtain ⟨ha', hb', y1, y2, rfl, fa, fb, _, _⟩ := wide_rcv wy
  have hQ' : queues g p q = d :: rest := by rw [← hA.2.2.2]; exact hQ
  exact pick_mono fa fb (lrecv g hA.1 hA.2.1 ey hQ').1

/-- termination: every role is done and every queue is empty -/
def Final (C : Cfg) : Prop := (∀ r, C.c r = .done) ∧ ∀ x y, C.Q x y = []

/-- progress: an associated configuration is final or can step -/
theorem progress {g : G} {C : Cfg} (hA : Assoc g C) : Final C ∨ ∃ α C', AStep C α C' := by
  obtain ⟨w, _, pr, hq⟩ := hA
  obtain ⟨c, Q⟩ := C
  simp only at pr hq
  cases g with
  | done =>
    left; refine ⟨fun r => ?_, fun x y => by rw [hq]; rfl⟩
    obtain ⟨y, ey, wy⟩ := pr r
    simp [proj] at ey; subst ey; exact wide_to_done wy
  | comm p q s ha hb a b =>
    right
    obtain ⟨hpq, hab, hs, -, -⟩ := w
    have ⟨d, hd⟩ : ∃ d, pick d ha hb = true := by
      cases s with
      | some e => exact ⟨e, hs e rfl⟩
      | none => cases ha
                · exact ⟨false, by simpa [pick] using hab⟩
                · exact ⟨true, rfl⟩
    cases s with
    | none =>
      obtain ⟨y, ey, wy⟩ := pr p
      obtain ⟨x1, x2, -, -, hb'⟩ := proj_comm.1 ey
      simp only [pbody, if_neg hpq, if_true] at hb'
      cases hb'
      cases hcp : c p with
      | snd q' ha' hb' k1 k2 =>
        rw [hcp] at wy; cases wy
        exact ⟨_, _, AStep.send hcp hd⟩
      | done => rw [hcp] at wy; cases wy
      | rcv _ _ _ _ _ => rw [hcp] at wy; cases wy
    | some e =>
      have hd := hs e rfl
      obtain ⟨y, ey, wy⟩ := pr q
      obtain ⟨x1, x2, -, -, hb'⟩ := proj_comm.1 ey
      simp only [pbody, if_true] at hb'
      cases hb'
      have hQ : Q p q = e :: (if e then queues a p q else queues b p q) := by rw [hq]; simp [queues]
      cases hcq : c q with
      | rcv p' ha' hb' k1 k2 =>
        rw [hcq] at wy
        obtain ⟨_, _, _, _, he, fa, fb, _, _⟩ := wide_rcv wy
        cases he
        exact ⟨_, _, AStep.recv hcq hQ (pick_mono fa fb hd)⟩
      | done => rw [hcq] at wy; cases wy
      | snd _ _ _ _ _ => rw [hcq] at wy; cases wy

/-- reachability by asynchronous steps -/
inductive Reach : Cfg → Cfg → Prop
  | refl {C : Cfg} : Reach C C
  | step {C C' C'' : Cfg} {α : Act} : AStep C α C' → Reach C' C'' → Reach C C''

theorem reach_assoc {C C' : Cfg} (h : Reach C C') : ∀ {g : G}, Assoc g C → ∃ g', Assoc g' C' := by
  induction h with
  | refl => intro g hA; exact ⟨g, hA⟩
  | step st _ ih =>
    intro g hA
    obtain ⟨g', _, hA'⟩ := sr hA st
    exact ih hA'

/-- no interaction has been sent yet -/
def Static : G → Prop
  | .done => True
  | .comm _ _ s _ _ a b => s = none ∧ Static a ∧ Static b

theorem static_queues : ∀ {g : G}, Static g → ∀ x y, queues g x y = []
  | .done, _, _, _ => rfl
  | .comm _ _ none ha _ a b, h, x, y => by
    simp only [queues]; cases ha
    · exact static_queues h.2.2 x y
    · exact static_queues h.2.1 x y
  | .comm _ _ (some _) _ _ _ _, h, _, _ => by cases h.1

theorem static_inv : ∀ {g : G}, Static g → Inv g
  | .done, _ => trivial
  | .comm _ _ none _ _ a b, h => by
    refine ⟨fun _ => static_inv h.2.1, fun _ => static_inv h.2.2, fun _ _ => ?_,
      fun _ x => static_queues h.2.1 _ x, fun _ x => static_queues h.2.2 _ x⟩
    funext x y; rw [static_queues h.2.1, static_queues h.2.2]
  | .comm _ _ (some _) _ _ _ _, h => by cases h.1

/-- deadlock freedom: start every role at the projection of a well-formed
    global type, with empty queues; every reachable configuration is final
    or can step, and every receiver accepts the head of its queue -/
theorem deadlock_free {g : G} {c : Nat → L} (hw : WF g) (hs : Static g) (hc : ∀ r, proj g r = some (c r))
    {C : Cfg} (hr : Reach ⟨c, fun _ _ => []⟩ C) :
    (Final C ∨ ∃ α C', AStep C α C') ∧
    (∀ {p q ha hb k1 k2 d rest}, C.c q = .rcv p ha hb k1 k2 → C.Q p q = d :: rest → pick d ha hb = true) := by
  have hA : Assoc g ⟨c, fun _ _ => []⟩ :=
    ⟨hw, static_inv hs, fun r => ⟨c r, hc r, Wide.refl _⟩, by funext x y; exact (static_queues hs x y).symm⟩
  obtain ⟨g', hA'⟩ := reach_assoc hr hA
  exact ⟨progress hA', fun h1 h2 => safe hA' h1 h2⟩

end RCE.SessionAsync
