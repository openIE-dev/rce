/-! T28p: deadlock freedom by priorities, for any session graph (cycles
    included).

    Kobayashi (CONCUR 2006) annotates each action with levels and forbids a
    process to block on an action while it owes an action of lower level;
    Padovani (CSL-LICS 2014) uses one priority per linear channel, and the
    two ends of a channel compose only at the same priority. Here a process
    is the sequence of exchanges it still has to do; the `t`-th exchange on
    channel `c` is the action `(c, t)` and both ends of `c` do the same
    action, so one priority `pr (c, t)` serves both ends (Padovani's
    composition rule). The typing condition is that each process does its
    actions in strictly increasing priority (`Inv.incr`): it never blocks
    on an action while it owes one of lower priority.

    The session graph (processes as nodes, channels as edges) may have
    cycles: two processes may share several channels, and channels may form
    a ring. `Inv` asks only that each channel has two distinct holders, that
    only they act on it, and that they agree on the exchanges still to do on
    it (`same`, what duality gives).

    `progress`: a state is final or can step. Proof: take the process whose
    next action `a` has the least priority among the next actions. Its
    partner on `a`'s channel owes `a` too; the partner's next action is at
    most `pr a` (its actions increase) and not below it (minimality), so it
    is `a` itself (priorities increase strictly): both can do `a`.
    `preserve` keeps `Inv` along steps, so `deadlock_free` holds for every
    reachable state.

    `chk` is the executable check over a finite network with a priority
    function; `chk_sound` gives `Inv`. `decide` computes priorities as
    longest paths by relaxation and then runs `chk`, so its soundness does
    not depend on the relaxation. `cycle_refutes`: an action cycle (each
    action right before the next in some process, the last before the first)
    admits no priorities, so `chk` fails for every `pr`. -/
namespace RCE.SessionPrio

/-- the `t`-th exchange on channel `c`, as `(c, t)` -/
abbrev Act := Nat × Nat

/-- the actions each process still has to do -/
abbrev State := Nat → List Act

/-- the two holders of each channel -/
abbrev Ends := Nat → Nat × Nat

def filt (c : Nat) (l : List Act) : List Act := l.filter (fun a => a.1 == c)

structure Inv (E : Ends) (pr : Act → Nat) (s : State) : Prop where
  holds : ∀ p a, a ∈ s p → p = (E a.1).1 ∨ p = (E a.1).2
  apart : ∀ c, (E c).1 ≠ (E c).2
  same : ∀ c, filt c (s (E c).1) = filt c (s (E c).2)
  incr : ∀ p, (s p).Pairwise (fun x y => pr x < pr y)

def upd (s : State) (p : Nat) (l : List Act) : State := fun i => if i = p then l else s i

/-- two distinct processes whose next action is the same exchange do it -/
inductive Step (s : State) : State → Prop
  | sync {p q : Nat} {a : Act} {l1 l2 : List Act} :
      p ≠ q → s p = a :: l1 → s q = a :: l2 → Step s (upd (upd s p l1) q l2)

def Final (s : State) : Prop := ∀ p, s p = []

inductive Reach (s0 : State) : State → Prop
  | refl : Reach s0 s0
  | step {s s'} : Reach s0 s → Step s s' → Reach s0 s'

variable {E : Ends} {pr : Act → Nat}

theorem mem_filt_self {a : Act} {l : List Act} (h : a ∈ l) : a ∈ filt a.1 l :=
  List.mem_filter.2 ⟨h, by simp⟩

/-- the other holder of the channel of a pending action owes it too -/
theorem partner {s : State} (I : Inv E pr s) {p : Nat} {a : Act} {l : List Act}
    (h : s p = a :: l) : ∃ q, q ≠ p ∧ a ∈ s q := by
  have ha : a ∈ s p := h ▸ List.mem_cons_self a l
  have hf := mem_filt_self ha
  rcases I.holds p a ha with e | e
  · refine ⟨(E a.1).2, fun h' => I.apart a.1 (e.symm.trans h'.symm), ?_⟩
    rw [e, I.same a.1] at hf
    exact (List.mem_filter.1 hf).1
  · refine ⟨(E a.1).1, fun h' => I.apart a.1 (h'.trans e), ?_⟩
    rw [e, ← I.same a.1] at hf
    exact (List.mem_filter.1 hf).1

theorem descend {s : State} (I : Inv E pr s) :
    ∀ v p a l, s p = a :: l → pr a < v → ∃ s', Step s s' := by
  intro v
  induction v with
  | zero => intro _ _ _ _ h; exact absurd h (Nat.not_lt_zero _)
  | succ v ih =>
    intro p a l hp hv
    have ⟨q, hqp, hq⟩ := partner I hp
    match hs : s q with
    | [] => rw [hs] at hq; cases hq
    | b :: l2 =>
      by_cases hb : b = a
      · subst hb; exact ⟨_, .sync (Ne.symm hqp) hp hs⟩
      · rw [hs] at hq
        have hin : a ∈ l2 := by
          rcases List.mem_cons.1 hq with e | e
          · exact absurd e.symm hb
          · exact e
        have hlt : pr b < pr a := by
          have := I.incr q; rw [hs] at this
          exact (List.pairwise_cons.1 this).1 a hin
        exact ih q b l2 hs (by omega)

theorem progress {s : State} (I : Inv E pr s) : Final s ∨ ∃ s', Step s s' := by
  by_cases hf : ∀ p, s p = []
  · exact Or.inl hf
  · right
    have ⟨p, hp⟩ := Classical.not_forall.1 hf
    match hs : s p with
    | [] => exact absurd hs hp
    | a :: l => exact descend I (pr a + 1) p a l hs (Nat.lt_succ_self _)

theorem filt_tail_ne {c : Nat} {a : Act} {l : List Act} (h : a.1 ≠ c) :
    filt c (a :: l) = filt c l := by
  simp [filt, List.filter_cons, h]

theorem filt_tail_eq {c : Nat} {a : Act} {l : List Act} (h : a.1 = c) :
    filt c (a :: l) = a :: filt c l := by
  simp [filt, List.filter_cons, h]

theorem preserve {s s' : State} (I : Inv E pr s) (st : Step s s') : Inv E pr s' := by
  cases st with
  | @sync p q a l1 l2 hpq hp hq =>
    have v : ∀ i, upd (upd s p l1) q l2 i = if i = q then l2 else if i = p then l1 else s i := by
      intro i; simp [upd]
    constructor
    · intro i b hb
      rw [v] at hb
      split at hb
      · next e => subst e; exact I.holds _ b (by rw [hq]; exact List.mem_cons_of_mem _ hb)
      · split at hb
        · next e => subst e; exact I.holds _ b (by rw [hp]; exact List.mem_cons_of_mem _ hb)
        · exact I.holds i b hb
    · exact I.apart
    · intro c
      rw [v, v]
      by_cases hc : a.1 = c
      · -- the step is on c: its holders are p and q
        have hpE := I.holds p a (by rw [hp]; exact List.mem_cons_self _ _)
        have hqE := I.holds q a (by rw [hq]; exact List.mem_cons_self _ _)
        have hs := I.same c
        subst hc
        have ap := I.apart a.1
        rcases hpE with e1 | e1 <;> rcases hqE with e2 | e2
        · exact absurd (e1.trans e2.symm) hpq
        · rw [← e1, ← e2, hp, hq, filt_tail_eq rfl, filt_tail_eq rfl] at hs
          have hs := (List.cons.inj hs).2
          rw [← e1, ← e2]; simp [hpq, Ne.symm hpq, hs]
        · rw [← e1, ← e2, hp, hq, filt_tail_eq rfl, filt_tail_eq rfl] at hs
          have hs := (List.cons.inj hs).2
          rw [← e1, ← e2]; simp [hpq, Ne.symm hpq, hs]
        · exact absurd (e1.trans e2.symm) hpq
      · -- the step is on another channel: c's view is unchanged
        have keep : ∀ i, filt c (if i = q then l2 else if i = p then l1 else s i) = filt c (s i) := by
          intro i
          split
          · next e => subst e; rw [hq, filt_tail_ne hc]
          · split
            · next e => subst e; rw [hp, filt_tail_ne hc]
            · rfl
        rw [keep, keep]; exact I.same c
    · intro i
      rw [v]
      split
      · next e => subst e; have := I.incr i; rw [hq] at this; exact (List.pairwise_cons.1 this).2
      · split
        · next e => subst e; have := I.incr i; rw [hp] at this; exact (List.pairwise_cons.1 this).2
        · exact I.incr i

theorem reach_inv {s0 s : State} (I : Inv E pr s0) (r : Reach s0 s) : Inv E pr s := by
  induction r with
  | refl => exact I
  | step _ st ih => exact preserve ih st

/-- T28p -/
theorem deadlock_free {s0 s : State} (I : Inv E pr s0) (r : Reach s0 s) :
    Final s ∨ ∃ s', Step s s' :=
  progress (reach_inv I r)

/-! ## The executable check -/

def toState (L : List (List Act)) : State := fun p => L.getD p []
def toEnds (es : List (Nat × Nat)) : Ends := fun c => es.getD c (0, 1)

def incrB (pr : Act → Nat) : List Act → Bool
  | [] => true
  | x :: l => l.all (fun y => decide (pr x < pr y)) && incrB pr l

theorem incrB_sound {pr : Act → Nat} : ∀ {l : List Act}, incrB pr l = true → l.Pairwise (fun x y => pr x < pr y)
  | [], _ => List.Pairwise.nil
  | x :: l, h => by
    simp only [incrB, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
    exact List.Pairwise.cons h.1 (incrB_sound h.2)

def chk (L : List (List Act)) (es : List (Nat × Nat)) (pr : Act → Nat) : Bool :=
  let E := toEnds es
  let S := toState L
  (List.range L.length).all (fun p => (S p).all (fun a =>
      decide (a.1 < es.length) && (p == (E a.1).1 || p == (E a.1).2))) &&
  es.all (fun e => e.1 != e.2) &&
  (List.range es.length).all (fun c => filt c (S (E c).1) == filt c (S (E c).2)) &&
  L.all (incrB pr)

theorem toState_out {L : List (List Act)} {p : Nat} (h : L.length ≤ p) : toState L p = [] := by
  simp [toState, List.getD, List.getElem?_eq_none h]

theorem toEnds_out {es : List (Nat × Nat)} {c : Nat} (h : es.length ≤ c) : toEnds es c = (0, 1) := by
  simp [toEnds, List.getD, List.getElem?_eq_none h]

theorem toState_mem {L : List (List Act)} {p : Nat} (h : p < L.length) : toState L p ∈ L := by
  simp [toState, List.getD, List.getElem?_eq_getElem h]
  exact List.getElem_mem _ _ _

theorem toEnds_mem {es : List (Nat × Nat)} {c : Nat} (h : c < es.length) : toEnds es c ∈ es := by
  simp [toEnds, List.getD, List.getElem?_eq_getElem h]
  exact List.getElem_mem _ _ _

theorem filt_out {L : List (List Act)} {es : List (Nat × Nat)}
    (hch : ∀ p a, a ∈ toState L p → a.1 < es.length) {c : Nat} (hc : es.length ≤ c) (p : Nat) :
    filt c (toState L p) = [] := by
  apply List.eq_nil_iff_forall_not_mem.2
  intro a ha
  have ⟨h1, h2⟩ := List.mem_filter.1 ha
  have := hch p a h1
  simp at h2; omega

theorem chk_sound {L : List (List Act)} {es : List (Nat × Nat)} {pr : Act → Nat}
    (h : chk L es pr = true) : Inv (toEnds es) pr (toState L) := by
  simp only [chk, Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  obtain ⟨⟨⟨hh, ha⟩, hs⟩, hi⟩ := h
  have hp : ∀ p a, a ∈ toState L p → a.1 < es.length ∧ (p = (toEnds es a.1).1 ∨ p = (toEnds es a.1).2) := by
    intro p a hm
    by_cases hp : p < L.length
    · have := hh p hp a hm
      simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true, beq_iff_eq] at this
      exact this
    · rw [toState_out (by omega)] at hm; cases hm
  refine ⟨fun p a hm => (hp p a hm).2, ?_, ?_, ?_⟩
  · intro c
    by_cases hc : c < es.length
    · have := ha _ (toEnds_mem hc); simpa using this
    · rw [toEnds_out (by omega)]; decide
  · intro c
    by_cases hc : c < es.length
    · have := hs c hc; simpa using this
    · rw [filt_out (fun p a hm => (hp p a hm).1) (by omega), filt_out (fun p a hm => (hp p a hm).1) (by omega)]
  · intro p
    by_cases hpl : p < L.length
    · exact incrB_sound (hi _ (toState_mem hpl))
    · rw [toState_out (by omega)]; exact List.Pairwise.nil

/-- any priorities that pass `chk` make the network deadlock free -/
theorem chk_deadlock_free {L : List (List Act)} {es : List (Nat × Nat)} {pr : Act → Nat}
    (h : chk L es pr = true) {s : State} (r : Reach (toState L) s) : Final s ∨ ∃ s', Step s s' :=
  deadlock_free (chk_sound h) r

/-! ## Computing priorities -/

def look (m : List (Act × Nat)) (a : Act) : Nat :=
  match m.find? (fun e => e.1 == a) with
  | some e => e.2
  | none => 0

def bump (m : List (Act × Nat)) (a : Act) (v : Nat) : List (Act × Nat) :=
  if look m a < v then (a, v) :: m.filter (fun e => e.1 != a) else m

/-- one sweep: every action is above its predecessor in each process -/
def sweep (m : List (Act × Nat)) : List (List Act) → List (Act × Nat)
  | [] => m
  | l :: ls => sweep (go m l) ls
where
  go (m : List (Act × Nat)) : List Act → List (Act × Nat)
    | x :: y :: l => go (bump m y (look m x + 1)) (y :: l)
    | _ => m

/-- longest-path priorities after one sweep per action (enough on an
    acyclic order) -/
def prOf (L : List (List Act)) : Act → Nat :=
  let n := (L.map List.length).foldl (· + ·) 0
  look ((List.range (n + 1)).foldl (fun m _ => sweep m L) [])

/-- the decision used by the checker link: priorities exist (computed by
    `prOf`) and pass `chk` -/
def decideP (L : List (List Act)) (es : List (Nat × Nat)) : Bool := chk L es (prOf L)

theorem decideP_sound {L : List (List Act)} {es : List (Nat × Nat)} (h : decideP L es = true)
    {s : State} (r : Reach (toState L) s) : Final s ∨ ∃ s', Step s s' :=
  chk_deadlock_free h r

/-! ## Cycles refute priorities -/

/-- `x` comes right before `y` in some process -/
def Before (L : List (List Act)) (x y : Act) : Prop := ∃ l ∈ L, ∃ i, l[i]? = some x ∧ l[i + 1]? = some y

theorem before_lt {L : List (List Act)} {pr : Act → Nat} (hi : ∀ l ∈ L, l.Pairwise (fun x y => pr x < pr y))
    {x y : Act} (h : Before L x y) : pr x < pr y := by
  obtain ⟨l, hl, i, hx, hy⟩ := h
  have hp := hi l hl
  have hi1 : i + 1 < l.length := by
    rcases Nat.lt_or_ge (i + 1) l.length with h | h
    · exact h
    · rw [List.getElem?_eq_none h] at hy; cases hy
  have e1 : l[i] = x := by rw [List.getElem?_eq_getElem (by omega)] at hx; exact Option.some.inj hx
  have e2 : l[i + 1] = y := by rw [List.getElem?_eq_getElem hi1] at hy; exact Option.some.inj hy
  rw [← e1, ← e2]
  exact List.pairwise_iff_getElem.1 hp i (i + 1) (by omega) hi1 (by omega)

/-- a chain `x0 → x1 → … → xk` of `Before` steps -/
def Chain (L : List (List Act)) : Act → List Act → Prop
  | _, [] => True
  | x, y :: ys => Before L x y ∧ Chain L y ys

theorem chain_le {L : List (List Act)} {pr : Act → Nat} (hi : ∀ l ∈ L, l.Pairwise (fun x y => pr x < pr y)) :
    ∀ {x : Act} {ys : List Act}, Chain L x ys → ∀ z, (ys.getLast? = some z) → pr x < pr z
  | _, [], _, _, h => by cases h
  | x, y :: ys, ⟨hb, hc⟩, z, hz => by
    have h1 := before_lt hi hb
    cases ys with
    | nil => simp at hz; subst hz; exact h1
    | cons w ws =>
      have : (w :: ws).getLast? = some z := by
        simp only [List.getLast?_cons_cons] at hz; exact hz
      exact Nat.lt_trans h1 (chain_le hi hc z this)

/-- an action cycle: no priorities make every process increasing -/
theorem cycle_refutes {L : List (List Act)} {x : Act} {ys : List Act}
    (hc : Chain L x ys) (hl : ys.getLast? = some x) (pr : Act → Nat) :
    ¬ ∀ l ∈ L, l.Pairwise (fun a b => pr a < pr b) := by
  intro hi
  exact Nat.lt_irrefl _ (chain_le hi hc x hl)

theorem chk_incr {L : List (List Act)} {es : List (Nat × Nat)} {pr : Act → Nat}
    (h : chk L es pr = true) : ∀ l ∈ L, l.Pairwise (fun a b => pr a < pr b) := by
  simp only [chk, Bool.and_eq_true, List.all_eq_true] at h
  exact fun l hl => incrB_sound (h.2 l hl)

/-- with a cycle, `chk` fails whatever the priorities -/
theorem cycle_chk_false {L : List (List Act)} {es : List (Nat × Nat)} {x : Act} {ys : List Act}
    (hc : Chain L x ys) (hl : ys.getLast? = some x) (pr : Act → Nat) : chk L es pr = false := by
  cases h : chk L es pr
  · rfl
  · exact absurd (chk_incr h) (cycle_refutes hc hl pr)

/-! ## A cyclic network the tree rule does not admit

    Three processes in a ring: 0 and 1 share channel 0, 1 and 2 share
    channel 1, 2 and 0 share channel 2. Each process first serves the
    channel to its left, then the one to its right, except process 0, which
    starts the ring. -/
def ring : List (List Act) := [[(0, 0), (2, 0)], [(0, 0), (1, 0)], [(1, 0), (2, 0)]]
def ringEnds : List (Nat × Nat) := [(0, 1), (1, 2), (2, 0)]

example : decideP ring ringEnds = true := by decide

/-- the same ring with process 0 serving channel 2 first deadlocks, and
    the cycle (0,0) → (1,0) → (2,0) → (0,0) refutes every priority -/
def ringBad : List (List Act) := [[(2, 0), (0, 0)], [(0, 0), (1, 0)], [(1, 0), (2, 0)]]

example (pr : Act → Nat) : chk ringBad ringEnds pr = false := by
  apply cycle_chk_false (x := (0, 0)) (ys := [(1, 0), (2, 0), (0, 0)]) _ rfl
  refine ⟨⟨_, List.mem_cons_of_mem _ (List.mem_cons_self _ _), 0, rfl, rfl⟩,
    ⟨_, List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_self _ _)), 0, rfl, rfl⟩,
    ⟨_, List.mem_cons_self _ _, 0, rfl, rfl⟩, trivial⟩

end RCE.SessionPrio
