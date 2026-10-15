/-
  E.2b: quorum intersection, Election Safety and Leader Completeness for the
  replication protocol of `spec/checker/src/raft.rs`, modeled on whole logs.

  Model. Replicas are ids; a quorum is a duplicate-free list of ids below
  `n` with more than half of them. Replica `i` holds `rs i = (lt, log)`,
  where `lt` is the term of the leader that last wrote its log (the
  abstraction of Raft's last log term), and `prom i`, the highest term it
  has promised (durable term and vote). Ghost state records, per term `u`,
  the leader's current log `ldr u`, the log it was elected with `seed u`,
  that log's term `slt u`, its voters `voters u`, and the list `cm` of
  commits. Every run starts from `init`; the steps are
  * `elect u c Q`: every member of quorum `Q` has promised less than `u`
    and its (lt, length) is at most that of candidate `c ∈ Q` (the vote
    rule); the leader of `u` starts from `c`'s log and `Q` promises `u`;
  * `append u x`: the leader of `u` appends an entry to its log;
  * `write u i L`: replica `i` with `prom i ≤ u` (the term fence) takes a
    log `L` with `seed u <+: L <+: ldr u`, never shrinking a log it already
    holds from `u` (log matching keeps agreeing entries);
  * `commit u C Q`: every member of quorum `Q` holds `C` from term `u`.
  Message loss, delay, duplication and reordering only choose which steps
  happen and in what order, so they are covered.

  Proved with no hypotheses beyond reachability: `elect_fresh` (Election
  Safety: a term already won cannot be won again), and
  `leader_completeness`: if `C` is committed at `t`, the log of every leader
  of a term above `t` contains `C`; `committed_in_later_replicas` gives the
  same for every replica last written above `t`. The induction is the one
  Raft's own argument uses: `commit_seed` is strong induction on terms (a
  later leader shares a voter with the committing quorum, and that voter's
  vote forces the leader's starting log to be at least as up to date, so it
  either holds `C` directly or came from an earlier later term, which holds
  `C` by induction), and `Good` is the inductive invariant of all runs.
  Model simplification: a replica that has accepted a write in term `u`
  does not vote in `u`, which removes behaviors and is weaker than Raft's
  votedFor only in liveness.
-/
namespace RCE.Quorum
open List

theorem card_le : ∀ {l S : List Nat}, l.Nodup → (∀ x ∈ l, x ∈ S) → l.length ≤ S.length
  | [], _, _, _ => Nat.zero_le _
  | a :: t, S, hl, hs => by
    rw [nodup_cons] at hl
    have ha : a ∈ S := hs a (mem_cons_self a t)
    have h2 : ∀ x ∈ t, x ∈ S.erase a := by
      intro x hx
      have hne : x ≠ a := fun h => hl.1 (h ▸ hx)
      exact (mem_erase_of_ne hne).2 (hs x (mem_cons_of_mem a hx))
    have ih := card_le hl.2 h2
    rw [length_erase_of_mem ha] at ih
    have hp : 0 < S.length := length_pos.2 (ne_nil_of_mem ha)
    simp only [length_cons]
    omega

def IsQuorum (n : Nat) (q : List Nat) : Prop :=
  q.Nodup ∧ (∀ x ∈ q, x < n) ∧ n < 2 * q.length

theorem quorum_intersect {n : Nat} {q₁ q₂ : List Nat} (h₁ : IsQuorum n q₁) (h₂ : IsQuorum n q₂) :
    ∃ x, x ∈ q₁ ∧ x ∈ q₂ := by
  apply Classical.byContradiction
  intro hno
  have disj : ∀ a ∈ q₁, ∀ b ∈ q₂, a ≠ b := by
    intro a ha b hb hab
    exact hno ⟨a, ha, hab ▸ hb⟩
  have nd : (q₁ ++ q₂).Nodup := by
    unfold Nodup
    rw [pairwise_append]
    exact ⟨h₁.1, h₂.1, disj⟩
  have sub : ∀ x ∈ q₁ ++ q₂, x ∈ range n := by
    intro x hx
    rw [mem_range]
    rcases mem_append.1 hx with h | h
    · exact h₁.2.1 x h
    · exact h₂.2.1 x h
  have := card_le nd sub
  rw [length_append, length_range] at this
  have := h₁.2.2
  have := h₂.2.2
  omega

/-- Election Safety: each replica votes for at most one candidate in a term
    (the durable vote), so two candidates that each hold a quorum of votes
    in the same term are the same candidate. -/
theorem election_safety {n : Nat} (vote : Nat → Option Nat) {q₁ q₂ : List Nat} {c₁ c₂ : Nat}
    (h₁ : IsQuorum n q₁) (h₂ : IsQuorum n q₂)
    (v₁ : ∀ v ∈ q₁, vote v = some c₁) (v₂ : ∀ v ∈ q₂, vote v = some c₂) : c₁ = c₂ := by
  obtain ⟨x, hx₁, hx₂⟩ := quorum_intersect h₁ h₂
  have := (v₁ x hx₁).symm.trans (v₂ x hx₂)
  exact Option.some.inj this


section Protocol
variable {β : Type}

structure Rep (β : Type) where
  lt : Nat
  log : List β

structure Commit (β : Type) where
  C : List β
  t : Nat
  Q : List Nat

structure St (β : Type) where
  rs : Nat → Rep β
  prom : Nat → Nat
  ldr : Nat → Option (List β)
  seed : Nat → List β
  slt : Nat → Nat
  voters : Nat → List Nat
  cm : List (Commit β)

def init : St β where
  rs := fun _ => ⟨0, []⟩
  prom := fun _ => 0
  ldr := fun w => if w = 0 then some [] else none
  seed := fun _ => []
  slt := fun _ => 0
  voters := fun _ => []
  cm := []

/-- `r` is at most `c` by (lt, length): the vote rule. -/
def UpLe (r c : Rep β) : Prop := r.lt < c.lt ∨ (r.lt = c.lt ∧ r.log.length ≤ c.log.length)

def Upto (r : Rep β) (lt : Nat) (S : List β) : Prop := r.lt < lt ∨ (r.lt = lt ∧ r.log <+: S)

inductive Step (n : Nat) : St β → St β → Prop
  | elect (s : St β) (u c : Nat) (Q : List Nat) :
      0 < u → IsQuorum n Q → c ∈ Q → (∀ j ∈ Q, s.prom j < u) → (∀ j ∈ Q, UpLe (s.rs j) (s.rs c)) →
      Step n s { s with
        ldr := fun w => if w = u then some (s.rs c).log else s.ldr w,
        seed := fun w => if w = u then (s.rs c).log else s.seed w,
        slt := fun w => if w = u then (s.rs c).lt else s.slt w,
        voters := fun w => if w = u then Q else s.voters w,
        prom := fun j => if j ∈ Q then u else s.prom j }
  | append (s : St β) (u : Nat) (L : List β) (x : β) :
      s.ldr u = some L → Step n s { s with ldr := fun w => if w = u then some (L ++ [x]) else s.ldr w }
  | write (s : St β) (u i : Nat) (Lu L : List β) :
      s.ldr u = some Lu → s.prom i ≤ u → s.seed u <+: L → L <+: Lu →
      ((s.rs i).lt = u → (s.rs i).log.length ≤ L.length) →
      Step n s { s with rs := fun j => if j = i then ⟨u, L⟩ else s.rs j,
                        prom := fun j => if j = i then u else s.prom j }
  | commit (s : St β) (u : Nat) (C : List β) (Q : List Nat) :
      IsQuorum n Q → (∀ j ∈ Q, (s.rs j).lt = u ∧ C <+: (s.rs j).log) →
      Step n s { s with cm := ⟨C, u, Q⟩ :: s.cm }

inductive Reach (n : Nat) : St β → Prop
  | init : Reach n init
  | step {s s' : St β} : Reach n s → Step n s s' → Reach n s'

def I1 (s : St β) : Prop :=
  ∀ i, ∃ L, s.ldr (s.rs i).lt = some L ∧ s.seed (s.rs i).lt <+: (s.rs i).log ∧ (s.rs i).log <+: L
def I2 (s : St β) : Prop := ∀ i, (s.rs i).lt ≤ s.prom i
def I3 (s : St β) : Prop := ∀ u L, s.ldr u = some L → s.seed u <+: L
def I4 (s : St β) : Prop :=
  ∀ u L, s.ldr u = some L → 0 < u →
    s.slt u < u ∧ (∃ L', s.ldr (s.slt u) = some L') ∧ s.seed (s.slt u) <+: s.seed u
def I5 (n : Nat) (s : St β) : Prop :=
  ∀ u L, s.ldr u = some L → 0 < u → IsQuorum n (s.voters u) ∧
    ∀ j ∈ s.voters u, u ≤ s.prom j ∧ ((s.rs j).lt < u → Upto (s.rs j) (s.slt u) (s.seed u))
def CI (n : Nat) (s : St β) (e : Commit β) : Prop :=
  IsQuorum n e.Q ∧ (∃ L, s.ldr e.t = some L ∧ e.C <+: L) ∧
  (∀ u L, e.t < u → s.ldr u = some L → e.C <+: s.seed u) ∧
  (∀ i ∈ e.Q, e.t ≤ (s.rs i).lt ∧ ((s.rs i).lt = e.t → e.C <+: (s.rs i).log))

def Good (n : Nat) (s : St β) : Prop :=
  I1 s ∧ I2 s ∧ I3 s ∧ I4 s ∧ I5 n s ∧ ∀ e ∈ s.cm, CI n s e

theorem good_init (n : Nat) : Good n (init : St β) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i; exact ⟨[], by simp [init], by simp [init], by simp [init]⟩
  · intro i; simp [init]
  · intro u L h; simp [init]
  · intro u L h hu; simp [init] at h; omega
  · intro u L h hu; simp [init] at h; omega
  · intro e he; simp [init] at he

/-- Election Safety: a term that has a leader cannot be won again, because
    its voters promised it and any new quorum shares one of them. -/
theorem elect_fresh {n : Nat} {s : St β} (g : Good n s) {u : Nat} {Q : List Nat}
    (hu : 0 < u) (hQ : IsQuorum n Q) (hp : ∀ j ∈ Q, s.prom j < u) : s.ldr u = none := by
  cases h : s.ldr u with
  | none => rfl
  | some L =>
    obtain ⟨hq, hv⟩ := g.2.2.2.2.1 u L h hu
    obtain ⟨x, hx1, hx2⟩ := quorum_intersect hQ hq
    have := hp x hx1; have := (hv x hx2).1; omega

theorem commit_seed {n : Nat} {s : St β} (g : Good n s) {C : List β} {t : Nat} {Q : List Nat}
    (hQ : IsQuorum n Q) (hc : ∀ j ∈ Q, (s.rs j).lt = t ∧ C <+: (s.rs j).log) :
    ∀ u L, t < u → s.ldr u = some L → C <+: s.seed u := by
  intro u
  induction u using Nat.strongRecOn with
  | ind u ih =>
    intro L htu hL
    have hu0 : 0 < u := by omega
    obtain ⟨hq2, hv⟩ := g.2.2.2.2.1 u L hL hu0
    obtain ⟨v, hv1, hv2⟩ := quorum_intersect hQ hq2
    obtain ⟨hvt, hvC⟩ := hc v hv1
    obtain ⟨hslt, ⟨L', hL'⟩, hss⟩ := g.2.2.2.1 u L hL hu0
    rcases (hv v hv2).2 (by omega) with hlt | ⟨_, hpre⟩
    · exact (ih (s.slt u) hslt L' (by omega) hL').trans hss
    · exact hvC.trans hpre

theorem pre_same {s : St β} (h1 : I1 s) {i j : Nat} (he : (s.rs i).lt = (s.rs j).lt)
    (hl : (s.rs i).log.length ≤ (s.rs j).log.length) : (s.rs i).log <+: (s.rs j).log := by
  obtain ⟨Li, hi, _, hi3⟩ := h1 i
  obtain ⟨Lj, hj, _, hj3⟩ := h1 j
  rw [he, hj] at hi
  cases hi
  exact prefix_of_prefix_length_le hi3 hj3 hl

theorem good_elect {n : Nat} {s : St β} (g : Good n s) (u c : Nat) (Q : List Nat)
    (hu : 0 < u) (hQ : IsQuorum n Q) (hc : c ∈ Q) (hp : ∀ j ∈ Q, s.prom j < u)
    (hle : ∀ j ∈ Q, UpLe (s.rs j) (s.rs c)) :
    Good n { s with
        ldr := fun w => if w = u then some (s.rs c).log else s.ldr w,
        seed := fun w => if w = u then (s.rs c).log else s.seed w,
        slt := fun w => if w = u then (s.rs c).lt else s.slt w,
        voters := fun w => if w = u then Q else s.voters w,
        prom := fun j => if j ∈ Q then u else s.prom j } := by
  obtain ⟨h1, h2, h3, h4, h5, hcm⟩ := g
  have hn : s.ldr u = none := elect_fresh ⟨h1, h2, h3, h4, h5, hcm⟩ hu hQ hp
  have ne_of : ∀ w L, s.ldr w = some L → w ≠ u := by
    intro w L h e; rw [e, hn] at h; cases h
  have hcu : (s.rs c).lt < u := Nat.lt_of_le_of_lt (h2 c) (hp c hc)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i
    obtain ⟨L, a, b, d⟩ := h1 i
    have := ne_of _ _ a
    exact ⟨L, by simp [this, a], by simpa [this] using b, d⟩
  · intro i
    by_cases hi : i ∈ Q
    · have := h2 i; have := hp i hi; simp [hi]; omega
    · simpa [hi] using h2 i
  · intro w L h
    by_cases hw : w = u
    · subst hw; simp at h; subst h; simp
    · simp [hw] at h ⊢; exact h3 w L h
  · intro w L h hw0
    by_cases hw : w = u
    · subst hw
      obtain ⟨Lc, a, b, _⟩ := h1 c
      have := ne_of _ _ a
      exact ⟨by simpa using hcu, ⟨Lc, by simp [this, a]⟩, by simpa [this] using b⟩
    · simp [hw] at h ⊢
      obtain ⟨a, ⟨L', b⟩, d⟩ := h4 w L h hw0
      have := ne_of _ _ b
      exact ⟨a, ⟨L', by simp [this, b]⟩, by simpa [this] using d⟩
  · intro w L h hw0
    by_cases hw : w = u
    · subst hw
      refine ⟨by simpa using hQ, ?_⟩
      intro j hj
      simp at hj
      refine ⟨by simp [hj], ?_⟩
      intro _
      simp
      rcases hle j hj with a | ⟨a, b⟩
      · exact Or.inl a
      · exact Or.inr ⟨a, pre_same h1 a b⟩
    · simp [hw] at h ⊢
      obtain ⟨a, b⟩ := h5 w L h hw0
      refine ⟨a, ?_⟩
      intro j hj
      obtain ⟨d, e⟩ := b j hj
      refine ⟨?_, e⟩
      by_cases hjq : j ∈ Q
      · have := hp j hjq; simp [hjq]; omega
      · simpa [hjq] using d
  · intro e he
    obtain ⟨q, ⟨Lt, a, b⟩, d, f⟩ := hcm e he
    have hte := ne_of _ _ a
    refine ⟨q, ⟨Lt, by simp [hte, a], b⟩, ?_, f⟩
    intro w L htw h
    by_cases hw : w = u
    · subst hw
      simp
      obtain ⟨v, hv1, hv2⟩ := quorum_intersect q hQ
      obtain ⟨hvt, hvC⟩ := f v hv1
      have upC : e.t < (s.rs c).lt → e.C <+: (s.rs c).log := by
        intro hlt
        obtain ⟨Lc, x, y, _⟩ := h1 c
        exact (d _ _ hlt x).trans y
      rcases hle v hv2 with x | ⟨x, y⟩
      · exact upC (by omega)
      · by_cases ht : (s.rs v).lt = e.t
        · exact (hvC ht).trans (pre_same h1 x y)
        · exact upC (by omega)
    · simp [hw] at h ⊢; exact d w L htw h

theorem good_append {n : Nat} {s : St β} (g : Good n s) (u : Nat) (L : List β) (x : β)
    (hL : s.ldr u = some L) :
    Good n { s with ldr := fun w => if w = u then some (L ++ [x]) else s.ldr w } := by
  obtain ⟨h1, h2, h3, h4, h5, hcm⟩ := g
  have back : ∀ w L', (if w = u then some (L ++ [x]) else s.ldr w) = some L' →
      ∃ L0, s.ldr w = some L0 ∧ L0 <+: L' := by
    intro w L' h
    by_cases hw : w = u
    · subst hw; simp at h; subst h; exact ⟨L, hL, prefix_append L [x]⟩
    · simp [hw] at h; exact ⟨L', h, prefix_refl L'⟩
  have fwd : ∀ w L0, s.ldr w = some L0 →
      ∃ L', (if w = u then some (L ++ [x]) else s.ldr w) = some L' ∧ L0 <+: L' := by
    intro w L0 h
    by_cases hw : w = u
    · subst hw; rw [hL] at h; cases h; exact ⟨L ++ [x], by simp, prefix_append L [x]⟩
    · exact ⟨L0, by simp [hw, h], prefix_refl L0⟩
  refine ⟨?_, h2, ?_, ?_, ?_, ?_⟩
  · intro i
    obtain ⟨L0, a, b, d⟩ := h1 i
    obtain ⟨L', e, f⟩ := fwd _ _ a
    exact ⟨L', e, b, d.trans f⟩
  · intro w L' h
    obtain ⟨L0, a, b⟩ := back w L' h
    exact (h3 w L0 a).trans b
  · intro w L' h hw0
    obtain ⟨L0, a, _⟩ := back w L' h
    obtain ⟨p, ⟨L1, q⟩, r⟩ := h4 w L0 a hw0
    obtain ⟨L2, e, _⟩ := fwd _ _ q
    exact ⟨p, ⟨L2, e⟩, r⟩
  · intro w L' h hw0
    obtain ⟨L0, a, _⟩ := back w L' h
    exact h5 w L0 a hw0
  · intro e he
    obtain ⟨q, ⟨Lt, a, b⟩, d, f⟩ := hcm e he
    obtain ⟨L', a', b'⟩ := fwd _ _ a
    refine ⟨q, ⟨L', a', b.trans b'⟩, ?_, f⟩
    intro w L2 htw h
    obtain ⟨L0, a0, _⟩ := back w L2 h
    exact d w L0 htw a0

theorem good_write {n : Nat} {s : St β} (g : Good n s) (u i : Nat) (Lu L : List β)
    (hLu : s.ldr u = some Lu) (hp : s.prom i ≤ u) (hs : s.seed u <+: L) (hL : L <+: Lu)
    (hgrow : (s.rs i).lt = u → (s.rs i).log.length ≤ L.length) :
    Good n { s with rs := fun j => if j = i then ⟨u, L⟩ else s.rs j,
                    prom := fun j => if j = i then u else s.prom j } := by
  obtain ⟨h1, h2, h3, h4, h5, hcm⟩ := g
  refine ⟨?_, ?_, h3, h4, ?_, ?_⟩
  · intro j
    by_cases hj : j = i
    · subst hj; exact ⟨Lu, by simp [hLu], by simpa using hs, by simpa using hL⟩
    · simpa [hj] using h1 j
  · intro j
    by_cases hj : j = i
    · subst hj; simp
    · simpa [hj] using h2 j
  · intro w L' h hw0
    obtain ⟨a, b⟩ := h5 w L' h hw0
    refine ⟨a, ?_⟩
    intro j hj
    obtain ⟨d, e⟩ := b j hj
    by_cases hji : j = i
    · subst hji
      refine ⟨by simp; omega, ?_⟩
      intro hlt; simp at hlt; omega
    · simpa [hji] using (⟨d, e⟩ : w ≤ s.prom j ∧ ((s.rs j).lt < w → Upto (s.rs j) (s.slt w) (s.seed w)))
  · intro e he
    obtain ⟨q, a, d, f⟩ := hcm e he
    refine ⟨q, a, d, ?_⟩
    intro j hj
    obtain ⟨x, y⟩ := f j hj
    by_cases hji : j = i
    · subst hji
      have := h2 j
      simp
      refine ⟨by omega, ?_⟩
      intro hut
      have hlt : (s.rs j).lt = e.t := by omega
      have hC := y hlt
      obtain ⟨L0, a0, _, c0⟩ := h1 j
      rw [hlt, ← hut, hLu] at a0
      cases a0
      exact hC.trans (prefix_of_prefix_length_le c0 hL (hgrow (by omega)))
    · simpa [hji] using (⟨x, y⟩ : e.t ≤ (s.rs j).lt ∧ ((s.rs j).lt = e.t → e.C <+: (s.rs j).log))

theorem good_commit {n : Nat} {s : St β} (g : Good n s) (u : Nat) (C : List β) (Q : List Nat)
    (hQ : IsQuorum n Q) (hc : ∀ j ∈ Q, (s.rs j).lt = u ∧ C <+: (s.rs j).log) :
    Good n { s with cm := ⟨C, u, Q⟩ :: s.cm } := by
  have hseed := commit_seed g hQ hc
  obtain ⟨h1, h2, h3, h4, h5, hcm⟩ := g
  refine ⟨h1, h2, h3, h4, h5, ?_⟩
  intro e he
  simp only [mem_cons] at he
  rcases he with he | he
  · subst he
    obtain ⟨v, hv, _⟩ := quorum_intersect hQ hQ
    obtain ⟨hvt, hvC⟩ := hc v hv
    obtain ⟨L, a, _, b⟩ := h1 v
    rw [hvt] at a
    refine ⟨hQ, ⟨L, a, hvC.trans b⟩, hseed, ?_⟩
    intro j hj
    obtain ⟨x, y⟩ := hc j hj
    exact ⟨Nat.le_of_eq x.symm, fun _ => y⟩
  · exact hcm e he

theorem good_step {n : Nat} {s s' : St β} (g : Good n s) (h : Step n s s') : Good n s' := by
  cases h with
  | elect u c Q a b d e f => exact good_elect g u c Q a b d e f
  | append u L x a => exact good_append g u L x a
  | write u i Lu L a b d e f => exact good_write g u i Lu L a b d e f
  | commit u C Q a b => exact good_commit g u C Q a b

theorem reach_good {n : Nat} {s : St β} (h : Reach n s) : Good n s := by
  induction h with
  | init => exact good_init n
  | step _ st ih => exact good_step ih st

/-- Leader Completeness: in every reachable state, a prefix committed at
    term `t` is contained in the log of the leader of every term above `t`. -/
theorem leader_completeness {n : Nat} {s : St β} (h : Reach n s) {e : Commit β} (he : e ∈ s.cm)
    {u : Nat} {L : List β} (hu : e.t < u) (hL : s.ldr u = some L) : e.C <+: L := by
  obtain ⟨_, _, h3, _, _, hcm⟩ := reach_good h
  exact ((hcm e he).2.2.1 u L hu hL).trans (h3 u L hL)

/-- Every replica whose log was last written by a leader above `t` holds
    the prefix committed at `t`. -/
theorem committed_in_later_replicas {n : Nat} {s : St β} (h : Reach n s) {e : Commit β}
    (he : e ∈ s.cm) (i : Nat) (hi : e.t < (s.rs i).lt) : e.C <+: (s.rs i).log := by
  obtain ⟨h1, _, _, _, _, hcm⟩ := reach_good h
  obtain ⟨L, a, b, _⟩ := h1 i
  exact ((hcm e he).2.2.1 _ L hi a).trans b

end Protocol
end RCE.Quorum
