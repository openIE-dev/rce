import RCE.SessionRec
/-! Deadlock freedom for a tree-shaped session graph (binary sessions).

    A network has `n` processes and `m` channels. Each channel `c` has two
    ends, side `true` and side `false`; `holder c s` is the process that
    owns end `s` and `ty c s` its protocol (`RCE.SessionRec.P`). A process
    is either blocked on one of its endpoints (`wait p = some (c, s)`, the
    protocol of that end is not `done`) or has finished every endpoint it
    holds (`wait p = none`). This abstracts any sequential body: a process
    that is not finished is about to act on some endpoint.

    The session graph has processes as nodes and channels as edges. `Tree`
    says it is a forest given as depths: the two ends of every channel are
    one level apart and the channel is the parent edge of its deeper end
    (so a process has at most one parent edge, and no channel joins a
    process to itself).

    `deadlock_free`: if every channel's ends are compatible (`Comp`,
    closed) then either every process has finished or some channel has
    both ends blocked on it, and those two ends can take a step (`Red`).
    With `t28r_preserve` this gives progress for the whole network as
    long as the graph stays a tree.

    Proof idea: if no channel has both ends ready, the partner of a
    blocked process is blocked on another channel. From a blocked process
    walk to a partner; going up the tree the depth falls, and once a step
    goes down to a child the child's other channels all lead further down.
    Depths are bounded, so the walk cannot go on: some channel is ready.
    Two channels between the same two processes (a cycle) break this; see
    Gay and Hole's example `x?.y! | y?.x!`. -/
namespace RCE.SessionDead
open RCE.SessionRec

theorem comp_done_r : ∀ {a : P}, Comp a .done → a = .done
  | _, .done => rfl

theorem comp_done_l : ∀ {b : P}, Comp .done b → b = .done
  | _, .done => rfl

structure Net (n m : Nat) where
  holder : Fin m → Bool → Fin n
  ty : Fin m → Bool → P
  depth : Fin n → Nat
  parent : Fin n → Option (Fin m)
  wait : Fin n → Option (Fin m × Bool)

variable {n m : Nat}

def Tree (N : Net n m) : Prop :=
  ∀ c : Fin m, ∃ s : Bool,
    N.depth (N.holder c (!s)) = N.depth (N.holder c s) + 1 ∧ N.parent (N.holder c (!s)) = some c

def Compat (N : Net n m) : Prop :=
  ∀ c, Comp (N.ty c true) (N.ty c false) ∧ closedAt 0 (N.ty c true)

def Waits (N : Net n m) : Prop :=
  ∀ p, (∀ c s, N.wait p = some (c, s) → N.holder c s = p ∧ N.ty c s ≠ .done) ∧
    (N.wait p = none → ∀ c s, N.holder c s = p → N.ty c s = .done)

def Ready (N : Net n m) (c : Fin m) : Prop :=
  N.wait (N.holder c true) = some (c, true) ∧ N.wait (N.holder c false) = some (c, false)

/-- the partner of a non-finished end is non-finished -/
theorem other_live {N : Net n m} (hc : Compat N) {c : Fin m} {s : Bool}
    (h : N.ty c s ≠ .done) : N.ty c (!s) ≠ .done := by
  intro h'
  cases s with
  | true => exact h (comp_done_r (h' ▸ (hc c).1))
  | false => exact h (comp_done_l (h' ▸ (hc c).1))

theorem holders_differ {N : Net n m} (ht : Tree N) (c : Fin m) (s : Bool) :
    N.holder c s ≠ N.holder c (!s) := by
  intro e
  have ⟨s0, hd, _⟩ := ht c
  cases s <;> cases s0 <;> simp_all <;> omega

/-- with no ready channel, the partner of a blocked process is blocked on
    another channel -/
theorem next {N : Net n m} (ht : Tree N) (hc : Compat N) (hw : Waits N)
    (nr : ∀ c, ¬ Ready N c) {p : Fin n} {c : Fin m} {s : Bool} (h : N.wait p = some (c, s)) :
    ∃ c' s', N.wait (N.holder c (!s)) = some (c', s') ∧ c' ≠ c := by
  have ⟨hp, hl⟩ := (hw p).1 c s h
  have hl' := other_live hc hl
  cases hq : N.wait (N.holder c (!s)) with
  | none => exact absurd ((hw _).2 hq c (!s) rfl) hl'
  | some w =>
    obtain ⟨c', s'⟩ := w
    refine ⟨c', s', rfl, ?_⟩
    intro e
    rw [e] at hq
    have ⟨hq', _⟩ := (hw _).1 c s' hq
    cases s' <;> cases s
    · exact holders_differ ht c false (by simp_all)
    · exact nr c ⟨hp ▸ h, by simp_all⟩
    · exact nr c ⟨by simp_all, hp ▸ h⟩
    · exact holders_differ ht c true (by simp_all)

/-- `p` is blocked on a channel that leads down to its child -/
def Down (N : Net n m) (p : Fin n) : Prop :=
  ∃ c s, N.wait p = some (c, s) ∧ N.parent (N.holder c (!s)) = some c ∧
    N.depth (N.holder c (!s)) = N.depth p + 1

theorem to_down {N : Net n m} (ht : Tree N) (hc : Compat N) (hw : Waits N)
    (nr : ∀ c, ¬ Ready N c) : ∀ k (p : Fin n), N.depth p ≤ k → (∃ w, N.wait p = some w) →
    ∃ p', Down N p' := by
  intro k
  induction k with
  | zero =>
    intro p hd ⟨⟨c, s⟩, h⟩
    have ⟨hp, _⟩ := (hw p).1 c s h
    have ⟨s0, h1, h2⟩ := ht c
    by_cases e : s0 = s
    · subst e; exact ⟨p, c, s0, h, h2, by rw [hp] at h1; exact h1⟩
    · have e' : s0 = !s := by cases s0 <;> cases s <;> simp_all
      subst e'
      simp only [Bool.not_not] at h1
      rw [hp] at h1
      omega
  | succ k ih =>
    intro p hd ⟨⟨c, s⟩, h⟩
    have ⟨hp, _⟩ := (hw p).1 c s h
    have ⟨s0, h1, h2⟩ := ht c
    have ⟨c', s', hq, _⟩ := next ht hc hw nr h
    by_cases e : s0 = s
    · subst e; exact ⟨p, c, s0, h, h2, by rw [hp] at h1; exact h1⟩
    · have e' : s0 = !s := by cases s0 <;> cases s <;> simp_all
      subst e'
      simp only [Bool.not_not] at h1
      rw [hp] at h1
      exact ih _ (by omega) ⟨_, hq⟩

theorem down_step {N : Net n m} (ht : Tree N) (hc : Compat N) (hw : Waits N)
    (nr : ∀ c, ¬ Ready N c) {p : Fin n} (hd : Down N p) :
    ∃ r, Down N r ∧ N.depth r = N.depth p + 1 := by
  obtain ⟨c, s, h, hpar, hdep⟩ := hd
  have ⟨c', s', hq, ne⟩ := next ht hc hw nr h
  refine ⟨N.holder c (!s), ?_, hdep⟩
  have ⟨hr, _⟩ := (hw _).1 c' s' hq
  have ⟨s1, h1, h2⟩ := ht c'
  by_cases e : s1 = s'
  · subst e; exact ⟨c', s1, hq, h2, by rw [hr] at h1; exact h1⟩
  · have e' : s1 = !s' := by cases s1 <;> cases s' <;> simp_all
    subst e'
    simp only [Bool.not_not] at h2
    rw [hr, hpar] at h2
    exact absurd (Option.some.inj h2).symm ne

theorem bounded : ∀ {n : Nat} (f : Fin n → Nat), ∃ D, ∀ i, f i ≤ D
  | 0, _ => ⟨0, fun i => i.elim0⟩
  | _ + 1, f =>
    let ⟨D, hD⟩ := bounded (fun i => f i.succ)
    ⟨max D (f 0), fun i => Fin.cases (Nat.le_max_right _ _) (fun j => Nat.le_trans (hD j) (Nat.le_max_left _ _)) i⟩

theorem deadlock_free {N : Net n m} (ht : Tree N) (hc : Compat N) (hw : Waits N) :
    (∀ p, N.wait p = none) ∨
      ∃ c, Ready N c ∧ ∃ a' b', Red (N.ty c true) (N.ty c false) a' b' := by
  by_cases hr : ∃ c, Ready N c
  · obtain ⟨c, hrc⟩ := hr
    refine Or.inr ⟨c, hrc, ?_⟩
    have ⟨_, hl⟩ := (hw _).1 c true hrc.1
    rcases t28r_progress (hc c).1 (hc c).2 with ⟨e, _⟩ | st
    · exact absurd e hl
    · exact st
  · have nr : ∀ c, ¬ Ready N c := fun c h => hr ⟨c, h⟩
    refine Or.inl (fun p => ?_)
    cases hp : N.wait p with
    | none => rfl
    | some w =>
      exfalso
      have ⟨p0, h0⟩ := to_down ht hc hw nr _ p (Nat.le_refl _) ⟨w, hp⟩
      have ⟨D, hD⟩ := bounded N.depth
      have deep : ∀ k, ∃ r, Down N r ∧ N.depth p0 + k ≤ N.depth r := by
        intro k
        induction k with
        | zero => exact ⟨p0, h0, Nat.le_refl _⟩
        | succ k ih =>
          obtain ⟨r, hr', hk⟩ := ih
          obtain ⟨r', hr'', e⟩ := down_step ht hc hw nr hr'
          exact ⟨r', hr'', by omega⟩
      obtain ⟨r, _, hk⟩ := deep (D + 1)
      have := hD r
      omega

end RCE.SessionDead
