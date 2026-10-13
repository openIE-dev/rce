/-! T24 in both directions (Mattern 1988), on the standard model of a run:
    events are (process, index) pairs; `le` is happens-before-or-equal, a
    partial order containing program order; the clock of an event counts,
    per process, the events that happen before or equal it. Then
    `a` happens strictly before `b` iff `V a < V b` (`vc_iff`).
    `RCE.Par.hb_lt` shows the forward direction for the tick rules of the
    runtime; `RCE.TickClock.count` proves that the tick rules compute
    exactly such a counting clock, with `hb_iff`/`hb_lt_iff` there giving
    both directions directly for the tick-rule clock. -/
namespace RCE.VClock

abbrev Ev := Nat × Nat  -- (process, index on that process)

structure Run where
  le : Ev → Ev → Prop
  refl : ∀ a, le a a
  trans : ∀ a b c, le a b → le b c → le a c
  antisymm : ∀ a b, le a b → le b a → a = b
  /-- program order: on one process, `le` is the index order -/
  local_iff : ∀ p k k', le (p, k) (p, k') ↔ k ≤ k'

/-- happens strictly before -/
def Run.lt (r : Run) (a b : Ev) : Prop := r.le a b ∧ a ≠ b

/-- A clock that counts: on process p, the events with index below `V e p`
    are exactly those that happen before or equal `e`. -/
structure Clock (r : Run) where
  V : Ev → Nat → Nat
  count : ∀ e p k, k < V e p ↔ r.le (p, k) e

def vle (u v : Nat → Nat) : Prop := ∀ i, u i ≤ v i
def vlt (u v : Nat → Nat) : Prop := vle u v ∧ ∃ i, u i < v i

variable {r : Run} (c : Clock r)

theorem own (a : Ev) : c.V a a.1 = a.2 + 1 := by
  have h1 : a.2 < c.V a a.1 := (c.count a a.1 a.2).2 (r.refl a)
  have h2 : ¬ (a.2 + 1 < c.V a a.1) := by
    intro h
    have := (c.count a a.1 (a.2 + 1)).1 h
    have := (r.local_iff a.1 (a.2 + 1) a.2).1 this
    omega
  omega

theorem mono {a b : Ev} (h : r.le a b) : vle (c.V a) (c.V b) := by
  intro p
  apply Nat.le_of_not_lt
  intro hlt
  have := (c.count a p (c.V b p)).1 hlt
  have := (c.count b p (c.V b p)).2 (r.trans _ _ _ this h)
  omega

/-- forward: a → b implies V a < V b -/
theorem lt_vlt {a b : Ev} (h : r.lt a b) : vlt (c.V a) (c.V b) := by
  refine ⟨mono c h.1, b.1, ?_⟩
  apply Nat.lt_of_not_le
  intro hle
  have hb : b.2 < c.V b b.1 := (c.count b b.1 b.2).2 (r.refl b)
  have hba : r.le (b.1, b.2) a := (c.count a b.1 b.2).1 (Nat.lt_of_lt_of_le hb hle)
  exact h.2 (r.antisymm a b h.1 hba)

/-- converse: V a < V b implies a → b -/
theorem vlt_lt {a b : Ev} (h : vlt (c.V a) (c.V b)) : r.lt a b := by
  obtain ⟨hle, i, hi⟩ := h
  have ha : a.2 < c.V a a.1 := (c.count a a.1 a.2).2 (r.refl a)
  have hab : r.le (a.1, a.2) b := (c.count b a.1 a.2).1 (Nat.lt_of_lt_of_le ha (hle a.1))
  refine ⟨hab, ?_⟩
  intro e
  subst e
  omega

theorem vc_iff (a b : Ev) : r.lt a b ↔ vlt (c.V a) (c.V b) :=
  ⟨lt_vlt c, vlt_lt c⟩

/-- concurrent events are exactly those with incomparable clocks -/
theorem concurrent_iff (a b : Ev) :
    (¬ r.lt a b ∧ ¬ r.lt b a) ↔ (¬ vlt (c.V a) (c.V b) ∧ ¬ vlt (c.V b) (c.V a)) := by
  rw [vc_iff c a b, vc_iff c b a]

end RCE.VClock
