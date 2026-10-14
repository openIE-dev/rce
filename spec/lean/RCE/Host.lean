import RCE.Replay
/-! Appendix E, the host contract. Three models:
    * T37 host replay (HC-8, HC-10, HC-12): events are program events or
      host entries; δ is the identity on host entries; replaying the log
      with no adapter reproduces the live run, and placement, a function of
      the logged host entries, is the same on replay.
    * T38 verify on open (HS-4): the verified prefix of a hash-chained log
      is a prefix, a valid chain, the whole log when it is valid, and a
      valid log followed by an entry that does not link is cut back to it.
      The hash function is abstract, so nothing about SHA-256 is assumed.
    * T39 fatal barrier (HS-3): after a failed barrier nothing is appended,
      and every entry of a running admitter's log passed its barrier. -/
namespace RCE.Host

/-! ## T37 host replay -/
section Replay
variable {S E F A P : Type}

/-- δ extended to host entries: the identity on `host.*` entries. -/
def hostStep (δ : S → E → A → S) : S → (E ⊕ F) → A → S
  | s, .inl e, a => δ s e a
  | s, .inr _, _ => s

theorem host_entry_identity (δ : S → E → A → S) (s : S) (f : F) (a : A) :
    hostStep δ s (.inr f) a = s := rfl

/-- The host entries of a log, in log order. -/
def hostEntries : List ((E ⊕ F) × A) → List F
  | [] => []
  | (.inl _, _) :: l => hostEntries l
  | (.inr f, _) :: l => f :: hostEntries l

/-- HC-8: replay with every adapter disconnected (no oracle) equals the
live run, whatever the adapters answered. -/
theorem t37_replay (δ : S → E → A → S) (o : S → (E ⊕ F) → A) (s : S)
    (es : List (E ⊕ F)) :
    Replay.replay (hostStep δ) s (Replay.live (hostStep δ) o s es).2 =
      (Replay.live (hostStep δ) o s es).1 :=
  Replay.replay_live _ _ _ _

/-- HC-12: placement is a function of the logged host entries, so two runs
whose logs agree make the same placement, and so does replay. -/
theorem t37_placement (place : List F → P) (l₁ l₂ : List ((E ⊕ F) × A))
    (h : l₁ = l₂) : place (hostEntries l₁) = place (hostEntries l₂) := by
  rw [h]

/-- Host entries change nothing: a run of only host entries leaves the
state as it was. -/
theorem t37_host_only (δ : S → E → A → S) (s : S) (l : List (F × A)) :
    Replay.replay (hostStep δ) s (l.map (fun x => ((Sum.inr x.1 : E ⊕ F), x.2))) = s := by
  induction l with
  | nil => rfl
  | cons x l ih => simp only [List.map, Replay.replay]; exact ih
end Replay

/-! ## T38 verify on open -/
section Chain
variable {Ev H : Type} [DecidableEq H]

structure Entry (Ev H : Type) where
  event : Ev
  prev : H
  stateHash : H
  hash : H

/-- An entry links after hash `p` when its `prev` is `p` and its `hash` is
the hash of its body (6.1). -/
def links (hf : H → Ev → H → H) (p : H) (x : Entry Ev H) : Bool :=
  decide (x.prev = p) && decide (x.hash = hf p x.event x.stateHash)

def valid (hf : H → Ev → H → H) : H → List (Entry Ev H) → Prop
  | _, [] => True
  | p, x :: xs => links hf p x = true ∧ valid hf x.hash xs

/-- The hash after a chain. -/
def last : H → List (Entry Ev H) → H
  | p, [] => p
  | _, x :: xs => last x.hash xs

/-- HS-4: keep the longest prefix that links from the genesis hash. -/
def verify (hf : H → Ev → H → H) : H → List (Entry Ev H) → List (Entry Ev H)
  | _, [] => []
  | p, x :: xs => if links hf p x then x :: verify hf x.hash xs else []

theorem verify_prefix (hf : H → Ev → H → H) (p : H) (l : List (Entry Ev H)) :
    ∃ t, verify hf p l ++ t = l := by
  induction l generalizing p with
  | nil => exact ⟨[], rfl⟩
  | cons x xs ih =>
    unfold verify
    by_cases h : links hf p x = true
    · obtain ⟨t, ht⟩ := ih x.hash
      exact ⟨t, by rw [if_pos h, List.cons_append, ht]⟩
    · exact ⟨x :: xs, by rw [if_neg h]; rfl⟩

theorem verify_valid (hf : H → Ev → H → H) (p : H) (l : List (Entry Ev H)) :
    valid hf p (verify hf p l) := by
  induction l generalizing p with
  | nil => trivial
  | cons x xs ih =>
    unfold verify
    by_cases h : links hf p x = true
    · rw [if_pos h]; exact ⟨h, ih x.hash⟩
    · rw [if_neg h]; trivial

theorem verify_intact (hf : H → Ev → H → H) (p : H) (l : List (Entry Ev H))
    (hv : valid hf p l) : verify hf p l = l := by
  induction l generalizing p with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨h, hr⟩ := hv
    unfold verify
    rw [if_pos h, ih x.hash hr]

theorem verify_append (hf : H → Ev → H → H) (p : H) (l t : List (Entry Ev H))
    (hv : valid hf p l) : verify hf p (l ++ t) = l ++ verify hf (last p l) t := by
  induction l generalizing p with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨h, hr⟩ := hv
    simp only [List.cons_append]
    rw [verify.eq_2, if_pos h, ih x.hash hr]
    rfl

/-- A torn or corrupted tail entry is cut, and the valid log is kept. -/
theorem t38_torn (hf : H → Ev → H → H) (p : H) (l : List (Entry Ev H))
    (b : Entry Ev H) (rest : List (Entry Ev H)) (hv : valid hf p l)
    (hb : links hf (last p l) b = false) :
    verify hf p (l ++ b :: rest) = l := by
  rw [verify_append hf p l _ hv]
  unfold verify
  rw [hb]
  simp
end Chain

/-! ## T39 fatal barrier -/
section Barrier
variable {Ev : Type}

inductive Adm (Ev : Type)
  | run (log : List Ev)
  | halted (log : List Ev)

/-- One append: the barrier `sync` decides success; on failure the
admitter halts and keeps the log it had. -/
def append (sync : List Ev → Ev → Bool) : Adm Ev → Ev → Adm Ev
  | .run l, e => if sync l e then .run (l ++ [e]) else .halted l
  | .halted l, _ => .halted l

def appendAll (sync : List Ev → Ev → Bool) (a : Adm Ev) (es : List Ev) : Adm Ev :=
  es.foldl (append sync) a

theorem t39_halted (sync : List Ev → Ev → Bool) (l : List Ev) (es : List Ev) :
    appendAll sync (.halted l) es = .halted l := by
  induction es with
  | nil => rfl
  | cons e es ih => simp only [appendAll, List.foldl, append]; exact ih

/-- A running admitter's log is exactly the entries it was given: no
failed barrier is ever followed by a run, so nothing is silently dropped. -/
theorem t39_run_complete (sync : List Ev → Ev → Bool) (l₀ l : List Ev)
    (es : List Ev) (h : appendAll sync (.run l₀) es = .run l) : l = l₀ ++ es := by
  induction es generalizing l₀ with
  | nil => simp only [appendAll, List.foldl] at h; cases h; simp
  | cons e es ih =>
    simp only [appendAll, List.foldl, append] at h
    cases hs : sync l₀ e with
    | false =>
      rw [hs] at h
      simp only [Bool.false_eq_true, ite_false] at h
      rw [show List.foldl (append sync) (Adm.halted l₀) es = appendAll sync (.halted l₀) es from rfl,
        t39_halted] at h
      cases h
    | true =>
      rw [hs] at h
      simp only [ite_true] at h
      rw [ih (l₀ ++ [e]) h]; simp

theorem t39_fault_stops (sync : List Ev → Ev → Bool) (l : List Ev) (e : Ev)
    (es : List Ev) (h : sync l e = false) :
    appendAll sync (.run l) (e :: es) = .halted l := by
  simp only [appendAll, List.foldl, append, h]
  exact t39_halted sync l es
end Barrier

end RCE.Host
