/-
  E.2a / E.2b: canonical admission windows and the admitter role across terms.

  Model. An input is any type `α` with a key `k : α → Nat`. The key stands
  for the bytes of (priority descending, source id, source seq, event JSON)
  ordered lexicographically; it is injective on inputs because the event
  bytes and (source, seq) identify the input (HC-15, HC-16). A window's
  entries are `window k w`: the inputs sorted by key, adjacent copies of one
  key kept once (dedupe by (source, seq), HC-16). D's admitted inputs are
  the concatenation of the windows in order.

  Proved here:
  * `window_perm`: the entries of a window do not depend on the order the
    inputs arrived in (any permutation of the window's inputs);
  * `log_permwise`: hence D does not depend on arrival order within each
    window, for any sequence of windows;
  * `window_complete`: sorting loses no input and adds none;
  * `fenced_append_unchanged`: a replica that promised a higher term ignores
    an append from a lower one (T39 for a deposed admitter);
  * `committed_prefix_preserved`: under Leader Completeness (Raft's
    property, a hypothesis here, not proved), every committed prefix is a
    prefix of D after any number of term changes (T38/T39 under leader
    change).
  Not modeled: quorum intersection, message loss, the Rust code.
-/
namespace RCE.Admit

section Window
variable {α : Type} (k : α → Nat)

def ins (a : α) : List α → List α
  | [] => [a]
  | b :: t => if k a < k b then a :: b :: t else b :: ins a t

def sortK : List α → List α
  | [] => []
  | a :: t => ins k a (sortK t)

def dedupAdj : List α → List α
  | a :: b :: t => if k a = k b then dedupAdj (b :: t) else a :: dedupAdj (b :: t)
  | l => l

def window (w : List α) : List α := dedupAdj k (sortK k w)

theorem ins_comm (hk : ∀ x y, k x = k y → x = y) (a b : α) :
    ∀ l, ins k a (ins k b l) = ins k b (ins k a l) := by
  intro l
  by_cases hab : a = b
  · subst hab; rfl
  have hne : k a ≠ k b := fun h => hab (hk a b h)
  induction l with
  | nil =>
    by_cases h : k a < k b
    · have : ¬ k b < k a := by omega
      simp [ins, h, this]
    · have : k b < k a := by omega
      simp [ins, h, this]
  | cons c t ih =>
    by_cases h1 : k a < k c <;> by_cases h2 : k b < k c <;> by_cases h3 : k a < k b <;>
      by_cases h4 : k b < k a <;> by_cases h5 : k c < k a <;> by_cases h6 : k c < k b <;>
      simp [ins, h1, h2, h3, h4, h5, h6, ih] <;> omega

theorem sortK_perm (hk : ∀ x y, k x = k y → x = y) {l₁ l₂ : List α} (h : l₁.Perm l₂) :
    sortK k l₁ = sortK k l₂ := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp [sortK, ih]
  | swap x y l => simp [sortK]; exact ins_comm k hk y x _
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

theorem window_perm (hk : ∀ x y, k x = k y → x = y) {w₁ w₂ : List α} (h : w₁.Perm w₂) :
    window k w₁ = window k w₂ := by
  simp [window, sortK_perm k hk h]

theorem ins_perm (a : α) : ∀ l, (ins k a l).Perm (a :: l)
  | [] => List.Perm.refl _
  | b :: t => by
    unfold ins
    split
    · exact List.Perm.refl _
    · exact ((ins_perm a t).cons b).trans (List.Perm.swap a b t)

theorem window_complete : ∀ l : List α, (sortK k l).Perm l
  | [] => List.Perm.nil
  | a :: t => (ins_perm k a _).trans ((window_complete t).cons a)

/-- Two sequences of windows with the same inputs per window. -/
inductive PermWise : List (List α) → List (List α) → Prop
  | nil : PermWise [] []
  | cons {w₁ w₂ ws₁ ws₂} : w₁.Perm w₂ → PermWise ws₁ ws₂ → PermWise (w₁ :: ws₁) (w₂ :: ws₂)

def logOf : List (List α) → List α
  | [] => []
  | w :: ws => window k w ++ logOf ws

theorem log_permwise (hk : ∀ x y, k x = k y → x = y) {ws₁ ws₂ : List (List α)}
    (h : PermWise ws₁ ws₂) : logOf k ws₁ = logOf k ws₂ := by
  induction h with
  | nil => rfl
  | cons hw _ ih => simp [logOf, window_perm k hk hw, ih]

end Window

section Terms
variable {β : Type}

structure Replica (β : Type) where
  promised : Nat
  log : List β

/-- An append at term `t` (HC-18): not performed when a higher term was promised. -/
def appendAt (t : Nat) (r : Replica β) (l : List β) : Replica β :=
  if t < r.promised then r else { promised := t, log := l }

theorem fenced_append_unchanged (t : Nat) (r : Replica β) (l : List β) (h : t < r.promised) :
    appendAt t r l = r := by
  simp [appendAt, h]

theorem accepted_append (t : Nat) (r : Replica β) (l : List β) (h : r.promised ≤ t) :
    (appendAt t r l).log = l := by
  have : ¬ t < r.promised := by omega
  simp [appendAt, this]

/-- A history of terms: each new leader starts from a log that contains the
    committed prefix (Leader Completeness, hypothesis) and only appends. -/
inductive History : List β → List β → Prop
  | start (L : List β) : History L L
  | term {C L L' : List β} (new : List β) :
      History C L → C <+: L' → History C (L' ++ new)

theorem committed_prefix_preserved {C L : List β} (h : History C L) : C <+: L := by
  induction h with
  | start => exact List.prefix_refl _
  | term new _ hc _ => exact hc.trans (List.prefix_append _ new)

end Terms
end RCE.Admit
