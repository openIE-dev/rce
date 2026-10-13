/-
Abstract models for T19b, T21, T22, T25-T31 (0.6b concurrency).
These are models, not proofs about the Rust evaluator. T27, T29 and T30
are linked to the evaluator by generated test vectors (ModelVectors.lean,
checker/tests/lean_models.rs); the others are not linked. ParSem.lean and
ParRefine.lean connect to an operational semantics (core + par); T24 for
the tick rules is TickClock.lean.
-/
namespace RCE.Models

/-- T19b: if waits-for is acyclic (a rank strictly decreases along waits),
any nonempty set of tasks contains one that waits on no task in the set. -/
theorem t19b {T : Type} (ts : List T) (waits : T → T → Prop) (rank : T → Nat)
    (acyc : ∀ a b, waits a b → rank b < rank a) (t0 : T) (h0 : t0 ∈ ts) :
    ∃ m ∈ ts, ∀ u ∈ ts, ¬ waits m u := by
  have key : ∀ n, ∀ t ∈ ts, rank t < n → ∃ m ∈ ts, ∀ u ∈ ts, ¬ waits m u := by
    intro n
    induction n with
    | zero => intro t _ h; omega
    | succ n ih =>
      intro t ht hr
      by_cases hex : ∃ u ∈ ts, waits t u
      · obtain ⟨u, hu, hw⟩ := hex
        exact ih u hu (by have := acyc _ _ hw; omega)
      · exact ⟨t, ht, fun u hu hw => hex ⟨u, hu, hw⟩⟩
  exact key _ t0 h0 (Nat.lt_succ_self _)

/-- T21: a nondeterministic run driven by an oracle logs its choices;
replaying the log reproduces the final state. -/
def runO {S : Type} (step : S → Nat → S) (o : Nat → Nat) : Nat → S → S × List Nat
  | 0, s => (s, [])
  | k+1, s => let c := o k
              let r := runO step o k (step s c)
              (r.1, c :: r.2)

def replay {S : Type} (step : S → Nat → S) : List Nat → S → S
  | [], s => s
  | c :: cs, s => replay step cs (step s c)

theorem t21 {S : Type} (step : S → Nat → S) (o : Nat → Nat) (k : Nat) (s : S) :
    replay step (runO step o k s).2 s = (runO step o k s).1 := by
  induction k generalizing s with
  | zero => rfl
  | succ k ih => simp [runO, replay, ih]

/-- T22: a deadline block runs the body if it fits, else the else block;
exactly one outcome, and elapsed time is bounded by deadline + else cost. -/
def deadline (c d e : Nat) : Bool × Nat :=
  if c ≤ d then (true, c) else (false, d + e)

theorem t22 (c d e : Nat) :
    (deadline c d e).2 ≤ d + e ∧ ((deadline c d e).1 = true ↔ c ≤ d) := by
  unfold deadline; split <;> simp_all <;> omega

def maxL : List Nat → Nat
  | [] => 0
  | x :: xs => max x (maxL xs)

theorem le_maxL : ∀ (l : List Nat), ∀ x ∈ l, x ≤ maxL l
  | [], _, h => by cases h
  | y :: ys, x, h => by
    cases h with
    | head => simp [maxL]; omega
    | tail _ h => have := le_maxL ys x h; simp [maxL]; omega

theorem maxL_mem : ∀ (l : List Nat), l ≠ [] → maxL l ∈ l
  | [], h => absurd rfl h
  | [x], _ => by simp [maxL]
  | x :: y :: ys, _ => by
    have ih := maxL_mem (y :: ys) (by simp)
    show max x (maxL (y :: ys)) ∈ x :: y :: ys
    by_cases h : x ≤ maxL (y :: ys)
    · rw [Nat.max_eq_right h]; exact List.mem_cons_of_mem _ ih
    · rw [Nat.max_eq_left (by omega)]; exact List.mem_cons_self _ _

/-- T25: priority admission admits a pending request whose priority is at
least that of every other pending request. -/
theorem t25 (pending : List Nat) (h : pending ≠ []) :
    maxL pending ∈ pending ∧ ∀ p ∈ pending, p ≤ maxL pending :=
  ⟨maxL_mem _ h, le_maxL _⟩

/-- T26: a scope ends no earlier than the finish time of every child. -/
def scopeEnd (start : Nat) (children : List Nat) : Nat := max start (maxL children)

theorem t26 (start : Nat) (cs : List Nat) :
    start ≤ scopeEnd start cs ∧ ∀ c ∈ cs, c ≤ scopeEnd start cs := by
  refine ⟨Nat.le_max_left _ _, fun c hc => ?_⟩
  have := le_maxL cs c hc; unfold scopeEnd; omega

/-- T27: an actor processing messages in some order answers each message
exactly as the sequential specification would at that position. -/
def actor {S M R : Type} (f : S → M → S) (r : S → M → R) : S → List M → List R
  | _, [] => []
  | s, m :: ms => r s m :: actor f r (f s m) ms

def foldS {S M : Type} (f : S → M → S) (s : S) (ms : List M) : S := ms.foldl f s

theorem actor_len {S M R : Type} (f : S → M → S) (r : S → M → R) :
    ∀ (ms : List M) (s : S), (actor f r s ms).length = ms.length
  | [], _ => rfl
  | _ :: ms, s => by simp [actor, actor_len f r ms]

theorem t27 {S M R : Type} (f : S → M → S) (r : S → M → R) :
    ∀ (ms : List M) (s : S) (i : Nat) (h : i < ms.length),
      (actor f r s ms)[i]'(by rw [actor_len]; exact h) = r (foldS f s (ms.take i)) (ms[i])
  | [], _, _, h => by cases h
  | m :: ms, s, 0, _ => by simp [actor, foldS]
  | m :: ms, s, i+1, h => by
    have := t27 f r ms (f s m) i (by simp at h; omega)
    simp only [actor, List.getElem_cons_succ, List.take_succ_cons, foldS, List.foldl] at *
    exact this

/-- T28: finite session types with choice. The dual of a protocol is
compatible with it, and dual is an involution. Recursive protocols are not
modelled here; see RCE.SessionRec for the recursive version. -/
inductive P where
  | done
  | send : P → P
  | recv : P → P
  | choose : P → P → P
  | offer : P → P → P

def dual : P → P
  | .done => .done
  | .send k => .recv (dual k)
  | .recv k => .send (dual k)
  | .choose a b => .offer (dual a) (dual b)
  | .offer a b => .choose (dual a) (dual b)

inductive Comp : P → P → Prop
  | done : Comp .done .done
  | sr {a b} : Comp a b → Comp (.send a) (.recv b)
  | rs {a b} : Comp a b → Comp (.recv a) (.send b)
  | co {a b c d} : Comp a c → Comp b d → Comp (.choose a b) (.offer c d)
  | oc {a b c d} : Comp a c → Comp b d → Comp (.offer a b) (.choose c d)

theorem t28 : ∀ p, Comp p (dual p) ∧ dual (dual p) = p
  | .done => ⟨.done, rfl⟩
  | .send k => let ⟨h, e⟩ := t28 k; ⟨.sr h, by simp [dual, e]⟩
  | .recv k => let ⟨h, e⟩ := t28 k; ⟨.rs h, by simp [dual, e]⟩
  | .choose a b => let ⟨h, e⟩ := t28 a; let ⟨h', e'⟩ := t28 b; ⟨.co h h', by simp [dual, e, e']⟩
  | .offer a b => let ⟨h, e⟩ := t28 a; let ⟨h', e'⟩ := t28 b; ⟨.oc h h', by simp [dual, e, e']⟩

/-- Compatible endpoints never deadlock: unless both are done, one can send
or choose and the other is ready to receive or offer. -/
theorem t28_progress {a b : P} (h : Comp a b) :
    (a = .done ∧ b = .done) ∨ ∃ x y, Comp x y := by
  cases h with
  | done => exact Or.inl ⟨rfl, rfl⟩
  | sr h => exact Or.inr ⟨_, _, h⟩
  | rs h => exact Or.inr ⟨_, _, h⟩
  | co h _ => exact Or.inr ⟨_, _, h⟩
  | oc h _ => exact Or.inr ⟨_, _, h⟩

/-- T29 with a threshold read: lattice (Nat, max). The final value is
independent of put order, and a single-threshold read gives the same answer
under every order. -/
theorem foldl_max_comm (a b : Nat) (l : List Nat) :
    l.foldl max (max a b) = max a (l.foldl max b) := by
  induction l generalizing b with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl]; rw [Nat.max_assoc, ih]

theorem t29_perm {l l' : List Nat} (h : l.Perm l') (s : Nat) :
    l.foldl max s = l'.foldl max s := by
  induction h generalizing s with
  | nil => rfl
  | cons x _ ih => exact ih _
  | swap x y l => simp only [List.foldl]; rw [Nat.max_assoc, Nat.max_assoc, Nat.max_comm x y]
  | trans _ _ ih1 ih2 => rw [ih1, ih2]

def thresholdRead (t : Nat) (v : Nat) : Option Nat := if t ≤ v then some t else none

theorem t29 {l l' : List Nat} (h : l.Perm l') (s t : Nat) :
    thresholdRead t (l.foldl max s) = thresholdRead t (l'.foldl max s) := by
  rw [t29_perm h]

/-- T30: an atomic batch applies all operations or none. -/
def batch {S O : Type} (ok : S → O → Bool) (ap : S → O → S) : S → List O → Option S
  | s, [] => some s
  | s, o :: os => if ok s o then batch ok ap (ap s o) os else none

def atomic {S O : Type} (ok : S → O → Bool) (ap : S → O → S) (s : S) (os : List O) : S :=
  (batch ok ap s os).getD s

theorem t30 {S O : Type} (ok : S → O → Bool) (ap : S → O → S) (s : S) (os : List O) :
    atomic ok ap s os = s ∨ atomic ok ap s os = os.foldl ap s := by
  unfold atomic
  suffices ∀ s, batch ok ap s os = none ∨ batch ok ap s os = some (os.foldl ap s) by
    rcases this s with h | h <;> simp [h]
  induction os with
  | nil => intro s; simp [batch]
  | cons o os ih =>
    intro s; simp only [batch]; split
    · exact ih _
    · exact Or.inl rfl

/-- T31: a replica that has applied a prefix of the common log holds the
fold of that prefix; replicas that applied equally many entries agree. -/
theorem t31 {S E : Type} (f : S → E → S) (s0 : S) (L : List E) (i j : Nat) (h : i = j) :
    (L.take i).foldl f s0 = (L.take j).foldl f s0 := by subst h; rfl

theorem t31_ext {S E : Type} (f : S → E → S) (s0 : S) (L : List E) (i : Nat) :
    (L.take (i+1)).foldl f s0 =
      match L[i]? with
      | some e => f ((L.take i).foldl f s0) e
      | none => (L.take i).foldl f s0 := by
  induction L generalizing i s0 with
  | nil => simp
  | cons x xs ih =>
    cases i with
    | zero => cases xs <;> simp
    | succ i => simp only [List.take, List.foldl, List.getElem?_cons_succ]; exact ih _ _

end RCE.Models
