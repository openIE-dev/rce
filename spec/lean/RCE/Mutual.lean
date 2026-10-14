/-
  Mutual recursion by a summed size measure (W-REC-2, draft 0.7 extension
  of spec 6.7). Model: a family of functions indexed by Nat, each taking a
  list of argument sizes. One evaluation of function f on arguments a makes
  a finite list of calls (g, a') and combines their results; the bodies are
  otherwise total (T2). Evaluation runs with explicit fuel, one unit per
  call level.

  Each function f has an ordered list P f of designated parameter
  positions; its measure is the sum of the sizes at those positions. The
  checker's condition on a call f → g inside one cycle is: the designated
  positions of g map one to one into the designated positions of f, the
  size passed at each is at most the size received there (the same
  parameter, or a component of it), and at least one is strictly smaller
  (a component); a call where none is smaller is allowed when such calls
  form no cycle. `call_dec` proves the measure strictly decreases on the
  first kind, `lex_dec` that the measure paired with a rank decreases on
  both, so the theorems below apply to that combined measure; `mutual_total` that fuel above the measure always suffices, and
  `mutual_fuel_irrelevant` that extra fuel changes nothing.
-/
namespace RCE.Mutual

def lsum : List Nat → Nat
  | [] => 0
  | x :: xs => x + lsum xs

/-- pointwise ≤ on lists of the same length, with at least one < -/
inductive LeLt : List Nat → List Nat → Bool → Prop
  | nil : LeLt [] [] false
  | cons {x y xs ys b} : x ≤ y → LeLt xs ys b → LeLt (x :: xs) (y :: ys) (b || decide (x < y))

theorem lelt_sum : ∀ {xs ys b}, LeLt xs ys b → lsum xs ≤ lsum ys ∧ (b = true → lsum xs < lsum ys)
  | _, _, _, .nil => by simp [lsum]
  | _, _, _, .cons (x := x) (y := y) h t => by
    have ⟨h1, h2⟩ := lelt_sum t
    constructor
    · simp [lsum]; omega
    · intro hb
      simp at hb
      rcases hb with hb | hb
      · have := h2 hb; simp [lsum]; omega
      · simp [lsum]; omega

/-- sublist of a list of naturals (kept positions of f, in order) -/
inductive Sub : List Nat → List Nat → Prop
  | nil : Sub [] []
  | skip {xs ys y} : Sub xs ys → Sub xs (y :: ys)
  | keep {xs ys y} : Sub xs ys → Sub (y :: xs) (y :: ys)

theorem sub_sum : ∀ {xs ys}, Sub xs ys → lsum xs ≤ lsum ys
  | _, _, .nil => Nat.le_refl _
  | _, _, .skip h => by have := sub_sum h; simp [lsum]; omega
  | _, _, .keep h => by have := sub_sum h; simp [lsum]; omega

theorem perm_sum : ∀ {xs ys : List Nat}, xs.Perm ys → lsum xs = lsum ys := by
  intro xs ys h
  induction h with
  | nil => rfl
  | cons x _ ih => simp [lsum, ih]
  | swap x y l => simp [lsum]; omega
  | trans _ _ ih1 ih2 => exact ih1.trans ih2

/-- The checker's call condition implies a strict decrease of the summed
    measure: `passed` are the sizes at g's designated positions, `recv` the
    sizes of the distinct f positions they map to, pointwise ≤ with one <;
    `recv` is a reordering of a sublist `kept` of f's designated sizes
    `mine` (the map is injective). -/
theorem call_dec (passed recv kept mine : List Nat) (h : LeLt passed recv true)
    (hp : recv.Perm kept) (hs : Sub kept mine) : lsum passed < lsum mine := by
  have h1 := (lelt_sum h).2 rfl
  have h2 := perm_sum hp
  have h3 := sub_sum hs
  omega

/-- Calls that do not shrink the summed measure are allowed when they form
    no cycle: with ranks r (r g < r f on every such call f → g, all ranks
    below R), the combined measure μ·R + r falls at every call. -/
theorem lex_dec (mf mg rf rg R : Nat) (hg : rg < R)
    (h : mg < mf ∨ (mg ≤ mf ∧ rg < rf)) : mg * R + rg < mf * R + rf := by
  rcases h with h | ⟨h1, h2⟩
  · have : (mg + 1) * R ≤ mf * R := Nat.mul_le_mul_right R h
    rw [Nat.succ_mul] at this
    omega
  · have : mg * R ≤ mf * R := Nat.mul_le_mul_right R h1
    omega

/-- A program: for function f on arguments a, the calls it makes and how it
    combines their results. -/
structure Prog where
  calls : Nat → List Nat → List (Nat × List Nat)
  comb : Nat → List Nat → List Int → Int

def evalAll (ev : Nat → List Nat → Option Int) : List (Nat × List Nat) → Option (List Int)
  | [] => some []
  | (g, a) :: cs =>
    match ev g a, evalAll ev cs with
    | some v, some vs => some (v :: vs)
    | _, _ => none

def evalF (p : Prog) : Nat → Nat → List Nat → Option Int
  | 0, _, _ => none
  | n + 1, f, a => (evalAll (evalF p n) (p.calls f a)).map (p.comb f a)

/-- Every call strictly lowers the measure μ. -/
def Decreasing (p : Prog) (μ : Nat → List Nat → Nat) : Prop :=
  ∀ f a g a', (g, a') ∈ p.calls f a → μ g a' < μ f a

theorem evalAll_some (ev : Nat → List Nat → Option Int) (r : Nat → List Nat → Int) :
    ∀ cs : List (Nat × List Nat), (∀ c ∈ cs, ev c.1 c.2 = some (r c.1 c.2)) →
      evalAll ev cs = some (cs.map (fun c => r c.1 c.2))
  | [], _ => rfl
  | (g, a) :: cs, h => by
    have h1 := h (g, a) (List.mem_cons_self _ _)
    have h2 := evalAll_some ev r cs (fun c hc => h c (List.mem_cons_of_mem _ hc))
    simp [evalAll, h1, h2]

/-- The value a program denotes when μ decreases: `go` unfolded μ + 1
    levels (`go_stable` shows deeper unfolding gives the same value). -/
def den (p : Prog) (μ : Nat → List Nat → Nat) : Nat → List Nat → Int :=
  fun f a => go (μ f a + 1) f a
where
  go : Nat → Nat → List Nat → Int
  | 0, _, _ => 0
  | n + 1, f, a => p.comb f a ((p.calls f a).map (fun c => go n c.1 c.2))

/-- Termination with an explicit bound: with fuel above the measure the
    evaluation returns a value, and that value does not depend on the fuel. -/
theorem mutual_total (p : Prog) (μ : Nat → List Nat → Nat) (hd : Decreasing p μ) :
    ∀ n f a, μ f a < n → evalF p n f a = some (den.go p n f a) := by
  intro n
  induction n with
  | zero => intro f a h; omega
  | succ n ih =>
    intro f a h
    have hall : ∀ c ∈ p.calls f a, evalF p n c.1 c.2 = some (den.go p n c.1 c.2) := by
      intro c hc
      have := hd f a c.1 c.2 hc
      exact ih c.1 c.2 (by omega)
    simp [evalF, den.go, evalAll_some (evalF p n) (den.go p n) _ hall]

theorem go_stable (p : Prog) (μ : Nat → List Nat → Nat) (hd : Decreasing p μ) :
    ∀ n m f a, μ f a < n → n ≤ m → den.go p n f a = den.go p m f a := by
  intro n
  induction n with
  | zero => intro m f a h; omega
  | succ n ih =>
    intro m f a h hm
    cases m with
    | zero => omega
    | succ m =>
      simp only [den.go]
      congr 1
      apply List.map_congr_left
      intro c hc
      have := hd f a c.1 c.2 hc
      exact ih m c.1 c.2 (by omega) (by omega)

/-- Extra fuel never changes the result. -/
theorem mutual_fuel_irrelevant (p : Prog) (μ : Nat → List Nat → Nat) (hd : Decreasing p μ)
    (f : Nat) (a : List Nat) (m n : Nat) (hm : μ f a < m) (hn : μ f a < n) :
    evalF p m f a = evalF p n f a := by
  rw [mutual_total p μ hd m f a hm, mutual_total p μ hd n f a hn]
  have h1 := go_stable p μ hd (μ f a + 1) m f a (by omega) (by omega)
  have h2 := go_stable p μ hd (μ f a + 1) n f a (by omega) (by omega)
  rw [← h1, ← h2]

/-- Depth bound: along any chain of calls the measure falls by at least one
    per call, so a chain from (f, a) has at most μ f a + 1 calls. -/
inductive Chain (p : Prog) : Nat → List Nat → Nat → Prop
  | one {f a} : Chain p f a 1
  | step {f a g a' k} : (g, a') ∈ p.calls f a → Chain p g a' k → Chain p f a (k + 1)

theorem mutual_depth (p : Prog) (μ : Nat → List Nat → Nat) (hd : Decreasing p μ) :
    ∀ f a k, Chain p f a k → k ≤ μ f a + 1 := by
  intro f a k h
  induction h with
  | one => omega
  | step hc _ ih => have := hd _ _ _ _ hc; omega

/-! ### Lexicographic layers (6.7, W-REC-1)

The checker's rule for every recursive cycle, including a function that
calls only itself: layers 1..k, each a choice of designated positions per
function, and a final rank. Every call of the cycle keeps the summed
measure of layer 1; the calls that shrink it are done; every remaining call
keeps layer 2, and so on; the calls left after layer k form no cycle among
the functions (rank). So the tuple (μ₁, …, μ_k, rank) falls
lexicographically at every call, and the lexicographic order on a tuple of
fixed length is well founded. -/

/-- A call that keeps the summed measure (no position strictly smaller is
    needed): it does not grow. -/
theorem call_keep (passed recv kept mine : List Nat) {b : Bool} (h : LeLt passed recv b)
    (hp : recv.Perm kept) (hs : Sub kept mine) : lsum passed ≤ lsum mine := by
  have h1 := (lelt_sum h).1
  have h2 := perm_sum hp
  have h3 := sub_sum hs
  omega

/-- k layer measures and a rank. -/
@[reducible] def Tup : Nat → Type
  | 0 => Nat
  | k + 1 => Nat × Tup k

def tupWf : (k : Nat) → WellFoundedRelation (Tup k)
  | 0 => Nat.lt_wfRel
  | k + 1 => Prod.lex Nat.lt_wfRel (tupWf k)

/-- The checker's condition on one call: at each layer the measure does not
    grow, and either it shrinks at some layer (the call is done there) or
    the rank falls. -/
def LayerDec : (k : Nat) → Tup k → Tup k → Prop
  | 0, r', r => @LT.lt Nat _ r' r
  | k + 1, t', t => t'.1 < t.1 ∨ (t'.1 ≤ t.1 ∧ LayerDec k t'.2 t.2)

theorem layer_dec : ∀ (k : Nat) (t' t : Tup k), LayerDec k t' t → (tupWf k).rel t' t
  | 0, _, _, h => h
  | k + 1, (m', t'), (m, t), h => by
    simp only [LayerDec] at h
    show Prod.Lex Nat.lt_wfRel.rel (tupWf k).rel (m', t') (m, t)
    rcases h with h | ⟨h1, h2⟩
    · exact Prod.Lex.left _ _ h
    · rcases Nat.lt_or_eq_of_le h1 with h1 | h1
      · exact Prod.Lex.left _ _ h1
      · subst h1; exact Prod.Lex.right _ (layer_dec k t' t h2)

theorem evalAll_mono (ev ev' : Nat → List Nat → Option Int)
    (h : ∀ g a v, ev g a = some v → ev' g a = some v) :
    ∀ cs vs, evalAll ev cs = some vs → evalAll ev' cs = some vs
  | [], _, hv => hv
  | (g, a) :: cs, vs, hv => by
    simp only [evalAll] at hv
    split at hv
    · rename_i v ws h1 h2
      simp only [evalAll, h g a v h1, evalAll_mono ev ev' h cs ws h2]
      exact hv
    · contradiction

/-- More fuel never takes a result away or changes it. -/
theorem evalF_mono (p : Prog) : ∀ n f a v, evalF p n f a = some v → evalF p (n + 1) f a = some v := by
  intro n
  induction n with
  | zero => intro f a v h; simp [evalF] at h
  | succ n ih =>
    intro f a v h
    simp only [evalF, Option.map_eq_some'] at h ⊢
    obtain ⟨vs, h1, h2⟩ := h
    exact ⟨vs, evalAll_mono _ _ ih _ _ h1, h2⟩

theorem evalF_mono_le (p : Prog) (n m f a v) (hm : n ≤ m) (h : evalF p n f a = some v) :
    evalF p m f a = some v := by
  induction hm with
  | refl => exact h
  | step _ ih => exact evalF_mono p _ f a v ih

theorem evalAll_exists (p : Prog) :
    ∀ cs : List (Nat × List Nat), (∀ c ∈ cs, ∃ n v, evalF p n c.1 c.2 = some v) →
      ∃ n vs, evalAll (evalF p n) cs = some vs
  | [], _ => ⟨0, [], rfl⟩
  | (g, a) :: cs, h => by
    obtain ⟨n1, v, hv⟩ := h (g, a) (List.mem_cons_self _ _)
    obtain ⟨n2, vs, hvs⟩ := evalAll_exists p cs (fun c hc => h c (List.mem_cons_of_mem _ hc))
    refine ⟨max n1 n2, v :: vs, ?_⟩
    have e1 := evalF_mono_le p n1 (max n1 n2) g a v (Nat.le_max_left _ _) hv
    have e2 := evalAll_mono (evalF p n2) (evalF p (max n1 n2))
      (fun g' a' v' h' => evalF_mono_le p n2 _ g' a' v' (Nat.le_max_right _ _) h') cs vs hvs
    simp [evalAll, e1, e2]

/-- Termination: when every call lowers a measure in a well-founded order
    (`layer_dec` gives one from the checker's layers), every evaluation
    returns a value for some fuel. -/
theorem wf_total (p : Prog) {M : Type} (R : WellFoundedRelation M) (μ : Nat → List Nat → M)
    (hd : ∀ f a g a', (g, a') ∈ p.calls f a → R.rel (μ g a') (μ f a)) :
    ∀ f a, ∃ n v, evalF p n f a = some v := by
  have key : ∀ x, ∀ f a, μ f a = x → ∃ n v, evalF p n f a = some v := by
    intro x
    induction x using R.wf.induction with
    | _ x ih =>
      intro f a hx
      obtain ⟨n, vs, h⟩ := evalAll_exists p (p.calls f a) (fun c hc =>
        ih (μ c.1 c.2) (hx ▸ hd f a c.1 c.2 hc) c.1 c.2 rfl)
      exact ⟨n + 1, p.comb f a vs, by simp [evalF, h]⟩
  exact fun f a => key _ f a rfl

/-- The value does not depend on the fuel. -/
theorem wf_fuel_irrelevant (p : Prog) (f : Nat) (a : List Nat) (n m : Nat) (v w : Int)
    (hn : evalF p n f a = some v) (hm : evalF p m f a = some w) : v = w := by
  have h1 := evalF_mono_le p n (max n m) f a v (Nat.le_max_left _ _) hn
  have h2 := evalF_mono_le p m (max n m) f a w (Nat.le_max_right _ _) hm
  rw [h1] at h2
  exact Option.some.inj h2

/-! ### Declared bounds (6.7, `bound n`)

A function declared with `bound n` may have at most n active calls; the
call that would be the (n + 1)-th traps. The counts are part of the
evaluation state (a `Prog` can carry them in its argument list). The
slack Σ (n_f − c_f) over the bounded functions falls by one at every
admitted call of a bounded function (`bump_slack`) and is unchanged by
other calls; the checker requires the calls between unbounded functions
of a cycle to form no cycle, so with the rank of `lex_dec` (bounded
functions ranked above all others) the pair (slack, rank) falls at every
call and `mutual_total`, `mutual_fuel_irrelevant` and `mutual_depth`
apply: every run ends, by a value or the trap, within (slack + 1) · R
nested calls. -/

/-- (bound, active calls) of each bounded function -/
def slack : List (Nat × Nat) → Nat
  | [] => 0
  | (n, c) :: xs => (n - c) + slack xs

def bump : List (Nat × Nat) → Nat → List (Nat × Nat)
  | [], _ => []
  | (n, c) :: xs, 0 => (n, c + 1) :: xs
  | x :: xs, i + 1 => x :: bump xs i

/-- the call is admitted: function i has fewer active calls than its bound -/
def below : List (Nat × Nat) → Nat → Prop
  | [], _ => False
  | (n, c) :: _, 0 => c < n
  | _ :: xs, i + 1 => below xs i

theorem bump_slack : ∀ (xs : List (Nat × Nat)) (i : Nat), below xs i → slack (bump xs i) + 1 = slack xs
  | [], _, h => absurd h (by simp [below])
  | (n, c) :: xs, 0, h => by simp only [below] at h; simp only [bump, slack]; omega
  | (n, c) :: xs, i + 1, h => by
    simp only [below] at h
    have := bump_slack xs i h
    simp only [bump, slack]; omega

/-- With the rank of `lex_dec`, every admitted call lowers (slack, rank):
    a call of a bounded function lowers the slack, any other keeps it and
    lowers the rank. -/
theorem bound_dec (xs : List (Nat × Nat)) (i : Nat) (h : below xs i) (rf rg R : Nat) (hg : rg < R) :
    slack (bump xs i) * R + rg < slack xs * R + rf :=
  lex_dec (slack xs) (slack (bump xs i)) rf rg R hg (Or.inl (by have := bump_slack xs i h; omega))

end RCE.Mutual

#print axioms RCE.Mutual.call_dec
#print axioms RCE.Mutual.lex_dec
#print axioms RCE.Mutual.mutual_total
#print axioms RCE.Mutual.mutual_fuel_irrelevant
#print axioms RCE.Mutual.mutual_depth
#print axioms RCE.Mutual.call_keep
#print axioms RCE.Mutual.layer_dec
#print axioms RCE.Mutual.wf_total
#print axioms RCE.Mutual.wf_fuel_irrelevant
#print axioms RCE.Mutual.bump_slack
#print axioms RCE.Mutual.bound_dec
