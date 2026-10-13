/-
T34 Order-preserving A-normal form. The emitters flatten an expression
that contains write-back calls into let-bound steps (spec 6.2,
`spec/checker/src/anf.rs`). On a core fragment with reads, write-back
calls (which update the store and charge), operators (which charge) and
emits, running the flattened bindings gives the same value, the same
final store and the same trace, where the trace records every charge and
every emit in order. So reads see the same store, budget charges happen
in the same order (a budget traps at the same point) and emits
interleave identically. The model binds every non-literal node (full
ANF); the implementation leaves more nodes in place, which is tested.
-/
namespace RCE.Anf

inductive Ev | charge (c : Nat) | out (v : Int)
  deriving DecidableEq

structure St where
  σ : Nat → Int
  tr : List Ev

def upd (f : Nat → Int) (x : Nat) (v : Int) : Nat → Int := fun y => if y = x then v else f y

inductive E | lit (n : Int) | rd (x : Nat) | wb (x : Nat) | add (a b : E) | emit (a : E)

/-- Source semantics: left to right. `wb x` increments x, charges 4 and
returns the new value; `add` charges 1 after its operands. -/
def eval : E → St → Int × St
  | .lit n, s => (n, s)
  | .rd x, s => (s.σ x, s)
  | .wb x, s => (s.σ x + 1, ⟨upd s.σ x (s.σ x + 1), s.tr ++ [.charge 4]⟩)
  | .add a b, s =>
    let r1 := eval a s
    let r2 := eval b r1.2
    (r1.1 + r2.1, ⟨r2.2.σ, r2.2.tr ++ [.charge 1]⟩)
  | .emit a, s =>
    let r := eval a s
    (r.1, ⟨r.2.σ, r.2.tr ++ [.out r.1]⟩)

inductive A | lit (n : Int) | tmp (i : Nat)
inductive Op | rd (x : Nat) | wb (x : Nat) | add (a b : A) | emit (a : A)

def aval (ρ : Nat → Int) : A → Int
  | .lit n => n
  | .tmp i => ρ i

def op (o : Op) (ρ : Nat → Int) (s : St) : Int × St :=
  match o with
  | .rd x => (s.σ x, s)
  | .wb x => (s.σ x + 1, ⟨upd s.σ x (s.σ x + 1), s.tr ++ [.charge 4]⟩)
  | .add a b => (aval ρ a + aval ρ b, ⟨s.σ, s.tr ++ [.charge 1]⟩)
  | .emit a => (aval ρ a, ⟨s.σ, s.tr ++ [.out (aval ρ a)]⟩)

/-- Run let-bindings `tmp i := op` in order. -/
def run : List (Nat × Op) → (Nat → Int) → St → (Nat → Int) × St
  | [], ρ, s => (ρ, s)
  | (i, o) :: p, ρ, s => let r := op o ρ s; run p (upd ρ i r.1) r.2

/-- Full ANF with fresh temporaries from k: (bindings, atom, next). -/
def anf : E → Nat → List (Nat × Op) × A × Nat
  | .lit n, k => ([], .lit n, k)
  | .rd x, k => ([(k, .rd x)], .tmp k, k + 1)
  | .wb x, k => ([(k, .wb x)], .tmp k, k + 1)
  | .add a b, k =>
    let ra := anf a k
    let rb := anf b ra.2.2
    (ra.1 ++ rb.1 ++ [(rb.2.2, .add ra.2.1 rb.2.1)], .tmp rb.2.2, rb.2.2 + 1)
  | .emit a, k =>
    let ra := anf a k
    (ra.1 ++ [(ra.2.2, .emit ra.2.1)], .tmp ra.2.2, ra.2.2 + 1)

theorem run_append (p q : List (Nat × Op)) (ρ : Nat → Int) (s : St) :
    run (p ++ q) ρ s = run q (run p ρ s).1 (run p ρ s).2 := by
  induction p generalizing ρ s with
  | nil => rfl
  | cons b p ih => obtain ⟨i, o⟩ := b; simp only [List.cons_append, run]; exact ih _ _

/-- Binders of `anf e k` lie in [k, next), and k ≤ next. -/
theorem anf_bounds (e : E) (k : Nat) :
    k ≤ (anf e k).2.2 ∧ ∀ b ∈ (anf e k).1, k ≤ b.1 ∧ b.1 < (anf e k).2.2 := by
  induction e generalizing k with
  | lit _ => simp [anf]
  | rd x => simp [anf]
  | wb x => simp [anf]
  | add a b iha ihb =>
    obtain ⟨ha1, ha2⟩ := iha k
    obtain ⟨hb1, hb2⟩ := ihb (anf a k).2.2
    refine ⟨by simp only [anf]; omega, ?_⟩
    intro c hc
    simp only [anf, List.mem_append, List.mem_singleton] at hc
    rcases hc with (hc | hc) | hc
    · have := ha2 c hc; simp only [anf]; omega
    · have := hb2 c hc; simp only [anf]; omega
    · subst hc; simp only [anf]; omega
  | emit a iha =>
    obtain ⟨ha1, ha2⟩ := iha k
    refine ⟨by simp only [anf]; omega, ?_⟩
    intro c hc
    simp only [anf, List.mem_append, List.mem_singleton] at hc
    rcases hc with hc | hc
    · have := ha2 c hc; simp only [anf]; omega
    · subst hc; simp only [anf]; omega

/-- The atom of `anf e k` is a literal or a temporary in [k, next). -/
theorem anf_atom (e : E) (k : Nat) :
    (∃ n, (anf e k).2.1 = .lit n) ∨ ∃ i, (anf e k).2.1 = .tmp i ∧ k ≤ i ∧ i < (anf e k).2.2 := by
  cases e with
  | lit _ => exact Or.inl ⟨_, rfl⟩
  | rd x => exact Or.inr ⟨k, rfl, Nat.le_refl _, Nat.lt_succ_self _⟩
  | wb x => exact Or.inr ⟨k, rfl, Nat.le_refl _, Nat.lt_succ_self _⟩
  | add a b =>
    have h1 := (anf_bounds a k).1
    have h2 := (anf_bounds b (anf a k).2.2).1
    refine Or.inr ⟨(anf b (anf a k).2.2).2.2, rfl, by omega, ?_⟩
    show _ < (anf b (anf a k).2.2).2.2 + 1; omega
  | emit a =>
    have h1 := (anf_bounds a k).1
    refine Or.inr ⟨(anf a k).2.2, rfl, h1, ?_⟩
    show _ < (anf a k).2.2 + 1; omega

/-- Running bindings whose binders are all ≥ k leaves temporaries below k. -/
theorem run_frame (p : List (Nat × Op)) (k : Nat) (h : ∀ b ∈ p, k ≤ b.1)
    (ρ : Nat → Int) (s : St) (i : Nat) (hi : i < k) : (run p ρ s).1 i = ρ i := by
  induction p generalizing ρ s with
  | nil => rfl
  | cons b p ih =>
    obtain ⟨j, o⟩ := b
    simp only [run]
    rw [ih (fun c hc => h c (List.mem_cons_of_mem _ hc))]
    have : k ≤ j := h (j, o) (List.mem_cons_self _ _)
    simp only [upd]
    have hne : i ≠ j := by omega
    simp [hne]

theorem aval_upd_tmp (ρ : Nat → Int) (i : Nat) (v : Int) : aval (upd ρ i v) (.tmp i) = v := by
  simp [aval, upd]

/-- T34: the flattened bindings give the source value, store and trace. -/
theorem t34 (e : E) (k : Nat) (ρ : Nat → Int) (s : St) :
    aval (run (anf e k).1 ρ s).1 (anf e k).2.1 = (eval e s).1 ∧
    (run (anf e k).1 ρ s).2 = (eval e s).2 := by
  induction e generalizing k ρ s with
  | lit n => exact ⟨rfl, rfl⟩
  | rd x => simp [anf, run, op, aval, eval, upd]
  | wb x => simp [anf, run, op, aval, eval, upd]
  | add a b iha ihb =>
    obtain ⟨va, sa⟩ := iha k ρ s
    let ra := anf a k
    let ρ1 := (run ra.1 ρ s).1
    let s1 := (run ra.1 ρ s).2
    obtain ⟨vb, sb⟩ := ihb ra.2.2 ρ1 s1
    -- the atom of a survives the bindings of b
    have keep : aval (run (anf b ra.2.2).1 ρ1 s1).1 ra.2.1 = aval ρ1 ra.2.1 := by
      rcases anf_atom a k with ⟨n, hn⟩ | ⟨i, hi, _, hi2⟩
      · simp only [ra, hn, aval]
      · simp only [ra, hi, aval]
        exact run_frame _ ra.2.2 (fun c hc => ((anf_bounds b ra.2.2).2 c hc).1) ρ1 s1 i hi2
    simp only [anf, run_append, run, op, aval, eval]
    refine ⟨?_, ?_⟩
    · simp only [upd, if_pos rfl]
      show aval (run (anf b ra.2.2).1 ρ1 s1).1 ra.2.1 + aval (run (anf b ra.2.2).1 ρ1 s1).1 (anf b ra.2.2).2.1 = _
      rw [keep, vb]; simp only [ρ1, va, s1, sa]
    · show (⟨(run (anf b ra.2.2).1 ρ1 s1).2.σ, (run (anf b ra.2.2).1 ρ1 s1).2.tr ++ [.charge 1]⟩ : St) = _
      rw [sb]; simp only [s1, sa]
  | emit a iha =>
    obtain ⟨va, sa⟩ := iha k ρ s
    simp only [anf, run_append, run, op, eval, aval_upd_tmp]
    rw [va, sa]; exact ⟨rfl, rfl⟩

end RCE.Anf
