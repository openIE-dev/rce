/-
T2 composition. T2 bounds δ through the call graph: a function's cost
is bounded by a bound computed from its argument, and calls compose.
Here a costed function returns its result and the steps it took; if f
and g are bounded by Bf and Bg, then g ∘ f is bounded by
Bf a + Bg (f a), and a chain (an acyclic call path) by the sum along
the path. This is the composition step the T2 bound B(P) relies on.
-/
namespace RCE.Compose

structure Costed (α β : Type) where
  run : α → β × Nat

def Bounded {α β : Type} (f : Costed α β) (B : α → Nat) : Prop :=
  ∀ a, (f.run a).2 ≤ B a

def comp {α β γ : Type} (f : Costed α β) (g : Costed β γ) : Costed α γ :=
  ⟨fun a => let r := f.run a; let q := g.run r.1; (q.1, r.2 + q.2)⟩

theorem comp_bounded {α β γ : Type} (f : Costed α β) (g : Costed β γ)
    (Bf : α → Nat) (Bg : β → Nat) (hf : Bounded f Bf) (hg : Bounded g Bg) :
    Bounded (comp f g) (fun a => Bf a + Bg (f.run a).1) := by
  intro a
  exact Nat.add_le_add (hf a) (hg _)

/-- The result of a composition is the composition of the results. -/
theorem comp_result {α β γ : Type} (f : Costed α β) (g : Costed β γ) (a : α) :
    ((comp f g).run a).1 = (g.run (f.run a).1).1 := rfl

/-- An acyclic chain of endofunctions, each bounded by a constant. -/
def chain {α : Type} : List (Costed α α) → Costed α α
  | [] => ⟨fun a => (a, 0)⟩
  | f :: fs => comp f (chain fs)

theorem chain_bounded {α : Type} (fs : List (Costed α α)) (bs : List Nat)
    (hlen : fs.length = bs.length)
    (h : ∀ i (hi : i < fs.length), Bounded (fs.get ⟨i, hi⟩) (fun _ => bs.get ⟨i, hlen ▸ hi⟩)) :
    Bounded (chain fs) (fun _ => bs.foldr (· + ·) 0) := by
  induction fs generalizing bs with
  | nil => intro _; simp [chain]
  | cons f fs ih =>
    cases bs with
    | nil => cases hlen
    | cons b bs =>
      intro a
      have h0 := h 0 (Nat.zero_lt_succ _)
      have ht : Bounded (chain fs) (fun _ => bs.foldr (· + ·) 0) :=
        ih bs (Nat.succ.inj hlen) (fun i hi => h (i + 1) (Nat.succ_lt_succ hi))
      simp only [chain, comp, List.foldr]
      exact Nat.add_le_add (h0 a) (ht _)

end RCE.Compose
