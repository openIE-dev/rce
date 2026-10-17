/-
The canonical representation of sets and maps is unique (7.1, C.11.5).

The evaluator and the four emitter runtimes keep a `Set[τ]` as a list
sorted strictly by the canonical order (`canon_cmp`), and a `Map[k, v]` as
an association list sorted strictly by key. These theorems say that the
representation is determined by the value:

* `set_canon`: two strictly sorted lists with the same elements are equal;
* `map_canon`: two association lists strictly sorted by key with the same
  lookup are equal.

So equality of representations is extensional equality, which is what the
Merge laws of `MergeLaws.setSL` / `mapSL` and `MergeCert.unionS` / `mapS`
(stated on membership predicates and lookup functions) need to transfer to
the runtime values. The order needs only to be irreflexive and
asymmetric; that `canon_cmp` is a strict total order on canonical values
is assumed here (it is a lexicographic order on tagged values).
-/
namespace RCE.Canon

variable {α : Type} (lt : α → α → Prop)

theorem set_canon (irr : ∀ a, ¬ lt a a) (asym : ∀ a b, lt a b → ¬ lt b a) :
    ∀ {l1 l2 : List α}, l1.Pairwise lt → l2.Pairwise lt → (∀ x, x ∈ l1 ↔ x ∈ l2) → l1 = l2
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => absurd ((h b).2 (List.mem_cons_self _ _)) (List.not_mem_nil _)
  | a :: _, [], _, _, h => absurd ((h a).1 (List.mem_cons_self _ _)) (List.not_mem_nil _)
  | a :: t1, b :: t2, p1, p2, h => by
    rw [List.pairwise_cons] at p1 p2
    have hab : a = b := by
      apply Classical.byContradiction
      intro ne
      have ha : a ∈ t2 := by
        rcases List.mem_cons.1 ((h a).1 (List.mem_cons_self _ _)) with e | e
        · exact absurd e ne
        · exact e
      have hb : b ∈ t1 := by
        rcases List.mem_cons.1 ((h b).2 (List.mem_cons_self _ _)) with e | e
        · exact absurd e.symm ne
        · exact e
      exact asym _ _ (p1.1 b hb) (p2.1 a ha)
    subst hab
    have ht : t1 = t2 := set_canon irr asym p1.2 p2.2 (fun x => by
      constructor
      · intro hx
        have hne : x ≠ a := fun e => by subst e; exact irr _ (p1.1 _ hx)
        rcases List.mem_cons.1 ((h x).1 (List.mem_cons_of_mem _ hx)) with e | e
        · exact absurd e hne
        · exact e
      · intro hx
        have hne : x ≠ a := fun e => by subst e; exact irr _ (p2.1 _ hx)
        rcases List.mem_cons.1 ((h x).2 (List.mem_cons_of_mem _ hx)) with e | e
        · exact absurd e hne
        · exact e)
    rw [ht]

variable {β : Type} [DecidableEq α]

def lookup (k : α) : List (α × β) → Option β
  | [] => none
  | (k', v) :: t => if k' = k then some v else lookup k t

/-- in a list strictly sorted by key, a pair is present iff lookup finds it -/
theorem mem_iff_lookup (irr : ∀ a, ¬ lt a a) :
    ∀ {l : List (α × β)}, l.Pairwise (fun p q => lt p.1 q.1) →
      ∀ k v, (k, v) ∈ l ↔ lookup k l = some v
  | [], _, k, v => by simp [lookup]
  | (k', v') :: t, p, k, v => by
    rw [List.pairwise_cons] at p
    simp only [lookup, List.mem_cons, Prod.mk.injEq]
    by_cases e : k' = k
    · subst e
      simp only [if_true, Option.some.injEq, true_and]
      constructor
      · rintro (h | h)
        · exact h.symm
        · exact absurd (p.1 _ h) (irr _)
      · intro h; exact Or.inl h.symm
    · rw [if_neg e, ← mem_iff_lookup irr p.2 k v]
      constructor
      · rintro (h | h)
        · exact absurd h.1.symm e
        · exact h
      · intro h; exact Or.inr h

theorem map_canon (irr : ∀ a, ¬ lt a a) (asym : ∀ a b, lt a b → ¬ lt b a)
    {l1 l2 : List (α × β)} (p1 : l1.Pairwise (fun p q => lt p.1 q.1))
    (p2 : l2.Pairwise (fun p q => lt p.1 q.1)) (h : ∀ k, lookup k l1 = lookup k l2) : l1 = l2 :=
  set_canon (fun p q => lt p.1 q.1) (fun a => irr a.1) (fun a b => asym a.1 b.1) p1 p2
    (fun ⟨k, v⟩ => by rw [mem_iff_lookup lt irr p1, mem_iff_lookup lt irr p2, h k])

end RCE.Canon
