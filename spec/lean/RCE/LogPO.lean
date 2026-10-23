/-! Model of the partial-order log (spec/proposals/partial-order-log.md):
    entries with a read and a write footprint over locations, an abstract
    store, independence as disjointness of footprints, and the theorem
    that two orderings of a layer that agree on every dependency give the
    same store (T62, a model in the sense of chapter 8).

    In `lake build` and the standing axiom audit. The proofs are by
    induction on the swap sequence and use only `simp`, `funext` and the
    footprint lemmas. -/
namespace RCE.LogPO

-- Locations and values are abstract.
variable {Loc Val : Type} [DecidableEq Loc]

abbrev Store (Loc Val : Type) := Loc → Val

/-- An entry: what it reads, what it writes, and how it writes, as a
    function of the values it read. `writes` lists the locations it may
    write; `apply` may change only those (`apply_frame` below is the
    assumption that makes this a footprint). -/
structure Entry (Loc Val : Type) where
  reads : List Loc
  writes : List Loc
  apply : Store Loc Val → Store Loc Val
  /-- the entry writes only its write footprint -/
  frame : ∀ s l, l ∉ writes → apply s l = s l
  /-- the entry's writes depend only on what it reads -/
  reads_only : ∀ s s', (∀ l ∈ reads, s l = s' l) → ∀ l ∈ writes, apply s l = apply s' l

/-- Two entries are independent when neither reads or writes what the
    other writes (W-PAR-1 footprint disjointness). -/
def Indep (a b : Entry Loc Val) : Prop :=
  (∀ l ∈ a.writes, l ∉ b.writes ∧ l ∉ b.reads) ∧ (∀ l ∈ b.writes, l ∉ a.reads)

/-- Independent entries commute on every store. -/
theorem commute (a b : Entry Loc Val) (h : Indep a b) (s : Store Loc Val) :
    b.apply (a.apply s) = a.apply (b.apply s) := by
  funext l
  by_cases hb : l ∈ b.writes
  · -- b writes l: a does not write l, and b's write reads nothing a wrote
    have ha : l ∉ a.writes := fun ha => (h.1 l ha).1 hb
    have h1 : b.apply (a.apply s) l = b.apply s l := by
      apply b.reads_only
      · intro r hr
        apply a.frame
        intro hw
        exact (h.1 r hw).2 hr
      · exact hb
    rw [h1]
    rw [a.frame (b.apply s) l ha]
  · by_cases ha : l ∈ a.writes
    · -- a writes l: b does not write l, and a's write reads nothing b wrote
      have h1 : a.apply (b.apply s) l = a.apply s l := by
        apply a.reads_only
        · intro r hr
          apply b.frame
          intro hw
          exact (h.2 r hw) hr
        · exact ha
      rw [b.frame (a.apply s) l hb, h1]
    · rw [b.frame (a.apply s) l hb, a.frame s l ha, a.frame (b.apply s) l ha, b.frame s l hb]

/-- Folding a list of entries over a store, in list order. -/
def fold : List (Entry Loc Val) → Store Loc Val → Store Loc Val
  | [], s => s
  | e :: es, s => fold es (e.apply s)

/-- One adjacent swap of independent entries. -/
inductive Swap : List (Entry Loc Val) → List (Entry Loc Val) → Prop where
  | here {a b : Entry Loc Val} {es} : Indep a b → Swap (a :: b :: es) (b :: a :: es)
  | there {e : Entry Loc Val} {xs ys} : Swap xs ys → Swap (e :: xs) (e :: ys)

/-- Dep-consistent reorderings: the reflexive transitive closure of swaps.
    Mazurkiewicz: two linearizations of one trace are related exactly by
    such a sequence, so this is the model of "any order that agrees with
    every chain's seq order and every deps edge". -/
inductive Reorder : List (Entry Loc Val) → List (Entry Loc Val) → Prop where
  | refl {xs} : Reorder xs xs
  | step {xs ys zs} : Swap xs ys → Reorder ys zs → Reorder xs zs

theorem fold_swap {xs ys : List (Entry Loc Val)} (h : Swap xs ys) (s : Store Loc Val) :
    fold xs s = fold ys s := by
  induction h generalizing s with
  | here hi => simp [fold, commute _ _ hi]
  | there _ ih => simp [fold, ih]

/-- **T62 (model)**: every dep-consistent ordering of a layer folds to the
    same store from the same cut. -/
theorem fold_perm_eq {xs ys : List (Entry Loc Val)} (h : Reorder xs ys) (s : Store Loc Val) :
    fold xs s = fold ys s := by
  induction h with
  | refl => rfl
  | step hs _ ih => rw [fold_swap hs s, ih]

end RCE.LogPO
