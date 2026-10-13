/-! Draft 0.6 / 0.6b parallel constructs (the draft 0.6 concurrency chapter, not published here C.7, C.11.14).
    What is mechanized here, on small models:
    * T16 core: branches with disjoint write sets give the source-order
      result under every interleaving of their writes (`interleave_eq_seq`).
    * T18 core: the fixed reduction tree equals the sequential left fold for
      any associative operation with a two-sided identity (`tree_eq_foldl`).
    * C.11.2 core: a block scan (scan both halves, offset the right half by
      the left total) equals the sequential exclusive scan (`scan_append`).
    * T24 one direction: with the standard vector clock update rules,
      happens-before implies strictly smaller clocks (`hb_lt`).
    * T29 core: puts into an LVar that join in a commutative, associative
      operation give the same state under every interleaving of two branches
      (`lvar_interleave`); a conflict is a join to an absorbing top, so the
      trap is reached on every schedule or on none (`lvar_top_absorbs`).
    Not mechanized: the link from these models to the RCE small-step
    semantics (footprints, W-PAR-1..4, W-GRP-1..3), the converse of T24, the
    threshold reads of T29, and T17, T19 to T23, T25 to T28, T30, T31. -/
namespace RCE.Par

/-! ## T16: disjoint writes commute -/

abbrev Store := Nat → Int
abbrev Writes := List (Nat × Int)

def upd (s : Store) (k : Nat) (v : Int) : Store := fun j => if j = k then v else s j

def apply (s : Store) : Writes → Store
  | [] => s
  | (k, v) :: ws => apply (upd s k v) ws

/-- Interleavings of two lists. -/
inductive Interleave {α : Type} : List α → List α → List α → Prop
  | nil : Interleave [] [] []
  | left (a : α) {xs ys zs} : Interleave xs ys zs → Interleave (a :: xs) ys (a :: zs)
  | right (b : α) {xs ys zs} : Interleave xs ys zs → Interleave xs (b :: ys) (b :: zs)

def keys (ws : Writes) : List Nat := ws.map Prod.fst

def Disjoint (xs ys : Writes) : Prop := ∀ k, k ∈ keys xs → k ∉ keys ys

theorem upd_comm (s : Store) {k k' : Nat} (v v' : Int) (h : k ≠ k') :
    upd (upd s k v) k' v' = upd (upd s k' v') k v := by
  funext j; unfold upd
  by_cases h1 : j = k' <;> by_cases h2 : j = k <;> simp_all

/-- A write to a key that `ws` never writes commutes past all of `ws`. -/
theorem apply_upd (s : Store) (k : Nat) (v : Int) (ws : Writes)
    (h : k ∉ keys ws) : apply (upd s k v) ws = upd (apply s ws) k v := by
  induction ws generalizing s with
  | nil => rfl
  | cons p ws ih =>
    obtain ⟨k', v'⟩ := p
    simp only [keys, List.map, List.mem_cons, not_or] at h
    simp only [apply]
    rw [upd_comm s v v' h.1]
    exact ih _ h.2

/-- T16 core: every interleaving of two branches with disjoint write sets
    gives the source-order result. -/
theorem interleave_eq_seq :
    ∀ {xs ys zs : Writes}, Interleave xs ys zs → Disjoint xs ys →
      ∀ s, apply s zs = apply (apply s xs) ys := by
  intro xs ys zs hi
  induction hi with
  | nil => intro _ s; rfl
  | left a _ ih =>
    intro hd s
    obtain ⟨k, v⟩ := a
    exact ih (fun j hj => hd j (List.mem_cons_of_mem _ hj)) _
  | right b _ ih =>
    intro hd s
    obtain ⟨k, v⟩ := b
    have hd' : Disjoint _ _ := fun j hj hy => hd j hj (List.mem_cons_of_mem _ hy)
    have hk : k ∉ keys _ := fun hx => hd k hx (List.mem_cons_self _ _)
    show apply (upd s k v) _ = apply (upd (apply s _) k v) _
    rw [ih hd', apply_upd s k v _ hk]

/-! ## T18: the fixed tree equals the left fold -/

section Tree
variable {α : Type} (f : α → α → α) (e : α)

/-- Reduction tree over a list: split at half, combine with `f`.
    `fuel` bounds the depth; `tree` passes the length. -/
def treeN : Nat → List α → α
  | 0, _ => e
  | _ + 1, [] => e
  | _ + 1, [x] => x
  | n + 1, xs => f (treeN n (xs.take (xs.length / 2))) (treeN n (xs.drop (xs.length / 2)))

def tree (xs : List α) : α := treeN f e xs.length xs

variable (assoc : ∀ a b c, f (f a b) c = f a (f b c))
variable (idl : ∀ a, f e a = a) (idr : ∀ a, f a e = a)
include assoc idl idr

theorem foldl_shift (a : α) (xs : List α) :
    xs.foldl f a = f a (xs.foldl f e) := by
  induction xs generalizing a with
  | nil => simp [List.foldl]; exact (idr a).symm
  | cons x xs ih =>
    simp only [List.foldl]
    rw [ih (f a x), ih (f e x), idl, assoc]

theorem foldl_append2 (xs ys : List α) :
    (xs ++ ys).foldl f e = f (xs.foldl f e) (ys.foldl f e) := by
  rw [List.foldl_append, foldl_shift f e assoc idl idr]

theorem treeN_eq (n : Nat) : ∀ xs : List α, xs.length ≤ n →
    treeN f e n xs = xs.foldl f e := by
  induction n with
  | zero =>
    intro xs h
    have : xs = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this; rfl
  | succ n ih =>
    intro xs h
    match xs, h with
    | [], _ => rfl
    | [x], _ => simp [treeN, List.foldl, idl]
    | x :: y :: zs, h =>
      have hl : (x :: y :: zs).length = zs.length + 2 := rfl
      simp only [treeN]
      rw [ih _ (by simp only [List.length_take, hl]; omega),
          ih _ (by simp only [List.length_drop, hl]; omega),
          ← foldl_append2 f e assoc idl idr, List.take_append_drop]

theorem tree_eq_foldl (xs : List α) : tree f e xs = xs.foldl f e :=
  treeN_eq f e assoc idl idr _ xs (Nat.le_refl _)

/-! ## C.11.2: block scan equals sequential scan -/

/-- Exclusive scan starting from accumulator `a`. -/
def scanx : α → List α → List α
  | _, [] => []
  | a, x :: xs => a :: scanx (f a x) xs

theorem scanx_shift (a : α) (xs : List α) :
    scanx f a xs = (scanx f e xs).map (f a) := by
  induction xs generalizing a with
  | nil => rfl
  | cons x xs ih =>
    simp only [scanx, List.map]
    rw [ih (f a x), ih (f e x), idr, idl, List.map_map]
    congr 1
    apply List.map_congr_left
    intro b _
    simp [Function.comp, assoc]

theorem scan_append (xs ys : List α) :
    scanx f e (xs ++ ys) =
      scanx f e xs ++ (scanx f e ys).map (f (xs.foldl f e)) := by
  suffices h : ∀ a, scanx f a (xs ++ ys) = scanx f a xs ++ scanx f (xs.foldl f a) ys by
    rw [h, scanx_shift f e assoc idl idr (xs.foldl f e)]
  intro a
  induction xs generalizing a with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.cons_append, scanx, List.foldl]
    exact congrArg _ (ih _)

end Tree

/-! ## T24 (one direction): happens-before implies smaller clocks -/

abbrev VC := Nat → Nat

def VC.le (a b : VC) : Prop := ∀ i, a i ≤ b i
def VC.lt (a b : VC) : Prop := VC.le a b ∧ ∃ i, a i < b i

theorem VC.lt_trans {a b c : VC} (h1 : VC.lt a b) (h2 : VC.lt b c) : VC.lt a c := by
  obtain ⟨l1, i, hi⟩ := h1
  obtain ⟨l2, _⟩ := h2
  exact ⟨fun j => Nat.le_trans (l1 j) (l2 j), i, Nat.lt_of_lt_of_le hi (l2 i)⟩

/-- Tick for process `p` after merging `m` (use `m = fun _ => 0` for a local step). -/
def tick (p : Nat) (own m : VC) : VC :=
  fun i => if i = p then max (own i) (m i) + 1 else max (own i) (m i)

theorem tick_gt_own (p : Nat) (own m : VC) : VC.lt own (tick p own m) := by
  refine ⟨fun i => ?_, p, ?_⟩
  · unfold tick; split <;> omega
  · simp [tick]; omega

theorem tick_gt_msg (p : Nat) (own m : VC) : VC.lt m (tick p own m) := by
  refine ⟨fun i => ?_, p, ?_⟩
  · unfold tick; split <;> omega
  · simp [tick]; omega

/-- A run: events with clocks; `next a b` when b follows a on the same
    process, `msg a b` when b receives the message sent at a. -/
structure Run (E : Type) where
  clock : E → VC
  proc : E → Nat
  next : E → E → Prop
  msg : E → E → Prop
  /-- Every event's clock is the tick of its predecessor merged with the message clock. -/
  step_next : ∀ a b, next a b → ∃ m, clock b = tick (proc b) (clock a) m
  step_msg : ∀ a b, msg a b → ∃ own, clock b = tick (proc b) own (clock a)

inductive HB {E : Type} (r : Run E) : E → E → Prop
  | next {a b} : r.next a b → HB r a b
  | msg {a b} : r.msg a b → HB r a b
  | trans {a b c} : HB r a b → HB r b c → HB r a c

theorem hb_lt {E : Type} (r : Run E) {a b : E} (h : HB r a b) :
    VC.lt (r.clock a) (r.clock b) := by
  induction h with
  | next hn =>
    obtain ⟨m, hm⟩ := r.step_next _ _ hn
    rw [hm]; exact tick_gt_own _ _ _
  | msg hm =>
    obtain ⟨o, ho⟩ := r.step_msg _ _ hm
    rw [ho]; exact tick_gt_msg _ _ _
  | trans _ _ ih1 ih2 => exact VC.lt_trans ih1 ih2

/-! ## T29: LVar puts are schedule independent -/

section LVar
variable {α : Type} (j : α → α → α)
variable (jassoc : ∀ a b c, j (j a b) c = j a (j b c)) (jcomm : ∀ a b, j a b = j b a)
include jassoc jcomm

theorem foldl_put_comm (s x : α) (xs : List α) :
    xs.foldl j (j s x) = j (xs.foldl j s) x := by
  induction xs generalizing s with
  | nil => rfl
  | cons y ys ih =>
    simp only [List.foldl]
    rw [← ih, jassoc, jcomm x y, ← jassoc]

theorem lvar_interleave (s : α) :
    ∀ {xs ys zs : List α}, Interleave xs ys zs →
      zs.foldl j s = (xs ++ ys).foldl j s := by
  intro xs ys zs hi
  induction hi generalizing s with
  | nil => rfl
  | left a _ ih => exact ih _
  | right b _ ih =>
    simp only [List.foldl]
    rw [ih, List.foldl_append, List.foldl_append, List.foldl_cons,
        foldl_put_comm j jassoc jcomm]

omit jassoc jcomm in
/-- A conflict joins to an absorbing top, so once reached it stays. -/
theorem lvar_top_absorbs (top : α) (htop : ∀ a, j top a = top) (xs : List α) :
    xs.foldl j top = top := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl, htop, ih]

end LVar
end RCE.Par
