/-
T29q (C.11.8): LVars over a general join-semilattice, with threshold reads
in the lattice's own order and a frozen flag.

The lattice is any join-semilattice with a least element `bot` and an
absorbing `top` (the error state). This follows the lattice of Kuper and
Newton (FHPC 2013, Sec. 3.1) and the frozen status bit of Kuper, Turon,
Krishnaswami and Newton (POPL 2014, Sec. 4.2). The calculus is the
put / get / freeze fragment over one LVar shared by any number of threads;
a schedule picks any thread whose next operation can step:

* `put d`: the value becomes `v ⊔ d`; the step is `error` when `v ⊔ d = top`
  (E-PUT-ERR) or when the LVar is frozen and `v ⊔ d ≠ v` (a put that would
  change a frozen LVar);
* `get Q`: `Q` must be pairwise incompatible (`incomp`, E-GET); it returns
  an element `q ∈ Q` with `q ⊑ v`, in the same order `⊑` that `put` uses;
* `freeze`: sets the frozen bit and returns the exact value (E-FREEZE-SIMPLE).

Results (no `sorry`; `#print axioms` lists only propext, Quot.sound and
Classical.choice):
* `quasi_det`: two runs that complete every thread without error reach the
  same configuration: the same value, the same frozen bit and the same
  observations in every thread (POPL 2014, Theorem 1, shape; no location
  permutation is needed because there is one LVar and no allocation);
* `det_no_err`: for a freeze-free program, if one run completes then no run
  reaches `error`; with `quasi_det` this is determinism (FHPC 2013, Thm. 1,
  for complete runs);
* `quasi_not_det`: with `freeze` the error is schedule dependent, so the
  quasi-determinism statement cannot be strengthened.

Not covered: runs that block forever on a `get` (the theorems are about
complete and erroring runs), handlers and quiescence, and several LVars
with allocation. This is a model; the link to the evaluator is the rule
table in RCE-SPEC.md C.11.8 and the cross-target tests
(checker/tests/lvar_freeze.rs).
-/
namespace RCE.LVarQD

set_option linter.unusedSectionVars false
set_option linter.constructorNameAsVariable false

/-- A join-semilattice with bottom and an absorbing top. -/
class Lat (α : Type) where
  join : α → α → α
  bot : α
  top : α
  join_assoc : ∀ a b c, join (join a b) c = join a (join b c)
  join_comm : ∀ a b, join a b = join b a
  join_idem : ∀ a, join a a = a
  bot_join : ∀ a, join bot a = a
  top_join : ∀ a, join top a = top
  bot_ne_top : bot ≠ top

open Lat

section
variable {α : Type} [Lat α]

instance instAssocJoin : Std.Associative (α := α) join := ⟨join_assoc⟩
instance instCommJoin : Std.Commutative (α := α) join := ⟨join_comm⟩

/-- the order induced by the join: `a ⊑ b` iff `a ⊔ b = b` -/
def Le (a b : α) : Prop := join a b = b

theorem join_bot (a : α) : join a bot = a := by rw [join_comm]; exact bot_join a

theorem le_refl' (a : α) : Le a a := join_idem a

theorem le_join_left (a b : α) : Le a (join a b) := by
  unfold Le; rw [← join_assoc, join_idem]

theorem le_join_right (a b : α) : Le b (join a b) := by
  unfold Le; rw [← join_assoc, join_comm b a, join_assoc, join_idem]

theorem le_trans' {a b c : α} (h1 : Le a b) (h2 : Le b c) : Le a c := by
  unfold Le at *; rw [← h2, ← join_assoc, h1]

theorem join_le {a b c : α} (h1 : Le a c) (h2 : Le b c) : Le (join a b) c := by
  unfold Le at *; rw [join_assoc, h2, h1]

theorem top_le {v : α} (h : Le top v) : v = top := by
  unfold Le at h; rw [top_join] at h; exact h.symm

theorem join_mono {v a b : α} (h : Le a b) : Le (join v a) (join v b) :=
  join_le (le_join_left v b) (le_trans' h (le_join_right v b))

/-- pairwise incompatible threshold set (FHPC 2013 `incomp`) -/
def Incomp (q : List α) : Prop := ∀ a ∈ q, ∀ b ∈ q, a ≠ b → join a b = top

/-- at a value other than top, at most one threshold of an incompatible set
is reached, so `get` has one possible answer -/
theorem incomp_unique {q : List α} {v : α} (hq : Incomp q) (hv : v ≠ top) {a b : α}
    (ha : a ∈ q) (hb : b ∈ q) (hav : Le a v) (hbv : Le b v) : a = b := by
  apply Classical.byContradiction
  intro hne
  have h := join_le hav hbv
  rw [hq a ha b hb hne] at h
  exact hv (top_le h)

end

inductive Op (α : Type) where
  | put : α → Op α
  | get : List α → Op α
  | freeze : Op α

/-- a thread: the operations done, each with its result, and those to do -/
structure Th (α : Type) where
  done : List (Op α × α)
  todo : List (Op α)

structure Cfg (α : Type) where
  val : α
  frz : Bool
  ths : List (Th α)

inductive St (α : Type) where
  | err : St α
  | ok : Cfg α → St α

section
variable {α : Type} [Lat α]

/-- one step of one thread (`pre ++ t :: post` picks the thread) -/
inductive Step : Cfg α → St α → Prop where
  | put {v : α} {f : Bool} {pre post : List (Th α)} {dn : List (Op α × α)} {d : α}
      {tl : List (Op α)} :
      join v d ≠ top → (f = true → join v d = v) →
      Step ⟨v, f, pre ++ ⟨dn, .put d :: tl⟩ :: post⟩
        (.ok ⟨join v d, f, pre ++ ⟨dn ++ [(.put d, d)], tl⟩ :: post⟩)
  | put_top {v : α} {f : Bool} {pre post : List (Th α)} {dn : List (Op α × α)} {d : α}
      {tl : List (Op α)} :
      join v d = top → Step ⟨v, f, pre ++ ⟨dn, .put d :: tl⟩ :: post⟩ .err
  | put_frozen {v : α} {pre post : List (Th α)} {dn : List (Op α × α)} {d : α}
      {tl : List (Op α)} :
      join v d ≠ v → Step ⟨v, true, pre ++ ⟨dn, .put d :: tl⟩ :: post⟩ .err
  | get {v : α} {f : Bool} {pre post : List (Th α)} {dn : List (Op α × α)} {q : List α}
      {o : α} {tl : List (Op α)} :
      Incomp q → o ∈ q → Le o v →
      Step ⟨v, f, pre ++ ⟨dn, .get q :: tl⟩ :: post⟩
        (.ok ⟨v, f, pre ++ ⟨dn ++ [(.get q, o)], tl⟩ :: post⟩)
  | freeze {v : α} {f : Bool} {pre post : List (Th α)} {dn : List (Op α × α)}
      {tl : List (Op α)} :
      Step ⟨v, f, pre ++ ⟨dn, .freeze :: tl⟩ :: post⟩
        (.ok ⟨v, true, pre ++ ⟨dn ++ [(.freeze, v)], tl⟩ :: post⟩)

/-- runs from `s` -/
inductive Reach (s : St α) : St α → Prop where
  | refl : Reach s s
  | tail {c : Cfg α} {t : St α} : Reach s (.ok c) → Step c t → Reach s t

def init (P : List (List (Op α))) : Cfg α :=
  ⟨bot, false, P.map (fun ops => ⟨[], ops⟩)⟩

def Complete (c : Cfg α) : Prop := ∀ t ∈ c.ths, t.todo = []

def isFrz : Op α → Bool
  | .freeze => true
  | _ => false

def FreezeFree (P : List (List (Op α))) : Prop := ∀ ops ∈ P, ∀ o ∈ ops, isFrz o = false

/-- join of the values still to be put -/
def putsOf : List (Op α) → α
  | [] => bot
  | .put d :: tl => join d (putsOf tl)
  | .get _ :: tl => putsOf tl
  | .freeze :: tl => putsOf tl

def rem : List (Th α) → α
  | [] => bot
  | t :: ts => join (putsOf t.todo) (rem ts)

/-- the join of every put in the program -/
def J (P : List (List (Op α))) : α := rem (init P).ths

def prog (ths : List (Th α)) : List (List (Op α)) :=
  ths.map (fun t => t.done.map Prod.fst ++ t.todo)

def anyFrz : List (Op α × α) → Bool
  | [] => false
  | e :: es => isFrz e.1 || anyFrz es

def frozenDone : List (Th α) → Bool
  | [] => false
  | t :: ts => anyFrz t.done || frozenDone ts

/-- what a recorded result says at value `v` and frozen bit `f` -/
def Good (v : α) (f : Bool) : Op α × α → Prop
  | (.put d, o) => o = d
  | (.get q, o) => Incomp q ∧ o ∈ q ∧ Le o v
  | (.freeze, o) => f = true ∧ o = v

structure Inv (P : List (List (Op α))) (c : Cfg α) : Prop where
  sum : join c.val (rem c.ths) = J P
  ntop : c.val ≠ top
  shape : prog c.ths = P
  good : ∀ t ∈ c.ths, ∀ e ∈ t.done, Good c.val c.frz e
  frz : c.frz = frozenDone c.ths

theorem rem_append (a b : List (Th α)) : rem (a ++ b) = join (rem a) (rem b) := by
  induction a with
  | nil => simp only [List.nil_append, rem, bot_join]
  | cons t ts ih =>
    show join (putsOf t.todo) (rem (ts ++ b)) = join (join (putsOf t.todo) (rem ts)) (rem b)
    rw [ih, join_assoc]

theorem anyFrz_append (a b : List (Op α × α)) : anyFrz (a ++ b) = (anyFrz a || anyFrz b) := by
  induction a with
  | nil => simp [anyFrz]
  | cons e es ih => simp [anyFrz, ih, Bool.or_assoc]

theorem frozenDone_append (a b : List (Th α)) :
    frozenDone (a ++ b) = (frozenDone a || frozenDone b) := by
  induction a with
  | nil => simp [frozenDone]
  | cons e es ih => simp [frozenDone, ih, Bool.or_assoc]

theorem good_mono {v v' : α} {f f' : Bool} (hle : Le v v')
    (hf : f = true → f' = true ∧ v' = v) : ∀ e, Good v f e → Good v' f' e
  | (.put _, _), h => h
  | (.get _, _), h => ⟨h.1, h.2.1, le_trans' h.2.2 hle⟩
  | (.freeze, _), h => by
      obtain ⟨h1, h2⟩ := hf h.1
      exact ⟨h1, by rw [h2]; exact h.2⟩

theorem prog_mid (pre post : List (Th α)) (dn : List (Op α × α)) (op : Op α) (o : α)
    (tl : List (Op α)) :
    prog (pre ++ ⟨dn ++ [(op, o)], tl⟩ :: post) = prog (pre ++ ⟨dn, op :: tl⟩ :: post) := by
  simp [prog]

/-- the invariant moves to the new thread list -/
theorem good_mid {v v' : α} {f f' : Bool} {pre post : List (Th α)} {dn : List (Op α × α)}
    {op : Op α} {o : α} {tl : List (Op α)}
    (hg : ∀ t ∈ pre ++ ⟨dn, op :: tl⟩ :: post, ∀ e ∈ t.done, Good v f e)
    (hm : ∀ e, Good v f e → Good v' f' e) (hn : Good v' f' (op, o)) :
    ∀ t ∈ pre ++ ⟨dn ++ [(op, o)], tl⟩ :: post, ∀ e ∈ t.done, Good v' f' e := by
  intro t ht e he
  simp only [List.mem_append, List.mem_cons] at ht
  rcases ht with h | h | h
  · exact hm e (hg t (by simp [h]) e he)
  · subst h
    simp only [List.mem_append, List.mem_singleton] at he
    rcases he with he | he
    · exact hm e (hg ⟨dn, op :: tl⟩ (by simp) e he)
    · subst he; exact hn
  · exact hm e (hg t (by simp [h]) e he)

theorem inv_init (P : List (List (Op α))) : Inv P (init P) where
  sum := by simp only [init, bot_join, J]
  ntop := bot_ne_top
  shape := by
    simp only [prog, init, List.map_map]
    have : ((fun t : Th α => t.done.map Prod.fst ++ t.todo) ∘ fun ops => (⟨[], ops⟩ : Th α)) = id := by
      funext x; rfl
    rw [this, List.map_id]
  good := by
    intro t ht e he
    simp only [init, List.mem_map] at ht
    obtain ⟨_, _, rfl⟩ := ht
    simp at he
  frz := by
    simp only [init]
    induction P with
    | nil => rfl
    | cons _ _ ih => simp only [List.map_cons, frozenDone, anyFrz]; exact ih

theorem inv_step {P : List (List (Op α))} {c c' : Cfg α} (h : Inv P c)
    (hs : Step c (.ok c')) : Inv P c' := by
  cases hs with
  | @put v f pre post dn d tl hnt hfz =>
    refine ⟨?_, hnt, ?_, ?_, ?_⟩
    · rw [← h.sum]; simp only [rem_append, rem, putsOf]; ac_rfl
    · rw [prog_mid]; exact h.shape
    · exact good_mid h.good
        (good_mono (le_join_left v d) (fun hf => ⟨hf, hfz hf⟩)) rfl
    · have := h.frz
      simp only [frozenDone_append, frozenDone, anyFrz_append, anyFrz, isFrz] at this ⊢
      simpa using this
  | @get v f pre post dn q o tl hq hm hl =>
    refine ⟨?_, h.ntop, ?_, ?_, ?_⟩
    · rw [← h.sum]; simp only [rem_append, rem, putsOf]
    · rw [prog_mid]; exact h.shape
    · exact good_mid h.good (good_mono (le_refl' v) (fun hf => ⟨hf, rfl⟩)) ⟨hq, hm, hl⟩
    · have := h.frz
      simp only [frozenDone_append, frozenDone, anyFrz_append, anyFrz, isFrz] at this ⊢
      simpa using this
  | @freeze v f pre post dn tl =>
    refine ⟨?_, h.ntop, ?_, ?_, ?_⟩
    · rw [← h.sum]; simp only [rem_append, rem, putsOf]
    · rw [prog_mid]; exact h.shape
    · exact good_mid h.good (good_mono (le_refl' v) (fun _ => ⟨rfl, rfl⟩)) ⟨rfl, rfl⟩
    · simp [frozenDone_append, frozenDone, anyFrz_append, anyFrz, isFrz]

theorem inv_reach {P : List (List (Op α))} {s : St α} (h : Reach (.ok (init P)) s) :
    ∀ c, s = .ok c → Inv P c := by
  induction h with
  | refl => intro c hc; injection hc with hc; subst hc; exact inv_init P
  | tail _ hst ih => intro c hc; subst hc; exact inv_step (ih _ rfl) hst

theorem rem_complete : ∀ {ths : List (Th α)}, (∀ t ∈ ths, t.todo = []) → rem ths = bot
  | [], _ => rfl
  | t :: ts, h => by
    simp only [rem]
    rw [h t (List.mem_cons_self _ _), rem_complete (fun t' ht' => h t' (List.mem_cons_of_mem _ ht'))]
    simp only [putsOf, bot_join]

theorem val_complete {P : List (List (Op α))} {c : Cfg α} (h : Inv P c) (k : Complete c) :
    c.val = J P := by
  have := h.sum; rw [rem_complete k, join_bot] at this; exact this

/-- at the final value every recorded result is determined by the operation -/
theorem good_det {Jv : α} (hJ : Jv ≠ top) {f1 f2 : Bool} :
    ∀ {e1 e2 : Op α × α}, Good Jv f1 e1 → Good Jv f2 e2 → e1.1 = e2.1 → e1 = e2
  | (.put _, _), (.put _, _), h1, h2, hop => by
      simp only [Good, Op.put.injEq] at h1 h2 hop ⊢; subst hop; rw [h1, h2]
  | (.get _, _), (.get _, _), h1, h2, hop => by
      simp only [Op.get.injEq] at hop; subst hop
      simp only [Good] at h1 h2
      rw [incomp_unique h1.1 hJ h1.2.1 h2.2.1 h1.2.2 h2.2.2]
  | (.freeze, _), (.freeze, _), h1, h2, _ => by
      simp only [Good] at h1 h2; rw [h1.2, h2.2]
  | (.put _, _), (.get _, _), _, _, hop => by simp at hop
  | (.put _, _), (.freeze, _), _, _, hop => by simp at hop
  | (.get _, _), (.put _, _), _, _, hop => by simp at hop
  | (.get _, _), (.freeze, _), _, _, hop => by simp at hop
  | (.freeze, _), (.put _, _), _, _, hop => by simp at hop
  | (.freeze, _), (.get _, _), _, _, hop => by simp at hop

theorem list_det {G : Op α × α → Prop} (hG : ∀ e1 e2, G e1 → G e2 → e1.1 = e2.1 → e1 = e2) :
    ∀ (l1 l2 : List (Op α × α)), l1.map Prod.fst = l2.map Prod.fst →
      (∀ e ∈ l1, G e) → (∀ e ∈ l2, G e) → l1 = l2
  | [], [], _, _, _ => rfl
  | [], _ :: _, h, _, _ => by simp at h
  | _ :: _, [], h, _, _ => by simp at h
  | a :: as, b :: bs, h, g1, g2 => by
      simp only [List.map_cons, List.cons.injEq] at h
      rw [hG a b (g1 a (List.mem_cons_self _ _)) (g2 b (List.mem_cons_self _ _)) h.1,
        list_det hG as bs h.2 (fun e he => g1 e (List.mem_cons_of_mem _ he))
          (fun e he => g2 e (List.mem_cons_of_mem _ he))]

theorem ths_det {G : Op α × α → Prop} (hG : ∀ e1 e2, G e1 → G e2 → e1.1 = e2.1 → e1 = e2) :
    ∀ (t1 t2 : List (Th α)), prog t1 = prog t2 →
      (∀ t ∈ t1, t.todo = []) → (∀ t ∈ t2, t.todo = []) →
      (∀ t ∈ t1, ∀ e ∈ t.done, G e) → (∀ t ∈ t2, ∀ e ∈ t.done, G e) → t1 = t2
  | [], [], _, _, _, _, _ => rfl
  | [], _ :: _, h, _, _, _, _ => by simp [prog] at h
  | _ :: _, [], h, _, _, _, _ => by simp [prog] at h
  | ⟨ad, atd⟩ :: as, ⟨bd, btd⟩ :: bs, h, k1, k2, g1, g2 => by
      have ta : atd = [] := k1 _ (List.mem_cons_self _ _)
      have tb : btd = [] := k2 _ (List.mem_cons_self _ _)
      subst ta; subst tb
      simp only [prog, List.map_cons, List.cons.injEq, List.append_nil] at h
      have hd : ad = bd := list_det hG ad bd h.1 (g1 _ (List.mem_cons_self _ _))
        (g2 _ (List.mem_cons_self _ _))
      have hr : as = bs := ths_det hG as bs h.2
        (fun t ht => k1 t (List.mem_cons_of_mem _ ht)) (fun t ht => k2 t (List.mem_cons_of_mem _ ht))
        (fun t ht => g1 t (List.mem_cons_of_mem _ ht)) (fun t ht => g2 t (List.mem_cons_of_mem _ ht))
      rw [hd, hr]

/-- **Quasi-determinism.** Two runs of the same program that complete every
thread without error end in the same configuration, with or without
`freeze`. -/
theorem quasi_det {P : List (List (Op α))} {c1 c2 : Cfg α}
    (h1 : Reach (.ok (init P)) (.ok c1)) (h2 : Reach (.ok (init P)) (.ok c2))
    (k1 : Complete c1) (k2 : Complete c2) : c1 = c2 := by
  have i1 := inv_reach h1 c1 rfl
  have i2 := inv_reach h2 c2 rfl
  have v1 := val_complete i1 k1
  have v2 := val_complete i2 k2
  have hJ : J P ≠ top := by rw [← v1]; exact i1.ntop
  have hths : c1.ths = c2.ths :=
    ths_det (G := fun e => ∃ f, Good (J P) f e)
      (fun _ _ ⟨_, g1⟩ ⟨_, g2⟩ hop => good_det hJ g1 g2 hop) c1.ths c2.ths
      (by rw [i1.shape, i2.shape]) k1 k2
      (fun t ht e he => ⟨c1.frz, by rw [← v1]; exact i1.good t ht e he⟩)
      (fun t ht e he => ⟨c2.frz, by rw [← v2]; exact i2.good t ht e he⟩)
  have hf : c1.frz = c2.frz := by rw [i1.frz, i2.frz, hths]
  obtain ⟨a1, f1, t1⟩ := c1
  obtain ⟨a2, f2, t2⟩ := c2
  simp only at v1 v2 hths hf
  subst v1; subst v2; subst hf; subst hths; rfl

theorem frozenDone_false : ∀ {ths : List (Th α)},
    (∀ t ∈ ths, ∀ e ∈ t.done, isFrz e.1 = false) → frozenDone ths = false
  | [], _ => rfl
  | t :: ts, h => by
    have ha : ∀ {l : List (Op α × α)}, (∀ e ∈ l, isFrz e.1 = false) → anyFrz l = false := by
      intro l
      induction l with
      | nil => intro _; rfl
      | cons e es ih =>
        intro hl
        simp only [anyFrz, hl e (List.mem_cons_self _ _), ih (fun x hx => hl x (List.mem_cons_of_mem _ hx)),
          Bool.or_self]
    simp only [frozenDone, ha (h t (List.mem_cons_self _ _)),
      frozenDone_false (fun t' ht' => h t' (List.mem_cons_of_mem _ ht')), Bool.or_self]

/-- **Determinism without freeze.** If a freeze-free program has a run that
completes, no run of it reaches `error`. -/
theorem det_no_err {P : List (List (Op α))} (hP : FreezeFree P) {c : Cfg α}
    (h1 : Reach (.ok (init P)) (.ok c)) (k : Complete c) : ¬ Reach (.ok (init P)) .err := by
  intro he
  have ic := inv_reach h1 c rfl
  have hJ : J P ≠ top := by rw [← val_complete ic k]; exact ic.ntop
  cases he with
  | tail hr hs =>
    rename_i c0
    have i0 := inv_reach hr c0 rfl
    cases hs with
    | @put_top v f pre post dn d tl htop =>
      have hd : Le d (rem (pre ++ ⟨dn, .put d :: tl⟩ :: post)) := by
        simp only [rem_append, rem, putsOf]
        exact le_trans' (le_trans' (le_join_left d _) (le_join_left _ _)) (le_join_right _ _)
      have := join_mono (v := v) hd
      rw [htop, i0.sum] at this
      exact hJ (top_le this)
    | @put_frozen v pre post dn d tl _ =>
      have hf := i0.frz
      simp only at hf
      rw [frozenDone_false] at hf
      · exact Bool.noConfusion hf
      · intro t ht e he
        have hm : t.done.map Prod.fst ++ t.todo ∈ P := by
          rw [← i0.shape]; exact List.mem_map.2 ⟨t, ht, rfl⟩
        exact hP _ hm e.1 (List.mem_append_left _ (List.mem_map.2 ⟨e, he, rfl⟩))

end

/-! A three-point lattice `b ⊑ m ⊑ t`, and a program where `freeze` races
with a `put`: one schedule completes, another errs. -/

inductive Three where
  | b | m | t
  deriving DecidableEq

def Three.join : Three → Three → Three
  | .b, x => x
  | x, .b => x
  | .m, .m => .m
  | _, _ => .t

instance : Lat Three where
  join := Three.join
  bot := .b
  top := .t
  join_assoc := by intro a b c; cases a <;> cases b <;> cases c <;> rfl
  join_comm := by intro a b; cases a <;> cases b <;> rfl
  join_idem := by intro a; cases a <;> rfl
  bot_join := by intro a; cases a <;> rfl
  top_join := by intro a; cases a <;> rfl
  bot_ne_top := by intro h; cases h

def race : List (List (Op Three)) := [[.put .m], [.freeze]]

theorem quasi_not_det :
    (∃ c, Reach (.ok (init race)) (.ok c) ∧ Complete c) ∧ Reach (.ok (init race)) .err := by
  refine ⟨⟨⟨.m, true, [⟨[(.put .m, .m)], []⟩, ⟨[(.freeze, .m)], []⟩]⟩, ?_, ?_⟩, ?_⟩
  · -- put, then freeze
    have s1 : Step (α := Three) ⟨.b, false, [] ++ ⟨[], [.put .m]⟩ :: [⟨[], [.freeze]⟩]⟩
        (.ok ⟨.m, false, [] ++ ⟨[] ++ [(.put .m, .m)], []⟩ :: [⟨[], [.freeze]⟩]⟩) :=
      Step.put (v := Three.b) (d := Three.m) (by decide) (by intro h; cases h)
    have s2 : Step (α := Three) ⟨.m, false, [⟨[(.put .m, .m)], []⟩] ++ ⟨[], [.freeze]⟩ :: []⟩
        (.ok ⟨.m, true, [⟨[(.put .m, .m)], []⟩] ++ ⟨[] ++ [(.freeze, .m)], []⟩ :: []⟩) :=
      Step.freeze
    exact Reach.tail (Reach.tail Reach.refl s1) s2
  · intro t ht; simp at ht; rcases ht with rfl | rfl <;> rfl
  · -- freeze, then put: the put would change a frozen LVar
    have s1 : Step (α := Three) ⟨.b, false, [⟨[], [.put .m]⟩] ++ ⟨[], [.freeze]⟩ :: []⟩
        (.ok ⟨.b, true, [⟨[], [.put .m]⟩] ++ ⟨[] ++ [(.freeze, .b)], []⟩ :: []⟩) :=
      Step.freeze
    have s2 : Step (α := Three) ⟨.b, true, [] ++ ⟨[], [.put .m]⟩ :: [⟨[(.freeze, .b)], []⟩]⟩ .err :=
      Step.put_frozen (v := Three.b) (d := Three.m) (by decide)
    exact Reach.tail (Reach.tail Reach.refl s1) s2

end RCE.LVarQD
