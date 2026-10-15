/-
T29s (C.11.8): quasi-determinism for a store of many LVars shared by any
number of threads.

The store is a map from locations `ι` to values of a lattice `α` (the
`Lat` of LVarQD.lean), with a frozen bit per location. Its states are the
product lattice `ι → α` ordered pointwise; the store is in error as soon as
one location reaches `top`. Operations name their location:

* `put l d`: location `l` becomes `v l ⊔ d`; error when that is `top`, or
  when `l` is frozen and the put would change it;
* `get l Q`: `Q` pairwise incompatible, returns the `q ∈ Q` with `q ⊑ v l`;
* `freeze l`: sets the frozen bit of `l` and returns `v l`.

Locations stand for the `lvar_new` sites of a program, allocated before the
threads start (the store is the product over all of them; a location never
used is `bot` throughout). Dynamic allocation with renaming of locations
(POPL 2014, Sec. 5, permutations) is not modelled.

Results (no `sorry`):
* `store_quasi_det`: two runs that complete every thread without error reach
  the same store, the same frozen bits and the same observations;
* `store_det_no_err`: a freeze-free program with one complete run has no
  run that reaches `error`.
-/
import RCE.LVarQD

namespace RCE.LVarStore

open RCE.LVarQD RCE.LVarQD.Lat

set_option linter.unusedSectionVars false
set_option linter.constructorNameAsVariable false

variable {ι α : Type} [DecidableEq ι] [Lat α]

inductive Op (ι α : Type) where
  | put : ι → α → Op ι α
  | get : ι → List α → Op ι α
  | freeze : ι → Op ι α

structure Th (ι α : Type) where
  done : List (Op ι α × α)
  todo : List (Op ι α)

structure Cfg (ι α : Type) where
  val : ι → α
  frz : ι → Bool
  ths : List (Th ι α)

inductive St (ι α : Type) where
  | err : St ι α
  | ok : Cfg ι α → St ι α

def upd {β : Type} (v : ι → β) (l : ι) (x : β) : ι → β := fun k => if k = l then x else v k

inductive Step : Cfg ι α → St ι α → Prop where
  | put {v : ι → α} {f : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {d : α} {tl : List (Op ι α)} :
      join (v l) d ≠ top → (f l = true → join (v l) d = v l) →
      Step ⟨v, f, pre ++ ⟨dn, .put l d :: tl⟩ :: post⟩
        (.ok ⟨upd v l (join (v l) d), f, pre ++ ⟨dn ++ [(.put l d, d)], tl⟩ :: post⟩)
  | put_top {v : ι → α} {f : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {d : α} {tl : List (Op ι α)} :
      join (v l) d = top → Step ⟨v, f, pre ++ ⟨dn, .put l d :: tl⟩ :: post⟩ .err
  | put_frozen {v : ι → α} {f : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {d : α} {tl : List (Op ι α)} :
      f l = true → join (v l) d ≠ v l → Step ⟨v, f, pre ++ ⟨dn, .put l d :: tl⟩ :: post⟩ .err
  | get {v : ι → α} {f : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {q : List α} {o : α} {tl : List (Op ι α)} :
      Incomp q → o ∈ q → Le o (v l) →
      Step ⟨v, f, pre ++ ⟨dn, .get l q :: tl⟩ :: post⟩
        (.ok ⟨v, f, pre ++ ⟨dn ++ [(.get l q, o)], tl⟩ :: post⟩)
  | freeze {v : ι → α} {f : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {tl : List (Op ι α)} :
      Step ⟨v, f, pre ++ ⟨dn, .freeze l :: tl⟩ :: post⟩
        (.ok ⟨v, upd f l true, pre ++ ⟨dn ++ [(.freeze l, v l)], tl⟩ :: post⟩)

inductive Reach (s : St ι α) : St ι α → Prop where
  | refl : Reach s s
  | tail {c : Cfg ι α} {t : St ι α} : Reach s (.ok c) → Step c t → Reach s t

def init (P : List (List (Op ι α))) : Cfg ι α :=
  ⟨fun _ => bot, fun _ => false, P.map (fun ops => ⟨[], ops⟩)⟩

def Complete (c : Cfg ι α) : Prop := ∀ t ∈ c.ths, t.todo = []

def frzAt (k : ι) : Op ι α → Bool
  | .freeze l => decide (l = k)
  | _ => false

def isFrz : Op ι α → Bool
  | .freeze _ => true
  | _ => false

def FreezeFree (P : List (List (Op ι α))) : Prop := ∀ ops ∈ P, ∀ o ∈ ops, isFrz o = false

def putsOf (k : ι) : List (Op ι α) → α
  | [] => bot
  | .put l d :: tl => if l = k then join d (putsOf k tl) else putsOf k tl
  | .get _ _ :: tl => putsOf k tl
  | .freeze _ :: tl => putsOf k tl

def rem (k : ι) : List (Th ι α) → α
  | [] => bot
  | t :: ts => join (putsOf k t.todo) (rem k ts)

def J (P : List (List (Op ι α))) (k : ι) : α := rem k (init P).ths

def prog (ths : List (Th ι α)) : List (List (Op ι α)) :=
  ths.map (fun t => t.done.map Prod.fst ++ t.todo)

def anyFrz (k : ι) : List (Op ι α × α) → Bool
  | [] => false
  | e :: es => frzAt k e.1 || anyFrz k es

def frozenDone (k : ι) : List (Th ι α) → Bool
  | [] => false
  | t :: ts => anyFrz k t.done || frozenDone k ts

def Good (v : ι → α) (f : ι → Bool) : Op ι α × α → Prop
  | (.put _ d, o) => o = d
  | (.get l q, o) => Incomp q ∧ o ∈ q ∧ Le o (v l)
  | (.freeze l, o) => f l = true ∧ o = v l

structure Inv (P : List (List (Op ι α))) (c : Cfg ι α) : Prop where
  sum : ∀ k, join (c.val k) (rem k c.ths) = J P k
  ntop : ∀ k, c.val k ≠ top
  shape : prog c.ths = P
  good : ∀ t ∈ c.ths, ∀ e ∈ t.done, Good c.val c.frz e
  frz : ∀ k, c.frz k = frozenDone k c.ths

theorem rem_append (k : ι) (a b : List (Th ι α)) : rem k (a ++ b) = join (rem k a) (rem k b) := by
  induction a with
  | nil => simp only [List.nil_append, rem, bot_join]
  | cons t ts ih =>
    show join (putsOf k t.todo) (rem k (ts ++ b)) = join (join (putsOf k t.todo) (rem k ts)) (rem k b)
    rw [ih, join_assoc]

theorem anyFrz_append (k : ι) (a b : List (Op ι α × α)) :
    anyFrz k (a ++ b) = (anyFrz k a || anyFrz k b) := by
  induction a with
  | nil => simp [anyFrz]
  | cons e es ih => simp [anyFrz, ih, Bool.or_assoc]

theorem frozenDone_append (k : ι) (a b : List (Th ι α)) :
    frozenDone k (a ++ b) = (frozenDone k a || frozenDone k b) := by
  induction a with
  | nil => simp [frozenDone]
  | cons e es ih => simp [frozenDone, ih, Bool.or_assoc]

theorem good_mono {v v' : ι → α} {f f' : ι → Bool} (hle : ∀ k, Le (v k) (v' k))
    (hf : ∀ k, f k = true → f' k = true ∧ v' k = v k) : ∀ e, Good v f e → Good v' f' e
  | (.put _ _, _), h => h
  | (.get l _, _), h => ⟨h.1, h.2.1, le_trans' h.2.2 (hle l)⟩
  | (.freeze l, _), h => by
      obtain ⟨h1, h2⟩ := hf l h.1
      exact ⟨h1, by rw [h2]; exact h.2⟩

theorem prog_mid (pre post : List (Th ι α)) (dn : List (Op ι α × α)) (op : Op ι α) (o : α)
    (tl : List (Op ι α)) :
    prog (pre ++ ⟨dn ++ [(op, o)], tl⟩ :: post) = prog (pre ++ ⟨dn, op :: tl⟩ :: post) := by
  simp [prog]

theorem good_mid {v v' : ι → α} {f f' : ι → Bool} {pre post : List (Th ι α)}
    {dn : List (Op ι α × α)} {op : Op ι α} {o : α} {tl : List (Op ι α)}
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

theorem inv_init (P : List (List (Op ι α))) : Inv P (init P) where
  sum k := by simp only [init, bot_join, J]
  ntop _ := bot_ne_top
  shape := by
    simp only [prog, init, List.map_map]
    have : ((fun t : Th ι α => t.done.map Prod.fst ++ t.todo) ∘
        fun ops => (⟨[], ops⟩ : Th ι α)) = id := by
      funext x; rfl
    rw [this, List.map_id]
  good := by
    intro t ht e he
    simp only [init, List.mem_map] at ht
    obtain ⟨_, _, rfl⟩ := ht
    simp at he
  frz k := by
    simp only [init]
    induction P with
    | nil => rfl
    | cons _ _ ih => simp only [List.map_cons, frozenDone, anyFrz]; exact ih

theorem inv_step {P : List (List (Op ι α))} {c c' : Cfg ι α} (h : Inv P c)
    (hs : Step c (.ok c')) : Inv P c' := by
  cases hs with
  | @put v f pre post dn l d tl hnt hfz =>
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · intro k
      have hk := h.sum k
      simp only [rem_append, rem, putsOf] at hk ⊢
      by_cases e : k = l
      · subst e; simp only [upd, if_pos] at hk ⊢; rw [← hk]; ac_rfl
      · have e' : ¬ l = k := fun h => e h.symm
        simp only [upd, if_neg e, if_neg e'] at hk ⊢; exact hk
    · intro k; by_cases e : k = l
      · subst e; simp only [upd, if_pos]; exact hnt
      · simp only [upd, if_neg e]; exact h.ntop k
    · rw [prog_mid]; exact h.shape
    · refine good_mid h.good (good_mono ?_ ?_) rfl
      · intro k; by_cases e : k = l
        · subst e; simp only [upd, if_pos]; exact le_join_left _ _
        · simp only [upd, if_neg e]; exact le_refl' _
      · intro k hf; refine ⟨hf, ?_⟩; by_cases e : k = l
        · subst e; simp only [upd, if_pos]; exact hfz hf
        · simp only [upd, if_neg e]
    · intro k
      have := h.frz k
      simp only [frozenDone_append, frozenDone, anyFrz_append, anyFrz, frzAt] at this ⊢
      simpa using this
  | @get v f pre post dn l q o tl hq hm hl =>
    refine ⟨?_, h.ntop, ?_, ?_, ?_⟩
    · intro k; have hk := h.sum k; simp only [rem_append, rem, putsOf] at hk ⊢; exact hk
    · rw [prog_mid]; exact h.shape
    · exact good_mid h.good (good_mono (fun _ => le_refl' _) (fun _ hf => ⟨hf, rfl⟩)) ⟨hq, hm, hl⟩
    · intro k
      have := h.frz k
      simp only [frozenDone_append, frozenDone, anyFrz_append, anyFrz, frzAt] at this ⊢
      simpa using this
  | @freeze v f pre post dn l tl =>
    refine ⟨?_, h.ntop, ?_, ?_, ?_⟩
    · intro k; have hk := h.sum k; simp only [rem_append, rem, putsOf] at hk ⊢; exact hk
    · rw [prog_mid]; exact h.shape
    · refine good_mid h.good (good_mono (fun _ => le_refl' _) ?_) ⟨by simp [upd], rfl⟩
      intro k hf; refine ⟨?_, rfl⟩
      by_cases e : k = l
      · subst e; simp [upd]
      · simp only [upd, if_neg e]; exact hf
    · intro k
      have := h.frz k
      by_cases e : k = l
      · subst e
        simp [upd, frozenDone_append, frozenDone, anyFrz_append, anyFrz, frzAt]
      · simp only [frozenDone_append, frozenDone, anyFrz_append, anyFrz, frzAt] at this ⊢
        simp only [upd, if_neg e]
        have e' : ¬ l = k := fun h' => e h'.symm
        simpa [e'] using this

theorem inv_reach {P : List (List (Op ι α))} {s : St ι α} (h : Reach (.ok (init P)) s) :
    ∀ c, s = .ok c → Inv P c := by
  induction h with
  | refl => intro c hc; injection hc with hc; subst hc; exact inv_init P
  | tail _ hst ih => intro c hc; subst hc; exact inv_step (ih _ rfl) hst

theorem rem_complete (k : ι) : ∀ {ths : List (Th ι α)}, (∀ t ∈ ths, t.todo = []) → rem k ths = bot
  | [], _ => rfl
  | t :: ts, h => by
    simp only [rem]
    rw [h t (List.mem_cons_self _ _),
      rem_complete k (fun t' ht' => h t' (List.mem_cons_of_mem _ ht'))]
    simp only [putsOf, bot_join]

theorem val_complete {P : List (List (Op ι α))} {c : Cfg ι α} (h : Inv P c) (kc : Complete c) :
    c.val = J P := by
  funext k
  have := h.sum k; rw [rem_complete k kc, join_bot] at this; exact this

theorem good_det {Jv : ι → α} (hJ : ∀ k, Jv k ≠ top) {f1 f2 : ι → Bool} :
    ∀ {e1 e2 : Op ι α × α}, Good Jv f1 e1 → Good Jv f2 e2 → e1.1 = e2.1 → e1 = e2
  | (.put _ _, _), (.put _ _, _), h1, h2, hop => by
      simp only [Good, Op.put.injEq] at h1 h2 hop ⊢; obtain ⟨rfl, rfl⟩ := hop; rw [h1, h2]
  | (.get _ _, _), (.get _ _, _), h1, h2, hop => by
      simp only [Op.get.injEq] at hop; obtain ⟨rfl, rfl⟩ := hop
      simp only [Good] at h1 h2
      rw [incomp_unique h1.1 (hJ _) h1.2.1 h2.2.1 h1.2.2 h2.2.2]
  | (.freeze _, _), (.freeze _, _), h1, h2, hop => by
      simp only [Op.freeze.injEq] at hop; subst hop
      simp only [Good] at h1 h2; rw [h1.2, h2.2]
  | (.put _ _, _), (.get _ _, _), _, _, hop => by simp at hop
  | (.put _ _, _), (.freeze _, _), _, _, hop => by simp at hop
  | (.get _ _, _), (.put _ _, _), _, _, hop => by simp at hop
  | (.get _ _, _), (.freeze _, _), _, _, hop => by simp at hop
  | (.freeze _, _), (.put _ _, _), _, _, hop => by simp at hop
  | (.freeze _, _), (.get _ _, _), _, _, hop => by simp at hop

theorem list_det {G : Op ι α × α → Prop} (hG : ∀ e1 e2, G e1 → G e2 → e1.1 = e2.1 → e1 = e2) :
    ∀ (l1 l2 : List (Op ι α × α)), l1.map Prod.fst = l2.map Prod.fst →
      (∀ e ∈ l1, G e) → (∀ e ∈ l2, G e) → l1 = l2
  | [], [], _, _, _ => rfl
  | [], _ :: _, h, _, _ => by simp at h
  | _ :: _, [], h, _, _ => by simp at h
  | a :: as, b :: bs, h, g1, g2 => by
      simp only [List.map_cons, List.cons.injEq] at h
      rw [hG a b (g1 a (List.mem_cons_self _ _)) (g2 b (List.mem_cons_self _ _)) h.1,
        list_det hG as bs h.2 (fun e he => g1 e (List.mem_cons_of_mem _ he))
          (fun e he => g2 e (List.mem_cons_of_mem _ he))]

theorem ths_det {G : Op ι α × α → Prop} (hG : ∀ e1 e2, G e1 → G e2 → e1.1 = e2.1 → e1 = e2) :
    ∀ (t1 t2 : List (Th ι α)), prog t1 = prog t2 →
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

/-- **Quasi-determinism for a store of LVars.** -/
theorem store_quasi_det {P : List (List (Op ι α))} {c1 c2 : Cfg ι α}
    (h1 : Reach (.ok (init P)) (.ok c1)) (h2 : Reach (.ok (init P)) (.ok c2))
    (k1 : Complete c1) (k2 : Complete c2) : c1 = c2 := by
  have i1 := inv_reach h1 c1 rfl
  have i2 := inv_reach h2 c2 rfl
  have v1 := val_complete i1 k1
  have v2 := val_complete i2 k2
  have hJ : ∀ k, J P k ≠ top := by intro k; rw [← v1]; exact i1.ntop k
  have hths : c1.ths = c2.ths :=
    ths_det (G := fun e => ∃ f, Good (J P) f e)
      (fun _ _ ⟨_, g1⟩ ⟨_, g2⟩ hop => good_det hJ g1 g2 hop) c1.ths c2.ths
      (by rw [i1.shape, i2.shape]) k1 k2
      (fun t ht e he => ⟨c1.frz, by rw [← v1]; exact i1.good t ht e he⟩)
      (fun t ht e he => ⟨c2.frz, by rw [← v2]; exact i2.good t ht e he⟩)
  have hf : c1.frz = c2.frz := by funext k; rw [i1.frz, i2.frz, hths]
  obtain ⟨a1, f1, t1⟩ := c1
  obtain ⟨a2, f2, t2⟩ := c2
  simp only at v1 v2 hths hf
  subst v1; subst v2; subst hf; subst hths; rfl

theorem anyFrz_false (k : ι) : ∀ {l : List (Op ι α × α)}, (∀ e ∈ l, isFrz e.1 = false) →
    anyFrz k l = false
  | [], _ => rfl
  | e :: es, hl => by
    have h0 : frzAt k e.1 = false := by
      have := hl e (List.mem_cons_self _ _)
      revert this; cases e.1 <;> simp [isFrz, frzAt]
    simp only [anyFrz, h0, anyFrz_false k (fun x hx => hl x (List.mem_cons_of_mem _ hx)),
      Bool.or_self]

theorem frozenDone_false (k : ι) : ∀ {ths : List (Th ι α)},
    (∀ t ∈ ths, ∀ e ∈ t.done, isFrz e.1 = false) → frozenDone k ths = false
  | [], _ => rfl
  | t :: ts, h => by
    simp only [frozenDone, anyFrz_false k (h t (List.mem_cons_self _ _)),
      frozenDone_false k (fun t' ht' => h t' (List.mem_cons_of_mem _ ht')), Bool.or_self]

/-- **Determinism without freeze, for a store.** -/
theorem store_det_no_err {P : List (List (Op ι α))} (hP : FreezeFree P) {c : Cfg ι α}
    (h1 : Reach (.ok (init P)) (.ok c)) (kc : Complete c) : ¬ Reach (.ok (init P)) .err := by
  intro he
  have ic := inv_reach h1 c rfl
  have hJ : ∀ k, J P k ≠ top := by intro k; rw [← val_complete ic kc]; exact ic.ntop k
  cases he with
  | tail hr hs =>
    rename_i c0
    have i0 := inv_reach hr c0 rfl
    cases hs with
    | @put_top v f pre post dn l d tl htop =>
      have hd : Le d (rem l (pre ++ ⟨dn, .put l d :: tl⟩ :: post)) := by
        simp only [rem_append, rem, putsOf, if_pos]
        exact le_trans' (le_trans' (le_join_left d _) (le_join_left _ _)) (le_join_right _ _)
      have := join_mono (v := v l) hd
      rw [htop, i0.sum l] at this
      exact hJ l (top_le this)
    | @put_frozen v f pre post dn l d tl hfl _ =>
      have hf := i0.frz l
      simp only at hf
      rw [frozenDone_false l, hfl] at hf
      · exact Bool.noConfusion hf
      · intro t ht e he
        have hm : t.done.map Prod.fst ++ t.todo ∈ P := by
          rw [← i0.shape]; exact List.mem_map.2 ⟨t, ht, rfl⟩
        exact hP _ hm e.1 (List.mem_append_left _ (List.mem_map.2 ⟨e, he, rfl⟩))

end RCE.LVarStore
