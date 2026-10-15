/-
T29a (C.11.8): quasi-determinism for a store of LVars with dynamic
allocation, and an executable semantics proved sound against the relation.

Extends LVarStore.lean (T29s) with `new l`, which allocates location `l`:
it steps only when `l` is not yet allocated, and `put`, `get` and `freeze`
on `l` step only after it is. Locations come from a deterministic counter:
`Counter P` says the k-th `new` of thread i allocates `(i, k)`, so the
location of a cell does not depend on the schedule (`counter_fresh`: no
location is allocated twice). Results:

* `alloc_quasi_det`: two runs that complete every thread without error
  reach the same store, frozen bits, allocation map and observations;
* `alloc_det_no_err`: without `freeze`, one complete run rules out error;
* `stepAt` is an executable step function (the thread to run is the
  schedule's choice) and `stepAt_sound` proves each of its steps is a
  `Step`, so the runs `exec` computes are runs of the relation. The vectors
  of LVarVectors.lean come from `exec` and are checked against the
  evaluator by checker/tests/lean_lvar.rs.
-/
import RCE.LVarQD

namespace RCE.LVarAlloc

open RCE.LVarQD RCE.LVarQD.Lat

set_option linter.unusedSectionVars false
set_option linter.constructorNameAsVariable false

variable {ι α : Type} [DecidableEq ι] [Lat α]

inductive Op (ι α : Type) where
  | put : ι → α → Op ι α
  | get : ι → List α → Op ι α
  | freeze : ι → Op ι α
  | new : ι → Op ι α

structure Th (ι α : Type) where
  done : List (Op ι α × α)
  todo : List (Op ι α)

structure Cfg (ι α : Type) where
  val : ι → α
  frz : ι → Bool
  alc : ι → Bool
  ths : List (Th ι α)

inductive St (ι α : Type) where
  | err : St ι α
  | ok : Cfg ι α → St ι α

def upd {β : Type} (v : ι → β) (l : ι) (x : β) : ι → β := fun k => if k = l then x else v k

inductive Step : Cfg ι α → St ι α → Prop where
  | put {v : ι → α} {f a : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {d : α} {tl : List (Op ι α)} :
      a l = true → join (v l) d ≠ top → (f l = true → join (v l) d = v l) →
      Step ⟨v, f, a, pre ++ ⟨dn, .put l d :: tl⟩ :: post⟩
        (.ok ⟨upd v l (join (v l) d), f, a, pre ++ ⟨dn ++ [(.put l d, d)], tl⟩ :: post⟩)
  | put_top {v : ι → α} {f a : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {d : α} {tl : List (Op ι α)} :
      a l = true → join (v l) d = top → Step ⟨v, f, a, pre ++ ⟨dn, .put l d :: tl⟩ :: post⟩ .err
  | put_frozen {v : ι → α} {f a : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {d : α} {tl : List (Op ι α)} :
      a l = true → f l = true → join (v l) d ≠ v l →
      Step ⟨v, f, a, pre ++ ⟨dn, .put l d :: tl⟩ :: post⟩ .err
  | get {v : ι → α} {f a : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {q : List α} {o : α} {tl : List (Op ι α)} :
      a l = true → Incomp q → o ∈ q → Le o (v l) →
      Step ⟨v, f, a, pre ++ ⟨dn, .get l q :: tl⟩ :: post⟩
        (.ok ⟨v, f, a, pre ++ ⟨dn ++ [(.get l q, o)], tl⟩ :: post⟩)
  | freeze {v : ι → α} {f a : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {tl : List (Op ι α)} :
      a l = true →
      Step ⟨v, f, a, pre ++ ⟨dn, .freeze l :: tl⟩ :: post⟩
        (.ok ⟨v, upd f l true, a, pre ++ ⟨dn ++ [(.freeze l, v l)], tl⟩ :: post⟩)
  | new {v : ι → α} {f a : ι → Bool} {pre post : List (Th ι α)} {dn : List (Op ι α × α)}
      {l : ι} {tl : List (Op ι α)} :
      a l = false →
      Step ⟨v, f, a, pre ++ ⟨dn, .new l :: tl⟩ :: post⟩
        (.ok ⟨v, f, upd a l true, pre ++ ⟨dn ++ [(.new l, bot)], tl⟩ :: post⟩)

inductive Reach (s : St ι α) : St ι α → Prop where
  | refl : Reach s s
  | tail {c : Cfg ι α} {t : St ι α} : Reach s (.ok c) → Step c t → Reach s t

def init (P : List (List (Op ι α))) : Cfg ι α :=
  ⟨fun _ => bot, fun _ => false, fun _ => false, P.map (fun ops => ⟨[], ops⟩)⟩

def Complete (c : Cfg ι α) : Prop := ∀ t ∈ c.ths, t.todo = []

def frzAt (k : ι) : Op ι α → Bool
  | .freeze l => decide (l = k)
  | _ => false

def newAt (k : ι) : Op ι α → Bool
  | .new l => decide (l = k)
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
  | .new _ :: tl => putsOf k tl

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

def anyNew (k : ι) : List (Op ι α × α) → Bool
  | [] => false
  | e :: es => newAt k e.1 || anyNew k es

def allocDone (k : ι) : List (Th ι α) → Bool
  | [] => false
  | t :: ts => anyNew k t.done || allocDone k ts

def Good (v : ι → α) (f : ι → Bool) : Op ι α × α → Prop
  | (.put _ d, o) => o = d
  | (.get l q, o) => Incomp q ∧ o ∈ q ∧ Le o (v l)
  | (.freeze l, o) => f l = true ∧ o = v l
  | (.new _, o) => o = bot

structure Inv (P : List (List (Op ι α))) (c : Cfg ι α) : Prop where
  sum : ∀ k, join (c.val k) (rem k c.ths) = J P k
  ntop : ∀ k, c.val k ≠ top
  shape : prog c.ths = P
  good : ∀ t ∈ c.ths, ∀ e ∈ t.done, Good c.val c.frz e
  frz : ∀ k, c.frz k = frozenDone k c.ths
  alc : ∀ k, c.alc k = allocDone k c.ths

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
  | (.new _, _), h => h
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

theorem anyNew_append (k : ι) (a b : List (Op ι α × α)) :
    anyNew k (a ++ b) = (anyNew k a || anyNew k b) := by
  induction a with
  | nil => simp [anyNew]
  | cons e es ih => simp [anyNew, ih, Bool.or_assoc]

theorem allocDone_append (k : ι) (a b : List (Th ι α)) :
    allocDone k (a ++ b) = (allocDone k a || allocDone k b) := by
  induction a with
  | nil => simp [allocDone]
  | cons e es ih => simp [allocDone, ih, Bool.or_assoc]

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
  alc k := by
    simp only [init]
    induction P with
    | nil => rfl
    | cons _ _ ih => simp only [List.map_cons, allocDone, anyNew]; exact ih

theorem inv_step {P : List (List (Op ι α))} {c c' : Cfg ι α} (h : Inv P c)
    (hs : Step c (.ok c')) : Inv P c' := by
  cases hs with
  | @put v f a pre post dn l d tl _ hnt hfz =>
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
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
    · intro k
      have := h.alc k
      simp only [allocDone_append, allocDone, anyNew_append, anyNew, newAt] at this ⊢
      simpa using this
  | @get v f a pre post dn l q o tl _ hq hm hl =>
    refine ⟨?_, h.ntop, ?_, ?_, ?_, ?_⟩
    · intro k; have hk := h.sum k; simp only [rem_append, rem, putsOf] at hk ⊢; exact hk
    · rw [prog_mid]; exact h.shape
    · exact good_mid h.good (good_mono (fun _ => le_refl' _) (fun _ hf => ⟨hf, rfl⟩)) ⟨hq, hm, hl⟩
    · intro k
      have := h.frz k
      simp only [frozenDone_append, frozenDone, anyFrz_append, anyFrz, frzAt] at this ⊢
      simpa using this
    · intro k
      have := h.alc k
      simp only [allocDone_append, allocDone, anyNew_append, anyNew, newAt] at this ⊢
      simpa using this
  | @freeze v f a pre post dn l tl _ =>
    refine ⟨?_, h.ntop, ?_, ?_, ?_, ?_⟩
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
    · intro k
      have := h.alc k
      simp only [allocDone_append, allocDone, anyNew_append, anyNew, newAt] at this ⊢
      simpa using this
  | @new v f a pre post dn l tl _ =>
    refine ⟨?_, h.ntop, ?_, ?_, ?_, ?_⟩
    · intro k; have hk := h.sum k; simp only [rem_append, rem, putsOf] at hk ⊢; exact hk
    · rw [prog_mid]; exact h.shape
    · exact good_mid h.good (good_mono (fun _ => le_refl' _) (fun _ hf => ⟨hf, rfl⟩)) rfl
    · intro k
      have := h.frz k
      simp only [frozenDone_append, frozenDone, anyFrz_append, anyFrz, frzAt] at this ⊢
      simpa using this
    · intro k
      have := h.alc k
      by_cases e : k = l
      · subst e
        simp [upd, allocDone_append, allocDone, anyNew_append, anyNew, newAt]
      · simp only [allocDone_append, allocDone, anyNew_append, anyNew, newAt] at this ⊢
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
  | (.new _, _), (.new _, _), h1, h2, hop => by
      simp only [Op.new.injEq] at hop; subst hop
      simp only [Good] at h1 h2; rw [h1, h2]
  | (.new _, _), (.put _ _, _), _, _, hop => by simp at hop
  | (.new _, _), (.get _ _, _), _, _, hop => by simp at hop
  | (.new _, _), (.freeze _, _), _, _, hop => by simp at hop
  | (.put _ _, _), (.new _, _), _, _, hop => by simp at hop
  | (.get _ _, _), (.new _, _), _, _, hop => by simp at hop
  | (.freeze _, _), (.new _, _), _, _, hop => by simp at hop
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
theorem alloc_quasi_det {P : List (List (Op ι α))} {c1 c2 : Cfg ι α}
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
  have ha : c1.alc = c2.alc := by funext k; rw [i1.alc, i2.alc, hths]
  obtain ⟨a1, f1, l1, t1⟩ := c1
  obtain ⟨a2, f2, l2, t2⟩ := c2
  simp only at v1 v2 hths hf ha
  subst v1; subst v2; subst hf; subst ha; subst hths; rfl

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
theorem alloc_det_no_err {P : List (List (Op ι α))} (hP : FreezeFree P) {c : Cfg ι α}
    (h1 : Reach (.ok (init P)) (.ok c)) (kc : Complete c) : ¬ Reach (.ok (init P)) .err := by
  intro he
  have ic := inv_reach h1 c rfl
  have hJ : ∀ k, J P k ≠ top := by intro k; rw [← val_complete ic kc]; exact ic.ntop k
  cases he with
  | tail hr hs =>
    rename_i c0
    have i0 := inv_reach hr c0 rfl
    cases hs with
    | @put_top v f a pre post dn l d tl _ htop =>
      have hd : Le d (rem l (pre ++ ⟨dn, .put l d :: tl⟩ :: post)) := by
        simp only [rem_append, rem, putsOf, if_pos]
        exact le_trans' (le_trans' (le_join_left d _) (le_join_left _ _)) (le_join_right _ _)
      have := join_mono (v := v l) hd
      rw [htop, i0.sum l] at this
      exact hJ l (top_le this)
    | @put_frozen v f a pre post dn l d tl _ hfl _ =>
      have hf := i0.frz l
      simp only at hf
      rw [frozenDone_false l, hfl] at hf
      · exact Bool.noConfusion hf
      · intro t ht e he
        have hm : t.done.map Prod.fst ++ t.todo ∈ P := by
          rw [← i0.shape]; exact List.mem_map.2 ⟨t, ht, rfl⟩
        exact hP _ hm e.1 (List.mem_append_left _ (List.mem_map.2 ⟨e, he, rfl⟩))

/-! Locations from a deterministic counter. -/

def newsOf : List (Op ι α) → List ι
  | [] => []
  | .new l :: tl => l :: newsOf tl
  | _ :: tl => newsOf tl

/-- the k-th `new` of thread i allocates `(i, k)` -/
def Counter (P : List (List (Op (Nat × Nat) α))) : Prop :=
  ∀ i, newsOf (P.getD i []) = (List.range (newsOf (P.getD i [])).length).map (fun k => (i, k))

/-- under the counter a location is allocated only by its own thread, at
its own index, whatever the schedule -/
theorem counter_owner {P : List (List (Op (Nat × Nat) α))} (h : Counter P) {i j k : Nat}
    (hm : (j, k) ∈ newsOf (P.getD i [])) : j = i ∧ k < (newsOf (P.getD i [])).length := by
  rw [h i] at hm
  simp only [List.mem_map, List.mem_range, Prod.mk.injEq] at hm
  obtain ⟨k', hk, rfl, rfl⟩ := hm
  refine ⟨rfl, ?_⟩
  have := congrArg List.length (h i)
  simp only [List.length_map, List.length_range] at this
  omega

/-! An executable semantics: `stepAt c i` runs thread `i` one step. -/

def pick : List (Th ι α) → Nat → Option (List (Th ι α) × Th ι α × List (Th ι α))
  | [], _ => none
  | t :: ts, 0 => some ([], t, ts)
  | t :: ts, n + 1 => (pick ts n).map (fun (p, x, q) => (t :: p, x, q))

theorem pick_spec : ∀ {ths : List (Th ι α)} {i : Nat} {pre t post},
    pick ths i = some (pre, t, post) → ths = pre ++ t :: post
  | [], _, _, _, _, h => by simp [pick] at h
  | t :: ts, 0, pre, x, post, h => by
      simp only [pick, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h; rfl
  | t :: ts, n + 1, pre, x, post, h => by
      simp only [pick, Option.map_eq_some'] at h
      obtain ⟨⟨p, y, q⟩, hp, he⟩ := h
      simp only [Prod.mk.injEq] at he
      obtain ⟨rfl, rfl, rfl⟩ := he
      rw [pick_spec hp]; rfl

section exec
variable [DecidableEq α]

def incompB (q : List α) : Bool :=
  q.all fun a => q.all fun b => decide (a = b) || decide (join a b = (top : α))

theorem incompB_sound {q : List α} (h : incompB q = true) : Incomp q := by
  intro a ha b hb hne
  simp only [incompB, List.all_eq_true, Bool.or_eq_true, decide_eq_true_eq] at h
  rcases h a ha b hb with e | e
  · exact absurd e hne
  · exact e

def stepOp (v : ι → α) (f a : ι → Bool) (pre : List (Th ι α)) (dn : List (Op ι α × α))
    (post : List (Th ι α)) : List (Op ι α) → Option (St ι α)
  | [] => none
  | .put l d :: tl =>
      if a l = true then
        if join (v l) d = top then some .err
        else if f l = true ∧ join (v l) d ≠ v l then some .err
        else some (.ok ⟨upd v l (join (v l) d), f, a, pre ++ ⟨dn ++ [(.put l d, d)], tl⟩ :: post⟩)
      else none
  | .get l q :: tl =>
      if a l = true ∧ incompB q = true then
        match q.find? (fun o => decide (join o (v l) = v l)) with
        | some o => some (.ok ⟨v, f, a, pre ++ ⟨dn ++ [(.get l q, o)], tl⟩ :: post⟩)
        | none => none
      else none
  | .freeze l :: tl =>
      if a l = true then some (.ok ⟨v, upd f l true, a, pre ++ ⟨dn ++ [(.freeze l, v l)], tl⟩ :: post⟩)
      else none
  | .new l :: tl =>
      if a l = false then some (.ok ⟨v, f, upd a l true, pre ++ ⟨dn ++ [(.new l, bot)], tl⟩ :: post⟩)
      else none

def stepAt (c : Cfg ι α) (i : Nat) : Option (St ι α) :=
  match pick c.ths i with
  | none => none
  | some (pre, t, post) => stepOp c.val c.frz c.alc pre t.done post t.todo

theorem stepOp_sound {v : ι → α} {f a : ι → Bool} {pre post : List (Th ι α)}
    {dn : List (Op ι α × α)} {ops : List (Op ι α)} {s : St ι α}
    (h : stepOp v f a pre dn post ops = some s) : Step ⟨v, f, a, pre ++ ⟨dn, ops⟩ :: post⟩ s := by
  match ops, h with
  | [], h => simp [stepOp] at h
  | .put l d :: tl, h =>
    simp only [stepOp] at h
    by_cases ha : a l = true
    · by_cases ht : join (v l) d = top
      · rw [if_pos ha, if_pos ht, Option.some.injEq] at h; subst h; exact Step.put_top ha ht
      · by_cases hf : f l = true ∧ join (v l) d ≠ v l
        · rw [if_pos ha, if_neg ht, if_pos hf, Option.some.injEq] at h; subst h
          exact Step.put_frozen ha hf.1 hf.2
        · rw [if_pos ha, if_neg ht, if_neg hf, Option.some.injEq] at h; subst h
          refine Step.put ha ht ?_
          intro hfl; apply Classical.byContradiction; intro hne; exact hf ⟨hfl, hne⟩
    · simp [ha] at h
  | .get l q :: tl, h =>
    simp only [stepOp] at h
    by_cases hc : a l = true ∧ incompB q = true
    · rw [if_pos hc] at h
      split at h
      · rename_i o hfind
        simp only [Option.some.injEq] at h; subst h
        have hm := List.mem_of_find?_eq_some hfind
        have hp := List.find?_some hfind
        simp only [decide_eq_true_eq] at hp
        exact Step.get hc.1 (incompB_sound hc.2) hm hp
      · simp at h
    · rw [if_neg hc] at h; simp at h
  | .freeze l :: tl, h =>
    simp only [stepOp] at h
    by_cases ha : a l = true
    · rw [if_pos ha, Option.some.injEq] at h; subst h; exact Step.freeze ha
    · simp [ha] at h
  | .new l :: tl, h =>
    simp only [stepOp] at h
    by_cases ha : a l = false
    · rw [if_pos ha, Option.some.injEq] at h; subst h; exact Step.new ha
    · simp [ha] at h

/-- **Soundness of the executable semantics**: every step `stepAt` takes
is a step of the relation. -/
theorem stepAt_sound {c : Cfg ι α} {i : Nat} {s : St ι α} (h : stepAt c i = some s) : Step c s := by
  obtain ⟨v, f, a, ths⟩ := c
  unfold stepAt at h
  split at h
  · simp at h
  · rename_i pre t post hp
    have e := pick_spec hp
    simp only at e
    subst e
    exact stepOp_sound h

/-- run a schedule: each entry names a thread; a blocked thread is skipped;
the threads actually stepped are returned with the final state -/
def exec : Cfg ι α → List Nat → St ι α × List Nat
  | c, [] => (.ok c, [])
  | c, i :: is => match stepAt c i with
    | none => exec c is
    | some .err => (.err, [i])
    | some (.ok c') => let r := exec c' is; (r.1, i :: r.2)

theorem exec_sound : ∀ (c : Cfg ι α) (sch : List Nat), Reach (.ok c) (exec c sch).1
  | c, [] => Reach.refl
  | c, i :: is => by
    unfold exec
    split
    · exact exec_sound c is
    · rename_i h; exact Reach.tail Reach.refl (stepAt_sound h)
    · rename_i c' h
      exact reach_trans (Reach.tail Reach.refl (stepAt_sound h)) (exec_sound c' is)
where
  reach_trans {s : St ι α} {c' : Cfg ι α} {t : St ι α} (h1 : Reach s (.ok c')) (h2 : Reach (.ok c') t) :
      Reach s t := by
    induction h2 with
    | refl => exact h1
    | tail _ hs ih => exact Reach.tail ih hs

end exec

end RCE.LVarAlloc
