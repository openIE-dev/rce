/-
The canonical order of runtime values is a strict total order (7.1).

`Val` models the runtime values that `cmp_canon` sorts (set elements, map
keys, `select` candidates) as the Form values of 7.1 (`FV` in interp.rs):
null, bool, integers, floats, text, lists and records. `cmp` is the order
the platform worker fixed for `FV::cmp_canon` (logged 2026-10-15 14:42 in
the platform progress log, checked against interp.rs):

* rank first: null 0 < bool 1 < int 2 < float 3 < text 4 < list 5 < record 6
  (tuples and arrays embed as lists, maps and structs as records);
* bool: false < true; int: by mathematical value;
* float: IEEE 754 totalOrder (`f64::total_cmp`), through the key of its
  64-bit pattern (`fkey`: b when the sign bit is clear, -(b - 2^63) - 1 when
  it is set, compared as integers), so -0 < +0 and an int is never equal
  to a float;
* text: its Unicode scalar values, lexicographically;
* list: elements lexicographically, a proper prefix first;
* record: its sorted entries, each modelled as the list `[text key, value]`,
  lexicographically (key, then value), fewer entries first.

Theorems: `cmp a b = .eq ↔ a = b` (`cmp_eq_iff`), `cmp b a = (cmp a b).swap`
(`cmp_swap`), and `lt` (`cmp a b = .lt`) is transitive (`cmp_trans`), so `lt`
is a strict total order (`lt_irrefl`, `lt_asymm`, `lt_trans`, `lt_total`).
`set_canon_val` and `map_canon_val` instantiate `Canon` at these values:
a set of values, and a map with value keys, have one sorted representation.
-/
import RCE.Canon

namespace RCE.ValOrder

inductive Val where
  | unit
  | bool (b : Bool)
  | int (i : Int)
  | flt (bits : Nat)
  | text (bytes : List Nat)
  | list (xs : List Val)
  | record (entries : List Val)
  deriving Inhabited

def tag : Val → Nat
  | .unit => 0 | .bool _ => 1 | .int _ => 2 | .flt _ => 3 | .text _ => 4
  | .list _ => 5 | .record _ => 6

def icmp (a b : Int) : Ordering := if a < b then .lt else if a = b then .eq else .gt

theorem icmp_eq_iff (a b : Int) : icmp a b = .eq ↔ a = b := by
  unfold icmp
  repeat' split
  all_goals first | (exact ⟨fun _ => by omega, fun _ => rfl⟩) | (exact ⟨fun h => (by cases h), fun _ => by omega⟩)
theorem icmp_swap (a b : Int) : icmp b a = (icmp a b).swap := by
  unfold icmp
  repeat' split
  all_goals first | rfl | (exfalso; omega)
theorem icmp_trans {a b c : Int} : icmp a b = .lt → icmp b c = .lt → icmp a c = .lt := by
  unfold icmp
  repeat' split
  all_goals first | (intros; rfl) | (intros; exfalso; omega) | (intro h; cases h; done) | (intro _ h; cases h; done)
theorem icmp_refl (a : Int) : icmp a a = .eq := (icmp_eq_iff a a).2 rfl

/-- IEEE 754 totalOrder key of a 64-bit pattern -/
def fkey (b : Nat) : Int := if b < 2 ^ 63 then (b : Int) else -((b : Int) - 2 ^ 63) - 1

theorem fkey_inj {a b : Nat} (h : fkey a = fkey b) : a = b := by
  unfold fkey at h; split at h <;> split at h <;> omega

/-- lexicographic order on byte lists -/
def lexN : List Nat → List Nat → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | a :: as, b :: bs => match icmp a b with
    | .eq => lexN as bs
    | o => o

theorem lexN_eq_iff : ∀ (x y : List Nat), lexN x y = .eq ↔ x = y
  | [], [] => by simp [lexN]
  | [], _ :: _ => by simp [lexN]
  | _ :: _, [] => by simp [lexN]
  | a :: as, b :: bs => by
    simp only [lexN, List.cons.injEq]
    cases e : icmp (↑a) (↑b) with
    | eq =>
      have : a = b := by have := (icmp_eq_iff _ _).1 e; omega
      subst this; simp [lexN_eq_iff as bs]
    | lt =>
      refine ⟨fun h => (by cases h), fun ⟨hab, _⟩ => ?_⟩
      subst hab; rw [icmp_refl] at e; cases e
    | gt =>
      refine ⟨fun h => (by cases h), fun ⟨hab, _⟩ => ?_⟩
      subst hab; rw [icmp_refl] at e; cases e

theorem lexN_swap : ∀ (x y : List Nat), lexN y x = (lexN x y).swap
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | a :: as, b :: bs => by
    simp only [lexN]
    rw [icmp_swap (↑a) (↑b)]
    cases icmp (↑a) (↑b) <;> simp [Ordering.swap, lexN_swap as bs]

theorem lexN_trans : ∀ {x y z : List Nat}, lexN x y = .lt → lexN y z = .lt → lexN x z = .lt
  | [], [], _, h, _ => by simp only [lexN] at h; cases h
  | [], _ :: _, [], _, h => by simp only [lexN] at h; cases h
  | [], _ :: _, _ :: _, _, _ => rfl
  | _ :: _, [], _, h, _ => by simp only [lexN] at h; cases h
  | _ :: _, _ :: _, [], _, h => by simp only [lexN] at h; cases h
  | a :: as, b :: bs, c :: cs, h1, h2 => by
    simp only [lexN] at *
    cases e1 : icmp (↑a) (↑b) <;> cases e2 : icmp (↑b) (↑c) <;> rw [e1] at h1 <;> rw [e2] at h2 <;>
      (try (cases h1; done)) <;> (try (cases h2; done))
    · rw [icmp_trans e1 e2]
    · have : b = c := by have := (icmp_eq_iff _ _).1 e2; omega
      subst_vars; rw [e1]
    · have : a = b := by have := (icmp_eq_iff _ _).1 e1; omega
      subst_vars; rw [e2]
    · have : a = b := by have := (icmp_eq_iff _ _).1 e1; omega
      have : b = c := by have := (icmp_eq_iff _ _).1 e2; omega
      subst_vars; rw [icmp_refl]; exact lexN_trans h1 h2

mutual
def cmp : Val → Val → Ordering
  | .unit, .unit => .eq
  | .bool x, .bool y => icmp (if x then 1 else 0) (if y then 1 else 0)
  | .int x, .int y => icmp x y
  | .flt x, .flt y => icmp (fkey x) (fkey y)
  | .text x, .text y => lexN x y
  | .list x, .list y => cmpL x y
  | .record x, .record y => cmpL x y
  | a, b => icmp (tag a) (tag b)
def cmpL : List Val → List Val → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | a :: as, b :: bs => match cmp a b with
    | .eq => cmpL as bs
    | o => o
end

theorem cmp_off {a b : Val} (h : tag a ≠ tag b) : cmp a b = icmp (tag a) (tag b) := by
  cases a <;> cases b <;> simp [tag] at h <;> simp [cmp, tag]

theorem tag_lt {a b : Val} (h : tag a ≠ tag b) : cmp a b ≠ .eq := by
  rw [cmp_off h]; intro e; exact h (by have := (icmp_eq_iff _ _).1 e; omega)

mutual
theorem cmp_eq_iff (a b : Val) : cmp a b = .eq ↔ a = b := by
  by_cases h : tag a = tag b
  · cases a <;> cases b <;> simp [tag] at h
    · simp [cmp]
    · rename_i x y; simp only [cmp, icmp_eq_iff, Val.bool.injEq]; cases x <;> cases y <;> simp
    · rename_i x y; simp only [cmp, icmp_eq_iff, Val.int.injEq]
    · rename_i x y
      simp only [cmp, icmp_eq_iff, Val.flt.injEq]; exact ⟨fkey_inj, fun h => h ▸ rfl⟩
    · rename_i x y; simp only [cmp, lexN_eq_iff, Val.text.injEq]
    · rename_i x y; simp only [cmp, Val.list.injEq]; exact cmpL_eq_iff x y
    · rename_i x y; simp only [cmp, Val.record.injEq]; exact cmpL_eq_iff x y
  · exact ⟨fun e => absurd e (tag_lt h), fun e => absurd (congrArg tag e) h⟩
theorem cmpL_eq_iff : ∀ (xs ys : List Val), cmpL xs ys = .eq ↔ xs = ys
  | [], [] => by simp [cmpL]
  | [], _ :: _ => by simp [cmpL]
  | _ :: _, [] => by simp [cmpL]
  | a :: as, b :: bs => by
    simp only [cmpL, List.cons.injEq]
    have ih := cmp_eq_iff a b
    cases e : cmp a b with
    | eq => simp [ih.1 e, cmpL_eq_iff as bs]
    | lt => exact ⟨fun h => (by cases h), fun ⟨hab, _⟩ => by rw [ih.2 hab] at e; cases e⟩
    | gt => exact ⟨fun h => (by cases h), fun ⟨hab, _⟩ => by rw [ih.2 hab] at e; cases e⟩
end

theorem cmp_refl (a : Val) : cmp a a = .eq := (cmp_eq_iff a a).2 rfl

mutual
theorem cmp_swap (a b : Val) : cmp b a = (cmp a b).swap := by
  by_cases h : tag a = tag b
  · cases a <;> cases b <;> simp [tag] at h
    · rfl
    · rename_i x y; simp only [cmp]; exact icmp_swap _ _
    · rename_i x y; simp only [cmp]; exact icmp_swap _ _
    · rename_i x y; simp only [cmp]; exact icmp_swap _ _
    · rename_i x y; simp only [cmp]; exact lexN_swap _ _
    · rename_i x y; simp only [cmp]; exact cmpL_swap x y
    · rename_i x y; simp only [cmp]; exact cmpL_swap x y
  · rw [cmp_off h, cmp_off (Ne.symm h)]; exact icmp_swap _ _
theorem cmpL_swap : ∀ (xs ys : List Val), cmpL ys xs = (cmpL xs ys).swap
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | a :: as, b :: bs => by
    simp only [cmpL]
    rw [cmp_swap a b]
    cases cmp a b <;> simp [Ordering.swap]
    exact cmpL_swap as bs
end

theorem tag_le {a b : Val} (h : cmp a b = .lt) : tag a ≤ tag b := by
  by_cases t : tag a = tag b
  · omega
  · rw [cmp_off t] at h; unfold icmp at h; split at h
    · omega
    · split at h <;> cases h

theorem off_lt {a c : Val} (h : tag a < tag c) : cmp a c = .lt := by
  rw [cmp_off (by omega)]; unfold icmp; simp; omega

mutual
theorem cmp_trans (a b c : Val) (h1 : cmp a b = .lt) (h2 : cmp b c = .lt) : cmp a c = .lt := by
  have l1 := tag_le h1; have l2 := tag_le h2
  by_cases t1 : tag a = tag b
  · by_cases t2 : tag b = tag c
    · cases a <;> cases b <;> simp [tag] at t1 <;> cases c <;> simp [tag] at t2
      · simp [cmp] at h1
      · simp only [cmp] at *; exact icmp_trans h1 h2
      · simp only [cmp] at *; exact icmp_trans h1 h2
      · simp only [cmp] at *; exact icmp_trans h1 h2
      · simp only [cmp] at *; exact lexN_trans h1 h2
      · rename_i x y z; simp only [cmp] at *; exact cmpL_trans x y z h1 h2
      · rename_i x y z; simp only [cmp] at *; exact cmpL_trans x y z h1 h2
    · exact off_lt (by omega)
  · exact off_lt (by omega)
theorem cmpL_trans : ∀ (xs ys zs : List Val), cmpL xs ys = .lt → cmpL ys zs = .lt → cmpL xs zs = .lt
  | [], [], _, h, _ => by simp only [cmpL] at h; cases h
  | [], _ :: _, [], _, h => by simp only [cmpL] at h; cases h
  | [], _ :: _, _ :: _, _, _ => rfl
  | _ :: _, [], _, h, _ => by simp only [cmpL] at h; cases h
  | _ :: _, _ :: _, [], _, h => by simp only [cmpL] at h; cases h
  | a :: as, b :: bs, c :: cs, h1, h2 => by
    simp only [cmpL] at *
    cases e1 : cmp a b <;> cases e2 : cmp b c <;> rw [e1] at h1 <;> rw [e2] at h2 <;>
      (try (cases h1; done)) <;> (try (cases h2; done))
    · rw [cmp_trans a b c e1 e2]
    · rw [(cmp_eq_iff b c).1 e2] at e1; rw [e1]
    · rw [← (cmp_eq_iff a b).1 e1] at e2; rw [e2]
    · rw [(cmp_eq_iff a b).1 e1, (cmp_eq_iff b c).1 e2, cmp_refl]
      exact cmpL_trans as bs cs h1 h2
end

/-- equality of runtime values is decided by `cmp` -/
instance : DecidableEq Val := fun a b => decidable_of_iff _ (cmp_eq_iff a b)

/-- the strict order the runtimes sort by -/
def lt (a b : Val) : Prop := cmp a b = .lt

theorem lt_irrefl (a : Val) : ¬ lt a a := by unfold lt; rw [cmp_refl]; intro h; cases h
theorem lt_asymm {a b : Val} (h : lt a b) : ¬ lt b a := by
  unfold lt at *; rw [cmp_swap, h]; intro e; cases e
theorem lt_trans {a b c : Val} : lt a b → lt b c → lt a c := cmp_trans a b c
theorem lt_total (a b : Val) : lt a b ∨ a = b ∨ lt b a := by
  unfold lt; rw [cmp_swap a b]
  cases e : cmp a b
  · exact Or.inl rfl
  · exact Or.inr (Or.inl ((cmp_eq_iff a b).1 e))
  · exact Or.inr (Or.inr rfl)

/-- a set of runtime values has one sorted representation -/
theorem set_canon_val {l1 l2 : List Val} (p1 : l1.Pairwise lt) (p2 : l2.Pairwise lt)
    (h : ∀ x, x ∈ l1 ↔ x ∈ l2) : l1 = l2 :=
  RCE.Canon.set_canon lt lt_irrefl (fun _ _ => lt_asymm) p1 p2 h

/-- a map with runtime-value keys has one key-sorted representation -/
theorem map_canon_val {β : Type} {l1 l2 : List (Val × β)}
    (p1 : l1.Pairwise (fun p q => lt p.1 q.1)) (p2 : l2.Pairwise (fun p q => lt p.1 q.1))
    (h : ∀ k, RCE.Canon.lookup k l1 = RCE.Canon.lookup k l2) : l1 = l2 :=
  RCE.Canon.map_canon lt lt_irrefl (fun _ _ => lt_asymm) p1 p2 h

end RCE.ValOrder
