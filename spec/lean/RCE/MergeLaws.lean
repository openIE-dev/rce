/-
C.11.5 / W-CRDT-1: the Merge laws for the built-in instances, proved.

`SL α` is a join-semilattice (associative, commutative, idempotent join)
with an identity `bot`. The built-in `merge_builtin` instances of the
evaluator (conc.rs, `merge_builtin`) are modelled as:
* integers under `max` (`intMax`), with any chosen bottom `b` an identity
  on the values at or above `b` (`intMax_bot`),
* `bool` under `or` (`boolOr`),
* `none` as bottom over any instance (`optSL`),
* tuples and records componentwise (`prodSL`),
* maps pointwise over keys (`mapSL`, maps as functions to `Option`),
* sets and lists under union, as membership predicates (`setSL`).

`withTop` adds an absorbing top to any instance, which gives the lattice
`Lat` of LVarQD.lean for an LVar declared without a `top`. So T29q
(quasi-determinism) holds for every LVar whose type is built from these
instances.

Open: the evaluator keeps sets and maps as sorted lists without duplicates
(`canon_cmp`); that this representation is canonical, so that equality of
the predicates is equality of the values, is not proved here.
-/
import RCE.LVarQD

namespace RCE.MergeLaws

structure SL (α : Type) where
  join : α → α → α
  bot : α
  assoc : ∀ a b c, join (join a b) c = join a (join b c)
  comm : ∀ a b, join a b = join b a
  idem : ∀ a, join a a = a
  bot_join : ∀ a, join bot a = a

def boolOr : SL Bool where
  join := (· || ·)
  bot := false
  assoc := Bool.or_assoc
  comm := Bool.or_comm
  idem := Bool.or_self
  bot_join := Bool.false_or

/-- `max` on integers is associative, commutative and idempotent -/
theorem intMax_assoc (a b c : Int) : max (max a b) c = max a (max b c) := by
  simp only [Int.max_def]; split <;> split <;> (try split) <;> (try split) <;> omega

theorem intMax_comm (a b : Int) : max a b = max b a := by
  simp only [Int.max_def]; split <;> split <;> omega

theorem intMax_idem (a : Int) : max a a = a := by simp [Int.max_def]

/-- a chosen bottom `b` is an identity on the values at or above it -/
theorem intMax_bot (b a : Int) (h : b ≤ a) : max b a = a := by
  simp only [Int.max_def]; split <;> omega

/-- integers at or above a bottom `b` under `max` -/
def intMax (b : Int) : SL {a : Int // b ≤ a} where
  join x y := ⟨max x.1 y.1, by have := x.2; simp only [Int.max_def]; split <;> omega⟩
  bot := ⟨b, Int.le_refl b⟩
  assoc x y z := Subtype.ext (intMax_assoc x.1 y.1 z.1)
  comm x y := Subtype.ext (intMax_comm x.1 y.1)
  idem x := Subtype.ext (intMax_idem x.1)
  bot_join x := Subtype.ext (intMax_bot b x.1 x.2)

def optJoin {α : Type} (s : SL α) : Option α → Option α → Option α
  | none, y => y
  | x, none => x
  | some x, some y => some (s.join x y)

/-- `none` as bottom -/
def optSL {α : Type} (s : SL α) : SL (Option α) where
  join := optJoin s
  bot := none
  assoc a b c := by
    cases a <;> cases b <;> cases c <;> simp [optJoin, s.assoc]
  comm a b := by cases a <;> cases b <;> simp [optJoin, s.comm]
  idem a := by cases a <;> simp [optJoin, s.idem]
  bot_join a := by cases a <;> rfl

/-- componentwise, for tuples and records -/
def prodSL {α β : Type} (s : SL α) (t : SL β) : SL (α × β) where
  join p q := (s.join p.1 q.1, t.join p.2 q.2)
  bot := (s.bot, t.bot)
  assoc a b c := by simp [s.assoc, t.assoc]
  comm a b := by simp [s.comm, t.comm]
  idem a := by simp [s.idem, t.idem]
  bot_join a := by simp [s.bot_join, t.bot_join]

/-- maps: pointwise over keys, a missing key is `none` -/
def mapSL {K V : Type} (s : SL V) : SL (K → Option V) where
  join f g k := optJoin s (f k) (g k)
  bot _ := none
  assoc f g h := funext fun k => (optSL s).assoc (f k) (g k) (h k)
  comm f g := funext fun k => (optSL s).comm (f k) (g k)
  idem f := funext fun k => (optSL s).idem (f k)
  bot_join f := funext fun k => (optSL s).bot_join (f k)

/-- sets and lists under union, as membership predicates -/
def setSL (α : Type) : SL (α → Bool) where
  join f g x := f x || g x
  bot _ := false
  assoc f g h := funext fun x => Bool.or_assoc (f x) (g x) (h x)
  comm f g := funext fun x => Bool.or_comm (f x) (g x)
  idem f := funext fun x => Bool.or_self (f x)
  bot_join f := funext fun x => Bool.false_or (f x)

/-- an absorbing top over any instance: the lattice of an LVar declared
without `top` (no put reaches it) -/
def topJoin {α : Type} (s : SL α) : Option α → Option α → Option α
  | some x, some y => some (s.join x y)
  | _, _ => none

def withTop {α : Type} (s : SL α) : RCE.LVarQD.Lat (Option α) where
  join := topJoin s
  bot := some s.bot
  top := none
  join_assoc a b c := by cases a <;> cases b <;> cases c <;> simp [topJoin, s.assoc]
  join_comm a b := by cases a <;> cases b <;> simp [topJoin, s.comm]
  join_idem a := by cases a <;> simp [topJoin, s.idem]
  bot_join a := by cases a <;> simp [topJoin, s.bot_join]
  top_join a := by cases a <;> rfl
  bot_ne_top := by intro h; cases h

/-- a finite carrier: the laws are decided by enumeration, which is what the
checker does for a user `impl Merge` on a kind of nullary variants
(`crdt_gate`, exhaustive case). -/
theorem laws_of_enum {α : Type} (xs : List α) (hall : ∀ a, a ∈ xs) (j : α → α → α)
    (h : ∀ a ∈ xs, ∀ b ∈ xs, ∀ c ∈ xs, j (j a b) c = j a (j b c) ∧ j a b = j b a ∧ j a a = a) :
    (∀ a b c, j (j a b) c = j a (j b c)) ∧ (∀ a b, j a b = j b a) ∧ (∀ a, j a a = a) :=
  ⟨fun a b c => (h a (hall a) b (hall b) c (hall c)).1,
   fun a b => (h a (hall a) b (hall b) a (hall a)).2.1,
   fun a => (h a (hall a) a (hall a) a (hall a)).2.2⟩

end RCE.MergeLaws
