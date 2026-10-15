/-
W-CRDT-1 certificates (C.11.5). A user `impl Merge` is admitted when the
checker derives a certificate `Cert` for its `merge` body: the body is read
as a term built from proven combinators, and `Cert.semi` gives the laws
(associativity, commutativity, idempotence) of every such term by
construction. No sampling is involved: soundness is `Cert.semi`, a total
function from certificates to proofs.

Combinators:
* `maxI`: integers under `max`; `minI`: integers under `min` (the dual
  order);
* `orB`, `andB`: booleans under `or` and `and`;
* `prod a b`: componentwise (records are right-nested products, one
  certificate per field);
* `lex b`: lexicographic on `(Int, β)`: the larger key wins and equal keys
  merge the payload with `b`. The proved side condition is that the key is
  a chain (a total order); a lexicographic join with a non-chain key is not
  associative in general;
* `map v`: pointwise over keys (missing key is the bottom of the option);
* `union`, `inter`: sets as membership predicates;
* `self`: an already certified `impl Merge` of a field type, by reference.

Bottom is not part of the certificate: a user `bottom()` is any element,
and the reachable states of an LVar are the up-set of its bottom, which is
closed under the join, so the laws on the carrier are the laws on the
up-set (compare `MergeLaws.intMax`).
-/
namespace RCE.MergeCert

set_option linter.unusedVariables false

/-- associative, commutative, idempotent -/
structure Semi (α : Type) where
  join : α → α → α
  assoc : ∀ a b c, join (join a b) c = join a (join b c)
  comm : ∀ a b, join a b = join b a
  idem : ∀ a, join a a = a

def maxS : Semi Int where
  join a b := max a b
  assoc a b c := by simp only [Int.max_def]; split <;> split <;> (try split) <;> (try split) <;> omega
  comm a b := by simp only [Int.max_def]; split <;> split <;> omega
  idem a := by simp [Int.max_def]

def minS : Semi Int where
  join a b := min a b
  assoc a b c := by simp only [Int.min_def]; split <;> split <;> (try split) <;> (try split) <;> omega
  comm a b := by simp only [Int.min_def]; split <;> split <;> omega
  idem a := by simp [Int.min_def]

def orS : Semi Bool := ⟨(· || ·), Bool.or_assoc, Bool.or_comm, Bool.or_self⟩
def andS : Semi Bool := ⟨(· && ·), Bool.and_assoc, Bool.and_comm, Bool.and_self⟩

def prodS {α β : Type} (s : Semi α) (t : Semi β) : Semi (α × β) where
  join p q := (s.join p.1 q.1, t.join p.2 q.2)
  assoc a b c := by simp [s.assoc, t.assoc]
  comm a b := by simp [s.comm, t.comm]
  idem a := by simp [s.idem, t.idem]

def lexJ {β : Type} (t : Semi β) (p q : Int × β) : Int × β :=
  if p.1 < q.1 then q else if q.1 < p.1 then p else (p.1, t.join p.2 q.2)

def lexS {β : Type} (t : Semi β) : Semi (Int × β) where
  join := lexJ t
  assoc a b c := by
    obtain ⟨a1, a2⟩ := a; obtain ⟨b1, b2⟩ := b; obtain ⟨c1, c2⟩ := c
    simp only [lexJ]
    by_cases h1 : a1 < b1 <;> by_cases h2 : b1 < a1 <;> by_cases h3 : b1 < c1 <;>
      by_cases h4 : c1 < b1 <;> by_cases h5 : a1 < c1 <;> by_cases h6 : c1 < a1 <;>
      simp [h1, h2, h3, h4, h5, h6, t.assoc] <;> first | omega | (have : a1 = b1 := by omega); subst this; simp_all [t.assoc] | skip
  comm a b := by
    obtain ⟨a1, a2⟩ := a; obtain ⟨b1, b2⟩ := b
    simp only [lexJ]
    by_cases h1 : a1 < b1 <;> by_cases h2 : b1 < a1 <;> simp [h1, h2] <;>
      first | omega | (have : a1 = b1 := by omega); subst this; simp [t.comm]
  idem a := by obtain ⟨a1, a2⟩ := a; simp [lexJ, t.idem]

def optJ {α : Type} (s : Semi α) : Option α → Option α → Option α
  | none, y => y
  | x, none => x
  | some x, some y => some (s.join x y)

theorem optJ_assoc {α : Type} (s : Semi α) (a b c : Option α) :
    optJ s (optJ s a b) c = optJ s a (optJ s b c) := by
  cases a <;> cases b <;> cases c <;> simp [optJ, s.assoc]

theorem optJ_comm {α : Type} (s : Semi α) (a b : Option α) : optJ s a b = optJ s b a := by
  cases a <;> cases b <;> simp [optJ, s.comm]

theorem optJ_idem {α : Type} (s : Semi α) (a : Option α) : optJ s a a = a := by
  cases a <;> simp [optJ, s.idem]

def mapS {K V : Type} (s : Semi V) : Semi (K → Option V) where
  join f g k := optJ s (f k) (g k)
  assoc f g h := funext fun k => optJ_assoc s (f k) (g k) (h k)
  comm f g := funext fun k => optJ_comm s (f k) (g k)
  idem f := funext fun k => optJ_idem s (f k)

def unionS (α : Type) : Semi (α → Bool) :=
  ⟨fun f g x => f x || g x, fun f g h => funext fun x => Bool.or_assoc _ _ _,
   fun f g => funext fun x => Bool.or_comm _ _, fun f => funext fun x => Bool.or_self _⟩

def interS (α : Type) : Semi (α → Bool) :=
  ⟨fun f g x => f x && g x, fun f g h => funext fun x => Bool.and_assoc _ _ _,
   fun f g => funext fun x => Bool.and_comm _ _, fun f => funext fun x => Bool.and_self _⟩

/-- the certificate the checker derives (checker/src/conc.rs `Cert`) -/
inductive Cert where
  | maxI | minI | orB | andB
  | prod : Cert → Cert → Cert
  | lex : Cert → Cert
  | map : Cert → Cert
  | union | inter
  | unit

/-- carrier of a certificate (keys and set elements are abstract, `Int`) -/
def Cert.ty : Cert → Type
  | .maxI | .minI => Int
  | .orB | .andB => Bool
  | .prod a b => a.ty × b.ty
  | .lex b => Int × b.ty
  | .map v => Int → Option v.ty
  | .union | .inter => Int → Bool
  | .unit => Unit

/-- **Soundness of certificates**: every certificate denotes a join with
the three laws. -/
def Cert.semi : (c : Cert) → Semi c.ty
  | .maxI => maxS
  | .minI => minS
  | .orB => orS
  | .andB => andS
  | .prod a b => prodS a.semi b.semi
  | .lex b => lexS b.semi
  | .map v => mapS v.semi
  | .union => unionS Int
  | .inter => interS Int
  | .unit => ⟨fun _ _ => (), fun _ _ _ => rfl, fun _ _ => rfl, fun _ => rfl⟩

theorem cert_laws (c : Cert) :
    (∀ a b d : c.ty, c.semi.join (c.semi.join a b) d = c.semi.join a (c.semi.join b d)) ∧
    (∀ a b : c.ty, c.semi.join a b = c.semi.join b a) ∧
    (∀ a : c.ty, c.semi.join a a = a) :=
  ⟨c.semi.assoc, c.semi.comm, c.semi.idem⟩

/-- the side condition of `lex` is needed: with a key order that is not a
chain (here: equality only, so any two distinct keys are incomparable and
the left one is kept) the lexicographic join is not commutative. -/
def badLex (p q : Int × Bool) : Int × Bool := if p.1 = q.1 then (p.1, p.2 || q.2) else p

theorem lex_needs_chain : badLex (0, false) (1, false) ≠ badLex (1, false) (0, false) := by
  decide

end RCE.MergeCert
