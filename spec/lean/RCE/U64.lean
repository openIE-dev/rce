/-
  Full-range u64 arithmetic (spec 6.3 N8, draft 0.5).
  Form ops `uadd usub umul udiv urem` are checked over [0, 2^64): an exact
  result in range is returned, anything else traps (`none`).
  Theorems: results stay in range, are exact whenever defined, a defined
  result is unique (so every target that implements the checked op agrees),
  and division satisfies the Euclidean identity.
-/
namespace RCE.U64

def M : Nat := 2 ^ 64

def inR (x : Nat) : Bool := decide (x < M)

def uadd (a b : Nat) : Option Nat := if a + b < M then some (a + b) else none
def usub (a b : Nat) : Option Nat := if b ≤ a then some (a - b) else none
def umul (a b : Nat) : Option Nat := if a * b < M then some (a * b) else none
def udiv (a b : Nat) : Option Nat := if b = 0 then none else some (a / b)
def urem (a b : Nat) : Option Nat := if b = 0 then none else some (a % b)

theorem uadd_spec (a b r : Nat) : uadd a b = some r ↔ r = a + b ∧ r < M := by
  unfold uadd; split <;> constructor <;> intro h <;> simp_all <;> omega

theorem usub_spec (a b r : Nat) (ha : a < M) : usub a b = some r → r = a - b ∧ r < M ∧ r + b = a := by
  unfold usub; split <;> intro h <;> simp_all <;> omega

theorem umul_spec (a b r : Nat) : umul a b = some r ↔ r = a * b ∧ r < M := by
  unfold umul; split <;> constructor <;> intro h <;> simp_all <;> omega

theorem udiv_urem (a b q r : Nat) (hq : udiv a b = some q) (hr : urem a b = some r) :
    b * q + r = a ∧ r < b := by
  unfold udiv at hq; unfold urem at hr
  split at hq
  · simp at hq
  · rename_i hb
    simp [hb] at hq hr; subst hq hr
    exact ⟨Nat.div_add_mod a b, Nat.mod_lt a (by omega)⟩

theorem udiv_range (a b q : Nat) (ha : a < M) (h : udiv a b = some q) : q < M := by
  unfold udiv at h; split at h
  · simp at h
  · simp at h; subst h; exact Nat.lt_of_le_of_lt (Nat.div_le_self a b) ha

theorem urem_range (a b r : Nat) (ha : a < M) (h : urem a b = some r) : r < M := by
  unfold urem at h; split at h
  · simp at h
  · simp at h; subst h; exact Nat.lt_of_le_of_lt (Nat.mod_le a b) ha

/-- The top of the range is reachable and one more traps. -/
example : uadd (M - 1) 0 = some (M - 1) := by decide
example : uadd (M - 1) 1 = none := by decide
example : usub 0 1 = none := by decide

end RCE.U64
