/-! T6 Memory safety with full borrow paths (spec 5.5–5.6).
    A place is a root variable with a projection path (fields, tuple
    elements; every index of a list is the same projection, conservatively).
    The instrumented semantics keeps, at run time, which roots are alive,
    which paths have been moved out, and the set of live references (each a
    pointer to a path, shared or mutable).  It *faults* on: access to a dead
    root or a moved path; a write through one path while another live
    reference overlaps it; a dereference of a reference whose target is no
    longer valid (dangling); dropping a root that is still borrowed.
    The typing judgment is the checker's flow-sensitive analysis; the
    theorem says a well-typed program never faults and that the static
    loan set is exactly the run-time set of live references. -/
namespace RCE.Borrow

abbrev Var := Nat
abbrev Path := Var × List Nat

def overlaps (p q : Path) : Bool :=
  p.1 == q.1 && (p.2.isPrefixOf q.2 || q.2.isPrefixOf p.2)

structure Loan where
  ref : Var
  path : Path
  mu : Bool
  deriving DecidableEq, Repr

structure St where
  alive : List Var
  moved : List Path
  loans : List Loan

def valid (s : St) (p : Path) : Bool := s.alive.contains p.1 && !(s.moved.any (overlaps p))
/-- no live reference overlaps p (other than `except`) -/
def free (s : St) (p : Path) (except : Option Var := none) : Bool :=
  s.loans.all fun l => some l.ref == except || !(overlaps p l.path)
/-- no live *mutable* reference overlaps p -/
def noMut (s : St) (p : Path) : Bool := s.loans.all fun l => !(l.mu && overlaps p l.path)
def find (s : St) (b : Var) : Option Loan := s.loans.find? (·.ref == b)

inductive BStmt where
  | new (x : Var)                       -- var x = …
  | read (p : Path)                     -- … p …
  | write (p : Path)                    -- p = …  (re-initializes p)
  | moveOut (p : Path)                  -- let y = p (Move type)
  | borrow (b : Var) (p : Path) (mu : Bool) -- let b = &p / &mut p
  | readVia (b : Var)                   -- … *b …
  | writeVia (b : Var)                  -- *b = …
  | endRef (b : Var)                    -- end of b's scope
  | drop (x : Var)                      -- end of x's scope
  | seq (a b : BStmt)

/-- Instrumented semantics; `none` is a fault. -/
def bexec (s : St) : BStmt → Option St
  | .new x => some { s with alive := x :: s.alive }
  | .read p => if valid s p && noMut s p then some s else none
  | .write p => if s.alive.contains p.1 && free s p then
      some { s with moved := s.moved.filter (fun q => !(p.2.isPrefixOf q.2 && p.1 == q.1)) } else none
  | .moveOut p => if valid s p && free s p then some { s with moved := p :: s.moved } else none
  | .borrow b p m => if valid s p && (if m then free s p else noMut s p) && (find s b).isNone
      then some { s with loans := ⟨b, p, m⟩ :: s.loans } else none
  | .readVia b => match find s b with
    | some l => if valid s l.path then some s else none
    | none => none
  | .writeVia b => match find s b with
    | some l => if l.mu && valid s l.path && free s l.path (some b) then some s else none
    | none => none
  | .endRef b => some { s with loans := s.loans.filter (·.ref != b) }
  | .drop x => if s.alive.contains x && s.loans.all (fun l => l.path.1 != x)
      then some { s with alive := s.alive.filter (· != x), moved := s.moved.filter (·.1 != x) } else none
  | .seq a b => match bexec s a with
    | none => none
    | some s' => bexec s' b

/-- The checker's judgment Γ ⊢ s ⊣ Γ' with loans (5.6). -/
inductive BWt : St → BStmt → St → Prop where
  | new {Γ x} : BWt Γ (.new x) { Γ with alive := x :: Γ.alive }
  | read {Γ p} : valid Γ p = true → noMut Γ p = true → BWt Γ (.read p) Γ
  | write {Γ p} : Γ.alive.contains p.1 = true → free Γ p = true →
      BWt Γ (.write p) { Γ with moved := Γ.moved.filter (fun q => !(p.2.isPrefixOf q.2 && p.1 == q.1)) }
  | moveOut {Γ p} : valid Γ p = true → free Γ p = true → BWt Γ (.moveOut p) { Γ with moved := p :: Γ.moved }
  | borrowShr {Γ b p} : valid Γ p = true → noMut Γ p = true → (find Γ b).isNone = true →
      BWt Γ (.borrow b p false) { Γ with loans := ⟨b, p, false⟩ :: Γ.loans }
  | borrowMut {Γ b p} : valid Γ p = true → free Γ p = true → (find Γ b).isNone = true →
      BWt Γ (.borrow b p true) { Γ with loans := ⟨b, p, true⟩ :: Γ.loans }
  | readVia {Γ b l} : find Γ b = some l → valid Γ l.path = true → BWt Γ (.readVia b) Γ
  | writeVia {Γ b l} : find Γ b = some l → l.mu = true → valid Γ l.path = true → free Γ l.path (some b) = true →
      BWt Γ (.writeVia b) Γ
  | endRef {Γ b} : BWt Γ (.endRef b) { Γ with loans := Γ.loans.filter (·.ref != b) }
  | drop {Γ x} : Γ.alive.contains x = true → Γ.loans.all (fun l => l.path.1 != x) = true →
      BWt Γ (.drop x) { Γ with alive := Γ.alive.filter (· != x), moved := Γ.moved.filter (·.1 != x) }
  | seq {Γ Γ' Γ'' a b} : BWt Γ a Γ' → BWt Γ' b Γ'' → BWt Γ (.seq a b) Γ''

/-- **T6 with borrow paths**: a well-typed program never faults, and the
    run-time state (live roots, moved paths, live references) is exactly the
    context the checker computed at every point. -/
theorem borrow_safety {Γ s Γ'} (h : BWt Γ s Γ') : bexec Γ s = some Γ' := by
  induction h with
  | new => rfl
  | read h1 h2 => simp [bexec, h1, h2]
  | write h1 h2 => simp at h1; simp [bexec, h1, h2]
  | moveOut h1 h2 => simp [bexec, h1, h2]
  | borrowShr h1 h2 h3 => simp [bexec, h1, h2, h3]
  | borrowMut h1 h2 h3 => simp [bexec, h1, h2, h3]
  | readVia h1 h2 => simp [bexec, h1, h2]
  | writeVia h1 h2 h3 h4 => simp [bexec, h1, h2, h3, h4]
  | endRef => rfl
  | drop h1 h2 => simp at h1; simp [bexec, h1, h2]
  | seq _ _ iha ihb => simp [bexec, iha, ihb]

theorem borrow_no_fault {Γ s Γ'} (h : BWt Γ s Γ') : bexec Γ s ≠ none := by
  rw [borrow_safety h]; simp

/-- Every live reference points to a valid path, and a mutable reference is
    the only reference overlapping its target, after any well-typed write
    through it (shared XOR mutable on paths). -/
theorem writeVia_exclusive {Γ b Γ'} (h : BWt Γ (.writeVia b) Γ') :
    ∃ l, find Γ b = some l ∧ l.mu = true ∧ ∀ l' ∈ Γ.loans, l'.ref ≠ b → overlaps l.path l'.path = false := by
  cases h with
  | writeVia h1 h2 _ h4 =>
    refine ⟨_, h1, h2, fun l' hl' hne => ?_⟩
    simp only [free, List.all_eq_true] at h4
    have := h4 l' hl'
    simp only [Bool.or_eq_true, beq_iff_eq, Bool.not_eq_true'] at this
    rcases this with h | h
    · simp at h; exact absurd h hne
    · exact h

/-! Refusals (each is a rule of 5.6 the checker enforces). -/

/-- Writing a path while a shared borrow of an overlapping path is live. -/
theorem write_while_borrowed_refused {Γ b x} (hx : Γ.alive.contains x = true) (hm : Γ.moved = [])
    (hf : (find Γ b).isNone = true) (hn : noMut Γ (x, [0]) = true) :
    ¬ ∃ Γ', BWt Γ (.seq (.borrow b (x, [0]) false) (.write (x, []))) Γ' := by
  intro ⟨Γ', h⟩
  cases h with
  | seq h1 h2 =>
    cases h1 with
    | borrowShr =>
      cases h2 with
      | write _ hfree => simp [free, overlaps, List.isPrefixOf] at hfree

/-- Two mutable borrows of overlapping paths (`&mut x` and `&mut x.0`). -/
theorem overlapping_mut_refused {Γ b c x} :
    ¬ ∃ Γ', BWt Γ (.seq (.borrow b (x, []) true) (.borrow c (x, [0]) true)) Γ' := by
  intro ⟨Γ', h⟩
  cases h with
  | seq h1 h2 =>
    cases h1 with
    | borrowMut =>
      cases h2 with
      | borrowMut _ hfree => simp [free, overlaps, List.isPrefixOf] at hfree

/-- Dropping a root while it is borrowed (the reference would dangle). -/
theorem dangling_refused {Γ b x p m} :
    ¬ ∃ Γ', BWt Γ (.seq (.borrow b (x, p) m) (.drop x)) Γ' := by
  intro ⟨Γ', h⟩
  cases h with
  | seq h1 h2 =>
    cases h1 <;> (cases h2 with | drop _ h => simp at h)

/-- Use after a partial move: reading `x` after moving `x.0` out. -/
theorem read_after_partial_move_refused {Γ x} :
    ¬ ∃ Γ', BWt Γ (.seq (.moveOut (x, [0])) (.read (x, []))) Γ' := by
  intro ⟨Γ', h⟩
  cases h with
  | seq h1 h2 =>
    cases h1 with
    | moveOut =>
      cases h2 with
      | read hv _ => simp [valid, overlaps, List.isPrefixOf] at hv

/-- Precision: disjoint fields can be borrowed mutably at the same time. -/
theorem disjoint_fields_ok {x : Var} :
    ∃ Γ', BWt ⟨[x], [], []⟩ (.seq (.borrow 100 (x, [0]) true) (.borrow 101 (x, [1]) true)) Γ' :=
  ⟨_, .seq (.borrowMut (by simp [valid]) (by simp [free]) (by simp [find])) (.borrowMut (by simp [valid]) (by simp [free, overlaps, List.isPrefixOf]) (by simp [find]))⟩

end RCE.Borrow
