/-!
  Schedule-side catch adapter, as a definition in this directory.

  `rce-form.md` §6 says a Form `catch` returns `sum(tup(outputs…), halt)`
  and ω. `spec/checker/src/one/ir.rs` types a core `catch` as
  `sum(ret_F, Halt)`. §6 also says the core catch catches every halt, the
  fuel trap included, and the schedule writes a Branch after it that halts
  again on the fuel trap. §13 reads that pair back as one Form `catch`.

  The definitions below are that adapter on values: pack and unpack by the
  declared output list, keep ω beside the sum, and rethrow the fuel trap.
  They do not change a core type. They are not extracted from
  `spec/checker/src/form/bridge.rs`, and no theorem in this file says they
  agree with that Rust. On the commit these proofs were written against,
  the bridge refuses catch.
-/
namespace RCE.FormBridge.Catch

/-- Working context carried beside a catch result. `fuel` and `spent` are
    numbers in this definition. They are not `fuel.tick` nodes. `outputs`
    is the list gathered before the halt or the return. -/
structure Order where
  fuel : Nat
  spent : Nat
  outputs : List Nat
  deriving DecidableEq, Repr

/-- A payload. A tuple is one payload, so a list of payloads can hold one
    tuple or several outputs. Equality of payloads is proved by cases.
    This nested inductive does not carry a derived `DecidableEq`. -/
inductive Payload where
  | num (n : Nat)
  | tup (vs : List Payload)

/-- A halt value. The fuel trap is the trap text the reference uses. -/
inductive HaltKind where
  | trap (text : String)
  | declared (name : String)
  deriving DecidableEq, Repr

def fuelTrap : String := "comptime fuel exhausted"

def isFuel : HaltKind → Bool
  | .trap t => decide (t = fuelTrap)
  | .declared _ => false

/-- The callee's outputs, with an explicit arity `vals.length`. -/
structure Outputs where
  vals : List Payload

def Outputs.arity (o : Outputs) : Nat := o.vals.length

/-- How the callee ended. ω is the context the halt consumed, or the
    context the return produced. It is not restored to an earlier value. -/
inductive CalleeEnd where
  | ret (o : Outputs) (ω : Order)
  | halt (h : HaltKind) (ω : Order)

/-- `sum(tup(outputs…), halt)` with ω beside the sum, not inside it. -/
inductive Packed where
  | values (vs : List Payload) (ω : Order)
  | halted (h : HaltKind) (ω : Order)

def pack : CalleeEnd → Packed
  | .ret o ω => .values o.vals ω
  | .halt h ω => .halted h ω

def unpack : Packed → CalleeEnd
  | .values vs ω => .ret ⟨vs⟩ ω
  | .halted h ω => .halt h ω

theorem unpack_pack (e : CalleeEnd) : unpack (pack e) = e := by
  cases e <;> rfl

theorem pack_unpack (p : Packed) : pack (unpack p) = p := by
  cases p <;> rfl

theorem pack_ret_keeps_arity (o : Outputs) (ω : Order) :
    pack (.ret o ω) = .values o.vals ω ∧ (o.vals).length = o.arity := by
  exact ⟨rfl, rfl⟩

/-- The schedule adapter. A return stays a value list. A fuel trap is
    rethrown. Any other halt is caught as a halt payload. -/
inductive Adapt where
  | returned (vs : List Payload) (ω : Order)
  | caught (h : HaltKind) (ω : Order)
  | rethrow (h : HaltKind) (ω : Order)

def adapt : CalleeEnd → Adapt
  | .ret o ω => .returned o.vals ω
  | .halt h ω => if isFuel h then .rethrow h ω else .caught h ω

theorem adapt_return (o : Outputs) (ω : Order) :
    adapt (.ret o ω) = .returned o.vals ω := rfl

theorem adapt_fuel (ω : Order) :
    adapt (.halt (.trap fuelTrap) ω) = .rethrow (.trap fuelTrap) ω := by
  simp [adapt, isFuel, fuelTrap]

theorem adapt_declared (name : String) (ω : Order) :
    adapt (.halt (.declared name) ω) = .caught (.declared name) ω := rfl

theorem adapt_loop_bound (ω : Order) :
    adapt (.halt (.trap "loop bound") ω) = .caught (.trap "loop bound") ω := by
  simp [adapt, isFuel, fuelTrap]

/-- Zero outputs, one output, one output that is itself a tuple, and two
    outputs stay four different lists. -/
theorem arity_zero (ω : Order) :
    adapt (.ret ⟨[]⟩ ω) = .returned [] ω ∧ ([] : List Payload).length = 0 :=
  ⟨rfl, rfl⟩

theorem arity_singleton (ω : Order) :
    adapt (.ret ⟨[.num 7]⟩ ω) = .returned [.num 7] ω ∧
      ([.num 7] : List Payload).length = 1 :=
  ⟨rfl, rfl⟩

theorem arity_tuple_singleton (ω : Order) :
    let one : List Payload := [.tup [.num 1, .num 2]]
    adapt (.ret ⟨one⟩ ω) = .returned one ω ∧ one.length = 1 := by
  exact ⟨rfl, rfl⟩

theorem arity_two (ω : Order) :
    adapt (.ret ⟨[.num 1, .num 2]⟩ ω) = .returned [.num 1, .num 2] ω ∧
      ([.num 1, .num 2] : List Payload).length = 2 :=
  ⟨rfl, rfl⟩

theorem tuple_singleton_ne_two_outputs (ω : Order) :
    adapt (.ret ⟨[.tup [.num 1, .num 2]]⟩ ω) ≠
      adapt (.ret ⟨[.num 1, .num 2]⟩ ω) := by
  intro h
  simp [adapt] at h

/-- Flattening one tuple payload into its fields drops a layer. The adapter
    does not do that. -/
def flatten : Payload → List Payload
  | .tup vs => vs
  | p => [p]

theorem no_silent_flatten (ω : Order) :
    adapt (.ret ⟨[.tup [.num 1, .num 2]]⟩ ω) =
      .returned [.tup [.num 1, .num 2]] ω ∧
    flatten (.tup [.num 1, .num 2]) = [.num 1, .num 2] ∧
    ([.tup [.num 1, .num 2]] : List Payload) ≠ [.num 1, .num 2] := by
  refine ⟨rfl, rfl, ?_⟩
  intro h
  cases h

/-- Core `catch` stays `sum(ret_F, Halt)`. `ret_F` is one value. Several
    outputs are that one value as a tuple. This is the schedule view of the
    value, not a second core type. -/
inductive CoreCatch where
  | ok (v : Payload)
  | halt (h : HaltKind)

def toCore : CalleeEnd → CoreCatch × Order
  | .ret o ω => (.ok (.tup o.vals), ω)
  | .halt h ω => (.halt h, ω)

theorem toCore_one_value (o : Outputs) (ω : Order) :
    toCore (.ret o ω) = (.ok (.tup o.vals), ω) := rfl

theorem one_tuple_result_ne_several_outputs (a b : Payload) (ω : Order) :
    toCore (.ret ⟨[.tup [a, b]]⟩ ω) ≠ toCore (.ret ⟨[a, b]⟩ ω) := by
  intro h
  simp [toCore] at h

/-- A map is bijective when it is injective and surjective. Defined here
    so the statement does not depend on a name outside this file. -/
def Injective {α β : Type} (f : α → β) : Prop :=
  ∀ a₁ a₂, f a₁ = f a₂ → a₁ = a₂

def Surjective {α β : Type} (f : α → β) : Prop :=
  ∀ b, ∃ a, f a = b

def Bijective {α β : Type} (f : α → β) : Prop :=
  Injective f ∧ Surjective f

/-- Swap 0 and 1 inside a raw numeric success value. Halt values stay,
    and a tuple success value stays. This is a bijection of the core sum.
    It moves `.ok (.num 0)`. It does not move the tuple that `toCore`
    builds for one numeric output, so the bijection is not the adapter. -/
def flipNum : Nat → Nat
  | 0 => 1
  | 1 => 0
  | n + 2 => n + 2

theorem flipNum_involutive : ∀ n, flipNum (flipNum n) = n
  | 0 => rfl
  | 1 => rfl
  | _ + 2 => rfl

def twist : CoreCatch → CoreCatch
  | .ok (.num n) => .ok (.num (flipNum n))
  | .ok (.tup vs) => .ok (.tup vs)
  | .halt h => .halt h

theorem twist_involutive (c : CoreCatch) : twist (twist c) = c := by
  cases c with
  | halt h => rfl
  | ok v =>
      cases v with
      | tup _ => rfl
      | num n =>
          show CoreCatch.ok (.num (flipNum (flipNum n))) = .ok (.num n)
          rw [flipNum_involutive]

theorem twist_bijective : Bijective twist := by
  refine ⟨?_, ?_⟩
  · intro a b h
    have h' := congrArg twist h
    rw [twist_involutive, twist_involutive] at h'
    exact h'
  · intro c
    exact ⟨twist c, twist_involutive c⟩

theorem bijection_is_not_evaluator :
    twist (.ok (.num 0)) ≠ .ok (.num 0) := by
  intro h
  simp [twist, flipNum] at h

theorem twist_leaves_packed_singleton (ω : Order) :
    twist (toCore (.ret ⟨[.num 0]⟩ ω)).1 =
      (toCore (.ret ⟨[.num 0]⟩ ω)).1 ∧
    (toCore (.ret ⟨[.num 0]⟩ ω)).1 = .ok (.tup [.num 0]) := by
  simp [twist, toCore]

/-- Run the adapter. The handler is applied only on `.caught`. It receives
    the halt payload and the ω the callee consumed. A return does not apply
    it. A rethrow does not apply it, and the ω of the rethrow is that same
    consumed ω. -/
def dispatch (e : CalleeEnd) (handler : HaltKind → Order → Outputs × Order) :
    Except (HaltKind × Order) (Outputs × Order) :=
  match adapt e with
  | .returned vs ω => .ok (⟨vs⟩, ω)
  | .caught h ω => .ok (handler h ω)
  | .rethrow h ω => .error (h, ω)

theorem return_skips_handler (o : Outputs) (ω : Order)
    (handler : HaltKind → Order → Outputs × Order) :
    dispatch (.ret o ω) handler = .ok (o, ω) := by
  cases o
  simp [dispatch, adapt]

theorem fuel_skips_handler (ω : Order)
    (handler : HaltKind → Order → Outputs × Order) :
    dispatch (.halt (.trap fuelTrap) ω) handler =
      .error (.trap fuelTrap, ω) := by
  simp [dispatch, adapt, isFuel, fuelTrap]

theorem caught_sees_halt_and_omega (name : String) (ω : Order)
    (handler : HaltKind → Order → Outputs × Order) :
    dispatch (.halt (.declared name) ω) handler =
      .ok (handler (.declared name) ω) := rfl

theorem uncaught_trap_is_not_a_declared_error (ω : Order) :
    adapt (.halt (.trap "loop bound") ω) ≠
      adapt (.halt (.declared "loop bound") ω) := by
  intro h
  simp [adapt, isFuel, fuelTrap] at h

/-- The fuel trap does not come back with a fresh order value. -/
theorem fuel_does_not_restore_omega :
    let consumed : Order := ⟨4, 2, [7]⟩
    let fresh : Order := ⟨0, 0, []⟩
    adapt (.halt (.trap fuelTrap) consumed) =
      .rethrow (.trap fuelTrap) consumed ∧
    adapt (.halt (.trap fuelTrap) consumed) ≠
      .rethrow (.trap fuelTrap) fresh := by
  refine ⟨?_, ?_⟩
  · simp [adapt, isFuel, fuelTrap]
  · intro h
    simp [adapt, isFuel, fuelTrap] at h

/-- Storing ω's fuel inside the value list is a different term from the
    adapter, which keeps ω last, beside the sum. -/
def buryOmega (vs : List Payload) (ω : Order) : List Payload :=
  vs ++ [.num ω.fuel]

theorem omega_stays_beside_the_sum :
    let ω : Order := ⟨9, 1, [4]⟩
    adapt (.ret ⟨[.num 1]⟩ ω) = .returned [.num 1] ω ∧
    buryOmega [.num 1] ω = [.num 1, .num 9] ∧
    buryOmega [.num 1] ω ≠ [.num 1] := by
  refine ⟨rfl, rfl, ?_⟩
  intro h
  cases h

end RCE.FormBridge.Catch
