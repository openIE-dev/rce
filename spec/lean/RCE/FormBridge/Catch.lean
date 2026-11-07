/-!
  Schedule-side catch adapter, as a definition in this directory.

  `rce-form.md` §6 says a Form `catch` returns `sum(tup(outputs…), halt)`
  and ω. `spec/checker/src/one/ir.rs` types a core `catch` as
  `sum(ret_F, Halt)`. §6 also says the core catch catches every halt, the
  fuel trap included, and the schedule writes a Branch after it that halts
  again on the fuel trap. §13 reads that pair back as one Form `catch`.

  Two maps are defined, and they are not the same map.

  `pack` and `adapt` are the Form declared list: `sum` of that list with a
  halt, ω beside the sum, and a fuel rethrow. That is the schedule story in
  `rce-form.md` §6. `coreCatch` is the core evaluator in
  `spec/checker/src/one/eval.rs`: variant 0 holds the callee's one return
  value unchanged, and every halt, the fuel trap included, is variant 1.
  `sig_of` in `bridge.rs` admits a core return only when the non-ω outputs
  number exactly one. Zero and several results are outside that slice.

  Neither map is extracted from `bridge.rs`. The bridge still refuses a
  catch. Pull request 153, which is not on this main, is not assumed here.
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

/-- The core sum `sum(ret_F, Halt)`. The success payload is `ret_F` itself. -/
inductive CoreCatch where
  | ok (v : Payload)
  | halt (h : HaltKind)

/-- What `eval.rs` sees from the callee. `.other` is not a halt and is not
    caught. -/
inductive CoreFlow where
  | returned (v : Payload)
  | halt (h : HaltKind)
  | other

/-- `P::Catch` in `eval.rs`: `Ok(v)` becomes `Sum(0, v)` and `Flow::Halt`
    becomes `Sum(1, h)`. The value `v` is not wrapped again. -/
def coreCatch : CoreFlow → Except Unit CoreCatch
  | .returned v => .ok (.ok v)
  | .halt h => .ok (.halt h)
  | .other => .error ()

/-- `sig_of` admits the core return only for exactly one non-ω result. -/
def coreSliceAdmits (nonOmegaResults : Nat) : Bool :=
  decide (nonOmegaResults = 1)

theorem core_slice_admits_one : coreSliceAdmits 1 = true := rfl

theorem core_slice_refuses_zero : coreSliceAdmits 0 = false := rfl

theorem core_slice_refuses_two : coreSliceAdmits 2 = false := rfl

theorem core_keeps_scalar :
    coreCatch (.returned (.num 7)) = .ok (.ok (.num 7)) := rfl

theorem core_keeps_tuple :
    coreCatch (.returned (.tup [.num 1, .num 2])) =
      .ok (.ok (.tup [.num 1, .num 2])) := rfl

theorem core_does_not_wrap_scalar :
    coreCatch (.returned (.num 7)) ≠ .ok (.ok (.tup [.num 7])) := by
  intro h
  cases h

theorem core_does_not_nest_tuple (a b : Payload) :
    coreCatch (.returned (.tup [a, b])) ≠
      .ok (.ok (.tup [.tup [a, b]])) := by
  intro h
  simp [coreCatch] at h

/-- A declared list of one scalar is the Form packing. The core payload of
    that scalar is the scalar. The inequality of a tuple-valued declared
    output with two declared outputs is `tuple_singleton_ne_two_outputs`,
    and it is a fact about those lists, not about `coreCatch`. -/
theorem form_list_is_not_core_payload (ω : Order) :
    adapt (.ret ⟨[.num 7]⟩ ω) = .returned [.num 7] ω ∧
    coreCatch (.returned (.num 7)) = .ok (.ok (.num 7)) :=
  ⟨rfl, rfl⟩

theorem core_catches_fuel :
    coreCatch (.halt (.trap fuelTrap)) = .ok (.halt (.trap fuelTrap)) := rfl

theorem schedule_rethrow_is_not_the_core_catch (ω : Order) :
    adapt (.halt (.trap fuelTrap) ω) = .rethrow (.trap fuelTrap) ω ∧
    coreCatch (.halt (.trap fuelTrap)) = .ok (.halt (.trap fuelTrap)) := by
  refine ⟨?_, rfl⟩
  simp [adapt, isFuel, fuelTrap]

/-- A map is bijective when it is injective and surjective. Defined here
    so the statement does not depend on a name outside this file. -/
def Injective {α β : Type} (f : α → β) : Prop :=
  ∀ a₁ a₂, f a₁ = f a₂ → a₁ = a₂

def Surjective {α β : Type} (f : α → β) : Prop :=
  ∀ b, ∃ a, f a = b

def Bijective {α β : Type} (f : α → β) : Prop :=
  Injective f ∧ Surjective f

/-- Swap 0 and 1 inside a raw numeric success value. Halt values stay,
    and a tuple success value stays. -/
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

theorem id_bijective : Bijective (fun c : CoreCatch => c) := by
  refine ⟨?_, ?_⟩
  · intro a b h
    exact h
  · intro c
    exact ⟨c, rfl⟩

/-- The identity is bijective, and so is `twist`, and they disagree on
    `.ok (.num 0)`. A proof that a map is bijective does not say which
    value the evaluator returns. -/
theorem bijective_insufficient :
    Bijective (fun c : CoreCatch => c) ∧
    Bijective twist ∧
    (fun c : CoreCatch => c) (.ok (.num 0)) ≠ twist (.ok (.num 0)) := by
  refine ⟨id_bijective, twist_bijective, ?_⟩
  intro h
  simp [twist, flipNum] at h

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
