/-
Obligations T42 to T56 (specification Appendix F.9), stated and NOT proved.

This file is not part of `lake build` (the library target is `RCE`) and is not
read by `scripts/lean-audit.sh`; it is checked alone with
`lake env lean obligations/Obligations.lean`. It contains no theorem, no
`sorry` and no axiom: each obligation is a `def ... : Prop` over an abstract
model record that names only what the statement needs. A `def` here says what
a future proof would have to establish about the real semantics; it is
evidence of nothing until the record is instantiated with the semantics of
`RCE/Core`, `RCE/Conc` and `RCE/Host` and the `Prop` is proved. T42 to T56
stay "stated, unproved" in the theorem table (spec/book/10-theorems.md).
-/
namespace RCE.Obligations

/-- T42 (F.4.2) Noninterference with labels. -/
structure LabelModel where
  Input : Type
  Outcome : Type            -- a value, a trap or the refusal outcome
  Public : Type             -- what a public observer sees of an outcome
  run : Input → Outcome
  agreePublic : Input → Input → Prop   -- equal on every public-label field
  observe : Outcome → Public
def T42 (m : LabelModel) : Prop :=
  ∀ i j, m.agreePublic i j → m.observe (m.run i) = m.observe (m.run j)

/-- T43 (F.4.2) Robust declassification. -/
structure ReleaseModel where
  Path : Type
  Operand : Type
  reachesRelease : Path → Prop
  controlTrusted : Path → Prop
  deciding : Path → Operand → Prop
  operandTrusted : Operand → Prop
def T43 (m : ReleaseModel) : Prop :=
  ∀ p, m.reachesRelease p → m.controlTrusted p ∧ ∀ o, m.deciding p o → m.operandTrusted o

/-- T44 (F.5.2, F.6.1) Typed output. -/
structure RenderModel where
  Template : Type
  Position : Type
  rendered : Template → List Nat           -- the rendered text, as code units
  structural : Template → Position → Prop  -- a structural metacharacter at a position
  fromLiteral : Template → Position → Prop -- the position lies in a literal segment
def T44 (m : RenderModel) : Prop :=
  ∀ t p, m.structural t p → m.fromLiteral t p

/-- T45 (F.5.4) Constant time. -/
structure CtModel where
  Input : Type
  Decision : Type
  secretOnlyDiffer : Input → Input → Prop
  decisions : Input → List Decision        -- branch, index and loop decisions of a `ct fn`
def T45 (m : CtModel) : Prop :=
  ∀ i j, m.secretOnlyDiffer i j → m.decisions i = m.decisions j

/-- T46 (F.6.2) Contract enforcement. -/
inductive CallResult (V : Type) where
  | returned (v : V)
  | kept                                  -- the event is refused with the fixed record
structure ContractModel where
  State : Type
  Value : Type
  call : State → CallResult Value
  requires : State → Prop                    -- at entry
  ensures : State → Value → Prop             -- at the return
def T46 (m : ContractModel) : Prop :=
  ∀ s v, m.call s = .returned v → m.requires s ∧ m.ensures s v

/-- T47 (F.4.4) Transactional calls. -/
structure TxModel where
  Caller : Type                              -- the caller's `&mut` arguments and locals
  Value : Type
  call : Caller → Caller × CallResult Value
def T47 (m : TxModel) : Prop :=
  ∀ c, (m.call c).2 = CallResult.kept → (m.call c).1 = c

-- T48 is proved of a model in RCE/Discharged.lean (in the audit); not stated here.
/-- T49 (F.6.3) Ghost erasure. -/
structure GhostModel where
  Program : Type
  Observation : Type                         -- δ, λ and the log
  eraseGhost : Program → Program
  observe : Program → Observation
def T49 (m : GhostModel) : Prop :=
  ∀ p, m.observe (m.eraseGhost p) = m.observe p

/-- T50 (F.4.3) Definition hash. -/
structure HashModel where
  Def : Type
  Hash : Type
  Meaning : Type
  Input : Type
  Result : Type
  hash : Def → Hash
  meaning : Def → Meaning
  compatible : Def → Def → Prop              -- `rce api-diff` says compatible
  requires : Def → Input → Prop
  result : Def → Input → Result
def T50 (m : HashModel) : Prop :=
  (∀ a b, m.hash a = m.hash b → m.meaning a = m.meaning b) ∧
  (∀ a b, m.compatible a b → ∀ i, m.requires b i → m.result a i = m.result b i)

-- T51 is proved of a model in RCE/Discharged.lean (in the audit); not stated here.
/-- T52 (F.5.5) Budgets. -/
structure BudgetModel where
  Target : Type
  Program : Type
  Op : Type
  work : Target → Program → Nat              -- units of work done before the run ends
  budget : Program → Nat
  trapsAt : Target → Program → Option Op     -- the operation at which `trap("budget")` falls
def T52 (m : BudgetModel) : Prop :=
  ∀ p t, m.work t p ≤ m.budget p ∧ ∀ t', m.trapsAt t p = m.trapsAt t' p

-- T53 is proved of a model in RCE/Discharged.lean (in the audit); not stated here.
/-- T54 (F.5.3) Supply chain integrity, under collision resistance of the hash. -/
structure SupplyModel where
  Module : Type
  Digest : Type
  digest : Module → Digest
  manifest : List Digest
  acceptedBuild : List Module → Prop
def T54 (m : SupplyModel) : Prop :=
  (∀ a b, m.digest a = m.digest b → a = b) →
  ∀ ms, m.acceptedBuild ms → ms.map m.digest = m.manifest

/-- T55 (F.7) Broker invariant, by induction on grant events. -/
structure BrokerModel where
  Session : Type
  Event : Type
  isGrant : Event → Prop
  holdsAllThree : List Event → Session → Prop      -- private data, untrusted content, external action
  explicitGrant : List Event → Session → Prop
def T55 (m : BrokerModel) : Prop :=
  ∀ log s, m.holdsAllThree log s → m.explicitGrant log s

/-- T56 (5.6) Relaxed sequential borrows preserve T6 (the relaxation is not adopted). -/
structure BorrowModel where
  Program : Type
  strictAccepted : Program → Prop
  relaxedAccepted : Program → Prop
  T6 : Program → Prop                              -- the guarantee of T6 for a program
def T56 (m : BorrowModel) : Prop :=
  ∀ p, m.relaxedAccepted p → m.T6 p

end RCE.Obligations
