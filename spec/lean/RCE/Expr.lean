/-! RCE core: values, types, expressions; small-step semantics with traps
    (spec 6.2/6.3 N2), typing (5.3), determinism, progress, preservation,
    and agreement of the evaluator with the small-step relation. -/
namespace RCE

inductive Ty where
  | int (lo hi : Int)
  | bool
  deriving DecidableEq, Repr

inductive Val where
  | int (n : Int)
  | bool (b : Bool)
  deriving DecidableEq, Repr

def Val.hasTy : Val → Ty → Prop
  | .int n, .int lo hi => lo ≤ n ∧ n ≤ hi
  | .bool _, .bool => True
  | _, _ => False

inductive Trap where
  | overflow | divZero | badIndex | loopBound
  deriving DecidableEq, Repr

inductive BinOp where
  | add | sub | mul | div | rem | lt | eq
  deriving DecidableEq, Repr

inductive Expr where
  | lit (v : Val)
  | trap (t : Trap)
  | field (i : Nat)
  | arg (i : Nat)
  /-- `bin op ta a b`: operator `op` at operand type `ta` (the elaborated
      core records the static operand type, which fixes the checked range). -/
  | bin (op : BinOp) (ta : Ty) (a b : Expr)
  | ite (c a b : Expr)
  deriving DecidableEq, Repr

/-- Truncating division (spec N2: `/` truncates toward zero). -/
def tdiv (a b : Int) : Int :=
  if (a < 0) = (b < 0) then (Int.natAbs a / Int.natAbs b : Nat)
  else -((Int.natAbs a / Int.natAbs b : Nat) : Int)

/-- Remainder with the sign of the dividend. -/
def trem (a b : Int) : Int := a - b * tdiv a b

/-- Checked range: the result must lie in the static type's range (N2). -/
def chk (lo hi r : Int) : Except Trap Val :=
  if lo ≤ r ∧ r ≤ hi then .ok (.int r) else .error .overflow

/-- The meaning of a binary operator at a static type (spec 6.3). -/
def binop (op : BinOp) (t : Ty) (a b : Val) : Except Trap Val :=
  match op, t, a, b with
  | .add, .int lo hi, .int x, .int y => chk lo hi (x + y)
  | .sub, .int lo hi, .int x, .int y => chk lo hi (x - y)
  | .mul, .int lo hi, .int x, .int y => chk lo hi (x * y)
  | .div, .int lo hi, .int x, .int y => if y = 0 then .error .divZero else chk lo hi (tdiv x y)
  | .rem, .int lo hi, .int x, .int y => if y = 0 then .error .divZero else chk lo hi (trem x y)
  | .lt, _, .int x, .int y => .ok (.bool (decide (x < y)))
  | .eq, _, x, y => .ok (.bool (decide (x = y)))
  | _, _, _, _ => .error .badIndex

/-- Environment: the stepped region's fields and the event payload. -/
structure Env where
  fields : List Val
  args : List Val

/-- Typing context: field types, argument types. -/
structure Ctx where
  fields : List Ty
  args : List Ty

end RCE

namespace RCE

/-- Operator typing: operand type `ta`, result type `t`. -/
inductive OpTy : BinOp → Ty → Ty → Prop where
  | add {lo hi} : OpTy .add (.int lo hi) (.int lo hi)
  | sub {lo hi} : OpTy .sub (.int lo hi) (.int lo hi)
  | mul {lo hi} : OpTy .mul (.int lo hi) (.int lo hi)
  | div {lo hi} : OpTy .div (.int lo hi) (.int lo hi)
  | rem {lo hi} : OpTy .rem (.int lo hi) (.int lo hi)
  | lt {lo hi} : OpTy .lt (.int lo hi) .bool
  | eq {t} : OpTy .eq t .bool

/-- Σ;Γ ⊢ e : τ for the expression core (spec 5.3). -/
inductive HasTy (Γ : Ctx) : Expr → Ty → Prop where
  | lit {v t} : v.hasTy t → HasTy Γ (.lit v) t
  | trap {m t} : HasTy Γ (.trap m) t
  | field {i t} : Γ.fields.get? i = some t → HasTy Γ (.field i) t
  | arg {i t} : Γ.args.get? i = some t → HasTy Γ (.arg i) t
  | bin {op ta a b t} : OpTy op ta t → HasTy Γ a ta → HasTy Γ b ta → HasTy Γ (.bin op ta a b) t
  | ite {c a b t} : HasTy Γ c .bool → HasTy Γ a t → HasTy Γ b t → HasTy Γ (.ite c a b) t

/-- The environment agrees with the context. -/
def WtEnv (Γ : Ctx) (E : Env) : Prop :=
  (∀ i v t, E.fields.get? i = some v → Γ.fields.get? i = some t → v.hasTy t) ∧
  (∀ i v t, E.args.get? i = some v → Γ.args.get? i = some t → v.hasTy t)

/-- Small-step reduction (spec 6.2, left-to-right, traps propagate). -/
inductive Step (E : Env) : Expr → Expr → Prop where
  | field {i v} : E.fields.get? i = some v → Step E (.field i) (.lit v)
  | fieldBad {i} : E.fields.get? i = none → Step E (.field i) (.trap .badIndex)
  | arg {i v} : E.args.get? i = some v → Step E (.arg i) (.lit v)
  | argBad {i} : E.args.get? i = none → Step E (.arg i) (.trap .badIndex)
  | binL {op ta a a' b} : Step E a a' → Step E (.bin op ta a b) (.bin op ta a' b)
  | binR {op ta v b b'} : Step E b b' → Step E (.bin op ta (.lit v) b) (.bin op ta (.lit v) b')
  | binTrapL {op ta m b} : Step E (.bin op ta (.trap m) b) (.trap m)
  | binTrapR {op ta v m} : Step E (.bin op ta (.lit v) (.trap m)) (.trap m)
  | binOk {op ta x y r} : binop op ta x y = .ok r → Step E (.bin op ta (.lit x) (.lit y)) (.lit r)
  | binErr {op ta x y m} : binop op ta x y = .error m → Step E (.bin op ta (.lit x) (.lit y)) (.trap m)
  | iteC {c c' a b} : Step E c c' → Step E (.ite c a b) (.ite c' a b)
  | iteT {a b} : Step E (.ite (.lit (.bool true)) a b) a
  | iteF {a b} : Step E (.ite (.lit (.bool false)) a b) b
  | iteTrap {m a b} : Step E (.ite (.trap m) a b) (.trap m)
  | iteBad {n a b} : Step E (.ite (.lit (.int n)) a b) (.trap .badIndex)

theorem lit_nostep {E v e} : ¬ Step E (.lit v) e := by intro h; cases h
theorem trap_nostep {E m e} : ¬ Step E (.trap m) e := by intro h; cases h

/-- **T1 (expressions): the small-step relation is deterministic.** -/
theorem step_det {E e e1 e2} (h1 : Step E e e1) (h2 : Step E e e2) : e1 = e2 := by
  induction h1 generalizing e2 with
  | field h => cases h2 with
    | field h' => rw [h] at h'; cases h'; rfl
    | fieldBad h' => rw [h] at h'; cases h'
  | fieldBad h => cases h2 with
    | field h' => rw [h] at h'; cases h'
    | fieldBad => rfl
  | arg h => cases h2 with
    | arg h' => rw [h] at h'; cases h'; rfl
    | argBad h' => rw [h] at h'; cases h'
  | argBad h => cases h2 with
    | arg h' => rw [h] at h'; cases h'
    | argBad => rfl
  | binL h ih =>
    cases h2 with
    | binL h' => rw [ih h']
    | binR _ => exact absurd h lit_nostep
    | binTrapL => exact absurd h trap_nostep
    | binTrapR => exact absurd h lit_nostep
    | binOk _ => exact absurd h lit_nostep
    | binErr _ => exact absurd h lit_nostep
  | binR h ih =>
    cases h2 with
    | binL h' => exact absurd h' lit_nostep
    | binR h' => rw [ih h']
    | binTrapR => exact absurd h trap_nostep
    | binOk _ => exact absurd h lit_nostep
    | binErr _ => exact absurd h lit_nostep
  | binTrapL =>
    cases h2 with
    | binL h' => exact absurd h' trap_nostep
    | binTrapL => rfl
  | binTrapR =>
    cases h2 with
    | binL h' => exact absurd h' lit_nostep
    | binR h' => exact absurd h' trap_nostep
    | binTrapR => rfl
  | binOk h =>
    cases h2 with
    | binL h' => exact absurd h' lit_nostep
    | binR h' => exact absurd h' lit_nostep
    | binOk h' => rw [h] at h'; cases h'; rfl
    | binErr h' => rw [h] at h'; cases h'
  | binErr h =>
    cases h2 with
    | binL h' => exact absurd h' lit_nostep
    | binR h' => exact absurd h' lit_nostep
    | binOk h' => rw [h] at h'; cases h'
    | binErr h' => rw [h] at h'; cases h'; rfl
  | iteC h ih =>
    cases h2 with
    | iteC h' => rw [ih h']
    | iteT => exact absurd h lit_nostep
    | iteF => exact absurd h lit_nostep
    | iteTrap => exact absurd h trap_nostep
    | iteBad => exact absurd h lit_nostep
  | iteT => cases h2 with
    | iteC h' => exact absurd h' lit_nostep
    | iteT => rfl
  | iteF => cases h2 with
    | iteC h' => exact absurd h' lit_nostep
    | iteF => rfl
  | iteTrap => cases h2 with
    | iteC h' => exact absurd h' trap_nostep
    | iteTrap => rfl
  | iteBad => cases h2 with
    | iteC h' => exact absurd h' lit_nostep
    | iteBad => rfl

/-- **T4 Progress** (expressions): a well-typed expression is a value, a
    defined trap, or steps. -/
theorem progress {Γ E e t} (ht : HasTy Γ e t) :
    (∃ v, e = .lit v) ∨ (∃ m, e = .trap m) ∨ ∃ e', Step E e e' := by
  induction ht with
  | lit _ => exact .inl ⟨_, rfl⟩
  | trap => exact .inr (.inl ⟨_, rfl⟩)
  | @field i _ _ =>
    right; right
    cases h : E.fields.get? i with
    | none => exact ⟨_, .fieldBad h⟩
    | some v => exact ⟨_, .field h⟩
  | @arg i _ _ =>
    right; right
    cases h : E.args.get? i with
    | none => exact ⟨_, .argBad h⟩
    | some v => exact ⟨_, .arg h⟩
  | @bin op ta a b t _ _ _ iha ihb =>
    right; right
    rcases iha with ⟨va, rfl⟩ | ⟨m, rfl⟩ | ⟨a', ha⟩
    · rcases ihb with ⟨vb, rfl⟩ | ⟨m, rfl⟩ | ⟨b', hb⟩
      · cases h : binop op ta va vb with
        | ok r => exact ⟨_, .binOk h⟩
        | error m => exact ⟨_, .binErr h⟩
      · exact ⟨_, .binTrapR⟩
      · exact ⟨_, .binR hb⟩
    · exact ⟨_, .binTrapL⟩
    · exact ⟨_, .binL ha⟩
  | @ite c a b t _ _ _ ihc _ _ =>
    right; right
    rcases ihc with ⟨v, rfl⟩ | ⟨m, rfl⟩ | ⟨c', hc⟩
    · cases v with
      | int n => exact ⟨_, .iteBad⟩
      | bool b => cases b
                  · exact ⟨_, .iteF⟩
                  · exact ⟨_, .iteT⟩
    · exact ⟨_, .iteTrap⟩
    · exact ⟨_, .iteC hc⟩

theorem chk_ty {lo hi r v} (h : chk lo hi r = .ok v) : v.hasTy (.int lo hi) := by
  unfold chk at h
  split at h
  · cases h; simp [Val.hasTy]; omega
  · cases h

/-- Operators preserve types (each checked result lies in its range). -/
theorem binop_ty {op ta t x y r} (ho : OpTy op ta t) (hx : x.hasTy ta) (hy : y.hasTy ta)
    (h : binop op ta x y = .ok r) : r.hasTy t := by
  cases ho <;> cases x <;> cases y <;> simp [binop] at h <;> simp [Val.hasTy] at hx hy
  all_goals first
    | exact chk_ty h
    | (cases h; simp [Val.hasTy])
    | (split at h <;> first | cases h | exact chk_ty h)

/-- **T5 Preservation** (expressions). -/
theorem preservation {Γ E e e' t} (hE : WtEnv Γ E) (hs : Step E e e') (ht : HasTy Γ e t) :
    HasTy Γ e' t := by
  induction hs generalizing t with
  | field h => cases ht with | field h' => exact .lit (hE.1 _ _ _ h h')
  | fieldBad _ => exact .trap
  | arg h => cases ht with | arg h' => exact .lit (hE.2 _ _ _ h h')
  | argBad _ => exact .trap
  | binL _ ih => cases ht with | bin ho ha hb => exact .bin ho (ih ha) hb
  | binR _ ih => cases ht with | bin ho ha hb => exact .bin ho ha (ih hb)
  | binTrapL => exact .trap
  | binTrapR => exact .trap
  | binOk h => cases ht with
    | bin ho ha hb => cases ha with | lit hx => cases hb with | lit hy => exact .lit (binop_ty ho hx hy h)
  | binErr _ => exact .trap
  | iteC _ ih => cases ht with | ite hc ha hb => exact .ite (ih hc) ha hb
  | iteT => cases ht with | ite _ ha _ => exact ha
  | iteF => cases ht with | ite _ _ hb => exact hb
  | iteTrap => exact .trap
  | iteBad => exact .trap

/-- Reflexive-transitive closure. -/
inductive Steps (E : Env) : Expr → Expr → Prop where
  | refl {e} : Steps E e e
  | step {e e' e''} : Step E e e' → Steps E e' e'' → Steps E e e''

theorem Steps.trans {E a b c} (h1 : Steps E a b) (h2 : Steps E b c) : Steps E a c := by
  induction h1 with
  | refl => exact h2
  | step s _ ih => exact .step s (ih h2)

theorem Steps.binL {E op ta a a' b} (h : Steps E a a') : Steps E (.bin op ta a b) (.bin op ta a' b) := by
  induction h with
  | refl => exact .refl
  | step s _ ih => exact .step (.binL s) ih
theorem Steps.binR {E op ta v b b'} (h : Steps E b b') :
    Steps E (.bin op ta (.lit v) b) (.bin op ta (.lit v) b') := by
  induction h with
  | refl => exact .refl
  | step s _ ih => exact .step (.binR s) ih
theorem Steps.iteC {E c c' a b} (h : Steps E c c') : Steps E (.ite c a b) (.ite c' a b) := by
  induction h with
  | refl => exact .refl
  | step s _ ih => exact .step (.iteC s) ih

/-- The reference evaluator: what implementations compute. -/
def eval (E : Env) : Expr → Except Trap Val
  | .lit v => .ok v
  | .trap m => .error m
  | .field i => match E.fields.get? i with | some v => .ok v | none => .error .badIndex
  | .arg i => match E.args.get? i with | some v => .ok v | none => .error .badIndex
  | .bin op ta a b => do let x ← eval E a; let y ← eval E b; binop op ta x y
  | .ite c a b => match eval E c with
    | .ok (.bool true) => eval E a
    | .ok (.bool false) => eval E b
    | .ok (.int _) => .error .badIndex
    | .error m => .error m

def Res.toExpr : Except Trap Val → Expr
  | .ok v => .lit v
  | .error m => .trap m

/-- **Adequacy**: the evaluator agrees with the small-step semantics. -/
theorem eval_steps (E : Env) (e : Expr) : Steps E e (Res.toExpr (eval E e)) := by
  induction e with
  | lit v => exact .refl
  | trap m => exact .refl
  | field i =>
    simp only [eval]; cases h : E.fields.get? i with
    | none => exact .step (.fieldBad h) .refl
    | some v => exact .step (.field h) .refl
  | arg i =>
    simp only [eval]; cases h : E.args.get? i with
    | none => exact .step (.argBad h) .refl
    | some v => exact .step (.arg h) .refl
  | bin op ta a b iha ihb =>
    simp only [eval, bind, Except.bind]
    cases ha : eval E a with
    | error m => rw [ha] at iha; dsimp only; exact (Steps.binL iha).trans (.step .binTrapL .refl)
    | ok x =>
      rw [ha] at iha
      cases hb : eval E b with
      | error m => rw [hb] at ihb; dsimp only; exact (Steps.binL iha).trans ((Steps.binR ihb).trans (.step .binTrapR .refl))
      | ok y =>
        rw [hb] at ihb
        dsimp only
        refine (Steps.binL iha).trans ((Steps.binR ihb).trans ?_)
        cases hr : binop op ta x y with
        | ok r => exact .step (.binOk hr) .refl
        | error m => exact .step (.binErr hr) .refl
  | ite c a b ihc iha ihb =>
    simp only [eval]
    cases hc : eval E c with
    | error m => rw [hc] at ihc; exact (Steps.iteC ihc).trans (.step .iteTrap .refl)
    | ok v =>
      rw [hc] at ihc
      cases v with
      | int n => exact (Steps.iteC ihc).trans (.step .iteBad .refl)
      | bool bb => cases bb
                   · exact (Steps.iteC ihc).trans (.step .iteF ihb)
                   · exact (Steps.iteC ihc).trans (.step .iteT iha)

/-- Typed evaluation yields a value of the static type (type soundness of
    the evaluator; follows the same case analysis as preservation). -/
theorem eval_ty {Γ E e t v} (hE : WtEnv Γ E) (ht : HasTy Γ e t) (h : eval E e = .ok v) :
    v.hasTy t := by
  induction ht generalizing v with
  | lit hv => simp [eval] at h; subst h; exact hv
  | trap => simp [eval] at h
  | @field i _ hg =>
    simp only [eval] at h; cases hf : E.fields.get? i <;> rw [hf] at h <;> simp at h
    subst h; exact hE.1 _ _ _ hf hg
  | @arg i _ hg =>
    simp only [eval] at h; cases hf : E.args.get? i <;> rw [hf] at h <;> simp at h
    subst h; exact hE.2 _ _ _ hf hg
  | @bin op ta a b _ ho _ _ iha ihb =>
    simp only [eval, bind, Except.bind] at h
    cases hx : eval E a with
    | error m => rw [hx] at h; cases h
    | ok x =>
      cases hy : eval E b with
      | error m => rw [hx, hy] at h; cases h
      | ok y => rw [hx, hy] at h; exact binop_ty ho (iha hx) (ihb hy) h
  | ite _ _ _ _ iha ihb =>
    simp only [eval] at h
    split at h
    · exact iha h
    · exact ihb h
    · cases h
    · cases h

end RCE
