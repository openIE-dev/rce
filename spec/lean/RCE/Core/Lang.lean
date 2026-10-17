import RCE.Core.F64
import RCE.Core.F32Fmt
/-! Soundness stream S1 (proposals/soundness.md): the core expression
    language with fuel. Ranged integers, booleans, text, products (structs
    elaborate to nested products), sums (enums elaborate to nested sums),
    `match` on sums, `let`, declared errors with `catch` (traps are not caught), and calls to
    top-level functions. Evaluation is a fuel-indexed definitional
    interpreter. Its result is a value, a defined trap,
    exhaustion, or `stuck`; type soundness is the statement that a
    well-typed program is never `stuck`. -/
namespace RCE.Core

inductive Ty where
  | int (lo hi : Int)
  | bool
  | text
  | unit
  | prod (a b : Ty)
  | sum (a b : Ty)
  | f64
  | f32
  deriving DecidableEq, Repr

inductive Val where
  | int (n : Int)
  | bool (b : Bool)
  | text (s : String)
  | unit
  | pair (a b : Val)
  | inl (v : Val)
  | inr (v : Val)
  | f64 (bits : Nat)
  /-- an f32 value, held as the binary64 pattern of the same real -/
  | f32 (bits : Nat)
  deriving DecidableEq, Repr

inductive VTy : Val → Ty → Prop where
  | int {n lo hi} : lo ≤ n → n ≤ hi → VTy (.int n) (.int lo hi)
  | bool {b} : VTy (.bool b) .bool
  | text {s} : VTy (.text s) .text
  | unit : VTy .unit .unit
  | pair {a b ta tb} : VTy a ta → VTy b tb → VTy (.pair a b) (.prod ta tb)
  | inl {v ta tb} : VTy v ta → VTy (.inl v) (.sum ta tb)
  | inr {v ta tb} : VTy v tb → VTy (.inr v) (.sum ta tb)
  | f64 {b} : VTy (.f64 b) .f64
  | f32 {b} : VTy (.f32 b) .f32

/-- Defined traps: checked-range overflow, division by zero, and a raised
    program error with its code. -/
inductive Trap where
  | overflow (v lo hi : Int) | divZero | shift | nonFinite | user (code : Nat)
  /-- a `while` ran past its bound (interp.rs `run_loop`) -/
  | loopBound
  deriving DecidableEq, Repr

inductive BinOp where
  | add | sub | mul | div | rem | lt | eq | and | or | cat
  | band | bor | bxor | shl | shr
  /-- methods (book 6.3 N2, N5): `min max`, `saturating_*`, `wrapping_*` -/
  | min | max | sadd | ssub | smul | wadd | wsub | wmul
  deriving DecidableEq, Repr

/-- Unary methods (book 6.3 N2, N5): integer `abs`, float `abs sqrt floor
    ceil round`. -/
inductive UnOp where
  | iabs | fabs | sqrt | floor | ceil | round
  /-- `-e` checked into `[lo, hi]` (6.3 N2), and float negation (exact). -/
  | ineg (lo hi : Int) | fneg
  /-- f32 (6.3 N7): `x as f32` from an integer or f64, `x as f64` from f32,
      `x as int[lo, hi]` from f32 (truncate, check) -/
  | tof32 | widen | trunc32 (lo hi : Int)
  deriving DecidableEq, Repr

inductive Expr where
  | var (i : Nat)
  | lit (v : Val)
  | bin (op : BinOp) (ta : Ty) (a b : Expr)
  | ite (c a b : Expr)
  | let_ (a body : Expr)
  | pair (a b : Expr)
  | fst (e : Expr)
  | snd (e : Expr)
  | inl (e : Expr)
  | inr (e : Expr)
  /-- `case e l r`: `l` and `r` bind the payload as variable 0. -/
  | case (e l r : Expr)
  | raise (code : Nat)
  /-- `catch a h`: a declared error raised by `a` is handled by `h`; traps
      propagate (book 7, E-Catch: `catch` handles only declared errors). -/
  | catch_ (a h : Expr)
  | call (f : Nat) (arg : Expr)
  /-- One unit of fuel, then `e`: the entry charge of a call. -/
  | tick (e : Expr)
  /-- `try e else X` over an option or result encoded as `sum absent present`
      (`none` is `inl ()`): the present payload, or the declared error `code`
      (book 7 E-Try-F, E-TryElse). The error payload is not carried yet. -/
  | try_ (e : Expr) (code : Nat)
  /-- `e as int[lo, hi]`: checked integer conversion (spec 6.3 N2). -/
  | cast (lo hi : Int) (e : Expr)
  /-- `e as f64` from an integer, rounded to nearest (S3, book 6.3 N3). -/
  | tof (e : Expr)
  /-- `e.m()` for a unary method `m` at operand type `ta`. -/
  | un (op : UnOp) (ta : Ty) (e : Expr)
  deriving Repr

inductive Res where
  | ok (v : Val)
  | err (t : Trap)
  | oof
  | stuck
  deriving DecidableEq, Repr

/-- Truncating division (spec 6.3 N2). -/
def tdiv (a b : Int) : Int :=
  if (a < 0) = (b < 0) then (Int.natAbs a / Int.natAbs b : Nat)
  else -((Int.natAbs a / Int.natAbs b : Nat) : Int)

def trem (a b : Int) : Int := a - b * tdiv a b

def chk (lo hi r : Int) : Res :=
  if lo ≤ r ∧ r ≤ hi then .ok (.int r) else .err (.overflow r lo hi)

/-! Two's-complement bitwise operations on `Int` (`-(a+1)` is `~a`). -/
def iand : Int → Int → Int
  | .ofNat a, .ofNat b => .ofNat (a &&& b)
  | .ofNat a, .negSucc b => .ofNat (a - (a &&& b))
  | .negSucc a, .ofNat b => .ofNat (b - (a &&& b))
  | .negSucc a, .negSucc b => .negSucc (a ||| b)

def ior : Int → Int → Int
  | .ofNat a, .ofNat b => .ofNat (a ||| b)
  | .ofNat a, .negSucc b => .negSucc (b - (a &&& b))
  | .negSucc a, .ofNat b => .negSucc (a - (a &&& b))
  | .negSucc a, .negSucc b => .negSucc (a &&& b)

def ixor : Int → Int → Int
  | .ofNat a, .ofNat b => .ofNat (a ^^^ b)
  | .ofNat a, .negSucc b => .negSucc (a ^^^ b)
  | .negSucc a, .ofNat b => .negSucc (a ^^^ b)
  | .negSucc a, .negSucc b => .ofNat (a ^^^ b)

/-- Keep the low bits of the width `hi - lo + 1`, read signed when `lo < 0`. -/
def wrap (lo hi r : Int) : Int :=
  let m := hi - lo + 1
  let u := r % m
  if lo < 0 ∧ u > hi then u - m else u

def fres : Option Nat → Res
  | some b => .ok (.f64 b)
  | none => .err .nonFinite

/-- S3: binary64 by exact rounding (`RCE.F64`); a non-finite result traps (N3). -/
def fbinop : BinOp → Nat → Nat → Res
  | .add, x, y => fres (RCE.F64.add x y)
  | .sub, x, y => fres (RCE.F64.sub x y)
  | .mul, x, y => fres (RCE.F64.mul x y)
  | .div, x, y => fres (RCE.F64.div x y)
  | .lt, x, y => .ok (.bool (RCE.F64.lt x y))
  | .eq, x, y => .ok (.bool (RCE.F64.eq x y))
  -- 6.3 N5: IEEE order; the receiver on a tie
  | .min, x, y => .ok (.f64 (if RCE.F64.lt x y || RCE.F64.eq x y then x else y))
  | .max, x, y => .ok (.f64 (if RCE.F64.lt y x || RCE.F64.eq x y then x else y))
  | _, _, _ => .stuck

/-- Clamp to `[lo, hi]` (saturating methods). -/
def clamp (lo hi r : Int) : Int := max lo (min hi r)

/-- Residue in `[lo, lo + w)`, `w = hi - lo + 1` (wrapping methods). -/
def wrapR (lo hi r : Int) : Int := (r - lo) % (hi - lo + 1) + lo

def fres32 : Option Nat → Res
  | some b => .ok (.f32 b)
  | none => .err .nonFinite

/-- f32 arithmetic: the f64 operation, rounded again to f32 (N7). -/
def fbinop32 : BinOp → Nat → Nat → Res
  | .add, x, y => fres32 (RCE.F64.add32 x y)
  | .sub, x, y => fres32 (RCE.F64.sub32 x y)
  | .mul, x, y => fres32 (RCE.F64.mul32 x y)
  | .div, x, y => fres32 (RCE.F64.div32 x y)
  | .lt, x, y => .ok (.bool (RCE.F64.lt x y))
  | .eq, x, y => .ok (.bool (RCE.F64.eq x y))
  | _, _, _ => .stuck

def ibinop : BinOp → Ty → Val → Val → Res
  | .add, .int lo hi, .int x, .int y => chk lo hi (x + y)
  | .sub, .int lo hi, .int x, .int y => chk lo hi (x - y)
  | .mul, .int lo hi, .int x, .int y => chk lo hi (x * y)
  | .div, .int lo hi, .int x, .int y => if y = 0 then .err .divZero else chk lo hi (tdiv x y)
  -- book 6.3 N2: `i64.MIN % -1` traps like `i64.MIN / -1` (the quotient is out of range)
  | .rem, .int lo hi, .int x, .int y =>
    if y = 0 then .err .divZero else
    match chk lo hi (tdiv x y) with
    | .err t => .err t
    | _ => chk lo hi (trem x y)
  -- book 6.3 N2: bitwise on the two's-complement representation; shifts
  -- trap when the amount is negative or at least the bit width
  | .band, .int lo hi, .int x, .int y => chk lo hi (iand x y)
  | .bor, .int lo hi, .int x, .int y => chk lo hi (ior x y)
  | .bxor, .int lo hi, .int x, .int y => chk lo hi (ixor x y)
  | .shl, .int lo hi, .int x, .int y =>
    if y < 0 ∨ 2 ^ y.toNat ≥ hi - lo + 1 then .err .shift else chk lo hi (wrap lo hi (x * 2 ^ y.toNat))
  | .shr, .int lo hi, .int x, .int y =>
    if y < 0 ∨ 2 ^ y.toNat ≥ hi - lo + 1 then .err .shift else chk lo hi (Int.shiftRight x y.toNat)
  | .min, .int lo hi, .int x, .int y => chk lo hi (min x y)
  | .max, .int lo hi, .int x, .int y => chk lo hi (max x y)
  | .sadd, .int lo hi, .int x, .int y => chk lo hi (clamp lo hi (x + y))
  | .ssub, .int lo hi, .int x, .int y => chk lo hi (clamp lo hi (x - y))
  | .smul, .int lo hi, .int x, .int y => chk lo hi (clamp lo hi (x * y))
  | .wadd, .int lo hi, .int x, .int y => chk lo hi (wrapR lo hi (x + y))
  | .wsub, .int lo hi, .int x, .int y => chk lo hi (wrapR lo hi (x - y))
  | .wmul, .int lo hi, .int x, .int y => chk lo hi (wrapR lo hi (x * y))
  | .lt, .int _ _, .int x, .int y => .ok (.bool (decide (x < y)))
  | .eq, _, x, y => .ok (.bool (decide (x = y)))
  | .and, .bool, .bool x, .bool y => .ok (.bool (x && y))
  | .or, .bool, .bool x, .bool y => .ok (.bool (x || y))
  | .cat, .text, .text x, .text y => .ok (.text (x ++ y))
  | _, _, _, _ => .stuck

def unop : UnOp → Ty → Val → Res
  | .iabs, .int lo hi, .int x => chk lo hi (Int.natAbs x)
  | .fabs, .f64, .f64 b => .ok (.f64 (RCE.F64.abs b))
  | .sqrt, .f64, .f64 b => fres (RCE.F64.sqrt b)
  | .floor, .f64, .f64 b => fres (RCE.F64.toIntegral .floor b)
  | .ceil, .f64, .f64 b => fres (RCE.F64.toIntegral .ceil b)
  | .round, .f64, .f64 b => fres (RCE.F64.toIntegral .away b)
  | .ineg lo hi, .int _ _, .int x => chk lo hi (-x)
  | .fneg, .f64, .f64 b => .ok (.f64 (RCE.F64.neg b))
  | .fneg, .f32, .f32 b => .ok (.f32 (RCE.F64.neg b))
  | .fabs, .f32, .f32 b => .ok (.f32 (RCE.F64.abs b))
  | .tof32, .int _ _, .int x => fres32 (RCE.F64.ofInt32 x)
  | .tof32, .f64, .f64 b => fres32 (RCE.F64.to32 b)
  | .widen, .f32, .f32 b => .ok (.f64 b)
  | .trunc32 lo hi, .f32, .f32 b => chk lo hi (RCE.F64.trunc b)
  | _, _, _ => .stuck

def binop (op : BinOp) (ta : Ty) (x y : Val) : Res :=
  match ta, x, y with
  | .f64, .f64 a, .f64 b => fbinop op a b
  | .f32, .f32 a, .f32 b => fbinop32 op a b
  | _, _, _ => ibinop op ta x y

/-- A program: each function has an argument type, a result type, a body. -/
abbrev Prog := List (Ty × Ty × Expr)

/-- The definitional interpreter. Every node costs one unit of fuel, and a
    call costs one more on entry (book 12 Fuel: one tick per evaluated
    expression and per call of a user function), so tick counts can be
    compared with the reference evaluator. -/
def eval (P : Prog) : Nat → List Val → Expr → Res
  | 0, _, _ => .oof
  | n+1, ρ, e =>
    match e with
    | .var i => match ρ.get? i with | some v => .ok v | none => .stuck
    | .lit v => .ok v
    | .bin op ta a b =>
      match eval P n ρ a with
      | .ok x => match eval P n ρ b with
        | .ok y => binop op ta x y
        | r => r
      | r => r
    | .ite c a b =>
      match eval P n ρ c with
      | .ok (.bool true) => eval P n ρ a
      | .ok (.bool false) => eval P n ρ b
      | .ok _ => .stuck
      | r => r
    | .let_ a b =>
      match eval P n ρ a with
      | .ok x => eval P n (x :: ρ) b
      | r => r
    | .pair a b =>
      match eval P n ρ a with
      | .ok x => match eval P n ρ b with
        | .ok y => .ok (.pair x y)
        | r => r
      | r => r
    | .fst a => match eval P n ρ a with
      | .ok (.pair x _) => .ok x
      | .ok _ => .stuck
      | r => r
    | .snd a => match eval P n ρ a with
      | .ok (.pair _ y) => .ok y
      | .ok _ => .stuck
      | r => r
    | .inl a => match eval P n ρ a with
      | .ok x => .ok (.inl x)
      | r => r
    | .inr a => match eval P n ρ a with
      | .ok x => .ok (.inr x)
      | r => r
    | .case s l r =>
      match eval P n ρ s with
      | .ok (.inl x) => eval P n (x :: ρ) l
      | .ok (.inr x) => eval P n (x :: ρ) r
      | .ok _ => .stuck
      | q => q
    | .raise c => .err (.user c)
    | .catch_ a h =>
      match eval P n ρ a with
      | .err (.user _) => eval P n ρ h
      | r => r
    | .call f a =>
      match eval P n ρ a with
      | .ok x => match P.get? f with
        | some (_, _, body) => eval P n [x] (.tick body)
        | none => .stuck
      | r => r
    | .tick e => eval P n ρ e
    | .try_ e c => match eval P n ρ e with
      | .ok (.inr v) => .ok v
      | .ok (.inl _) => .err (.user c)
      | .ok _ => .stuck
      | r => r
    | .cast lo hi e => match eval P n ρ e with
      | .ok (.int x) => chk lo hi x
      | .ok (.f64 b) => chk lo hi (RCE.F64.trunc b)
      | .ok _ => .stuck
      | r => r
    | .tof e => match eval P n ρ e with
      | .ok (.int x) => fres (RCE.F64.ofInt x)
      | .ok _ => .stuck
      | r => r
    | .un op ta e => match eval P n ρ e with
      | .ok x => unop op ta x
      | r => r

/-! Static semantics. -/

inductive OpTy : BinOp → Ty → Ty → Prop where
  | add {lo hi} : OpTy .add (.int lo hi) (.int lo hi)
  | sub {lo hi} : OpTy .sub (.int lo hi) (.int lo hi)
  | mul {lo hi} : OpTy .mul (.int lo hi) (.int lo hi)
  | div {lo hi} : OpTy .div (.int lo hi) (.int lo hi)
  | rem {lo hi} : OpTy .rem (.int lo hi) (.int lo hi)
  | band {lo hi} : OpTy .band (.int lo hi) (.int lo hi)
  | bor {lo hi} : OpTy .bor (.int lo hi) (.int lo hi)
  | bxor {lo hi} : OpTy .bxor (.int lo hi) (.int lo hi)
  | shl {lo hi} : OpTy .shl (.int lo hi) (.int lo hi)
  | shr {lo hi} : OpTy .shr (.int lo hi) (.int lo hi)
  | min {lo hi} : OpTy .min (.int lo hi) (.int lo hi)
  | max {lo hi} : OpTy .max (.int lo hi) (.int lo hi)
  | sadd {lo hi} : OpTy .sadd (.int lo hi) (.int lo hi)
  | ssub {lo hi} : OpTy .ssub (.int lo hi) (.int lo hi)
  | smul {lo hi} : OpTy .smul (.int lo hi) (.int lo hi)
  | wadd {lo hi} : OpTy .wadd (.int lo hi) (.int lo hi)
  | wsub {lo hi} : OpTy .wsub (.int lo hi) (.int lo hi)
  | wmul {lo hi} : OpTy .wmul (.int lo hi) (.int lo hi)
  | f32add : OpTy .add .f32 .f32
  | f32sub : OpTy .sub .f32 .f32
  | f32mul : OpTy .mul .f32 .f32
  | f32div : OpTy .div .f32 .f32
  | f32lt : OpTy .lt .f32 .bool
  | fmin : OpTy .min .f64 .f64
  | fmax : OpTy .max .f64 .f64
  | fadd : OpTy .add .f64 .f64
  | fsub : OpTy .sub .f64 .f64
  | fmul : OpTy .mul .f64 .f64
  | fdiv : OpTy .div .f64 .f64
  | flt : OpTy .lt .f64 .bool
  | lt {lo hi} : OpTy .lt (.int lo hi) .bool
  | eq {t} : OpTy .eq t .bool
  | and : OpTy .and .bool .bool
  | or : OpTy .or .bool .bool
  | cat : OpTy .cat .text .text

inductive UnTy : UnOp → Ty → Ty → Prop where
  | iabs {lo hi} : UnTy .iabs (.int lo hi) (.int lo hi)
  | fabs : UnTy .fabs .f64 .f64
  | sqrt : UnTy .sqrt .f64 .f64
  | floor : UnTy .floor .f64 .f64
  | ceil : UnTy .ceil .f64 .f64
  | round : UnTy .round .f64 .f64
  | ineg {lo hi lo' hi'} : UnTy (.ineg lo hi) (.int lo' hi') (.int lo hi)
  | fneg : UnTy .fneg .f64 .f64
  | fneg32 : UnTy .fneg .f32 .f32
  | fabs32 : UnTy .fabs .f32 .f32
  | tof32 {lo hi} : UnTy .tof32 (.int lo hi) .f32
  | to32 : UnTy .tof32 .f64 .f32
  | widen : UnTy .widen .f32 .f64
  | trunc32 {lo hi} : UnTy (.trunc32 lo hi) .f32 (.int lo hi)

abbrev Sig := List (Ty × Ty)

def sig (P : Prog) : Sig := P.map (fun d => (d.1, d.2.1))

inductive HasTy (S : Sig) : List Ty → Expr → Ty → Prop where
  | var {Γ i t} : Γ.get? i = some t → HasTy S Γ (.var i) t
  | lit {Γ v t} : VTy v t → HasTy S Γ (.lit v) t
  | bin {Γ op ta a b t} : OpTy op ta t → HasTy S Γ a ta → HasTy S Γ b ta →
      HasTy S Γ (.bin op ta a b) t
  | ite {Γ c a b t} : HasTy S Γ c .bool → HasTy S Γ a t → HasTy S Γ b t →
      HasTy S Γ (.ite c a b) t
  | let_ {Γ a b ta t} : HasTy S Γ a ta → HasTy S (ta :: Γ) b t → HasTy S Γ (.let_ a b) t
  | pair {Γ a b ta tb} : HasTy S Γ a ta → HasTy S Γ b tb → HasTy S Γ (.pair a b) (.prod ta tb)
  | fst {Γ e ta tb} : HasTy S Γ e (.prod ta tb) → HasTy S Γ (.fst e) ta
  | snd {Γ e ta tb} : HasTy S Γ e (.prod ta tb) → HasTy S Γ (.snd e) tb
  | inl {Γ e ta tb} : HasTy S Γ e ta → HasTy S Γ (.inl e) (.sum ta tb)
  | inr {Γ e ta tb} : HasTy S Γ e tb → HasTy S Γ (.inr e) (.sum ta tb)
  | case {Γ s l r ta tb t} : HasTy S Γ s (.sum ta tb) → HasTy S (ta :: Γ) l t →
      HasTy S (tb :: Γ) r t → HasTy S Γ (.case s l r) t
  | raise {Γ c t} : HasTy S Γ (.raise c) t
  | catch_ {Γ a h t} : HasTy S Γ a t → HasTy S Γ h t → HasTy S Γ (.catch_ a h) t
  | call {Γ f a ta t} : S.get? f = some (ta, t) → HasTy S Γ a ta → HasTy S Γ (.call f a) t
  | tick {Γ e t} : HasTy S Γ e t → HasTy S Γ (.tick e) t
  | try_ {Γ e ta t c} : HasTy S Γ e (.sum ta t) → HasTy S Γ (.try_ e c) t
  | cast {Γ e lo hi lo' hi'} : HasTy S Γ e (.int lo' hi') → HasTy S Γ (.cast lo hi e) (.int lo hi)
  | fcast {Γ e lo hi} : HasTy S Γ e .f64 → HasTy S Γ (.cast lo hi e) (.int lo hi)
  | tof {Γ e lo hi} : HasTy S Γ e (.int lo hi) → HasTy S Γ (.tof e) .f64
  | un {Γ op ta t e} : UnTy op ta t → HasTy S Γ e ta → HasTy S Γ (.un op ta e) t

def WtProg (P : Prog) : Prop :=
  ∀ f ta t b, P.get? f = some (ta, t, b) → HasTy (sig P) [ta] b t

/-- The environment agrees with the context pointwise. -/
inductive WtEnv : List Ty → List Val → Prop where
  | nil : WtEnv [] []
  | cons {t v Γ ρ} : VTy v t → WtEnv Γ ρ → WtEnv (t :: Γ) (v :: ρ)

/-- A result agrees with a type: never stuck, and a value has the type. -/
def RTy : Res → Ty → Prop
  | .ok v, t => VTy v t
  | .err _, _ => True
  | .oof, _ => True
  | .stuck, _ => False

end RCE.Core
