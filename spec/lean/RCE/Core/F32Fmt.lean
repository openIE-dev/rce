import RCE.Core.F64
/-! S3 binary32 (book 6.3 N7). An f32 value is held exactly as the binary64
    pattern of the same real (as the reference holds `F32(f64)`). An f32
    operation is the f64 operation (rounded once, `RCE.F64`) rounded again
    to binary32, as the reference computes it; a non-finite result traps. -/
namespace RCE.F64

/-- Round the positive exact value `N/D·2^k` to binary32, ties to even, and
    return the binary64 pattern of the result; `none` on overflow. With
    `A = N·2^(k+bias)`: `Eb = max (⌊log2 (A/D)⌋ - 23) (bias - 149)` and the
    significand is the nearest integer to `A/(D·2^Eb)` (at most `2^24`). -/
def round32 (N D : Nat) (k : Int) : Option Nat :=
  let A := N * 2 ^ (k + bias).toNat
  let Eb := max (lg (A / D) - 23) (bias - 149)
  let q := rq A (D * 2 ^ Eb)
  if q = 0 then some 0
  else if q * 2 ^ Eb ≥ 2 ^ (128 + bias) then none
  else round q 1 ((Eb : Int) - bias)

/-- A binary64 pattern rounded to binary32 (`f as f32`); the sign is kept,
    so an underflow is a signed zero. -/
def to32 (b : Nat) : Option Nat :=
  let (s, m, e) := decode b
  if m = 0 then some b else (round32 m 1 e).map (withSign s)

/-- `x as f32` from an integer: one rounding to binary32. -/
def ofInt32 (x : Int) : Option Nat :=
  if x = 0 then some 0 else (round32 x.natAbs 1 0).map (withSign (decide (x < 0)))

def lift32 : Option Nat → Option Nat
  | some r => to32 r
  | none => none

def add32 (x y : Nat) : Option Nat := lift32 (add x y)
def sub32 (x y : Nat) : Option Nat := lift32 (sub x y)
def mul32 (x y : Nat) : Option Nat := lift32 (mul x y)
def div32 (x y : Nat) : Option Nat := lift32 (div x y)

/-- The binary32 value nearest `d·10^j` (as a binary64 pattern). -/
def ofDec32 (d : Nat) (j : Int) : Option Nat :=
  if d = 0 then some 0 else if j ≥ 0 then round32 (d * 10 ^ j.toNat) 1 0 else round32 d (10 ^ (-j).toNat) 0

/-- Shortest digits that read back to the same binary32 value (closest at
    each length), as Rust's `f32` `Display`. -/
def shortest32 (b : Nat) : Nat × Int :=
  let (_, m, e) := decode b
  let (N, D) := if e ≥ 0 then (m * 2 ^ e.toNat, 1) else (m, 2 ^ (-e).toNat)
  let E := flog10 N D
  let pick := (List.range 9).findSome? fun (i : Nat) =>
    let p : Int := (i : Int) + 1
    let d := roundDec N D (E - p + 1)
    if ofDec32 d (E - p + 1) = some b then some (d, p) else none
  match pick with
  | some (d, p) => if d = 10 ^ p.toNat then (d / 10, E + 1) else (d, E)
  | none => (roundDec N D (E - 8), E)

/-- Rust `f32` `Display`: plain decimal, never an exponent, `-0` kept. -/
def text32 (b : Nat) : String :=
  let neg := b ≥ 2 ^ 63
  let a := if neg then b - 2 ^ 63 else b
  let sgn := if neg then "-" else ""
  if a = 0 then sgn ++ "0" else
  let (d, E) := shortest32 a
  let ds := stripZeros (digits d)
  let ds := if ds.isEmpty then "0" else ds
  let n := ds.length
  if E ≥ 0 then
    let ip := E.toNat + 1
    if n ≤ ip then sgn ++ ds ++ String.mk (List.replicate (ip - n) '0')
    else sgn ++ (ds.take ip) ++ "." ++ (ds.drop ip)
  else sgn ++ "0." ++ String.mk (List.replicate ((-E).toNat - 1) '0') ++ ds

end RCE.F64
