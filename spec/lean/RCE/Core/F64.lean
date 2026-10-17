/-! S3: IEEE 754 binary64 on bit patterns, by exact arithmetic. A float is
    its 64-bit pattern as a `Nat`. Each operation computes the exact real
    result as a fraction and rounds it once to nearest, ties to even, so
    the result is the one IEEE 754-2019 defines. Operations whose result
    is not finite give `none`; RCE traps on them ("non-finite float",
    book 6.3 N3). Display is the shortest decimal that rounds back to the
    same bits, spelled as the reference's `float_text`. All definitions
    use structural recursion only, so `decide` can check vectors. -/
namespace RCE.F64

/-- `⌊log2 n⌋` for `0 < n < 2^6464`: whole 64-bit words first, then bits
    (structural, with small fuel, so the kernel reduces it). -/
def lgBits : Nat → Nat → Nat → Nat
  | _, 0, acc => acc
  | n, f + 1, acc => if n < 2 then acc else lgBits (n / 2) f (acc + 1)

def lgWords : Nat → Nat → Nat → Nat
  | n, 0, acc => lgBits n 64 acc
  | n, f + 1, acc => if n < 2 ^ 64 then lgBits n 64 acc else lgWords (n / 2 ^ 64) f (acc + 64)

def lg (n : Nat) : Nat := lgWords n 100 0

def p2 (k : Nat) : Nat := 2 ^ k

/-- sign, integer significand, exponent: the value is `±m·2^e`. -/
def decode (b : Nat) : Bool × Nat × Int :=
  let s := decide (b ≥ 2 ^ 63)
  let ex : Nat := (b / 2 ^ 52) % 2048
  let fr : Nat := b % 2 ^ 52
  if ex = 0 then (s, fr, -1074) else (s, fr + 2 ^ 52, (ex : Int) - 1075)

def isFinite (b : Nat) : Bool := (b / 2 ^ 52) % 2048 != 2047

def withSign (s : Bool) (b : Nat) : Nat := if s then b + 2 ^ 63 else b

/-- `N/D·2^k` scaled to the integer fraction `num/den` at unit `2^E`. -/
def scaleAt (N D : Nat) (k E : Int) : Nat × Nat :=
  if k - E ≥ 0 then (N * 2 ^ (k - E).toNat, D) else (N, D * 2 ^ (E - k).toNat)

/-- `⌊log2 (N/D)⌋` for positive `N`, `D`. -/
def flog2 (N D : Nat) : Int :=
  let j : Int := (lg N : Int) - lg D
  -- 2^j ≤ N/D < 2^(j+1) or 2^(j-1) ≤ N/D < 2^j
  let ge : Bool := if j ≥ 0 then decide (N ≥ D * 2 ^ j.toNat) else decide (N * 2 ^ (-j).toNat ≥ D)
  if ge then j else j - 1

/-- Nearest integer to `num/den`, ties to even. -/
def rq (num den : Nat) : Nat :=
  if 2 * (num % den) > den ∨ (2 * (num % den) = den ∧ (num / den) % 2 = 1)
  then num / den + 1 else num / den

/-- `|a - b|` on naturals. -/
def ndist (a b : Nat) : Nat := if b ≤ a then a - b else b - a

/-- Exponent bias: every exponent the operations produce is above `-bias`,
    so the rounding works with natural exponents only. -/
def bias : Nat := 2200

/-- Round the positive exact value `N/D·2^k` to a nonnegative binary64
    pattern, ties to even; `none` on overflow. With `A = N·2^(k+bias)`
    the value is `A/(D·2^bias)`; `L = ⌊log2 (A/D)⌋` is `⌊log2 x⌋ + bias`,
    the grid exponent is `Eb - bias` with `Eb = max (L - 52) 1126`
    (1126 = bias - 1074, the subnormal unit), and the significand is the
    nearest integer to `A/(D·2^Eb)`. -/
def round (N D : Nat) (k : Int) : Option Nat :=
  let A := N * 2 ^ (k + bias).toNat
  let Eb := max (lg (A / D) - 52) 1126
  let q := rq A (D * 2 ^ Eb)
  if q < 2 ^ 52 then some q
  else if q = 2 ^ 53 then
    (if Eb + 1 - 1125 ≥ 2047 then none else some ((Eb + 1 - 1125) * 2 ^ 52))
  else if Eb - 1125 ≥ 2047 then none else some ((Eb - 1125) * 2 ^ 52 + (q - 2 ^ 52))

/-- The signed exact value as an integer at a common unit `2^u`. -/
def atUnit (s : Bool) (m : Nat) (e u : Int) : Int :=
  let v : Int := m * 2 ^ (e - u).toNat
  if s then -v else v

def add (x y : Nat) : Option Nat :=
  let (s1, m1, e1) := decode x
  let (s2, m2, e2) := decode y
  let u := min e1 e2
  let S := atUnit s1 m1 e1 u + atUnit s2 m2 e2 u
  if S = 0 then some (withSign (s1 && s2) 0)
  else (round S.natAbs 1 u).map (withSign (decide (S < 0)))

/-- Truncation toward zero of a finite pattern (reference `f.trunc() as i128`). -/
def trunc (x : Nat) : Int :=
  let (s, m, e) := decode x
  let a : Nat := if e ≥ 0 then m * 2 ^ e.toNat else m / 2 ^ (-e).toNat
  if s then -(a : Int) else a

/-- The pattern nearest the integer `x` (reference `v as f64`). -/
def ofInt (x : Int) : Option Nat :=
  if x = 0 then some 0 else (round x.natAbs 1 0).map (withSign (decide (x < 0)))

def neg (x : Nat) : Nat := if x ≥ 2 ^ 63 then x - 2 ^ 63 else x + 2 ^ 63

def sub (x y : Nat) : Option Nat := add x (neg y)

def mul (x y : Nat) : Option Nat :=
  let (s1, m1, e1) := decode x
  let (s2, m2, e2) := decode y
  let s := s1 != s2
  if m1 * m2 = 0 then some (withSign s 0) else (round (m1 * m2) 1 (e1 + e2)).map (withSign s)

def div (x y : Nat) : Option Nat :=
  let (s1, m1, e1) := decode x
  let (s2, m2, e2) := decode y
  let s := s1 != s2
  if m2 = 0 then none
  else if m1 = 0 then some (withSign s 0)
  else (round m1 m2 (e1 - e2)).map (withSign s)

/-- `|x|`: clear the sign bit (exact). -/
def abs (x : Nat) : Nat := if x ≥ 2 ^ 63 then x - 2 ^ 63 else x

/-- Integer rounding modes of floor, ceil and round (half away from zero,
    Rust `f64::round`); the result keeps the sign of `x`, so it is `-0`
    for a negative `x` that rounds to zero. -/
inductive RMode where | floor | ceil | away
  deriving DecidableEq, Repr

def toIntegral (md : RMode) (x : Nat) : Option Nat :=
  let (s, m, e) := decode x
  if e ≥ 0 then some x else
  let k := (-e).toNat
  let q := m / 2 ^ k
  let r := m % 2 ^ k
  let up : Bool := match md with
    | .floor => s && r != 0
    | .ceil => !s && r != 0
    | .away => decide (2 * r ≥ 2 ^ k)
  let mag := if up then q + 1 else q
  if mag = 0 then some (withSign s 0) else (round mag 1 0).map (withSign s)

/-- `⌊√n⌋` bit by bit (structural, 128 steps cover `n < 2^256`). -/
def isqrtGo : Nat → Nat → Nat → Nat
  | 0, _, acc => acc
  | i + 1, n, acc =>
    let c := acc + 2 ^ i
    isqrtGo i n (if c * c ≤ n then c else acc)

def isqrt (n : Nat) : Nat := isqrtGo 128 n 0

/-- Correctly rounded square root; a negative nonzero operand has no real
    root (`none`, the non-finite trap); `√-0 = -0`. The significand is
    scaled by `4^60` so `q = ⌊√M⌋` has at least 60 bits; `2q + sticky`
    at unit `2^(e/2 - 61)` lies strictly between the same rounding
    boundaries as the exact root, so one rounding is exact (IEEE 754 5.4.1). -/
def sqrt (x : Nat) : Option Nat :=
  let (s, m, e) := decode x
  if m = 0 then some x
  else if s then none
  else
    let (M, E) : Nat × Int := if e % 2 = 0 then (m, e) else (2 * m, e - 1)
    let A := M * 2 ^ 120
    let q := isqrt A
    let st := if q * q = A then 0 else 1
    round (2 * q + st) 1 (E / 2 - 61)

/-- Three-way comparison of the values (`-0 = 0`). -/
def cmp (x y : Nat) : Ordering :=
  let (s1, m1, e1) := decode x
  let (s2, m2, e2) := decode y
  let u := min e1 e2
  compare (atUnit s1 m1 e1 u) (atUnit s2 m2 e2 u)

/-- IEEE 754 totalOrder on finite patterns (`-0 < +0`), the float leg of
    `cmp_canon` (checker interp.rs, `f64::total_cmp`). -/
def totalCmp (x y : Nat) : Ordering :=
  match cmp x y with
  | .eq => compare (decide (x < 2 ^ 63)) (decide (y < 2 ^ 63))
  | o => o

def lt (x y : Nat) : Bool := cmp x y == .lt
def eq (x y : Nat) : Bool := cmp x y == .eq

/-! Display. -/

def p10 (k : Nat) : Nat := 10 ^ k

/-- `N/D ≥ 10^j`. -/
def ge10 (N D : Nat) (j : Int) : Bool :=
  if j ≥ 0 then decide (N ≥ D * 10 ^ j.toNat) else decide (N * 10 ^ (-j).toNat ≥ D)

/-- `⌊log10 (N/D)⌋` for positive `N/D` (search from a binary estimate). -/
def flog10 (N D : Nat) : Int :=
  let est : Int := ((flog2 N D) * 30103) / 100000
  let j := (List.range 4).foldl (fun j _ => if ge10 N D (j + 1) then j + 1 else j) (est - 2)
  j

/-- Round `N/D / 10^j` to an integer, ties to even. -/
def roundDec (N D : Nat) (j : Int) : Nat :=
  let (num, den) := if j ≥ 0 then (N, D * 10 ^ j.toNat) else (N * 10 ^ (-j).toNat, D)
  let q := num / den
  let r := num % den
  if 2 * r > den ∨ (2 * r = den ∧ q % 2 = 1) then q + 1 else q

/-- The pattern nearest `d·10^j`. -/
def ofDec (d : Nat) (j : Int) : Option Nat :=
  if d = 0 then some 0 else if j ≥ 0 then round (d * 10 ^ j.toNat) 1 0 else round d (10 ^ (-j).toNat) 0

/-- Shortest digits `d` and exponent `E10` (value `d·10^(E10-len+1)`) of a
    positive finite pattern. -/
def shortest (b : Nat) : Nat × Int :=
  let (_, m, e) := decode b
  let (N, D) := if e ≥ 0 then (m * 2 ^ e.toNat, 1) else (m, 2 ^ (-e).toNat)
  let E := flog10 N D
  let pick := (List.range 17).findSome? fun (i : Nat) =>
    let p : Int := (i : Int) + 1
    let d := roundDec N D (E - p + 1)
    if ofDec d (E - p + 1) = some b then some (d, p) else none
  match pick with
  | some (d, p) =>
    -- a carry (d = 10^p) moves the exponent
    if d = 10 ^ p.toNat then (d / 10, E + 1) else (d, E)
  | none => (roundDec N D (E - 16), E)

def digits (n : Nat) : String := toString n

def stripZeros (s : String) : String := (s.toList.reverse.dropWhile (· == '0')).reverse.asString

/-- The reference's `float_text` (checker interp.rs): `0`/`-0`; plain
    decimal for magnitudes in [1e-5, 1e16); otherwise `d.ddde±X`. -/
def text (b : Nat) : String :=
  let neg := b ≥ 2 ^ 63
  let a := if neg then b - 2 ^ 63 else b
  let sgn := if neg then "-" else ""
  if a = 0 then sgn ++ "0" else
  let (d, E) := shortest a
  let ds := stripZeros (digits d)
  let ds := if ds.isEmpty then "0" else ds
  let n := ds.length
  let plain := ge10Pat a (-5) && !ge10Pat a 16
  if plain then
    if E ≥ 0 then
      let ip := E.toNat + 1
      if n ≤ ip then sgn ++ ds ++ String.mk (List.replicate (ip - n) '0')
      else sgn ++ (ds.take ip) ++ "." ++ (ds.drop ip)
    else sgn ++ "0." ++ String.mk (List.replicate ((-E).toNat - 1) '0') ++ ds
  else
    let mant := if n = 1 then ds else ds.take 1 ++ "." ++ ds.drop 1
    sgn ++ mant ++ "e" ++ (if E ≥ 0 then "+" else "") ++ toString E
where
  ge10Pat (a : Nat) (j : Int) : Bool :=
    let (_, m, e) := decode a
    let (N, D) := if e ≥ 0 then (m * 2 ^ e.toNat, 1) else (m, 2 ^ (-e).toNat)
    ge10 N D j

/-! Vectors (checked by the kernel). -/
set_option maxRecDepth 200000
example : add 0x3FB999999999999A 0x3FC999999999999A = some 0x3FD3333333333334 := by decide
example : text 0x3FD3333333333334 = "0.30000000000000004" := by decide
example : text 0x8000000000000000 = "-0" := by decide
example : text 0x4415AF1D78B58C40 = "1e+20" := by decide
example : text 1 = "5e-324" := by decide
example : div 0x3FF0000000000000 0 = none := by decide
example : mul 0x7FEFFFFFFFFFFFFF 0x4000000000000000 = none := by decide
example : (div 0x3FF0000000000000 0x4008000000000000).map text = some "0.3333333333333333" := by decide
example : totalCmp 0x8000000000000000 0 = .lt := by decide
example : eq 0x8000000000000000 0 = true := by decide

end RCE.F64
