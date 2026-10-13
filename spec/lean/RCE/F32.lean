/-
  f32 determinism (spec 6.3 N7, draft 0.5).
  Form computes an f32 operation by widening the operands to f64, applying
  the f64 operation, and rounding the result once to f32.
  Model: a binary float is a pair (m, e) meaning m·2^e; precision p means
  |m| < 2^p. A rounding to precision p is any function that leaves values
  already representable in precision p unchanged (round-to-nearest-even is
  one; the result below holds for every such function).
  Theorems:
  * `widen_exact`: widening an f32 to f64 is exact.
  * `f32_mul_det`: for multiplication the f64 step is exact (24 + 24 ≤ 53), so
    the Form result equals the f32 rounding of the exact product: a single
    rounding, the same on every target.
  * `f32_op_det`: for every op, the Form result is a function of the two f32
    inputs alone, so targets that implement the f64 op and the f32 rounding
    per IEEE 754 produce identical bits.
  For + − / the stronger statement (f64-then-f32 equals direct f32
  rounding) is Figueroa's theorem (p' ≥ 2p + 2: 53 ≥ 50); it is cited in the
  spec, not proved here.
-/
namespace RCE.F32

structure Fl where
  m : Int
  e : Int
deriving DecidableEq

def repr (p : Nat) (x : Fl) : Prop := x.m.natAbs < 2 ^ p

/-- A rounding to precision `p`: anything that fixes representable values. -/
structure Rounding (p : Nat) where
  f : Fl → Fl
  fixes : ∀ x, repr p x → f x = x

def mulExact (x y : Fl) : Fl := ⟨x.m * y.m, x.e + y.e⟩

theorem widen_exact (r64 : Rounding 53) (x : Fl) (hx : repr 24 x) : r64.f x = x := by
  apply r64.fixes
  unfold repr at *
  have : (2:Nat) ^ 24 ≤ 2 ^ 53 := Nat.pow_le_pow_right (by decide) (by decide)
  omega

theorem mul_repr (x y : Fl) (hx : repr 24 x) (hy : repr 24 y) : repr 53 (mulExact x y) := by
  unfold repr mulExact at *
  simp only [Int.natAbs_mul]
  have h1 : x.m.natAbs * y.m.natAbs < 2 ^ 24 * 2 ^ 24 := Nat.mul_lt_mul_of_lt_of_le hx (Nat.le_of_lt hy) (by omega)
  omega

/-- The Form f32 product. -/
def f32mul (r32 : Rounding 24) (r64 : Rounding 53) (x y : Fl) : Fl :=
  r32.f (r64.f (mulExact (r64.f x) (r64.f y)))

theorem f32_mul_det (r32 : Rounding 24) (r64 : Rounding 53) (x y : Fl)
    (hx : repr 24 x) (hy : repr 24 y) :
    f32mul r32 r64 x y = r32.f (mulExact x y) := by
  unfold f32mul
  rw [widen_exact r64 x hx, widen_exact r64 y hy, r64.fixes _ (mul_repr x y hx hy)]

/-- Any f32 op in Form: widen, apply the f64 op, round once. -/
def f32op (r32 : Rounding 24) (r64 : Rounding 53) (op64 : Fl → Fl → Fl) (x y : Fl) : Fl :=
  r32.f (r64.f (op64 (r64.f x) (r64.f y)))

/-- Two targets that agree on the f64 operation and on the f32 rounding
    (both fixed by IEEE 754) agree on every f32 result, and the widening of
    the inputs never changes them. -/
theorem f32_op_det (r32 r32' : Rounding 24) (r64 r64' : Rounding 53) (op64 : Fl → Fl → Fl)
    (h32 : ∀ z, r32.f z = r32'.f z) (h64 : ∀ z, r64.f z = r64'.f z) (x y : Fl)
    (hx : repr 24 x) (hy : repr 24 y) :
    f32op r32 r64 op64 x y = f32op r32' r64' op64 x y := by
  unfold f32op
  rw [widen_exact r64 x hx, widen_exact r64 y hy, widen_exact r64' x hx, widen_exact r64' y hy, h64, h32]

/-- The result of an f32 op is again an f32 value (rounding output is representable). -/
theorem f32op_closed (r32 : Rounding 24) (r64 : Rounding 53) (op64 : Fl → Fl → Fl)
    (hr : ∀ z, repr 24 (r32.f z)) (x y : Fl) : repr 24 (f32op r32 r64 op64 x y) := hr _

end RCE.F32
