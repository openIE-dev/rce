# SPDX-License-Identifier: MIT OR Apache-2.0
# IEEE 754 binary64 arithmetic on float, and the math module.
import math

a, b = 0.1, 0.2
print(a + b, a + b == 0.3, a * 3, 1 - 0.9)
print(1 / 3, 2 / 3, 1 / 3 * 3)
print(1e300 * 1e10, -1e300 * 1e10, 1e300 * 1e300 / 1e300)
print(math.sqrt(2), math.sqrt(2) ** 2, abs(-2.5), math.fabs(-0.0))
print(math.floor(-2.5), math.ceil(-2.5), math.trunc(-2.5), round(-2.5), round(2.5), round(3.5))
print(math.fmod(7.5, 2), math.fmod(-7.5, 2), 7.5 % 2, -7.5 % 2, math.remainder(7.5, 2))
x = 0.0
for _ in range(10):
    x += 0.1
print(x, x == 1, math.isclose(x, 1), math.fsum([0.1] * 10))
print(math.pi, math.e, math.tau, math.inf, -math.inf, math.nan)
print(math.factorial(20), math.gcd(12, 18), math.lcm(4, 6), math.comb(10, 3), math.perm(5, 2), math.isqrt(10 ** 20 + 1))
print(math.hypot(3, 4), math.copysign(1, -0.0), math.frexp(8.0), math.ldexp(0.5, 4), math.modf(-2.75))
