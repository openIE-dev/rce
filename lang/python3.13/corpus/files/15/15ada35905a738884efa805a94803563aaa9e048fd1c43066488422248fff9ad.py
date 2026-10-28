# SPDX-License-Identifier: MIT OR Apache-2.0
# Operations on infinities and NaN, and conversions that raise.
import math

inf, nan = math.inf, math.nan
print(inf - inf, inf * 0, inf + 1, -inf < inf, math.isnan(inf - inf), nan != nan)
print(max(nan, 1), max(1, nan), min(-0.0, 0.0), min(0.0, -0.0), sorted([3, nan, 1]))
print(math.sqrt(4), math.copysign(0.0, -1), 1e308 * 10, -1e308 * 10, 2.0 ** 1023 * 2)
for f in (lambda: int(nan), lambda: int(inf), lambda: math.sqrt(-1), lambda: math.log(0), lambda: 2.0 ** 10000,
          lambda: math.exp(1000), lambda: 1.0 / 0, lambda: 0.0 % 0.0, lambda: float(10 ** 400), lambda: (-8) ** (1 / 3)):
    try:
        print(repr(f()))
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        print(type(e).__name__, e)
print(int(1e300) == 10 ** 300, int(1e22) == 10 ** 22, int(-2.9), int(2.9), math.floor(-0.5), math.ceil(-0.5))
print(math.isfinite(1e308), math.isinf(-inf), math.isclose(0.1 + 0.2, 0.3), math.ulp(1.0), math.nextafter(1.0, 2.0))
