# SPDX-License-Identifier: MIT OR Apache-2.0
# Transcendental functions are not required to be correctly rounded;
# CPython computes most of them with the C library it loads, which the
# oracle record pins by digest.
import math

for x in [0.5, 1.0, 2.0, 10.0, 1e-8, 100.0]:
    print(x, repr(math.sin(x)), repr(math.cos(x)), repr(math.tan(x)), repr(math.exp(x)), repr(math.log(x)), repr(math.atan(x)))
print(repr(math.log10(1000)), repr(math.log2(1024)), repr(math.log(8, 2)), repr(math.pow(2, 0.5)), repr(2 ** 0.5), repr(math.cbrt(27)))
print(repr(math.erf(0.5)), repr(math.gamma(0.5)), repr(math.lgamma(10)), repr(math.sinh(1)), repr(math.atan2(1, -1)))
