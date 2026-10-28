# SPDX-License-Identifier: MIT OR Apache-2.0
# Fibonacci numbers as unbounded integers, past the 64-bit range, and the
# same sums reduced modulo 2^64 to show where a 64-bit type would wrap.
a, b = 0, 1
for i in range(201):
    if i >= 88 and (i <= 100 or i % 25 == 0):
        print(i, a, a % 2 ** 64, a.bit_length())
    a, b = b, a + b
