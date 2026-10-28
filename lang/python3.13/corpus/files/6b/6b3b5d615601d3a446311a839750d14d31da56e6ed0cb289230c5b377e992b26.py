# SPDX-License-Identifier: MIT OR Apache-2.0
# Euclid's greatest common divisor, the least common multiple, and the
# extended algorithm's Bezout coefficients, checked against math.gcd.
import math


def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def egcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y


for a, b in [(12, 18), (17, 5), (0, 9), (-24, 36), (1071, 462), (2 ** 61 - 1, 2 ** 31 - 1), (10 ** 30, 10 ** 20 + 10)]:
    g = gcd(a, b)
    lcm = a // g * b if g else 0
    print(a, b, g, lcm, g == math.gcd(a, b))
g, x, y = egcd(240, 46)
print(g, x, y, 240 * x + 46 * y)
acc = 1
for i in range(1, 41):
    acc = acc // gcd(acc, i) * i
print(acc, acc == math.lcm(*range(1, 41)))
