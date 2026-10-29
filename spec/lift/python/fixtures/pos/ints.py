# SPDX-License-Identifier: MIT OR Apache-2.0
# Integer arithmetic of the lift: floor division and modulo for every sign,
# powers, shifts and bitwise operators, and recursion.
def gcd(a, b):
    if b == 0:
        return abs(a)
    return gcd(b, a % b)


def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)


for a, b in [(7, 2), (-7, 2), (7, -2), (-7, -2), (0, 5)]:
    print(f"{a} // {b} = {a // b}, {a} % {b} = {a % b}", divmod(a, b))
print(2 ** 10, 3 ** 0, -2 ** 2, (-2) ** 3, 1 << 40, -1 >> 3, 255 >> 70, 6 & 3, 6 | 3, 6 ^ 3, ~5)
print(gcd(1071, 462), gcd(-24, 36), [fact(i) for i in range(0, 21, 5)])
print(pow(3, 200, 1000003), abs(-9), max(3, 7, 5), min(3, 7, 5), sum([1, 2, 3, 4]))
