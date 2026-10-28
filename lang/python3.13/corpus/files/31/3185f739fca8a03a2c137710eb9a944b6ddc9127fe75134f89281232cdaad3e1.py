# SPDX-License-Identifier: MIT OR Apache-2.0
# Integer matrix multiplication, a large power, and the determinant by
# cofactor expansion and by exact fractions.
from fractions import Fraction


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def det(a):
    if len(a) == 1:
        return a[0][0]
    return sum((-1) ** c * a[0][c] * det([row[:c] + row[c + 1:] for row in a[1:]]) for c in range(len(a)))


def det_frac(a):
    m = [[Fraction(x) for x in row] for row in a]
    n, d = len(m), Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if m[r][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            m[c], m[p] = m[p], m[c]
            d = -d
        d *= m[c][c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return d


print(mul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]))
r = [[1, 0], [0, 1]]
for _ in range(100):
    r = mul(r, [[1, 1], [1, 0]])
print(r)
for m in ([[2, -3, 1], [2, 0, -1], [1, 4, 5]], [[1, 2], [2, 4]], [[6, 1, 1, 3], [4, -2, 5, 1], [2, 8, 7, 6], [3, 1, 9, 7]]):
    print(det(m), det_frac(m))
print(Fraction(1, 3) + Fraction(1, 6), Fraction(0.1), Fraction("3.14"), Fraction(-6, 4))
