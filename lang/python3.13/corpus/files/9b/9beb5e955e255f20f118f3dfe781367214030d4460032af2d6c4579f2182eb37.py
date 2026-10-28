# SPDX-License-Identifier: MIT OR Apache-2.0
# The number of ways to place n non-attacking queens, for n from 1 to 9,
# and the first solution for n = 8.


def solve(n):
    cols, d1, d2 = set(), set(), set()
    pos = [0] * n
    first = []
    count = 0

    def place(row):
        nonlocal count
        if row == n:
            if not first:
                first.extend(pos)
            count += 1
            return
        for c in range(n):
            if c in cols or row + c in d1 or row - c in d2:
                continue
            cols.add(c)
            d1.add(row + c)
            d2.add(row - c)
            pos[row] = c
            place(row + 1)
            cols.discard(c)
            d1.discard(row + c)
            d2.discard(row - c)

    place(0)
    return count, first


for n in range(1, 10):
    k, first = solve(n)
    print(n, k)
    if n == 8:
        print(first)
