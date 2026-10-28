# SPDX-License-Identifier: MIT OR Apache-2.0
# Comparisons between int and float are exact: an int is not converted to
# float first.
n = 2 ** 53 + 1
f = float(2 ** 53)
print(n == f, n > f, n - 1 == f, float(n) == f, int(f) == n - 1)
print(10 ** 400 > 1e308, 10 ** 400 < float("inf"), -(10 ** 400) > float("-inf"))
print(2 ** 63 == 2.0 ** 63, 2 ** 64 - 1 == 2.0 ** 64, 2 ** 64 - 1 < 2.0 ** 64)
print(1 / 3 == 0.3333333333333333, 0.1 * 3 == 0.3, 3 * 0.1 > 0.3)
print(hash(2 ** 53 + 1) == hash(float(2 ** 53 + 1)), {2 ** 53: "a", float(2 ** 53): "b"})
print(sorted([2 ** 53 + 1, float(2 ** 53), 2 ** 53]), max(1, 1.0), max(1.0, 1))
