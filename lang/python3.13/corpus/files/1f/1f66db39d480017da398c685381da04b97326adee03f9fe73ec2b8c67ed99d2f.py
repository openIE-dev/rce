# SPDX-License-Identifier: MIT OR Apache-2.0
# Comparison of numbers, strings, sequences and mixed types; chained
# comparisons.
print(3 < 5, 5 <= 5, 7 > 7, -1 >= -2, 4 == 4, 4 != 4, 1 == 1.0, None is not False)
nan = float("nan")
print(nan == nan, nan < 1, nan > 1, nan != nan, [nan] == [nan])
print(0.0 == -0.0, 2 ** 53 == 2.0 ** 53, 2 ** 53 + 1 == 2.0 ** 53 + 1, 2 ** 53 + 1 == float(2 ** 53 + 1))
print("apple" < "banana", "apple" < "Apple", "a" < "aa", "" < "a", "Z" < "a")
print([1, 2, 3] < [1, 2, 4], [1, 2] < [1, 2, 0], (1, "a") < (1, "b"), [3] > [2, 99])
print(1 < 2 < 3, 3 > 2 > 1 > 0, 1 < 3 < 2, (1 < 3) < 2)
print({1, 2} < {1, 2, 3}, {1, 2} <= {1, 2}, {1: "a"} == {1: "a"}, None == 0, True == 1, False < True)
try:
    print(1 < "a")
except TypeError as e:
    print("TypeError", e)
try:
    print(None < None)
except TypeError as e:
    print("TypeError", e)
