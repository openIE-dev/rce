# SPDX-License-Identifier: MIT OR Apache-2.0
# Sets have no defined order; CPython's iteration order for small ints
# follows its hash table layout, which the pinned oracle fixes.
s = {5, 3, 1, 100, 9, 33, 64, 0, -1}
print(s, sorted(s), len(s))
print({1, 9, 17}, {17, 9, 1}, set(range(10, 0, -1)))
t = set()
for x in [40, 8, 16, 32, 24]:
    t.add(x)
print(t, list(t))
print({1, 2, 3} | {3, 4}, {1, 2, 3} & {2, 3, 4}, {1, 2, 3} - {2}, {1, 2, 3} ^ {3, 4})
print(frozenset([3, 1, 2]), sorted({"b", "a", "c"}), 2 in {1, 2}, {(1, 2), (0, 1)} == {(0, 1), (1, 2)})
