# SPDX-License-Identifier: MIT OR Apache-2.0
# List, set, dict and generator comprehensions, with nested loops and
# conditions.
print([x * x for x in range(6)])
print([x for x in range(20) if x % 3 == 0 if x % 2 == 0])
print([(x, y) for x in range(3) for y in range(x)])
print([[r * c for c in range(1, 4)] for r in range(1, 4)])
print({x % 4 for x in range(10)})
print({w: len(w) for w in ["apple", "fig", "kiwi"]})
print(sum(x * x for x in range(101)), any(x > 5 for x in [1, 9]), all(x > 0 for x in [1, -1]))
print([c.upper() if c in "aeiou" else c for c in "banana"])
matrix = [[1, 2, 3], [4, 5, 6]]
print([list(col) for col in zip(*matrix)], [v for row in matrix for v in row])
g = (x * 2 for x in range(3))
print(next(g), list(g), list(g))
