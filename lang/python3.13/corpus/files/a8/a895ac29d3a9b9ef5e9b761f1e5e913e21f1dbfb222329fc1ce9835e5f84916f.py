# SPDX-License-Identifier: MIT OR Apache-2.0
# sorted and list.sort: keys, reverse, stability and comparison of tuples.
xs = [5, 2, 9, 1, 5, 6, -3, 0]
print(sorted(xs), sorted(xs, reverse=True), xs)
xs.sort()
print(xs)
ws = ["pear", "Apple", "fig", "banana", "apple"]
print(sorted(ws), sorted(ws, key=str.lower), sorted(ws, key=len), sorted(ws, key=lambda w: (len(w), w)))
people = [("Ann", 30), ("Bob", 25), ("Cid", 30), ("Dee", 25), ("Eve", 35)]
print(sorted(people, key=lambda p: p[1]))
print(sorted(people, key=lambda p: p[1], reverse=True))
print(sorted([(2, "b"), (1, "z"), (2, "a"), (1, "a")]))
print(sorted("hello"), sorted({3: "c", 1: "a", 2: "b"}), sorted({3, 1, 2}))
print(sorted([3.5, 1, 2.25, -1, 0]), sorted([True, False, 2, 0]))
nested = [[3, 1], [2], [], [1, 1, 1]]
print(sorted(nested), sorted(nested, key=sum), max(nested, key=len), min(nested))
