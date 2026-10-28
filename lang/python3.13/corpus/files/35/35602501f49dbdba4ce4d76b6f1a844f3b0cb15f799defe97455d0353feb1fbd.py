# SPDX-License-Identifier: MIT OR Apache-2.0
# Dictionaries: insertion, lookup, get, setdefault, pop, deletion, views
# and comprehensions.
ages = {"ann": 31, "bob": 27}
ages["cid"] = 45
ages["ann"] += 1
print(ages, len(ages))
print(ages.get("dee"), ages.get("dee", 0), "bob" in ages, "dee" not in ages)
try:
    ages["dee"]
except KeyError as e:
    print("KeyError", e)
print(ages.pop("bob"), ages.pop("nobody", None), ages)
del ages["cid"]
print(ages, ages.setdefault("eve", 1), ages)
counts = {}
for ch in "mississippi":
    counts[ch] = counts.get(ch, 0) + 1
print(counts, list(counts.keys()), list(counts.values()), list(counts.items()))
sq = {n: n * n for n in range(5)}
print(sq, {v: k for k, v in sq.items()}, dict(zip("abc", [1, 2, 3])), dict(a=1, b=2))
merged = {**{"x": 1, "y": 2}, **{"y": 3, "z": 4}}
print(merged, {"x": 1} | {"x": 2}, dict.fromkeys("ab", 0))
d = {1: "a"}
d.update([(2, "b")], c=3)
print(d, d.popitem(), d)
