# SPDX-License-Identifier: MIT OR Apache-2.0
# Dictionaries keep insertion order: updating a key keeps its place,
# deleting and inserting again moves it to the end.
d = {}
for k in ["b", "a", "c"]:
    d[k] = len(d)
print(d, list(d))
d["a"] = 99
print(d)
del d["b"]
d["b"] = 7
print(d)
d.pop("c")
d["c"] = 0
d.update({"a": 1, "z": 2})
print(d, list(d.items()))
print(dict(sorted(d.items())), dict(reversed(d.items())), list(reversed(d)))
e = {3: "x", 1: "y", 2: "z"}
print(e, sorted(e), {**e, 1: "w"}, {1: "w", **e})
keys = {}
for w in "the quick brown the lazy the end".split():
    keys.setdefault(w, []).append(len(keys))
print(keys)
