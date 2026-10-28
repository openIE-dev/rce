# SPDX-License-Identifier: MIT OR Apache-2.0
# Lists: append, extend, insert, pop, remove, index, slices, membership
# and iteration.
s = []
for i in range(1, 6):
    s.append(i * i)
print(s, len(s), s[0], s[-1], s[1:3], s[:2], s[3:])
s.extend([0, 0])
s.insert(1, 99)
print(s, s.pop(), s.pop(0), s)
s.remove(0)
print(s, s.index(16), 9 in s, 7 in s, s.count(9))
s.reverse()
print(s, sum(s), max(s), min(s))
t = s.copy()
t[0] = -1
print(s, t, s + t[:1], [0] * 3)
del t[1:3]
print(t)
m = [["a"], ["b", "c"]]
m[1].append("d")
print(m, len(m[1]), [len(r) for r in m])
r = [v for v in s if v % 2 == 0]
print(r, list(reversed(r)), list(enumerate(r, 1)), list(zip(r, "xyz")))
s.clear()
print(s)
