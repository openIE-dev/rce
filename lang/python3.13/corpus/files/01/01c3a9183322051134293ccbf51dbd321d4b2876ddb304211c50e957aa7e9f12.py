# SPDX-License-Identifier: MIT OR Apache-2.0
# Tuples, packing and unpacking, starred targets, swaps and named tuples.
from collections import namedtuple

t = 1, "two", 3.0
a, b, c = t
print(t, a, b, c, len(t), t[1], t[-1], t[::-1])
first, *middle, last = range(6)
print(first, middle, last)
*init, tail = "abc"
print(init, tail)
x, y = 1, 2
x, y = y, x
print(x, y)
(p, q), r = (1, 2), 3
print(p, q, r)
for i, (k, v) in enumerate({"a": 1, "b": 2}.items()):
    print(i, k, v)
Point = namedtuple("Point", "x y")
pt = Point(3, 4)
print(pt, pt.x, pt._asdict(), pt._replace(y=0), tuple(pt), pt == (3, 4))
print((1,) * 3, (1, 2) + (3,), (1, 2) < (1, 3), tuple("ab"), (), (5,))
try:
    a, b = [1, 2, 3]
except ValueError as e:
    print("ValueError", e)
