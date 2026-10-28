# SPDX-License-Identifier: MIT OR Apache-2.0
# Names refer to objects: repeated rows share one list, += mutates in place
# while + makes a new list, and a mutable default argument is shared
# between calls.
grid = [[0] * 3] * 3
grid[0][0] = 1
print(grid)
grid2 = [[0] * 3 for _ in range(3)]
grid2[0][0] = 1
print(grid2)
a = [1, 2]
b = a
b += [3]
print(a, b, a is b)
c = a
c = c + [4]
print(a, c, a is c)
t = ([1],)
try:
    t[0] += [2]
except TypeError as e:
    print("TypeError", e)
print(t)


def append_to(x, acc=[]):
    acc.append(x)
    return acc


print(append_to(1), append_to(2), append_to(3, []), append_to(4))
s = [1, 2, 3]
for x in s[:]:
    s.remove(x)
print(s)
d = {"k": [1]}
e = dict(d)
e["k"].append(2)
print(d, e, d == e)
x = [1, 2, 3]
x[1:2] = [7, 8, 9]
print(x, x[::-1], x[10:], x[-100:2])
