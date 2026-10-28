# SPDX-License-Identifier: MIT OR Apache-2.0
# Slices: negative indices and steps, bounds past the ends, assignment to
# slices and deletion.
a = list(range(10))
print(a[2:5], a[-3:], a[:-7], a[::3], a[::-2], a[5:2:-1], a[-1:-4:-1], a[100:], a[-100:3], a[3:3])
s = "abcdefg"
print(s[1::2], s[::-1], s[-2:], s[:100], s[slice(1, 4)], s[5:1], s[::-3])
b = a[:]
b[2:5] = ["x"]
print(b)
b[::2] = range(len(b[::2]))
print(b)
del b[1::2]
print(b)
try:
    b[::2] = [1]
except ValueError as e:
    print("ValueError", e)
try:
    a[10]
except IndexError as e:
    print("IndexError", e)
print(slice(1, 10, 3).indices(5), len(range(0, 10, 3)[1:]), range(10)[::-1][:3])
