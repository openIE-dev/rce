# SPDX-License-Identifier: MIT OR Apache-2.0
# Addition, subtraction, multiplication, power and precedence on int.
a, b, c = 17, 5, -3
print(a + b, a - b, a * b, b - a, a * c)
print(a + b * c, (a + b) * c, a - b - c, a - (b - c))
print(-a, - -a, +c, 2 ** 10, -2 ** 2, (-2) ** 2, 2 ** -1, 2 ** 3 ** 2)
x = 1
for i in range(10):
    x = x * 3 + i
print(x)
x += 5
x -= 2
x *= 4
x **= 2
print(x)
print(abs(-9), max(3, 7, 5), min(3, 7, 5), sum([1, 2, 3, 4]), pow(3, 4), pow(3, 4, 5))
