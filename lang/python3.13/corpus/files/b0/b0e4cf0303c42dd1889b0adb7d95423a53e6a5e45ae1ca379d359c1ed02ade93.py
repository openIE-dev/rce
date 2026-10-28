# SPDX-License-Identifier: MIT OR Apache-2.0
# Floor division rounds toward negative infinity and the remainder has the
# sign of the divisor, for every combination of signs; / always gives a
# float.
pairs = [(7, 2), (-7, 2), (7, -2), (-7, -2), (6, 3), (-6, 3), (0, 5), (0, -5), (1, 7), (-1, 7), (13, 13), (-13, 13)]
for a, b in pairs:
    print(f"{a} // {b} = {a // b}, {a} % {b} = {a % b}, divmod = {divmod(a, b)}, {a} / {b} = {a / b}")
print(7 // 2.0, -7 // 2.0, 7 % -2.5, -7.5 % 2, divmod(-7.5, 2))
print(int(-7 / 2), int(7 / -2), round(-7 / 2))
