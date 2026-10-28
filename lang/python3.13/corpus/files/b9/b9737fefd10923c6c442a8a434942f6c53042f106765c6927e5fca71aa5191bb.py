# SPDX-License-Identifier: MIT OR Apache-2.0
# Division by zero raises ZeroDivisionError for int and float alike; here it
# is not caught.
print(10 / 4, 10 // 4, 10 % 4)
for d in [2, 1, 0]:
    print(10 // d)
