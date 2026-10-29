# SPDX-License-Identifier: MIT OR Apache-2.0
try:
    x = 10 // 0
except ZeroDivisionError:
    x = 0
print(x)
