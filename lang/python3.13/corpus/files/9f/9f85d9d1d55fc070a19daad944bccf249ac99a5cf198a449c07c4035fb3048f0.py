# SPDX-License-Identifier: MIT OR Apache-2.0
# round rounds halves to even, and rounds the exact binary value, so 2.675
# rounds down; format specifications do the same.
print([round(x) for x in (0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5)])
print(round(2.675, 2), round(1.005, 2), round(0.125, 2), round(0.375, 2), round(1234.5678, -2), round(-0.0), round(5, -1), round(15, -1), round(25, -1))
print(f"{2.5:.0f} {3.5:.0f} {0.125:.2f} {2.675:.2f} {1e16 + 1:.0f} {-0.5:.0f}")
print(type(round(2.5)), type(round(2.5, 0)), round(2.5, 0), round(7, 2))
print(round(float("inf"), 2), round(float("nan"), 1))
try:
    round(float("inf"))
except OverflowError as e:
    print("OverflowError", e)
