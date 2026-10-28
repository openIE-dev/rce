# SPDX-License-Identifier: MIT OR Apache-2.0
# Integers are unbounded: arithmetic past 64 bits stays exact.
import math

print(2 ** 64, 2 ** 64 - 1, -2 ** 63, 2 ** 200)
print(math.factorial(50))
print(int("9" * 60) + 1, -(10 ** 40) // 7, -(10 ** 40) % 7)
a = 2 ** 127 - 1
print(a * a, a.bit_length(), (a * a).bit_length())
print(pow(2, 1000, 10 ** 9 + 7), pow(3, -1, 7), pow(10, 50) // 3 ** 50)
print(abs(-(2 ** 100)), divmod(-(2 ** 70), 3), 2 ** 70 // -3)
print(sum(range(10 ** 6)), sum(i * i for i in range(10 ** 4)))
print(-1 >> 100, (2 ** 100) >> 99, 1 << 100, ~(2 ** 64), (2 ** 64) & -(2 ** 64), (-5) & 0xFF)
print(int(1e300) // 10 ** 290, int(2.0 ** 100), float(2 ** 100), 10 ** 20 / 3)
