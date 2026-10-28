# SPDX-License-Identifier: MIT OR Apache-2.0
# Two linear congruential generators with fixed seeds: one modulo 2^31 - 1
# (MINSTD) and one modulo 2^64.
x = 1
out = []
for _ in range(10):
    x = x * 48271 % 2147483647
    out.append(x)
print(*out)
s = 42
M = 2 ** 64
out = []
for _ in range(10):
    s = (s * 6364136223846793005 + 1442695040888963407) % M
    out.append(s >> 33)
print(*out)
hist = [0] * 10
for _ in range(100000):
    s = (s * 6364136223846793005 + 1442695040888963407) % M
    hist[(s >> 33) % 10] += 1
print(hist)
