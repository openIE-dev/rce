# SPDX-License-Identifier: MIT OR Apache-2.0
# The longest Collatz chain that starts below 100000, with a memo table.
limit = 100000
memo = [0] * limit
memo[1] = 1
best, best_n = 0, 0
for start in range(2, limit):
    n, steps = start, 0
    while n >= limit or memo[n] == 0:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    memo[start] = steps + memo[n]
    if memo[start] > best:
        best, best_n = memo[start], start
print(best_n, best)
print(memo[27], memo[97], memo[871])
