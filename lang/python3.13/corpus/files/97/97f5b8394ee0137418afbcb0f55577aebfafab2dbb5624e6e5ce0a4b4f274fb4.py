# SPDX-License-Identifier: MIT OR Apache-2.0
# The 0/1 knapsack problem by dynamic programming over capacities.
weights = [12, 7, 11, 8, 9, 5, 3, 14]
values = [24, 13, 23, 15, 16, 9, 5, 30]
for capacity in [0, 10, 26, 40, 69]:
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            best[c] = max(best[c], best[c - w] + v)
    print(capacity, best[capacity])
