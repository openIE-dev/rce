// SPDX-License-Identifier: MIT OR Apache-2.0
// The 0/1 knapsack problem by dynamic programming over capacities.
const weights = [12, 7, 11, 8, 9, 5, 3, 14];
const values = [24, 13, 23, 15, 16, 9, 5, 30];
for (const capacity of [0, 10, 26, 40, 69]) {
  const best = new Array(capacity + 1).fill(0);
  weights.forEach((w, i) => {
    for (let c = capacity; c >= w; c--) best[c] = Math.max(best[c], best[c - w] + values[i]);
  });
  console.log(capacity, best[capacity]);
}
