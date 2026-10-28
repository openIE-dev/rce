// SPDX-License-Identifier: MIT OR Apache-2.0
// The longest Collatz chain that starts below 100000, with a memo table.
const limit = 100000;
const memo = new Int32Array(limit);
memo[1] = 1;
let best = 0, bestN = 0;
for (let start = 2; start < limit; start++) {
  let n = start, steps = 0;
  while (n >= limit || memo[n] === 0) {
    n = n % 2 === 0 ? n / 2 : 3 * n + 1;
    steps++;
  }
  memo[start] = steps + memo[n];
  if (memo[start] > best) {
    best = memo[start];
    bestN = start;
  }
}
console.log(bestN, best);
console.log(memo[27], memo[97], memo[871]);
