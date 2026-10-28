// SPDX-License-Identifier: MIT OR Apache-2.0
// The number of ways to place n non-attacking queens, for n from 1 to 9,
// and the first solution for n = 8, with bit masks.
function solve(n) {
  let count = 0;
  let first = null;
  const pos = [];
  const full = (1 << n) - 1;
  (function place(row, cols, d1, d2) {
    if (row === n) {
      count++;
      if (!first) first = pos.slice();
      return;
    }
    let avail = full & ~(cols | d1 | d2);
    while (avail) {
      const bit = avail & -avail;
      avail ^= bit;
      pos[row] = 31 - Math.clz32(bit);
      place(row + 1, cols | bit, ((d1 | bit) << 1) & full, (d2 | bit) >> 1);
    }
  })(0, 0, 0, 0);
  return [count, first];
}
for (let n = 1; n <= 9; n++) {
  const [k, first] = solve(n);
  console.log(n, k);
  if (n === 8) console.log(first.join(" "));
}
