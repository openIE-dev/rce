// SPDX-License-Identifier: MIT OR Apache-2.0
// Quicksort with Lomuto partitioning on a sequence from a fixed linear
// congruential generator: Math.imul and >>> 0 keep its arithmetic modulo
// 2^32.
function quicksort(a, lo, hi) {
  if (hi - lo < 2) return;
  const p = a[hi - 1];
  let i = lo;
  for (let j = lo; j < hi - 1; j++) {
    if (a[j] < p) {
      [a[i], a[j]] = [a[j], a[i]];
      i++;
    }
  }
  [a[i], a[hi - 1]] = [a[hi - 1], a[i]];
  quicksort(a, lo, i);
  quicksort(a, i + 1, hi);
}
let s = 12345;
const a = [];
for (let k = 0; k < 40; k++) {
  s = (Math.imul(s, 1103515245) + 12345) >>> 0;
  a.push((s >>> 16) % 1000);
}
console.log(a.join(" "));
quicksort(a, 0, a.length);
console.log(a.join(" "));
console.log(a.every((v, i) => i === 0 || a[i - 1] <= v));
