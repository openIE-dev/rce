// SPDX-License-Identifier: MIT OR Apache-2.0
// Binary search over a sorted array, for present and absent values.
function search(xs, target) {
  let lo = 0, hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (xs[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return [lo, lo < xs.length && xs[lo] === target];
}
const xs = Array.from({ length: 50 }, (_, i) => i * i - 100);
for (const t of [-100, -99, 0, 1, 44, 2301, 2302, 5000, -1000]) {
  const [i, ok] = search(xs, t);
  console.log(t, i, ok, ok === xs.includes(t));
}
