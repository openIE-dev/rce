// SPDX-License-Identifier: MIT OR Apache-2.0
// A stable merge sort of records by key; equal keys keep their order.
function mergesort(a) {
  if (a.length <= 1) return a;
  const m = a.length >> 1;
  const l = mergesort(a.slice(0, m)), r = mergesort(a.slice(m));
  const out = [];
  let i = 0, j = 0;
  while (i < l.length && j < r.length) out.push(r[j][0] < l[i][0] ? r[j++] : l[i++]);
  return out.concat(l.slice(i), r.slice(j));
}
const data = [[3, "a"], [1, "b"], [2, "c"], [3, "d"], [1, "e"], [2, "f"], [0, "g"], [3, "h"]];
console.log(JSON.stringify(mergesort(data)));
console.log(JSON.stringify(mergesort(data)) === JSON.stringify([...data].sort((p, q) => p[0] - q[0])));
console.log(JSON.stringify(data));
