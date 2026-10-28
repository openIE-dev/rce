// SPDX-License-Identifier: MIT OR Apache-2.0
// sort with no comparator compares the values' strings; undefined sorts
// last and holes after it; sort is stable.
console.log([10, 9, 1, 2, 100, -1, -10].sort());
console.log([3, undefined, 1, null, 2, NaN].sort(), [true, false, "true", 1, "1"].sort());
const holes = [3, , 1, undefined, 2];
holes.sort();
console.log(holes, holes.length, 1 in holes, 4 in holes);
const recs = ["b1", "a1", "b2", "a2", "c1", "a3"].map((s) => ({ k: s[0], n: s[1] }));
console.log(recs.sort((x, y) => (x.k < y.k ? -1 : x.k > y.k ? 1 : 0)).map((r) => r.k + r.n).join(" "));
console.log([0.1, 0.01, 1e-7, 1e21, 5].sort(), [[2, 1], [10], [1, 9]].sort(), ["é", "e", "f", "E"].sort());
const arr = [5, 1, 4];
const same = arr.sort();
console.log(same === arr, arr);
