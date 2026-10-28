// SPDX-License-Identifier: MIT OR Apache-2.0
// Array methods at their edges: reduce with no start on an empty array,
// indexOf against includes for NaN, splice and copyWithin, length writes.
try {
  [].reduce((a, b) => a + b);
} catch (e) {
  console.log(e.name);
}
console.log([NaN].indexOf(NaN), [NaN].includes(NaN), [-0].includes(0), [1, 2, 3].lastIndexOf(3, -2));
const a = [1, 2, 3, 4, 5];
console.log(a.splice(-2), a, a.splice(1, 0, "x"), a, a.splice(10), [1, 2, 3, 4, 5].copyWithin(0, 3), [1, 2, 3].fill(0, 1));
console.log([1, 2, 3].slice(-2, -1), [3, 2, 1].reverse(), [1, [2], [[3]]].flat(2), ["a", "b"].entries().next().value);
console.log([..."hello"].map((c, i, arr) => arr.length - i).join(""), Array.prototype.concat.call(1, 2), [[1], [2]].join());
const big = [];
big[2 ** 32 - 2] = "last";
console.log(big.length);
try {
  big.length = 2 ** 32;
} catch (e) {
  console.log(e.name);
}
console.log(Array.from("abc", (c) => c.toUpperCase()), Array.from(new Set([3, 3, 1])), Array.isArray(Array.prototype));
