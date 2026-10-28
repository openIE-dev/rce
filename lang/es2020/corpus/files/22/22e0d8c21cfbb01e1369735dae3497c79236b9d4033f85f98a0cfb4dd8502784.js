// SPDX-License-Identifier: MIT OR Apache-2.0
// Arrays: push, pop, shift, unshift, splice, slice, indexOf, includes,
// iteration and the functional methods.
const s = [];
for (let i = 1; i <= 5; i++) s.push(i * i);
console.log(s, s.length, s[0], s[s.length - 1], s.slice(1, 3), s.slice(-2));
s.unshift(0);
console.log(s.pop(), s.shift(), s);
const removed = s.splice(1, 2, "x", "y", "z");
console.log(removed, s, s.indexOf("y"), s.includes(9), s.includes(16));
console.log([3, 1, 2].reverse(), [1, [2, [3, [4]]]].flat(), [1, [2, [3, [4]]]].flat(Infinity), [1, 2].flatMap((v) => [v, v * 10]));
const nums = [1, 2, 3, 4, 5, 6];
console.log(nums.map((v) => v * v), nums.filter((v) => v % 2), nums.reduce((a, v) => a + v), nums.reduceRight((a, v) => a + v, ""));
console.log(nums.find((v) => v > 3), nums.findIndex((v) => v > 3), nums.some((v) => v > 5), nums.every((v) => v > 0));
console.log(Array.from({ length: 4 }, (_, i) => i * 2), Array.of(7, 8), new Array(3).fill(0), [..."abc"], [1, 2].concat([3], 4, [[5]]));
const m = [["a"], ["b", "c"]];
m[1].push("d");
console.log(m, m[1].length, m.map((r) => r.length));
for (const [i, v] of ["p", "q"].entries()) console.log(i, v);
