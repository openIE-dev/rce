// SPDX-License-Identifier: MIT OR Apache-2.0
// Destructuring of arrays and objects: defaults, rest, renaming, nesting,
// swapping and parameters.
const [a, b = 2, ...rest] = [1, undefined, 3, 4];
console.log(a, b, rest);
const { x, y: why = 5, ...others } = { x: 1, z: 3, w: 4 };
console.log(x, why, others);
let p = 1, q = 2;
[p, q] = [q, p];
console.log(p, q);
const { inner: { deep = "d" } = {} } = { inner: {} };
console.log(deep);
function f({ name, tags = [] } = {}, [first, second] = []) {
  return `${name}:${tags.length}:${first}:${second}`;
}
console.log(f(), f({ name: "n", tags: [1] }, ["a"]));
const [, , third] = "abcd";
const { length } = "hello";
console.log(third, length);
for (const [k, v] of Object.entries({ m: 1, n: 2 })) console.log(k, v);
const { 0: zero, [1 + 1]: two } = ["z", "o", "t"];
console.log(zero, two);
