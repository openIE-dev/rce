// SPDX-License-Identifier: MIT OR Apache-2.0
// Default parameters, rest parameters, spread arguments, arrow functions,
// arguments, function length and name.
function divmod(a, b) {
  return [Math.floor(a / b), a - Math.floor(a / b) * b];
}
function greet(name, greeting = "hello", punct = name.length > 3 ? "!" : "?") {
  return `${greeting}, ${name}${punct}`;
}
function total(start, ...nums) {
  return nums.reduce((s, n) => s + n, start);
}
function args() {
  return [arguments.length, Array.from(arguments).join("|")];
}
const apply = (f, xs) => xs.map(f);
console.log(divmod(17, 5), divmod(-17, 5));
console.log(greet("ann"), greet("bob", "hi"), greet("cid", undefined, "."), greet("dee", null));
console.log(total(0), total(0, 1, 2, 3), total(10, ...[4, 5]), Math.max(...[1, 9, 3]));
console.log(args(), args(1, "a", null), apply((x) => x * x, [1, 2, 3]), apply(Math.abs, [-1, 2]));
console.log(greet.length, total.length, ((a, b = 1, c) => 0).length, greet.name, (() => 0).name, (function () {}).name === "");
const curried = (a) => (b) => (c) => a + b + c;
console.log(curried(1)(2)(3), typeof curried(1));
