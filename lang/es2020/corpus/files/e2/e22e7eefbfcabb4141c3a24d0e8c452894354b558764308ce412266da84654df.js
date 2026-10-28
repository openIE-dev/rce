// SPDX-License-Identifier: MIT OR Apache-2.0
// typeof and instanceof over every kind of value, including an undeclared
// name, which typeof alone may read.
const vals = [1, NaN, "s", true, undefined, null, {}, [], () => 0, class {}, Symbol("x"), 10n, new Number(1), new String("s"), /re/, new Map(), Object.create(null)];
for (const v of vals) {
  console.log(typeof v, Object.prototype.toString.call(v), v instanceof Object, Array.isArray(v));
}
console.log(typeof undeclaredName, typeof typeof 1, typeof Math, typeof JSON, typeof globalThis, typeof console.log);
console.log(new Number(1) === 1, new Number(1) == 1, typeof new Number(1).valueOf(), [] instanceof Array, Object(1n) instanceof BigInt);
