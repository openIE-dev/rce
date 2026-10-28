// SPDX-License-Identifier: MIT OR Apache-2.0
// Node's console.log formatting: printf-like directives in the first
// argument, nesting depth, long arrays, quoting, and special values.
console.log("%s is %d years and %i days, %f, %j, %o, %%, %c.", "x", 42.5, 42.5, "1.5", { a: 1 }, [1], "css");
console.log("%s", "only one", "extra", 3);
console.log({ a: { b: { c: { d: { e: 1 } } } } }, [[[[["deep"]]]]]);
console.log(Array.from({ length: 120 }, (_, i) => i));
console.log({ "key with space": 1, valid_id: 2, 3: 3, "it's": "it's", q: 'say "hi"', both: "'\"" });
console.log([-0, 0, 1e21, 1e-7, 2n ** 70n, NaN], -0, [undefined, null], { u: undefined });
class Point {
  constructor() {
    this.x = 1;
    this.y = 2;
  }
}
console.log(new Point(), Object.create(null), [new Point()], { p: new Point() });
console.log(new Map([[{ k: 1 }, new Set([1])]]), [new Error("in array").message], "a".repeat(200).length);
console.log([, 1, , 2], new Array(5), [1, 2, 3].keys(), Symbol.for("reg"), function named() {}, class Klass {});
console.log("x".repeat(30), ["x".repeat(30), "y".repeat(30), "z".repeat(30)]);
