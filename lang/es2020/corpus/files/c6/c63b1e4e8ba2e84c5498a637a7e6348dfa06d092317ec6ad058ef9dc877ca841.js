// SPDX-License-Identifier: MIT OR Apache-2.0
// The same operations in sloppy and strict code: assignment to an
// undeclared name, to a read-only property, and deletion of a
// non-configurable property.
function sloppy() {
  undeclaredSloppy = 1;
  const o = Object.freeze({ a: 1 });
  o.a = 2;
  return [typeof undeclaredSloppy, o.a, delete Object.prototype];
}
function strict() {
  "use strict";
  const out = [];
  for (const op of [() => { undeclaredStrict = 1; }, () => { Object.freeze({ a: 1 }).a = 2; }, () => delete Object.prototype, () => { NaN = 1; }]) {
    try {
      op();
      out.push("ok");
    } catch (e) {
      out.push(e.name);
    }
  }
  return out;
}
console.log(sloppy().join(","), strict().join(","), typeof undeclaredStrict, typeof undeclaredSloppy);
