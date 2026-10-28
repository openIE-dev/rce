// SPDX-License-Identifier: MIT OR Apache-2.0
// How this is bound: method calls, extracted methods in strict and sloppy
// functions, arrow functions, call, apply and bind.
const obj = {
  name: "obj",
  regular() {
    return this && this.name;
  },
  arrow: () => typeof this,
  nested() {
    return [1].map(() => this.name)[0];
  },
};
function sloppy() {
  return this === globalThis;
}
function strict() {
  "use strict";
  return this;
}
const extracted = obj.regular;
console.log(obj.regular(), extracted.call({ name: "other" }), obj.arrow(), obj.nested());
console.log(sloppy(), strict(), strict.call(5), typeof sloppy.call(5), sloppy.call(null));
const bound = obj.regular.bind({ name: "bound" });
console.log(bound(), bound.call({ name: "ignored" }), Math.max.apply(null, [1, 5, 3]));
try {
  (function () {
    "use strict";
    return extracted.call(undefined);
  })();
  console.log("undefined this returned", extracted.call(undefined));
} catch (e) {
  console.log(e.name);
}
