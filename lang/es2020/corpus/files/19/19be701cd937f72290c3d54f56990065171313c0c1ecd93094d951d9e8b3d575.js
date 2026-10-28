// SPDX-License-Identifier: MIT OR Apache-2.0
// Prototype chains, Object.create, property descriptors, getters and
// setters, and frozen objects in sloppy and strict code.
const base = { greet() { return "hi from " + this.name; } };
const child = Object.create(base);
child.name = "child";
console.log(child.greet(), Object.getPrototypeOf(child) === base, "greet" in child, child.hasOwnProperty("greet"));
const bare = Object.create(null);
bare.k = 1;
console.log(Object.keys(bare), typeof bare.toString, "toString" in bare);
const o = {};
Object.defineProperty(o, "hidden", { value: 1, enumerable: false, writable: false });
o.hidden = 2;
console.log(o.hidden, Object.keys(o), JSON.stringify(o), Object.getOwnPropertyNames(o));
let store = 0;
const acc = {
  get v() { return store * 2; },
  set v(x) { store = x + 1; },
};
acc.v = 4;
console.log(acc.v, store, JSON.stringify(acc));
const frozen = Object.freeze({ a: 1, inner: { b: 1 } });
frozen.a = 2;
frozen.inner.b = 2;
console.log(frozen.a, frozen.inner.b, Object.isFrozen(frozen));
try {
  (function () {
    "use strict";
    frozen.a = 3;
  })();
} catch (e) {
  console.log(e.name);
}
function Legacy(n) {
  this.n = n;
}
Legacy.prototype.double = function () {
  return this.n * 2;
};
console.log(new Legacy(21).double(), Legacy.prototype.constructor === Legacy, Object.keys(new Legacy(1)));
