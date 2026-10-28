// SPDX-License-Identifier: MIT OR Apache-2.0
// Closures capture variables: a counter, an adder, and loops whose let
// binding is fresh on each iteration while var is shared.
function counter() {
  let n = 0;
  return () => ++n;
}
const c1 = counter(), c2 = counter();
console.log(c1(), c1(), c1(), c2());
const adder = (base) => (x) => base + x;
const add5 = adder(5);
console.log(add5(1), add5(10));
const withLet = [];
for (let i = 0; i < 3; i++) withLet.push(() => i * 10);
const withVar = [];
for (var j = 0; j < 3; j++) withVar.push(() => j * 10);
console.log(withLet.map((f) => f()), withVar.map((f) => f()), j);
const fibGen = (() => {
  let a = 0, b = 1;
  return () => {
    const r = a;
    [a, b] = [b, a + b];
    return r;
  };
})();
console.log(Array.from({ length: 8 }, fibGen).join(" "));
let shared = 0;
const inc = () => shared++;
inc();
inc();
console.log(shared);
