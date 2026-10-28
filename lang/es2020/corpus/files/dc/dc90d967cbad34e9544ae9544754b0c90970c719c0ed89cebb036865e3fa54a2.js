// SPDX-License-Identifier: MIT OR Apache-2.0
// var is function-scoped and hoisted; let and const are block-scoped and
// may not be read before their declaration.
console.log(typeof hoisted, typeof declared, declared());
var hoisted = 1;
function declared() {
  return "function declarations are hoisted whole";
}
const fs = [];
for (var i = 0; i < 3; i++) fs.push(() => i);
const gs = [];
for (let j = 0; j < 3; j++) gs.push(() => j);
console.log(fs.map((f) => f()).join(","), gs.map((g) => g()).join(","), i);
try {
  console.log(later);
  let later = 1;
} catch (e) {
  console.log(e.name);
}
{
  var leaks = "var leaks out of a block";
  let stays = "let stays";
  console.log(stays);
}
console.log(leaks, typeof stays);
const c = { n: 1 };
c.n = 2;
try {
  c = {};
} catch (e) {
  console.log(e.name, c.n);
}
