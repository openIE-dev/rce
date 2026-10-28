// SPDX-License-Identifier: MIT OR Apache-2.0
// Own property order: keys that are array indices (canonical integers below
// 2^32 - 1) come first in ascending order, then other strings in insertion
// order, then symbols.
const o = {};
o.b = 1;
o[2] = 1;
o.a = 1;
o[1] = 1;
o["-1"] = 1;
o["01"] = 1;
o[4294967295] = 1;
o[4294967294] = 1;
o[Symbol("s")] = 1;
o["1.5"] = 1;
o[0] = 1;
console.log(Object.keys(o).join(" "));
console.log(JSON.stringify(o));
const forIn = [];
for (const k in o) forIn.push(k);
console.log(forIn.join(" "), Reflect.ownKeys(o).length, Object.getOwnPropertySymbols(o).length);
const merged = Object.assign({ z: 1, 10: 1 }, { y: 1, 5: 1 });
console.log(Object.keys(merged).join(" "), Object.entries({ b: 2, a: 1, 1: 0 }).join(";"));
delete o.b;
o.b = 2;
console.log(Object.keys(o).slice(-2).join(" "));
