// SPDX-License-Identifier: MIT OR Apache-2.0
// Objects: properties, computed keys, shorthand, methods, delete, in,
// Object.keys, values, entries, assign and spread.
const key = "dyn";
const o = { a: 1, "b c": 2, [key + "amic"]: 3, f() { return this.a; } };
o.d = 4;
o["e"] = 5;
console.log(o.a, o["b c"], o.dynamic, o.f(), o.missing, "a" in o, "z" in o, delete o.a, "a" in o);
console.log(Object.keys(o), Object.values(o).length, JSON.stringify(Object.entries({ x: 1, y: [2] })));
const merged = Object.assign({}, { x: 1, y: 2 }, { y: 3, z: 4 });
console.log(merged, { ...merged, x: 0 }, { x: 0, ...merged });
const counts = {};
for (const ch of "mississippi") counts[ch] = (counts[ch] || 0) + 1;
console.log(counts, Object.fromEntries([["p", 1], ["q", 2]]));
const nested = { a: { b: { c: 1 } } };
const copy = { ...nested };
copy.a.b.c = 2;
console.log(nested.a.b.c, copy === nested, copy.a === nested.a);
console.log(Object.keys([4, 5]), Object.keys("hi"), typeof o, Array.isArray(o));
