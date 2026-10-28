// SPDX-License-Identifier: MIT OR Apache-2.0
// Map and Set keep insertion order, compare keys by SameValueZero (NaN
// equals NaN, -0 equals 0), and visit entries added during iteration.
const m = new Map([[NaN, "nan"], [0, "zero"]]);
m.set(NaN, "nan again");
m.set(-0, "minus zero");
console.log(m.size, m.get(NaN), m.get(0), [...m.keys()].map((k) => Object.is(k, -0) ? "-0" : String(k)).join(","));
const s = new Set([1, 2, 3]);
const visited = [];
for (const v of s) {
  visited.push(v);
  if (v === 1) {
    s.delete(2);
    s.add(4);
  }
}
console.log(visited.join(","), [...s].join(","));
const order = new Map();
for (const k of ["c", "a", "b", "a"]) order.set(k, (order.get(k) || 0) + 1);
order.delete("c");
order.set("c", 9);
console.log(JSON.stringify([...order]));
const keys = [{}, {}, "1", 1];
const km = new Map(keys.map((k, i) => [k, i]));
console.log(km.size, km.get("1"), km.get(1), km.get({}));
