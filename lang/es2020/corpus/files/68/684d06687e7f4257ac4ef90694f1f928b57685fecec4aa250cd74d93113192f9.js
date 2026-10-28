// SPDX-License-Identifier: MIT OR Apache-2.0
// Map and Set: insertion order, keys of any type, size and deletion.
const m = new Map();
m.set("b", 2).set("a", 1).set(3, "three").set(true, "yes");
console.log(m.size, m.get("a"), m.get("z"), m.has(3), m.has("3"), [...m.keys()], [...m.values()]);
m.delete("b");
m.set("b", 22);
console.log([...m.entries()].map(([k, v]) => `${typeof k}:${String(k)}=${v}`).join(" "));
const objKey = { id: 1 };
m.set(objKey, "object");
console.log(m.get(objKey), m.get({ id: 1 }));
const s = new Set([3, 1, 3, 2, 1]);
s.add(0);
console.log([...s], s.size, s.has(2), s.delete(3), [...s]);
const words = ["the", "quick", "the", "lazy", "the"];
const freq = new Map();
for (const w of words) freq.set(w, (freq.get(w) || 0) + 1);
console.log(JSON.stringify([...freq]), Object.fromEntries(freq));
m.forEach((v, k) => {
  if (typeof k === "string") console.log("entry", k, v);
});
