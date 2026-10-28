// SPDX-License-Identifier: MIT OR Apache-2.0
// JSON at its edges: values JSON cannot hold, toJSON, key order, duplicate
// keys, number precision and lone surrogates.
console.log(JSON.stringify({ u: undefined, f() {}, s: Symbol("x"), n: NaN, i: -Infinity, z: -0, d: 1e21, e: 1e-7 }));
console.log(JSON.stringify([undefined, () => 0, Symbol("y"), NaN]), JSON.stringify(undefined), JSON.stringify(() => 0), JSON.stringify(null));
console.log(JSON.stringify({ toJSON() { return { replaced: true }; } }), JSON.stringify({ a: { toJSON(key) { return "key:" + key; } } }));
console.log(JSON.stringify({ b: 1, a: 2, 2: 3, 1: 4 }), JSON.stringify("\u2028\u2029\ud800\u007f\u0000"), JSON.stringify("😀"));
const parsed = JSON.parse('{"a": 1, "a": 2, "__proto__": {"p": 1}, "big": 12345678901234567890, "neg0": -0, "exp": 1E400}');
console.log(parsed.a, Object.keys(parsed).join(","), parsed.big, Object.is(parsed.neg0, -0), parsed.exp, parsed.p, Object.getPrototypeOf(parsed) === Object.prototype);
console.log(JSON.parse("[1, 2]", (k, v) => (typeof v === "number" ? v * 10 : v)), JSON.parse('"\\u00e9"'), JSON.parse(" 1 "));
for (const bad of ["{'a': 1}", "[1,]", "01", "NaN", '"\t"', ""]) {
  try {
    JSON.parse(bad);
    console.log("parsed", bad);
  } catch (e) {
    console.log(e.name, JSON.stringify(bad));
  }
}
console.log(JSON.stringify([new Map([[1, 2]]), new Set([1])]));
