// SPDX-License-Identifier: MIT OR Apache-2.0
// JSON.stringify and JSON.parse on ordinary values, with indentation and a
// replacer.
const v = { name: "x", list: [1, 2.5, "s", true, null], nested: { k: "v" } };
console.log(JSON.stringify(v));
console.log(JSON.stringify(v, null, 2));
console.log(JSON.stringify(v, ["name", "nested"]), JSON.stringify(v, (k, val) => (typeof val === "number" ? val * 10 : val)));
const back = JSON.parse('{"a": [1, {"b": null}], "c": "\\u00e9\\n", "d": 1e2}');
console.log(back.a[1].b, back.c, back.d, JSON.stringify(back));
console.log(JSON.stringify("quote\" back\\ tab\t é 😀"), JSON.stringify([]), JSON.stringify({}), JSON.stringify(0.1 + 0.2));
