// SPDX-License-Identifier: MIT OR Apache-2.0
// && and || return one of their operands and evaluate the right one only
// when needed; ! converts to a boolean.
const calls = [];
const t = (name) => (calls.push(name), true);
const f = (name) => (calls.push(name), false);
console.log(true && false, true || false, !true, !null);
const r1 = f("a") && t("b");
const r2 = t("c") || f("d");
const r3 = (t("e") && f("f")) || t("g");
const r4 = !(f("h") || f("i"));
console.log(r1, r2, r3, r4);
console.log(calls.join(","));
console.log(0 || "default", "" || null || undefined, 3 && 4, 0 && 1 / 0, [] || {}, "x" && "y");
console.log([0, 0.0, -0, "", null, undefined, NaN, "0", [], {}, "false", 0n, 1n].map(Boolean).join(" "));
const n = 5;
console.log(1 < n && n < 10, 1 < n > 3, n === 5 && n !== 6, !!n);
