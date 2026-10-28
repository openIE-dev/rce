// SPDX-License-Identifier: MIT OR Apache-2.0
// The loose equality table: == converts its operands by the abstract
// equality algorithm; === and Object.is do not convert.
const vals = [0, -0, 1, "", "0", "1", "a", null, undefined, NaN, false, true, [], [0], [1], {}];
const name = (v) => (Object.is(v, -0) ? "-0" : Array.isArray(v) ? `[${v}]` : typeof v === "object" && v !== null ? "{}" : JSON.stringify(v) === undefined ? String(v) : typeof v === "number" ? String(v) : JSON.stringify(v));
for (const a of vals) {
  console.log(name(a).padEnd(9), vals.map((b) => (a == b ? "=" : ".")).join(""), vals.map((b) => (Object.is(a, b) ? "o" : ".")).join(""));
}
console.log(null >= 0, null > 0, null == 0, undefined >= 0, "" < 1, "b" > "a", [2] > 1, ({}) < 1, ({}) > 1);
