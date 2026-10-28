// SPDX-License-Identifier: MIT OR Apache-2.0
// Template literals: substitutions convert with ToString, nest, and tagged
// templates receive the cooked and raw strings.
const n = 42, s = "go", x = 3.14159;
console.log(`${n}|${s}|${x}|${x.toFixed(2)}|${n * 2}|${[1, 2]}|${{}}|${null}|${undefined}|${-0}|${1e21}`);
console.log(`nested ${`inner ${n}`} done`, `${"a" + `b${"c"}`}`);
function tag(strings, ...vals) {
  return JSON.stringify(strings) + " " + JSON.stringify(strings.raw) + " " + JSON.stringify(vals);
}
console.log(tag`a${1}b\n${"two"}c`);
console.log(`line1
line2`.split("\n").length, `\u{41}\x42`);
