// SPDX-License-Identifier: MIT OR Apache-2.0
// Conversion to and from Roman numerals, checked by a round trip.
const VALS = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
const DIGIT = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };
function toRoman(n) {
  let out = "";
  for (const [v, s] of VALS) {
    while (n >= v) {
      out += s;
      n -= v;
    }
  }
  return out;
}
function fromRoman(s) {
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    const v = DIGIT[s[i]];
    total += i + 1 < s.length && DIGIT[s[i + 1]] > v ? -v : v;
  }
  return total;
}
for (const n of [1, 4, 9, 14, 40, 90, 400, 1994, 2024, 3999]) {
  const r = toRoman(n);
  console.log(n, r, fromRoman(r));
}
let bad = 0;
for (let n = 1; n < 4000; n++) if (fromRoman(toRoman(n)) !== n) bad++;
console.log(bad);
