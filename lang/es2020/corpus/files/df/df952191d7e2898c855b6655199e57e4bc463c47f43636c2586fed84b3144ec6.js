// SPDX-License-Identifier: MIT OR Apache-2.0
// Number to text at the edges of the rules: the exponent switches,
// rounding in toFixed and toPrecision, and radix conversion of integers.
const vals = [1e21, 999999999999999900000, 1e-6, 1e-7, 123e-20, 0.000001234, 1.5e300, 5e-324, 2 ** 53, -1e-7, 100, 1e100];
for (const v of vals) console.log(String(v), v.toExponential(), v.toPrecision(1), v.toPrecision(21));
console.log((0.5).toFixed(0), (1.5).toFixed(0), (2.5).toFixed(0), (-2.5).toFixed(0), (1.25).toFixed(1), (1.35).toFixed(1));
console.log((123.456).toFixed(10), (1e-10).toFixed(5), (0.000001).toFixed(7), (123456789).toFixed(), (1.45).toPrecision(2));
console.log((2 ** 53).toString(2), (-(2 ** 31)).toString(16), (35).toString(36), (1e21).toString(16) === (1e21).toString(16));
console.log(String(1 / 3), (1 / 3).toPrecision(17), (0.1).toPrecision(21), (123.456).toExponential(0), (0).toExponential(2));
try {
  (1).toFixed(101);
} catch (e) {
  console.log(e.name);
}
