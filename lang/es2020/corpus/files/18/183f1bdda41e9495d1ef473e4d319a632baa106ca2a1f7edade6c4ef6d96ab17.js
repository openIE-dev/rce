// SPDX-License-Identifier: MIT OR Apache-2.0
// Number to text: String uses the shortest round-trip digits and switches
// to exponent form at 1e21 and below 1e-6; toFixed, toPrecision and
// toExponential round as the specification says.
const vals = [0, 1, -1, 0.5, 0.1, 1 / 3, 100, 123456, 1234567, 1e6, 1e15, 1e16, 1e20, 1e21, 1e-4, 1e-6, 1e-7, 12345.678, 2.5e-8];
for (const v of vals) {
  console.log([String(v), v.toFixed(2), v.toPrecision(4), v.toExponential(3), v.toExponential(), v.toString(10)].join("|"));
}
console.log((255).toString(16), (255).toString(2), (-255).toString(36), (0.5).toString(2), (1e21).toFixed(2));
console.log((1.005).toFixed(2), (1.45).toFixed(1), (8.345).toFixed(2), (0).toFixed(0), (-1.5).toFixed(0), (2.5).toFixed(0));
