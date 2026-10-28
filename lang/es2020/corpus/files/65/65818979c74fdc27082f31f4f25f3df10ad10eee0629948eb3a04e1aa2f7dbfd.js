// SPDX-License-Identifier: MIT OR Apache-2.0
// Transcendental Math functions, whose results the specification leaves to
// the implementation's approximation; the pinned engine fixes them.
for (const x of [0.5, 1, 2, 10, 1e-8, 100]) {
  console.log(x, Math.sin(x), Math.cos(x), Math.tan(x), Math.exp(x), Math.log(x), Math.atan(x));
}
console.log(Math.log10(1000), Math.log2(1024), Math.log(8) / Math.log(2), Math.pow(2, 0.5), Math.sinh(1), Math.atan2(1, -1), Math.expm1(1e-10), Math.log1p(1e-10));
