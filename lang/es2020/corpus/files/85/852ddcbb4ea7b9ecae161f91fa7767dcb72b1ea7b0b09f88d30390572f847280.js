// SPDX-License-Identifier: MIT OR Apache-2.0
// IEEE 754 binary64 arithmetic and the Math functions. Math.sqrt and the
// rounding functions are exact; the specification lets Math.hypot and
// Math.cbrt approximate, and the pinned engine fixes their results.
const a = 0.1, b = 0.2;
console.log(a + b, a + b === 0.3, a * 3, 1 - 0.9);
console.log(1 / 3, 2 / 3, (1 / 3) * 3);
console.log(1e300 * 1e10, -1e300 * 1e10, (1e300 * 1e300) / 1e300);
console.log(Math.sqrt(2), Math.sqrt(2) ** 2, Math.abs(-2.5));
console.log(Math.floor(-2.5), Math.ceil(-2.5), Math.trunc(-2.5), Math.round(-2.5), Math.round(2.5), Math.round(3.5));
let x = 0;
for (let i = 0; i < 10; i++) x += 0.1;
console.log(x, x === 1, Math.abs(x - 1) < Number.EPSILON);
console.log(Math.PI, Math.E, Number.MAX_VALUE, Number.MIN_VALUE, Number.EPSILON, Number.MAX_SAFE_INTEGER);
console.log(Math.hypot(3, 4), Math.sign(-3), Math.cbrt(27), Math.fround(0.1), Math.clz32(1), Math.max(), Math.min());
