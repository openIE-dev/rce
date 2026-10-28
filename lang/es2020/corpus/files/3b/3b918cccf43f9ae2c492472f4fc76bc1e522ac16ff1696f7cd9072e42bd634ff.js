// SPDX-License-Identifier: MIT OR Apache-2.0
// Math functions at their edges. Rounding, sign, min and max with NaN and
// -0, fround, sqrt and the special cases of pow and hypot are fixed by the
// specification; Math.cbrt(-8) and 2 ** 0.5 are approximations that the
// pinned engine fixes.
console.log(Math.round(-0.5), Math.round(0.5), Math.round(2.5), Math.round(-2.5), Math.round(1.4999999999999998), Math.round(0.49999999999999994));
console.log(Math.max(), Math.min(), Math.max(NaN, 1), Math.max(1, "3", null), Math.min(-0, 0), Math.max([2], [1]));
console.log(Math.sign(-3), Math.sign(0), Math.sign(-0), Math.sign(NaN), Math.trunc(-4.7), Math.trunc("x"), Math.abs(-0), Math.abs(-Infinity));
console.log(Math.fround(5.5), Math.fround(5.05), Math.fround(2 ** 128), Math.fround(1e-46), Math.sqrt(-1), Math.sqrt(-0), Math.sqrt(2));
console.log(Math.floor(-0.5), Math.ceil(-0.5), Math.ceil(0.2), Math.floor(1e21 + 0.5), Math.hypot(), Math.hypot(NaN, Infinity), Math.cbrt(-8));
console.log(Math.pow(NaN, 0), Math.pow(1, Infinity), (-8) ** (1 / 3), 2 ** 0.5, Math.pow(-0, -1), Math.pow(0, -1));
