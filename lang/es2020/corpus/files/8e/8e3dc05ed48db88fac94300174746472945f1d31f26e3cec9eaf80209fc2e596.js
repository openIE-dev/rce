// SPDX-License-Identifier: MIT OR Apache-2.0
// Division by zero gives infinities or NaN and never throws for Number;
// the sign of zero is observable through division and Object.is.
console.log(1 / 0, -1 / 0, 0 / 0, 5 % 0, -5 % Infinity, Infinity % 2, Infinity / Infinity);
const negZero = -0;
console.log(1 / negZero, negZero === 0, Object.is(negZero, 0), Object.is(negZero, -0), String(negZero), JSON.stringify(negZero));
console.log(Math.sign(-0), 1 / Math.sign(-0), 0 * -1, 1 / (0 * -1), -0 + 0, 1 / (-0 + 0), -0 - 0, 1 / (-0 - 0));
console.log([-0].includes(0), [-0].indexOf(0), new Set([0, -0]).size, Math.max(-0, 0), 1 / Math.max(-0, 0), 1 / Math.min(-0, 0));
console.log(Math.round(-0.4), 1 / Math.round(-0.4), Math.trunc(-0.5), (-0).toFixed(1), (-0.0001).toFixed(2), -0 < 0);
try {
  console.log(1n / 0n);
} catch (e) {
  console.log(e.name, e.message);
}
