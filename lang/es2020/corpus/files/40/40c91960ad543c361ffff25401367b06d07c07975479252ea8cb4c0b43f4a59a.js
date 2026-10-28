// SPDX-License-Identifier: MIT OR Apache-2.0
// / is float division; % takes the sign of the dividend; Math.trunc and
// Math.floor give the two integer divisions.
const pairs = [[7, 2], [-7, 2], [7, -2], [-7, -2], [6, 3], [-6, 3], [0, 5], [0, -5], [1, 7], [-1, 7], [13, 13], [-13, 13]];
for (const [a, b] of pairs) {
  console.log(`${a} / ${b} = ${a / b}, trunc ${Math.trunc(a / b)}, floor ${Math.floor(a / b)}, ${a} % ${b} = ${a % b}`);
}
console.log(7.5 % 2, -7.5 % 2, 7 % -2.5, 5.5 % 1, -0 % 5, Object.is(-0 % 5, -0), Object.is(-6 % 3, -0));
console.log(((-7 % 2) + 2) % 2, (7 / 2) | 0, (-7 / 2) | 0, ~~(-7 / 2));
