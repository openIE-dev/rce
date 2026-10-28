// SPDX-License-Identifier: MIT OR Apache-2.0
// Arithmetic on Number values, which are IEEE 754 binary64, and
// precedence.
const a = 17, b = 5, c = -3;
console.log(a + b, a - b, a * b, b - a, a * c, a / b, a % b, a ** 2, 2 ** 10, (-2) ** 2, 2 ** -1, 2 ** 3 ** 2);
console.log(a + b * c, (a + b) * c, a - b - c, a - (b - c), -a, - -a, +c);
let x = 1;
for (let i = 0; i < 10; i++) {
  x = x * 3 + i;
}
console.log(x);
x += 5;
x -= 2;
x *= 4;
x **= 2;
x /= 7;
console.log(x);
let y = 5;
console.log(y++, y, ++y, y--, --y);
console.log(Math.abs(-9), Math.max(3, 7, 5), Math.min(3, 7, 5), [1, 2, 3, 4].reduce((s, v) => s + v, 0));
