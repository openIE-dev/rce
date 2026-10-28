// SPDX-License-Identifier: MIT OR Apache-2.0
// Euclid's greatest common divisor and the least common multiple on Number
// values, and on BigInt values past 2^53.
function gcd(a, b) {
  while (b) [a, b] = [b, a % b];
  return Math.abs(a);
}
function gcdBig(a, b) {
  while (b) [a, b] = [b, a % b];
  return a < 0n ? -a : a;
}
function egcd(a, b) {
  if (b === 0) return [a, 1, 0];
  const [g, x, y] = egcd(b, a % b);
  return [g, y, x - Math.floor(a / b) * y];
}
for (const [a, b] of [[12, 18], [17, 5], [0, 9], [-24, 36], [1071, 462], [2147483647, 2147483646]]) {
  const g = gcd(a, b);
  console.log(a, b, g, g ? (a / g) * b : 0);
}
const [g, x, y] = egcd(240, 46);
console.log(g, x, y, 240 * x + 46 * y);
let acc = 1;
for (let i = 1; i <= 40; i++) acc = (acc / gcd(acc, i)) * i;
let accBig = 1n;
for (let i = 1n; i <= 40n; i++) accBig = (accBig / gcdBig(accBig, i)) * i;
console.log(acc, String(accBig), Number.isSafeInteger(acc), BigInt(acc) === accBig);
