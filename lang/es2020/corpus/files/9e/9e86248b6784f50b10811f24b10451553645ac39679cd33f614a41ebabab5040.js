// SPDX-License-Identifier: MIT OR Apache-2.0
// Fibonacci numbers as Number lose exactness past 2^53 (from F(79)); as
// BigInt they stay exact.
let a = 0, b = 1, x = 0n, y = 1n;
for (let i = 0; i <= 100; i++) {
  if (i >= 76 && (i <= 82 || i % 10 === 0)) {
    console.log(i, a, String(x), Number.isSafeInteger(a), BigInt(a) === x);
  }
  [a, b] = [b, a + b];
  [x, y] = [y, x + y];
}
console.log(String(x), x.toString(16), BigInt.asUintN(64, x));
