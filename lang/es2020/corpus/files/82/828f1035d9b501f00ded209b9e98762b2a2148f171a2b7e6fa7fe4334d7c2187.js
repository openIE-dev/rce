// SPDX-License-Identifier: MIT OR Apache-2.0
// Recursive functions: factorial, Fibonacci, Ackermann and mutual recursion.
const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
const fib = (n) => (n < 2 ? n : fib(n - 1) + fib(n - 2));
function ack(m, n) {
  if (m === 0) return n + 1;
  if (n === 0) return ack(m - 1, 1);
  return ack(m - 1, ack(m, n - 1));
}
function isEven(n) {
  return n === 0 ? true : isOdd(n - 1);
}
function isOdd(n) {
  return n === 0 ? false : isEven(n - 1);
}
for (let i = 0; i <= 25; i += 5) console.log(i, fact(i));
console.log(Array.from({ length: 15 }, (_, i) => fib(i)).join(" "));
console.log(ack(2, 3), ack(3, 3), isEven(10), isOdd(7), isEven(7));
