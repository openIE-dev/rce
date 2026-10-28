// SPDX-License-Identifier: MIT OR Apache-2.0
// The towers of Hanoi: the moves for four discs and the count for twenty.
function hanoi(n, from, to, via, moves) {
  if (n === 0) return;
  hanoi(n - 1, from, via, to, moves);
  moves.push(from + to);
  hanoi(n - 1, via, to, from, moves);
}
const moves = [];
hanoi(4, "A", "C", "B", moves);
console.log(moves.length, moves.join(" "));
const big = [];
hanoi(20, "A", "C", "B", big);
console.log(big.length, 2 ** 20 - 1, big[big.length - 1]);
