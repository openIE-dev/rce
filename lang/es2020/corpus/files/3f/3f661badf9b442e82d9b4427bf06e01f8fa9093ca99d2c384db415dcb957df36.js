// SPDX-License-Identifier: MIT OR Apache-2.0
// Matrix multiplication on Number values, a large power on BigInt values,
// and the determinant by cofactor expansion.
const mul = (a, b) => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], typeof row[0] === "bigint" ? 0n : 0)));
function det(a) {
  if (a.length === 1) return a[0][0];
  return a[0].reduce((s, v, c) => s + (c % 2 ? -1 : 1) * v * det(a.slice(1).map((r) => r.filter((_, k) => k !== c))), 0);
}
console.log(JSON.stringify(mul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])));
let r = [[1n, 0n], [0n, 1n]];
for (let i = 0; i < 100; i++) r = mul(r, [[1n, 1n], [1n, 0n]]);
console.log(r.map((row) => row.map(String).join(" ")).join(" | "));
let rn = [[1, 0], [0, 1]];
for (let i = 0; i < 100; i++) rn = mul(rn, [[1, 1], [1, 0]]);
console.log(rn.map((row) => row.join(" ")).join(" | "));
console.log(det([[2, -3, 1], [2, 0, -1], [1, 4, 5]]), det([[1, 2], [2, 4]]), det([[6, 1, 1, 3], [4, -2, 5, 1], [2, 8, 7, 6], [3, 1, 9, 7]]));
