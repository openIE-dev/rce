// SPDX-License-Identifier: MIT OR Apache-2.0
// Prime factorisation by trial division on BigInt values, including
// numbers past 2^53 and 2^64.
function factors(n) {
  const fs = [];
  for (let p = 2n; p * p <= n; p += p === 2n ? 1n : 2n) {
    while (n % p === 0n) {
      fs.push(p);
      n /= p;
    }
  }
  if (n > 1n) fs.push(n);
  return fs;
}
for (const n of [1n, 2n, 12n, 97n, 360n, 1001n, 65536n, 600851475143n, 9999999967n, 2n ** 64n - 1n, 2n ** 64n + 1n]) {
  console.log(String(n), factors(n).join(" "));
}
