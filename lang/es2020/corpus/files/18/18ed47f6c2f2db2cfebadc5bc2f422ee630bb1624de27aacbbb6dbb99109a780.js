// SPDX-License-Identifier: MIT OR Apache-2.0
// Two linear congruential generators with fixed seeds: MINSTD, whose
// products stay below 2^53 and so are exact in Number, and one modulo 2^64
// in BigInt.
let x = 1;
const out = [];
for (let i = 0; i < 10; i++) {
  x = (x * 48271) % 2147483647;
  out.push(x);
}
console.log(out.join(" "));
let s = 42n;
const M = 6364136223846793005n, C = 1442695040888963407n;
const out2 = [];
for (let i = 0; i < 10; i++) {
  s = BigInt.asUintN(64, s * M + C);
  out2.push(String(s >> 33n));
}
console.log(out2.join(" "));
const hist = new Array(10).fill(0);
for (let i = 0; i < 100000; i++) {
  s = BigInt.asUintN(64, s * M + C);
  hist[Number((s >> 33n) % 10n)]++;
}
console.log(hist.join(" "));
