// SPDX-License-Identifier: MIT OR Apache-2.0
// The sieve of Eratosthenes: the primes below 200 and the count below 100000.
function sieve(n) {
  const composite = new Uint8Array(n);
  for (let i = 2; i * i < n; i++) {
    if (!composite[i]) {
      for (let j = i * i; j < n; j += i) composite[j] = 1;
    }
  }
  const out = [];
  for (let k = 2; k < n; k++) if (!composite[k]) out.push(k);
  return out;
}
console.log(sieve(200).join(" "));
console.log(sieve(100000).length);
