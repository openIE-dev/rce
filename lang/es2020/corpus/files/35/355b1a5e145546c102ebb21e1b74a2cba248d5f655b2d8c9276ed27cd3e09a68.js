// SPDX-License-Identifier: MIT OR Apache-2.0
// for, while, do-while, for-of over arrays and strings, and for-in over
// object keys.
const out = [];
for (let i = 0; i < 3; i++) out.push(i);
console.log(out.join(" "));
let n = 1;
while (n < 100) n *= 3;
console.log(n);
let k = 10;
do {
  k++;
} while (k < 5);
console.log(k);
for (let i = 1; i <= 3; i++) {
  const row = [];
  for (let j = 1; j <= 3; j++) row.push(i * j);
  console.log(row.join(" "));
}
for (const c of "go!") console.log(c);
for (const key in { x: 1, y: 2, 0: "z" }) console.log("key", key);
for (const v of [10, 20].values()) console.log("value", v);
let count = 0;
for (;;) {
  if (++count >= 4) break;
}
console.log(count);
