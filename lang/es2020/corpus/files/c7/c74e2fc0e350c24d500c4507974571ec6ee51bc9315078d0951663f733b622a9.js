// SPDX-License-Identifier: MIT OR Apache-2.0
// Sparse arrays: holes are absent properties, skipped by forEach, map and
// for-in, but visited by for-of as undefined.
const a = [1, , 3];
a[6] = 7;
console.log(a, a.length, 1 in a, a[1], Object.keys(a));
const seen = [];
a.forEach((v, i) => seen.push(i));
console.log(seen.join(","), a.map((v) => v * 2), a.filter(() => true), a.join("-"));
const ofs = [];
for (const v of a) ofs.push(String(v));
console.log(ofs.join(","), [...a].length, Array.from(a).includes(undefined), a.indexOf(undefined), a.includes(undefined));
const b = new Array(3);
console.log(b, b.length, Array.from({ length: 3 }), [...Array(2).keys()]);
const c = [1, 2, 3, 4];
c.length = 2;
console.log(c);
c.length = 4;
console.log(c, delete c[0], c);
console.log(JSON.stringify([, 1]), String([null, undefined, , 4]), [, 1].length, [1, ,].length);
