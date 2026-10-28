// SPDX-License-Identifier: MIT OR Apache-2.0
// FNV-1a in 32 bits with Math.imul and >>> 0, and in 64 bits with BigInt
// and BigInt.asUintN, over the UTF-8 bytes of each string.
function utf8(s) {
  const out = [];
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
    else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  return out;
}
function fnv32a(bytes) {
  let h = 2166136261;
  for (const b of bytes) h = Math.imul(h ^ b, 16777619) >>> 0;
  return h;
}
function fnv64a(bytes) {
  let h = 14695981039346656037n;
  for (const b of bytes) h = BigInt.asUintN(64, (h ^ BigInt(b)) * 1099511628211n);
  return h;
}
for (const s of ["", "a", "foobar", "hello, world", "日本"]) {
  const b = utf8(s);
  console.log(JSON.stringify(s), fnv32a(b).toString(16).padStart(8, "0"), fnv64a(b).toString(16).padStart(16, "0"), b.length);
}
let naive = 2166136261;
for (const b of utf8("foobar")) naive = (naive ^ b) * 16777619;
console.log(naive, Number.isSafeInteger(naive), naive >>> 0 === fnv32a(utf8("foobar")));
