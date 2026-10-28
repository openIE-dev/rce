// SPDX-License-Identifier: MIT OR Apache-2.0
// The CRC-32 of IEEE 802.3 computed with a table built at run time;
// >>> 0 keeps each value an unsigned 32-bit integer.
const table = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  table[i] = c;
}
for (const s of ["", "a", "123456789", "The quick brown fox jumps over the lazy dog"]) {
  let c = 0xffffffff;
  for (let i = 0; i < s.length; i++) c = table[(c ^ s.charCodeAt(i)) & 0xff] ^ (c >>> 8);
  c = (c ^ 0xffffffff) >>> 0;
  console.log(c.toString(16).padStart(8, "0"), c);
}
