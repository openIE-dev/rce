// SPDX-License-Identifier: MIT OR Apache-2.0
// Bitwise operators convert their operands to 32-bit integers: | & ^ ~ <<
// and >> to signed, >>> to unsigned, and shift counts are taken modulo 32.
console.log(2 ** 31 | 0, 2 ** 32 | 0, (2 ** 32 + 5) | 0, -1 >>> 0, -1 >>> 31, 1 << 31, 1 << 32, 1 << 33, 1 >> 32);
console.log(~5, ~-1, ~~3.7, ~~-3.7, 3.9 | 0, -3.9 | 0, NaN | 0, Infinity | 0, 1e21 | 0, 2 ** 53 | 0);
console.log(0xffffffff & 0xff, 0xffffffff, 0xffffffff | 0, 0x80000000 >> 0, 0x80000000 >>> 0, -0x80000000 >> 31);
console.log(Math.imul(0x7fffffff, 2), 0x7fffffff * 2, Math.imul(-1, 8), Math.imul(65536, 65536), (65536 * 65536) | 0);
console.log(5 << -1, 5 >>> -1, 5 >> 33, "12" | 0, "0x10" | 0, "abc" | 0, [] | 0, [7] | 0, null | 0, true << 2);
console.log((-7 >> 1), (-7 >>> 1), (-7 / 2) >> 0, Math.floor(-7 / 2), Math.clz32(0), Math.clz32(-1), Math.clz32(0.5));
