// SPDX-License-Identifier: MIT OR Apache-2.0
// Base64 encoding and decoding written with shifts and masks over UTF-8
// bytes, checked by a round trip.
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
function encode(bytes) {
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const k = Math.min(3, bytes.length - i);
    const n = (bytes[i] << 16) | ((bytes[i + 1] || 0) << 8) | (bytes[i + 2] || 0);
    for (let j = 0; j < 4; j++) out += j <= k ? ALPHABET[(n >> (18 - 6 * j)) & 63] : "=";
  }
  return out;
}
function decode(s) {
  const out = [];
  for (let i = 0; i < s.length; i += 4) {
    let n = 0, pad = 0;
    for (let j = 0; j < 4; j++) {
      const c = s[i + j];
      if (c === "=") pad++;
      n = (n << 6) | (c === "=" ? 0 : ALPHABET.indexOf(c));
    }
    out.push((n >> 16) & 255, (n >> 8) & 255, n & 255);
    out.length -= pad;
  }
  return out;
}
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
for (const bytes of [[], [102], [102, 111], [102, 111, 111], utf8("foobar"), utf8("héllo wörld"), [0, 255, 254]]) {
  const e = encode(bytes);
  console.log(JSON.stringify(bytes), e, JSON.stringify(decode(e)) === JSON.stringify(bytes));
}
