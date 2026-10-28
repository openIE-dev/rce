// SPDX-License-Identifier: MIT OR Apache-2.0
// Run-length encoding and decoding over code points.
function encode(s) {
  const cps = Array.from(s);
  let out = "";
  for (let i = 0; i < cps.length; ) {
    let j = i;
    while (j < cps.length && cps[j] === cps[i]) j++;
    out += String(j - i) + cps[i];
    i = j;
  }
  return out;
}
function decode(s) {
  let out = "";
  for (const m of s.matchAll(/(\d+)(\D)/gu)) out += m[2].repeat(Number(m[1]));
  return out;
}
for (const s of ["", "a", "aaabccddddde", "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW", "ééé日日x", "😀😀a"]) {
  const e = encode(s);
  console.log(JSON.stringify(s), JSON.stringify(e), decode(e) === s);
}
