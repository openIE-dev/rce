// SPDX-License-Identifier: MIT OR Apache-2.0
// Strings are sequences of UTF-16 code units: length counts units, while
// for-of and Array.from iterate by code point.
const words = ["abc", "héllo", "日本語", "😀", "e\u0301", "👍🏽", "é", "👨\u200d👩\u200d👧"];
for (const w of words) {
  const cps = Array.from(w).map((c) => c.codePointAt(0).toString(16));
  console.log(w, w.length, Array.from(w).length, cps.join(" "), encodeURIComponent(w).split("%").length - 1);
}
let i = 0;
for (const c of "aé日😀") {
  console.log(i++, c.codePointAt(0), c, c.length);
}
console.log("😀".charCodeAt(0), "😀".charCodeAt(1), "😀".codePointAt(0), "😀".codePointAt(1), String.fromCodePoint(0x1f600) === "😀");
