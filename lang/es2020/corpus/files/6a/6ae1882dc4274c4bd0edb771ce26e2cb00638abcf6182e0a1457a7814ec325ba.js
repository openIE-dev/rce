// SPDX-License-Identifier: MIT OR Apache-2.0
// Strings are UTF-16 code units: indexing and slicing can split a
// surrogate pair, and a lone surrogate reaches standard output as U+FFFD.
const s = "a😀b";
console.log(s.length, s[1], s[2], s.charCodeAt(1).toString(16), s.charCodeAt(2).toString(16), s.codePointAt(1).toString(16), s.codePointAt(2).toString(16));
console.log(s.split(""), Array.from(s), [...s].length, s.slice(0, 2), s.slice(2), s.substring(1, 2).length);
console.log(JSON.stringify(s.slice(0, 2)), encodeURIComponent(s), String.fromCharCode(0xd83d, 0xde00) === "😀");
console.log(s.slice(0, 2));
const lone = "\ud800";
console.log(JSON.stringify(lone), lone.length, lone.codePointAt(0), [...lone].length);
try {
  encodeURIComponent(lone);
} catch (e) {
  console.log(e.name);
}
console.log("e\u0301".length, "e\u0301".normalize("NFC").length, "é".normalize("NFD").length, "ﬁ".normalize("NFKC"));
console.log("😀".split("").reverse().join("") === "😀", [..."😀ab"].reverse().join(""), "x".concat("\udc00", "\ud800").length);
