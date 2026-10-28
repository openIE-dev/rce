// SPDX-License-Identifier: MIT OR Apache-2.0
// Concatenation, length in UTF-16 code units, indexing, slicing and
// immutability.
const s = "hello";
const t = s + ", " + "world";
console.log(t, t.length);
console.log(s[0], s.charAt(1), s.slice(1, 3), s.substring(3, 1), s.slice(-3), s[10], s.charAt(10) === "");
console.log(s.repeat(3), "ab".repeat(0) === "", s.indexOf("l"), s.lastIndexOf("l"), s.includes("ell"), s.search(/l+/));
const u = s.toUpperCase();
console.log(s, u, s === "hello");
let w = "";
for (let i = 0; i < 5; i++) w += String.fromCharCode(97 + i);
console.log(w, "A".charCodeAt(0), "☃", "\x41é\u{1F600}", "tab\tq\"b\\");
console.log("con" + "cat", `multi
line`, String.raw`raw\n${1 + 1}`);
