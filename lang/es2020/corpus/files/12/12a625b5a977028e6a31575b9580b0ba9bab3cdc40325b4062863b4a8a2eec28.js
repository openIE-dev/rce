// SPDX-License-Identifier: MIT OR Apache-2.0
// Methods of String on ASCII text.
const s = "  The quick brown fox  ";
console.log(`[${s.trim()}]`, `[${s.trimStart()}]`, `[${s.trimEnd()}]`);
console.log(s.toUpperCase(), s.toLowerCase());
console.log(JSON.stringify(s.trim().split(" ")), JSON.stringify("a,b,,c".split(",")), JSON.stringify("a,b,c,d".split(",", 2)));
console.log(JSON.stringify("".split(",")), JSON.stringify("".split("")), JSON.stringify("abc".split("")), JSON.stringify("a1b2c3".split(/\d/)));
console.log(["x", "y", "z"].join("-"), [].join(","), [null, undefined, 1].join(","), [[1, 2], [3]].join(";"));
console.log("aaaa".replace("a", "b"), "banana".replace(/an/g, "AN"), "abc".replace("", "-"), "a-b-c".split("-").join("+"));
console.log("seafood".startsWith("sea"), "golang".endsWith("ng"), "chicken".indexOf("ken"), "chicken".indexOf("dmr"));
console.log("42".padStart(5, "0"), "-42".padStart(5, "0"), "ab".padEnd(4, "*") + "|", "abc".padStart(2), "x".padStart(5, "ab"));
console.log("Hello".concat(" ", "World"));
