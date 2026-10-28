// SPDX-License-Identifier: MIT OR Apache-2.0
// Regular expressions: groups and named groups, the global flag's
// lastIndex, matchAll, replace with patterns and functions, split, sticky,
// unicode and lookbehind.
const date = /(?<y>\d{4})-(?<m>\d{2})-(?<d>\d{2})/;
const m = "on 2026-10-28 at".match(date);
console.log(m.index, m[0], m[1], m.groups.y, m.groups.d, JSON.stringify([...m]));
const g = /a(\d)/g;
console.log(g.exec("a1 a2 a3")[1], g.lastIndex, g.exec("a1 a2 a3")[1], g.lastIndex);
console.log(JSON.stringify([..."x1y22z333".matchAll(/\d+/g)].map((r) => [r[0], r.index])));
console.log("2026-10-28".replace(date, "$<d>/$<m>/$<y>"), "aaa".replace(/a/g, "$&$&"), "abc".replace(/b/, (s, off) => `[${s}@${off}]`));
console.log("a1b2c3".split(/(\d)/), "a, b,c ,d".split(/\s*,\s*/), "test".split(/(?:)/u), "😀😁".split(/(?:)/u).length, "😀😁".split("").length);
const sticky = /\d/y;
console.log(sticky.test("1a"), sticky.lastIndex, sticky.test("1a"), sticky.lastIndex);
console.log(/^.$/.test("😀"), /^.$/u.test("😀"), /\p{L}+/u.exec("123 été!")[0], /(?<=\$)\d+/.exec("cost $42")[0], /a.c/s.test("a\nc"));
console.log("Hello World".replace(/o/g, "0"), /[A-Z]/g[Symbol.replace]("aBc", "_"), "x".padEnd(3, "-").search(/-/));
