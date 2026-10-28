// SPDX-License-Identifier: MIT OR Apache-2.0
// Strings compare by UTF-16 code units, so U+FFFF sorts after U+10000,
// whose first unit is a surrogate.
const pairs = [["\uffff", "\u{10000}"], ["é", "f"], ["Z", "a"], ["abc", "abd"], ["ab", "abc"], ["\x00", ""], ["é", "e\u0301"]];
for (const [a, b] of pairs) console.log(JSON.stringify(a), JSON.stringify(b), a < b, a === b);
const ws = ["\u{1F600}", "\uffff", "zebra", "Zebra", "éclair", "eclair", "", "10", "9"];
console.log(JSON.stringify([...ws].sort()));
console.log(["b", "B", "a", "A"].sort().join(""), ["10", "9", "1"].sort().join(","), [10, 9, 1].sort().join(","));
