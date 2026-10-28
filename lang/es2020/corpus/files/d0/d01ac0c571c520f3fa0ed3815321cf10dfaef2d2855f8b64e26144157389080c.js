// SPDX-License-Identifier: MIT OR Apache-2.0
// Text to Number: Number and unary + read the whole string; parseInt and
// parseFloat read a prefix; whitespace, signs and radix prefixes differ.
const inputs = ["", " ", "42", " 42\n", "+42", "-42", "4 2", "42px", "0x1F", "-0x1F", "0b101", "0o17", "017", "08", "1e3", "1E-2", ".5", "5.", "1_000", "Infinity", "-Infinity", "infinity", "NaN", "\u00a042\u2028"];
for (const s of inputs) {
  console.log(JSON.stringify(s), Number(s), +s, parseInt(s), parseInt(s, 10), parseFloat(s));
}
console.log(Number(null), Number(undefined), Number(true), Number([]), Number([5]), Number([1, 2]), Number({}), Number("12345678901234567890"));
console.log(parseInt("123", 0), parseInt("11", 2), parseInt("11", 1), parseInt("11", 37), parseInt("z", 36), parseInt(0.0000005), parseInt(1e21));
