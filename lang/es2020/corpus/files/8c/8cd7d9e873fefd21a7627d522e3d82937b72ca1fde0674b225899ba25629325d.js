// SPDX-License-Identifier: MIT OR Apache-2.0
// Number literals in every base and exponent form, and conversion to and
// from text in bases.
console.log(255, 0xff, 0XFF, 0o377, 0b11111111, 1e3, 1.5e-3, .5, 5., 0.1e1);
const n = 48879;
console.log(n.toString(16), n.toString(8), n.toString(2), (-n).toString(16), n.toString(36));
console.log(parseInt("ff", 16), parseInt("0x1f"), parseInt("777", 8), parseInt("zz", 36), parseInt(" 42 "), parseInt("12px"), parseInt("px"));
console.log(Number("0x1f"), Number("0b101"), Number("0o17"), Number(" 42 "), Number("12px"), Number(""), Number("1e3"), Number("-0"));
console.log(parseFloat("3.14abc"), parseFloat(".5"), parseFloat("-.5e2"), parseFloat("Infinity"), parseFloat("1e400"));
console.log(Number.isInteger(5.0), Number.isSafeInteger(2 ** 53), Number.isNaN("x"), isNaN("x"), Number.parseFloat === parseFloat);
