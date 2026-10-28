// SPDX-License-Identifier: MIT OR Apache-2.0
// Number is binary64: integers are exact only up to 2^53, and literals
// round to the nearest representable value.
console.log(2 ** 53, 2 ** 53 + 1, 2 ** 53 + 2, 9007199254740993, 9007199254740995, -(2 ** 53) - 1);
console.log(Number.MAX_SAFE_INTEGER + 2, Number.isSafeInteger(2 ** 53 - 1), Number.isSafeInteger(2 ** 53));
console.log(0.1 + 0.2, 0.1 + 0.7, 1.1 * 3, 4.35 * 100, 0.57 * 100, 1e16 + 1, 123456789012345678901234567890);
console.log(Number.MAX_VALUE * 2, Number.MIN_VALUE / 2, Number.MIN_VALUE, 2 ** 1023 * 2, 2 ** -1074, 2 ** -1075);
console.log(0.1 * 3 === 0.3, 9007199254740992 === 9007199254740993, 1e21 + 1 === 1e21, (0.1).toFixed(20));
console.log(Number("9007199254740993"), parseInt("9007199254740993"), JSON.parse("9007199254740993"), +"1e400", +"-1e-400");
