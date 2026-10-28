// SPDX-License-Identifier: MIT OR Apache-2.0
// Strict and loose equality, and relational comparison of numbers, strings
// and mixed types.
console.log(3 < 5, 5 <= 5, 7 > 7, -1 >= -2, 4 === 4, 4 !== 4, 1 === 1.0);
console.log(NaN === NaN, NaN < 1, NaN > 1, NaN !== NaN, [NaN].includes(NaN), [NaN].indexOf(NaN));
console.log(0 === -0, Object.is(0, -0), Object.is(NaN, NaN), 2 ** 53 === 2 ** 53 + 1);
console.log("apple" < "banana", "apple" < "Apple", "a" < "aa", "" < "a", "Z" < "a", "10" < "9", 10 < 9, "10" < 9);
console.log(1 == "1", 1 === "1", 0 == "", null == undefined, null === undefined, null == 0, null >= 0, undefined == 0);
console.log([1, 2] == "1,2", [] == 0, [0] == false, "1" == true, "2" == true, ({}) == "[object Object]");
console.log("b" > "a" > "0", 1 < 2 < 3, 3 > 2 > 1, null < 1, undefined < 1, "abc" < "abd");
