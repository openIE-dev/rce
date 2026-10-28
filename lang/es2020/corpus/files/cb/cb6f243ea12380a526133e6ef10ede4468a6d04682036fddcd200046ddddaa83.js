// SPDX-License-Identifier: MIT OR Apache-2.0
// if, else if and else, the conditional operator and truthiness.
function classify(n) {
  if (n < 0) {
    return "negative";
  } else if (n === 0) {
    return "zero";
  } else if (n % 2 === 0) {
    return "even";
  } else {
    return "odd";
  }
}
for (const n of [-5, 0, 4, 7, 2.5, NaN]) console.log(n, classify(n));
const grade = 83;
console.log(grade >= 90 ? "A" : grade >= 80 ? "B" : "C");
for (const v of [0, 1, "", "x", [], [0], null, undefined, NaN, {}]) {
  console.log(JSON.stringify(v) === undefined ? "undefined" : JSON.stringify(v), v ? "truthy" : "falsy");
}
