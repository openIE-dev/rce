// SPDX-License-Identifier: MIT OR Apache-2.0
// Recursion with no base case passes the engine's call stack limit, which
// throws a RangeError; here it is not caught.
function down(n) {
  return down(n + 1) + 1;
}
console.log("start");
console.log(down(0));
