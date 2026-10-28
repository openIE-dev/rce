// SPDX-License-Identifier: MIT OR Apache-2.0
// An uncaught exception ends the program with exit status 1, after what was
// printed and after finally blocks.
function check(n) {
  if (n > 2) throw new TypeError(`n too large: ${n}`);
  console.log("ok", n);
}
try {
  for (let i = 0; i < 5; i++) check(i);
} finally {
  console.log("finally runs");
}
