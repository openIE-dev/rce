// SPDX-License-Identifier: MIT OR Apache-2.0
// break and continue, with labels on loops and on a block.
const evens = [];
for (let i = 0; i < 10; i++) {
  if (i % 2 === 1) continue;
  if (i > 6) break;
  evens.push(i);
}
console.log(evens.join(" "));
const pairs = [];
outer: for (let i = 0; i < 4; i++) {
  for (let j = 0; j < 4; j++) {
    if (j === 2) continue outer;
    if (i === 3) break outer;
    pairs.push(`${i}${j}`);
  }
}
console.log(pairs.join(" "));
block: {
  console.log("in block");
  if (pairs.length > 0) break block;
  console.log("not reached");
}
let found = -1;
search: for (const [r, row] of [[1, 2], [3, 4], [5, 6]].entries()) {
  for (const v of row) {
    if (v === 4) {
      found = r;
      break search;
    }
  }
}
console.log(found);
