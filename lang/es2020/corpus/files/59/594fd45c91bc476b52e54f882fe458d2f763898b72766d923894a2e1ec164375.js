// SPDX-License-Identifier: MIT OR Apache-2.0
// switch compares with ===, falls through without break, and may place
// default anywhere.
function day(d) {
  const out = [];
  switch (d) {
    case 1:
    case 7:
      out.push("weekend");
      break;
    case 2:
      out.push("monday");
    // falls through
    case 3:
      out.push("early week");
      break;
    default:
      out.push("late week");
    case 0:
      out.push("after default");
  }
  return out.join(",");
}
for (const d of [1, 2, 3, 6, 7, 0, "1"]) console.log(JSON.stringify(d), day(d));
const x = 15;
switch (true) {
  case x > 10 && x < 20:
    console.log("teens");
    break;
  case x > 5:
    console.log("not reached");
}
