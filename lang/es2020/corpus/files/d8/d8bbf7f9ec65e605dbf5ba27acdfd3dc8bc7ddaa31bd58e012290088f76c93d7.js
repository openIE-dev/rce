// SPDX-License-Identifier: MIT OR Apache-2.0
// switch over strings and mixed types with ===, a default in the middle,
// and break out of nested labelled statements.
function kind(v) {
  switch (typeof v) {
    case "number":
      if (Number.isNaN(v)) return "nan";
    // falls through
    case "bigint":
      return "numeric";
    default:
      return "other:" + typeof v;
    case "string":
      return "text";
  }
}
console.log([1, NaN, 1n, "s", null, undefined, {}].map(kind).join(" "));
const out = [];
outer: for (let i = 0; i < 3; i++) {
  inner: for (let j = 0; j < 3; j++) {
    switch (j) {
      case 1:
        continue inner;
      case 2:
        if (i === 1) break outer;
        break;
    }
    out.push(`${i}${j}`);
  }
}
console.log(out.join(" "));
