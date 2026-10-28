// SPDX-License-Identifier: MIT OR Apache-2.0
// try, catch and finally, the order in which they run, rethrowing and the
// built-in error types.
function trial(x) {
  const out = [];
  try {
    out.push("try");
    if (x === 0) throw new RangeError("zero");
    if (x === 1) return out.concat("returned");
    if (x === 2) null.prop;
    out.push("body done");
  } catch (e) {
    out.push(`${e.name}(${e.message})`);
  } finally {
    out.push("finally");
  }
  return out;
}
for (let x = 0; x < 4; x++) console.log(x, trial(x).join(","));
try {
  try {
    undefinedFunction();
  } catch (e) {
    console.log("inner", e.name, e instanceof ReferenceError);
    throw e;
  }
} catch (e) {
  console.log("outer", e.constructor.name, e.message);
}
try {
  JSON.parse("{bad");
} catch {
  console.log("catch without a binding");
}
const e = new TypeError("msg");
console.log(String(e), e.name, e.message, e instanceof Error, Object.prototype.toString.call(e));
