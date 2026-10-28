// SPDX-License-Identifier: MIT OR Apache-2.0
// finally runs on every exit and its return overrides the try's return or
// throw; any value can be thrown; Error subclasses keep their name.
function f1() {
  try {
    return "try";
  } finally {
    console.log("finally before return");
  }
}
function f2() {
  try {
    throw new Error("lost");
  } finally {
    return "finally wins";
  }
}
function f3() {
  for (let i = 0; i < 3; i++) {
    try {
      if (i === 1) continue;
      if (i === 2) break;
    } finally {
      console.log("finally", i);
    }
  }
  return "loop done";
}
console.log(f1(), f2(), f3());
class AppError extends Error {
  constructor(msg, code) {
    super(msg);
    this.name = "AppError";
    this.code = code;
  }
}
for (const thrown of [new AppError("bad", 7), 42, "text", null, { custom: true }, undefined]) {
  try {
    throw thrown;
  } catch (e) {
    console.log(typeof e, e instanceof Error, e instanceof AppError, String(e), e && e.code);
  }
}
console.log(Object.prototype.toString.call(new AppError("x")), new AppError("y").toString());
