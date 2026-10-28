// SPDX-License-Identifier: MIT OR Apache-2.0
// Balanced brackets with a stack; reports the offset of the first error.
const PAIR = { ")": "(", "]": "[", "}": "{" };
function check(s) {
  const stack = [];
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if ("([{".includes(c)) stack.push(c);
    else if (c in PAIR) {
      if (!stack.length || stack[stack.length - 1] !== PAIR[c]) return [false, i];
      stack.pop();
    }
  }
  return stack.length ? [false, s.length] : [true, -1];
}
for (const s of ["", "()", "([]{})", "(]", "((())", "{[()()]}x", "}{", "a(b[c]d)e"]) {
  const [ok, at] = check(s);
  console.log(JSON.stringify(s).padEnd(12), ok, at);
}
