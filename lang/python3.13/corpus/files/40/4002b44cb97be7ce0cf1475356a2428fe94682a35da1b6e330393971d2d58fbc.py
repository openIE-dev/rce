# SPDX-License-Identifier: MIT OR Apache-2.0
# Balanced brackets with a stack; reports the offset of the first error.
PAIR = {")": "(", "]": "[", "}": "{"}


def check(s):
    stack = []
    for i, c in enumerate(s):
        if c in "([{":
            stack.append(c)
        elif c in PAIR:
            if not stack or stack[-1] != PAIR[c]:
                return False, i
            stack.pop()
    return (False, len(s)) if stack else (True, -1)


for s in ["", "()", "([]{})", "(]", "((())", "{[()()]}x", "}{", "a(b[c]d)e"]:
    ok, at = check(s)
    print(f"{s!r:<12} {ok} {at}")
