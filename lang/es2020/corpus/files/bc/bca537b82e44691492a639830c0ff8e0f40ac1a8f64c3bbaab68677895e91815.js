// SPDX-License-Identifier: MIT OR Apache-2.0
// Levenshtein distance, counted over code points and over UTF-16 code
// units, which can differ for characters outside the Basic Multilingual
// Plane.
function lev(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
}
const pairs = [["kitten", "sitting"], ["flaw", "lawn"], ["", "abc"], ["abc", ""], ["same", "same"], ["héllo", "hello"], ["日本語", "日本"], ["😀a", "😃a"], ["😀", "a"]];
for (const [a, b] of pairs) console.log(a, b, lev(Array.from(a), Array.from(b)), lev(a, b));
