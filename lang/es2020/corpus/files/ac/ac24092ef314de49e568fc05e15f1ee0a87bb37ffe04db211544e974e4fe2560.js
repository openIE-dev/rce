// SPDX-License-Identifier: MIT OR Apache-2.0
// Knuth-Morris-Pratt substring search, checked against indexOf.
function failure(p) {
  const f = new Array(p.length).fill(0);
  for (let i = 1, k = 0; i < p.length; i++) {
    while (k && p[i] !== p[k]) k = f[k - 1];
    if (p[i] === p[k]) k++;
    f[i] = k;
  }
  return f;
}
function findAll(text, p) {
  if (!p) return [];
  const f = failure(p), out = [];
  for (let i = 0, k = 0; i < text.length; i++) {
    while (k && text[i] !== p[k]) k = f[k - 1];
    if (text[i] === p[k]) k++;
    if (k === p.length) {
      out.push(i - p.length + 1);
      k = f[k - 1];
    }
  }
  return out;
}
console.log(failure("ababaca").join(" "), failure("aaaa").join(" "));
const text = "abababcabababcababab";
for (const p of ["abab", "ababc", "c", "x", text]) {
  const found = findAll(text, p);
  console.log(p, JSON.stringify(found), (found.length ? found[0] : -1) === text.indexOf(p));
}
