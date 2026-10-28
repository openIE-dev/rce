// SPDX-License-Identifier: MIT OR Apache-2.0
// The longest common subsequence by dynamic programming, with one
// subsequence recovered from the table.
function lcs(a, b) {
  const m = a.length, n = b.length;
  const t = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      t[i][j] = a[i - 1] === b[j - 1] ? t[i - 1][j - 1] + 1 : Math.max(t[i - 1][j], t[i][j - 1]);
    }
  }
  let out = "";
  for (let i = m, j = n; i > 0 && j > 0; ) {
    if (a[i - 1] === b[j - 1]) {
      out = a[i - 1] + out;
      i--;
      j--;
    } else if (t[i - 1][j] >= t[i][j - 1]) i--;
    else j--;
  }
  return [t[m][n], out];
}
for (const [a, b] of [["ABCBDAB", "BDCABA"], ["AGGTAB", "GXTXAYB"], ["", "abc"], ["same", "same"], ["kitten", "sitting"]]) {
  console.log(JSON.stringify(a), JSON.stringify(b), ...lcs(a, b));
}
