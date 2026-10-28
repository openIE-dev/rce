// SPDX-License-Identifier: MIT OR Apache-2.0
// All permutations of four letters in lexicographic order by the
// next-permutation step, and the count for a multiset.
function nextPerm(a) {
  let i = a.length - 2;
  while (i >= 0 && a[i] >= a[i + 1]) i--;
  if (i < 0) return false;
  let j = a.length - 1;
  while (a[j] <= a[i]) j--;
  [a[i], a[j]] = [a[j], a[i]];
  const tail = a.splice(i + 1).reverse();
  a.push(...tail);
  return true;
}
const a = [..."abcd"];
const seen = [a.join("")];
while (nextPerm(a)) seen.push(a.join(""));
for (let k = 0; k < seen.length; k += 6) console.log(seen.slice(k, k + 6).join(" "));
console.log(seen.length);
const b = [..."aabb"];
let m = 1;
while (nextPerm(b)) m++;
console.log(m, b.join(""));
