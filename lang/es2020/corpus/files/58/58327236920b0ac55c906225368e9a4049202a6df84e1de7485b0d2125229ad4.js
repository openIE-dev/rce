// SPDX-License-Identifier: MIT OR Apache-2.0
// A binary min-heap as a class over an array.
class MinHeap {
  constructor() {
    this.a = [];
  }
  push(x) {
    const a = this.a;
    a.push(x);
    let i = a.length - 1;
    while (i > 0 && a[(i - 1) >> 1] > a[i]) {
      const p = (i - 1) >> 1;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && a[l] < a[m]) m = l;
        if (r < a.length && a[r] < a[m]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
  get size() {
    return this.a.length;
  }
}
const data = [9, 4, 7, 1, 8, 2, 2, 6, 3, 5, 0];
const h = new MinHeap();
data.forEach((d) => h.push(d));
const out = [];
while (h.size) out.push(h.pop());
console.log(out.join(" "), h.pop());
